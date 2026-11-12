"""
haiku_mt.py -- Translation 2 on the Space: Claude Haiku + the rule-book tools (study arm C).

2026-11-12, user decision: replaces "fine-tuned + rule book" as Translation 2. This module only wraps
the existing harness (HaikuHarness/agent_loop.py, prompts.py) and the frozen tools
(agent_translate/tools.py, rulebook-v2); it adds what a public Space needs:

  - a spending guard (user decision): a daily dollar cap for the whole Space plus a per-visitor rate
    limit. Both are in memory, so a restart resets them; set a spend limit on the Anthropic key too.
  - the retrieval corpus (clean train pairs) is downloaded at startup from a private HF dataset
    (user decision), never shipped as files in the public Space repo.
  - status text instead of exceptions: no API key, budget reached, rate limited, corpus missing.

Env / Space secrets: ANTHROPIC_API_KEY (required), HF_TOKEN (read access to the corpus dataset),
HAIKU_DAILY_CAP_USD (default 1.0), HAIKU_PER_VISITOR_PER_HOUR (default 10),
RULEBOOK_DATA_REPO (default mpily/kitaita-rulebook-data).
"""
from __future__ import annotations

import collections
import hashlib
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Repo layout: kitaita-project/{hf_space, HaikuHarness, agent_translate, rulebook}. Space layout: the
# hf_space files at the app root, with HaikuHarness/, agent_translate/ and rulebook/ beside them.
APP_ROOT = HERE if (HERE / "agent_translate").exists() else HERE.parent
for p in (APP_ROOT / "HaikuHarness", APP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DAILY_CAP_USD = float(os.environ.get("HAIKU_DAILY_CAP_USD", "1.0"))
PER_VISITOR_PER_HOUR = int(os.environ.get("HAIKU_PER_VISITOR_PER_HOUR", "10"))
DATA_REPO = os.environ.get("RULEBOOK_DATA_REPO", "mpily/kitaita-rulebook-data")
MAX_CHARS = 400
REQUEST_TIMEOUT_S = 90.0

_lock = threading.Lock()
_spent = {"day": None, "usd": 0.0}
_visits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
_state = {"corpus": "not started", "error": None}


class BudgetExceeded(Exception):
    pass


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _spent_today() -> float:
    with _lock:
        if _spent["day"] != _today():
            _spent.update(day=_today(), usd=0.0)
        return _spent["usd"]


def _add_cost(usd: float) -> None:
    with _lock:
        if _spent["day"] != _today():
            _spent.update(day=_today(), usd=0.0)
        _spent["usd"] += usd


def _visitor_ok(visitor: str) -> bool:
    """Sliding one-hour window per visitor (hashed; nothing is stored beyond this process)."""
    now = time.time()
    with _lock:
        q = _visits[visitor]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= PER_VISITOR_PER_HOUR:
            return False
        q.append(now)
        return True


def visitor_id(request) -> str:
    """Salted hash of the caller's IP (first X-Forwarded-For hop behind the HF proxy)."""
    ip = "unknown"
    try:
        fwd = request.headers.get("x-forwarded-for") if request is not None else None
        ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request is not None else "unknown")
    except Exception:
        pass
    salt = os.environ.get("PHONE_HASH_SALT", "") or os.environ.get("HAIKU_VISITOR_SALT", "")
    return hashlib.sha256((salt + ip).encode()).hexdigest()[:16]


class _GuardedClient:
    """anthropic.Anthropic() stand-in for agent_loop: checks the daily cap before each request and
    books its cost after (the cap can be overshot by at most one request)."""

    def __init__(self, inner, model: str):
        self._inner, self._model, self.messages = inner, model, self
        self.usd = 0.0

    def create(self, **kw):
        from score import cost
        from one_call import usage_of
        if _spent_today() >= DAILY_CAP_USD:
            raise BudgetExceeded()
        r = self._inner.messages.create(**kw)
        c = cost(usage_of(r), self._model)
        self.usd += c
        _add_cost(c)
        return r


def _fetch_corpus() -> None:
    """Download the clean-train pairs to the path tools.py reads (retrieval only)."""
    try:
        from agent_translate import tools
        if tools.TRAIN.exists():
            _state["corpus"] = "ready"
        else:
            _state["corpus"] = "downloading"
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(DATA_REPO, "train.jsonl", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
            tools.TRAIN.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, tools.TRAIN)
            _state["corpus"] = "ready"
        tools.retrieve_examples("habari", 1)   # builds the TF-IDF index once
    except Exception as e:                     # Haiku still runs, without retrieved examples
        _state.update(corpus="missing", error=f"{type(e).__name__}: {e}")


def start() -> None:
    """Call once at Space startup. tools.py (frozen) looks for the hf_space files under
    <root>/hf_space/; on the Space they sit at the root, so a hf_space -> . link is added there."""
    link = APP_ROOT / "hf_space"
    if not link.exists():
        try:
            link.symlink_to(".", target_is_directory=True)
        except OSError as e:
            _state["error"] = f"hf_space link: {e}"
    threading.Thread(target=_fetch_corpus, daemon=True).start()


def configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def translate(text: str, visitor: str = "unknown") -> dict:
    """{'kitaita', 'status', 'notes', 'tool_calls', 'usd', ...}; kitaita is None when nothing was produced."""
    base = {"kitaita": None, "notes": "", "tool_calls": [], "usd": 0.0, "corpus": _state["corpus"]}
    text = (text or "").strip()
    if not configured():
        return {**base, "status": "not configured on this Space (no ANTHROPIC_API_KEY secret)"}
    if len(text) > MAX_CHARS:
        return {**base, "status": f"sentence too long for this mode (max {MAX_CHARS} characters)"}
    if _spent_today() >= DAILY_CAP_USD:
        return {**base, "status": "daily limit reached for this mode; try again tomorrow (UTC)"}
    if not _visitor_ok(visitor):
        return {**base, "status": f"limit of {PER_VISITOR_PER_HOUR} sentences per hour reached; try again later"}
    import anthropic
    import agent_loop
    import prompts
    client = _GuardedClient(anthropic.Anthropic(timeout=REQUEST_TIMEOUT_S, max_retries=2), agent_loop.MODEL)
    try:
        r = agent_loop.translate_with_tools(text, client=client)
    except BudgetExceeded:
        return {**base, "usd": client.usd, "status": "daily limit reached for this mode; try again tomorrow (UTC)"}
    except Exception as e:
        return {**base, "usd": client.usd, "status": f"error: {type(e).__name__}"}
    notes = (r["reply"] or "").split("<translation>")[0].strip()
    base["corpus"] = _state["corpus"]
    return {**base, "kitaita": r["translation"], "status": "ready" if r["translation"] else f"no translation ({r['stop_reason']})",
            "notes": notes, "tool_calls": r["tool_calls"], "usd": client.usd, "n_requests": r["n_requests"],
            "stop_reason": r["stop_reason"], "prompt_sha16": prompts.prompt_sha16(), "model": agent_loop.MODEL}


def system_label() -> str:
    try:
        import prompts
        return f"haiku+tools:rulebook-v2:prompt-{prompts.prompt_sha16()}"
    except Exception:
        return "haiku+tools:rulebook-v2"
