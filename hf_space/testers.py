"""
testers.py -- paid testers on the Translator tab (2026-11-13, user decision).

The user got funding to pay native-speaker testers. Each tester enters a phone number once (it stays in their
browser); every submission carries only a salted hash of it. User decisions (2026-11-13):
  - storage: the raw number goes ONLY to a separate private dataset (TESTERS_DATASET_REPO, one file per tester:
    number <-> hash). The feedback dataset never contains a number. This is a dated exception to CLAUDE.md
    "never store raw phone numbers"; delete the numbers after the final payment.
  - paid unit: a COMPLETE submission = all three translations rated AND the tester's own Kitaita translation.
    Payment once per 100 complete submissions (scripts/tester_payouts.py). Each Swahili sentence counts once
    per tester (norm_sentence; the page shows the same count since 2026-11-17).
The hash is the same as the SMS API's (api.util.hash_identifier, prefix "ph", PHONE_HASH_SALT), so one person
is one tester across channels.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
from datetime import datetime, timezone

TESTERS_DATASET_REPO = os.environ.get("TESTERS_DATASET_REPO", "mpily/bantugrammar-testers")
PAY_EVERY = 100
_registered: set[str] = set()
_lock = threading.Lock()


def normalize_phone(raw: str) -> str | None:
    """Kenyan numbers to +254XXXXXXXXX (07.., 01.., 7.., 254.., +254..); other +<country> numbers kept as
    + and 8-15 digits. None if it does not look like a phone number."""
    d = re.sub(r"[^\d+]", "", raw or "")
    if d.startswith("00"):
        d = "+" + d[2:]
    if re.fullmatch(r"0[17]\d{8}", d):
        return "+254" + d[1:]
    if re.fullmatch(r"[17]\d{8}", d):
        return "+254" + d
    if re.fullmatch(r"254[17]\d{8}", d):
        return "+" + d
    if re.fullmatch(r"\+254[17]\d{8}", d):
        return d
    if re.fullmatch(r"\+[1-9]\d{7,14}", d) and not d.startswith("+254"):
        return d
    return None


def tester_id(phone: str) -> str:
    """Salted hash of a normalized number (raises if PHONE_HASH_SALT is not set)."""
    from api.util import hash_identifier
    return hash_identifier(phone, os.environ.get("PHONE_HASH_SALT", ""), "ph")


def is_complete(ratings: dict, correction: str | None) -> bool:
    return all(ratings.get(k) for k in ("A", "C", "E")) and bool((correction or "").strip())


def register(phone: str, tid: str, token: str) -> None:
    """Write number <-> hash to the private testers dataset (once per tester per process)."""
    with _lock:
        if tid in _registered:
            return
    from huggingface_hub import HfApi
    rec = {"tester_id": tid, "phone": phone, "registered": datetime.now(timezone.utc).isoformat(),
           "consent": "entered on the Translator tab under the payment notice"}
    api = HfApi(token=token)
    api.create_repo(TESTERS_DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
    path = f"testers/{tid}.json"
    if path not in set(api.list_repo_files(TESTERS_DATASET_REPO, repo_type="dataset")):
        api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(rec, indent=1).encode()), path_in_repo=path,
                        repo_id=TESTERS_DATASET_REPO, repo_type="dataset")
    with _lock:
        _registered.add(tid)


def complete_count(files: list[str], tid: str) -> int:
    """Complete submissions of one tester, from feedback file names (feedback_ab/<ts>_<tid>_complete_<id>.json).
    An upper bound (repeats counted); the page uses unique_complete since 2026-11-17."""
    return sum(1 for f in files if f"_{tid}_complete_" in f)


# --------------------------------------------------------------------------- 2026-11-17, user request
# Three testers were told "100 complete" while the payout script (one count per Swahili sentence) found 92-99:
# the page counted repeats. Repeats came from (1) the sentence button drawing with replacement and (2) one
# Submit saved several times (clicks queued while the upload ran: 6 saves in 12 s). Fixes: one sentence key
# shared by the page and the payout script; the page counts unique sentences; per-tester memory of rated
# sentences (for the button); and a short window that drops an identical resubmission.

def norm_sentence(s: str | None) -> str:
    """The key under which a Swahili sentence counts once per tester (page, payout script, analysis)."""
    return " ".join((s or "").lower().split()).rstrip(" .!?")


DUPLICATE_WINDOW_S = 120
_recent: dict[tuple, float] = {}
_rated: dict[str, dict[str, bool]] = {}   # tester id -> {sentence key: complete?}
_loading: set[str] = set()
_loaded: set[str] = set()


def is_duplicate_submit(key: tuple, now: float) -> bool:
    """True if the identical submission (key) was already accepted within DUPLICATE_WINDOW_S seconds;
    otherwise remembers it and returns False."""
    with _lock:
        for k in [k for k, t in _recent.items() if now - t > DUPLICATE_WINDOW_S]:
            del _recent[k]
        if key in _recent:
            return True
        _recent[key] = now
        return False


def forget_submit(key: tuple) -> None:
    """Undo is_duplicate_submit's memory (the save failed, so a retry must go through)."""
    with _lock:
        _recent.pop(key, None)


def load_rated(tid: str, token: str, repo: str, cache_dir: str | None = None) -> dict[str, bool]:
    """{sentence key: complete?} for one tester, from their feedback files (downloaded once per process,
    then kept up to date by record_rated)."""
    with _lock:
        if tid in _loaded:
            return dict(_rated[tid])
    import glob
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo, repo_type="dataset", token=token, allow_patterns=[f"feedback_ab/*_{tid}_*"],
                          local_dir=cache_dir or f"/tmp/kitaita_rated/{tid}", max_workers=8)
    out: dict[str, bool] = {}
    for f in glob.glob(f"{d}/feedback_ab/*_{tid}_*.json"):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        k = norm_sentence(r.get("swahili_input"))
        ratings = {a: (r.get("arms", {}).get(a) or {}).get("rating") for a in ("A", "C", "E")}
        out[k] = out.get(k, False) or is_complete(ratings, r.get("native_speaker_correction"))
    with _lock:
        d = _rated.setdefault(tid, {})
        for k, v in out.items():
            d[k] = d.get(k, False) or v
        _loading.discard(tid)
        _loaded.add(tid)
        return dict(d)


def load_rated_in_background(tid: str, token: str, repo: str) -> None:
    with _lock:
        if tid in _loaded or tid in _loading:
            return
        _loading.add(tid)
    def run():
        try:
            load_rated(tid, token, repo)
        except Exception:
            with _lock:
                _loading.discard(tid)
    threading.Thread(target=run, daemon=True).start()


def rated_if_loaded(tid: str) -> set[str]:
    with _lock:
        return set(_rated.get(tid, {}))


def record_rated(tid: str, sentence: str, complete: bool) -> None:
    with _lock:
        d = _rated.setdefault(tid, {})
        k = norm_sentence(sentence)
        d[k] = d.get(k, False) or complete


def unique_complete(tid: str) -> int:
    with _lock:
        return sum(_rated.get(tid, {}).values())
