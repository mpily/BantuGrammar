"""
kitaita_mined_lexicon.py

2026-11-02. Last-resort vocabulary tier from the round-1 grammar mining
(docs/status.md 2026-10-31 and 2026-11-02). Used ONLY for words the pipeline
would otherwise keep as Swahili: passthrough tokens and secondary nouns with no
Kitaita form. It never replaces a verified or Bible-derived form, and it never
fills class-dependent words (possessives, demonstratives, associatives; see
Code_Experiments/grammar_mining/build_pipeline_lexicon.py).

Every word filled here is machine-mined and NOT speaker-verified. Each fill is
recorded in `d.mined_fills` and as a flag, so the trace shows it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).parent / "kitaita_mined_lexicon_round1.json"
CONF_RANK = {"high": 3, "medium": 2, "low": 1, "derived": 0}
# Lowest confidence the fallback accepts. Chosen on the mining HOLDOUT (1,000
# dav_swa train pairs never used for mining), not on the test split -- see
# docs/status.md 2026-11-02 cont. 3. None disables the tier.
MIN_CONFIDENCE: str | None = "derived"
# Experiment 2 (2026-11-02): among an entry's listed forms, use the one davswa
# translators used most often in the mining data (entry["davswa"]["form_counts"]);
# if they most often kept the Swahili word ("=swahili"), keep it. Needs >= 3
# davswa pairs; ties keep the agent's first-listed form.
PREFER_DAVSWA_FORM: bool = False
MIN_DAVSWA_PAIRS = 3
_PUNCT = ".,!?;:\"'“”‘’()[]…"


@lru_cache(maxsize=1)
def _table() -> dict:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))["entries"]
    except FileNotFoundError:
        return {}


def mined_lookup(token: str, min_confidence: str | None = None) -> dict | None:
    """Mined entry for one Swahili token, or None if absent / below threshold."""
    floor = MIN_CONFIDENCE if min_confidence is None else min_confidence
    if floor is None:
        return None
    entry = _table().get((token or "").strip(_PUNCT).lower().replace("’", "'"))
    if entry is None or CONF_RANK[entry["confidence"]] < CONF_RANK[floor]:
        return None
    if PREFER_DAVSWA_FORM:
        reg = entry.get("davswa") or {}
        counts = reg.get("form_counts") or {}
        if reg.get("pairs", 0) >= MIN_DAVSWA_PAIRS and counts:
            best = max(counts.values())
            if counts.get(entry["kitaita"], 0) < best:
                choice = next(f for f, c in counts.items() if c == best)
                if choice == "=swahili":
                    return None
                return {**entry, "kitaita": choice, "davswa_preferred": True}
    return entry


def apply_mined_lexicon_fallback(d, min_confidence: str | None = None) -> None:
    """Fill passthrough tokens and Kitaita-less secondary nouns from the mined
    table, in place. Filled passthrough tokens move from d.passthrough to
    d.mined_fills, so the API no longer lists them as unknown; filled
    secondary nouns get the mined form as their Kitaita word."""
    fills = getattr(d, "mined_fills", None)
    if fills is None:
        return
    keep = []
    for idx, tok in d.passthrough:
        e = mined_lookup(tok, min_confidence)
        if e is None:
            keep.append((idx, tok))
        else:
            fills.append((idx, tok, e["kitaita"], e["confidence"]))
    d.passthrough = keep
    new_secondary = []
    for idx, lemma, kit in d.secondary_nouns:
        if not kit:
            e = mined_lookup(lemma, min_confidence)
            if e is not None:
                kit = e["kitaita"]
                fills.append((idx, lemma, kit, e["confidence"]))
        new_secondary.append((idx, lemma, kit))
    d.secondary_nouns = new_secondary
    if fills:
        d.flags.append("Mined-lexicon fallback (round 1, NOT speaker-verified): "
                       + ", ".join(f"{sw}->{kit} ({c})" for _, sw, kit, c in fills))
