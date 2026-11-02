"""
rulebook_correct.py

2026-11-02. Version B of the user's two-system comparison: deterministic
corrections applied to the fine-tuned NLLB's draft (system A). Every step can
be switched off on its own, so each is evaluated as a separate change.

  rerank    -- among the model's n-best candidates, take the one with the fewest
               noun-phrase agreement violations (a noun directly followed by a
               concord word, or a demonstrative directly before a noun, of a
               class the noun never takes in the training text; data:
               kitaita_agreement_data.json). Ties keep the model's
               order, so this only ever picks text the model itself produced.
  fill      -- a word the flagger marks as copied Swahili gets the round-1 mined
               Kidaw'ida form if one exists (kitaita_mined_lexicon; class-
               dependent words are never filled). Otherwise it stays Swahili.
  unknown   -- a word not in the training vocabulary is replaced by the Swahili
               source word, but only when the alignment is unambiguous: exactly
               one unknown word and exactly one source word that nothing in the
               output accounts for. Otherwise the model's word is kept.
  repeat    -- 3rd+ consecutive copies of a token are dropped.

Flagged words are never shown as [?] (user decision, 2026-11-02: brackets would
bias the speakers judging A vs B); the default is the Swahili word.
Nothing here is speaker-verified.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from kitaita_mined_lexicon import mined_lookup
from rulebook_flagger import flag, norm

_DATA = Path(__file__).parent / "kitaita_agreement_data.json"
ALL_STEPS = ("rerank", "fill", "unknown", "repeat")


@lru_cache(maxsize=1)
def _agreement():
    d = json.loads(_DATA.read_text(encoding="utf-8"))
    return ({f: set(c) for f, c in d["concord_forms"].items()},
            {w: set(v["classes"]) for w, v in d["noun_classes"].items()},
            set(d.get("demonstratives", [])))


def violations(hyp: str) -> list[tuple[str, str]]:
    """(noun, concord word) pairs whose classes do not overlap: a noun directly
    followed by a concord word, or a demonstrative directly followed by a noun."""
    concord, nouns, dems = _agreement()
    toks = [norm(t) for t in hyp.split()]
    out = []
    for a, b in zip(toks, toks[1:]):
        if a in nouns and b in concord and not (nouns[a] & concord[b]):
            out.append((a, b))
        elif a in dems and b in nouns and b not in concord and not (nouns[b] & concord[a]):
            out.append((b, a))
    return out


def _swap(raw: str, new: str) -> str:
    """Replace the word inside a raw token, keeping its punctuation and initial capital."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", raw)
    lead, core, trail = m.groups() if m else ("", raw, "")
    if core[:1].isupper():
        new = new[:1].upper() + new[1:]
    return f"{lead}{new}{trail}"


def _covered_source_words(src: str, hyp_tokens: list[str]) -> set[str]:
    """Source words the output accounts for: copied as is, or present as one of
    their mined Kidaw'ida forms."""
    hyp = set(hyp_tokens)
    covered = set()
    for s in {norm(t) for t in src.split()} - {""}:
        e = mined_lookup(s, "derived")
        forms = {s} | ({norm(e["kitaita"])} | {norm(f) for f in e["alternatives"]} if e else set())
        if forms & hyp:
            covered.add(s)
    return covered


def correct(src: str, nbest: list[str], steps=ALL_STEPS) -> dict:
    """nbest: the model's candidates, best first. Returns the corrected text and
    a list of what changed."""
    changes = []
    chosen = 0
    if "rerank" in steps and len(nbest) > 1:
        counts = [len(violations(h)) for h in nbest]
        chosen = min(range(len(nbest)), key=lambda i: (counts[i], i))
        if chosen != 0:
            changes.append(("rerank", f"candidate {chosen + 1} (agreement violations {counts[0]} -> {counts[chosen]})"))
    hyp = nbest[chosen]
    tokens = flag(src, hyp)
    words = [t for t, _ in tokens]

    if "fill" in steps:
        for i, (t, c) in enumerate(tokens):
            if c == "swahili":
                e = mined_lookup(norm(t), "derived")
                if e:
                    words[i] = _swap(t, e["kitaita"])
                    changes.append(("fill", f"{norm(t)} -> {e['kitaita']}"))

    if "unknown" in steps:
        unknown = [i for i, (_, c) in enumerate(tokens) if c == "unknown"]
        if len(unknown) == 1:
            others = [norm(w) for j, w in enumerate(words) if j != unknown[0]]
            src_words = [norm(t) for t in src.split() if norm(t)]
            free = [s for s in dict.fromkeys(src_words) if s not in _covered_source_words(src, others)]
            if len(free) == 1:
                i = unknown[0]
                changes.append(("unknown", f"{norm(words[i])} -> {free[0]} (Swahili)"))
                words[i] = _swap(words[i], free[0])

    if "repeat" in steps:
        kept = [w for w, (_, c) in zip(words, tokens) if c != "repeat"]
        if len(kept) < len(words):
            changes.append(("repeat", f"dropped {len(words) - len(kept)} repeated token(s)"))
        words = kept

    edited = any(k != "rerank" for k, _ in changes)
    # Re-joining tokens can alter spacing (e.g. "a,b" -> "a, b"); when no word was
    # edited, return the chosen candidate exactly as the model wrote it.
    text = " ".join(words) if edited else hyp
    return {"kitaita": text, "draft": nbest[0], "chosen_candidate": chosen, "changes": changes}
