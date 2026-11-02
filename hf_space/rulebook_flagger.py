"""
rulebook_flagger.py

2026-11-02. Version B of the user's two-system design (fine-tuned NLLB draft
-> deterministic rule book). Step 1: flag words in an NLLB Kidaw'ida draft
that are likely hallucinated or untranslated. No correction here.

A token is classed as:
  name        -- a proper name carried over from the source (never flagged)
  swahili     -- copied from the Swahili source and not an established
                 Kidaw'ida word (davswa translators rarely keep it)
  unknown     -- not in the model's training vocabulary (dav_swa train + the
                 Bible pairs), and its stem (after stripping verb prefixes and
                 a final suffix) matches no stem of a known word
  repeat      -- 3rd+ consecutive copy of the same token (repetition collapse)
  ok          -- everything else

Vocabulary = Kidaw'ida side of MINE + HOLDOUT (= dav_swa train + Bible NEN
pairs, the same text arm 2 was trained on). Built from
~/languagemodels/mining/ (not in git).
"""
from __future__ import annotations

import collections
import json
import re
from functools import lru_cache
from pathlib import Path

VOCAB_PATH = Path(__file__).parent / "kitaita_training_vocab.json"
_Q = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_PUNCT = ".,!?;:\"()[]«»“”…-"

# Kidaw'ida verb prefix material (round-1 verb_morphology tables): subject
# markers, SM+tense fusions, tense/aspect/negation markers, object markers.
VERB_PREFIXES = sorted(set("""
ni ku u di m mu wi ghu i ji ghi chi vi ri lu
na kwa wa da mwa ghwa ya ja gha cha va ra lwa
ne ko o de mo we gho e je ghe che ve re lo
ka ki se seke nde si nda mm mmb
""".split()), key=len, reverse=True)
FINAL_SUFFIXES = sorted(["agha", "aa", "ieghe", "eghe", "ie", "ire", "ere", "enyi", "ienyi", "wa", "a", "e", "i", "o"],
                        key=len, reverse=True)
MIN_STEM = 3


def norm(tok: str) -> str:
    return tok.translate(_Q).lower().replace("'", "").strip(_PUNCT)


def candidate_stems(word: str, max_prefixes: int = 3) -> set[str]:
    """All stems reachable by stripping up to `max_prefixes` verb prefixes and
    one final suffix, keeping at least MIN_STEM characters."""
    out = set()
    frontier = {word}
    for _ in range(max_prefixes + 1):
        nxt = set()
        for w in frontier:
            for suf in [""] + FINAL_SUFFIXES:
                if suf and not w.endswith(suf):
                    continue
                core = w[: len(w) - len(suf)] if suf else w
                if len(core) >= MIN_STEM:
                    out.add(core)
            for p in VERB_PREFIXES:
                if w.startswith(p) and len(w) - len(p) >= MIN_STEM:
                    nxt.add(w[len(p):])
        frontier = nxt
    return out


@lru_cache(maxsize=1)
def resources():
    data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    vocab = collections.Counter(data["vocab"])
    dav_n = collections.Counter(data["davswa_dav"])
    swa_n = collections.Counter(data["davswa_swa"])
    stems = set()
    for w in vocab:
        stems |= candidate_stems(w)
    return vocab, stems, dav_n, swa_n


def is_established_shared(tok: str, min_count: int = 3, min_ratio: float = 0.2) -> bool:
    """Swahili word that davswa translators also use as Kidaw'ida (na, kwa,
    muhimu, ...): appears on the Kidaw'ida side often enough relative to the
    Swahili side."""
    _, _, dav_n, swa_n = resources()
    return dav_n[tok] >= min_count and dav_n[tok] >= min_ratio * max(swa_n[tok], 1)


def source_names(src: str) -> set[str]:
    toks = src.translate(_Q).split()
    return {norm(t) for i, t in enumerate(toks) if i > 0 and t[:1].isupper() and norm(t)}


def flag(src: str, hyp: str) -> list[tuple[str, str]]:
    """[(token, category)] for every whitespace token of hyp."""
    vocab, stems, _, _ = resources()
    src_toks = {norm(t) for t in src.split()}
    names = source_names(src)
    out, prev, run = [], None, 0
    for raw in re.sub(r"([,;:!?])(?=\S)", r"\1 ", hyp).split():
        t = norm(raw)
        run = run + 1 if t == prev else 1
        prev = t
        if not t or not re.search(r"[a-z]", t):
            cat = "ok"
        elif run >= 3:
            cat = "repeat"
        elif t in names:
            cat = "name"
        elif t in src_toks and not is_established_shared(t):
            cat = "swahili"
        elif t in vocab or t in src_toks:
            cat = "ok"
        elif candidate_stems(t) & stems:
            cat = "ok"
        else:
            cat = "unknown"
        out.append((raw, cat))
    return out
