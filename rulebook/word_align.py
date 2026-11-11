"""
word_align.py

2026-11-11. Word-level evidence for verb rules, using the bootstrapped lexicon
(rules applied recursively: the lexicon learned with the rules now checks them).

A Swahili verb in a clean-train pair is aligned to the Kidaw'ida word in the same
pair whose lemma is a known translation of the Swahili lemma (bootstrap_lexicon.json,
any tier except blocked; if several, the first). Then for a rule (Swahili feature
-> Kidaw'ida shape): n aligned verbs with the feature, k whose Kidaw'ida word has
the shape, rate with Wilson CI, and counterexamples (aligned words without it).
"""
from __future__ import annotations

import collections
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))
import attest as A  # noqa: E402
import bootstrap_lexicon as B  # noqa: E402
from api.stats import rate_ci  # noqa: E402

N_COUNTER = 3


@lru_cache(maxsize=1)
def aligned() -> list[tuple[int, str, frozenset, str]]:
    lex = json.loads(B.OUT.read_text(encoding="utf-8"))
    known = collections.defaultdict(set)
    for e in lex["entries"]:
        known[e["sw_lemma"]].add(e["dav_lemma"])
    sw_lem, dv_lem = B.lemmatizers()
    out = []
    for i, r in enumerate(A.pairs()):
        dl = [(w, dv_lem.get(w, w)) for w in r["dv"]]
        for w in r["sw"]:
            f = A.sw_features(w)
            if not f:
                continue
            targets = known.get(sw_lem.get(w, w), ())
            hit = next((dw for dw, l in dl if l in targets), None)
            if hit:
                out.append((i, w, frozenset(f), hit))
    return out


def word_level(feature, dav_re: str) -> dict:
    rx = re.compile(dav_re)
    has = feature if callable(feature) else (lambda fs: feature in fs)
    k, n, counter, ex = 0, 0, [], []
    ps = A.pairs()
    for i, sw, fs, dw in aligned():
        if not has(fs):
            continue
        n += 1
        if rx.fullmatch(dw):
            k += 1
            if len(ex) < N_COUNTER:
                ex.append({"pair_id": ps[i]["id"], "sw_word": sw, "dav_word": dw})
        elif len(counter) < N_COUNTER:
            counter.append({"pair_id": ps[i]["id"], "sw_word": sw, "dav_word": dw})
    return {"method": "aligned_verbs (bootstrap lexicon)", "n_aligned": n, "k_match": k,
            "rate": rate_ci(k, n) if n else None, "examples": ex, "counterexamples": counter}
