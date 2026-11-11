"""
induce_verb_rules.py

2026-11-11. Verb rules induced from aligned verbs (word_align.aligned: Swahili verb
<-> Kidaw'ida word whose lemma is a known translation of the Swahili lemma).

The Kidaw'ida lemma marks the root, so the word splits into prefix + root + suffix.
The prefix is read with kitaita_verb.parse_prefix (canonical code) as
[NEG] SM(series bare/+a/+e) [TAM] [OM]; the reading whose SM row fits the Swahili
SM is preferred. Shape = (neg, series, tam, has_om, final). For each Swahili TAM /
negation feature, the shapes with their counts become rule readings:
  active     n >= 5 and share >= 10%
  tentative  n >= 2 and share >= 3%   (lowered bar, user 2026-11-11)
Output: rulebook/induced_verb_rules.json (counts, shares with Wilson CIs, example
pair IDs + words; no sentences). build_rules.py merges these into rules.json.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))
import attest as A  # noqa: E402
import bootstrap_lexicon as B  # noqa: E402
import word_align as W  # noqa: E402
from api.stats import rate_ci  # noqa: E402
from kitaita_verb import SW_SM_ROWS, parse_prefix  # noqa: E402

OUT = HERE / "induced_verb_rules.json"
FEATURES = ("tam:me", "tam:sha", "tam:li", "tam:ta", "tam:na", "tam:ka", "tam:ki", "tam:nge", "reltam:li",
            "reltam:na", "rel:lipo", "hab", "prohib", "neg:pres", "neg:ku", "neg:ta", "neg:ja", "neg:nge")
ACTIVE = (5, 0.10)
TENTATIVE = (2, 0.03)


def split(word: str, lemma: str):
    j = word.find(lemma)
    if j < 0:
        return None
    return word[:j], word[j + len(lemma):]


def shape(prefix: str, suffix: str, sw_sm: str | None):
    readings = parse_prefix(prefix) if prefix else [{"neg": None, "row": None, "variant": None, "tam": None,
                                                      "om": None}]
    if not readings:
        return None
    rows = SW_SM_ROWS.get(sw_sm or "", ())
    best = next((r for r in readings if r["row"] in rows), readings[0])
    return (best["neg"] or "-", best["variant"] or "none", best["tam"] or "-", "OM" if best["om"] else "-",
            suffix or "-"), best


def main():
    sw_lem, dv_lem = B.lemmatizers()
    ps = A.pairs()
    counts = {f: collections.Counter() for f in FEATURES}
    ex = collections.defaultdict(list)
    n_feat = collections.Counter()
    unparsed = collections.Counter()
    for i, sw, fs, dw in W.aligned():
        lem = dv_lem.get(dw, dw)
        sp = split(dw, lem)
        p = A.sw_parse(sw)
        for f in FEATURES:
            if f not in fs:
                continue
            n_feat[f] += 1
            if not sp or len(dw) <= len(lem):
                unparsed[f] += 1
                continue
            sh = shape(sp[0], sp[1], p["sm"] if p else None)
            if not sh:
                unparsed[f] += 1
                continue
            counts[f][sh[0]] += 1
            if len(ex[(f, sh[0])]) < 3:
                ex[(f, sh[0])].append({"pair_id": ps[i]["id"], "sw_word": sw, "dav_word": dw, "root": lem})
    out = {}
    for f in FEATURES:
        n = n_feat[f]
        rows = []
        for sh, k in counts[f].most_common():
            share = k / n if n else 0
            tier = "active" if k >= ACTIVE[0] and share >= ACTIVE[1] else \
                "tentative" if k >= TENTATIVE[0] and share >= TENTATIVE[1] else None
            if not tier:
                continue
            neg, series, tam, om, final = sh
            tmpl = "-".join(x for x in ((neg if neg != "-" else ""), {"bare": "SM", "a": "SM+a", "e": "SM+e",
                            "none": ""}[series], (tam if tam != "-" else ""), ("OM" if om == "OM" else ""), "ROOT")
                            if x) + (f"-{final}" if final != "-" else "")
            rows.append({"template": tmpl, "shape": dict(zip(("neg", "series", "tam", "om", "final"), sh)),
                         "n": k, "share": rate_ci(k, n), "tier": tier, "examples": ex[(f, sh)]})
        out[f] = {"n_aligned": n, "n_unparsed": unparsed[f], "readings": rows}
    meta = {"built_by": "rulebook/induce_verb_rules.py", "script_sha16": A.sha16(Path(__file__)),
            "bootstrap_lexicon_sha16": A.sha16(B.OUT), "train_sha16": A.sha16(A.TRAIN),
            "thresholds": {"active": ACTIVE, "tentative": TENTATIVE}}
    OUT.write_text(json.dumps({"_meta": meta, "features": out}, ensure_ascii=False, indent=1))
    for f, v in out.items():
        print(f"{f} n={v['n_aligned']} unparsed={v['n_unparsed']}")
        for r in v["readings"][:7]:
            print(f"   {r['template']:<28} {r['n']:>5} {r['share']['rate']:.2f} {r['tier']:<9} "
                  f"{[e['sw_word'] + '>' + e['dav_word'] for e in r['examples'][:2]]}")


if __name__ == "__main__":
    main()
