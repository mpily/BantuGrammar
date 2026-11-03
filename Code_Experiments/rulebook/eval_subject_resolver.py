"""
eval_subject_resolver.py

2026-11-03. How well does the subject-class resolver (hf_space/
subject_class_resolver.py) predict the speaker-verified subject marker of a
noun when the verified tier is switched off? Gold: VERIFIED_SUBJECT_MARKERS
(incl. the Class Sorter batch). Number is the gold marker's own number (i- read
as singular), passed as a neutral Swahili marker (ki- / vi-).

Votes are rebuilt in each of 5 folds from the other folds' nouns only
(build_subject_classes.learn), so no noun is scored with votes learned from
itself. concord and round1 never saw the gold labels. Reports coverage and
accuracy per tier alone and for tier orders, with Wilson CIs.
"""
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
sys.path.insert(0, str(Path(__file__).parent))
import build_subject_classes as B  # noqa: E402
import subject_class_resolver as R  # noqa: E402
from rulebook_flagger import norm  # noqa: E402

SEED = 20261103
ORDERS = [("votes",), ("concord",), ("round1",), ("votes", "concord", "round1"), ("votes", "round1", "concord"),
          ("round1", "votes", "concord"), ("concord", "votes", "round1"),
          ("votes", "infinitive", "concord", "round1"), ("votes", "infinitive", "round1", "concord")]

if __name__ == "__main__":
    follow = collections.defaultdict(list)
    for name in ("mine.jsonl", "holdout.jsonl"):
        for line in open(B.MINING / name, encoding="utf-8"):
            toks = [norm(t) for t in json.loads(line)["dav_tok"]]
            for a, b in zip(toks, toks[1:]):
                p = B.patterns(b)
                if a and b and p:
                    follow[a].append(p)
    gold = dict(B.GOLD)
    swahili_of = {norm(k): v for v, k in ((sw, R.VERIFIED_NOUN_TRANSLATIONS[sw]) for sw in R.VERIFIED_NOUN_TRANSLATIONS)}
    for sw, e in R._data()[3].items():
        for f in e["kitaita"]:
            swahili_of.setdefault(f, sw)
    nouns = sorted(gold)
    rng = random.Random(SEED)
    rng.shuffle(nouns)
    folds = [nouns[i::5] for i in range(5)]
    verified, _, concord, round1, by_form = R._data()
    res = {o: collections.Counter() for o in ORDERS}
    by_class = {o: collections.defaultdict(collections.Counter) for o in ORDERS}
    errors = collections.defaultdict(list)
    for k in range(5):
        test = set(folds[k])
        model = B.learn({n: gold[n] for n in nouns if n not in test}, follow)
        fold_votes = {}
        for n in test:
            c, _ = B.predict(n, model, follow)
            if c:
                fold_votes[n] = c

        R._data = (lambda fv: (lambda: (verified, fv, concord, round1, by_form)))(fold_votes)
        for n in test:
            g = gold[n]
            number = "sg" if g == "i" or R.MARKER_TO_NUMS.get(g, (0,))[0] in R.SG else "pl"
            sm = "ki" if number == "sg" else "vi"
            for o in ORDERS:
                pred, why = R.resolve(swahili_of.get(n), n, sm, tiers=o)
                res[o]["n"] += 1
                if pred is not None:
                    res[o]["answered"] += 1
                    res[o]["right"] += pred == g
                    by_class[o][g]["answered"] += 1
                    by_class[o][g]["right"] += pred == g
                    if pred != g and len(errors[o]) < 30:
                        errors[o].append({"noun": n, "swahili": swahili_of.get(n), "gold": g, "pred": pred, "why": why})
    out = {}
    for o in ORDERS:
        r = res[o]
        out["+".join(o)] = {"n": r["n"], "coverage": B.wilson(r["answered"], r["n"]),
                            "accuracy_when_answered": B.wilson(r["right"], r["answered"]),
                            "right_of_all": B.wilson(r["right"], r["n"]),
                            "by_gold_class": {c: dict(v) for c, v in sorted(by_class[o].items())}}
        print(f"{'+'.join(o):24s} coverage {out['+'.join(o)]['coverage']}  "
              f"accuracy {out['+'.join(o)]['accuracy_when_answered']}  right/all {out['+'.join(o)]['right_of_all']}")
    print("errors", errors[ORDERS[-1]])
    (Path(__file__).parent / "eval_subject_resolver_results.json").write_text(json.dumps(
        {"_meta": {"date": "2026-11-03", "seed": SEED, "gold": "VERIFIED_SUBJECT_MARKERS (verified tier off)",
                   "votes": "rebuilt per fold without held-out nouns"},
         "results": out, "errors": {"+".join(o): e for o, e in errors.items()}}, ensure_ascii=False, indent=1))
