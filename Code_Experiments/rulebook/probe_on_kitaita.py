"""
probe_on_kitaita.py

2026-11-03. User question: do the relation probes work on Kitaita text, so the
rule book can find each verb's subject in the draft itself instead of pairing
draft verbs with Swahili verbs?

The probes (saved_identification_probes, NLLB-1.3B encoder, trained on Swahili
only) are run unchanged on Kidaw'ida reference sentences (HOLDOUT, human
translations), encoded as swh_Latn.

Silver gold, from agreement in the reference: a verb whose subject marker
(kitaita_subject_markers.json surfaces) matches exactly one candidate noun's
class (subject_class_resolver), in a sentence with at least one other noun of a
different class. Gold subject = that noun. Not speaker-annotated.

Compared on the same cases: the probe (identify_relations, one verb at a
time), nearest noun before the verb (else after), first noun, random noun.
Control (does the probe just read the agreement?): the verb's subject marker
is swapped to the other noun's class surface at the same tense; a probe that
follows agreement switches to the other noun.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
sys.path.insert(0, str(Path(__file__).parent))
import os  # noqa: E402

os.chdir(ROOT / "hf_space")
from build_subject_classes import wilson  # noqa: E402
from rulebook_correct import _noun_forms, _subject_markers, verb_prefix  # noqa: E402
from rulebook_flagger import norm  # noqa: E402
from subject_class_resolver import resolve  # noqa: E402

SEED = 20261103
MAX_CASES = 300
HOLDOUT = Path.home() / "languagemodels" / "mining" / "holdout.jsonl"


def noun_markers(form):
    """Possible subject markers of a noun form (both numbers for invariant forms)."""
    out = {resolve(None, form, sm)[0] for sm in ("ki", "vi")}
    return out - {None}


def cases():
    nouns = _noun_forms()
    bases = sorted(_subject_markers()[1])
    out = []
    for line in open(HOLDOUT, encoding="utf-8"):
        row = json.loads(line)
        toks = row["dav"].split()
        low = [norm(t) for t in toks]
        n_idx = {i: noun_markers(t) for i, t in enumerate(low) if t in nouns}
        n_idx = {i: m for i, m in n_idx.items() if m}
        if len(n_idx) < 2:
            continue
        for v, t in enumerate(low):
            if v in n_idx or t in nouns or len(t) < 4 or t.endswith("nyi") or t.startswith("ku"):
                continue
            sm = {b: h for b in bases if (h := verb_prefix(t, b))}
            if not sm:
                continue
            match = [i for i, m in n_idx.items() if m & set(sm)]
            other = [i for i, m in n_idx.items() if not (m & set(sm))]
            if len(match) == 1 and other:
                out.append({"id": row["id"], "toks": toks, "nouns": sorted(n_idx), "verb": v, "gold": match[0],
                            "verb_sm": sm, "other": other, "noun_markers": {i: sorted(m) for i, m in n_idx.items()}})
    return out


def corrupt(c):
    """Swap the verb's subject marker to an 'other' noun's class at the same tense."""
    dominant = _subject_markers()[0]
    t = norm(c["toks"][c["verb"]])
    surface, tam = next(iter(c["verb_sm"].values()))
    for o in c["other"]:
        for m in c["noun_markers"][o]:
            s2 = dominant.get((m, tam))
            if s2 and s2 != surface:
                toks = list(c["toks"])
                toks[c["verb"]] = s2 + t[len(surface):]
                return toks, o
    return None, None


if __name__ == "__main__":
    from nllb_probe_identification import identify_relations
    rng = random.Random(SEED)
    all_cases = cases()
    rng.shuffle(all_cases)
    sel = all_cases[:MAX_CASES]
    print(len(all_cases), "silver cases;", len(sel), "scored", flush=True)
    score = {k: [0, 0] for k in ("probe", "nearest_before", "first_noun", "random")}
    ctrl = {"n": 0, "follows_agreement": 0, "keeps_gold": 0, "other": 0}
    for k, c in enumerate(sel):
        v, gold, nouns = c["verb"], c["gold"], c["nouns"]
        r = identify_relations(c["toks"], nouns, [v], [])
        p = (r.verb_results or {}).get(v, {}).get("subject_idx")
        before = [i for i in nouns if i < v]
        preds = {"probe": p, "nearest_before": before[-1] if before else min(nouns, key=lambda i: abs(i - v)),
                 "first_noun": nouns[0], "random": rng.choice(nouns)}
        for name, x in preds.items():
            score[name][1] += 1
            score[name][0] += x == gold
        toks2, o = corrupt(c)
        if toks2:
            r2 = identify_relations(toks2, nouns, [v], [])
            p2 = (r2.verb_results or {}).get(v, {}).get("subject_idx")
            ctrl["n"] += 1
            ctrl["follows_agreement"] += p2 == o
            ctrl["keeps_gold"] += p2 == gold
            ctrl["other"] += p2 not in (o, gold)
        if k % 50 == 0:
            print(k, {n: s[0] for n, s in score.items()}, flush=True)
    res = {n: {"right": s[0], "n": s[1], "acc": wilson(s[0], s[1])} for n, s in score.items()}
    res["probe_answered"] = None
    out = {"_meta": {"date": "2026-11-03", "seed": SEED, "holdout": "dav_swa HOLDOUT references",
                     "gold": "silver: agreement in the reference (see docstring)", "n_cases_total": len(all_cases)},
           "accuracy": res,
           "control_swapped_marker": {**ctrl, "follows_agreement_rate": wilson(ctrl["follows_agreement"], ctrl["n"]),
                                      "keeps_gold_rate": wilson(ctrl["keeps_gold"], ctrl["n"])},
           "gold_position": {"subject_before_verb": sum(c["gold"] < c["verb"] for c in sel), "n": len(sel)}}
    print(json.dumps(out, indent=1))
    (Path(__file__).parent / "probe_on_kitaita_results.json").write_text(json.dumps(out, indent=1))
