"""
build_subject_classes.py

2026-11-03. Kidaw'ida noun -> subject-marker class, voted by the verb that
follows the noun in the training text (MINE + HOLDOUT). User question: improve
the subject noun-class table with all the data we have (the version-B subject
step failed mostly on guessed classes: docs/status.md 2026-11-03).

Patterns: the word after a noun is split into subject marker + tense with the
class surfaces of build_subject_markers.py (e-re, ya-, ra-, je-re, ...).
Where the Bible and the modern (dav_swa) text disagree, the Bible decides
(user decision 2026-11-03: the Bible is more standardised).
P(class | pattern) is learned from speaker-verified nouns
(VERIFIED_SUBJECT_MARKERS); a noun's class is the argmax of its summed
P(class | pattern) over occurrences, kept if it has >= MIN_VOTES votes and the
top class has >= MIN_SHARE of them.

Check: 5-fold cross-validation over the verified nouns (patterns learned
without the held-out nouns), against (a) the pipeline's corpus class table
(noun_class_of, what the pipeline uses when a noun is not verified), (b) the
noun-shape predictor predict_kitaita_class_from_prefix, (c) the majority class,
and (d) a control: the same voting with the training nouns' classes shuffled.
Output: hf_space/kitaita_subject_classes.json (word -> class + vote counts, no
sentences) and Code_Experiments/rulebook/build_subject_classes_results.json.
Not speaker-verified.
"""
import collections
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
sys.path.insert(0, str(Path(__file__).parent))
from build_subject_markers import TAMS, surfaces  # noqa: E402
from kitaita_verified_corrections import TAM_SM_ALIAS, VERIFIED_SUBJECT_MARKERS  # noqa: E402
from rulebook_flagger import norm  # noqa: E402

MINING = Path.home() / "languagemodels" / "mining"
OUT = ROOT / "hf_space" / "kitaita_subject_classes.json"
RESULTS = Path(__file__).parent / "build_subject_classes_results.json"
MIN_PATTERN_HITS = 5
MIN_VOTES = 3
MIN_SHARE = 0.5
SEED = 20261103

GOLD = {norm(n): TAM_SM_ALIAS.get(c, c) for n, c in VERIFIED_SUBJECT_MARKERS.items()}
CLASSES = sorted(set(GOLD.values()))
ALL_SURFACES = sorted({s for c in CLASSES for s in surfaces(c)}, key=len, reverse=True)


def patterns(token: str) -> set[str]:
    """Every (marker, tense) split of a word, as 'marker-tense' ('ya-' = present)."""
    out = set()
    for s in ALL_SURFACES:
        for t in TAMS:
            if token.startswith(s + t) and len(token) > len(s + t) + 1:
                out.add(f"{s}-{t}")
        rest = token[len(s):]
        if s.endswith("a") and token.startswith(s) and len(rest) >= 3 and rest[0] not in "aeiou":
            out.add(f"{s}-")
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 3), round(c - h, 3), round(c + h, 3)]


def learn(nouns_to_class, follow):
    """pattern -> {class: P(class | pattern)} from the given labelled nouns."""
    cnt = collections.defaultdict(collections.Counter)
    for n, c in nouns_to_class.items():
        for pats in follow.get(n, []):
            for p in pats:
                cnt[p][c] += 1
    return {p: {c: k / sum(v.values()) for c, k in v.items()}
            for p, v in cnt.items() if sum(v.values()) >= MIN_PATTERN_HITS}


def predict(noun, model, follow):
    votes = collections.Counter()
    for pats in follow.get(noun, []):
        known = [p for p in pats if p in model]
        for p in known:
            for c, w in model[p].items():
                votes[c] += w / len(known)
    total = sum(votes.values())
    if total < MIN_VOTES:
        return None, votes
    c, v = votes.most_common(1)[0]
    return (c if v / total >= MIN_SHARE else None), votes


if __name__ == "__main__":
    follow = collections.defaultdict(list)  # noun form -> [patterns of the next word], per occurrence
    by_src = {"bible": collections.defaultdict(list), "davswa": collections.defaultdict(list)}
    sources = {}
    for name in ("mine.jsonl", "holdout.jsonl"):
        path = MINING / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            src = "bible" if row["id"].startswith("bible:") else "davswa"
            toks = [norm(t) for t in row["dav_tok"]]
            for a, b in zip(toks, toks[1:]):
                if a and b:
                    p = patterns(b)
                    if p:
                        follow[a].append(p)
                        by_src[src][a].append(p)

    import os
    os.chdir(ROOT / "hf_space")
    from kitaita_e2e_pipeline import build_lookup_tables, load_pages, predict_kitaita_class_from_prefix
    tables = build_lookup_tables(*load_pages())
    noun_kitaita, noun_class_of = tables[0], tables[3]

    gold = {n: c for n, c in GOLD.items() if follow.get(n)}
    nouns = sorted(gold)
    rng = random.Random(SEED)
    rng.shuffle(nouns)
    folds = [nouns[i::5] for i in range(5)]
    res = {m: {"right": 0, "answered": 0} for m in ("votes", "pipeline_class_table", "shape", "majority", "shuffled_control")}
    errors = []
    for k in range(5):
        test = set(folds[k])
        train = {n: gold[n] for n in nouns if n not in test}
        model = learn(train, follow)
        labels = list(train.values())
        rng.shuffle(labels)
        model_ctrl = learn(dict(zip(train, labels)), follow)
        majority = collections.Counter(train.values()).most_common(1)[0][0]
        for n in test:
            preds = {"votes": predict(n, model, follow)[0], "shuffled_control": predict(n, model_ctrl, follow)[0],
                     "pipeline_class_table": (noun_class_of.get(n) or {}).get("subject_marker"),
                     "shape": predict_kitaita_class_from_prefix(n, noun_class_of)[0], "majority": majority}
            for m, p in preds.items():
                p = TAM_SM_ALIAS.get(p, p) if p else p
                if p is not None:
                    res[m]["answered"] += 1
                    res[m]["right"] += p == gold[n]
            if preds["votes"] is not None and preds["votes"] != gold[n]:
                errors.append({"noun": n, "gold": gold[n], "votes": preds["votes"]})
    n_gold = len(nouns)
    summary = {m: {"coverage": wilson(r["answered"], n_gold), "accuracy_when_answered": wilson(r["right"], r["answered"]),
                   "right_of_all": wilson(r["right"], n_gold)} for m, r in res.items()}
    for m, s in summary.items():
        print(f"{m:22s} coverage {s['coverage']}  accuracy {s['accuracy_when_answered']}  right/all {s['right_of_all']}")
    print("errors (votes):", errors[:25])

    # final table: patterns from all verified nouns, applied to every noun form in the text
    model = learn(gold, follow)
    candidates = {norm(v["kitaita_word"]) for v in noun_kitaita.values() if v.get("kitaita_word")}
    mined = json.loads((ROOT / "hf_space" / "kitaita_mined_lexicon_round1.json").read_text(encoding="utf-8"))["entries"]
    candidates |= {norm(e["kitaita"]) for e in mined.values() if e.get("area") == "nouns" and e.get("kitaita")}
    candidates |= set(json.loads((ROOT / "hf_space" / "kitaita_agreement_data.json").read_text(encoding="utf-8"))["noun_classes"])
    # 2026-11-03, user decision: where the registers vary, the Bible (more
    # standardised) decides. class = the Bible-only vote when it is decisive,
    # else the vote over both. bible_votes are kept for every noun (also the
    # verified ones) so the resolver can check speaker entries against them.
    table = {}
    for n in sorted(candidates | set(gold)):
        c_all, votes = predict(n, model, follow)
        c_bible, v_bible = predict(n, model, by_src["bible"])
        c_mod, _ = predict(n, model, by_src["davswa"])
        c = c_bible or c_all
        if c or v_bible:
            table[n] = {"class": c, "votes": round(sum(votes.values()), 1),
                        "share": round(votes[c] / sum(votes.values()), 2) if c and votes else None,
                        "decided_by": "bible" if c_bible else ("both" if c_all else None),
                        "davswa_class": c_mod,
                        "bible_votes": {k: round(v, 1) for k, v in v_bible.most_common() if v >= 0.5}}
    print(len(candidates), "candidate noun forms,", sum(1 for v in table.values() if v["class"]), "with a voted class;",
          sum(1 for v in table.values() if v["class"] and v["davswa_class"] and v["davswa_class"] != v["class"]),
          "where the Bible overrules the modern vote")
    meta = {"built_by": "Code_Experiments/rulebook/build_subject_classes.py", "date": "2026-11-03",
            "source_sha256_16": sources, "min_pattern_hits": MIN_PATTERN_HITS, "min_votes": MIN_VOTES,
            "min_share": MIN_SHARE, "status": "corpus-voted, NOT speaker-verified"}
    OUT.write_text(json.dumps({"_meta": meta, "nouns": table}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    RESULTS.write_text(json.dumps({"_meta": {**meta, "seed": SEED, "n_gold_in_text": n_gold,
                                             "n_gold_total": len(GOLD)},
                                   "cv_5fold": summary, "vote_errors": errors}, ensure_ascii=False, indent=1))
