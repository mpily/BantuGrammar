"""
build_agreement_data.py

2026-11-02. Data for the version-B agreement checker (hf_space/rulebook_correct.py).

1. concord_forms: Kidaw'ida concord word -> set of classes it can mark, from
   the round-1 concord table (noun_classes_concord.json): demonstratives,
   associative, possessives, -ose, 'other', numerals 1-5, 'many', nominal
   adjectives. Demonstratives are also listed separately, because they can
   come before the noun (kwa ijo isanga). Verb-like series (SM, OM, stative adjectives) are left out:
   they attach to verbs, not to the word after a noun.
2. noun_classes: Kidaw'ida noun form -> classes it takes agreement in,
   assigned the way round 1 did it but for every form in the training text:
   each concord word directly after the noun votes for its classes (weight
   1/|set|); a class is kept if it has >= 20% of the weight; the noun is kept
   if it has >= MIN_VOTES votes, its top class has >= 50% of the weight and at
   most 3 classes survive (function words like na / ni spread across many).
   kwa is excluded as a concord form: it is also the preposition.

Training text = MINE + HOLDOUT (dav_swa train + Bible NEN), the same text arm 2
was trained on. Output: hf_space/kitaita_agreement_data.json (word lists and
counts only, no sentences). Not speaker-verified.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
from rulebook_flagger import norm  # noqa: E402

MINING = Path.home() / "languagemodels" / "mining"
OUT = ROOT / "hf_space" / "kitaita_agreement_data.json"
TYPES = {"DEM_PROX", "DEM_REF", "DEM_DIST", "ASSOC", "POSS_1SG", "POSS_2SG", "POSS_3SG", "POSS_1PL",
         "POSS_2PL", "POSS_3PL", "OSE", "OTHER", "NUM1", "NUM2", "NUM3", "NUM4", "NUM5", "MANY", "ADJ_NOM"}
MIN_VOTES = 3
MIN_SHARE = 0.2
# A real noun's votes concentrate on one class (pair); function words such as
# na / ni / kufuma spread across many. Keep a noun only if its top class has
# >= MIN_TOP_SHARE of the weight and at most MAX_CLASSES classes survive.
MIN_TOP_SHARE = 0.5
MAX_CLASSES = 3
# Homographs of non-concord words: kwa is also the preposition 'with/at/to'.
AMBIGUOUS_FORMS = {"kwa"}

src = MINING / "round1" / "noun_classes_concord.json"
entries = json.loads(src.read_text(encoding="utf-8"))["entries"]
concord = collections.defaultdict(set)
demonstratives = set()
for e in entries:
    if e["kind"] != "concord" or e["concord_type"] not in TYPES or e.get("confidence") == "low":
        continue
    cls = str(e["kitaita_class"]).strip()
    for f in e["kitaita"]:
        f = norm(re.sub(r"\s*\(.*\)", "", f))  # drop notes like "o- (before -re-)"
        if f and "-" not in f and " " not in f and f not in AMBIGUOUS_FORMS:
            concord[f].add(cls)
            if e["concord_type"].startswith("DEM_"):
                demonstratives.add(f)

votes = collections.defaultdict(collections.Counter)
sources = {}
for name in ("mine.jsonl", "holdout.jsonl"):
    path = MINING / name
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    for line in open(path, encoding="utf-8"):
        toks = [norm(t) for t in json.loads(line)["dav_tok"]]
        for a, b in zip(toks, toks[1:]):
            if a and b in concord and a not in concord:
                for c in concord[b]:
                    votes[a][c] += 1 / len(concord[b])
nouns = {}
for w, v in votes.items():
    total = sum(v.values())
    n_votes = sum(1 for _ in range(int(round(total))))
    if n_votes < MIN_VOTES:
        continue
    keep = sorted(c for c, x in v.items() if x / total >= MIN_SHARE)
    if max(v.values()) / total < MIN_TOP_SHARE or len(keep) > MAX_CLASSES:
        continue
    nouns[w] = {"classes": keep, "votes": round(total, 1)}

OUT.write_text(json.dumps({"_meta": {"built_by": "Code_Experiments/rulebook/build_agreement_data.py",
                                      "date": "2026-11-02", "source_sha256_16": {
                                          "noun_classes_concord.json": hashlib.sha256(src.read_bytes()).hexdigest()[:16],
                                          **sources},
                                      "min_votes": MIN_VOTES, "min_share": MIN_SHARE,
                                      "min_top_share": MIN_TOP_SHARE, "max_classes": MAX_CLASSES,
                                      "ambiguous_forms_excluded": sorted(AMBIGUOUS_FORMS),
                                      "status": "corpus-derived, NOT speaker-verified"},
                           "concord_forms": {f: sorted(c) for f, c in concord.items()},
                           "demonstratives": sorted(demonstratives),
                           "noun_classes": nouns}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
print(OUT, len(concord), "concord forms,", len(nouns), "nouns")
