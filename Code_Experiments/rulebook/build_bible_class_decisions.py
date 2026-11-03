"""
build_bible_class_decisions.py

2026-11-03. User decision: "use the bible corpus to decide in case of
variation as it is more standardized". Two Bible-only tables for the
subject-class resolver (hf_space/subject_class_resolver.py):

1. concord_pairs: noun -> class PAIR voted by the concord word after it
   (associative, demonstratives, possessives...: kitaita_agreement_data.json
   concord_forms), counting 9 and 10 (etc.) as one pair, so number-invariant
   nouns are not split (chia: ya-/iyo- 485, ra-/rose- 287, cha 38 -> 9/10).
   Within the pair: one member if it has >= 80% of the pair's votes, else
   "invariant" (the Swahili verb's number picks the member).
2. class14_sm: wu- nouns -> u- or ghu- on the verb after them (u-/o- vs
   ghu-/gho- + re/ka/cha/ki/de). The two registers disagree on class 14
   overall (modern ghu- 13 / u- 2; Bible u- 84 / ghu- 78), so per noun.

Counts only, no sentences: hf_space/kitaita_bible_class_decisions.json.
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
OUT = ROOT / "hf_space" / "kitaita_bible_class_decisions.json"
PAIR_OF = {"1": "1/2", "2": "1/2", "3": "3/4", "4": "3/4", "5": "5/6", "6": "5/6", "7": "7/8", "8": "7/8",
           "9": "9/10", "10": "9/10", "11": "11/10", "14": "14", "15": "15"}
MEMBER_SHARE = 0.8
VERB14 = re.compile(r"^(gho|ghu|o|u)(re|ka|cha|ki|de)[a-z]{2,}")

if __name__ == "__main__":
    concord = {f: c for f, c in json.loads((ROOT / "hf_space" / "kitaita_agreement_data.json").read_text(
        encoding="utf-8"))["concord_forms"].items()}
    pair_votes = collections.defaultdict(collections.Counter)
    member_votes = collections.defaultdict(collections.Counter)
    v14 = collections.defaultdict(collections.Counter)
    sources = {}
    for name in ("mine.jsonl", "holdout.jsonl"):
        path = MINING / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if not row["id"].startswith("bible:"):
                continue
            toks = [norm(t) for t in row["dav_tok"]]
            for a, b in zip(toks, toks[1:]):
                if not a or a in concord:
                    continue
                if b in concord:
                    pairs = {PAIR_OF.get(c) for c in concord[b]} - {None}
                    for p in pairs:
                        pair_votes[a][p] += 1 / len(pairs)
                    for c in concord[b]:
                        member_votes[a][c] += 1 / len(concord[b])
                if a.startswith("wu") and len(a) > 4:
                    m = VERB14.match(b)
                    if m:
                        v14[a]["ghu" if m.group(1).startswith("gh") else "u"] += 1
    concord_pairs = {}
    for n, pv in pair_votes.items():
        total = sum(pv.values())
        pair, k = pv.most_common(1)[0]
        members = {c: v for c, v in member_votes[n].items() if PAIR_OF.get(c) == pair}
        mtot = sum(members.values()) or 1
        top_c, top_v = max(members.items(), key=lambda x: x[1]) if members else (None, 0)
        concord_pairs[n] = {"pair": pair, "votes": round(total, 1), "share": round(k / total, 3),
                            "member": top_c if top_v / mtot >= MEMBER_SHARE else "invariant",
                            "members": {c: round(v, 1) for c, v in members.items()},
                            "pairs": {q: round(v, 1) for q, v in pv.items()}}
    class14 = {n: {"u": c["u"], "ghu": c["ghu"]} for n, c in v14.items()}
    OUT.write_text(json.dumps({"_meta": {"built_by": "Code_Experiments/rulebook/build_bible_class_decisions.py",
                                         "date": "2026-11-03", "source_sha256_16": sources,
                                         "member_share": MEMBER_SHARE, "text": "Bible rows only (id bible:*)",
                                         "status": "Bible counts, NOT speaker-verified"},
                               "concord_pairs": concord_pairs, "class14_sm": class14},
                              ensure_ascii=False, sort_keys=True), encoding="utf-8")
    for w in ("chia", "nguku", "vindo", "wughoma", "wurumwengu", "wurigha", "wulacha", "ijeshi", "nyamandu"):
        print(w, concord_pairs.get(w), class14.get(w))
    print(len(concord_pairs), "nouns with Bible concord votes,", len(class14), "wu- nouns with verb counts")
