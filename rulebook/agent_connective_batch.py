"""
agent_connective_batch.py

2026-11-12, user request: connecting words (lakini, kwa kuwa, ila, kwa hivyo, kama, ...) get their own review in
the final dictionary round (round 5). Many are wrong or missing in rulebook-v4 (na -> wandu 'people', au -> angu,
hivyo / yaani unknown, every multi-word connective unknown: the loop pairs single words only).

For each Swahili connective (single or multi-word, matched as contiguous tokens) the batch gives: number of
clean-train pairs, the v4 lookup result, Kidaw'ida 1-2-grams most associated with it (Dice over pairs), and 10
example pairs (6 dav_swa + 4 Bible, random among pairs of <= 20 Swahili tokens). Clean train only.

Run: ~/languagemodels/bert-env/bin/python rulebook/agent_connective_batch.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agent_translate"))
import attest as A  # noqa: E402

OUT = Path.home() / "languagemodels" / "rulebook_data" / "agents" / "round5"
SEED = 20261112

CONNECTIVES = """lakini | ila | bali | ingawa | ingawaje | ijapokuwa | japo | japokuwa | hata kama | ingawa hivyo |
hata hivyo | kama | kama vile | vile vile | vilevile | kwamba | kuwa | ili | ili kwamba | ili kusudi |
kwa sababu | sababu | kwa kuwa | kwani | kwa maana | maana | kwa sababu ya | kutokana na |
basi | hivyo | kwa hivyo | kwa hiyo | kwahiyo | hivyo basi | ndiyo maana | ndipo | ndio | ndiyo |
hata | au | ama | wala | na | pamoja na | pia | tena | halafu | kisha | baadaye | baada ya | baada ya hapo |
kabla | kabla ya | kabla ya hapo | mpaka | hadi | yaani | endapo | iwapo | ikiwa | kama ... basi |
wakati | wakati wa | wakati huo | ambapo | huku | mbali na | licha ya | zaidi ya | isipokuwa | badala ya |
bila | bila ya | kwa ajili ya | kuhusu | juu ya | kwa mfano | hasa | tu | pekee | je | labda | huenda |
kumbe | aidha | kwanza | mwishowe | hatimaye | kwa kweli | kweli | bado | tayari | sasa | mara | mara moja |
ghafla | kisha basi | lakini pia | wala si | si tu | ndani ya | nje ya | kati ya | chini ya | karibu na"""


def main():
    rng = random.Random(SEED)
    import tools as T
    phrases = [p.strip() for p in CONNECTIVES.replace("\n", " ").split("|") if p.strip() and "..." not in p]
    ps = A.pairs()

    def has(toks, ph):
        n = len(ph)
        return any(toks[i:i + n] == ph for i in range(len(toks) - n + 1))

    dv_df = collections.Counter()
    dv_grams = []
    for r in ps:
        g = set(r["dv"]) | {" ".join(r["dv"][i:i + 2]) for i in range(len(r["dv"]) - 1)}
        dv_grams.append(g)
        dv_df.update(g)
    items = []
    for ph in phrases:
        t = ph.split()
        ids = [i for i, r in enumerate(ps) if has(r["sw"], t)]
        co = collections.Counter()
        for i in ids:
            co.update(dv_grams[i])
        assoc = sorted(((2 * c / (len(ids) + dv_df[g]), g, c) for g, c in co.items() if c >= 3), reverse=True)[:10]
        short = [i for i in ids if len(ps[i]["sw"]) <= 20]
        dav = [i for i in short if ps[i]["src"] == "davswa"]
        bib = [i for i in short if ps[i]["src"] != "davswa"]
        pick = rng.sample(dav, min(6, len(dav))) + rng.sample(bib, min(4, len(bib)))
        lk = T.lookup_word(ph)
        items.append({"key": f"c-{len(items):03d}", "sw": ph, "n_train_pairs": len(ids),
                      "n_dav_swa": sum(ps[i]["src"] == "davswa" for i in ids),
                      "current_dictionary": [{"word": c.get("word") or c["lemma"], "evidence_count": c["evidence_count"]}
                                             for c in (lk.get("candidates") or [])[:4]],
                      "associated_dav": [{"dav": g, "pairs_with_both": c, "dice": round(d, 3)} for d, g, c in assoc],
                      "examples": [{"pair_id": ps[i]["id"], "corpus": ps[i]["src"], "sw": ps[i]["swa"], "dav": ps[i]["dav"]}
                                   for i in pick]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "batch_connectives.json").write_text(json.dumps({"round": 5, "items": items}, ensure_ascii=False, indent=1))
    print(json.dumps({"phrases": len(items), "with_no_pairs": [x["sw"] for x in items if not x["n_train_pairs"]],
                      "size_kb": (OUT / "batch_connectives.json").stat().st_size // 1024}))


if __name__ == "__main__":
    main()
