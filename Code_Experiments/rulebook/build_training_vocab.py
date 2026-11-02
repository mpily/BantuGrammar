"""
build_training_vocab.py

2026-11-02. Writes hf_space/kitaita_training_vocab.json for the rule-book
flagger (hf_space/rulebook_flagger.py): word counts only, no sentences.
  vocab     -- Kidaw'ida token counts over MINE + HOLDOUT (= dav_swa train +
               Bible NEN pairs, the text arm 2 was trained on)
  davswa_dav / davswa_swa -- number of davswa pairs whose Kidaw'ida / Swahili
               side contains each token (used to decide whether a Swahili
               word is also established Kidaw'ida usage, e.g. na, kwa, muhimu)
Input: ~/languagemodels/mining/{mine,holdout}.jsonl (not in git).
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
from rulebook_flagger import norm  # noqa: E402

MINING = Path.home() / "languagemodels" / "mining"
OUT = ROOT / "hf_space" / "kitaita_training_vocab.json"

vocab, dav_n, swa_n, sources = collections.Counter(), collections.Counter(), collections.Counter(), {}
for name in ("mine.jsonl", "holdout.jsonl"):
    path = MINING / name
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        dt = [norm(t) for t in r["dav_tok"]]
        vocab.update(t for t in dt if t)
        if r["src"] == "davswa":
            dav_n.update({t for t in dt if t})
            swa_n.update({norm(t) for t in r["swa_tok"]} - {""})
swa_kept = {t: c for t, c in swa_n.items() if dav_n[t]}  # only words that also occur on the Kidaw'ida side matter
OUT.write_text(json.dumps({"_meta": {"built_by": "Code_Experiments/rulebook/build_training_vocab.py",
                                      "date": "2026-11-02", "source_sha256_16": sources,
                                      "note": "token counts only; derived from dav_swa train + Bible NEN pairs"},
                           "vocab": dict(vocab), "davswa_dav": dict(dav_n), "davswa_swa": swa_kept},
                          ensure_ascii=False, sort_keys=True), encoding="utf-8")
print(OUT, len(vocab), "words")
