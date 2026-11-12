"""
build_vocab.py

2026-11-12. Kidaw'ida word counts over the clean train split only (rulebook_data/train.jsonl, repeats
counted once), for the tools' attestation check (agent_translate/tools.py). Replaces
hf_space/kitaita_training_vocab.json for this purpose: that file was built from the contaminated
pool (status 2026-11-11 step 1). Counts only, no sentences.
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attest as A  # noqa: E402

c = collections.Counter(w for r in A.pairs() for w in r["dv"])
out = HERE / "dav_vocab.json"
out.write_text(json.dumps({"_meta": {"built_by": "rulebook/build_vocab.py", "train_sha16": A.sha16(A.TRAIN),
                                     "n_types": len(c), "n_tokens": sum(c.values())},
                           "vocab": dict(c.most_common())}, ensure_ascii=False))
print(len(c), sum(c.values()))
