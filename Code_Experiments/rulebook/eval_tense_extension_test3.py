"""
eval_tense_extension_test3.py

2026-11-03. The pre-registered score (docs/status.md 2026-11-03 cont. 6) of the
frozen tense + extension steps on test_novel3 (500 novel test sentences, never
inspected). Conditions: A, B_default (Space now), B_new = B_default + tense +
extension. Writes the rule edits, shuffled A/B order per edit and without
saying which form the rule produced, for an agent judgement
(~/languagemodels/audit/test3_edits_blind.jsonl; key kept separately).
"""
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
sys.path.insert(0, str(Path(__file__).parent))
from eval_version_b import boot, mean_ci, score, wilson  # noqa: E402
from rulebook_correct import DEFAULT_STEPS, correct  # noqa: E402

lm = Path.home() / "languagemodels"
rows = [json.loads(l) for l in open(lm / "arm2_nbest_test_novel3.jsonl", encoding="utf-8")]
NEW = DEFAULT_STEPS + ("tense", "extension")
A = [score(r["nbest"][0]["hyp"], r["ref"]) for r in rows]
res_def = [correct(r["swa"], [c["hyp"] for c in r["nbest"]], DEFAULT_STEPS) for r in rows]
res_new = [correct(r["swa"], [c["hyp"] for c in r["nbest"]], NEW) for r in rows]
Bd = [score(x["kitaita"], r["ref"]) for x, r in zip(res_def, rows)]
Bn = [score(x["kitaita"], r["ref"]) for x, r in zip(res_new, rows)]
diff = [n - d for n, d in zip(Bn, Bd)]
changed = [i for i, (x, y) in enumerate(zip(res_def, res_new)) if x["kitaita"] != y["kitaita"]]
out = {"set": "test_novel3", "n": len(rows), "A": mean_ci(A), "B_default": mean_ci(Bd), "B_new": mean_ci(Bn),
       "B_new_minus_B_default": boot(diff), "B_new_minus_A": boot([n - a for n, a in zip(Bn, A)]),
       "B_default_minus_A": boot([d - a for d, a in zip(Bd, A)]),
       "sentences_changed_by_tense_extension": len(changed),
       "changed_better": sum(diff[i] > 0 for i in changed), "changed_worse": sum(diff[i] < 0 for i in changed),
       "edits": {k: sum(1 for x in res_new for c in x["changes"] if c[0] == k) for k in ("tense", "extension")}}
rng = random.Random(20261103)
blind, key = [], []
for x, r in zip(res_new, rows):
    for kind, desc in x["changes"]:
        if kind not in ("tense", "extension"):
            continue
        old, new = desc.split(" (")[0].split(" -> ")
        pair = [old, new]
        rng.shuffle(pair)
        eid = f"e{len(blind) + 1}"
        blind.append({"edit_id": eid, "swahili": r["swa"], "kitaita_sentence": r["nbest"][0]["hyp"],
                      "word_option_1": pair[0], "word_option_2": pair[1]})
        key.append({"edit_id": eid, "rule_form": new, "draft_form": old, "kind": kind, "change": desc,
                    "reference": r["ref"]})
(lm / "audit" / "test3_edits_blind.jsonl").write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in blind))
(lm / "audit" / "test3_edits_key.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in key))
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
out["config"] = {"git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
                 "nbest_sha256_16": h(lm / "arm2_nbest_test_novel3.jsonl"), "steps_new": list(NEW),
                 "bootstrap": {"B": 2000, "seed": 20261102}}
print(json.dumps(out, indent=1))
(Path(__file__).parent / "eval_tense_extension_test3_results.json").write_text(json.dumps(out, indent=1))
