"""
eval_version_b.py

2026-11-02. Version B (rule book on arm 2's draft) vs version A (arm 2's top-1),
on the n-best files written by decode_nbest.py. Each correction step is scored
alone (one change per experiment) and all four together.

Per condition: sentences changed, mean chrF (api.stats.chrf on normalize_kitaita,
as /api/progress), paired B-A difference with a bootstrap 95% CI, and on the
CHANGED sentences how many got better / worse / same. Also the agreement
checker's false-alarm rate on the real references.

The rules have no parameters tuned on this set. Usage:
  python eval_version_b.py --set test_novel   (the reported, clean set)
  python eval_version_b.py --set val          (debugging only: contaminated)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))

from api.stats import chrf, mean_ci  # noqa: E402
from api.util import normalize_kitaita  # noqa: E402
from rulebook_correct import ALL_STEPS, correct, violations  # noqa: E402

CONDITIONS = [(s,) for s in ALL_STEPS] + [ALL_STEPS]


def score(h, r):
    return chrf(normalize_kitaita(h), normalize_kitaita(r))


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(k / n, 4), round(c - h, 4), round(c + h, 4)]


def boot(d, seed=20261102, B=2000):
    if not d:
        return None
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(B))
    return [round(sum(d) / len(d), 3), round(m[int(0.025 * B)], 3), round(m[int(0.975 * B)], 3)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["val", "test_novel"])
    a = ap.parse_args()
    path = Path.home() / "languagemodels" / f"arm2_nbest_{a.set}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    base = [score(r["nbest"][0]["hyp"], r["ref"]) for r in rows]
    out = {"set": a.set, "n": len(rows), "A_chrf": mean_ci(base), "conditions": {}, "examples": {}}
    for steps in CONDITIONS:
        name = "+".join(steps) if len(steps) > 1 else steps[0]
        diffs, changed_diffs, changed, better, worse, ex = [], [], 0, 0, 0, []
        for r, b0 in zip(rows, base):
            res = correct(r["swa"], [c["hyp"] for c in r["nbest"]], steps)
            b = score(res["kitaita"], r["ref"])
            diffs.append(b - b0)
            if res["kitaita"] != r["nbest"][0]["hyp"]:
                changed += 1
                changed_diffs.append(b - b0)
                better += b > b0
                worse += b < b0
                if len(ex) < 8:
                    ex.append({"swa": r["swa"], "A": r["nbest"][0]["hyp"], "B": res["kitaita"], "ref": r["ref"],
                               "changes": res["changes"], "chrf_diff": round(b - b0, 1)})
        out["conditions"][name] = {
            "changed": changed, "changed_share": wilson(changed, len(rows)),
            "B_chrf": mean_ci([b0 + d for b0, d in zip(base, diffs)]),
            "paired_B_minus_A_all": boot(diffs),
            "paired_B_minus_A_changed_only": boot(changed_diffs),
            "changed_better": better, "changed_worse": worse, "changed_same": changed - better - worse}
        out["examples"][name] = ex
        print(name, json.dumps(out["conditions"][name]))
    refs_flagged = sum(bool(violations(r["ref"])) for r in rows)
    out["checker_false_alarm_refs"] = wilson(refs_flagged, len(rows))
    out["checker_flag_rate_A"] = wilson(sum(bool(violations(r["nbest"][0]["hyp"])) for r in rows), len(rows))
    print("refs with an agreement 'violation':", out["checker_false_alarm_refs"],
          "| A top-1:", out["checker_flag_rate_A"])
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    out["config"] = {"git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                               cwd=ROOT).stdout.strip(),
                     "nbest_file_sha256_16": h(path), "bootstrap": {"B": 2000, "seed": 20261102},
                     "files_sha256_16": {f: h(ROOT / "hf_space" / f) for f in
                                         ("rulebook_correct.py", "rulebook_flagger.py", "kitaita_agreement_data.json",
                                          "kitaita_training_vocab.json", "kitaita_mined_lexicon_round1.json")}}
    (Path(__file__).parent / f"eval_version_b_{a.set}_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
