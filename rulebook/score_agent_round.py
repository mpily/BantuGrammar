"""
score_agent_round.py

2026-11-11. Scores one agent round of the lexicon/rule loop:
  - audit precision: agent verdicts on the RANDOM audit sample (correct / partly /
    wrong), overall and by tier, Wilson CIs -- the round-to-round convergence measure;
  - blocked check: share of support-check blocks the agents judged correct (over-blocking);
  - proposals: counts by confidence, multi-word units, null counterparts;
  - rule notes: listed for review in the round's agent folder (outside git: may quote sentences).
Agent verdicts are an LLM's judgement, not speaker verification.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hf_space"))
from api.stats import rate_ci  # noqa: E402

AGENT_DIR = Path.home() / "languagemodels" / "rulebook_data" / "agents"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    d = AGENT_DIR / f"round{a.round}"
    rows, bad_lines, per_file = [], 0, {}
    for f in sorted(d.glob("out_agent*.jsonl")):
        n = 0
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            o["_file"] = f.name
            rows.append(o)
            n += 1
        per_file[f.name] = n
    rev = [o for o in rows if o.get("type") == "review"]
    audit = [o for o in rev if o.get("why_reviewed") == "random_audit"]
    tiers = {}
    batch_info = {}
    for f in d.glob("batch_agent*.json"):
        for it in json.loads(f.read_text(encoding="utf-8"))["review"]:
            batch_info[(it["sw_lemma"], it["dav_lemma"])] = it["stats"].get("tier")

    def verdicts(xs):
        c = collections.Counter(o.get("verdict") for o in xs)
        n = sum(c.values())
        return {"n": n, "counts": dict(c), "correct_or_partly": rate_ci(c["correct"] + c["partly"], n) if n else None,
                "correct": rate_ci(c["correct"], n) if n else None}

    by_tier = collections.defaultdict(list)
    for o in audit:
        by_tier[batch_info.get((o.get("sw_lemma"), o.get("dav_lemma")), "unknown")].append(o)
    props = [o for o in rows if o.get("type") == "proposal"]
    notes = [o for o in rows if o.get("type") == "rule_note"]
    out = {"round": a.round, "lines_per_file": per_file, "bad_lines": bad_lines,
           "audit": verdicts(audit), "audit_by_tier": {t: verdicts(x) for t, x in by_tier.items()},
           "blocked_check": verdicts([o for o in rev if o.get("why_reviewed") == "blocked_by_support_check"]),
           "low_support_tentative": verdicts([o for o in rev if o.get("why_reviewed") == "low_support_tentative"]),
           "proposals": {"n": len(props), "by_confidence": dict(collections.Counter(o.get("confidence") for o in props)),
                         "fixes_known": sum(bool(o.get("fixes_known")) for o in props),
                         "null_counterpart": sum(o.get("dav_word") is None or o.get("sw_word") is None for o in props),
                         "multiword": sum(len(str(o.get("sw_word") or "").split()) > 1 for o in props)},
           "rule_notes": {"n": len(notes), "by_rule": dict(collections.Counter(o.get("rule_id") for o in notes))}}
    od = HERE / "agent_rounds"
    od.mkdir(exist_ok=True)
    (od / f"round{a.round}_score.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (d / "rule_notes.json").write_text(json.dumps(  # may quote corpus text: outside git
        [{k: o.get(k) for k in ("rule_id", "observation", "pair_ids", "confidence", "_file")} for o in notes],
        indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
