"""Summary of the blind three-arm native-speaker evaluation on the Space (2026-11-15 onward).

Arms: A = fine-tuned NLLB-600M (arm 2), C = Claude Haiku + rule-book tools, E = Claude Haiku zero-shot.
Reads feedback_ab/ records from the private feedback dataset (only records with a random display_order,
i.e. from 2026-11-15, when the shuffle went live), and writes one JSON with the full config, the list of
record files used (snapshot) and their hash, and:
  - mean stars per arm (t 95% CI), share rated >= 4 (Wilson CI), chrF vs the tester's own translation
  - paired A - C star difference (bootstrap 95% CI), wins / ties / losses
  - the same per tester, per register (storybook / news / encyclopedia) and per time batch
  - mean rating per display slot (position check)
  - agreement between testers on sentences rated by >= 2 testers
  - how often the tester's translation is an exact copy of an arm's output
Repeated saves (same tester, sentence, ratings and translation within 120 s of the kept record: one click saved
several times, seen 2026-11-17) are dropped and counted; a sentence the button gave a tester again later is a
separate rating and kept.
Tester ids are salted hashes; no phone numbers are read or written.

Usage: bert-env python scripts/blind_eval_summary.py [--since 2026-11-15] [--out Results/blind_eval_<date>.json]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hf_space"))
from api.stats import chrf, mean_ci  # noqa: E402

REPO = "mpily/kitaita-translation-feedback"
ARMS = {"A": "fine-tuned NLLB-600M (arm 2)", "C": "Haiku + rule-book tools", "E": "Haiku zero-shot"}


def norm(s: str | None) -> str:
    return " ".join((s or "").lower().split()).rstrip(" .!?")   # = testers.norm_sentence


def rating(r: dict, arm: str) -> int | None:
    x = r["arms"].get(arm, {}).get("rating")
    return int(x) if str(x or "").isdigit() else None


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def boot_ci(x: list[float], seed: int, n_boot: int) -> list[float] | None:
    if len(x) < 3:
        return None
    rng = random.Random(seed)
    bs = sorted(st.mean(rng.choices(x, k=len(x))) for _ in range(n_boot))
    return [round(bs[int(0.025 * n_boot)], 4), round(bs[int(0.975 * n_boot)], 4)]


def summarize(rr: list[dict], seed: int, n_boot: int) -> dict:
    out = {"n_records": len(rr), "arms": {}}
    for a in ARMS:
        xs = [rating(r, a) for r in rr if rating(r, a)]
        cs = [chrf(r["arms"][a]["output"], r["native_speaker_correction"]) for r in rr
              if r.get("native_speaker_correction") and (r["arms"][a].get("output") or "").strip()]
        k = sum(x >= 4 for x in xs)
        out["arms"][a] = {"stars": mean_ci(xs), "share_ge4": round(k / len(xs), 4) if xs else None,
                          "share_ge4_wilson95": wilson(k, len(xs)), "chrf_vs_tester": mean_ci(cs)}
    d = [rating(r, "A") - rating(r, "C") for r in rr if rating(r, "A") and rating(r, "C")]
    out["paired_A_minus_C"] = {"n": len(d), "mean": round(st.mean(d), 4) if d else None,
                               "bootstrap95": boot_ci(d, seed, n_boot),
                               "A_wins": sum(v > 0 for v in d), "ties": sum(v == 0 for v in d),
                               "C_wins": sum(v < 0 for v in d)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-15")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=str(ROOT / ".cache" / "feedback_snapshot"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=80)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", allow_patterns=["feedback_ab/*"], local_dir=args.cache,
                      max_workers=16)
    files = sorted(glob.glob(f"{args.cache}/feedback_ab/*.json"))
    recs = []
    for f in files:
        r = json.load(open(f, encoding="utf-8"))
        if r.get("timestamp", "") >= args.since and r.get("display_order") and "arms" in r:
            recs.append((Path(f).name, r))
    recs.sort(key=lambda x: x[1]["timestamp"])
    kept, last_seen, dropped = [], {}, []
    for n, r in recs:
        key = (r.get("tester_id") or "anon", norm(r.get("swahili_input")),
               tuple(sorted((a, (v or {}).get("rating")) for a, v in r["arms"].items())),
               (r.get("native_speaker_correction") or "").strip())
        t = datetime.fromisoformat(r["timestamp"])
        if key in last_seen and (t - last_seen[key]).total_seconds() <= 120:
            dropped.append(n)
            continue
        last_seen[key] = t
        kept.append((n, r))
    recs = kept
    names = [n for n, _ in recs]
    R = [r for _, r in recs]

    res = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {**vars(args), "dataset": REPO, "arms": ARMS,
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "stats_sha256": hashlib.sha256((ROOT / "hf_space/api/stats.py").read_bytes()).hexdigest()},
        "snapshot": {"n_files": len(names), "files_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                     "first": R[0]["timestamp"] if R else None, "last": R[-1]["timestamp"] if R else None,
                     "files": names, "dropped_repeated_saves": dropped},
        "systems_seen": sorted({f"{a}={r['arms'][a].get('system')}" for r in R for a in r["arms"]}),
        "overall": summarize(R, args.seed, args.n_boot),
    }
    testers = sorted({r.get("tester_id") or "none" for r in R})
    res["per_tester"] = {t: {**summarize([r for r in R if (r.get("tester_id") or "none") == t], args.seed, args.n_boot),
                             "complete_for_payment": sum(bool(r.get("complete_for_payment")) for r in R
                                                         if (r.get("tester_id") or "none") == t)}
                         for t in testers}
    res["per_register"] = {str(g): summarize([r for r in R if (r.get("test_sentence") or {}).get("register") == g],
                                             args.seed, args.n_boot)
                           for g in ["storybook", "news", "encyclopedia", None]}
    res["per_batch"] = [{"first": R[i]["timestamp"], **summarize(R[i:i + args.batch], args.seed, args.n_boot)}
                        for i in range(0, len(R), args.batch)]
    res["per_slot"] = {str(p + 1): mean_ci([rating(r, r["display_order"][p]) for r in R
                                            if rating(r, r["display_order"][p])]) for p in range(3)}

    # agreement: sentences rated by >= 2 different testers; pairs of (tester, tester) ratings per arm
    by_sent = defaultdict(dict)
    for r in R:
        by_sent[norm(r["swahili_input"])][r.get("tester_id") or "none"] = r
    multi = {s: v for s, v in by_sent.items() if len(v) >= 2}
    agree = {"n_sentences_multi": len(multi), "n_unique_sentences": len(by_sent)}
    for a in list(ARMS) + ["A-C"]:
        pairs = []
        for v in multi.values():
            for r1, r2 in itertools.combinations(v.values(), 2):
                f = (lambda r: rating(r, "A") - rating(r, "C") if rating(r, "A") and rating(r, "C") else None) \
                    if a == "A-C" else (lambda r, a=a: rating(r, a))
                x, y = f(r1), f(r2)
                if x is not None and y is not None:
                    pairs.append((x, y))
        agree[a] = {"n_pairs": len(pairs),
                    "exact_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
                    "within_1": round(sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs), 4) if pairs else None,
                    "mean_abs_diff": round(st.mean(abs(x - y) for x, y in pairs), 4) if pairs else None}
        if a == "A-C" and pairs:
            agree[a]["same_sign"] = round(sum((x > 0) == (y > 0) and (x < 0) == (y < 0) for x, y in pairs)
                                          / len(pairs), 4)
    res["inter_tester"] = agree
    with_corr = [r for r in R if r.get("native_speaker_correction")]
    res["copy_check"] = {"n_with_translation": len(with_corr),
                         **{a: sum(norm(r["arms"][a].get("output")) == norm(r["native_speaker_correction"])
                                   for r in with_corr) for a in ARMS}}

    out = Path(args.out or ROOT / f"Results/blind_eval_{datetime.now().date().isoformat()}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    o = res["overall"]
    print(f"wrote {out}: n={o['n_records']} (dropped {len(dropped)} repeated saves), A-C {o['paired_A_minus_C']}")
    print(json.dumps(res["inter_tester"], indent=1))
    print(json.dumps(res["copy_check"]))


if __name__ == "__main__":
    main()
