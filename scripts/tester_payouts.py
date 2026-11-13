"""
tester_payouts.py -- who to pay, from the Space's feedback (2026-11-13, paid testers; hf_space/testers.py).

A tester earns one payment per 100 COMPLETE submissions (all three translations rated + own Kitaita
translation); each Swahili sentence counts once per tester. Paid batches are recorded in payouts/ledger.jsonl
in the private testers dataset, so a batch is never paid twice.

  python scripts/tester_payouts.py                       # table: tester, phone, complete, earned, paid, owed
  python scripts/tester_payouts.py --mark-paid ph_abc 1  # after paying: record 1 batch for that tester

Needs HF_TOKEN with read (and, for --mark-paid, write) access to both private datasets. Prints phone numbers:
run it on your own machine only and don't paste the output anywhere public.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hf_space"))
import testers  # noqa: E402

FEEDBACK = "mpily/kitaita-translation-feedback"


def _read_all(api, repo, prefix):
    out = []
    for f in api.list_repo_files(repo, repo_type="dataset"):
        if f.startswith(prefix):
            try:
                out.append((f, Path(api.hf_hub_download(repo, f, repo_type="dataset")).read_text(encoding="utf-8")))
            except Exception as e:
                print(f"skip {f}: {type(e).__name__}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark-paid", nargs=2, metavar=("TESTER_ID", "BATCHES"))
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = testers.TESTERS_DATASET_REPO
    ledger = collections.Counter()
    for _, txt in _read_all(api, repo, "payouts/"):
        for line in txt.splitlines():
            if line.strip():
                r = json.loads(line)
                ledger[r["tester_id"]] += int(r["batches"])
    if a.mark_paid:
        tid, n = a.mark_paid[0], int(a.mark_paid[1])
        rec = {"tester_id": tid, "batches": n, "paid_at": datetime.now(timezone.utc).isoformat()}
        api.upload_file(path_or_fileobj=io.BytesIO((json.dumps(rec) + "\n").encode()),
                        path_in_repo=f"payouts/{rec['paid_at'].replace(':', '-')}_{tid}.jsonl",
                        repo_id=repo, repo_type="dataset")
        print("recorded", rec)
        return
    phones = {json.loads(t)["tester_id"]: json.loads(t)["phone"] for _, t in _read_all(api, repo, "testers/")}
    done = collections.defaultdict(set)
    partial = collections.Counter()
    for f, txt in _read_all(api, FEEDBACK, "feedback_ab/"):
        if "_ph_" not in f:
            continue
        r = json.loads(txt)
        tid = r.get("tester_id")
        if not tid:
            continue
        ratings = {k: (r.get("arms", {}).get(k) or {}).get("rating") for k in ("A", "C", "E")}
        if testers.is_complete(ratings, r.get("native_speaker_correction")):
            done[tid].add(" ".join((r.get("swahili_input") or "").lower().split()).rstrip(" .!?"))
        else:
            partial[tid] += 1
    print(f"{'tester':30} {'phone':15} {'complete':>8} {'partial':>7} {'earned':>6} {'paid':>4} {'owed':>4}")
    for tid in sorted(set(done) | set(partial) | set(phones)):
        n = len(done[tid])
        earned = n // testers.PAY_EVERY
        print(f"{tid:30} {phones.get(tid, '?'):15} {n:8} {partial[tid]:7} {earned:6} {ledger[tid]:4} "
              f"{max(0, earned - ledger[tid]):4}")


if __name__ == "__main__":
    main()
