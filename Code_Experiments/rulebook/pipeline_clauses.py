"""
pipeline_clauses.py

2026-11-02. Runs the rule-based pipeline's probe path on each source sentence
of an n-best file (decode_nbest.py) and saves its per-verb subject analysis
(report["clauses"]) for the version-B subject step. Resumable: sentences
already in the output file are skipped. Output (outside git, derived from the
test split): ~/languagemodels/arm2_clauses_<set>.jsonl
Usage: python pipeline_clauses.py --set test_novel
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["val", "test_novel", "test_novel2", "test_novel3"])
    a = ap.parse_args()
    import os
    os.chdir(ROOT / "hf_space")
    from kitaita_e2e_pipeline import build_lookup_tables, load_pages, translate_sentence_with_probes
    tables = build_lookup_tables(*load_pages())
    src = Path.home() / "languagemodels" / f"arm2_nbest_{a.set}.jsonl"
    out = Path.home() / "languagemodels" / f"arm2_clauses_{a.set}.jsonl"
    done = {json.loads(l)["swa"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    with open(out, "a", encoding="utf-8") as f:
        for n, r in enumerate(rows):
            if r["swa"] in done:
                continue
            try:
                clauses, error = translate_sentence_with_probes(r["swa"], tables).get("clauses") or [], None
            except Exception as e:  # one bad input must not stop the run; logged, counted as no clauses
                clauses, error = [], f"{type(e).__name__}: {e}"
            f.write(json.dumps({"swa": r["swa"], "clauses": clauses, "error": error}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(n, len(rows), flush=True)
