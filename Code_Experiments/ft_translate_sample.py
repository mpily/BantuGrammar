# -*- coding: utf-8 -*-
"""
ft_translate_sample.py

2026-11-01. Translate a seeded random sample of Swahili sentences from a
{swa, dav} JSONL with an exported fine-tuned model, and score each output
against its Kidaw'ida reference (same chrF / normalisation as the dav_swa
evaluations). For inspecting failures, e.g. on Bible verses that arm 1 never
saw in training (out-of-domain and a different register).

Output JSONL: {id, swa, ref, hyp, chrf}. Corpus text: keep it out of git.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent / "hf_space")]

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from api.stats import chrf, mean_ci
from api.util import normalize_kitaita


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--src-filter", default=None, help="keep only rows whose 'src' equals this")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--max-words", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-beams", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    rows = [r for r in rows if (a.src_filter is None or r.get("src") == a.src_filter)
            and 3 <= len(r["swa"].split()) <= a.max_words]
    rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    tok = AutoTokenizer.from_pretrained(a.model, src_lang="swh_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).eval()
    tgt = tok.convert_tokens_to_ids("dav_Latn")
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            b = rows[i: i + 8]
            enc = tok([r["swa"] for r in b], return_tensors="pt", padding=True)
            gen = model.generate(**enc, forced_bos_token_id=tgt, num_beams=a.num_beams, max_new_tokens=128)
            for r, h in zip(b, tok.batch_decode(gen, skip_special_tokens=True)):
                out.append({"id": r["id"], "swa": r["swa"], "ref": r["dav"], "hyp": h,
                            "chrf": round(chrf(normalize_kitaita(h), normalize_kitaita(r["dav"])), 1)})
            print(f"{len(out)}/{len(rows)}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    m = mean_ci([o["chrf"] for o in out])
    print(f"chrF mean {m['mean']:.2f} {m['ci95']} n={m['n']}")


if __name__ == "__main__":
    main()
