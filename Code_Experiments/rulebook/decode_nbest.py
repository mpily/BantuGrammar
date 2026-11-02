"""
decode_nbest.py

2026-11-02. Decode arm 2 (fine-tuned NLLB, dav_swa + Bible) with beam 4 and
keep all 4 candidates + scores, for the version-B rule book (re-ranking needs
the n-best list; the saved test_predictions.jsonl only has the top-1).

Sets:
  val        -- arm 2's own 500-pair validation carve-out (reconstructed with
                nllb_finetune_dav.py's split, seed 20261028). Contaminated for
                the rule book (its sentences are in the mining data), so used
                for debugging only.
  test_novel -- a seeded random sample of the novel test subset (Swahili side
                never in train): clean for the model and the rule book.
Writes ~/languagemodels/arm2_nbest_<set>.jsonl (corpus text: not in git).
Resumable per item (CLAUDE.md: checkpoint runs > 10 min).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code_Experiments"))
from nllb_finetune_dav import _key, clean, load_split  # noqa: E402

MODEL = Path.home() / "languagemodels" / "arm2_davswa_hf"
SPLIT_SEED = 20261028
SAMPLE_SEED = 20261102


def sets(n_test: int):
    train_all, _ = load_split("train")
    test, _ = load_split("test")
    train_all, test = clean(train_all), clean(test)
    order = list(range(len(train_all)))
    random.Random(SPLIT_SEED).shuffle(order)
    val = [train_all[i] for i in order[:500]]
    train_keys = {_key(r["swa"]) for r in train_all}
    novel = [r for r in test if _key(r["swa"]) not in train_keys]
    return {"val": val, "test_novel": random.Random(SAMPLE_SEED).sample(novel, n_test)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["val", "test_novel"])
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    rows = sets(a.n_test)[a.set]
    out = Path.home() / "languagemodels" / f"arm2_nbest_{a.set}.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    tok = AutoTokenizer.from_pretrained(MODEL, src_lang="swh_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    tgt = tok.convert_tokens_to_ids("dav_Latn")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    with torch.no_grad(), open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(rows), a.batch):
            b = rows[i:i + a.batch]
            enc = tok([r["swa"] for r in b], return_tensors="pt", padding=True, truncation=True, max_length=128)
            g = model.generate(**enc, forced_bos_token_id=tgt, num_beams=a.beams, num_return_sequences=a.beams,
                               max_new_tokens=128, output_scores=True, return_dict_in_generate=True)
            hyps = tok.batch_decode(g.sequences, skip_special_tokens=True)
            scores = g.sequences_scores.tolist()
            for j, r in enumerate(b):
                cands = [{"hyp": hyps[j * a.beams + k], "score": round(scores[j * a.beams + k], 4)}
                         for k in range(a.beams)]
                f.write(json.dumps({"id": r["id"], "swa": r["swa"], "ref": r["dav"], "nbest": cands},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"{a.set}: {len(done) + i + len(b)}", flush=True)
