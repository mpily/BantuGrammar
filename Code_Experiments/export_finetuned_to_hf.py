# -*- coding: utf-8 -*-
"""
export_finetuned_to_hf.py

2026-10-29. Turns a nllb_finetune_dav.py checkpoint (best_model.pt, a raw
state_dict) into a standard Hugging Face model folder -- config, weights,
tokenizer with the added `dav_Latn` token, model card -- so the model can
be loaded anywhere with AutoModelForSeq2SeqLM.from_pretrained() and tested
independently. No retraining: the base model is rebuilt exactly as
nllb_finetune_dav.py built it (same add_target_language), then the
fine-tuned weights are loaded with strict=True.

Equivalence check (always run): the exported folder is reloaded from disk
and used through the PLAIN tokenizer API (src_lang="swh_Latn",
forced_bos_token_id=dav_Latn) -- the way an outside user would call it --
on sentences from the run's own test_predictions.jsonl, with the run's
decoding settings (beam/max-len from its results.json). Every output must
match the saved Colab prediction exactly; any mismatch is reported.

Pushing to the Hub is a separate, explicit step (--push-repo); default is
a private repo. Licences: NLLB-200 is CC-BY-NC-4.0 (inherited by the
fine-tune: non-commercial only); dav_swa is CC-BY-4.0 (cite Mbogho et al.
2024). A run trained on --extra-pairs Bible text must not be pushed
publicly: that text is rights-reserved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nllb_finetune_dav import SRC_LANG, TGT_LANG, add_target_language, encode

CARD = """---
license: cc-by-nc-4.0
base_model: {base}
language:
- sw
- dav
tags:
- translation
- nllb
- kidawida
- kitaita
datasets:
- thinkKenya/kenyan-low-resource-language-data
pipeline_tag: translation
---

# NLLB-200 ({base_name}) fine-tuned for Kiswahili → Kidaw'ida (Kitaita)

Research checkpoint. **Machine translation — not reviewed by native
speakers; do not present its output as verified Kidaw'ida.**

- Base model: `{base}` (CC-BY-NC-4.0, so this fine-tune is **non-commercial only**).
- Training data: `dav_swa` config of
  [thinkKenya/kenyan-low-resource-language-data](https://huggingface.co/datasets/thinkKenya/kenyan-low-resource-language-data)
  (CC-BY-4.0), train split minus a 500-sentence validation carve-out
  ({n_train} pairs){extra_note}.
- Kidaw'ida has no NLLB language code; a new token `dav_Latn` was added,
  initialised from `swh_Latn`.

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("<this repo>", src_lang="swh_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("<this repo>")
enc = tok(["Moto wa kuni umezimwa"], return_tensors="pt")
out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids("dav_Latn"),
                     num_beams={beams}, max_new_tokens={max_len})
print(tok.batch_decode(out, skip_special_tokens=True))
```

## Results (dav_swa test split, sentence-level chrF, apostrophe-normalised)

| Test set | n | chrF [95% CI] | exact match |
|---|---|---|---|
| full test | {n_test} | {chrf_all} | {exact_all} |
| novel (Swahili side never in train) | {n_novel} | {chrf_novel} | {exact_novel} |

Note: 34% of the released test split's Swahili sentences also occur in
train; the novel subset is the honest number. In-domain only — not
evaluated on other domains or by native speakers.

## Citation (training data)

```
@dataset{{mbogho_2024_low_resource_language_data,
  author       = {{Mbogho, Audrey and Kipkebut, Andrew and Wanzare, Lilian and
                  Awuor, Quin and Oloo, Vivian and Lugano, Rose}},
  title        = {{{{Low-Resource Language Data: Parallel Corpora for
                   Kiswahili and Kidaw'ida, Kalenjin, and Dholuo}}}},
  year         = 2024,
  publisher    = {{Tech Innovators Network (THiNK) on Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/thinkKenya/kenyan-low-resource-language-data}}}}
}}
```
"""


def fmt_ci(m: dict) -> str:
    return f"{m['mean']:.2f} [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}]"


def fmt_rate(r: dict) -> str:
    return f"{r['k']}/{r['n']} = {100 * r['rate']:.1f}%"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="folder holding best_model.pt, results.json, test_predictions.jsonl")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-check", required=True, type=int, help="test sentences to re-translate for the equivalence check")
    p.add_argument("--push-repo", default=None, help="e.g. mpily/nllb-600m-swh-dav (omit to only export locally)")
    p.add_argument("--public", action="store_true", help="push as a public repo (default private)")
    args = p.parse_args()

    results = json.loads((args.run_dir / "results.json").read_text())
    cfg = results["config"]
    if args.push_repo and args.public and cfg.get("extra_pairs"):
        sys.exit("refusing a PUBLIC push: this run was trained on --extra-pairs (Bible text is rights-reserved)")

    # rebuild exactly as nllb_finetune_dav.py did, then load the fine-tuned weights
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"])
    add_target_language(model, tok)
    state = torch.load(args.run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.generation_config.max_length = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    extra = cfg.get("extra_pairs")
    (args.out_dir / "README.md").write_text(CARD.format(
        base=cfg["model"], base_name=cfg["model"].split("/")[-1], n_train=cfg["n_train"],
        extra_note=(f", plus {extra['n_kept']} extra pairs from {Path(extra['path']).name}" if extra else ""),
        beams=cfg["num_beams"], max_len=cfg["max_len"],
        n_test=results["test"]["n"], chrf_all=fmt_ci(results["test"]["chrf"]),
        exact_all=fmt_rate(results["test"]["exact_match"]),
        n_novel=results["test_novel"]["n"], chrf_novel=fmt_ci(results["test_novel"]["chrf"]),
        exact_novel=fmt_rate(results["test_novel"]["exact_match"])))
    (args.out_dir / "training_config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    print(f"exported to {args.out_dir}")

    # equivalence check: reload from disk, plain tokenizer API, compare with saved predictions
    tok2 = AutoTokenizer.from_pretrained(args.out_dir, src_lang=SRC_LANG)
    model2 = AutoModelForSeq2SeqLM.from_pretrained(args.out_dir).eval()
    tgt_id = tok2.convert_tokens_to_ids(TGT_LANG)
    preds = [json.loads(l) for l in open(args.run_dir / "test_predictions.jsonl", encoding="utf-8")][: args.n_check]
    for o in preds[:3]:  # the plain tokenizer must produce what training used: [swh_Latn] ... [eos]
        manual = encode(tok, o["swa"], tok.convert_tokens_to_ids(SRC_LANG), cfg["max_len"])
        assert tok2(o["swa"])["input_ids"] == manual, "tokenizer layout differs from training"
    mismatches = []
    with torch.no_grad():
        for i in range(0, len(preds), 8):
            batch = preds[i: i + 8]
            enc = tok2([o["swa"] for o in batch], return_tensors="pt", padding=True)
            out = model2.generate(**enc, forced_bos_token_id=tgt_id, num_beams=cfg["num_beams"],
                                  max_new_tokens=cfg["max_len"])
            for o, h in zip(batch, tok2.batch_decode(out, skip_special_tokens=True)):
                if h != o["hyp"]:
                    mismatches.append({"swa": o["swa"], "saved": o["hyp"], "exported": h})
    print(f"equivalence check: {len(preds) - len(mismatches)}/{len(preds)} identical to the saved predictions")
    for m in mismatches[:10]:
        print("  MISMATCH", m)
    (args.out_dir / "equivalence_check.json").write_text(json.dumps(
        {"n_checked": len(preds), "n_identical": len(preds) - len(mismatches), "mismatches": mismatches},
        indent=1, ensure_ascii=False))

    if args.push_repo:
        if mismatches:
            sys.exit("not pushing: exported model does not reproduce the saved predictions")
        model2.push_to_hub(args.push_repo, private=not args.public)
        tok2.push_to_hub(args.push_repo, private=not args.public)
        from huggingface_hub import upload_file
        for f in ("README.md", "training_config.json", "equivalence_check.json"):
            upload_file(path_or_fileobj=str(args.out_dir / f), path_in_repo=f, repo_id=args.push_repo)
        print(f"pushed to https://huggingface.co/{args.push_repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
