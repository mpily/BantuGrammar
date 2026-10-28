# -*- coding: utf-8 -*-
"""
nllb_finetune_dav.py

2026-10-28 cont. 9. External-review item 6 (deferred in cont. 8): a
fine-tuned NLLB Swahili->Kidaw'ida baseline, scored on the SAME held-out
dav_swa test split and the SAME sentence-level metrics as the rule-based
pipeline (cont. 1/5: A = 30.75 chrF, D dictionary-only = 28.12, C null =
23.21), so "does the pipeline beat simply training on the data?" has an
answer.

Design (one change per experiment, rule 9):
  arm 1: dav_swa train split only (minus a seeded validation carve-out)
  arm 2: arm 1 + extra parallel pairs (e.g. Bible verses) via --extra-pairs
Everything else identical between arms; pass every hyperparameter on the
CLI (rule 4) -- the resolved config is written into the results JSON.

Language tag: NLLB-200 has no `dav` code. A new special token `dav_Latn`
is added, its input/output embedding initialised as a copy of `swh_Latn`
(closest covered relative). Source side is tagged `swh_Latn`.

Selection: the checkpoint kept for test decoding is the one with best
validation chrF (validation disjoint from test). Test is decoded once, at
the end, with that checkpoint.

Checkpoint/resume: every --save-every steps the full training state
(model, optimizer, scheduler, step, data order, RNG) is written to
<out-dir>/last/; re-running the same command resumes from it. Test
decoding resumes per batch from <out-dir>/test_predictions.jsonl.

Output: <out-dir>/test_predictions.jsonl (id, swa, ref, hyp) so the
slot-level metrics can be run on these hypotheses later, and
<out-dir>/results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from api.stats import chrf, mean_ci, rate_ci
from api.util import normalize_kitaita
from dav_swa_eval_common import tokens_match

DATASET_REPO = "thinkKenya/kenyan-low-resource-language-data"
CONFIG = "dav_swa"
SRC_LANG = "swh_Latn"
TGT_LANG = "dav_Latn"
INIT_FROM_LANG = "swh_Latn"


# ---------------------------------------------------------------- data

def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def git_sha() -> str:
    if not (HERE.parent / ".git").exists():
        bundle = HERE.parent / "BUNDLE_GIT_SHA"
        return bundle.read_text().strip() + "(bundle)" if bundle.exists() else "unknown"
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(Path(__file__).name)],
                           cwd=HERE, capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+dirty" if dirty else "")


def load_split(split: str) -> tuple[list[dict], Path]:
    path = Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset",
                                filename=f"{CONFIG}/{split}-00000-of-00001.parquet"))
    df = pd.read_parquet(path)
    rows = [{"id": f"{split}:{int(r['id'])}", "swa": r["translation"]["swa"], "dav": r["translation"]["dav"]}
            for _, r in df.iterrows()]
    return rows, path


def load_extra_pairs(path: Path) -> list[dict]:
    """JSONL, one object per line with at least "swa" and "dav" (optional "id")."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            rows.append({"id": o.get("id", f"extra:{i}"), "swa": o["swa"], "dav": o["dav"]})
    return rows


def _key(text: str) -> str:
    return normalize_kitaita(text).replace("'", "")


def drop_overlap(extra: list[dict], held_out: list[dict]) -> tuple[list[dict], int]:
    """Drop extra pairs whose Swahili OR Kitaita side equals any held-out
    (val/test) sentence after normalization -- extra data must never leak
    a held-out sentence into training."""
    seen = {_key(r["swa"]) for r in held_out} | {_key(r["dav"]) for r in held_out}
    kept = [r for r in extra if _key(r["swa"]) not in seen and _key(r["dav"]) not in seen]
    return kept, len(extra) - len(kept)


def clean(rows: list[dict]) -> list[dict]:
    """Drop empty sides and the literal string "nan" (86 train rows in the
    released dav_swa split are exactly that -- a data artifact)."""
    return [r for r in rows if _key(r["swa"]) not in ("", "nan") and _key(r["dav"]) not in ("", "nan")]


# ---------------------------------------------------------------- model

def add_target_language(model, tok) -> int:
    if TGT_LANG not in tok.get_vocab():
        tok.add_tokens([TGT_LANG], special_tokens=True)
        model.resize_token_embeddings(len(tok))
        new_id = tok.convert_tokens_to_ids(TGT_LANG)
        init_id = tok.convert_tokens_to_ids(INIT_FROM_LANG)
        with torch.no_grad():
            model.get_input_embeddings().weight[new_id] = model.get_input_embeddings().weight[init_id]
            out = model.get_output_embeddings()
            if out is not None and out.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
                out.weight[new_id] = out.weight[init_id]
    return tok.convert_tokens_to_ids(TGT_LANG)


def encode(tok, text: str, lang_id: int, max_len: int) -> list[int]:
    """NLLB (non-legacy) layout: [lang] tokens [eos], built by hand so it does
    not depend on the tokenizer knowing the new language code."""
    ids = tok(text, add_special_tokens=False)["input_ids"][: max_len - 2]
    return [lang_id] + ids + [tok.eos_token_id]


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    src_len = max(len(s) for s, _ in batch)
    tgt_len = max(len(t) for _, t in batch)
    input_ids = torch.full((len(batch), src_len), pad_id)
    attn = torch.zeros((len(batch), src_len), dtype=torch.long)
    labels = torch.full((len(batch), tgt_len), -100)
    for i, (s, t) in enumerate(batch):
        input_ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
        labels[i, : len(t)] = torch.tensor(t)
    return input_ids, attn, labels


@torch.no_grad()
def translate(model, tok, texts: list[str], src_id: int, tgt_id: int, args, device) -> list[str]:
    model.eval()
    enc = [encode(tok, t, src_id, args.max_len) for t in texts]
    input_ids, attn, _ = collate([(e, [0]) for e in enc], tok.pad_token_id)
    out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                         forced_bos_token_id=tgt_id, num_beams=args.num_beams,
                         max_new_tokens=args.max_len, max_length=None, use_cache=True)
    return tok.batch_decode(out, skip_special_tokens=True)


def score(hyps: list[str], refs: list[str]) -> dict:
    """Same scoring as dav_swa_corpus_eval.py (cont. 1/5): normalize_kitaita
    on both sides, apostrophe-insensitive exact match, sentence chrF mean."""
    exact = sum(int(tokens_match(normalize_kitaita(h), normalize_kitaita(r))) for h, r in zip(hyps, refs))
    scores = [chrf(normalize_kitaita(h), normalize_kitaita(r)) for h, r in zip(hyps, refs)]
    return {"exact_match": rate_ci(exact, len(refs)), "chrf": mean_ci(scores), "n": len(refs)}


def evaluate_val(model, tok, val: list[dict], src_id, tgt_id, args, device) -> dict:
    hyps = []
    for i in range(0, len(val), args.eval_batch_size):
        hyps += translate(model, tok, [r["swa"] for r in val[i: i + args.eval_batch_size]], src_id, tgt_id, args, device)
    return score(hyps, [r["dav"] for r in val])


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--extra-pairs", type=Path, default=None, help="JSONL of {swa, dav} pairs (arm 2)")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--val-size", required=True, type=int)
    p.add_argument("--epochs", required=True, type=float)
    p.add_argument("--lr", required=True, type=float)
    p.add_argument("--weight-decay", required=True, type=float)
    p.add_argument("--warmup-steps", required=True, type=int)
    p.add_argument("--batch-size", required=True, type=int)
    p.add_argument("--grad-accum", required=True, type=int)
    p.add_argument("--max-len", required=True, type=int)
    p.add_argument("--num-beams", required=True, type=int)
    p.add_argument("--eval-every", required=True, type=int, help="optimizer steps between validation runs")
    p.add_argument("--save-every", required=True, type=int, help="optimizer steps between resume checkpoints")
    p.add_argument("--eval-batch-size", required=True, type=int)
    p.add_argument("--max-steps", type=int, default=None, help="smoke tests only; caps optimizer steps")
    p.add_argument("--test-limit", type=int, default=None, help="smoke tests only; reported runs use the full split")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (GPU)")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="recompute activations in backward to save GPU memory; same math, ~30%% slower "
                        "(added after arm 2 -- long Bible verses -- ran out of memory on a T4)")
    p.add_argument("--delete-resume-state-when-done", action="store_true",
                   help="remove <out-dir>/last/ (~7.4 GB for 600M) once results.json is written")
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        sys.exit(f"{out}/results.json already exists -- this arm is finished; use a new --out-dir to rerun")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data: validation carved from TRAIN with a fixed seed; test untouched
    train_all, train_path = load_split("train")
    test, test_path = load_split("test")
    train_all, test = clean(train_all), clean(test)
    order = list(range(len(train_all)))
    random.Random(args.seed).shuffle(order)
    val = [train_all[i] for i in order[: args.val_size]]
    train = [train_all[i] for i in order[args.val_size:]]
    # train has internal duplicates, so a val sentence can recur in the rest of
    # train; drop those rows or checkpoint selection rewards memorisation.
    val_swa_keys = {_key(r["swa"]) for r in val}
    n_before = len(train)
    train = [r for r in train if _key(r["swa"]) not in val_swa_keys]
    n_train_dropped_val_dup = n_before - len(train)
    # The released split overlaps: 1,818/5,333 test Swahili sentences also occur
    # in train (1,518 as exact swa+dav pairs). Test is reported on the full split
    # (comparable with cont. 1) AND on the "novel" subset whose Swahili side never
    # occurs anywhere in train -- the novel subset is the headline number.
    train_swa_keys = {_key(r["swa"]) for r in train_all}
    novel_test_ids = {r["id"] for r in test if _key(r["swa"]) not in train_swa_keys}

    extra_info = None
    if args.extra_pairs:
        extra = clean(load_extra_pairs(args.extra_pairs))
        extra, n_dropped = drop_overlap(extra, val + test)
        train = train + extra
        extra_info = {"path": str(args.extra_pairs), "sha256_16": file_sha256(args.extra_pairs),
                      "n_kept": len(extra), "n_dropped_overlap_with_val_or_test": n_dropped}
    if args.test_limit:
        test = test[: args.test_limit]

    # --- model
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tgt_id = add_target_language(model, tok)
    src_id = tok.convert_tokens_to_ids(SRC_LANG)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model.to(device)

    examples = [(encode(tok, r["swa"], src_id, args.max_len), encode(tok, r["dav"], tgt_id, args.max_len))
                for r in train]
    steps_per_epoch = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, args.warmup_steps, total_steps)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update({"git_sha": git_sha(), "script_sha256_16": file_sha256(Path(__file__)), "device": str(device), "torch": torch.__version__,
                   "src_lang": SRC_LANG, "tgt_lang": TGT_LANG, "tgt_init_from": INIT_FROM_LANG,
                   "train_parquet_sha256_16": file_sha256(train_path),
                   "test_parquet_sha256_16": file_sha256(test_path),
                   "n_train": len(train), "n_val": len(val), "n_test": len(test),
                   "n_train_dropped_val_dup": n_train_dropped_val_dup,
                   "n_test_novel": len([r for r in test if r["id"] in novel_test_ids]),
                   "extra_pairs": extra_info, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps})
    print(json.dumps(config, indent=1), flush=True)

    # --- resume
    state = {"step": 0, "epoch_order": None, "pos": 0, "best_val_chrf": -1.0, "val_history": []}
    last = out / "last"
    if (last / "state.pt").exists():
        ck = torch.load(last / "state.pt", weights_only=False)
        if ck["config"]["n_train"] != config["n_train"] or ck["config"]["seed"] != args.seed:
            sys.exit("checkpoint in --out-dir was made with a different config; refusing to resume")
        model.load_state_dict(torch.load(last / "model.pt", map_location=device))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        state = ck["state"]
        random.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed at step {state['step']}", flush=True)

    def save_last():
        last.mkdir(exist_ok=True)
        torch.save(model.state_dict(), last / "model.pt.tmp")
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "state": state, "config": config,
                    "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()}, last / "state.pt.tmp")
        (last / "model.pt.tmp").replace(last / "model.pt")
        (last / "state.pt.tmp").replace(last / "state.pt")

    # --- train
    model.train()
    t0 = time.time()
    while state["step"] < total_steps:
        if state["epoch_order"] is None or state["pos"] >= len(state["epoch_order"]):
            state["epoch_order"] = random.sample(range(len(examples)), len(examples))
            state["pos"] = 0
        model.train()
        opt.zero_grad()
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            idx = state["epoch_order"][state["pos"]: state["pos"] + args.batch_size]
            state["pos"] += args.batch_size
            if not idx:
                break
            input_ids, attn, labels = collate([examples[i] for i in idx], tok.pad_token_id)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                             labels=labels.to(device)).loss / args.grad_accum
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        state["step"] += 1
        if state["step"] % 20 == 0:
            print(f"step {state['step']}/{total_steps} loss {loss_sum:.4f} "
                  f"{(time.time() - t0) / 20:.2f}s/step", flush=True)
            t0 = time.time()
        if state["step"] % args.eval_every == 0 or state["step"] == total_steps:
            v = evaluate_val(model, tok, val, src_id, tgt_id, args, device)
            state["val_history"].append({"step": state["step"], **v})
            print(f"VAL step {state['step']}: chrF {v['chrf']['mean']:.2f} "
                  f"exact {v['exact_match']['rate']:.4f} (n={v['n']})", flush=True)
            if v["chrf"]["mean"] > state["best_val_chrf"]:
                state["best_val_chrf"] = v["chrf"]["mean"]
                state["best_step"] = state["step"]
                torch.save(model.state_dict(), out / "best_model.pt")
        if state["step"] % args.save_every == 0 or state["step"] == total_steps:
            save_last()

    # --- test, once, with the best-validation checkpoint
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    pred_path = out / "test_predictions.jsonl"
    done = {}
    if pred_path.exists():
        for line in open(pred_path, encoding="utf-8"):
            o = json.loads(line)
            done[o["id"]] = o
    todo = [r for r in test if r["id"] not in done]
    with open(pred_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.eval_batch_size):
            batch = todo[i: i + args.eval_batch_size]
            hyps = translate(model, tok, [r["swa"] for r in batch], src_id, tgt_id, args, device)
            for r, h in zip(batch, hyps):
                o = {"id": r["id"], "swa": r["swa"], "ref": r["dav"], "hyp": h}
                done[r["id"]] = o
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
    preds = [done[r["id"]] for r in test]
    result = {"config": config, "best_step": state.get("best_step"), "best_val_chrf": state["best_val_chrf"],
              "val_history": state["val_history"],
              "test": score([o["hyp"] for o in preds], [o["ref"] for o in preds]),
              "test_novel": score([o["hyp"] for o in preds if o["id"] in novel_test_ids],
                                  [o["ref"] for o in preds if o["id"] in novel_test_ids]),
              "smoke_test_only": bool(args.max_steps or args.test_limit)}
    (out / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    if args.delete_resume_state_when_done:
        shutil.rmtree(last, ignore_errors=True)
    for name in ("test", "test_novel"):
        t = result[name]
        print(f"{name.upper()}: chrF {t['chrf']['mean']:.2f} {t['chrf']['ci95']} "
              f"exact {t['exact_match']['k']}/{t['n']} {t['exact_match']['ci95']}")


if __name__ == "__main__":
    main()
