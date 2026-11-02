"""
eval_mined_lexicon_fallback.py

2026-11-02. One experiment: does the round-1 mined-lexicon fallback tier
(hf_space/kitaita_mined_lexicon.py) improve the deterministic pipeline?

Step 1 (validation): the mining HOLDOUT (1,000 dav_swa train pairs that the
round-1 mining never used; ~/languagemodels/mining/holdout.jsonl). Sweep the
confidence floor (off / high / medium / low / derived) and pick the best by
mean chrF. Step 2 (test, run once): dav_swa TEST, off vs the chosen floor, on
all 5,333 rows and on the "novel" subset whose Swahili side never occurs in
train (same key as nllb_finetune_dav.py).

Path: translate_sentence() (rule-based, no NLLB), as in dav_swa_corpus_eval.py.
Scoring: api.stats.chrf / mean_ci on api.util.normalize_kitaita, as /api/progress.
Also counts output tokens identical to a Swahili source token (untranslated).
Writes Code_Experiments/grammar_mining/eval_mined_lexicon_fallback_results.json.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))

import pandas as pd
from huggingface_hub import hf_hub_download

import kitaita_mined_lexicon as ml
from kitaita_sentence_generator import load_pages, build_lookup_tables
from kitaita_e2e_pipeline import translate_sentence
from api.stats import chrf, mean_ci
from api.util import normalize_kitaita

HOLDOUT = Path.home() / "languagemodels" / "mining" / "holdout.jsonl"
FLOORS = [None, "high", "medium", "low", "derived"]
_Q = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


def load_split(split):
    p = hf_hub_download(repo_id="thinkKenya/kenyan-low-resource-language-data", repo_type="dataset",
                        filename=f"dav_swa/{split}-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    return [{"id": int(r["id"]), "dav": r["translation"]["dav"], "swa": r["translation"]["swa"]}
            for _, r in df.iterrows()], p


def toks(s):
    return [t.strip(".,!?;:\"()[]") for t in (s or "").translate(_Q).lower().split() if t.strip(".,!?;:\"()[]")]


def key(text):
    return normalize_kitaita(text).replace("'", "")


def run(rows, floor):
    ml.MIN_CONFIDENCE = floor
    scores, copied, total, fills, errors = [], 0, 0, 0, 0
    for r in rows:
        try:
            out = translate_sentence(r["swa"], TABLES)
        except Exception:
            errors += 1
            scores.append(0.0)
            continue
        hyp = out.get("kitaita_sentence") or ""
        scores.append(chrf(normalize_kitaita(hyp), normalize_kitaita(r["dav"])))
        src = set(toks(r["swa"]))
        ht = toks(hyp)
        total += len(ht)
        copied += sum(t in src for t in ht)
        fills += len(getattr(out.get("decomposition"), "mined_fills", []) or [])
    return {"floor": floor, "n": len(rows), "chrf": mean_ci(scores), "errors": errors,
            "copied_token_share": round(copied / max(total, 1), 4), "mined_fills": fills,
            "_scores": scores}


def paired_diff(a, b, seed=20261102, B=2000):
    """Mean paired chrF difference b-a with a bootstrap 95% CI over sentences."""
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(B))
    return {"mean": round(sum(d) / len(d), 3), "ci95": [round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3)],
            "improved": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d), "same": sum(x == 0 for x in d)}


def strip(res):
    return {k: v for k, v in res.items() if k != "_scores"}


if __name__ == "__main__":
    TABLES = build_lookup_tables(*load_pages())
    holdout = [json.loads(l) for l in open(HOLDOUT, encoding="utf-8")]
    val = [run(holdout, f) for f in FLOORS]
    for v in val:
        print("holdout", strip(v))
    best = max(val[1:], key=lambda v: v["chrf"]["mean"])
    chosen = best["floor"]
    print("chosen floor:", chosen)

    test, test_path = load_split("test")
    train, train_path = load_split("train")
    train_keys = {key(r["swa"]) for r in train}
    novel = [r for r in test if key(r["swa"]) not in train_keys]
    res = {}
    for name, rows in (("test", test), ("test_novel", novel)):
        off, on = run(rows, None), run(rows, chosen)
        res[name] = {"off": strip(off), "on": strip(on), "paired_on_minus_off": paired_diff(off["_scores"], on["_scores"])}
        print(name, json.dumps(res[name]))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    out = {
        "experiment": "round-1 mined-lexicon fallback tier (one change)",
        "date": "2026-11-02",
        "config": {"git_sha_base": sha, "path": "translate_sentence (rule-based, no NLLB)",
                   "floors_swept": FLOORS, "chosen_floor_on_holdout": chosen,
                   "bootstrap": {"B": 2000, "seed": 20261102},
                   "files_sha256_16": {
                       "kitaita_mined_lexicon_round1.json": h(ROOT / "hf_space" / "kitaita_mined_lexicon_round1.json"),
                       "kitaita_mined_lexicon.py": h(ROOT / "hf_space" / "kitaita_mined_lexicon.py"),
                       "kitaita_e2e_pipeline.py": h(ROOT / "hf_space" / "kitaita_e2e_pipeline.py"),
                       "holdout.jsonl": h(HOLDOUT), "test.parquet": h(test_path), "train.parquet": h(train_path)}},
        "validation_holdout": [strip(v) for v in val],
        "test": res,
    }
    (Path(__file__).parent / "eval_mined_lexicon_fallback_results.json").write_text(json.dumps(out, indent=1))


def shuffled_control(rows, floor, seed=20261102):
    """Control: same table, same fill positions, but each Swahili word gets the
    Kidaw'ida form of a different random entry (values permuted). Measures how
    much chrF rises just from inserting Kidaw'ida-looking strings."""
    table = ml._table()
    keys = list(table)
    vals = [table[k] for k in keys]
    random.Random(seed).shuffle(vals)
    saved = dict(table)
    table.clear()
    table.update({k: {**saved[k], "kitaita": v["kitaita"]} for k, v in zip(keys, vals)})
    try:
        return run(rows, floor)
    finally:
        table.clear()
        table.update(saved)
