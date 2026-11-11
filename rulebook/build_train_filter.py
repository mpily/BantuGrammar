"""
build_train_filter.py

2026-11-11, CLAUDE_rulebook.md §1 + §6. Builds the train-only data the rule
book, lexicon and retrieval index may use, and the dev slice for validation.

Excluded (never mined, indexed or cited):
  - the whole released dav_swa test split (cleaned) -- contains test_novel,
    test_novel2, test_novel3 (decode_nbest.py) and the overlapping rest;
  - arm A's validation carve-out (500 train rows, seed 20261028, used for
    checkpoint selection: nllb_finetune_dav.py).
Dropped from train (same near-duplicate check as status 2026-10-28 cont. 12):
  - exact match of the normalized Swahili OR Kidaw'ida side with an excluded row;
  - nearest excluded sentence by character 3-5-gram TF-IDF cosine (Swahili side,
    top 3 checked), word-level difflib ratio >= SW_WORD_RATIO ("1-2 words differ"
    band of cont. 12);
  - Kidaw'ida side char cosine >= DAV_COSINE with an excluded row (cont. 12's
    Kitaita near-identity cut).
Then dev = DEV_SHARE of the remaining pairs (seed 20261111, stratified by
corpus). Dev is held out from the rule book, lexicon evidence and index too, so
§6 coverage/round-trip numbers are not in-sample.

Inputs: ~/languagemodels/mining/{mine,holdout}.jsonl (= all cleaned dav_swa train
+ 27,003 Bible NEN pairs) and the dav_swa test parquet (HF cache).
Outputs:
  ~/languagemodels/rulebook_data/{train,dev}.jsonl   corpus text, NOT in git
  rulebook/split_manifest.json                        IDs, counts, filter, hashes
"""
from __future__ import annotations

import difflib
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code_Experiments"))
sys.path.insert(0, str(ROOT / "hf_space"))
from nllb_finetune_dav import _key, clean, load_split  # noqa: E402

MINING = Path.home() / "languagemodels" / "mining"
OUT_DIR = Path.home() / "languagemodels" / "rulebook_data"
MANIFEST = ROOT / "rulebook" / "split_manifest.json"
ARM_SPLIT_SEED, ARM_VAL_SIZE = 20261028, 500
DEV_SEED, DEV_SHARE = 20261111, 0.10
SW_WORD_RATIO, DAV_COSINE, TOPK = 0.8, 0.95, 3


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def excluded_rows() -> tuple[list[dict], dict]:
    train_all, train_path = load_split("train")
    test, test_path = load_split("test")
    train_all, test = clean(train_all), clean(test)
    order = list(range(len(train_all)))
    random.Random(ARM_SPLIT_SEED).shuffle(order)
    val = [train_all[i] for i in order[:ARM_VAL_SIZE]]
    for r in val:
        r["why"] = "arm_val"
    for r in test:
        r["why"] = "test"
    return test + val, {"train_parquet_sha16": sha16(train_path), "test_parquet_sha16": sha16(test_path),
                        "n_test_clean": len(test), "n_arm_val": len(val)}


def main():
    pool = read_jsonl(MINING / "mine.jsonl") + read_jsonl(MINING / "holdout.jsonl")
    excl, info = excluded_rows()
    excl_ids = {r["id"] for r in excl}
    reasons: dict[str, str] = {}

    # 1. the row itself is excluded (arm val rows are in the mining pool)
    for r in pool:
        if r["id"] in excl_ids:
            reasons[r["id"]] = "is_excluded_row"
    # 2. exact normalized match on either side
    ex_sw = {_key(r["swa"]) for r in excl}
    ex_dav = {_key(r["dav"]) for r in excl}
    for r in pool:
        if r["id"] not in reasons and (_key(r["swa"]) in ex_sw or _key(r["dav"]) in ex_dav):
            reasons[r["id"]] = "exact_dup"
    # 3. near duplicates
    rest = [r for r in pool if r["id"] not in reasons]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit([_key(r["swa"]) for r in excl] + [_key(r["swa"]) for r in rest])
    E = vec.transform([_key(r["swa"]) for r in excl])
    vd = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vd.fit([_key(r["dav"]) for r in excl] + [_key(r["dav"]) for r in rest])
    Ed = vd.transform([_key(r["dav"]) for r in excl])
    ex_words = [_key(r["swa"]).split() for r in excl]
    for s in range(0, len(rest), 2000):
        chunk = rest[s:s + 2000]
        sims = (vec.transform([_key(r["swa"]) for r in chunk]) @ E.T).toarray()
        dsims = (vd.transform([_key(r["dav"]) for r in chunk]) @ Ed.T).toarray()
        for i, r in enumerate(chunk):
            words = _key(r["swa"]).split()
            for j in np.argsort(-sims[i])[:TOPK]:
                if difflib.SequenceMatcher(None, words, ex_words[j]).ratio() >= SW_WORD_RATIO:
                    reasons[r["id"]] = "near_dup_swa"
                    break
            if r["id"] not in reasons and dsims[i].max() >= DAV_COSINE:
                reasons[r["id"]] = "near_dup_dav"
    kept = [r for r in pool if r["id"] not in reasons]

    # 4. dev slice, stratified by corpus
    dev_ids = set()
    for src in ("davswa", "bible"):
        ids = sorted(r["id"] for r in kept if r["src"] == src)
        random.Random(DEV_SEED).shuffle(ids)
        dev_ids |= set(ids[: round(DEV_SHARE * len(ids))])
    train = [r for r in kept if r["id"] not in dev_ids]
    dev = [r for r in kept if r["id"] in dev_ids]

    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def by(rows, key):
        out = {}
        for r in rows:
            out[key(r)] = out.get(key(r), 0) + 1
        return out

    dropped = [r for r in pool if r["id"] in reasons]
    MANIFEST.write_text(json.dumps({
        "_meta": {"built_by": "rulebook/build_train_filter.py", "script_sha16": sha16(Path(__file__)),
                  "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                            text=True).stdout.strip(),
                  "inputs": {"mine.jsonl": sha16(MINING / "mine.jsonl"),
                             "holdout.jsonl": sha16(MINING / "holdout.jsonl"), **info},
                  "filter": {"excluded": "dav_swa test (cleaned, all of it) + arm-A val (500, seed 20261028)",
                             "exact": "normalized swa OR dav side equal to an excluded row",
                             "near_dup_swa": f"char_wb 3-5 TF-IDF top-{TOPK} excluded neighbours, word difflib ratio >= {SW_WORD_RATIO}",
                             "near_dup_dav": f"dav-side char_wb 3-5 TF-IDF cosine >= {DAV_COSINE}",
                             "dev": f"{DEV_SHARE} of kept pairs per corpus, seed {DEV_SEED}"},
                  "outputs": {"train.jsonl": sha16(OUT_DIR / "train.jsonl"), "dev.jsonl": sha16(OUT_DIR / "dev.jsonl")}},
        "counts": {"pool": by(pool, lambda r: r["src"]), "dropped_by_reason": by(dropped, lambda r: reasons[r["id"]]),
                   "dropped_by_src": by(dropped, lambda r: r["src"]),
                   "train": by(train, lambda r: r["src"]), "dev": by(dev, lambda r: r["src"])},
        "excluded_ids": sorted(excl_ids),
        "dropped_train_ids": {i: reasons[i] for i in sorted(reasons)},
        "dev_ids": sorted(dev_ids),
    }, indent=1, ensure_ascii=False))
    print(json.dumps(json.loads(MANIFEST.read_text())["counts"], indent=1))


if __name__ == "__main__":
    main()
