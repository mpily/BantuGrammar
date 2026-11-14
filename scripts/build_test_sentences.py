"""
build_test_sentences.py -- the Space's "Give me a sentence to test" pool (hf_space/test_sentences.json).

2026-11-14, user request: a diverse set -- news as well as storybooks. Before this the pool was 999 African
Storybook sentences + 60 Bible verses. Now four registers, sampled evenly by the app (register first):
  storybook  African Storybook Initiative (github.com/global-asp/storybooks-swahili, CC-BY / CC-BY-NC 3.0;
             the existing 999 sentences, mined by Code_Experiments/mine_african_storybook.py)
  news       Tanzanian news, community-datasets/swahili_news (CC BY 4.0), test split
  news       MasakhaNEWS Swahili (masakhane/masakhanews, AFL-3.0), dev+test: business, health, politics,
             religion, sports, technology, entertainment
  encyclopedia  Swahili Wikipedia (wikimedia/wikipedia 20231101.sw, CC BY-SA 3.0)
The Bible top-up is dropped: Bible verses are in the fine-tuned model's (arm A) training data, which would
favour it in the blind comparison.

Filters: one sentence, 5-18 words, starts with a capital and ends in . ! ?, no digits, URLs, quotes,
brackets or colons, at most 25% capitalised words after the first (not a list of names). Any sentence whose
normalised text appears on the Swahili side of the training / dev data (rulebook train + dev, every Bible
parallel file, the Colab training bundle) is removed. Seeded sampling.

Run:  ~/languagemodels/bert-env/bin/python -I scripts/build_test_sentences.py <download dir>
  <download dir> holds news/test.parquet, masakha/{dev,test}.tsv, wiki/sw.parquet (from huggingface.co).
"""
from __future__ import annotations

import csv
import glob
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent
OUT = ROOT / "hf_space" / "test_sentences.json"
SEED = 20261114
N = {"storybook": 400, "news_tz": 250, "news_masakhane": 250, "wikipedia": 200}
SOURCES = {
    "storybook": ("storybook", "African Storybook Initiative", "CC BY / CC BY-NC 3.0"),
    "news_tz": ("news", "Swahili News Dataset (Tanzanian newspapers)", "CC BY 4.0"),
    "news_masakhane": ("news", "MasakhaNEWS Swahili", "AFL-3.0"),
    "wikipedia": ("encyclopedia", "Swahili Wikipedia", "CC BY-SA 3.0"),
}
BAD = re.compile(r"[\d\"“”«»()\[\]:;/@#_=|<>…]|https?|www\.|\.\w+\.|’’")


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower().replace("’", "'")).split())


def sentences(text: str):
    text = re.sub(r"(?<=[a-z])\.(?=[A-Z])", ". ", text or "")      # "kushamiri.Aidha" in the news text
    for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        yield s.strip()


def ok(s: str, min_words: int = 5) -> bool:
    w = s.split()
    if re.fullmatch(r"[A-Z][a-z]{0,2}\.", w[-1] if w else ""):     # cut at an abbreviation (Mr., Dk., St.)
        return False
    if not min_words <= len(w) <= 18 or not s[:1].isupper() or s[-1] not in ".!?" or BAD.search(s):
        return False
    if s.count(".") > 1 or s.count(",") > 3 or "?" in s[:-1] or "!" in s[:-1]:
        return False
    caps = sum(t[:1].isupper() for t in w[1:])
    if caps > 0.25 * (len(w) - 1) or any(t.isupper() and len(t) > 1 for t in w[1:]):
        return False
    return True


def seen_swahili() -> set[str]:
    out = set()
    files = [DATA / "rulebook_data" / "train.jsonl", DATA / "rulebook_data" / "dev.jsonl"]
    files += [Path(p) for p in glob.glob(str(ROOT / "bible collection" / "data" / "*" / "parallel.jsonl"))]
    files += [Path(p) for p in glob.glob(str(DATA / "kitaita_colab_bundle" / "**" / "*.jsonl"), recursive=True)]
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for k in ("swa", "sw", "SUV", "swahili", "source"):
                if isinstance(r.get(k), str):
                    out.add(norm(re.sub(r"#[^A-Z]*", " ", r[k])))
                    for s in sentences(r[k]):
                        out.add(norm(s))
    return out


def main():
    import pandas as pd
    dl = Path(sys.argv[1])
    rng = random.Random(SEED)
    seen = seen_swahili()
    pools: dict[str, list[dict]] = {k: [] for k in N}

    old = json.loads(OUT.read_text(encoding="utf-8"))
    pools["storybook"] = [{"id": x["id"], "sw": x["sw"]} for x in old
                          if x["id"].startswith("ASB.") and ok(x["sw"], min_words=4)]

    for i, t in enumerate(pd.read_parquet(dl / "news" / "test.parquet")["text"]):
        for j, s in enumerate(sentences(t)):
            if ok(s):
                pools["news_tz"].append({"id": f"SNEWS.test.{i}.{j}", "sw": s})
    for split in ("dev", "test"):
        with open(dl / "masakha" / f"{split}.tsv", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                for j, s in enumerate(sentences(r["text"])):
                    if ok(s):
                        pools["news_masakhane"].append({"id": f"MNEWS.{split}.{i}.{j}", "sw": s, "topic": r["category"],
                                                        "url": r.get("url")})
    w = pd.read_parquet(dl / "wiki" / "sw.parquet", columns=["id", "url", "text"])
    for _, r in w.sample(n=min(6000, len(w)), random_state=SEED).iterrows():
        for j, s in enumerate(sentences(r["text"].split("\n")[0] + " " + " ".join(r["text"].split("\n")[1:4]))):
            if ok(s):
                pools["wikipedia"].append({"id": f"SWWIKI.{r['id']}.{j}", "sw": s, "url": r["url"]})

    out, report, used = [], {}, set()
    for key, n in N.items():
        cand = [x for x in pools[key] if norm(x["sw"]) not in seen]
        uniq = {}
        for x in cand:
            uniq.setdefault(norm(x["sw"]), x)
        cand = list(uniq.values())
        rng.shuffle(cand)
        n_clean = len(cand)
        if key == "news_masakhane":       # spread over topics
            by = {}
            for x in cand:
                by.setdefault(x["topic"], []).append(x)
            cand = [x for grp in zip(*by.values()) for x in grp] + [x for g in by.values() for x in g]
        pick = []
        for x in cand:
            k = norm(x["sw"])
            if k not in used:
                used.add(k)
                pick.append(x)
            if len(pick) == n:
                break
        reg, name, lic = SOURCES[key]
        for x in pick:
            out.append({**x, "register": reg, "source": name, "licence": lic})
        report[key] = {"candidates": len(pools[key]), "after_training_overlap_and_dedup": n_clean, "picked": len(pick),
                       **({"topics": dict(Counter(x["topic"] for x in pick))} if key == "news_masakhane" else {})}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), "\ntotal", len(out), "seen_swahili", len(seen))


if __name__ == "__main__":
    main()
