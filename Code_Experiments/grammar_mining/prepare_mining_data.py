# -*- coding: utf-8 -*-
"""
prepare_mining_data.py

2026-10-31. Shared starting point for the multi-agent grammar/vocabulary
mining rounds (status.md 2026-10-31).

Data discipline:
  * MINE   = dav_swa TRAIN minus a seeded 1,000-pair holdout, plus the
             cleaned Bible (NEN) pairs. Agents derive rules/words from this.
  * HOLDOUT = those 1,000 dav_swa train pairs. Used ONLY to check
             predictions (lexicon precision, rule accuracy). Never mined.
  * dav_swa TEST is not touched here at all (it stays the final
             evaluation set for the pipeline / fine-tune comparison).
Corpus text is written OUTSIDE the repo (~/languagemodels/mining/):
dav_swa is CC-BY-4.0 but gated; the Bible is rights-reserved.

Outputs (in --out):
  mine.jsonl, holdout.jsonl     {id, src: davswa|bible, swa, dav, swa_tok, dav_tok}
  links.jsonl                   per MINE pair: symmetrised eflomal links (i-j)
  lexicon_candidates.tsv        swa_word, dav_word, links, swa_freq, dav_freq,
                                p(dav|swa), dice, n_davswa, n_bible
  dav_affix_stats.tsv           Kitaita word-initial / word-final strings (1-4
                                chars) with type and token counts
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / "hf_space")]

from nllb_finetune_dav import clean, load_split  # same loader + cleaning as the fine-tune
from api.util import normalize_kitaita

_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})


def tok(text: str) -> list[str]:
    """Lowercase, straight quotes, keep word-internal apostrophes (w'/ng'),
    strip surrounding punctuation. Same spirit as dav_swa_corpus_eval.tokenize."""
    out = []
    for t in normalize_kitaita(text).translate(_QUOTES).split():
        t = t.strip(".,!?;:\"()[]“”«»…-—–*")
        t = t.strip("'") if not re.match(r"^\w'\w", t) else t
        if t and re.search(r"\w", t):
            out.append(t)
    return out


def symmetrise(fwd: str, rev: str) -> set[tuple[int, int]]:
    """grow-diag style: intersection plus union links adjacent to an
    intersection link (keeps precision high for lexicon extraction)."""
    f = {tuple(map(int, p.split("-"))) for p in fwd.split()}
    r = {tuple(map(int, p.split("-"))) for p in rev.split()}
    inter, union = f & r, f | r
    out = set(inter)
    added = True
    while added:
        added = False
        for (i, j) in sorted(union - out):
            if any((i + di, j + dj) in out for di in (-1, 0, 1) for dj in (-1, 0, 1)) and \
                    (all(a != i for a, _ in out) or all(b != j for _, b in out)):
                out.add((i, j))
                added = True
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bible-pairs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--holdout-size", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = clean(load_split("train")[0])
    rng = random.Random(args.seed)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    hold = [dict(train[i], src="davswa") for i in idx[: args.holdout_size]]
    hold_keys = {normalize_kitaita(r["swa"]) for r in hold}
    mine = [dict(train[i], src="davswa") for i in idx[args.holdout_size:]
            if normalize_kitaita(train[i]["swa"]) not in hold_keys]
    for line in open(args.bible_pairs, encoding="utf-8"):
        o = json.loads(line)
        mine.append({"id": o["id"], "swa": o["swa"], "dav": o["dav"], "src": "bible"})
    for r in mine + hold:
        r["swa_tok"], r["dav_tok"] = tok(r["swa"]), tok(r["dav"])
    mine = [r for r in mine if r["swa_tok"] and r["dav_tok"]]
    for name, rows in (("mine", mine), ("holdout", hold)):
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"mine {len(mine)} (davswa {sum(r['src']=='davswa' for r in mine)}, bible {sum(r['src']=='bible' for r in mine)}), holdout {len(hold)}")

    # --- eflomal, both directions, on MINE only
    import eflomal
    src_txt, trg_txt = args.out / "_swa.txt", args.out / "_dav.txt"
    src_txt.write_text("\n".join(" ".join(r["swa_tok"]) for r in mine) + "\n", encoding="utf-8")
    trg_txt.write_text("\n".join(" ".join(r["dav_tok"]) for r in mine) + "\n", encoding="utf-8")
    fwd_p, rev_p = args.out / "_fwd.links", args.out / "_rev.links"
    with open(src_txt, encoding="utf-8") as s, open(trg_txt, encoding="utf-8") as t:
        eflomal.Aligner().align(s, t, links_filename_fwd=str(fwd_p), links_filename_rev=str(rev_p), quiet=True)
    fwd, rev = fwd_p.read_text().splitlines(), rev_p.read_text().splitlines()
    assert len(fwd) == len(mine) == len(rev), (len(fwd), len(mine), len(rev))

    pair_links = Counter()
    by_src = defaultdict(Counter)
    swa_freq, dav_freq = Counter(), Counter()
    with open(args.out / "links.jsonl", "w") as f:
        for r, a, b in zip(mine, fwd, rev):
            links = sorted(symmetrise(a, b))
            f.write(json.dumps({"id": r["id"], "links": links}) + "\n")
            swa_freq.update(set(r["swa_tok"]))
            dav_freq.update(set(r["dav_tok"]))
            for i, j in links:
                key = (r["swa_tok"][i], r["dav_tok"][j])
                pair_links[key] += 1
                by_src[key][r["src"]] += 1
    swa_total = Counter()
    for (s, _), c in pair_links.items():
        swa_total[s] += c
    with open(args.out / "lexicon_candidates.tsv", "w", encoding="utf-8") as f:
        f.write("swa\tdav\tlinks\tswa_freq\tdav_freq\tp_dav_given_swa\tdice\tn_davswa\tn_bible\n")
        for (s, d), c in sorted(pair_links.items(), key=lambda kv: (-kv[1], kv[0])):
            if c < 2:
                continue
            dice = 2 * c / (swa_freq[s] + dav_freq[d])
            f.write(f"{s}\t{d}\t{c}\t{swa_freq[s]}\t{dav_freq[d]}\t{c / swa_total[s]:.3f}\t{dice:.3f}"
                    f"\t{by_src[(s, d)]['davswa']}\t{by_src[(s, d)]['bible']}\n")

    # --- Kitaita affix statistics (types and tokens)
    tokc = Counter(w for r in mine for w in r["dav_tok"])
    rows = []
    for side in ("prefix", "suffix"):
        for n in range(1, 5):
            types, toks = Counter(), Counter()
            for w, c in tokc.items():
                if len(w) > n + 1:
                    a = w[:n] if side == "prefix" else w[-n:]
                    types[a] += 1
                    toks[a] += c
            rows += [(side, a, types[a], toks[a]) for a in types if types[a] >= 20]
    with open(args.out / "dav_affix_stats.tsv", "w", encoding="utf-8") as f:
        f.write("side\taffix\ttypes\ttokens\n")
        for row in sorted(rows, key=lambda x: (x[0], -x[2])):
            f.write("\t".join(map(str, row)) + "\n")
    print("lexicon candidate pairs (links>=2):", sum(1 for c in pair_links.values() if c >= 2))


if __name__ == "__main__":
    main()
