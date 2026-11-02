# -*- coding: utf-8 -*-
"""
mine_query.py -- shared query tool for the grammar/vocabulary mining agents.

Reads ~/languagemodels/mining/ (built by prepare_mining_data.py).
MINE data only, except `check`, which scores against the HOLDOUT.

  grep   --swa RE --dav RE [--src davswa|bible] [--limit N]
         sentence pairs whose tokens match (regex on whole tokens, fullmatch);
         matched words are [bracketed]; aligned counterpart shown after "=>"
  align  WORD [--side swa|dav] [--limit N]
         top aligned counterparts of WORD with counts by corpus
  count  --dav RE | --swa RE [--by-src]
         token/type counts of matching words (with the top types)
  check  LEXICON.json
         holdout precision of a {swa_word: dav_word | [dav_words]} lexicon:
         an entry is a hit on a holdout pair if the Swahili word occurs and one
         of its predicted Kitaita words occurs in that pair's Kitaita side
         (apostrophe-insensitive). Reports hit rate with a Wilson 95% CI and
         the same number for the alignment table's own top candidate (baseline).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

DATA = Path.home() / "languagemodels" / "mining"


def norm(w: str) -> str:
    return w.replace("'", "")


@lru_cache(None)
def load(name: str) -> list[dict]:
    return [json.loads(l) for l in open(DATA / f"{name}.jsonl", encoding="utf-8")]


@lru_cache(None)
def links() -> list[list[tuple[int, int]]]:
    return [[tuple(x) for x in json.loads(l)["links"]] for l in open(DATA / "links.jsonl")]


@lru_cache(None)
def table() -> dict[str, list[tuple[str, int, int, int]]]:
    t = defaultdict(list)
    with open(DATA / "lexicon_candidates.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, d, c, _, _, _, _, nd, nb = line.rstrip("\n").split("\t")
            t[s].append((d, int(c), int(nd), int(nb)))
    return t


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def cmd_grep(a):
    rs, rd = (re.compile(a.swa) if a.swa else None), (re.compile(a.dav) if a.dav else None)
    shown = total = 0
    for r, L in zip(load("mine"), links()):
        if a.src and r["src"] != a.src:
            continue
        si = [i for i, w in enumerate(r["swa_tok"]) if rs and rs.fullmatch(w)]
        di = [j for j, w in enumerate(r["dav_tok"]) if rd and rd.fullmatch(w)]
        if (rs and not si) or (rd and not di):
            continue
        total += 1
        if shown >= a.limit:
            continue
        shown += 1
        sw = " ".join(f"[{w}]" if i in si else w for i, w in enumerate(r["swa_tok"]))
        dv = " ".join(f"[{w}]" if j in di else w for j, w in enumerate(r["dav_tok"]))
        al = [f"{r['swa_tok'][i]}=>{r['dav_tok'][j]}" for i, j in L if i in si or j in di]
        print(f"{r['id']}\n  SW: {sw}\n  DV: {dv}\n  AL: {', '.join(al)}")
    print(f"-- {total} matching pairs ({shown} shown)")


def cmd_align(a):
    if a.side == "swa":
        rows = table().get(a.word, [])
    else:
        rows = [(s, c, nd, nb) for s, v in table().items() for d, c, nd, nb in v if d == a.word]
    rows.sort(key=lambda x: -x[1])
    tot = sum(x[1] for x in rows) or 1
    for w, c, nd, nb in rows[: a.limit]:
        print(f"{w:25s} links {c:6d}  share {c / tot:.2f}  davswa {nd:5d}  bible {nb:5d}")
    if not rows:
        print("-- no alignment rows (word unseen in MINE, or <2 links)")


def cmd_count(a):
    side, rx = ("dav_tok", a.dav) if a.dav else ("swa_tok", a.swa)
    r_ = re.compile(rx)
    c = defaultdict(Counter)
    for r in load("mine"):
        for w in r[side]:
            if r_.fullmatch(w):
                c[r["src"] if a.by_src else "all"][w] += 1
    for k, cc in c.items():
        print(f"[{k}] tokens {sum(cc.values())}, types {len(cc)}: " +
              ", ".join(f"{w}:{n}" for w, n in cc.most_common(25)))


def cmd_check(a):
    lex = json.load(open(a.lexicon, encoding="utf-8"))
    lex = {s: ([v] if isinstance(v, str) else v) for s, v in lex.items()}
    lex = {s: {norm(x) for x in v} for s, v in lex.items() if v}
    base = {s: {norm(table()[s][0][0])} for s in lex if table().get(s)}
    def score(L):
        k = n = 0
        per = Counter()
        for r in load("holdout"):
            dset = {norm(w) for w in r["dav_tok"]}
            for s in set(r["swa_tok"]):
                if s in L:
                    n += 1
                    hit = bool(L[s] & dset)
                    k += hit
                    per[(s, hit)] += 1
        return k, n, per
    k, n, per = score(lex)
    lo, hi = wilson(k, n)
    print(f"entries {len(lex)} | holdout occurrences {n} | hit {k}/{n} = {k / max(n, 1):.3f} [{lo:.3f}, {hi:.3f}]")
    kb, nb, _ = score(base)
    lo, hi = wilson(kb, nb)
    print(f"baseline (alignment top-1 for the same Swahili words, where available): {kb}/{nb} = {kb / max(nb, 1):.3f} [{lo:.3f}, {hi:.3f}]")
    miss = Counter({s: c for (s, h), c in per.items() if not h})
    print("most-missed entries:", ", ".join(f"{s}:{c}" for s, c in miss.most_common(15)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("grep"); g.add_argument("--swa"); g.add_argument("--dav"); g.add_argument("--src")
    g.add_argument("--limit", type=int, default=10)
    al = sp.add_parser("align"); al.add_argument("word"); al.add_argument("--side", default="swa")
    al.add_argument("--limit", type=int, default=15)
    c = sp.add_parser("count"); c.add_argument("--dav"); c.add_argument("--swa"); c.add_argument("--by-src", action="store_true")
    ch = sp.add_parser("check"); ch.add_argument("lexicon")
    a = p.parse_args()
    {"grep": cmd_grep, "align": cmd_align, "count": cmd_count, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
