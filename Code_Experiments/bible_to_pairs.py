# -*- coding: utf-8 -*-
"""
bible_to_pairs.py

2026-10-28 cont. 9. Converts a `bible_collect.py align` output
(parallel.jsonl: {"ids": [...], "TAITA": ..., "<SW_ABBR>": ...}) into the
{id, swa, dav} JSONL that nllb_finetune_dav.py --extra-pairs reads.

Checked directly on the scraped data before writing this:
  * every verse on both sides starts with its verse number ("1 Aho ...")
    -> stripped;
  * footnotes / cross-references leaked into the text as "# ..." despite
    bible_collect.py's docstring (TAITA: 3,332/31,103 verses; NEN: 1,205;
    SUV: 17,836). In NEN they sit MID-sentence with no end marker
    ("... mavumbi # 2:7 Neno la Kiebrania ... ya ardhi, ..."), so they
    cannot be cut out reliably -> any pair with "#" on either side is
    DROPPED, not repaired;
  * merged verses (len(ids) > 1) are dropped (only BHN has any: 4).
Counts of every drop reason are printed and written next to the output.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

VERSE_NUM = re.compile(r"^\s*\d+[a-z]?\s+")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", required=True, type=Path, help="bible_collect.py align output (parallel.jsonl)")
    p.add_argument("--sw-abbr", required=True, help="Swahili version key in the file, e.g. NEN")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    counts = Counter()
    kept = []
    for line in open(args.parallel, encoding="utf-8"):
        r = json.loads(line)
        counts["input"] += 1
        if len(r["ids"]) != 1:
            counts["drop_merged_verse"] += 1
            continue
        dav = VERSE_NUM.sub("", r["TAITA"]).strip()
        swa = VERSE_NUM.sub("", r[args.sw_abbr]).strip()
        if "#" in dav or "#" in swa:
            counts["drop_footnote"] += 1
            continue
        if not dav or not swa:
            counts["drop_empty"] += 1
            continue
        kept.append({"id": f"bible:{args.sw_abbr}:{r['ids'][0]}", "swa": swa, "dav": dav})
    counts["kept"] = len(kept)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in kept:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    args.out.with_suffix(".counts.json").write_text(json.dumps(dict(counts), indent=1))
    print(dict(counts))


if __name__ == "__main__":
    main()
