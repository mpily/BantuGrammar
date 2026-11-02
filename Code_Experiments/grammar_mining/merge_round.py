# -*- coding: utf-8 -*-
"""
merge_round.py -- pool one mining round's agent outputs and set up the next.

  python3 merge_round.py --round round1

Reads  ~/languagemodels/mining/<round>/*.json  (schema: BRIEFING.md)
Writes ~/languagemodels/mining/<round>/_merged/
  merged_entries.json   all entries, tagged with their area
  lexicon.json          union of every area's lexicon_for_check
  conflicts.json        Swahili words that different areas map to DISJOINT Kidaw'ida forms
  gaps.json             frequent Swahili words in MINE with no entry in any area
                        (with alignment top candidates, as round-2 work items)
  report.md             per-area entry counts by confidence/method, holdout scores
                        per area and pooled (same scorer as mine_query.py check)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import mine_query as mq

DATA = mq.DATA


def holdout_score(lex: dict[str, set[str]]):
    k = n = 0
    for r in mq.load("holdout"):
        d = {mq.norm(w) for w in r["dav_tok"]}
        for s in set(r["swa_tok"]):
            if s in lex:
                n += 1
                k += bool(lex[s] & d)
    lo, hi = mq.wilson(k, n)
    return k, n, lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", required=True)
    p.add_argument("--gap-top", type=int, default=600, help="how many frequent uncovered Swahili words to list")
    a = p.parse_args()
    rd = DATA / a.round
    out = rd / "_merged"
    out.mkdir(exist_ok=True)

    areas, entries, lex_by_area = {}, [], {}
    for f in sorted(rd.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"!! {f.name}: invalid JSON ({e})")
            continue
        area = d.get("area", f.stem)
        areas[area] = d
        for e in d.get("entries", []):
            entries.append(dict(e, area=area))
        L = {}
        for s, v in (d.get("lexicon_for_check") or {}).items():
            vs = [v] if isinstance(v, str) else list(v or [])
            if vs:
                L[s.lower()] = {mq.norm(x.lower()) for x in vs}
        lex_by_area[area] = L

    pooled = defaultdict(set)
    for L in lex_by_area.values():
        for s, v in L.items():
            pooled[s] |= v

    conflicts = []
    for s in pooled:
        claims = {ar: L[s] for ar, L in lex_by_area.items() if s in L}
        if len(claims) > 1:
            sets = list(claims.values())
            if any(not (x & y) for i, x in enumerate(sets) for y in sets[i + 1:]):
                conflicts.append({"swahili": s, "claims": {k: sorted(v) for k, v in claims.items()}})

    covered = set(pooled)
    for e in entries:
        sw = e.get("swahili")
        for w in ([sw] if isinstance(sw, str) else (sw or [])):
            covered.add(str(w).lower())
    freq = Counter(w for r in mq.load("mine") for w in set(r["swa_tok"]))
    gaps = []
    for w, c in freq.most_common():
        if w in covered or len(w) < 3 or w.isdigit():
            continue
        cands = sorted(mq.table().get(w, []), key=lambda x: -x[1])[:5]
        gaps.append({"swahili": w, "sentence_freq": c,
                     "alignment_top": [{"dav": d, "links": l, "davswa": nd, "bible": nb} for d, l, nd, nb in cands]})
        if len(gaps) >= a.gap_top:
            break

    (out / "merged_entries.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    (out / "lexicon.json").write_text(json.dumps({s: sorted(v) for s, v in pooled.items()}, ensure_ascii=False, indent=1))
    (out / "conflicts.json").write_text(json.dumps(conflicts, ensure_ascii=False, indent=1))
    (out / "gaps.json").write_text(json.dumps(gaps, ensure_ascii=False, indent=1))

    lines = [f"# {a.round} merge report", "", "| area | entries | high | medium | low | statistical | linguistic | both | lexicon keys | holdout hit [95% CI] (n) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for area, d in areas.items():
        es = d.get("entries", [])
        cc, mc = Counter(e.get("confidence") for e in es), Counter(e.get("method") for e in es)
        k, n, lo, hi = holdout_score(lex_by_area[area])
        hs = f"{k / n:.3f} [{lo:.3f}, {hi:.3f}] ({n})" if n else "--"
        lines.append(f"| {area} | {len(es)} | {cc['high']} | {cc['medium']} | {cc['low']} | {mc['statistical']} | {mc['linguistic']} | {mc['both']} | {len(lex_by_area[area])} | {hs} |")
    k, n, lo, hi = holdout_score(pooled)
    lines += ["", f"**Pooled lexicon**: {len(pooled)} Swahili words; holdout hit {k}/{n} = {k / max(n, 1):.3f} [{lo:.3f}, {hi:.3f}]",
              "", "Holdout hit = the Swahili word occurs in a holdout pair and one of its predicted Kidaw'ida forms occurs in that pair's Kidaw'ida side (apostrophe-insensitive). Free translation means a correct mapping can still miss; compare areas against each other, not against 100%.",
              "", f"Conflicts (areas disagree, disjoint forms): {len(conflicts)}", f"Gap list: top {len(gaps)} frequent uncovered Swahili words", ""]
    tok_cov = sum(c for w, c in freq.items() if w in pooled) / sum(freq.values())
    lines.append(f"Swahili token coverage of MINE by the pooled lexicon (sentence-level counts): {tok_cov:.1%}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
