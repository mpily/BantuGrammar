"""
induce_extensions.py

2026-11-11, CLAUDE_rulebook.md §3 sections 5-6 (extensions; final vowel / perfective / -ghe),
induced from aligned verbs like induce_verb_rules.py.

For a Swahili verb whose parse has a decomposition base + extensions (pig-an-a,
andik-i-w-a), the Kidaw'ida translation of the BASE comes from the bootstrapped lexicon;
if it occurs in the aligned Kidaw'ida verb, the piece between it and the end is split
into extension + final (finals: -a, -e, -o, -aa, -agha, -ie, -eghe/-ieghe). Counts per
Swahili extension, conditioned on the Kidaw'ida root shape (final segment palatal / vowel /
other; last vowel e/o vs other). Control: the same piece on aligned verbs of the same base
without that extension (lift).

Also scores the canonical version-B helpers on the same data (CLAUDE_rulebook.md §1.7:
wrap, don't fork): rulebook_correct.applicative_stem / applicative_perfective and
kitaita_verified_corrections.kitaita_passive_form.
Output: rulebook/induced_extension_rules.json (counts, words, pair IDs; no sentences).
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))
import attest as A  # noqa: E402
import bootstrap_lexicon as B  # noqa: E402
from api.stats import rate_ci  # noqa: E402
from kitaita_verified_corrections import kitaita_passive_form  # noqa: E402
from rulebook_correct import applicative_perfective, applicative_stem  # noqa: E402

OUT = HERE / "induced_extension_rules.json"
FINALS = ("ieghe", "eghe", "agha", "aa", "ie", "a", "e", "o")
PALATAL = ("ny", "nj", "sh", "ch", "s", "z", "j")
EXT_TYPES = ("applicative", "passive", "causative", "stative", "reciprocal")


def root_shape(r: str) -> str:
    end = "vowel" if r[-1] in "aeiou" else "palatal" if r.endswith(PALATAL) else "other"
    vs = re.findall(r"[aeiou]", r)
    return f"{end}/{'eo' if vs and vs[-1] in 'eo' else 'aiu'}"


def split_tail(tail: str):
    for f in FINALS:
        if tail.endswith(f):
            return tail[: -len(f)], f
    return tail, ""


def main():
    lex = json.loads(B.OUT.read_text(encoding="utf-8"))
    known = collections.defaultdict(set)
    for e in lex["entries"]:
        if len(e["dav_lemma"]) >= 2:
            known[e["sw_lemma"]].add(e["dav_lemma"])
    ps = A.pairs()
    obs = []          # (ext tuple, dav base, ext surface, final, pair idx, sw word, dav word)
    base_plain = collections.defaultdict(collections.Counter)   # (sw base, dav base) -> ext surfaces w/o Swahili ext
    for i, r in enumerate(ps):
        for sw in r["sw"]:
            p = A.sw_parse(sw)
            if not p:
                continue
            for base, exts in sorted(p["decompositions"], key=lambda x: -len(x[0])):
                hit = None
                for L in sorted(known.get(base, ()), key=len, reverse=True):
                    for dw in r["dv"]:
                        j = dw.find(L)
                        if j > 0 or (j == 0 and len(dw) > len(L)):
                            hit = (L, dw, dw[j + len(L):])
                            break
                    if hit:
                        break
                if not hit:
                    continue
                L, dw, tail = hit
                ext, fin = split_tail(tail)
                if exts:
                    obs.append((tuple(exts), L, ext, fin, i, sw, dw))
                else:
                    base_plain[L][ext] += 1
                break
    plain_total = collections.Counter()
    for c in base_plain.values():
        plain_total.update(c)
    n_plain = sum(plain_total.values())
    out = {"_meta": {"built_by": "rulebook/induce_extensions.py", "script_sha16": A.sha16(Path(__file__)),
                     "bootstrap_lexicon_sha16": A.sha16(B.OUT), "train_sha16": A.sha16(A.TRAIN),
                     "n_observations": len(obs), "n_plain_controls": n_plain}, "extensions": {}, "helpers": {}}
    for et in EXT_TYPES:
        xs = [o for o in obs if o[0] == (et,)]
        by_surface = collections.Counter(o[2] for o in xs)
        rows = []
        for surf, k in by_surface.most_common(12):
            share = k / len(xs)
            base_rate = plain_total[surf] / n_plain if n_plain else 0
            cond = collections.Counter(root_shape(o[1]) for o in xs if o[2] == surf)
            rows.append({"surface": surf or "(none)", "n": k, "share": rate_ci(k, len(xs)),
                         "lift_vs_no_ext": round(share / base_rate, 2) if base_rate else None,
                         "by_root_shape": dict(cond.most_common()),
                         "examples": [{"pair_id": ps[o[4]]["id"], "sw_word": o[5], "dav_word": o[6], "dav_root": o[1]}
                                      for o in xs if o[2] == surf][:3]})
        shape_table = {}
        for sh in sorted({root_shape(o[1]) for o in xs}):
            c = collections.Counter(o[2] for o in xs if root_shape(o[1]) == sh)
            n = sum(c.values())
            shape_table[sh] = {"n": n, "top": [(s or "(none)", k, round(k / n, 2)) for s, k in c.most_common(4)]}
        finals = collections.Counter(o[3] for o in xs)
        out["extensions"][et] = {"n": len(xs), "surfaces": rows, "by_root_shape": shape_table,
                                 "finals": dict(finals.most_common(8))}
    # section 6: perfective final by root shape (aligned verbs, no extension, Swahili -li-/-me-),
    # and post-final -ghe by Swahili tense
    import word_align as W
    sw_lem, dv_lem = B.lemmatizers()
    perf = collections.defaultdict(collections.Counter)
    perf_ex = collections.defaultdict(list)
    ghe = collections.defaultdict(collections.Counter)
    for i, sw, fs, dw in W.aligned():
        L = dv_lem.get(dw, dw)
        j = dw.find(L)
        if j < 0 or len(L) < 2:
            continue
        tail = dw[j + len(L):]
        for t in ("tam:li", "tam:me", "reltam:li", "neg:ku", "tam:na", "tam:ta"):
            if t in fs:
                ghe[t]["ghe" if dw.endswith("ghe") else "no_ghe"] += 1
        if not ({"tam:li", "tam:me", "reltam:li", "neg:ku"} & fs):
            continue
        m = re.fullmatch(r"(ie|ire|ere|e)(ghe)?", tail)
        if m:
            sh = root_shape(L)
            perf[sh][m.group(1)] += 1
            if len(perf_ex[(sh, m.group(1))]) < 3:
                perf_ex[(sh, m.group(1))].append({"pair_id": ps[i]["id"], "sw_word": sw, "dav_word": dw, "dav_root": L})
    out["perfective"] = {sh: {"n": sum(c.values()), "finals": dict(c.most_common()),
                              "examples": {f: perf_ex[(sh, f)] for f in c}} for sh, c in perf.items()}
    out["post_final_ghe"] = {t: {"n": sum(c.values()), "ghe": c["ghe"],
                                 "rate": rate_ci(c["ghe"], sum(c.values()))} for t, c in ghe.items()}
    # canonical helpers, scored on the same observations
    for name, ets, fn in (("applicative_stem", ("applicative",), lambda L: applicative_stem(L)),
                          ("applicative_perfective+ie", ("applicative",), lambda L: applicative_perfective(L) + "ie"),
                          ("kitaita_passive_form", ("passive",), lambda L: kitaita_passive_form(L))):
        xs = [o for o in obs if o[0] == ets]
        k = 0
        miss = []
        for o in xs:
            if fn(o[1]) in o[6]:
                k += 1
            elif len(miss) < 5:
                miss.append({"pair_id": ps[o[4]]["id"], "sw_word": o[5], "dav_word": o[6], "dav_root": o[1],
                             "predicted": fn(o[1])})
        vis = [o for o in xs if o[2]]
        kv = sum(fn(o[1]) in o[6] for o in vis)
        out["helpers"][name] = {"n": len(xs), "k": k, "rate": rate_ci(k, len(xs)) if xs else None,
                                "n_with_visible_extension": len(vis), "k_visible": kv,
                                "rate_visible": rate_ci(kv, len(vis)) if vis else None, "misses": miss}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for et, v in out["extensions"].items():
        print(f"{et}: n={v['n']} finals={v['finals']}")
        for r in v["surfaces"][:6]:
            print(f"   {r['surface']:<8} {r['n']:>4} {r['share']['rate']:.2f} lift={r['lift_vs_no_ext']} "
                  f"{r['by_root_shape']} {[e['sw_word'] + '>' + e['dav_word'] for e in r['examples'][:2]]}")
        for sh, t in v["by_root_shape"].items():
            print(f"      {sh:<12} n={t['n']:<4} {t['top']}")
    for h, v in out["helpers"].items():
        print(h, v["k"], "/", v["n"], [m["sw_word"] + ">" + m["dav_word"] + " pred " + m["predicted"] for m in v["misses"][:3]])


if __name__ == "__main__":
    main()
