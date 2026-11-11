"""
segment.py

2026-11-11. Root-inventory segmentation (user: a cheap pass to improve the word cuts, after agent
rounds 1-3 all named stem cutting as the main lexicon error).

Inventory = roots the agents gave in rounds 1-3 (proposals, high/medium confidence, root inside
the word) + root-list answers (type "root", segmentation pass) + speaker-verified noun forms and
verb roots. A word is analysed as
    [prefixes] ROOT [extensions] [final] [-ghe]
where prefixes are a NEG / SM / TAM / OM chain or a noun prefix (rule-book forms), ROOT is in the
inventory, and the tail is a valid sequence of rule-book extensions and finals. The longest root
wins; ties -> fewer prefix letters. m+w/b -> mmb is undone. Words with no analysis keep the
loop's own lemma (bootstrap_lexicon.choose_lemmas).
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))
from rulebook_flagger import norm  # noqa: E402

AGENT_DIR = Path.home() / "languagemodels" / "rulebook_data" / "agents"
DAV_EXT = r"(?:ir|er|il|el|ish|esh|is|es|ik|ek|lw|an|i|w|l|k|n|y)"
DAV_TAIL = re.compile(rf"({DAV_EXT}{{0,3}})(?:ieghe|ireghe|ereghe|iloghe|eloghe|iroghe|eroghe|eghe|agha|aa|ie|ire|ere|"
                      rf"ilo|elo|iro|ero|enyi|onyi|nyi|a|e|o|i|u)?(?:ghe)?(?:nyi)?")
SW_EXT = r"(?:esh|ish|ik|ek|iw|ew|w|an|li|le|i|e)"
SW_TAIL = re.compile(rf"({SW_EXT}{{0,3}})(?:a|e|i|u|o)?(?:ni)?")


def cut(lemma: str) -> str:
    """Inventory roots without the final vowel (kunda -> kund), like the loop's stems."""
    lemma = norm(lemma)
    return lemma[:-1] if len(lemma) >= 4 and lemma[-1] in "ae" else lemma


@lru_cache(maxsize=1)
def inventory() -> dict[str, collections.Counter]:
    """side -> root -> votes."""
    inv = {"dav": collections.Counter(), "sw": collections.Counter()}
    for f in sorted(AGENT_DIR.glob("round*/out_agent*.jsonl")) + sorted(AGENT_DIR.glob("segment*/out_agent*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("confidence") not in ("high", "medium"):
                continue
            if o.get("type") == "root" and o.get("root") and o.get("word"):
                r, w = cut(o["root"]), norm(o["word"])
                if len(r) >= 2 and r in w and o.get("side") in inv:
                    inv[o["side"]][r] += 2
            elif o.get("type") == "proposal":
                for side, wk, lk in (("sw", "sw_word", "sw_lemma"), ("dav", "dav_word", "dav_lemma")):
                    w, l = o.get(wk), o.get(lk)
                    if not w or not l or " " in str(w).strip() or " " in str(l).strip():
                        continue
                    r = cut(l)
                    if len(r) >= 2 and r in norm(w):
                        inv[side][r] += 1
    import kitaita_verified_corrections as V
    for sw, dv in V.VERIFIED_NOUN_TRANSLATIONS.items():
        if isinstance(dv, str) and " " not in dv:
            inv["dav"][cut(dv)] += 3
            inv["sw"][cut(sw)] += 3
    for sw, dv in V.VERIFIED_ROOTS.items():
        inv["dav"][norm(dv)] += 3
        inv["sw"][norm(sw)] += 3
    # clean: roots that still carry m+w -> mmb (mmbo) are artefacts; locative -(e)nyi is not part of a root
    clean = collections.Counter()
    for r, c in inv["dav"].items():
        if r.startswith("mmb"):
            continue
        for suf in ("enyi", "nyi"):
            if r.endswith(suf) and len(r) - len(suf) >= 3:
                r = r[: -len(suf)]
                break
        clean[r] += c
    inv["dav"] = clean
    return inv


def score(root: str, votes: int, ext_letters: int, prefix_letters: int) -> float:
    """Longer and better-attested roots win; each extension letter costs (enikira: e-ni-kir-a, not
    eni-k-ir-a); a well-attested root beats a longer stray one (wadakundana: kund-an-a, not ndan-a)."""
    return len(root) + 1.5 * math.log2(1 + votes) - 0.75 * ext_letters - 0.1 * prefix_letters


@lru_cache(maxsize=1)
def word_roots() -> dict[str, dict[str, str]]:
    """Direct answers of the root-list pass (type "root", high/medium): side -> word -> root."""
    out = {"dav": {}, "sw": {}}
    for f in sorted(AGENT_DIR.glob("segment*/out_agent*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") == "root" and o.get("confidence") in ("high", "medium") and o.get("side") in out:
                w, r = norm(o.get("word") or ""), cut(o.get("root") or "")
                if w and len(r) >= 2 and r in w:
                    out[o["side"]][w] = r
    return out


@lru_cache(maxsize=1)
def _prefixes():
    import bootstrap_lexicon as B
    sw_prefix = []
    import swahili_verb as S
    for neg in ("",) + tuple(S.NEG_SM):
        sw_prefix.append(neg)
    return B, (tuple(S.SM), tuple(S.TAM), tuple(S.REL), tuple(S.OM), tuple(S.NEG_SM))


@lru_cache(maxsize=None)
def dav_root(word: str) -> str | None:
    B, _ = _prefixes()
    inv = inventory()["dav"]
    if len(word) < 3:
        return None
    starts = B._strip_prefixes(word, (B.DAV_NEG, B.DAV_SM, B.DAV_TAM, B.DAV_OM)) | B._strip_prefixes(word, (B.DAV_NOUN,))
    for x in list(starts):
        if x.startswith("mb") and len(x) >= 4:
            starts |= {"w" + x[2:], "b" + x[2:]}
    best = None
    for x in starts:
        for k in range(len(x), 1, -1):
            r = x[:k]
            m = DAV_TAIL.fullmatch(x[k:])
            if r in inv and m:
                cand = (score(r, inv[r], len(m.group(1)), len(word) - len(x)), r)
                if best is None or cand > best:
                    best = cand
    return best[-1] if best else None


@lru_cache(maxsize=None)
def sw_root(word: str) -> str | None:
    B, (SM, TAM, REL, OM, NEG_SM) = _prefixes()
    inv = inventory()["sw"]
    if len(word) < 4:
        return None
    starts = B._strip_prefixes(word, (tuple(NEG_SM) + ("ku",), SM, TAM, REL, OM)) | \
        B._strip_prefixes(word, (B.SW_NOUN,))
    best = None
    for x in starts:
        for k in range(len(x), 1, -1):
            r = x[:k]
            m = SW_TAIL.fullmatch(x[k:])
            if r in inv and m:
                cand = (score(r, inv[r], len(m.group(1)), len(word) - len(x)), r)
                if best is None or cand > best:
                    best = cand
    return best[-1] if best else None
