"""
swahili_verb.py

2026-11-03. A small Swahili verb parser for the version-B tense and extension
steps (rulebook_correct.py). Pure Python (the pipeline's own parser lives in a
module that imports torch). Template, as in round-1 verb_morphology:

  NEG - SM - TAM - REL - OM - root - extensions - final

Many words have several readings; a reading is kept only if its root is in the
Swahili -> Kidaw'ida root table (kitaita_root_alignment.json), and the reading
with the longest root wins. Extensions are peeled from the end of the stem
(passive -w-, applicative -i-/-e-/-li-/-le-, causative -ish-/-esh-, stative
-ik-/-ek-, reciprocal -an-), but a stem that is itself in the root table is
kept whole (pigana 'fight' is a root, not pig + reciprocal). Not
speaker-verified; covers common forms, not all of Swahili.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SM = ("ni", "u", "a", "tu", "m", "mu", "wa", "i", "li", "ki", "vi", "zi", "ya", "ku", "pa", "yu")
NEG_SM = {"si": "ni", "hu": "u", "ha": "a", "hatu": "tu", "ham": "m", "hawa": "wa", "hai": "i", "hali": "li",
          "haki": "ki", "havi": "vi", "hazi": "zi", "haya": "ya", "hau": "u", "haku": "ku", "hapa": "pa"}
TAM = ("ngali", "nge", "li", "na", "me", "ta", "ka", "ki", "sha")
NEG_TAM = ("ku", "ja", "ta", "nge")
REL = ("cho", "vyo", "yo", "zo", "lo", "po", "ko", "mo", "ye", "o", "wo")
OM = ("ni", "ku", "m", "mw", "tu", "wa", "ki", "vi", "li", "ya", "zi", "i", "u", "ji", "pa")
# extensions, peeled longest first
EXT = (("esh", "causative"), ("ish", "causative"), ("ikan", "stative"), ("ekan", "stative"),
       ("ik", "stative"), ("ek", "stative"), ("liw", "passive"), ("lew", "passive"), ("iw", "passive"),
       ("ew", "passive"), ("w", "passive"), ("an", "reciprocal"), ("li", "applicative"), ("le", "applicative"),
       ("i", "applicative"), ("e", "applicative"))


@lru_cache(maxsize=1)
def roots() -> dict:
    p = Path(__file__).parent / "kitaita_root_alignment.json"
    return json.loads(p.read_text(encoding="utf-8"))["roots"]


def decompositions(stem: str) -> list[tuple[str, list[str]]]:
    """Every (root, [extensions]) split of a stem whose root is in the table:
    the whole stem first if it is a root, then peeled ones (pigani -> pigan +
    applicative). Roots of 1 letter or the copula ku/w are not peeled into."""
    r = roots()
    out = []
    if stem in r:
        out.append((stem, []))
    for suf, label in EXT:
        base = stem[:-len(suf)]
        if stem.endswith(suf) and (len(base) >= 2 or base in r):  # one-letter roots only if listed (p, l)
            for root, exts in decompositions(base):
                if root not in ("ku", "w") and (root, exts + [label]) not in out:
                    out.append((root, exts + [label]))
    return out


FINALS = ("eni", "a", "e", "i", "u")


def _readings(tok: str):
    """All (mood, neg, sm, tam, rel, om, stem, fv) splits of a token."""
    def tails(rest):
        for rel in (None,) + REL:
            r1 = rest
            if rel:
                if not r1.startswith(rel):
                    continue
                r1 = r1[len(rel):]
            for om in (None,) + OM:
                r2 = r1
                if om:
                    if not r2.startswith(om):
                        continue
                    r2 = r2[len(om):]
                for fv in FINALS:
                    if r2.endswith(fv) and len(r2) > len(fv):
                        yield rel, om, r2[:-len(fv)], fv

    for negp, sm in NEG_SM.items():
        if tok.startswith(negp):
            rest = tok[len(negp):]
            for tam in (None,) + NEG_TAM:
                r0 = rest
                if tam:
                    if not r0.startswith(tam):
                        continue
                    r0 = r0[len(tam):]
                for rel, om, stem, fv in tails(r0):
                    yield "negative", negp, sm, tam, rel, om, stem, fv
    for sm in SM:
        if tok.startswith(sm + "si"):
            for rel, om, stem, fv in tails(tok[len(sm) + 2:]):
                if fv == "e" and rel is None:
                    yield "prohibitive", None, sm, None, None, om, stem, fv
        for tam in TAM:
            if tok.startswith(sm + tam):
                for rel, om, stem, fv in tails(tok[len(sm) + len(tam):]):
                    yield "affirmative", None, sm, tam, rel, om, stem, fv
        if tok.startswith(sm):
            for rel, om, stem, fv in tails(tok[len(sm):]):
                if fv == "e" and rel is None:
                    yield "subjunctive", None, sm, None, None, om, stem, fv
    if tok.startswith("hu"):
        for rel, om, stem, fv in tails(tok[2:]):
            if rel is None:
                yield "habitual", None, None, "hu", None, om, stem, fv
    if tok.startswith("ku"):
        for rel, om, stem, fv in tails(tok[2:]):
            if rel is None and fv == "a":
                yield "infinitive", None, None, None, None, om, stem, fv
    for fv in ("eni", "a"):
        if tok.endswith(fv) and len(tok) > len(fv) + 1:
            yield "imperative", None, None, None, None, None, tok[:-len(fv)], fv


MOOD_RANK = {"negative": 6, "prohibitive": 5, "affirmative": 4, "habitual": 3, "infinitive": 2,
             "subjunctive": 1, "imperative": 0}


def parse(token: str) -> dict | None:
    """Best reading with a known root:
    {mood, neg, sm, tam, rel, om, root, exts, fv, decompositions}.
    Preference: longest whole-stem root, then mood (a marked reading over a
    bare imperative/subjunctive), then fewer peeled extensions."""
    tok = re.sub(r"[^a-z]", "", (token or "").lower())
    if len(tok) < 2:
        return None
    best = None
    tier = {"verified": 3, "round1": 2, "infinitive": 2, "pipeline": 0}
    for mood, neg, sm, tam, rel, om, stem, fv in _readings(tok):
        decs = decompositions(stem)
        if fv in ("i", "u", "e") and (stem + fv) in roots():
            decs = [(stem + fv, [])] + decs  # vowel-final loan root (fahamu, rudi)
        if not decs:
            continue
        root, exts = decs[0]
        # trusted root sources first (pipeline roots sometimes absorb an object
        # marker: nifaham), then the longest root, then a marked mood
        key = (tier.get(roots()[root]["source"], 0), len(root), MOOD_RANK[mood], -len(exts), -len(om or ""))
        if best is None or key > best[0]:
            best = (key, {"mood": mood, "neg": neg, "sm": sm, "tam": tam, "rel": rel, "om": om,
                          "root": root, "exts": exts, "fv": fv, "stem": stem, "decompositions": decs})
    return best[1] if best else None


def kitaita_roots(swahili_root: str) -> list[str]:
    e = roots().get(swahili_root)
    return e["kitaita"] if e else []
