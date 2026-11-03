"""
kitaita_verb.py

2026-11-03. Kidaw'ida verb prefixes, for the version-B tense and extension
steps (rulebook_correct.py). Tables from round-1 verb_morphology.md (corpus-
mined, NOT speaker-verified):

  template  [nde-|si-] SM - TAM - [OM] - root - [ext] - final
  SM series per person/class: bare / SM+a (perfect, present) / SM+e (past)

A prefix (the part of a draft word before the verb root) is split into
(neg, row, variant, tam, om) in every way the tables allow; the caller picks
the reading that fits the Swahili verb.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from rulebook_flagger import norm

# row -> (bare, SM+a, SM+e); 2pl bare also "m", class 9 SM+e also "ye"
SERIES = {
    "1sg": ("ni", "na", "ne"), "2sg": ("ku", "kwa", "ko"), "3sg": ("u", "wa", "o"), "1pl": ("di", "da", "de"),
    "2pl": ("mu", "mwa", "mo"), "c2": ("wi", "wa", "we"), "c3": ("ghu", "ghwa", "gho"), "c9": ("i", "ya", "e"),
    "c5": ("ji", "ja", "je"), "c6": ("ghi", "gha", "ghe"), "c7": ("chi", "cha", "che"), "c8": ("vi", "va", "ve"),
    "c10": ("ri", "ra", "re"), "c11": ("lu", "lwa", "lo"), "c15": ("ku", "kwa", "ko"),
}
EXTRA_FORMS = {("2pl", 0): ("m",), ("c9", 2): ("ye",)}
VARIANTS = ("bare", "a", "e")
TAMS = ("seke", "sake", "chaa", "cha", "wia", "we", "da", "de", "re", "ka", "ki", "se", "nda")
OMS = ("mmb", "ni", "ku", "m", "di", "wi", "ghi", "ri", "chi", "vi", "ji", "ghu", "lu", "i")
# Swahili subject marker -> Kidaw'ida rows it can stand for
SW_SM_ROWS = {"ni": ("1sg",), "u": ("2sg", "c3", "c11"), "a": ("3sg",), "yu": ("3sg",), "tu": ("1pl",),
              "m": ("2pl",), "mu": ("2pl",), "wa": ("c2",), "i": ("c9",), "li": ("c5",), "ya": ("c6",),
              "ki": ("c7",), "vi": ("c8",), "zi": ("c10",), "ku": ("c15",)}


@lru_cache(maxsize=1)
def _forms():
    out = []
    for row, forms in SERIES.items():
        for vi, f in enumerate(forms):
            for g in (f,) + EXTRA_FORMS.get((row, vi), ()):
                out.append((g, row, VARIANTS[vi]))
    return out


def parse_prefix(prefix: str) -> list[dict]:
    """Every reading of a verb prefix as [neg] SM [TAM] [OM]."""
    p = norm(prefix)
    out = []
    for neg in (None, "nde", "si"):
        r0 = p
        if neg:
            if not r0.startswith(neg):
                continue
            r0 = r0[len(neg):]
        for form, row, variant in _forms():
            if not r0.startswith(form):
                continue
            r1 = r0[len(form):]
            for tam in (None,) + TAMS:
                r2 = r1
                if tam:
                    if not r2.startswith(tam):
                        continue
                    r2 = r2[len(tam):]
                for om in (None,) + OMS:
                    if (om or "") == r2:
                        out.append({"neg": neg, "row": row, "variant": variant, "sm": form, "tam": tam, "om": om})
    return out


def build_prefix(neg, row, variant, tam, om) -> str:
    return (neg or "") + SERIES[row][VARIANTS.index(variant)] + (tam or "") + (om or "")


@lru_cache(maxsize=1)
def vocab() -> dict:
    d = json.loads((Path(__file__).parent / "kitaita_training_vocab.json").read_text(encoding="utf-8"))
    out = {}
    for w, c in d["vocab"].items():
        k = norm(w)
        out[k] = out.get(k, 0) + c
    return out


@lru_cache(maxsize=4096)
def attested_ending(stem_and_final: str) -> int:
    """Corpus tokens of words that end in this stem + final (any prefix)."""
    return sum(c for w, c in vocab().items() if w.endswith(stem_and_final) and len(w) > len(stem_and_final))
