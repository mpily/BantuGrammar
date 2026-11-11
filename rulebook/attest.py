"""
attest.py

2026-11-11. Corpus evidence for rule-book candidates, counted ONLY on the clean
train split (~/languagemodels/rulebook_data/train.jsonl, built by
build_train_filter.py; dev and every excluded split are never read here).

Two kinds of evidence, both sentence-pair level (no word aligner: the existing
eflomal links were trained on the contaminated pool):

  after_nouns(cls, dav_re)   words directly after (or before) a speaker-verified
                             noun of a class. Lift = rate after this class /
                             rate after nouns of all other classes.
  cooc(sw_feature, dav_re)   pairs whose Swahili side has a feature (from
                             swahili_verb.parse) AND whose Kidaw'ida side has a
                             word matching dav_re. Lift = P(dav | feature) /
                             P(dav | no feature).

Examples are the shortest matching pairs (easiest for a speaker to check), ties
by id. Counts are reported per corpus (davswa / bible).
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hf_space"))
from rulebook_flagger import norm  # noqa: E402
from swahili_verb import parse  # noqa: E402

TRAIN = Path.home() / "languagemodels" / "rulebook_data" / "train.jsonl"
N_EXAMPLES = 3


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def toks(text: str) -> list[str]:
    return [t for t in (norm(w) for w in text.split()) if t]


@lru_cache(maxsize=1)
def pairs() -> list[dict]:
    out = []
    for line in open(TRAIN, encoding="utf-8"):
        r = json.loads(line)
        out.append({"id": r["id"], "src": r["src"], "swa": r["swa"], "dav": r["dav"],
                    "sw": toks(r["swa"]), "dv": toks(r["dav"])})
    return out


@lru_cache(maxsize=200000)
def sw_parse(word: str):
    return parse(word)


SW_DEM_WORDS = set(("huyu hawa huu hii hili haya hiki hivi hizi huku hapa humu yule wale ule ile lile yale kile vile "
                     "zile kule pale mle huyo hao huo hiyo hilo hayo hicho hivyo hizo huko hapo humo").split())
SW_HAVE = {"nina", "una", "ana", "tuna", "mna", "wana", "kina", "ina", "zina", "vina", "lina"}
SW_COPULA_LOC = {"niko", "uko", "yuko", "tuko", "mko", "iko", "kiko", "viko", "ziko", "liko", "kuna", "pana", "mna"}


def sw_features(word: str) -> set[str]:
    """Swahili verb features used by the TAM / NEG / REL / OM rules."""
    from rulebook_correct import _SW_NEG_STOP  # haki, hali, hawa ... are not verbs (agent round 1: hachi)
    f = set()
    if word in SW_COPULA_LOC:
        return {"copula:ko"}
    if word in SW_HAVE:
        return {"have"}
    if len(word) >= 5 and word.endswith("eni") and not sw_parse(word):
        f.add("imp:pl")
    if word in _SW_NEG_STOP or word in SW_DEM_WORDS:   # vile, kile are demonstratives, not subjunctives
        return f
    p = sw_parse(word)
    if not p:
        if len(word) >= 6 and word.endswith("ni"):
            f.add("loc:ni")
        return f
    mood, tam, rel = p["mood"], p["tam"], p["rel"]
    if p["om"]:
        f.add(f"om:{p['om']}")
    if "passive" in p["exts"] or p["root"].endswith("w"):
        f.add("ext:passive")
    if p["root"] in ("kuw", "w") or word.endswith("likuwa"):
        f.add("kuwa")
    if mood == "prohibitive":
        f.add("prohib")
    elif mood == "negative":
        f.add(f"negsm:{p['sm']}")
        if p["neg"] == "hu" and p["fv"] == "a" and tam is None:
            f.add("hab")  # hu-R-a habitual (the parser files hu- under negative)
        elif tam in ("ku", "ja", "ta", "nge"):
            f.add(f"neg:{tam}")
        elif tam is None and p["fv"] == "i":
            f.add("neg:pres")
    elif mood == "habitual":
        f.add("hab")
    elif mood == "subjunctive":
        f.add("subj")  # SM-ROOT-e (nde- hortative in Kidaw'ida: agent round 2)
        f.add(f"sm:{p['sm']}")
    elif mood == "affirmative":
        if rel:
            f.add(f"rel:{rel}")
            f.add("rel:lipo" if (rel == "po" and tam == "li") else f"reltam:{tam}")
        else:
            f.add(f"tam:{tam}")
            f.add(f"sm:{p['sm']}")
    return f


@lru_cache(maxsize=1)
def pair_features() -> list[set[str]]:
    out = []
    for r in pairs():
        f = set()
        fs = [sw_features(w) for w in r["sw"]]
        for k, x in enumerate(fs):
            f |= x
            # past progressive: alikuwa akisoma (agent round 1)
            if "kuwa" in x and "tam:li" in x and k + 1 < len(fs) and "tam:ki" in fs[k + 1]:
                f.add("pastprog")
            if "tam:li" in x and "ext:passive" in x:
                f.add("past:passive")
        out.append(f)
    return out


def _examples(idx: list[int], words: dict[int, str] | None = None) -> list[dict]:
    ps = pairs()
    best = sorted(idx, key=lambda i: (len(ps[i]["sw"]) + len(ps[i]["dv"]), ps[i]["id"]))[:N_EXAMPLES]
    out = []
    for i in best:
        # no sentence text: corpus text stays out of git (Bible text is rights-reserved);
        # render.py adds the text to a local review copy from train.jsonl
        e = {"pair_id": ps[i]["id"]}
        if words and i in words:
            e["match"] = words[i]
        out.append(e)
    return out


def _by_src(idx) -> dict:
    c = collections.Counter(pairs()[i]["src"] for i in idx)
    return {"davswa": c.get("davswa", 0), "bible": c.get("bible", 0)}


def cooc(feature, dav_re: str, exclude: set[str] = frozenset()) -> dict:
    """feature: a feature string, or a predicate over the pair's feature set."""
    rx = re.compile(dav_re)
    has = feature if callable(feature) else (lambda fs: feature in fs)
    both, words = [], {}
    n_f = n_d = n_d_nof = 0
    for i, (r, fs) in enumerate(zip(pairs(), pair_features())):
        hit = next((w for w in r["dv"] if w not in exclude and rx.fullmatch(w)), None)
        f = has(fs)
        n_f += f
        n_d += hit is not None
        if hit is not None and f:
            both.append(i)
            words[i] = hit
        elif hit is not None:
            n_d_nof += 1
    n = len(pairs())
    p_f = len(both) / n_f if n_f else 0.0
    p_nof = n_d_nof / (n - n_f) if n - n_f else 0.0
    return {"method": "pair_cooccurrence", "dav_regex": dav_re, "evidence_count": len(both),
            "n_pairs_with_sw_feature": n_f, "share_of_feature_pairs": round(p_f, 4),
            "lift": round(p_f / p_nof, 2) if p_nof else None, "domains": _by_src(both),
            "examples": _examples(both, words)}


def after_nouns(nouns_by_class: dict[str, set[str]], cls: str, dav_re: str, before: bool = False,
                sw_re: str | None = None, same_form: frozenset = frozenset()) -> dict:
    """Words right after (or before) verified nouns of class `cls`. Classes in
    `same_form` use the identical form for this slot, so they are left out of
    the lift baseline (they would count as hits for the wrong reason)."""
    rx = re.compile(dav_re)
    srx = re.compile(sw_re) if sw_re else None
    of = {n: c for c, ns in nouns_by_class.items() for n in ns}
    hits_c, words = [], {}
    n_c = n_o = hit_o = 0
    for i, r in enumerate(pairs()):
        if srx and not any(srx.fullmatch(w) for w in r["sw"]):
            continue
        t = r["dv"]
        found = False
        for j, w in enumerate(t):
            c = of.get(w)
            k = j - 1 if before else j + 1
            if c is None or not 0 <= k < len(t):
                continue
            m = rx.fullmatch(t[k]) is not None
            if c == cls:
                n_c += 1
                if m and not found:
                    hits_c.append(i)
                    words[i] = f"{w} {t[k]}" if not before else f"{t[k]} {w}"
                    found = True
            elif c not in same_form:
                n_o += 1
                hit_o += m
    rate_c = len(hits_c) / n_c if n_c else 0.0
    rate_o = hit_o / n_o if n_o else 0.0
    return {"method": "after_verified_nouns" if not before else "before_verified_nouns",
            "dav_regex": dav_re, "sw_regex": sw_re, "baseline_excludes": sorted(same_form), "evidence_count": len(hits_c),
            "n_noun_tokens": n_c, "lift": round(rate_c / rate_o, 2) if rate_o else None,
            "domains": _by_src(hits_c), "examples": _examples(hits_c, words)}


def explore(feature: str, prefix_len: int = 5, min_count: int = 5, top: int = 15):
    """Kidaw'ida word starts most specific to pairs with a Swahili feature (for finding candidates)."""
    with_f, without = collections.Counter(), collections.Counter()
    n_f = 0
    for r, fs in zip(pairs(), pair_features()):
        starts = {w[:prefix_len] for w in r["dv"] if len(w) > prefix_len}
        if feature in fs:
            n_f += 1
            with_f.update(starts)
        else:
            without.update(starts)
    n_o = len(pairs()) - n_f
    rows = [(s, c, round((c / n_f) / ((without[s] + 1) / n_o), 1)) for s, c in with_f.items() if c >= min_count]
    return n_f, sorted(rows, key=lambda x: -x[2])[:top]
