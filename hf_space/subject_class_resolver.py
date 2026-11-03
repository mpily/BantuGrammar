"""
subject_class_resolver.py

2026-11-03. Subject-marker class of a sentence's subject, for the version-B
subject step (rulebook_correct.py). User request after the first subject step
failed on guessed classes (docs/status.md 2026-11-03): "build the resolver".

Class PAIR from the Kidaw'ida side, NUMBER from the Swahili verb:
  1. class pair of the Kidaw'ida noun, first source that has it:
       verified    VERIFIED_SUBJECT_MARKERS (speaker, incl. the Class Sorter batch)
       votes       the verb after the noun in the training text
                   (kitaita_subject_classes.json, build_subject_classes.py)
       infinitive  ku-...-a nouns -> class 15 (ku-)
       concord     noun-phrase concord after the noun (kitaita_agreement_data.json)
       round1      round-1 noun lexemes (kitaita_noun_classes_round1.json); last,
                   because its classes describe the noun's form, and some nouns
                   agree in another class (punda: class 9 form, wa- agreement)
     Order chosen by 5-fold cross-validation against the verified nouns
     (eval_subject_resolver.py), not on any test sentences: with the verified
     tier off it answers for 74% [67, 79] and is right 91% [86, 94] of the time.
  2. number from the Swahili verb's subject marker: a-/yu-/u-/li-/ki- singular,
     wa-/ya-/vi-/zi- plural, i- plural only after a mi- subject noun.
  3. the member of the pair with that number, as a subject marker.
Anything ambiguous returns None (no edit). Nothing here is speaker-verified
except tier 1.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from kitaita_verified_corrections import (NUMBER_INVARIANT_NOUNS, TAM_SM_ALIAS, VERIFIED_NOUN_TRANSLATIONS,
                                          VERIFIED_SUBJECT_MARKERS)
from rulebook_flagger import norm

_HERE = Path(__file__).parent
NUM_TO_MARKER = {1: "u", 2: "wi", 3: "ghu", 4: "i", 5: "ji", 6: "ghi", 7: "chi", 8: "va", 9: "i", 10: "ri",
                 11: "lu", 14: None, 15: "ku"}
# class 14 (wu- nouns) is left unresolved: on verbs the corpus splits u-ka 46 /
# ghu-ka 29, u-cha 25 / ghu-cha 26, o-re 11 / gho-re 16, and the speakers sorted
# most wu- nouns into u- (wukula: gh-). Only the verified table decides them.
MARKER_TO_NUMS = {"u": (1,), "wi": (2,), "ghu": (3, 14), "gh": (14,), "i": (9, 4), "ji": (5,), "ghi": (6,),
                  "chi": (7,), "va": (8,), "ri": (10,), "lu": (11,), "ku": (15,)}
SG_TO_PL = {1: 2, 3: 4, 5: 6, 7: 8, 9: 10, 11: 10, 14: 6}
PL_TO_SG = {2: (1,), 4: (3,), 6: (5, 14), 8: (7,), 10: (9, 11)}
SG, PL = set(SG_TO_PL) | {15}, set(PL_TO_SG)
SW_SG = {"a", "yu", "u", "li", "ki"}
SW_PL = {"wa", "ya", "vi", "zi"}
TIERS = ("verified", "bible14", "votes", "infinitive", "concord", "round1")
# 2026-11-03, user decision: where sources vary, the Bible (more standardised)
# decides. Bible concord pairs (kitaita_bible_class_decisions.json) are used
# before the both-register concord data, and overrule a verified entry only on
# strong evidence (>= BIBLE_OVERRIDE_VOTES votes, the verified class's pair
# < BIBLE_OVERRIDE_MAX_SHARE of them): chia, sorted chi-, is 9/10 in the Bible.
BIBLE_OVERRIDE_VOTES = 20
BIBLE_OVERRIDE_MAX_SHARE = 0.2
# Not overruled: in the Bible these spellings are other words or the evidence is thin
# (checked by the Swahili words of the same verses, 2026-11-03).
BIBLE_OVERRIDE_SKIP = {
    "maza": "Bible maza aligns with mambo 'matters', not mapacha 'twins'",
    "uzi": "40 votes split wa-/ya- (uzi wa kitani): too thin",
}
BIBLE_PAIR_MIN_VOTES = 10
BIBLE_PAIR_MIN_SHARE = 0.6
BIBLE14_MIN = 5
BIBLE14_SHARE = 0.7
PAIR_OF_MARKER = {"u": "1/2", "wi": "1/2", "ghu": "3/4", "ji": "5/6", "ghi": "5/6", "chi": "7/8", "va": "7/8",
                  "ri": "9/10", "lu": "11/10", "ku": "15", "gh": "14"}


@lru_cache(maxsize=1)
def _data():
    load = lambda f: json.loads((_HERE / f).read_text(encoding="utf-8"))
    votes = {w: v["class"] for w, v in load("kitaita_subject_classes.json")["nouns"].items() if v.get("class")}
    concord = {w: [int(c) for c in v["classes"] if c.isdigit()]
               for w, v in load("kitaita_agreement_data.json")["noun_classes"].items()}
    round1 = load("kitaita_noun_classes_round1.json")["nouns"]
    by_form = {}
    for e in round1.values():
        for f in e["kitaita"]:
            by_form.setdefault(f, e)
    verified = {norm(k): TAM_SM_ALIAS.get(v, v) for k, v in VERIFIED_SUBJECT_MARKERS.items()}
    return verified, votes, concord, round1, by_form


@lru_cache(maxsize=1)
def _bible():
    d = json.loads((_HERE / "kitaita_bible_class_decisions.json").read_text(encoding="utf-8"))
    return d["concord_pairs"], d["class14_sm"]


def _from_bible_pair(e: dict, number: str) -> tuple[str | None, str]:
    """Subject marker from a Bible concord pair entry and the Swahili number."""
    pair = e["pair"]
    if pair == "14":
        return None, "Bible: class 14 (u-/ghu- decided per noun only)"
    if e["member"] != "invariant":
        c = int(e["member"])
    else:
        nums = [int(x) for x in pair.split("/")]
        c = nums[0] if number == "sg" else nums[-1]
    m = NUM_TO_MARKER.get(c)
    return (m, f"Bible concord {pair}, class {c}") if m else (None, f"Bible concord {pair}: class {c} unresolved")


def swahili_number(swahili_sm: str | None, swahili_subject: str | None) -> str | None:
    if swahili_sm in SW_SG:
        return "sg"
    if swahili_sm in SW_PL:
        return "pl"
    if swahili_sm == "i":
        return "pl" if norm(swahili_subject or "").startswith("mi") else "sg"
    return None


def kitaita_forms(swahili_subject: str | None, kitaita_subject: str | None) -> list[str]:
    """Kidaw'ida form(s) of the subject: the pipeline's, then the verified
    translation, then round 1's listed forms."""
    out = []
    if kitaita_subject:
        out.append(norm(kitaita_subject))
    sw = norm(swahili_subject or "")
    if sw in VERIFIED_NOUN_TRANSLATIONS:
        out.append(norm(VERIFIED_NOUN_TRANSLATIONS[sw]))
    e = _data()[3].get(sw)
    if e:
        out.extend(e["kitaita"])
    return list(dict.fromkeys(f for f in out if f))


def _pairs_from_numbers(nums) -> set[tuple]:
    """Possible (singular, plural) class pairs for a set of class numbers."""
    pairs = set()
    for n in nums:
        if n in SG:
            pairs.add((n, SG_TO_PL.get(n)))
        elif n in PL:
            pairs.update((s, n) for s in PL_TO_SG[n])
    return pairs


def number_invariant(form: str) -> bool:
    """One Kidaw'ida form for singular and plural (nguku, samaki; round 1 '9/10')."""
    if form in {norm(n) for n in NUMBER_INVARIANT_NOUNS}:
        return True
    # any round-1 entry with a one-form class string ("9/10") that lists this form
    return any(form in e["kitaita"] and re.fullmatch(r"\d+/\d+", e["class"] or "")
               for e in _data()[3].values())


def _member(sg: int | None, pl: int | None, number: str, invariant: bool, own: int | None):
    """Class to use: the form's own class unless the form is number-invariant."""
    if not invariant and own is not None:
        return own
    return sg if number == "sg" else pl


def resolve(swahili_subject: str | None, kitaita_subject: str | None, swahili_sm: str | None,
            tiers=TIERS) -> tuple[str | None, str]:
    """(subject marker or None, explanation).

    The Kidaw'ida form's own class decides (vindo 'food' is class 8, va-, even
    for Swahili singular chakula); the Swahili verb's number is used only when
    the form is the same in both numbers (nguku: i- singular, ri- plural)."""
    number = swahili_number(swahili_sm, swahili_subject)
    if number is None:
        return None, f"number unknown (Swahili subject marker {swahili_sm!r})"
    verified, votes, concord, round1, by_form = _data()
    forms = kitaita_forms(swahili_subject, kitaita_subject)
    for tier in tiers:
        for f in forms:
            inv = number_invariant(f)
            if tier == "verified" and f in verified:
                # strong Bible evidence against the verified class pair overrules it
                be = _bible()[0].get(f)
                vp = PAIR_OF_MARKER.get(verified[f]) if verified[f] != "i" else None
                if (be and f not in BIBLE_OVERRIDE_SKIP and be["votes"] >= BIBLE_OVERRIDE_VOTES
                        and be["share"] >= BIBLE_PAIR_MIN_SHARE):
                    own = (be.get("pairs", {}).get(vp, 0) / be["votes"]) if vp else (
                        (be.get("pairs", {}).get("9/10", 0) + be.get("pairs", {}).get("3/4", 0)) / be["votes"])
                    if own < BIBLE_OVERRIDE_MAX_SHARE:
                        m2, why2 = _from_bible_pair(be, number)
                        if m2:
                            return m2, f"{why2} (overrules verified {verified[f]}-)"
            if tier == "bible14" and f.startswith("wu") and f in _bible()[1]:
                c14 = _bible()[1][f]
                tot = c14["u"] + c14["ghu"]
                if tot >= BIBLE14_MIN and max(c14.values()) / tot >= BIBLE14_SHARE:
                    m = max(c14, key=c14.get)
                    return m, f"Bible verbs after {f}: u- {c14['u']} / ghu- {c14['ghu']}"
            if tier in ("verified", "votes") and f in (verified if tier == "verified" else votes):
                m = (verified if tier == "verified" else votes)[f]
                if not inv:
                    return m, f"{tier}, the form's own class"
                if m == "i":
                    return ("i", f"{tier}, sg") if number == "sg" else ("ri", f"{tier} (class 9 form), pl -> 10")
                return None, f"{tier} {m}- on a number-invariant form: no rule"
            if tier == "infinitive" and f.startswith("ku") and f.endswith("a") and len(f) >= 5:
                return "ku", "infinitive (class 15)"  # ku- 607 / ku-ka 145 after ku- nouns
            if tier == "concord":
                be = _bible()[0].get(f)
                if be and be["votes"] >= BIBLE_PAIR_MIN_VOTES and be["share"] >= BIBLE_PAIR_MIN_SHARE:
                    return _from_bible_pair(be, number)
            if tier == "concord" and concord.get(f):
                cls = set(concord[f])
                both = any(sg in cls and pl in cls for sg, pl in SG_TO_PL.items())
                if both or inv:
                    pairs = _pairs_from_numbers(cls)
                    chosen = {p[0] if number == "sg" else p[1] for p in pairs}
                else:
                    chosen = cls
                markers = {NUM_TO_MARKER.get(c) for c in chosen}
                if len(markers) == 1 and None not in markers:
                    return markers.pop(), f"concord, classes {sorted(cls)}"
                return None, f"ambiguous (concord: classes {sorted(cls)} for {number})"
            if tier == "round1":
                e = round1.get(norm(swahili_subject or "")) or by_form.get(f)
                if e:
                    if e["sg"] and e["pl"]:
                        c = e["sg"] if number == "sg" else e["pl"]   # separate forms or 9/10: Swahili number
                    else:
                        c = e["sg"] or e["pl"]                       # singular- or plural-only noun
                    m = NUM_TO_MARKER.get(c)
                    return (m, f"round1, class {c}") if m else (None, f"round1: class {c} unresolved")
    return None, "no class data for the subject"
