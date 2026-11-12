"""
build_rules.py

2026-11-11, CLAUDE_rulebook.md §3, sections 1-4 (noun classes, verb template,
TAM, negation). Writes rulebook/rules.json and rulebook/build_rules_results.json
(rejected candidates + config + hashes).

Sources allowed (user decision 2026-11-11, status.md): speaker-verified tables
(kitaita_verified_corrections.py VERIFIED_*, kitaita_speaker_verified.json) and
counts on the clean train split (attest.py). Candidate forms may come from the
existing code (kitaita_verb.SERIES/TAMS/OMS, rulebook_correct._tense_target) or
from exploring the clean train split, but a candidate enters the rule book only
with evidence:
  speaker_verified  -> active (corpus counts attached where available)
  corpus            -> active if evidence_count >= MIN_EVIDENCE and lift >= MIN_LIFT
                       (lift None = no baseline hits); else rejected (results file)
  flag_only         -> known gaps / undecidable cases (tools return status "flag")
Kidaw'ida class labels are the project's subject-marker values (u, wi, ghu, i,
i_nasal, ji, ghi, chi, va, ri, lu, ku, gh), with Bantu numbers as aliases.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "hf_space"))
import attest as A  # noqa: E402
import word_align as W  # noqa: E402
import kitaita_verified_corrections as V  # noqa: E402
from kitaita_verb import SERIES, EXTRA_FORMS  # noqa: E402
from rulebook_flagger import norm  # noqa: E402

OUT = HERE / "rules.json"
RESULTS = HERE / "build_rules_results.json"
MIN_EVIDENCE, MIN_LIFT = 2, 1.5
# lowered bar (user, 2026-11-11): a candidate below the bar but with >= 1 pair and lift >= 1.0
# is kept as status "tentative" instead of being rejected
TENTATIVE_EVIDENCE, TENTATIVE_LIFT = 1, 1.0
VREF = "hf_space/kitaita_verified_corrections.py"

CLASSES = {  # label -> Bantu aliases, SERIES row
    "u": (["1", "14 (some wu- nouns)"], "3sg"), "wi": (["2"], "c2"), "ghu": (["3", "14 (some wu- nouns)"], "c3"),
    "i": (["9", "4"], "c9"), "i_nasal": (["9 (nasal-initial)"], "c9"), "ji": (["5"], "c5"), "ghi": (["6"], "c6"),
    "chi": (["7"], "c7"), "va": (["8"], "c8"), "ri": (["10"], "c10"), "lu": (["11"], "c11"), "ku": (["15"], "c15"),
    "gh": (["14 (wukula)"], "c3"),
}
PERSONS = {"1sg": "ni", "2sg": "u", "1pl": "tu", "2pl": "m"}  # Swahili SM per person
TAMS_RE = "(?:chaa|cha|re|de|ka|ki|we|nda|da|se|sake|seke)?"
SW_OM = {"ni": "1sg", "ku": "2sg", "m": "3sg", "mw": "3sg", "tu": "1pl", "ki": "c7", "vi": "c8", "li": "c5",
         "ji": "c5", "ya": "c6", "zi": "c10", "i": "c9", "u": "c3", "wa": "c2"}
SW_REL = {"ye": "u", "o": "wi", "wo": "ghu", "yo": "i", "lo": "ji", "cho": "chi", "vyo": "va", "zo": "ri", "ko": "ku"}
POSS = {"1sg": ("a", "po", "angu"), "2sg": ("a", "ko", "ako"), "3sg": ("a", "ke", "ake"),
        "1pl": ("e", "du", "etu"), "2pl": ("e", "nyu", "enu"), "3pl": ("a", "wo", "ao")}
ADJ_PREFIX = {"u": "m", "ghu": "m", "wi": "wa", "ji": "i", "ghi": "ma", "chi": "ki", "va": "vi", "i": "m",
              "i_nasal": "m", "ri": "m", "lu": "lu", "ku": "ku", "gh": "wu"}
ADJ_STEMS = "(?:baa|tini|lacha|ghenyi)"
DEM = {"near": V.KITAITA_DEM_PROXIMAL_VERIFIED, "mid": V.KITAITA_DEM_REFERENTIAL_VERIFIED,
       "far": V.KITAITA_DEM_DISTAL_VERIFIED}
KNOWN_NOUN_CONFLICTS = ("chia", "maza", "boi", "ngelo")


def forms(row: str, variant: int) -> list[str]:
    f = SERIES[row][variant]
    return [f, *EXTRA_FORMS.get((row, variant), ())]


def alt(xs) -> str:
    return "(?:" + "|".join(sorted(set(xs), key=len, reverse=True)) + ")"


ALL_BARE = [f for r in SERIES for f in forms(r, 0)]
ALL_A = [f for r in SERIES for f in forms(r, 1)]
ALL_E = [f for r in SERIES for f in forms(r, 2)]
ALL_SM = ALL_BARE + ALL_A + ALL_E
NOUNS = {}
for n, c in V.VERIFIED_SUBJECT_MARKERS.items():
    NOUNS.setdefault(c, set()).add(norm(n))


def merge_a(sm_a: str, stem: str) -> str:
    """SM+a + vowel-initial stem: a+o -> o (wose, ghose), a+i -> e (wengi, mengi)."""
    if stem[0] in "aeiou" and sm_a.endswith("a"):
        return sm_a[:-1] + {"o": "o", "i": "e"}.get(stem[0], stem[0]) + stem[1:]
    return sm_a + stem


def same(lab: str, form_of) -> frozenset:
    """Other classes whose form for this slot is identical (left out of the lift baseline)."""
    mine = form_of(lab)
    return frozenset(l for l in CLASSES if l != lab and form_of(l) == mine)


def concord_words() -> set[str]:
    out = set()
    for row in SERIES:
        a, e = forms(row, 1)[0], forms(row, 2)[0]
        out |= {a, e, merge_a(a, "ose")} | {(a if v == "a" else e) + s for v, s, _ in POSS.values()}
    for t in DEM.values():
        out |= set(t.values())
    return out


STOP = concord_words()
rules: list[dict] = []
rejected: list[dict] = []


def add(rid, section, slot, cond, then, ev=None, source="corpus_attested", verified_examples=None,
        phonology=(), conflict=False, readings=None, notes="", status=None):
    ev = ev or {}
    r = {"id": rid, "section": section, "slot": slot, "if": cond, "then": then, "phonology": list(phonology),
         "examples": ev.get("examples", []), "counterexamples": [], "evidence_count": ev.get("evidence_count", 0),
         "evidence": {k: v for k, v in ev.items() if k not in ("examples", "evidence_count")},
         "source": source, "conflict": conflict}
    if verified_examples:
        r["verified_examples"] = verified_examples
    if readings:
        r["readings"] = readings
    if notes:
        r["notes"] = notes
    if status:
        r["status"] = status
    elif source == "speaker_verified":
        r["status"] = "active"
    else:
        lift = ev.get("lift")
        ok = r["evidence_count"] >= MIN_EVIDENCE and (lift is None or lift >= MIN_LIFT)
        weak = r["evidence_count"] >= TENTATIVE_EVIDENCE and (lift is None or lift >= TENTATIVE_LIFT)
        if not ok and not weak:
            rejected.append({**r, "reason": f"evidence {r['evidence_count']}, lift {lift}"})
            return None
        r["status"] = "active" if ok else "tentative"
    rules.append(r)
    return r


def combine(evs: dict[str, dict]) -> dict:
    """Several forms in one rule: evidence summed, best examples kept, per-form counts."""
    ex = [e for ev in evs.values() for e in ev["examples"][:2]][:4]
    return {"evidence_count": sum(e["evidence_count"] for e in evs.values()), "examples": ex,
            "method": next(iter(evs.values()))["method"],
            "by_form": {f: {"n": e["evidence_count"], "lift": e.get("lift"), "domains": e["domains"],
                            "attested": attested(e)} for f, e in evs.items()}}


def attested(ev: dict) -> bool:
    return ev["evidence_count"] >= MIN_EVIDENCE and (ev.get("lift") is None or ev["lift"] >= MIN_LIFT)


# ------------------------------------------------------------- section 1
def section1():
    for lab, (bantu, row) in CLASSES.items():
        nouns = sorted(NOUNS.get(lab, ()))
        vex = [{"ref": f"{VREF}:VERIFIED_SUBJECT_MARKERS", "nouns": nouns[:8], "n_nouns": len(nouns)}]
        add(f"CLASS.{lab}", 1, "CLASS", {"kitaita_class": lab}, {"bantu": bantu, "series_row": row,
            "verified_nouns": len(nouns)}, source="speaker_verified", verified_examples=vex,
            conflict=lab in ("u", "ghu", "gh"),
            notes="class 14 wu- nouns are split between u and ghu (and gh: wukula), per noun, by the speaker; "
                  "round-1 concord notes gave ghu- for all of class 14" if lab in ("u", "ghu", "gh") else "")
        # SM, three series. Bare SM is the class label itself (speaker-verified per noun); +a/+e corpus.
        for vi, var in enumerate(("bare", "a", "e")):
            fs = forms(row, vi)
            if lab == "i_nasal" or lab == "gh":
                fs = forms(row, vi)
            sf = same(lab, lambda l, vi=vi: forms(CLASSES[l][1], vi)[0])
            evs = {f: A.after_nouns(NOUNS, lab, f"{f}{TAMS_RE}[a-z]{{3,}}", same_form=sf) for f in fs}
            ev = combine(evs)
            src = "corpus_attested"
            conflict, readings, notes = False, None, ""
            if var == "a" and lab in V.PRESENT_SM_VOWEL_CONFIRMED:
                notes = f"present SM+a confirmed against speaker answers (PRESENT_SM_VOWEL_CONFIRMED: " \
                        f"{V.PRESENT_SM_VOWEL_CONFIRMED[lab]})"
            if var == "a" and lab in V.PRESENT_SM_VOWEL_CONFIRMED and lab != "gh" \
                    and V.PRESENT_SM_VOWEL_CONFIRMED[lab] != fs[0]:
                conflict = True
                readings = [{"form": V.PRESENT_SM_VOWEL_CONFIRMED[lab], "source": "speaker (PRESENT_SM_VOWEL_CONFIRMED)",
                             "default": True}, {"form": fs[0], "source": "corpus", "default": False}]
                notes = f"speaker-confirmed present {V.PRESENT_SM_VOWEL_CONFIRMED[lab]}- vs corpus {fs[0]}- (validation 2026-11-12)"
            if lab == "gh" and var == "a":
                conflict = True
                readings = [{"form": "gha", "source": "corpus (PRESENT_SM_VOWEL_CONFIRMED)", "default": True},
                            {"form": "ghwa", "source": "speaker answer", "default": False}]
                notes = "speaker used ghwa-; corpus had 0 ghwa+root vs 10 gha+root (2026-10-07). Kept both."
            add(f"SM.{lab}.{var}", 1, "SM", {"subj_class": lab, "series": var},
                {"forms": [x["form"] for x in readings] if readings else fs}, ev, src, conflict=conflict,
                readings=readings, notes=notes)
        # possessives: SM+a/+e + po/ko/ke/du/nyu/wo
        a0, e0 = forms(row, 1)[0], forms(row, 2)[0]
        table = {p: (a0 if v == "a" else e0) + s for p, (v, s, _) in POSS.items()}
        sf = same(lab, lambda l: forms(CLASSES[l][1], 1)[0])
        evs = {table[p]: A.after_nouns(NOUNS, lab, table[p], sw_re=f"[a-z]*{cue}", same_form=sf)
               for p, (_, _, cue) in POSS.items()}
        add(f"POSS.{lab}", 1, "CONCORD", {"noun_class": lab, "kind": "possessive"},
            {"template": "SM+a + po|ko|ke|wo ; SM+e + du|nyu", "forms": table}, combine(evs))
        # associative 'of' = SM+a
        cue = "(?:wa|ya|la|cha|vya|za|kwa|cha|pa|mwa)"
        ev = A.after_nouns(NOUNS, lab, a0, sw_re=cue, same_form=sf)
        assoc_v = V.KITAITA_ASSOCIATIVE_VERIFIED.get(lab)
        add(f"ASSOC.{lab}", 1, "CONCORD", {"noun_class": lab, "kind": "associative"}, {"form": assoc_v or a0},
            ev, "speaker_verified" if assoc_v else "corpus_attested",
            verified_examples=[{"ref": f"{VREF}:KITAITA_ASSOCIATIVE_VERIFIED"}] if assoc_v else None)
        # -ose 'all'
        ose = merge_a(a0, "ose")
        ev = A.after_nouns(NOUNS, lab, ose, same_form=sf)
        add(f"QUANT.ose.{lab}", 1, "CONCORD", {"noun_class": lab, "kind": "all"}, {"form": ose}, ev,
            phonology=["a + o -> o"])
        # demonstratives (speaker-verified, 3 deixis), corpus counts before and after the noun
        if lab in DEM["near"]:
            dem = {d: t[lab] for d, t in DEM.items()}
            evs = {}
            for d, f in dem.items():
                dsf = same(lab, lambda l, t=DEM[d]: t.get(l))
                e1 = A.after_nouns(NOUNS, lab, f, same_form=dsf)
                e2 = A.after_nouns(NOUNS, lab, f, before=True, same_form=dsf)
                evs[f] = {**e1, "evidence_count": e1["evidence_count"] + e2["evidence_count"],
                          "examples": e1["examples"] + e2["examples"]}
            add(f"DEM.{lab}", 1, "CONCORD", {"noun_class": lab, "kind": "demonstrative"},
                {"near": dem["near"], "mid": dem["mid"], "far": dem["far"]}, combine(evs), "speaker_verified",
                verified_examples=[{"ref": f"{VREF}:KITAITA_DEM_{{PROXIMAL,REFERENTIAL,DISTAL}}_VERIFIED"}])
        # adjective prefix (nominal-type stems)
        pre = ADJ_PREFIX[lab]
        ev = A.after_nouns(NOUNS, lab, f"{pre}{ADJ_STEMS}", same_form=same(lab, ADJ_PREFIX.get))
        vadj = [{"sw": f"{s} {n}", "dav_adj": f} for (s, n), f in V.VERIFIED_ADJECTIVES.items()]
        add(f"ADJ.{lab}", 1, "CONCORD", {"noun_class": lab, "kind": "adjective", "stem_type": "nominal"},
            {"prefix": pre, "e.g.": pre + "baa"}, ev,
            notes="nominal stems only (-baa, -tini, -lacha, -ghenyi); stative adjectives take the SM "
                  "(baya mtu -> u-zamie, speaker-verified)")
    # verified adjective forms as one rule
    add("ADJ.verified_forms", 1, "CONCORD", {"kind": "adjective"},
        {"forms": {f"{s} {n}": f for (s, n), f in V.VERIFIED_ADJECTIVES.items()}}, source="speaker_verified",
        verified_examples=[{"ref": f"{VREF}:VERIFIED_ADJECTIVES"}])
    # persons
    for p, sw_sm in PERSONS.items():
        for vi, var in enumerate(("bare", "a", "e")):
            fs = forms(p, vi)
            evs = {f: A.cooc(f"sm:{sw_sm}", f"{f}{TAMS_RE}[a-z]{{3,}}", exclude=STOP) for f in fs}
            add(f"SM.{p}.{var}", 1, "SM", {"subj_person": p, "series": var}, {"forms": fs}, combine(evs))
    vp = {f"{p}/{t}": m for (p, t), m in V.PERSON_SUBJECT_MARKERS_BY_TAM.items()}
    ver = {k: v for k, v in vp.items() if k not in ("1pl/past", "2sg/ka")}
    add("SM.person.verified", 1, "SM", {"subj_person": "1sg|2sg|1pl|2pl"}, {"forms_by_person_tense": ver},
        source="speaker_verified", verified_examples=[{"ref": f"{VREF}:PERSON_SUBJECT_MARKERS_BY_TAM (s73-s84)"}],
        conflict=True, readings=[{"1pl/past": "de", "source": "corpus n=1"},
                                 {"1pl/past": "da", "source": "corpus n=2"}],
        notes="1pl past de/da unresolved, not asked; 2sg narrative u- not asked")
    # object markers
    oms = {"1sg": "ni", "2sg": "ku", "3sg": "m", "1pl": "di", "c2": "wi", "c3": "ghu", "c5": "ji", "c6": "ghi",
           "c7": "chi", "c8": "vi", "c9": "i", "c10": "ri", "c11": "lu"}
    pre = f"(?:nde|si)?{alt(ALL_SM)}{TAMS_RE}"
    for sw_om, row in SW_OM.items():
        om = oms[row]
        ev = A.cooc(f"om:{sw_om}", f"{pre}{om}[a-z]{{3,}}", exclude=STOP)
        add(f"OM.{row}.from_sw_{sw_om}", 1, "OM", {"sw_om": sw_om, "obj": row},
            {"form": om, "position": "after TAM, before root"}, ev)
    # relative: bare SM + R-agha (present) / -ie-ghe (past)
    for marker, lab in SW_REL.items():
        bare = forms(CLASSES[lab][1], 0)[0]
        evs = {f"{bare}-...-agha": A.cooc(f"rel:{marker}", f"{bare}[a-z]{{2,}}agha", exclude=STOP),
               f"{bare}-...-ie-ghe": A.cooc(f"rel:{marker}", f"{bare}[a-z]{{2,}}(?:ie|ire|ere)ghe", exclude=STOP)}
        add(f"REL.{lab}.from_sw_{marker}", 1, "REL", {"sw_rel": marker, "class": lab},
            {"template": "bare SM, no relative morpheme; present ROOT-agha, past ROOT-ie-ghe", "bare_sm": bare},
            combine(evs))
    # plural pairing from speaker-verified singular/plural pairs
    swaps = [("u", "wi", r"^mw?u?", "wa"), ("ghu", "i", r"^mu?", "mi"), ("ji", "ghi", r"^i?", "ma"),
             ("chi", "va", r"^ki", "vi"), ("chi", "va", r"^ch", "vy"), ("lu", "ri", r"^lu", "n"),
             ("i", "i", r"^", "")]
    for sg, pl, pat, rep in swaps:
        pairs_ = []
        for n in sorted(NOUNS.get(sg, ())):
            cand = re.sub(pat, rep, n, count=1)
            if cand != n and cand in NOUNS.get(pl, ()):
                pairs_.append(f"{n}/{cand}")
        if pairs_:
            add(f"PLURAL.{sg}_{pl}.{rep or 'same'}", 1, "CLASS", {"sg_class": sg, "prefix": pat},
                {"pl_class": pl, "rewrite": f"{pat} -> {rep}"}, {"evidence_count": len(pairs_),
                 "examples": [], "method": "verified singular/plural nouns", "pairs": pairs_[:10]},
                source="speaker_verified")
    add("PLURAL.number_invariant", 1, "CLASS", {"nouns": sorted(V.NUMBER_INVARIANT_NOUNS)},
        {"sg_sm": V.NUMBER_INVARIANT_SM_SINGULAR, "pl_sm": V.NUMBER_INVARIANT_SM_PLURAL,
         "how": "same noun form; the Swahili verb's number picks i- (sg) or ri- (pl)"},
        source="speaker_verified", verified_examples=[{"ref": f"{VREF}:NUMBER_INVARIANT_NOUNS"}])
    # animacy: Kidaw'ida animals keep i/ri concord; Swahili animacy override is flagged
    animals = [n for n in ("nguku", "nyonyi", "choka", "shimba", "ng'ombe", "samaki") if norm(n) in
               {x for s in NOUNS.values() for x in s}]
    add("CLASS.animate_override", 1, "CLASS", {"sw_noun_animate": True, "sw_shape_class": "9/10/7"},
        {"status": "flag", "kitaita": "animals keep their own class concord (i / i_nasal / ri), not class 1",
         "flag_nouns_sw": ["kipepeo", "paka", "nyoka", "chama"]},
        {"evidence_count": len(animals), "examples": [], "method": "verified animal nouns", "nouns": animals},
        source="speaker_verified", status="flag_only",
        notes="CLAUDE.md §8: Swahili gives class-1 concord to animates; a Swahili SM a-/wa- with an animal "
              "subject must not be copied. Tools return status flag for these nouns.")
    # known noun-level conflicts: verified class vs clean-train concord votes
    vote_forms = {}
    for lab, (_, row) in CLASSES.items():
        a0, e0 = forms(row, 1)[0], forms(row, 2)[0]
        for f in [a0, merge_a(a0, "ose")] + [(a0 if v == "a" else e0) + s for v, s, _ in POSS.values()]:
            vote_forms.setdefault(f, set()).add(lab)
    for n in KNOWN_NOUN_CONFLICTS:
        votes = {}
        ids = []
        for i, r in enumerate(A.pairs()):
            t = r["dv"]
            for j, w in enumerate(t[:-1]):
                if w == n and t[j + 1] in vote_forms:
                    for c in vote_forms[t[j + 1]]:
                        votes[c] = votes.get(c, 0) + 1 / len(vote_forms[t[j + 1]])
                    ids.append(i)
        votes = {c: round(v, 1) for c, v in sorted(votes.items(), key=lambda x: -x[1]) if v >= 1}
        add(f"NOUN.conflict.{n}", 1, "CLASS", {"kitaita_noun": n},
            {"default_class": V.VERIFIED_SUBJECT_MARKERS.get(n)},
            {"evidence_count": len(ids), "examples": A._examples(ids), "method": "concord word after noun",
             "corpus_votes": votes}, source="speaker_verified", conflict=True,
            readings=[{"class": V.VERIFIED_SUBJECT_MARKERS.get(n), "source": "speaker_verified", "default": True},
                      {"class_votes": votes, "source": "corpus_attested", "default": False}])
    wu_modifier_rule(vote_forms)


WU_NOUN = re.compile(r"^w'?u[^aeiou']")


def wu_modifier_rule(vote_forms):
    """rulebook-v2 (2026-11-12): wu- (class 14) nouns take ghu-series modifiers (ghodu, ghwake, ghose ...)
    even when their subject marker is u- (speaker table: wuhuru, wushu, wurigha ... SM u). Evidence: the
    concord word right after any wu- word on clean train, ghu-series vs u-series forms."""
    ghu = {f for f, cs in vote_forms.items() if "ghu" in cs}
    u = {f for f, cs in vote_forms.items() if "u" in cs and "ghu" not in cs}
    ids_g, ids_u, n_g, n_u = [], [], 0, 0
    for i, r in enumerate(A.pairs()):
        t = r["dv"]
        for j, w in enumerate(t[:-1]):
            if WU_NOUN.match(w) and len(w) >= 5:
                if t[j + 1] in ghu:
                    n_g += 1
                    ids_g.append(i)
                elif t[j + 1] in u:
                    n_u += 1
                    ids_u.append(i)
    sm_u = sorted(n for n, c in V.VERIFIED_SUBJECT_MARKERS.items() if WU_NOUN.match(norm(n)) and c == "u")
    r = add("CLASS.wu_prefix.modifier_ghu", 1, "CLASS",
            {"kitaita_noun_prefix": "wu- / w'u-", "slots": ["ADJ", "DEM_near", "DEM_mid", "DEM_far", "POSS", "ASSOC",
                                                             "OSE", "SELF", "num"]},
            {"modifier_class": "ghu", "subject_class": "unchanged (speaker table or concord votes)"},
            {"evidence_count": n_g, "examples": A._examples(ids_g), "method": "concord word after wu- noun", "pattern": WU_NOUN.pattern,
             "ghu_series_tokens": n_g, "u_series_tokens": n_u, "share_ghu": round(n_g / max(1, n_g + n_u), 4),
             "speaker_sm_u_nouns": sm_u},
            notes="rulebook-v2. Subject marker and modifier concord differ for these nouns (e.g. speaker: "
                  "'wuhuru ghodu' with SM u-). Used only for modifiers; subject markers are not changed.")
    if r and ids_u:
        r["counterexamples"] = A._examples(ids_u)


# ------------------------------------------------------------- section 2
def section2():
    rx = f"nde{alt(ALL_E)}re(?:ni|ku|m|di|wi|ghi|ri|chi|vi|ji|ghu|lu|i)[a-z]{{2,}}(?:ie|ire|ere|e)ghe"
    ev = A.cooc(lambda fs: True, rx)
    add("VERB.template", 2, "TEMPLATE", {},
        {"order": ["NEG (nde-|si-)", "SM", "TAM", "OM", "ROOT", "EXT", "FV (-a|-e|-ie|-agha)", "post-final -ghe"],
         "note": "no separate relative morpheme: relatives use the bare SM series"}, ev,
        notes="evidence: words showing NEG-SM-TAM-OM-ROOT-FV-ghe in one form, e.g. nde-we-re-ni-sikir-e-ghe")


# ------------------------------------------------------------- section 3 + 4
def tam(rid, section, slot, feature, dav_re, then, cond, **kw):
    ev = A.cooc(feature, dav_re, exclude=STOP)
    r = add(rid, section, slot, cond, then, ev, **kw)
    wl = W.word_level(feature, dav_re)
    target = r if r else (rejected[-1] if rejected and rejected[-1]["id"] == rid else None)
    if target is not None:
        target["word_level"] = wl
        target["counterexamples"] = wl["counterexamples"]
        # sentence-level pass but almost never the aligned verb's form -> tentative
        if r and r["status"] == "active" and wl["n_aligned"] >= 20 and wl["k_match"] / wl["n_aligned"] < 0.02:
            r["status"] = "tentative"
            r["notes"] = (r.get("notes", "") + f"; word-level {wl['k_match']}/{wl['n_aligned']} aligned verbs").strip("; ")
    return r


def section3():
    B, Aa, E = alt(ALL_BARE), alt(ALL_A), alt(ALL_E)
    R = "[a-z]{2,}"
    tam("TAM.perf.me", 3, "TAM", "tam:me", f"{Aa}{R}a", {"template": "SM+a ROOT-a"}, {"sw_tam": "me"})
    tam("TAM.perf.me_ie", 3, "TAM", "tam:me", f"{Aa}{R}(?:ie|ire|ere)", {"template": "SM+a ROOT-ie"},
        {"sw_tam": "me"}, notes="second reading of -me-: perfective final -ie (wa-mer-ie)")
    tam("TAM.perf.sha", 3, "TAM", "tam:sha", f"{Aa}{R}a", {"template": "SM+a ROOT-a"}, {"sw_tam": "sha"})
    tam("TAM.past.li", 3, "TAM", "tam:li", f"{E}re{R}(?:ie|ire|ere|e)(?:ghe)?",
        {"template": "SM+e-re-ROOT-ie(-ghe)", "kuwa": "SM+e-re-kogh-e (ne-re-kogh-e, speaker-verified)"},
        {"sw_tam": "li", "rel": None},
        verified_examples=[{"ref": f"{VREF}:PERSON_SUBJECT_MARKERS_BY_TAM 1sg/past (s84)", "dav": "nerekoghe"}])
    tam("TAM.past.li_de", 3, "TAM", "tam:li", f"{E}de{R}(?:ie|ire|ere|e)(?:ghe)?",
        {"template": "SM+e-de-ROOT-ie(-ghe)"}, {"sw_tam": "li", "rel": None},
        notes=f"VERIFIED_PAST_TAM_BY_SM: {V.VERIFIED_PAST_TAM_BY_SM} (ji class uses -de-)")
    add("TAM.past.verified_by_class", 3, "TAM", {"sw_tam": "li", "subj_class": sorted(V.VERIFIED_PAST_TAM_BY_SM)},
        {"tam_by_class": V.VERIFIED_PAST_TAM_BY_SM, "template": "SM+e-<tam>-ROOT-ie(-ghe)"}, source="speaker_verified",
        verified_examples=[{"ref": f"{VREF}:VERIFIED_PAST_TAM_BY_SM"}],
        notes="class-specific past marker (ji: -de-) overrides -re- (validation 2026-11-12)")
    tam("TAM.past.li_e_only", 3, "TAM", "tam:li", f"{E}(?!re|de){R}a",
        {"template": "SM+e ROOT-a (rulebook_correct _tense_target '-li- past -> SM+e')"}, {"sw_tam": "li", "rel": None})
    tam("TAM.fut.ta", 3, "TAM", "tam:ta", f"{B}cha{R}a", {"template": "SM-cha-ROOT-a"}, {"sw_tam": "ta"},
        verified_examples=[{"ref": f"{VREF}:PERSON_SUBJECT_MARKERS_BY_TAM future (s73-s79)"}])
    r0 = tam("TAM.pres.na_zero", 3, "TAM", "tam:na", f"{Aa}(?!we){R}a", {"template": "SM+a ROOT-a"},
             {"sw_tam": "na"})
    r1 = tam("TAM.pres.na_we", 3, "TAM", "tam:na", f"{Aa}we{R}a", {"template": "SM+a-we-ROOT-a"},
             {"sw_tam": "na"})
    for r, default in ((r1, True), (r0, False)):
        if r:
            r.update({"conflict": True, "source": "speaker_verified",
                      "readings": [{"form": "SM+a-we-ROOT-a", "default": True, "why": "speaker's stated preference"},
                                   {"form": "SM+a ROOT-a", "default": False, "why": "also grammatical; corpus-dominant"}],
                      "verified_examples": [{"ref": f"{VREF}:PRESENT_TAM_DEFAULT (2026-10-07 cont. 15)"}]})
    # no -na- for stative-like 'want': wanataka -> w'ekunda
    ids =[i for i, r in enumerate(A.pairs()) if any(re.fullmatch(r"[a-z]{1,3}nataka", w) for w in r["sw"])
           and any(re.fullmatch(f"{E}kunda", w) for w in r["dv"])]
    base = sum(any(re.fullmatch(r"[a-z]{1,3}nataka", w) for w in r["sw"]) for r in A.pairs())
    add("TAM.pres.na_stative_taka", 3, "TAM", {"sw_tam": "na", "sw_root": "tak"},
        {"template": "SM+e kunda (no -na- equivalent: past-looking form for the present)"},
        {"evidence_count": len(ids), "examples": A._examples(ids), "method": "pair_cooccurrence",
         "n_pairs_with_sw_feature": base, "lift": None},
        notes="lexical: stative-like 'want' (wanataka -> w'ekunda). Other stative verbs: see flag TAM.stative_present")
    tam("TAM.rel.past", 3, "TAM", "reltam:li", f"{E}re{R}(?:ie|ire|ere|e)ghe",
        {"template": "SM+e-re-ROOT-ie-ghe"}, {"sw_tam": "li", "rel": "any"})
    add("TAM.rel.past_cleft", 3, "TAM", {"context": "after ndio/niwo (cleft), Swahili -li-"},
        {"template": "bare SM ROOT-ie-ghe", "e.g.": "wi-lu-ie-ghe"},
        A.cooc("tam:li", f"{B}(?!re|de|ka|ki|cha){R}(?:ie|ire|ere)ghe", exclude=STOP), source="speaker_verified",
        verified_examples=[{"sw": "Hawa wazee ndio walipigania uhuru wetu",
                            "dav": "Awa waghosi niwo wiluieghe wuhuru ghodu", "ref": "kitaita_speaker_verified.json"}])
    tam("TAM.rel.pres", 3, "TAM", "reltam:na", f"{B}{R}agha", {"template": "bare SM ROOT-agha"},
        {"sw_tam": "na", "rel": "any"})
    tam("TAM.rel.lipo", 3, "TAM", "rel:lipo", f"{E}nda{R}", {"template": "SM+e-nda-ROOT-a"},
        {"sw_tam": "li", "rel": "po"})
    tam("TAM.hab.hu", 3, "TAM", "hab", f"{Aa}da{R}", {"template": "SM+a-da-ROOT-a(gha)"}, {"sw": "hu-ROOT-a"})
    tam("TAM.narr.ka", 3, "TAM", "tam:ka", f"{B}ka{R}", {"template": "SM-ka-ROOT-a"}, {"sw_tam": "ka"})
    tam("TAM.cond.ki", 3, "TAM", "tam:ki", f"{B}ki{R}", {"template": "SM-ki-ROOT-a"}, {"sw_tam": "ki"})
    tam("TAM.prohib.se", 3, "TAM", "prohib", f"{B}se{R}e", {"template": "SM-se-ROOT-e"}, {"sw": "SM-si-ROOT-e"})
    tam("TAM.prohib.sake", 3, "TAM", "prohib", f"{B}(?:sake|seke){R}", {"template": "SM-sake-ROOT-a"},
        {"sw": "SM-si-ROOT-e"}, notes="second prohibitive reading (m-sake-, ku-sake-)")
    for fid, cond, why in (("TAM.cond.nge", {"sw_tam": "nge|ngali"}, "no consistent Kidaw'ida form in clean train"),
                           ("TAM.progressive.wia", {"dav_tam": "wia"}, "-wia- progressive: in kitaita_verb.TAMS, "
                                                                     "no Swahili condition known"),
                           ("TAM.stative_present", {"sw_tam": "na", "stative": True},
                            "stative present (ghe-singa) missing from the notes")):
        add(fid, 3, "TAM", cond, {"status": "flag"}, source="grammar_notes", status="flag_only",
            notes=why + " (known gap: docs/status.md 2026-11-03 cont. 5; section 8 of the brief)")


def section_round1():
    """Candidates from agent round 1 rule notes (2026-11-11), attested like every other rule."""
    B, Aa, E = alt(ALL_BARE), alt(ALL_A), alt(ALL_E)
    R = "[a-z]{2,}"
    tam("TAM.past.passive", 3, "TAM", "past:passive", f"{E}re{R}(?:il|el|ir|er)o(?:ghe)?",
        {"template": "SM+e-re-ROOT-ilo/elo/iro/ero-ghe"}, {"sw_tam": "li", "ext": "passive"},
        notes="agent round 1 (4 agents): passive + final fuse to -o")
    tam("TAM.perf.passive_o", 3, "TAM", lambda fs: "tam:me" in fs and "ext:passive" in fs,
        f"{Aa}{R}(?:il|el|ir|er|l|w)o", {"template": "SM+a ROOT-ilo/-elo/-o"}, {"sw_tam": "me", "ext": "passive"})
    tam("TAM.narr.nga_1sg", 3, "TAM", lambda fs: "tam:ka" in fs and "sm:ni" in fs, f"nga{R}",
        {"template": "nga-ROOT-a (1sg narrative)"}, {"sw_tam": "ka", "subj": "1sg"}, notes="agent round 1")
    tam("TAM.cond.nga_1sg", 3, "TAM", lambda fs: "tam:ki" in fs and "sm:ni" in fs, f"nga{R}",
        {"template": "nga-ROOT-a (1sg conditional)"}, {"sw_tam": "ki", "subj": "1sg"}, notes="agent round 1")
    tam("TAM.pres.awuya", 3, "TAM", "tam:na", f"{Aa}a?wuya{R}", {"template": "SM+a-awuya-ROOT-a (continuative)"},
        {"sw_tam": "na"}, notes="agent round 1: -awuya- 'keep on'; Swahili condition not settled")
    tam("TAM.pastprog", 3, "TAM", "pastprog", f"{B}ki{R}",
        {"template": "SM+e-re-kogh-e SM-ki-ROOT-a (two words)"}, {"sw": "-li-kuwa + -ki- verb"},
        notes="agent round 1: orekoghe ukisoghoda; evidence counts the -ki- word only")
    tam("NEG.fut_3sg", 4, "NEG", lambda fs: "neg:ta" in fs and "negsm:a" in fs, f"ndechaa{R}",
        {"template": "nde-chaa-ROOT-a (3sg: no SM)"}, {"sw": "ha-ta-ROOT-a"}, notes="agent round 1")
    tam("NEG.past_3sg", 4, "NEG", lambda fs: "neg:ku" in fs and "negsm:a" in fs, f"ndere{R}",
        {"template": "nde-re-ROOT-ie-ghe (3sg: nde+o+re -> ndere)"}, {"sw": "ha-ku-ROOT-a"}, notes="agent round 1")
    tam("NEG.pres_3sg", 4, "NEG", lambda fs: "neg:pres" in fs and "negsm:a" in fs, f"nde(?!we|wi|ri|ji|ghi|chi|vi|ku|di|m){R}agha",
        {"template": "nde-ROOT-agha (3sg: no SM)"}, {"sw": "ha-ROOT-i"}, notes="agent round 1")
    tam("IMP.pl_enyi", 6, "FV", "imp:pl", f"{R}enyi", {"template": "ROOT-enyi (plural imperative)"},
        {"sw": "ROOT-eni"}, notes="agent round 1")
    tam("LOC.nyi", 7, "LOC", "loc:ni", f"{R}nyi", {"template": "NOUN-(e)nyi (locative)", "phonology": "a+i -> e"},
        {"sw": "NOUN-ni"}, notes="agent round 1: hekalunyi, isangenyi")
    tam("COPULA.ko", 7, "COPULA", "copula:ko", f"{alt(ALL_BARE + ALL_E)}ko",
        {"template": "SM(bare/+e)-ko 'be at / there is'; SM-ko na 'have'"}, {"sw": "SM-ko / kuna / pana"},
        notes="agent round 1: weko, neko, gheko, wiko")
    # agent round 2
    tam("TAM.subj.nde", 3, "TAM", "subj", f"nde{B}?{R}(?:e|o)",
        {"template": "nde-SM-ROOT-e / -o (subjunctive 'may/let'; passive -o) — NOT negation"},
        {"sw": "SM-ROOT-e (subjunctive)"}, notes="agent round 2: nde- also hortative; tools must not read it as NEG")
    tam("TAM.subj.plain", 3, "TAM", "subj", f"{B}(?!nde){R}e", {"template": "SM-ROOT-e (subjunctive)"},
        {"sw": "SM-ROOT-e (subjunctive)"}, notes="agent round 2")
    tam("OM.reflexive.ku", 1, "OM", "om:ji", f"(?:nde|si)?{alt(ALL_SM)}{TAMS_RE}ku[a-z]{{3,}}",
        {"form": "ku", "meaning": "reflexive (Swahili -ji-)"}, {"sw_om": "ji"},
        notes="agent rounds 1-2: Swahili -ji- is reflexive -> Kidaw'ida ku- (OM.c5.from_sw_ji treats it as class 5)")
    tam("OM.3sg.mmb", 1, "OM", lambda fs: "om:m" in fs or "om:mw" in fs,
        f"(?:nde|si)?{alt(ALL_SM)}{TAMS_RE}mmb[a-z]{{2,}}", {"form": "mmb", "phonology": "m + w/b -> mmb"},
        {"sw_om": "m", "dav_root": "w- / b-initial"}, notes="agent round 2: wichammbona, ummboniagha")
    tam("COPULA.have", 7, "COPULA", "have", f"{alt(ALL_BARE + ALL_E)}(?:e?ka|oka|ko)",
        {"template": "SM(+e)-(e)ka / -ko + na 'have'"}, {"sw": "SM-na 'have'"},
        notes="agent rounds 1-2: neka na, oka na, deka na; check that 'na' follows")
    # agent round 4: tens = mirongo + unit numeral; 'self' = SM+a-eni
    ids = [i for i, r in enumerate(A.pairs()) if any(w in r["sw"] for w in ("ishirini", "thelathini", "arobaini",
           "hamsini", "sitini", "sabini", "themanini", "tisini")) and "mirongo" in r["dv"]]
    base = sum(any(w in r["sw"] for w in ("ishirini", "thelathini", "arobaini", "hamsini", "sitini", "sabini",
               "themanini", "tisini")) for r in A.pairs())
    add("NUM.tens", 7, "CONCORD", {"sw": "ishirini ... tisini"}, {"template": "mirongo + unit numeral (mirongo idadu = 30)"},
        {"evidence_count": len(ids), "examples": A._examples(ids), "method": "pair_cooccurrence",
         "n_pairs_with_sw_feature": base, "lift": None}, notes="agent round 4")
    for lab, (_, row) in CLASSES.items():
        f = merge_a(forms(row, 1)[0], "eni")
        add(f"CONCORD.self.{lab}", 1, "CONCORD", {"noun_class": lab, "kind": "'self / same' (Swahili -enyewe)"},
            {"form": f}, A.after_nouns(NOUNS, lab, f), notes="agent rounds 3-4: weni, gheni, cheni")
    # agent round 3
    tam("TAM.fut.chaa", 3, "TAM", "tam:ta", f"{B}chaa{R}", {"template": "SM-chaa-ROOT-a (affirmative future)"},
        {"sw_tam": "ta"}, notes="agent round 3: -chaa- is not only negative; negation is nde-/si-")
    tam("OM.2pl.m", 1, "OM", "om:wa", f"(?:nde|si)?{alt(ALL_SM)}{TAMS_RE}mu?(?!mb)[a-z]{{3,}}",
        {"form": "m- / mu-", "meaning": "2pl object (Swahili -wa- ... -eni)"}, {"sw_om": "wa", "obj": "2pl"},
        notes="agent round 3; same surface as 3sg m-, so evidence also counts 3sg readings")
    tam("NEG.have", 4, "NEG", "neg:have", f"nde{alt(ALL_BARE + ALL_E)}?n[ae]",
        {"template": "nde-SM-na / nde-SM-ne 'not have'"}, {"sw": "SM-na negative (sina, hana, hakuna)"},
        notes="agent rounds 2-3: ndeuna, ndeghuna, ndew'ine, ndekune")
    tam("NEG.have_wadie", 4, "NEG", "neg:have", f"nde{alt(ALL_BARE + ALL_E)}?wadie",
        {"template": "nde-SM-wadie 'not have' (main form)"}, {"sw": "SM-na negative (sina, hana, hakuna)"},
        notes="round 3 check: ndekuwadie 180, ndewiwadie 22, ndediwadie 17 in neg-have pairs")
    tam("NEG.have_dependent", 4, "NEG", "neg:have", f"{alt(ALL_BARE)}sen[ae]",
        {"template": "SM-se-na (dependent clause)"}, {"sw": "SM-na negative"}, notes="agent round 3: usena, isena")
    tam("NEG.lest", 4, "NEG", "lest", f"{B}seche{R}", {"template": "SM-se-che-ROOT-a 'lest / so that not'"},
        {"sw": "-sije-"}, notes="agent round 3: disechetota")
    # concord variants named by agents
    for lab, f, kind in (("i", "iyo", "DEM.mid variant"), ("i_nasal", "iyo", "DEM.mid variant"),
                         ("ji", "jija", "DEM.far variant"), ("ku", "kose", "-ose 'all/everywhere'"),
                         ("ri", "rija", "DEM.far variant (round 2)"), ("chi", "chija", "DEM.far variant (round 2)"),
                         ("ghi", "ghaja", "DEM.far variant (round 2)"), ("wi", "waja", "DEM.far variant (round 2)"),
                         ("ghu", "ghose", "-ose 'all' (round 2; same form as ghi)")):
        ev = A.after_nouns(NOUNS, lab, f)
        e2 = A.after_nouns(NOUNS, lab, f, before=True)
        ev = {**ev, "evidence_count": ev["evidence_count"] + e2["evidence_count"],
              "examples": ev["examples"] + e2["examples"]}
        add(f"CONCORD.variant.{lab}.{f}", 1, "CONCORD", {"noun_class": lab, "kind": kind}, {"form": f}, ev,
            notes="agent round 1")
    for lab, (_, row) in CLASSES.items():
        f = forms(row, 1)[0] + "ro"
        sf = same(lab, lambda l: forms(CLASSES[l][1], 1)[0])
        add(f"POSS.ro.{lab}", 1, "CONCORD", {"noun_class": lab, "kind": "possessive 'its' (non-human possessor)"},
            {"form": f}, A.after_nouns(NOUNS, lab, f, same_form=sf), notes="agent round 1: varo, charo, gharo")


def section4():
    B, E = alt(ALL_BARE), alt(ALL_E)
    R = "[a-z]{2,}"
    tam("NEG.pres", 4, "NEG", "neg:pres", f"nde{B}?{R}agha", {"template": "nde-SM-ROOT-agha"},
        {"sw": "ha-SM-ROOT-i"})
    tam("NEG.pres_1sg", 4, "NEG", lambda fs: "neg:pres" in fs and "negsm:ni" in fs, f"si(?!cha){R}(?:agha|ire|ie|e)",
        {"template": "si-ROOT-agha / si-ROOT-ire"}, {"sw": "si-ROOT-i (1sg)"})
    tam("NEG.past", 4, "NEG", "neg:ku", f"nde{E}?re{R}ghe", {"template": "nde-SM+e-re-ROOT-ie-ghe"},
        {"sw": "ha-SM-ku-ROOT-a"})
    tam("NEG.past_ie", 4, "NEG", "neg:ku", f"nde{B}{R}(?:ie|ire|ere)",
        {"template": "nde-SM-ROOT-ie (rulebook_correct past negative)"}, {"sw": "ha-SM-ku-ROOT-a"})
    tam("NEG.fut", 4, "NEG", "neg:ta", f"nde{B}?chaa{R}", {"template": "nde-SM-chaa-ROOT-a"},
        {"sw": "ha-SM-ta-ROOT-a"})
    tam("NEG.fut_1sg", 4, "NEG", lambda fs: "neg:ta" in fs and "negsm:ni" in fs, f"sichaa{R}", {"template": "si-chaa-ROOT-a"}, {"sw": "si-ta-ROOT-a"})
    tam("NEG.ja", 4, "NEG", "neg:ja", f"(?:si|nde{B}?){R}(?:ie|ire|ere)", {"template": "si-/nde-SM-ROOT-ie"},
        {"sw": "ha-SM-ja-ROOT-a"})
    tam("NEG.word.hachi", 4, "NEG", "neg:pres", "hachi", {"form": "hachi", "use": "negator word"},
        {"sw": "negative present"})


def extensions():
    """Sections 5-6 from induce_extensions.py: extension surface per Swahili extension and Kidaw'ida
    root shape; perfective final per root shape; post-final -ghe per Swahili tense; and the
    version-B helpers' accuracy on the same aligned verbs."""
    path = HERE / "induced_extension_rules.json"
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    for et, v in d["extensions"].items():
        for sh, t in v["by_root_shape"].items():
            n = t["n"]
            for surf, k, share in t["top"]:
                if surf == "(none)":
                    continue
                tier = "active" if k >= 5 and share >= 0.10 else "tentative" if k >= 2 and share >= 0.03 else None
                if not tier:
                    continue
                ex = next((r["examples"] for r in v["surfaces"] if r["surface"] == surf), [])
                lift = next((r["lift_vs_no_ext"] for r in v["surfaces"] if r["surface"] == surf), None)
                add(f"EXT.{et}.{sh.replace('/', '_')}.{surf}", 5, "EXT",
                    {"sw_ext": et, "dav_root_shape": sh},
                    {"suffix": f"-{surf}-", "share_in_condition": f"{k}/{n}"},
                    {"evidence_count": k, "examples": ex, "method": "aligned verbs, base root from lexicon",
                     "n_condition": n, "lift": lift}, status=tier,
                    notes="root shape = final segment (palatal ny/nj/sh/ch/s/z/j, vowel, other) / last vowel")
    for name, h in d["helpers"].items():
        add(f"EXT.helper.{name}", 5, "EXT", {"helper": name},
            {"code": "hf_space (version B)", "accuracy_where_extension_visible":
             f"{h['k_visible']}/{h['n_with_visible_extension']}"},
            {"evidence_count": h["k_visible"], "examples": [], "method": "helper prediction found in aligned verb",
             "misses": h["misses"], "rate_visible": h["rate_visible"]},
            status="active" if h["rate_visible"] and h["rate_visible"]["rate"] >= 0.5 else "tentative",
            notes="canonical helper scored on clean-train aligned verbs (wrap, don't fork)")
    for sh, v in d["perfective"].items():
        n = v["n"]
        for f, k in v["finals"].items():
            share = k / n
            tier = "active" if k >= 5 and share >= 0.10 else "tentative" if k >= 2 and share >= 0.03 else None
            if tier:
                add(f"FV.perfective.{sh.replace('/', '_')}.{f}", 6, "FV", {"dav_root_shape": sh, "tense": "perfective"},
                    {"final": f"-{f}", "share_in_condition": f"{k}/{n}"},
                    {"evidence_count": k, "examples": v["examples"].get(f, []), "method": "aligned verbs",
                     "n_condition": n, "lift": None}, status=tier)
    # roots that already end in -ir / -er take a plain -e perfective (sikir-e-ghe, uzir-e-ghe): validation 2026-11-12
    B_ = alt(ALL_E)
    ev = A.cooc("tam:li", f"{B_}re[a-z]{{1,}}(?:ir|er)eghe")
    alt_ev = A.cooc("tam:li", f"{B_}re[a-z]{{1,}}(?:ir|er)ieghe")
    add("FV.perfective.r_final", 6, "FV", {"dav_root_ends": "-ir / -er (part of the root)", "tense": "perfective"},
        {"final": "-e", "vs": f"-ie {alt_ev['evidence_count']} pairs (mostly applicative -ir-ie)"}, ev,
        notes="dev round trip: oresikireghe, oreuzireghe; vocab -ireghe 665 vs -irieghe 412, -ereghe 589 vs -erieghe 159")
    for t, v in d["post_final_ghe"].items():
        rate = v["rate"]["rate"]
        add(f"FV.ghe.{t.replace(':', '_')}", 6, "FV", {"sw_feature": t},
            {"post_final": "-ghe", "rate": f"{v['ghe']}/{v['n']}"},
            {"evidence_count": v["ghe"], "examples": [], "method": "aligned verbs", "rate": v["rate"], "lift": None},
            status="active" if rate >= 0.2 else "tentative" if v["ghe"] >= 2 else None or "tentative",
            notes="-ghe marks narrative / remote past; rare with -me- and present / future")


FEATURE_SECTION = {"neg:pres": 4, "neg:ku": 4, "neg:ta": 4, "neg:ja": 4, "neg:nge": 4}


def induced():
    """Verb readings induced from aligned verbs (induce_verb_rules.py): one rule per
    (Swahili feature, Kidaw'ida shape), word-level evidence, active / tentative."""
    path = HERE / "induced_verb_rules.json"
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    for f, v in d["features"].items():
        for k, r in enumerate(v["readings"], 1):
            sec = FEATURE_SECTION.get(f, 3)
            ev = {"evidence_count": r["n"], "examples": r["examples"], "method": "aligned_verbs (induced)",
                  "n_aligned": v["n_aligned"], "share": r["share"], "shape": r["shape"]}
            add(f"{'NEG' if sec == 4 else 'TAM'}.induced.{f.replace(':', '_')}.{k}", sec,
                "NEG" if sec == 4 else "TAM", {"sw_feature": f}, {"template": r["template"]}, ev,
                status=r["tier"], notes="induced from aligned verbs; segmentation by kitaita_verb.parse_prefix")


def rank_readings():
    """Rules with the same slot + condition are alternative readings: the default is a
    speaker-preferred reading if one is marked, else the one with most evidence."""
    groups = {}
    for r in rules:
        if r["status"] in ("active", "tentative") and r["slot"] in ("TAM", "NEG"):
            groups.setdefault((r["slot"], json.dumps(r["if"], sort_keys=True)), []).append(r)
    for g in groups.values():
        g = [r for r in g]
        pref = [r for r in g if any(x.get("default") and x.get("form") == r["then"].get("template")
                                    for x in r.get("readings", []))]
        order = pref + sorted([r for r in g if r not in pref],
                              key=lambda r: (r["status"] != "active", -r["evidence_count"]))
        for i, r in enumerate(order):
            r["rank"] = i + 1
            r["default"] = i == 0
            r["alternatives"] = [x["id"] for x in order if x is not r]


def main():
    section1()
    section2()
    section3()
    section4()
    section_round1()
    induced()
    extensions()
    rank_readings()
    hashes_extra = {"bootstrap_lexicon.json": A.sha16(HERE / "bootstrap_lexicon.json"),
                    "induced_verb_rules.json": A.sha16(HERE / "induced_verb_rules.json"),
                    "induced_extension_rules.json": A.sha16(HERE / "induced_extension_rules.json"),
                    "word_align.py": A.sha16(HERE / "word_align.py")}
    hashes = {"train.jsonl": A.sha16(A.TRAIN), "attest.py": A.sha16(HERE / "attest.py"),
              "build_rules.py": A.sha16(Path(__file__)), **hashes_extra,
              "kitaita_verified_corrections.py": A.sha16(ROOT / "hf_space/kitaita_verified_corrections.py"),
              "kitaita_speaker_verified.json": A.sha16(ROOT / "hf_space/kitaita_speaker_verified.json"),
              "kitaita_verb.py": A.sha16(ROOT / "hf_space/kitaita_verb.py"),
              "swahili_verb.py": A.sha16(ROOT / "hf_space/swahili_verb.py"),
              "kitaita_root_alignment.json (Swahili parser filter only)":
                  A.sha16(ROOT / "hf_space/kitaita_root_alignment.json")}
    meta = {"built_by": "rulebook/build_rules.py", "sections_done": [1, 2, 3, 4, 5, 6, 7],
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                      text=True).stdout.strip(),
            "config": {"min_evidence": MIN_EVIDENCE, "min_lift": MIN_LIFT, "tentative_evidence": TENTATIVE_EVIDENCE,
                       "tentative_lift": TENTATIVE_LIFT, "n_train_pairs": len(A.pairs()),
                       "n_examples": A.N_EXAMPLES},
            "hashes_sha256_16": hashes,
            "priority": ["speaker_verified", "corpus_attested", "grammar_notes", "mined_alignment"],
            "note": "corpus evidence = sentence-pair co-occurrence on clean train, not word alignment"}
    OUT.write_text(json.dumps({"_meta": meta, "rules": rules}, indent=1, ensure_ascii=False))
    RESULTS.write_text(json.dumps({"_meta": meta, "n_rules": len(rules), "rejected": rejected}, indent=1,
                                  ensure_ascii=False))
    by = {}
    for r in rules:
        k = (r["slot"], r["source"], r["status"])
        by[k] = by.get(k, 0) + 1
    for k, v in sorted(by.items()):
        print(k, v)
    print("rejected:", [(r["id"], r["reason"]) for r in rejected])


def v2_patch():
    """rulebook-v2 = rulebook-v1 rules.json unchanged + CLASS.wu_prefix.modifier_ghu. A full rebuild is not
    used: set ordering reorders regex alternatives in 55 rules (same matches), which would muddy the diff."""
    base = json.loads(subprocess.run(["git", "show", "rulebook-v1:rulebook/rules.json"], cwd=ROOT,
                                     capture_output=True, text=True, check=True).stdout)
    vote_forms = {}
    for lab, (_, row) in CLASSES.items():
        a0, e0 = forms(row, 1)[0], forms(row, 2)[0]
        for f in [a0, merge_a(a0, "ose")] + [(a0 if v == "a" else e0) + s for v, s, _ in POSS.values()]:
            vote_forms.setdefault(f, set()).add(lab)
    wu_modifier_rule(vote_forms)
    new = [r for r in rules if r["id"] == "CLASS.wu_prefix.modifier_ghu"]
    assert new and not any(r["id"] == new[0]["id"] for r in base["rules"])
    base["rules"] += new
    base["_meta"]["rulebook_version"] = "v2"
    base["_meta"]["v2_patch"] = {"date": "2026-11-12", "added": [r["id"] for r in new], "base": "tag rulebook-v1",
                                 "built_by": "rulebook/build_rules.py --v2-patch",
                                 "build_rules_sha16": A.sha16(Path(__file__))}
    OUT.write_text(json.dumps(base, indent=1, ensure_ascii=False))
    print("v2:", len(base["rules"]), "rules; added", base["_meta"]["v2_patch"]["added"], new[0]["status"],
          new[0]["evidence_count"])


if __name__ == "__main__":
    v2_patch() if "--v2-patch" in sys.argv else main()
