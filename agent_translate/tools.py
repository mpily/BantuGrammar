"""
agent_translate/tools.py

2026-11-12, CLAUDE_rulebook.md §5. Pure-Python tools a small LLM (arm C) calls to translate
Swahili -> Kidaw'ida. The LLM analyses the Swahili (roles, which noun controls which slot, which
verb parse, which sense); these functions produce every Kidaw'ida verb form and concord prefix
from features, using only the frozen rule book (rulebook/rules.json), dictionary
(rulebook/lexicon.json, lemma maps in rulebook/bootstrap_lexicon.json) and a clean-train word count
(rulebook/dav_vocab.json). No torch, no network.

Every function takes and returns JSON-serialisable values, never raises, and returns
  status: ok | unknown | unattested | flag
plus rule_ids / source so every output word can be traced.
  ok          produced from active rules / entries, ending attested in the corpus
  unattested  produced, but the generated ending is attested < 2 times, or a rule used is tentative
  unknown     no rule / entry for the input
  flag        do not guess: coordinated subject, animacy-override noun, or no settled rule

Canonical helpers are wrapped, not forked (hard rule 10): swahili_verb.parse / decompositions,
kitaita_verb.SERIES / parse_prefix / SW_SM_ROWS, rulebook_correct.applicative_stem /
applicative_perfective, kitaita_verified_corrections.kitaita_passive_form.
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULEBOOK = ROOT / "rulebook"
sys.path.insert(0, str(ROOT / "hf_space"))

TRAIN = Path.home() / "languagemodels" / "rulebook_data" / "train.jsonl"   # retrieval index source (not in git)
RETRIEVAL_LOG = Path.home() / "languagemodels" / "rulebook_data" / "retrieval_log.jsonl"
MIN_ATTESTED = 2
PALATAL = ("ny", "nj", "sh", "ch", "s", "z", "j")

# Kidaw'ida class label -> kitaita_verb.SERIES row (rulebook CLASS.* rules); persons are rows too
CLASS_ROW = {"u": "3sg", "wi": "c2", "ghu": "c3", "i": "c9", "i_nasal": "c9", "ji": "c5", "ghi": "c6", "chi": "c7",
             "va": "c8", "ri": "c10", "lu": "c11", "ku": "c15", "gh": "c3",
             "1sg": "1sg", "2sg": "2sg", "3sg": "3sg", "1pl": "1pl", "2pl": "2pl"}
BANTU_TO_LABEL = {"1": "u", "2": "wi", "3": "ghu", "4": "i", "5": "ji", "6": "ghi", "7": "chi", "8": "va", "9": "i",
                  "10": "ri", "11": "lu", "14": "u", "15": "ku"}
OM_FORM = {"1sg": "ni", "2sg": "ku", "3sg": "m", "1pl": "di", "2pl": "m", "refl": "ku", "u": "m", "wi": "wi",
           "ghu": "ghu", "i": "i", "i_nasal": "i", "ji": "ji", "ghi": "ghi", "chi": "chi", "va": "vi", "ri": "ri",
           "lu": "lu", "ku": "ku"}
OM_RULE = {"2pl": "OM.2pl.m", "refl": "OM.reflexive.ku", "1sg": "OM.1sg.from_sw_ni", "2sg": "OM.2sg.from_sw_ku",
           "3sg": "OM.3sg.from_sw_m", "u": "OM.3sg.from_sw_m", "1pl": "OM.1pl.from_sw_tu", "wi": "OM.c2.from_sw_wa",
           "ghu": "OM.c3.from_sw_u", "i": "OM.c9.from_sw_i", "i_nasal": "OM.c9.from_sw_i", "ji": "OM.c5.from_sw_li",
           "ghi": "OM.c6.from_sw_ya", "chi": "OM.c7.from_sw_ki", "va": "OM.c8.from_sw_vi", "ri": "OM.c10.from_sw_zi"}

# tense key -> (SM series, TAM marker, final kind, post-final -ghe, rule ids). Final kind: a | e | agha | perf
TAM_TABLE = {
    "present": ("a", "we", "a", False, ["TAM.pres.na_we"]),            # speaker preference; zero form = alternative
    "present_zero": ("a", None, "a", False, ["TAM.pres.na_zero"]),
    "perfect": ("a", None, "a", False, ["TAM.perf.me"]),
    "perfect_ie": ("a", None, "perf", False, ["TAM.perf.me_ie"]),
    "past": ("e", "re", "perf", True, ["TAM.past.li", "FV.ghe.tam_li"]),
    "future": ("bare", "cha", "a", False, ["TAM.fut.ta"]),
    "habitual": ("a", "da", "a", False, ["TAM.hab.hu"]),
    "narrative": ("bare", "ka", "a", False, ["TAM.narr.ka"]),
    "conditional": ("bare", "ki", "a", False, ["TAM.cond.ki"]),
    "subjunctive": ("bare", None, "e", False, ["TAM.subj.plain"]),
    "prohibitive": ("bare", "se", "e", False, ["TAM.prohib.se"]),
    "rel_present": ("bare", None, "agha", False, ["TAM.rel.pres"]),
    "rel_past": ("e", "re", "perf", True, ["TAM.rel.past"]),
    "rel_past_cleft": ("bare", None, "perf", True, ["TAM.rel.past_cleft"]),
    "when_past": ("e", "nda", "a", False, ["TAM.rel.lipo"]),
}
NEG_TABLE = {   # tense key -> (series, TAM, final, -ghe, rule ids); 1sg uses si- without SM, 3sg no SM
    "present": ("bare", None, "agha", False, ["NEG.pres"], ["NEG.pres_1sg"], ["NEG.pres_3sg"]),
    "past": ("e", "re", "perf", True, ["NEG.past"], [], ["NEG.past_3sg"]),
    "future": ("bare", "chaa", "a", False, ["NEG.fut"], ["NEG.fut_1sg"], ["NEG.fut_3sg"]),
    "not_yet": ("bare", None, "e", False, ["NEG.induced.neg_ja.1"], [], []),
}
TOOL_NAMES = ("parse_swahili_verb", "lookup_word", "noun_info", "concord", "generate_verb", "generate_modifier",
              "retrieve_examples", "check_translation")


def _safe(fn):
    """Tools never raise: an internal error becomes status unknown with the reason."""
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            return {"status": "unknown", "error": f"{type(e).__name__}: {e}", "rule_ids": []}
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


# ------------------------------------------------------------------ data
@lru_cache(maxsize=1)
def _rules() -> dict[str, dict]:
    return {r["id"]: r for r in json.loads((RULEBOOK / "rules.json").read_text(encoding="utf-8"))["rules"]}


@lru_cache(maxsize=1)
def _lexicon() -> dict[str, dict]:
    return {e["sw"]: e for e in json.loads((RULEBOOK / "lexicon.json").read_text(encoding="utf-8"))["entries"]}


@lru_cache(maxsize=1)
def _lemmas() -> tuple[dict, dict]:
    d = json.loads((RULEBOOK / "bootstrap_lexicon.json").read_text(encoding="utf-8"))["lemmas"]
    return d["sw"], d["dav"]


@lru_cache(maxsize=1)
def _vocab() -> dict[str, int]:
    return json.loads((RULEBOOK / "dav_vocab.json").read_text(encoding="utf-8"))["vocab"]


@lru_cache(maxsize=1)
def _dav_index() -> dict[str, list[dict]]:
    """Kidaw'ida surface form / lemma -> lexicon candidates (with their Swahili lemma)."""
    out: dict[str, list[dict]] = {}
    for sw, e in _lexicon().items():
        for c in e["dav"]:
            for k in {c["lemma"], *c["forms"]}:
                out.setdefault(k, []).append({**c, "sw": sw})
    return out


@lru_cache(maxsize=1)
def _verified_stems() -> set:
    try:
        d = json.loads((ROOT / "hf_space" / "kitaita_speaker_verified.json").read_text(encoding="utf-8"))
        return set(d.get("morphology", {}).get("applicative_perfective_stem", {}))
    except (OSError, ValueError):
        return set()


SW_DEM = set(("huyu hawa huu hii hili haya hiki hivi hizi huku hapa humu yule wale ule ile lile yale kile vile "
              "zile kule pale mle huyo hao huo hiyo hilo hayo hicho hivyo hizo huko hapo humo").split())


def _norm(w: str) -> str:
    from rulebook_flagger import norm
    return norm(w or "")


def _rule_status(ids: list[str]) -> str:
    st = [_rules().get(i, {}).get("status", "missing") for i in ids]
    return "active" if all(s == "active" for s in st) else ("missing" if "missing" in st else "tentative")


@lru_cache(maxsize=4096)
def _ending_count(ending: str) -> int:
    """Clean-train tokens of words ending in this string (any prefix). Same idea as
    kitaita_verb.attested_ending, on the clean vocabulary."""
    if len(ending) < 3:
        return 0
    return sum(c for w, c in _vocab().items() if w.endswith(ending) and len(w) > len(ending))


def _speaker_overrides(subj: str, series: str, tense: str, sm: str, tam):
    """Speaker-verified forms beat corpus forms (CLAUDE_rulebook.md §1.4): verified person markers
    (2pl future m-), SM+a readings marked default in rules.json (ghu present gha-), and the class-specific
    past marker (ji: -de-)."""
    import kitaita_verified_corrections as V
    extra = []
    vt = {"future": "future", "present": "present", "past": "past"}.get(tense)
    if subj in ("1sg", "2sg", "1pl", "2pl") and vt and (subj, vt) in V.PERSON_SUBJECT_MARKERS_BY_TAM \
            and (subj, vt) not in (("1pl", "past"),):
        m = V.PERSON_SUBJECT_MARKERS_BY_TAM[(subj, vt)]
        if m != sm:
            sm = m
            extra.append("SM.person.verified")
    lab = "u" if subj == "3sg" else subj
    r = _rules().get(f"SM.{lab}.{series}")
    if series == "a" and r and r.get("readings") and tense in ("present", "present_zero"):   # speaker data: present only
        d = next((x["form"] for x in r["readings"] if x.get("default")), None)
        if d and d != sm:
            sm = d
            extra.append(r["id"])
    if tense in ("past", "rel_past") and tam == "re" and lab in V.VERIFIED_PAST_TAM_BY_SM:
        tam = V.VERIFIED_PAST_TAM_BY_SM[lab]
        extra.append("TAM.past.verified_by_class")
    return sm, tam, extra


def _sm_before_root(sm: str, row: str, series: str, first: str) -> str:
    """2pl bare SM: m- before a consonant (1,556 corpus tokens), mu- before a vowel (SM.2pl.bare)."""
    if row == "2pl" and series == "bare" and sm in ("m", "mu"):
        return "mu" if first in "aeiou" else "m"
    return sm


def _sw_lemma(word: str) -> str:
    from swahili_verb import parse
    w = _norm(word)
    sw, _ = _lemmas()
    if w in sw:
        return sw[w]
    p = parse(w)
    return p["root"] if p and len(p["root"]) >= 3 else w


def _root_shape(root: str) -> tuple[str, str]:
    end = "vowel" if root[-1:] in "aeiou" else "palatal" if root.endswith(PALATAL) else "other"
    vs = re.findall(r"[aeiou]", root)
    return end, ("eo" if vs and vs[-1] in "eo" else "aiu")


# ------------------------------------------------------------------ 1. parse_swahili_verb
def swahili_tam_key(p: dict) -> tuple[str | None, bool]:
    """Map a swahili_verb.parse reading to (tense key, negative) for generate_verb."""
    mood, tam, rel = p.get("mood"), p.get("tam"), p.get("rel")
    if mood == "prohibitive":
        return "prohibitive", False
    if mood == "subjunctive":
        return "subjunctive", False
    if mood == "habitual" or (mood == "negative" and p.get("neg") == "hu" and p.get("fv") == "a" and tam is None):
        return "habitual", False
    if mood == "negative":
        return {"ku": "past", "ta": "future", "ja": "not_yet", None: "present"}.get(tam), True
    if rel:
        if rel == "po" and tam == "li":
            return "when_past", False
        return {"li": "rel_past", "na": "rel_present"}.get(tam), False
    return {"li": "past", "me": "perfect", "sha": "perfect", "na": "present", "ta": "future", "ka": "narrative",
            "ki": "conditional"}.get(tam), False


@_safe
def parse_swahili_verb(word: str) -> dict:
    """Candidate parses of a Swahili verb (wraps swahili_verb.parse / decompositions)."""
    from kitaita_verb import SW_SM_ROWS
    from swahili_verb import parse
    w = _norm(word)
    p = parse(w)
    if not p:
        return {"status": "unknown", "word": w, "parses": [], "rule_ids": [], "source": "swahili_verb.parse"}
    key, neg = swahili_tam_key(p)
    rows = SW_SM_ROWS.get(p["sm"] or "", ())
    parses = []
    for base, exts in p["decompositions"]:
        parses.append({"neg": p["neg"], "mood": p["mood"], "sm": p["sm"], "sm_rows": list(rows), "tam": p["tam"],
                       "rel": p["rel"], "om": p["om"], "root": base, "exts": exts, "final": p["fv"],
                       "tense_key": key, "negative": neg})
    return {"status": "ok", "word": w, "parses": parses, "rule_ids": [], "source": "swahili_verb.parse"}


# ------------------------------------------------------------------ 2. lookup_word
@_safe
def lookup_word(sw: str, pos: str | None = None) -> dict:
    """Ranked Kidaw'ida candidates for a Swahili word or lemma (rulebook/lexicon.json)."""
    w = _norm(sw)
    lem = _sw_lemma(w)
    lex = _lexicon()
    # rulebook-v5: an exact connective / function-word entry wins over the lemmatizer's guess (kisha -> kish 'nao')
    e = (lex[w] if lex.get(w, {}).get("pos") == "FUNC" else None) or lex.get(lem) or lex.get(w)
    note = None
    if not e:
        # rulebook-v3: one-syllable verb roots (pa 'give', ona 'see', ita 'call', lia 'cry') lose ku- when an
        # object marker is present (alimpa, nimeona); _sw_lemma ignores roots under 3 letters, so these came back
        # unknown. Fallback only when nothing else matched: a short parse root with a verb entry, no negative
        # reading (hapa is not ha-p-a).
        from swahili_verb import parse
        p = parse(w)
        r = p["root"] if p and not p.get("neg") else None
        if r and len(r) < 3 and lex.get(r, {}).get("pos") == "V":
            e = lex[r]
            note = (f"matched through the short verb root '{r}' of the parse; check that '{w}' really is that verb "
                    f"(nouns and adjectives such as ukali can parse the same way)")
        # rulebook-v4: the bare citation form of a one-syllable verb (pa 'give', la 'eat') -> its root (p, l)
        elif len(w) <= 3 and w[-1:] in "aeiou" and lex.get(w[:-1], {}).get("pos") == "V":
            e = lex[w[:-1]]
            note = f"matched '{w}' to the one-syllable verb root '{w[:-1]}'"
    if not e:
        return {"status": "unknown", "sw": w, "lemma": lem, "candidates": [], "rule_ids": [], "source": None}
    # rulebook-v4: a part-of-speech mismatch no longer hides the candidates (the tags are unreliable: sasa
    # 'now' is tagged N, so a lookup with pos ADV came back unknown and Haiku wrote [?sasa])
    pos_note = None
    if pos and e["pos"] != pos and not (pos == "N" and e["pos"] in ("N", "NAME")):
        pos_note = f"entry is tagged {e['pos']}, not {pos}; tags are not reliable, candidates shown anyway"
    # rulebook-v4: for anything but verbs, `word` is the full form to write (the lemma can be a cut stem:
    # sasa -> lemma idan, word idana)
    cands = [{"word": (c["forms"][0] if c.get("forms") else c["lemma"]) if e["pos"] != "V" else None,
              **{k: c.get(k) for k in ("lemma", "forms", "source", "evidence_count", "domains", "class", "plural",
                                       "senses", "conflict")}} for c in e["dav"]]
    status = "ok" if e["source"] in ("speaker_verified", "corpus_attested") else "unattested"
    out = {"status": status, "sw": w, "lemma": e["sw"], "pos": e["pos"], "candidates": cands,
           "close_alternatives": e.get("close_alternatives", False), "rule_ids": [], "source": e["source"]}
    if note:
        out.update(note=note, rule_ids=["LEX.short_verb_root"])
    if pos_note:
        out["pos_note"] = pos_note
    return out


# ------------------------------------------------------------------ 3. noun_info
@lru_cache(maxsize=1)
def _animate_flags() -> tuple[set, set]:
    r = _rules().get("CLASS.animate_override", {})
    return set(r.get("then", {}).get("flag_nouns_sw", [])), set(r.get("evidence", {}).get("nouns", []))


@_safe
def noun_info(dav_noun: str, sw_noun: str | None = None) -> dict:
    """Class, plural, animacy flag, source and conflicts for a Kidaw'ida noun."""
    import kitaita_verified_corrections as V
    n = _norm(dav_noun)
    flag_sw, animals = _animate_flags()
    rules = _rules()
    conflict_rule = rules.get(f"NOUN.conflict.{n}")
    out = {"noun": n, "rule_ids": [], "conflict": bool(conflict_rule)}
    verified = {_norm(k): v for k, v in V.VERIFIED_SUBJECT_MARKERS.items()}
    if n in verified:
        cls = V.TAM_SM_ALIAS.get(verified[n], verified[n]) if verified[n] == "i_nasal" else verified[n]
        out.update({"class": verified[n], "sm_class": cls, "source": "speaker_verified"})
    else:
        c = next((x for x in _dav_index().get(n, []) if x.get("class")), None)
        if c:
            cls = BANTU_TO_LABEL.get(str(c["class"]), str(c["class"]))
            out.update({"class": cls, "sm_class": cls, "source": c.get("class_source"), "sw": c["sw"]})
    lexc = next(iter(_dav_index().get(n, [])), None)
    out["plural"] = lexc.get("plural") if lexc else None
    # rulebook-v2: modifiers (adj, dem, poss, ose, num) can follow another series than the subject marker
    if out.get("class"):
        m = next((x for x in _dav_index().get(n, []) if x.get("mod_class")), None)
        wu = rules.get("CLASS.wu_prefix.modifier_ghu")
        if m and m.get("mod_class") and m.get("mod_class_source") != "same_as_class":
            out["mod_class"] = BANTU_TO_LABEL.get(str(m["mod_class"]), str(m["mod_class"]))
        elif wu and re.match(wu["evidence"]["pattern"], n):
            out["mod_class"] = wu["then"]["modifier_class"]
        else:
            out["mod_class"] = out["class"]
        if out["mod_class"] != out["class"] and wu:
            out["rule_ids"].append(wu["id"])
        out["use"] = "sm_class for SM / OM / REL in verbs; mod_class for adj, dem, poss, ose, num, assoc"
    if n in V.NUMBER_INVARIANT_NOUNS:
        out.update({"plural": n, "number_invariant": True, "sg_sm": V.NUMBER_INVARIANT_SM_SINGULAR,
                    "pl_sm": V.NUMBER_INVARIANT_SM_PLURAL})
        out["rule_ids"].append("PLURAL.number_invariant")
    if conflict_rule:
        out["readings"] = conflict_rule.get("readings")
        out["rule_ids"].append(conflict_rule["id"])
    out["animate"] = n in animals
    if sw_noun and _norm(sw_noun) in flag_sw:
        out.update({"status": "flag", "flag": "animacy override: Swahili concord follows animacy; use the "
                    "Kidaw'ida noun's own class, check with a speaker"})
        out["rule_ids"].append("CLASS.animate_override")
        return out
    out["status"] = "ok" if out.get("class") else "unknown"
    return out


# ------------------------------------------------------------------ 4. concord
CONCORD_SLOTS = ("SM", "SM_a", "SM_e", "OM", "REL", "ADJ", "DEM_near", "DEM_mid", "DEM_far", "POSS", "ASSOC", "OSE",
                 "SELF")


@_safe
def concord(cls: str, slot: str) -> dict:
    """Concord prefix / word for a class (or person) and slot."""
    from kitaita_verb import SERIES
    c = BANTU_TO_LABEL.get(str(cls), str(cls))
    if c not in CLASS_ROW:
        return {"status": "unknown", "class": c, "slot": slot, "forms": [], "rule_ids": []}
    rules, row = _rules(), CLASS_ROW[c]
    lab = c if c in ("1sg", "2sg", "1pl", "2pl") else ("u" if c == "3sg" else c)

    def from_rule(rid, key=None):
        r = rules.get(rid)
        if not r:
            return None
        return r, (r["then"].get(key) if key else r["then"])
    if slot in ("SM", "SM_a", "SM_e"):
        var = {"SM": "bare", "SM_a": "a", "SM_e": "e"}[slot]
        rid = f"SM.{lab}.{var}"
        r = rules.get(rid)
        forms = list(SERIES[row][("bare", "a", "e").index(var)].split()) if not r else r["then"]["forms"]
        return {"status": "ok" if r and r["status"] == "active" else "unattested", "class": c, "slot": slot,
                "forms": forms, "rule_ids": [rid] if r else [], "source": r["source"] if r else "kitaita_verb.SERIES",
                **({"readings": r["readings"]} if r and r.get("readings") else {})}
    if slot == "OM":
        rid = OM_RULE.get(c)
        return {"status": "ok" if rid and _rule_status([rid]) == "active" else "unattested", "class": c, "slot": slot,
                "forms": [OM_FORM[c]] if c in OM_FORM else [], "rule_ids": [rid] if rid else [],
                "phonology": "m + w/b -> mmb (OM.3sg.mmb)" if OM_FORM.get(c) == "m" else None}
    if slot == "REL":
        rid = next((k for k in rules if k.startswith(f"REL.{lab}.")), None)
        bare = SERIES[row][0]
        return {"status": "ok" if rid else "unattested", "class": c, "slot": slot, "forms": [bare],
                "template": "bare SM; present ROOT-agha, past SM+e-re-ROOT-ie-ghe", "rule_ids": [rid] if rid else []}
    table = {"ADJ": (f"ADJ.{lab}", "prefix"), "DEM_near": (f"DEM.{lab}", "near"), "DEM_mid": (f"DEM.{lab}", "mid"),
             "DEM_far": (f"DEM.{lab}", "far"), "POSS": (f"POSS.{lab}", "forms"), "ASSOC": (f"ASSOC.{lab}", "form"),
             "OSE": (f"QUANT.ose.{lab}", "form"), "SELF": (f"CONCORD.self.{lab}", "form")}
    if slot not in table:
        return {"status": "unknown", "class": c, "slot": slot, "forms": [], "rule_ids": [],
                "note": f"slots: {', '.join(CONCORD_SLOTS)}"}
    rid, key = table[slot]
    got = from_rule(rid, key)
    if not got or got[1] is None:
        return {"status": "unknown", "class": c, "slot": slot, "forms": [], "rule_ids": []}
    r, val = got
    forms = val if isinstance(val, dict) else [val]
    variants = [k for k in rules if k.startswith(f"CONCORD.variant.{lab}.") and slot.startswith("DEM")]
    unatt = [f for f, v in r["evidence"].get("by_form", {}).items() if not v.get("attested")]
    status = "ok" if r["status"] == "active" and (r["source"] == "speaker_verified" or not unatt) else "unattested"
    return {"status": status, "class": c, "slot": slot, "forms": forms, "rule_ids": [rid] + variants,
            "source": r["source"], "unattested_forms": unatt,
            **({"variants": [rules[v]["then"]["form"] for v in variants]} if variants else {})}


# ------------------------------------------------------------------ 5. generate_verb
def _ext_stem(root: str, ext: str, perfective: bool) -> tuple[str, str, bool]:
    """Root + one extension (rule id, tentative?). Perfective passive uses -il-/-el-/-ir-/-er- + o."""
    from rulebook_correct import applicative_stem
    from kitaita_verified_corrections import kitaita_passive_form
    end, harm = _root_shape(root)
    if ext == "applicative":
        if end == "vowel":
            return root + "y", "EXT.applicative.vowel_aiu.y", True
        return applicative_stem(root), "EXT.helper.applicative_stem", False
    if ext == "passive":
        if perfective:
            inf = ("er" if harm == "eo" else "ir") if end == "palatal" else ("el" if harm == "eo" else "il")
            return root + inf, f"TAM.past.passive", False
        return kitaita_passive_form(root), "EXT.helper.kitaita_passive_form", False
    if ext == "causative":
        return root + ("esh" if harm == "eo" else "ish"), f"EXT.causative.other_{harm}.{'esh' if harm == 'eo' else 'ish'}", False
    if ext == "stative":
        if end == "vowel":
            return root + "k", "EXT.stative.vowel_aiu.k", False
        return root + ("ek" if harm == "eo" else "ik"), "EXT.stative.other_aiu.ik", harm == "eo"
    if ext == "reciprocal":
        return root + ("n" if end == "vowel" else "an"), "EXT.reciprocal.vowel_aiu.n", end != "vowel"
    raise ValueError(f"unknown extension {ext}")


def _perfective(stem: str, applicative_last: bool, base_root: str) -> tuple[str, str]:
    from rulebook_correct import applicative_perfective
    if applicative_last:
        return applicative_perfective(base_root) + "ie", "EXT.helper.applicative_perfective+ie"
    if stem.endswith(("ir", "er")) and len(stem) >= 4:
        return stem + "e", "FV.perfective.r_final"
    # rulebook-v3: vowel-less roots (ch 'come') take -ie, not the palatal -ire
    if not re.search(r"[aeiou]", stem) and "FV.perfective.no_vowel.ie" in _rules():
        return stem + "ie", "FV.perfective.no_vowel.ie"
    end, harm = _root_shape(stem)
    # rulebook-v3: e/o roots take -ie (counted per root); kogh 'be' and other listed roots keep -e
    v3 = _rules().get("FV.perfective.other_eo.ie_by_root")
    if v3 and end == "other" and harm == "eo":
        if stem in v3["then"]["exceptions_final_e"]:
            return stem + "e", "FV.perfective.other_eo.e"
        return stem + "ie", v3["id"]
    f = {"palatal": "ere" if harm == "eo" else "ire", "vowel": "e"}.get(end, "e" if harm == "eo" else "ie")
    return stem + f, f"FV.perfective.{end}_{harm}.{f}"


@_safe
def generate_verb(features: dict) -> dict:
    """Kidaw'ida verb form from features: dav_root, subj (class label / person), tense (tense key),
    neg, obj?, exts[], post_final?, subj_coordinated?, subj_noun_sw?."""
    from kitaita_verb import SERIES
    f = dict(features or {})
    root = _norm(f.get("dav_root", ""))
    subj = BANTU_TO_LABEL.get(str(f.get("subj", "")), str(f.get("subj", "")))
    tense, neg = f.get("tense"), bool(f.get("neg"))
    if f.get("subj_coordinated"):
        return {"status": "flag", "form": None, "rule_ids": [], "flag": "coordinated subject: agreement has no settled "
                "rule (CLAUDE.md §8) -- ask a speaker", "features": f}
    flag_sw, _ = _animate_flags()
    if f.get("subj_noun_sw") and _norm(f["subj_noun_sw"]) in flag_sw:
        return {"status": "flag", "form": None, "rule_ids": ["CLASS.animate_override"],
                "flag": "animacy-override subject noun: use the Kidaw'ida noun's class (noun_info)", "features": f}
    if not root or subj not in CLASS_ROW:
        return {"status": "unknown", "form": None, "rule_ids": [], "note": "need dav_root and a known subj", "features": f}
    table = NEG_TABLE if neg else TAM_TABLE
    if tense not in table:
        return {"status": "unknown", "form": None, "rule_ids": [],
                "note": f"tense must be one of {sorted(table)} (neg={neg})", "features": f}
    row = CLASS_ROW[subj]
    if neg:
        series, tam, fin, ghe, ids, ids_1sg, ids_3sg = table[tense]
        if subj == "1sg" and ids_1sg:
            prefix, ids = "si" + (tam or ""), ids_1sg
        elif subj in ("3sg", "u") and ids_3sg:
            prefix, ids = "nde" + (tam if tam else "") if tense != "past" else "ndere", ids_3sg
        else:
            sm = SERIES[row][("bare", "a", "e").index(series)]
            sm = _sm_before_root(sm, row, series, (tam or "")[:1] or (f.get("obj") and "x") or root[:1])
            prefix = "nde" + sm + (tam or "")
    else:
        series, tam, fin, ghe, ids = table[tense]
        sm = SERIES[row][("bare", "a", "e").index(series)]
        sm, tam, extra = _speaker_overrides(subj, series, tense, sm, tam)
        sm = _sm_before_root(sm, row, series, (tam or "")[:1] or (f.get("obj") and "x") or root[:1])
        prefix = sm + (tam or "")
        ids = list(ids) + extra
    ids = list(ids)
    tentative = False
    stem, last_app = root, False
    exts = list(f.get("exts") or [])
    perf = fin == "perf"
    for k, e in enumerate(exts):
        stem2, rid, tent = _ext_stem(stem, e, perf and e == "passive")
        stem, last_app = stem2, (e == "applicative" and k == len(exts) - 1)
        ids.append(rid)
        tentative |= tent
    passive_perf = perf and "passive" in exts
    if passive_perf:
        tail = stem + "o"
    elif perf:
        tail, rid = _perfective(stem, last_app, root if len(exts) == 1 else stem[:-1])
        ids.append(rid)
    else:
        tail = stem + {"a": "a", "e": "e", "agha": "agha"}[fin]
    post = f.get("post_final", ghe)
    if post and not tail.endswith("ghe") and root != "kogh":
        tail += "ghe"
        ids.append("FV.ghe.tam_li")
    om = f.get("obj")
    if om:
        omc = BANTU_TO_LABEL.get(str(om), str(om))
        if omc not in OM_FORM:
            return {"status": "unknown", "form": None, "rule_ids": ids, "note": f"unknown obj {om}", "features": f}
        o = OM_FORM[omc]
        if o == "m" and tail[:1] in ("w", "b"):
            tail = "mmb" + tail[1:] if tail[0] == "w" else "mm" + tail
            ids.append("OM.3sg.mmb")
        else:
            tail = o + tail
        ids.append(OM_RULE.get(omc, ""))
    form = prefix + tail
    if "passive" in exts and root.endswith("w") and "EXT.passive.w_final.flag" in _rules():
        return {"status": "flag", "form": None, "rule_ids": ids + ["EXT.passive.w_final.flag"], "features": f,
                "flag": "passive of a w-final root (lw, fw ...) is not attested; ask a speaker"}
    ids = list(dict.fromkeys(i for i in ids if i))
    verified_stem = root in _verified_stems() and "applicative" in exts
    ending = tail if not om else tail[len(OM_FORM[BANTU_TO_LABEL.get(str(om), str(om))]):]
    att = _ending_count(ending)
    rs = _rule_status([i for i in ids if i])
    status = "ok" if (att >= MIN_ATTESTED and rs == "active" and not tentative) or verified_stem else "unattested"
    alts = []
    if not neg and tense == "present":
        z = generate_verb({**f, "tense": "present_zero"})
        alts.append({"form": z.get("form"), "rule_ids": z.get("rule_ids"), "status": z.get("status"),
                     "why": "zero present: also grammatical (speaker), corpus-dominant"})
    return {"status": status, "form": form, "rule_ids": ids, "rule_status": rs,
            "attested_count": att, "attested_form_count": _vocab().get(form, 0), "alternatives": alts, "features": f,
            "source": "rulebook/rules.json"}


# ------------------------------------------------------------------ 6. generate_modifier
@_safe
def generate_modifier(cls: str, dav_stem: str, kind: str) -> dict:
    """Agreeing modifier: kind adj (nominal stem) | adj_stative | dem_near | dem_mid | dem_far | poss (stem =
    1sg..3pl) | num | ose | self | assoc."""
    from kitaita_verb import SERIES
    c = BANTU_TO_LABEL.get(str(cls), str(cls))
    stem = _norm(dav_stem)
    if c not in CLASS_ROW:
        return {"status": "unknown", "form": None, "rule_ids": []}
    row = CLASS_ROW[c]
    if kind == "adj":
        r = concord(c, "ADJ")
        if r["status"] == "unknown":
            return {**r, "form": None}
        form = r["forms"][0] + stem
    elif kind == "adj_stative":
        form, r = SERIES[row][0] + stem, {"rule_ids": ["ADJ.verified_forms"], "status": "ok"}
    elif kind.startswith("dem_"):
        r = concord(c, {"dem_near": "DEM_near", "dem_mid": "DEM_mid", "dem_far": "DEM_far"}[kind])
        return {**r, "form": r["forms"][0] if r.get("forms") else None}
    elif kind == "poss":
        r = concord(c, "POSS")
        forms = r.get("forms") or {}
        return {**r, "form": forms.get(stem) if isinstance(forms, dict) else None}
    elif kind in ("ose", "self", "assoc"):
        r = concord(c, {"ose": "OSE", "self": "SELF", "assoc": "ASSOC"}[kind])
        return {**r, "form": r["forms"][0] if r.get("forms") else None}
    elif kind == "num":
        if stem[:1] in "aeiou":
            bare, a = SERIES[row][0], SERIES[row][1]
            cands = [bare[:-1] + stem if bare[-1:] == stem[0] else bare + stem,          # ji + ingi -> jingi
                     a[:-1] + {"o": "o", "i": "e"}.get(stem[0], stem[0]) + stem[1:]]     # wa + ingi -> wengi
            form = max(cands, key=lambda x: _vocab().get(x, 0))
        else:
            form = SERIES[row][0] + stem
        r = {"rule_ids": ["SM." + ("u" if c == "3sg" else c) + ".bare"], "status": "ok"}
    else:
        return {"status": "unknown", "form": None, "rule_ids": [], "note": "kind: adj | adj_stative | dem_near | "
                "dem_mid | dem_far | poss | num | ose | self | assoc"}
    n = _vocab().get(form, 0)
    return {"status": "ok" if n >= MIN_ATTESTED and r.get("status") == "ok" else "unattested", "form": form,
            "attested_form_count": n, "rule_ids": r.get("rule_ids", [])}


# ------------------------------------------------------------------ 7. retrieve_examples
@lru_cache(maxsize=1)
def _index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform([_norm_sentence(r["swa"]) for r in rows])
    return rows, vec, X


def _norm_sentence(s: str) -> str:
    return " ".join(t for t in (_norm(w) for w in s.split()) if t)


@_safe
def retrieve_examples(sw_sentence: str, k: int = 5) -> dict:
    """k most similar clean-train pairs (char 3-5-gram TF-IDF over the Swahili side). Every call is logged."""
    import time
    if not TRAIN.exists():
        return {"status": "unknown", "examples": [], "rule_ids": [], "note": "retrieval index source missing"}
    rows, vec, X = _index()
    q = vec.transform([_norm_sentence(sw_sentence)])
    sims = (X @ q.T).toarray().ravel()
    k = max(1, min(int(k), 20))
    top = sims.argsort()[::-1][:k]
    ex = [{"pair_id": rows[i]["id"], "corpus": rows[i]["src"], "swahili": rows[i]["swa"], "kidawida": rows[i]["dav"],
           "similarity": round(float(sims[i]), 3)} for i in top]
    try:
        with open(RETRIEVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "query": sw_sentence, "k": k, "ids": [e["pair_id"] for e in ex]},
                               ensure_ascii=False) + "\n")
    except OSError:
        pass
    return {"status": "ok", "examples": ex, "rule_ids": [], "source": "clean train (rulebook/split_manifest.json)"}


# ------------------------------------------------------------------ 8. check_translation
@_safe
def check_translation(sw: str, dav: str) -> dict:
    """Per Swahili verb: find its Kidaw'ida verb (lexicon root) and compare tense / negation / extensions /
    subject class against the rule book's expected shape. An adherence report, not a quality score."""
    from kitaita_verb import SW_SM_ROWS, parse_prefix
    from swahili_verb import parse
    dav_words = [_norm(w) for w in dav.split() if _norm(w)]
    report = []
    for raw in sw.split():
        w = _norm(raw)
        p = parse(w) if w not in SW_DEM else None
        if not p:
            continue
        key, neg = swahili_tam_key(p)
        lu = lookup_word(w)
        roots = [c["lemma"] for c in lu.get("candidates", [])]
        roots += [x[:-1] for x in roots if len(x) >= 4 and x[-1] in "ae"]
        hit = next(((dw, r) for r in roots for dw in dav_words if len(r) >= 2 and r in dw and dw.find(r) > 0), None)
        item = {"sw_word": w, "tense_key": key, "negative": neg, "sw_exts": p["exts"]}
        if not hit:
            report.append({**item, "status": "unknown", "note": "no aligned Kidaw'ida verb (root not found)"})
            continue
        dw, r = hit
        pre, tail = dw[: dw.find(r)], dw[dw.find(r) + len(r):]
        readings = parse_prefix(pre)
        rows = SW_SM_ROWS.get(p["sm"] or "", ())
        rd = next((x for x in readings if x["row"] in rows), readings[0] if readings else None)
        checks = {}
        if rd:
            exp = (NEG_TABLE if neg else TAM_TABLE).get(key)
            checks["neg"] = (rd["neg"] is not None) == neg
            if exp:
                checks["tam"] = (rd["tam"] or None) == (exp[1] or None)
                checks["series"] = rd["variant"] == exp[0] or (neg and key in ("present", "future"))
            checks["sm_class"] = (rd["row"] in rows) if rows else None
        ext_marks = {"passive": ("w", "lw", "il", "el", "ir", "er", "o"), "applicative": ("i", "ir", "er", "y"),
                     "causative": ("ish", "esh", "is", "es"), "stative": ("ik", "ek", "k"), "reciprocal": ("an", "n")}
        for e in p["exts"]:
            checks[f"ext_{e}"] = any(tail.startswith(m) for m in ext_marks.get(e, ()))
        bad = [k for k, v in checks.items() if v is False]
        report.append({**item, "dav_word": dw, "dav_root": r, "dav_prefix_reading": rd, "checks": checks,
                       "status": "ok" if not bad else "flag", "mismatches": bad})
    n = [x for x in report if x["status"] != "unknown"]
    return {"status": "ok" if report else "unknown", "verbs": report,
            "summary": {"n_verbs": len(report), "n_aligned": len(n),
                        "n_all_checks_pass": sum(1 for x in n if x["status"] == "ok")}, "rule_ids": []}


TOOLS = {n: globals()[n] for n in TOOL_NAMES}


def call(name: str, args: dict) -> dict:
    """Dispatch a tool-use call (name + JSON input) -- for the harness."""
    fn = TOOLS.get(name)
    if not fn:
        return {"status": "unknown", "error": f"no tool {name}", "rule_ids": []}
    return fn(**(args or {}))
