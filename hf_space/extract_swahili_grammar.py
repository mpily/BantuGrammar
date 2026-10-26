# -*- coding: utf-8 -*-
"""
extract_swahili_grammar.py

Populate a BantuGrammar for Swahili from the collected Genesis parallel
corpus (bible collection/data/parallel/parallel.jsonl, SUV side),
reusing this project's existing Swahili resources (hard rule #10:
extend, don't fork) rather than re-deriving them:
  - SM_TABLE (subject-marker-per-class) from nllb_concord_rewrite_
    pipeline.py, extended here with the classes it doesn't cover
    (11, 15) -- standard reference-grammar facts, same spirit as the
    original table.
  - the 319-noun swahili_noun_lexicon_agreement_v2.py lexicon, used as
    the primary (exact-match) noun-class source; unmatched nouns get a
    lower-confidence class guess from their own prefix shape.
  - a new TAM-marker table (verb tense/aspect prefixes) -- this
    project had no existing one; standard reference grammar, hand-
    built here the same way SM_TABLE/ADJ_BASE/ASSOC_TABLE were.

Everything here is "known"/"lexicon"/"prefix-pattern" confidence --
this is the higher-confidence HALF of the project; see extract_
kitaita_grammar.py for the much lower-confidence Kitaita side.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from bantu_grammar import BantuGrammar, NounClassEntry, ConcordForm, VerbFormEntry
from nllb_concord_rewrite_pipeline import SM_TABLE as _SM_TABLE_1_10
from swahili_noun_lexicon_agreement_v2 import NOUNS as SWAHILI_LEXICON

PARALLEL_PATH = Path(__file__).parent.parent / "bible collection" / "data" / "parallel" / "parallel.jsonl"

# Extend the existing 1-10 SM_TABLE with the classes it doesn't cover.
# Class 11 merged historically with 14 (abstract) in Swahili, SM "u" --
# same subject-prefix string as class 3. Class 15 (infinitive-as-noun)
# SM is "ku-". Standard reference grammar (Ashton; Mohammed, Modern
# Swahili Grammar), not derived from this corpus.
SM_TABLE = dict(_SM_TABLE_1_10)
SM_TABLE[11] = "u"
SM_TABLE[15] = "ku"
SM_TABLE["ANIMATE-SG"] = SM_TABLE[1]
SM_TABLE["ANIMATE-PL"] = SM_TABLE[2]

# 2026-10-11: object markers (verbal infixes, SM-TAM-OM-ROOT-FV), checked
# directly against SWH_UDT's own NounClass[obj] gold feature before
# building this (52 real instances) rather than assumed from reference
# grammar alone. Direct corpus confirmation: class 2 "wa" (9x: kuwaunganisha,
# kuwafanya, kuwapa, kuwashukuru, kuwaletea, kuwapata, kuwaridhisha,
# kuwasaidia, huwapelekea), class 5 "li" (5x: kulifuatilia, kulitumikia,
# aliyelizuia, ililitumia, ataliongoza), class 9 "i" (4x: kuipatia,
# kuiongelea, kuirudisha, ameitoa), class 1 "m"/vowel-conditioned "mw"
# (5x: kumtia, kumnyima, kumpokonya, kumkabidhi / alimwamuru), class 6
# "ya" (1x: kuyatatua), 1st-person-singular "ni" (1x: zilizonisaidia).
# Every CONFIRMED class (2,5,6,9) turns out to use the EXACT SAME string
# as SM_TABLE's subject marker for that class -- classes 3/4/7/8/10/11
# extrapolate on this same confirmed SM==OM pattern (standard for
# Swahili's non-human classes; Ashton 1944, Mohammed 2001), not directly
# attested in this corpus sample. Class 1 is the one CONFIRMED exception
# (SM "a" vs OM "m"/"mw") -- also standard, not a guess.
#
# IMPORTANT, found directly while building this: NounClass[obj] being
# present on a verb does NOT mean the OM STRING is actually there --
# e.g. gold 'walithibitisha' (Bantu1) and 'hawangetoa' (Bantu3) carry
# NO object-marker substring at all. Object marking is genuinely
# optional in Swahili; this table is used for PRESENCE DETECTION (try
# stripping a candidate, keep it only if the resulting root scores
# better against the Kitaita root dictionary), never for blind
# unconditional stripping -- see decompose_verb_with_object_marker().
OM_TABLE = dict(SM_TABLE)
OM_TABLE[1] = "m"
OM_TABLE["1_vowel"] = "mw"  # before a vowel-initial root (alimwamuru)

# Person object markers -- reference grammar (Ashton 1944), except 1sg
# which is corpus-confirmed above. 2sg "ku" collides by STRING with
# class-15/infinitive's "ku" (not a new ambiguity -- that collision
# already exists for SM_TABLE[15]); "2pl" is a KNOWN, UNRESOLVED
# ambiguity in Swahili ITSELF, not a gap in this table -- the object
# marker "wa" is genuinely identical for "you (plural)" and "them"
# (class 2); flagged, not force-disambiguated.
PERSON_OBJECT_MARKERS = {"1sg": "ni", "2sg": "ku", "1pl": "tu", "2pl": "wa"}
REFLEXIVE_OBJECT_MARKER = "ji"  # invariant; reference grammar.

_OM_CANDIDATES = sorted(
    {*OM_TABLE.values(), *PERSON_OBJECT_MARKERS.values(), REFLEXIVE_OBJECT_MARKER},
    key=len, reverse=True,
)

# 2026-10-11: relative-marker infixes (SM-TAM-REL-[OM]-ROOT-FV), checked
# directly against SWH_UDT's own NounClass[rel] gold feature (67 real
# instances) before building this -- same discipline as OM_TABLE. Direct
# corpus confirmation, multiple independent instances each: class 1
# "ye" (aliyesoma, aliyelizuia), class 2 "o" (waliokuwa, wanaoongoza,
# wasioamini, waliofariki x2), class 5 "lo" (linalosumbua, linalopaswa),
# class 6 "yo" (wanayohitaji, yaliyopita), class 7 "cho" (kilichotokea,
# kilichojitokeza, kinachotawala, kinachoitwa), class 8 "vyo"
# (alivyohukumiwa, wanavyowakilisha), class 9 "yo" (inayodumu,
# inayotumika, iliyotolewa), class 10 "zo" (zilizotajwa, zinazounga,
# zilizotolewa, zinazohusu, zinazoharibikiwa, zinazoendelea,
# zinazochafua, zijazo -- 8x), class 11 "o" (unaohusiana, usiojulikana).
# Classes 3/4/14 extrapolate on the confirmed pattern (not directly
# attested this round): for 4/5/6/7/8/9/10 the REL string is exactly
# ASSOC_TABLE's consonant + "o" instead of "a" (checked directly: ya->yo,
# la->lo, cha->cho, vya->vyo, za->zo -- every attested case matches);
# 1/2/3 are the confirmed exceptions (ye/o/o, not derivable from ASSOC).
# Slot order CONFIRMED directly: REL comes before any OBJECT marker, not
# after -- "inyayoikabili" = i-na-yo-i-kabili (REL "yo" then OM "i").
#
# IMPORTANT, same lesson as OM_TABLE: a verb being the acl:relcl/'mod@
# relcl' HEAD does not guarantee the REL string is actually present on
# THIS verb (e.g. gold 'ilishawaambia', tagged NounClass[rel]=Bantu9,
# has no detectable "yo" at all) -- presence detection, not blind
# stripping, same as object markers.
#
# KNOWN, NOT HANDLED: (1) the "-ja-...-zo" future/customary participle
# puts REL as a SUFFIX, not this infix (e.g. "zijazo" -- a genuinely
# different template, flagged not forced into this rule); (2) the
# free-standing "amba-" + REL-suffix relative pronoun ("ambaye"/
# "ambacho"/...) is a SEPARATE WORD, not an infix at all -- likely an
# easier rewrite (closed-set substitution like rewrite_demonstrative)
# but not checked this round, out of scope.
REL_TABLE = {1: "ye", 2: "o", 3: "o", 4: "yo", 5: "lo", 6: "yo", 7: "cho",
             8: "vyo", 9: "yo", 10: "zo", 11: "o", 14: "o", 15: "ko"}

# 2026-10-21: class 16 ("mahali"/locative-temporal, e.g. "alipoangusha" =
# a-li-po-angusha, "when he dropped") was missing entirely -- standard
# reference grammar (Ashton 1944 Sec. 182; Mohammed 2001), same tier as
# every other REL_TABLE entry, not corpus-derived this round (the SWH_UDT
# 67-instance pass that built the rest of this table happened not to
# contain a locative/temporal relative). Direct user bug report: without
# this, "-po-" was invisible to find_relative_marker_candidates() and the
# whole "po"+root string (e.g. "poangusha") was treated as one opaque
# root, corrupting every lookup for a verb used in this construction.
# Same "presence detection, not blind stripping" caveat applies -- adding
# this entry only makes "-po-" a candidate find_relative_marker_
# candidates() can enumerate, not something it assumes is always there.
REL_TABLE[16] = "po"

_REL_CANDIDATES = sorted(set(REL_TABLE.values()), key=len, reverse=True)


def find_relative_marker_candidates(stem_fv: str) -> list[tuple[str, str]]:
    """Enumerates every known relative-marker string that matches the
    START of `stem_fv` (the post-SM+TAM remainder) AND leaves a
    plausible remainder behind -- does NOT decide which, if any, is
    real (same enumerate-don't-strip discipline as object markers).
    Returns (rel_string, remainder_after_rel) tuples; the remainder may
    itself still contain an object marker before the root (checked
    separately, at rewrite time, via find_object_marker_candidates --
    not precomputed here to avoid combinatorial blowup for a question
    only the rewrite step actually needs to answer)."""
    out = []
    for rel in _REL_CANDIDATES:
        if not stem_fv.startswith(rel):
            continue
        remainder = stem_fv[len(rel):]
        if len(remainder) >= 2 and remainder[-1] in "aeiou":
            out.append((rel, remainder))
    return out


def find_object_marker_candidates(rest: str) -> list[tuple[str, str, str, bool]]:
    """Enumerates every known object-marker string that matches the START
    of `rest` (the post-SM+TAM remainder) AND leaves a plausible stem+FV
    behind (>=2 chars, ends in a vowel) -- does NOT decide which, if any,
    is real; see OM_TABLE's docstring comment for why (same surface-
    collision risk this project already hit and fixed for '-ki-' verb
    detection). Returns (om_string, candidate_root, candidate_fv,
    suppress_tam) tuples, longest om_string first (not a priority ranking
    the caller must respect, just a stable, readable order). suppress_tam
    is always False here -- see decompose_verb()'s own additional "tam
    string IS actually the OM" candidate for when it's True."""
    out = []
    for om in _OM_CANDIDATES:
        if not rest.startswith(om):
            continue
        stem_fv = rest[len(om):]
        if len(stem_fv) >= 2 and stem_fv[-1] in "aeiou":
            out.append((om, stem_fv[:-1], stem_fv[-1], False))
    return out

# Noun class prefixes (singular, plural) -- standard reference grammar,
# used only as a FALLBACK for words not found in the lexicon by exact
# match. Ordered longest-prefix-first where there's overlap.
NOUN_CLASS_PREFIXES = {
    "1/2": ("m", "wa"), "3/4": ("m", "mi"), "5/6": ("ji", "ma"),
    "7/8": ("ki", "vi"), "11": ("u", None), "14": ("u", None), "15": ("ku", None),
}

# Standard Swahili TAM (tense/aspect/mood) verb markers -- reference
# grammar, not derived from this corpus. Negative paradigm (ha-/sija-/
# etc.) is a known, separate gap -- not attempted here, flagged in notes.
TAM_MARKERS = {
    "na": "present", "li": "past", "ta": "future", "me": "perfect",
    "ki": "conditional/while", "nge": "conditional(hypothetical)",
    "ngali": "conditional(past-hypothetical)", "ja": "not-yet(negative-perfect)",
    "hu": "habitual",
    # 2026-10-09: narrative/consecutive -ka- was missing entirely -- a real gap,
    # not a special case for one caller. kitaita_sentence_generator.translate()
    # already has its own dedicated branch for Swahili tam=="ka" (narrative ->
    # Kitaita -ka-, attested identical retained form), so this decomposer should
    # recognize the Swahili side of that same marker rather than silently failing
    # every narrative-tense sentence (caught via kitaita_e2e_pipeline.py testing
    # "Mungu akawona" / "Mtu mkubwa akaona").
    "ka": "narrative/consecutive",
}

# Negation -- standard reference grammar (Ashton 1944; Mohammed 2001),
# added 2026-10-07 per direct user request; still only the GENERAL
# marker + the 1st-singular irregular, not the full negative paradigm
# (e.g. sija-, haku- compound negative-TAM forms remain a known gap --
# see extract_kitaita_grammar.py's notes; this is reference data, not
# derived from the corpus, so it carries no count/confidence fields).
NEGATION_MARKERS = {
    "ha": "negative, general (replaces/precedes the ordinary subject "
          "marker for every class except 1st person singular; present-"
          "tense final vowel changes -a -> -i, e.g. cl.1 a- + ha- -> "
          "ha- 'haendi' = s/he does not go; cl.2 wa- + ha- -> hawa- "
          "'hawaendi' = they do not go)",
    "si": "negative, 1st person singular only (replaces 'ni-'/the "
          "zero 1sg marker outright, not just prefixed to it, e.g. "
          "'siendi' = I do not go)",
}

# 2026-10-15: Swahili PERSON subject-marker prefixes (reference grammar,
# Ashton 1944/Mohammed 2001 -- same tier as SM_TABLE above, not corpus-
# derived). SM_TABLE only ever covered noun CLASSES; a verb with no overt
# noun subject but a person prefix ("Ninapenda" = I like, "Tunapenda" = we
# like) was previously simply unrecognized by decompose_verb() at all
# (none of SM_TABLE's class strings are a prefix of "ninapenda"/
# "tunapenda"/"mnapenda"). "u" (2sg, "you") is a genuine STRING COLLISION
# with SM_TABLE[3]/[11] -- both are real, valid readings of the same
# prefix; disambiguated only at the SENTENCE level (kitaita_e2e_pipeline.py
# prefers a class-3/11 noun subject when one exists in the sentence, and
# falls back to the person reading only when no subject noun was found at
# all). 3sg "a"/3pl "wa" are deliberately NOT included here even though
# they're also person prefixes -- those strings are already fully covered
# by SM_TABLE[1]/[2] (animate singular/plural), and resolving a BARE 3rd-
# person pro-drop with no noun in the sentence at all is a coreference
# problem (which earlier noun/sentence is "it"/"they"?) explicitly out of
# this project's scope (CLAUDE.md: "Cross-sentence object markers/pronouns
# are a coreference problem -- out of scope"), not something this table
# should paper over.
PERSON_SM_TABLE = {"ni": "1sg", "tu": "1pl", "m": "2pl", "u": "2sg"}

# 2026-10-15: Swahili NEGATIVE-PRESENT subject markers (reference grammar,
# Ashton 1944 ch. on the negative; same tier as SM_TABLE/PERSON_SM_TABLE,
# not corpus-derived). Negative present tense has NO separate tense marker
# at all (unlike the affirmative "na-") -- it is NEG-subject-prefix + root
# + final vowel '-i' (not '-a'), e.g. "sipendi" = si+pend+i = "I do not
# like". The person/class-1/class-2 forms are irregular/fused (si-, hu-,
# ha-, hatu-, ham-, hawa- -- NOT simply "ha" + the ordinary SM string: e.g.
# 2sg is "hu-", not "ha-u-"), so they're listed directly rather than
# derived. Every OTHER noun class (3,4,5,6,7,8,9,10,11,15) is fully
# transparent: "ha" + that class's own ordinary SM_TABLE string, with NO
# contraction (hau-, hai-, hali-, haya-, haki-, havi-, hai-, hazi-, hau-,
# haku-) -- built programmatically below from SM_TABLE itself rather than
# hand-copied, so it can't silently drift out of sync with it.
#
# Scope: PRESENT negative only. Other negative tenses (past "ha-SM-ku-
# root-a", future "ha-SM-ta-root-a", perfect "ha-ja-...") have different
# shapes entirely and are NOT attempted here -- same "one gap closed per
# change" discipline as every other addition in this file, flagged as a
# follow-on rather than silently half-implemented.
NEGATIVE_SM_IRREGULAR = {
    "si": "1sg", "hu": "2sg", "ha": "3sg/cl1", "hatu": "1pl", "ham": "2pl", "hawa": "3pl/cl2",
}
NEGATIVE_SM_TRANSPARENT = {
    cls: f"ha{sm}" for cls, sm in SM_TABLE.items() if cls not in (1, 2, "ANIMATE-SG", "ANIMATE-PL")
}
_NEGATIVE_SM_CANDIDATES = sorted(
    [(s, label) for s, label in NEGATIVE_SM_IRREGULAR.items()]
    + [(s, f"cl{cls}") for cls, s in NEGATIVE_SM_TRANSPARENT.items()],
    key=lambda sl: -len(sl[0]),
)


def decompose_negative_verb(word: str):
    """Recognize a Swahili negative-PRESENT verb form (see
    NEGATIVE_SM_IRREGULAR/NEGATIVE_SM_TRANSPARENT above). Requires the
    final vowel to be literally '-i' -- the one unambiguous morphological
    signal that this is negative-present rather than some unrelated word
    that happens to start with the same letters (affirmative Swahili verbs
    are overwhelmingly '-a'-final) -- same "require positive evidence,
    don't just pattern-match" discipline as decompose_verb()'s own TAM/FV
    check. Returns a VerbFormEntry (negated=True, tam_label=
    'negative_present') or None (flag, don't guess)."""
    wl = word.lower()
    for sm, label in _NEGATIVE_SM_CANDIDATES:
        if not wl.startswith(sm):
            continue
        rest = wl[len(sm):]
        if len(rest) >= 3 and rest[-1] == "i":
            is_person = "/" not in label and not label.startswith("cl")
            return VerbFormEntry(
                surface=word, subject_marker=sm, tam_marker="", tam_label="negative_present",
                root_guess=rest[:-1], final_vowel=rest[-1], confidence="reference-grammar",
                negated=True, person=label if is_person else None,
            )
    return None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text)


def load_swahili_tokens() -> Counter:
    counts = Counter()
    with PARALLEL_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("SUV", "")
            for tok in tokenize(text):
                counts[tok] += 1
    return counts


def guess_noun_class_by_prefix(word: str) -> tuple[str, str] | None:
    wl = word.lower()
    for cls, (sg_pre, pl_pre) in NOUN_CLASS_PREFIXES.items():
        if pl_pre and wl.startswith(pl_pre) and len(wl) > len(pl_pre) + 2:
            return cls, "plural-shape"
        if wl.startswith(sg_pre) and len(wl) > len(sg_pre) + 2:
            return cls, "singular-shape"
    return None


def decompose_verb(word: str) -> VerbFormEntry | None:
    """Greedy SM + TAM + stem + FV segmentation. Only returns a result
    when both the SM and TAM slot match known strings AND a non-trivial
    stem + a plausible final vowel remain -- otherwise returns None
    (flag, don't guess)."""
    wl = word.lower()
    # 2026-10-15: PERSON_SM_TABLE's strings ("ni"/"tu"/"m", plus "u" which
    # already existed as class 3/11's own string) are included here too, so
    # a pro-dropped-subject verb like "ninapenda"/"tunapenda"/"mnapenda" is
    # recognized at all -- previously NONE of SM_TABLE's class-only strings
    # matched these prefixes, so decompose_verb returned None outright for
    # every such form.
    sm_candidates = sorted({str(v) for v in SM_TABLE.values()} | set(PERSON_SM_TABLE), key=len, reverse=True)
    for sm in sm_candidates:
        if not wl.startswith(sm):
            continue
        rest = wl[len(sm):]
        for tam, label in sorted(TAM_MARKERS.items(), key=lambda kv: -len(kv[0])):
            if rest.startswith(tam):
                stem_fv = rest[len(tam):]
                if len(stem_fv) >= 3 and stem_fv[-1] in "aeiou":
                    # 2026-10-11: object-marker candidates are ENUMERATED here
                    # (root_guess/final_vowel below are UNCHANGED either way,
                    # every existing caller keeps its old behavior) -- see
                    # OM_TABLE's docstring for why this doesn't just strip.
                    om_candidates = find_object_marker_candidates(stem_fv)
                    # Infinitives (sm=="ku") take NO tense marker at all, so
                    # when the matched "tam" string ALSO happens to be a known
                    # OM string (e.g. "li" = both past-TAM and class-5 OM --
                    # confirmed real collision: "kulifuatilia" genuinely means
                    # "to follow IT(cls5) up", not "to-PAST follow-up"), add
                    # the alternate reading explicitly: root_guess/final_vowel
                    # are ALREADY the correct root+FV for this case (nothing
                    # else to strip), only the TAM-vs-OM interpretation of
                    # "tam" differs.
                    if sm == "ku" and tam in _OM_CANDIDATES:
                        om_candidates = om_candidates + [(tam, stem_fv[:-1], stem_fv[-1], True)]
                    # Relative-marker candidates -- finite forms only (this
                    # branch), relative clauses need a conjugated verb.
                    rel_candidates = find_relative_marker_candidates(stem_fv)
                    # 2026-10-23: see VerbFormEntry.ku_infinitive_alternate's
                    # own comment -- "ku-" + a string that ALSO happens to be
                    # a real TAM marker (e.g. "ta"/"na"/"ki"/"ka"/"li"/"da")
                    # is genuinely ambiguous between the infinitive reading
                    # (whole `rest` is the root, no TAM at all) and this
                    # branch's own TAM-consuming reading -- enumerated here,
                    # not decided (same discipline as the OM-candidate check
                    # just above).
                    ku_infinitive_alternate = None
                    if sm == "ku" and len(rest) >= 3 and rest[-1] in "aeiou":
                        ku_infinitive_alternate = (rest[:-1], rest[-1])
                    return VerbFormEntry(
                        surface=word, subject_marker=sm, tam_marker=tam, tam_label=label,
                        root_guess=stem_fv[:-1], final_vowel=stem_fv[-1], confidence="prefix-pattern",
                        object_marker_candidates=om_candidates,
                        relative_marker_candidates=rel_candidates,
                        person=PERSON_SM_TABLE.get(sm),
                        ku_infinitive_alternate=ku_infinitive_alternate,
                    )
        # 2026-10-11: bare infinitive (ku- + [object marker] + root + FV,
        # NO tense/aspect marker -- grammatical only for the infinitive
        # SM "ku"). Checked directly: kumtia/kuwafanya/kuyatatua/
        # kuwaunganisha/kuwasaidia and several more real gold NounClass[obj]
        # examples were ALL silently unrecognized before this (decompose_
        # verb returned None outright, not just root-corrupted) -- a
        # genuinely different gap from the TAM-present case above, not
        # previously documented.
        if sm == "ku" and len(rest) >= 3 and rest[-1] in "aeiou":
            return VerbFormEntry(
                surface=word, subject_marker=sm, tam_marker="", tam_label="infinitive/bare",
                root_guess=rest[:-1], final_vowel=rest[-1], confidence="prefix-pattern",
                object_marker_candidates=find_object_marker_candidates(rest),
            )
        # 2026-10-20, direct user bug report ("yeye aletaye uharibifu... hata
        # uharibifu uipate ngome" -> "no verb found", even though this
        # project already has usable Kitaita evidence for BOTH 'leta' and
        # 'pata'): decompose_verb() had NO path at all for a verb with NO
        # TAM marker for any SM other than 'ku' -- every other SM required
        # an explicit TAM string to match, so a genuinely tenseless Swahili
        # verb form was silently unrecognized outright, not just root-
        # corrupted. Two DIFFERENT tenseless constructions, checked in this
        # order (the first is more specific -- a plain final-vowel check
        # alone would wrongly also match it, see below):
        #
        # 1. The SUFFIXED "general/zero" relative ("a-leta-ye" = SM + root +
        #    FV + a relative-marker SUFFIX, e.g. class 1 '-ye' -- a
        #    genuinely different position from the ALREADY-handled INFIX
        #    relative above, SM+TAM+REL+root+FV, e.g. "a-na-ye-fanya"; this
        #    one has NO TAM slot at all). Tried FIRST: "aletaye" ends in
        #    'e', which the bare-subjunctive check below would otherwise
        #    match too, wrongly leaving the '-ye' suffix glued onto the root
        #    ("letay", not "let") -- checked directly before ordering these.
        for rel in _REL_CANDIDATES:
            if not rest.endswith(rel):
                continue
            stem_fv = rest[:-len(rel)]
            if len(stem_fv) >= 3 and stem_fv[-1] in "aeiou":
                return VerbFormEntry(
                    surface=word, subject_marker=sm, tam_marker="", tam_label="relative/tenseless (suffixed)",
                    root_guess=stem_fv[:-1], final_vowel=stem_fv[-1], confidence="prefix-pattern",
                    object_marker_candidates=find_object_marker_candidates(stem_fv),
                    relative_marker_candidates=[(rel, stem_fv)],
                    person=PERSON_SM_TABLE.get(sm),
                )
        # 2. Bare SUBJUNCTIVE (SM + [object marker] + root + '-e', no TAM --
        #    the one other Swahili mood that needs no explicit tense marker,
        #    always final-vowel '-e' rather than '-a'; distinguishes it from
        #    an ordinary indicative verb that's simply missing its TAM,
        #    which is NOT a real Swahili form and shouldn't be guessed at).
        #    `rest not in TAM_MARKERS` guard found directly while testing:
        #    "inge" (sm='i', rest='nge') was spuriously matching this branch
        #    as root 'ng'+'-e' -- but 'nge' is ITSELF the whole conditional-
        #    hypothetical TAM marker with nothing left over (the TAM-loop
        #    above already tried and correctly REJECTED that reading, since
        #    stem_fv would be empty) -- a string that IS a complete TAM
        #    marker and nothing else is strong evidence this is a TAM+no-
        #    root non-word, not a genuine 2-letter root, so it's excluded
        #    here rather than guessed at.
        if len(rest) >= 3 and rest[-1] == "e" and rest not in TAM_MARKERS:
            # 2026-10-21: direct user bug report ("Usiogope" -> root
            # "siogop", wrong -- it's negative "u-si-ogop-e", not an
            # affirmative subjunctive whose root happens to start with
            # "si"). See VerbFormEntry.negative_subjunctive_candidate's own
            # docstring for why this is ENUMERATED here, not decided --
            # "usikilize" (root "sikiliz", genuinely NOT negated) is the
            # same string shape and must not be silently reinterpreted.
            neg_candidate = rest[2:-1] if rest.startswith("si") and len(rest) >= 5 else None
            return VerbFormEntry(
                surface=word, subject_marker=sm, tam_marker="", tam_label="subjunctive/bare",
                root_guess=rest[:-1], final_vowel=rest[-1], confidence="prefix-pattern",
                object_marker_candidates=find_object_marker_candidates(rest),
                person=PERSON_SM_TABLE.get(sm),
                negative_subjunctive_candidate=neg_candidate,
            )
    # 2026-10-22, direct user bug report ("nafurahia" -> no verb found,
    # even though "ninafurahia" is the exact same verb and decomposes
    # fine): standard Swahili reference grammar (Ashton 1944; universally
    # attested in natural text) allows the 1sg present subject marker
    # "ni-" to be ELIDED directly before the "-na-" TAM marker --
    # "ninasoma"/"nasoma" are both grammatical for "I am reading". The
    # main loop above never tries this because "na" is a TAM string, not
    # an SM candidate, so a word that starts with "na" and has NO OTHER
    # SM match at all (every sm_candidates entry above already failed)
    # was previously just unrecognized outright, not merely root-
    # corrupted. Tried only here, after every SM-prefixed reading has
    # failed, so no existing recognized form changes behavior. Same
    # minimum-root-length + final-vowel-is-a-vowel discipline as every
    # other branch in this function.
    if wl.startswith("na"):
        rest = wl[2:]
        if len(rest) >= 3 and rest[-1] in "aeiou":
            return VerbFormEntry(
                surface=word, subject_marker="ni", tam_marker="na", tam_label="present",
                root_guess=rest[:-1], final_vowel=rest[-1], confidence="reference-grammar",
                object_marker_candidates=find_object_marker_candidates(rest),
                relative_marker_candidates=find_relative_marker_candidates(rest),
                person="1sg",
            )
    # 2026-10-15: tried only after every affirmative reading above has
    # failed, so no existing recognized form changes behavior -- this only
    # adds recognition for words that previously returned None outright.
    return decompose_negative_verb(word)


def build() -> BantuGrammar:
    g = BantuGrammar(language_name="Swahili", iso_code="swh")
    token_counts = load_swahili_tokens()
    print(f"loaded {sum(token_counts.values())} Swahili tokens ({len(token_counts)} types) "
          f"from the Genesis parallel corpus", flush=True)

    lexicon_by_surface = {lemma: (cls, gloss) for lemma, cls, gloss in SWAHILI_LEXICON}

    verb_count, noun_count = 0, 0
    for word, count in token_counts.items():
        wl = word.lower()
        if wl in lexicon_by_surface:
            cls, gloss = lexicon_by_surface[wl]
            g.noun_classes.setdefault(cls, []).append(
                NounClassEntry(lemma=wl, noun_class=cls, gloss=gloss, confidence="lexicon",
                                source="swahili_noun_lexicon_agreement_v2.py exact match",
                                count_in_corpus=count))
            noun_count += 1
            continue
        v = decompose_verb(word)
        if v is not None:
            v.count_in_corpus = count
            g.verb_forms.append(v)
            g.tam_marker_inventory.setdefault(v.tam_marker, {"label": v.tam_label, "count": 0,
                                                              "confidence": "prefix-pattern"})
            g.tam_marker_inventory[v.tam_marker]["count"] += count
            verb_count += 1
            continue
        guess = guess_noun_class_by_prefix(word)
        if guess is not None and count >= 2:  # require repetition to reduce prefix-coincidence noise
            cls, shape = guess
            g.noun_classes.setdefault(cls, []).append(
                NounClassEntry(lemma=wl, noun_class=cls, confidence="prefix-pattern",
                                source=f"prefix shape ({shape}), not lexicon-verified",
                                count_in_corpus=count))
            noun_count += 1

    for cls in SM_TABLE:
        g.concord_forms.append(ConcordForm(noun_class=str(cls), relation="subject",
                                            form=SM_TABLE[cls], confidence="known",
                                            example_word=None))

    g.notes.append(f"{noun_count} noun-class entries, {verb_count} decomposed verb tokens "
                    f"out of {len(token_counts)} distinct word types (many types are neither -- "
                    f"particles, adverbs, proper nouns, forms the segmenter correctly refused).")
    g.notes.append("Negative verb paradigm (ha-/sija-/haku- etc.) NOT attempted -- separate, "
                    "known gap, not guessed at.")
    g.notes.append("Class 5 SM 'li-' is string-identical to the past-tense TAM marker '-li-' -- "
                    "a genuine Swahili segmentation ambiguity, not a bug in this extractor.")
    return g


if __name__ == "__main__":
    g = build()
    print(g.summary())
    out = Path(__file__).parent / "bantu_grammar_swahili.json"
    import dataclasses
    out.write_text(json.dumps(dataclasses.asdict(g), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {out}")
