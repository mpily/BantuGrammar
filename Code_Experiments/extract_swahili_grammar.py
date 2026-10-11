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
    sm_candidates = sorted({str(v) for v in SM_TABLE.values()}, key=len, reverse=True)
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
                    return VerbFormEntry(
                        surface=word, subject_marker=sm, tam_marker=tam, tam_label=label,
                        root_guess=stem_fv[:-1], final_vowel=stem_fv[-1], confidence="prefix-pattern",
                        object_marker_candidates=om_candidates,
                        relative_marker_candidates=rel_candidates,
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
    return None


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
