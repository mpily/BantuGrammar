# -*- coding: utf-8 -*-
"""
kitaita_e2e_pipeline.py

2026-10-09. Direct user request: "put everything together and test the
end to end pipeline ... give a google translate kind of page where I
input a text in some language in and it gives the kitaita translation
so that we can see how far we have come and how far we have left to
go." This script is the LOCAL (non-artifact) half of that: it connects

  (1) real NLLB (facebook/nllb-200-distilled-1.3B, CPU, already cached
      locally -- confirmed directly, see docs/status.md) for an
      OPTIONAL English -> Swahili leg, to

  (2) a NEW free-text Swahili sentence decomposer (this file), which
      recovers the (subject noun, verb {sm,tam,root,fv}, object noun,
      adjective {stem,class}) tuple that kitaita_sentence_generator.
      translate() already expects -- reusing extract_swahili_grammar.
      py's decompose_verb()/SM_TABLE/TAM_MARKERS, the 319-noun
      swahili_noun_lexicon_agreement_v2 lexicon, page1_vocabulary.
      json's statistical noun candidates, and nllb_concord_rewrite_
      pipeline.py's ADJ_BASE concord-prefix table (hard rule #10:
      extend existing machinery, don't fork it) -- feeding

  (3) kitaita_sentence_generator.translate(), this project's own
      native-speaker-corrected rule-based Swahili -> Kitaita rewriter,
      unchanged.

SCOPE, stated up front rather than overclaimed (hard rule: never claim
coverage without checking): kitaita_sentence_generator.translate()
only ever accepts ONE Subject-Verb-(Object)-(Adjective) clause per
call, so decompose_sentence() (the HEURISTIC, adjacency-based path,
used when NLLB/the trained probes aren't available -- the browser
Artifact, most notably) still only attempts ONE such clause per
sentence; a second verb-shaped token is flagged and kept as Swahili,
since guessing its subject by adjacency is exactly the kind of
guessing this project avoids. decompose_sentence_with_probes() (used
when NLLB + the saved probes ARE available) does NOT have this limit
-- 2026-10-09, direct user directive: "remove the one verb
restriction... we can handle multiple verbs." It finds every verb,
asks the probe for EACH verb's own subject/object independently
(against the same candidate-noun pool, since a noun can legitimately
be the shared, pro-dropped subject of more than one verb), and calls
translate() once per verb -- see that function's docstring. Negation,
relative clauses, and a genuinely unidentifiable relation are still
FLAGGED as out-of-scope/unresolved in both paths, never silently
guessed at or dropped.

WORD ORDER, direct user correction 2026-10-09: this pipeline's job is
MARKER-ONLY SUBSTITUTION (identify each word/morpheme-slot's relation,
substitute subject/verb SM+TAM/object/adjective appropriately) -- it
never rearranges the sentence. decompose_sentence() below only
IDENTIFIES which token plays which role and WHERE it sits in the
input; assemble_in_order() places each substituted Kitaita word back
at that SAME position. An earlier version of this file mistakenly
treated kitaita_sentence_batch.py's own fixed Swahili-prompt-display
convention (adjective always trails, for isolating one test variable
at a time) as if it were evidence about real Kitaita word order,
and tried to "fix" adjective placement based on that -- wrong, and
retracted; see docs/status.md 2026-10-09 for the full account.

2026-10-09, UPDATE (direct user directive): the parent project's NLLB-
encoder relation PROBES (nllb_concord_rewrite_pipeline.py and
siblings) ARE now wired in -- see decompose_sentence_with_probes()/
translate_sentence_with_probes() below and nllb_probe_identification.
py. The note above (ATTENTION being an unreliable artifact) was about
raw attention heads, a DIFFERENT, cruder mechanism than the TRAINED
relation probes used here; it does not apply to this integration.
decompose_sentence()/translate_sentence() (the adjacency heuristic)
are kept, unchanged, as the fallback for contexts that can't load
NLLB (the browser Artifact, which cannot run Python/ML at all) and as
a baseline to compare the probe-based path against.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from extract_swahili_grammar import (
    decompose_verb, tokenize, guess_noun_class_by_prefix, SM_TABLE, TAM_MARKERS,
    find_object_marker_candidates,
)
from swahili_noun_lexicon_agreement_v2 import NOUNS as SWAHILI_LEXICON
from bantu_grammar import VerbFormEntry
from nllb_concord_rewrite_pipeline import ADJ_BASE, POSS_TABLE, DEM_PROXIMAL_TABLE, DEM_REFERENTIAL_TABLE, DEM_DISTAL_TABLE
from kitaita_verified_corrections import (
    VERIFIED_NOUN_TRANSLATIONS, VERIFIED_ROOTS, VERIFIED_IDIOMS, VERIFIED_SUBJECT_MARKERS, DEVERBAL_SM_HYPOTHESIS,
    NUMBER_INVARIANT_NOUNS, NUMBER_INVARIANT_SM_SINGULAR, BAYA_STEM, KUBWA_FORM_BY_SM, KUBWA_DEFAULT_FORM,
    ZURI_ROOT, SAFI_STEM, MOJA_ROOT, ADJ_SINGLE_CLASS_CANDIDATE, ADJ_SINGLE_ROOT_GENERALIZED, kitaita_adj_prefix, VERIFIED_ADJECTIVES,
    LACHA_FORM_BY_SM, LACHA_DEFAULT_FORM, FUNCTION_WORD_TRANSLATIONS, POSS_PERSON_FORM_CLASS1,
    POSS_PERSON_FORM_BY_CLASS, OTE_STEM, OTE_STEM_CONFIRMED_CLASSES, SIKU_PLURAL_TRIGGER,
    INGI_FORM_BY_SM, INGI_DEFAULT_FORM, INGI_SURFACE_FORMS,
    DOGO_FORM_BY_SM, DOGO_DEFAULT_FORM, GENI_FORM_BY_SM, GENI_DEFAULT_FORM,
    FUPI_FORM_BY_SM, FUPI_DEFAULT_FORM, ZITO_FORM_BY_SM, ZITO_DEFAULT_FORM,
    CHAFU_FORM_BY_SM, CHAFU_DEFAULT_FORM, TAM_SM_ALIAS, PRESENT_SM_VOWEL_CONFIRMED,
    KITAITA_ASSOCIATIVE_VERIFIED, KITAITA_ASSOCIATIVE_DESCRIPTOR_VERIFIED, KITAITA_DEM_PROXIMAL_VERIFIED,
    KITAITA_DEM_REFERENTIAL_VERIFIED, KITAITA_DEM_DISTAL_VERIFIED, SWAHILI_CLASS_DEFAULT_SM_GUESS,
)
from kitaita_sentence_generator import load_pages, build_lookup_tables, translate, resolve_verb_root

LEXICON_BY_SURFACE = {lemma: (cls, gloss) for lemma, cls, gloss in SWAHILI_LEXICON}
ADJ_PREFIXES = sorted({p for p in ADJ_BASE.values()}, key=len, reverse=True)  # "", "m", "wa", "mi", "ki", "vi", "n", "ma"
ADJ_ROOTS = ["kubwa", "baya", "zuri", "dogo", "geni", "refu", "zito", "chafu", "fupi", "safi", "moja"] + \
            sorted(INGI_SURFACE_FORMS)
# 2026-10-20: "-ingi" (many/much) is listed by its SURFACE forms (wengi/
# vingi/nyingi/mengi/mingi/wingi), not a bare root -- see INGI_SURFACE_
# FORMS's own comment (kitaita_verified_corrections.py) for why: Swahili's
# OWN vowel coalescence means it never decomposes as ADJ_PREFIXES+root the
# way every other entry here does.
# 2026-10-11: extended from just kubwa/baya to all 9 Bible-corpus-mined adjective
# families (kitaita_verified_corrections.py ZURI_ROOT/ADJ_SINGLE_CLASS_CANDIDATE) so
# they're actually recognized as adjective CANDIDATES in free text -- translate()'s own
# concord logic still decides per-stem whether real Kitaita data exists for the target
# class; most of the 7 non-kubwa/baya additions will currently come back flagged/None
# for most classes (single data points, not generalized), that's expected, not a bug.

# 2026-10-11: possessive-pronoun detection ("vyako"/"wangu"/"wake"/...).
# Direct user question: "why wasn't it wired together with the adjectives
# since it is adjective shaped" -- checked directly, not assumed: the POS
# probe tags 'vyako' as 'OTHER', not 'ADJ' (gold UD/SUD tags possessives
# as DET/PRON, a different category from true adjectives, even though
# the morphology -- concord prefix + stem -- looks the same). The
# adjective sweep gates on pos_tags[i]=="ADJ", so it structurally cannot
# catch these regardless of how the detection LOGIC is written -- this
# needs its own morphological check (POSS_TABLE prefix + a closed set of
# person-stems), not a POS-probe fallback.
POSS_PERSON_STEMS = ("angu", "ako", "ake", "etu", "enu", "ao")  # my/your(sg)/his-her/our/your(pl)/their
_POSS_PREFIXES = set(POSS_TABLE.values())  # {"w","y","l","ch","vy","z"}


def find_possessive_pronoun(tok: str) -> tuple[str, str] | None:
    """Returns (prefix, person_stem) if `tok` decomposes as a known
    POSS_TABLE prefix + a known person-stem suffix (e.g. 'vyako' ->
    ('vy','ako')), else None. Checked longest-stem-first for the same
    reason every other prefix/suffix match in this project is -- not
    actually ambiguous here (the 6 stems share no suffix relationship),
    but kept consistent with the project's standing discipline."""
    for stem in sorted(POSS_PERSON_STEMS, key=len, reverse=True):
        if tok.endswith(stem):
            prefix = tok[:-len(stem)]
            if prefix in _POSS_PREFIXES:
                return prefix, stem
    return None


# 2026-10-09 (cont. 12): three ZERO-NLLB-COST syntactic control rules for
# '-ki-' (participial/conditional) verb subjects, checked BEFORE falling
# back to the expensive live self-loss check (nllb_probe_identification.
# resolve_ki_subject_via_self_loss). Found by directly inspecting the 185
# raw candidates mine_ki_candidates.py mines from the Bible corpus (not
# assumed): the mined training/silver-label data for this gap is dominated
# by TWO constructions that are NOT genuine subject/object control
# ambiguity at all, but were being treated as such by every prior
# distillation attempt (1-4):
#   - 40/185 (21.6%) are "akisema"/"wakisema"-type reported-speech
#     introductions ("Mungu akavibarikia, akisema, ...") -- inspected
#     several directly: the two candidate nouns mined nearby are from the
#     QUOTED SPEECH that follows, unrelated to who is speaking; the real
#     controller (e.g. "Mungu") is the SAME SUBJECT as the preceding
#     clause and is standard Swahili narrative same-subject chaining, not
#     a genuine 2-noun contrast at all.
#   - another ~31/145 of the remainder are periphrastic "ali(po)kuwa ...
#     akiV" continuous/progressive constructions, where the '-ki-' verb
#     IS the real predicate and MUST share the preceding 'kuwa' auxiliary's
#     own subject by construction, not by inference.
# Mixing these into self-loss-silver-labeled training data as if they were
# genuine 2-candidate contrasts is itself a likely source of attempts
# 3-4's weak selectivity (comparing two candidates when neither is the
# real controller). These two cases are handled here as free, deterministic
# rules instead -- no probe or self-loss needed, and (unlike every marker-
# swapping method tried before) not affected by two candidates sharing one
# subject-marker class, since neither rule compares candidate MARKERS at
# all.
#
# A third rule targets the construction this project's OWN hand-built
# validation data (the mvulana/kalamu case; all 12 items in validate_
# roundtrip_and_attention.py) actually tests, which is rare in the Bible
# corpus (only ~2/145 clear instances, root 'on'/'mwon') but was always
# the motivating case: perception/cognition-verb small-clause complements
# ("saw/found/heard X doing Y"), a standard cross-linguistic OBJECT-
# CONTROL pattern. This rule is a pure SYNTACTIC rule (which argument is
# the object of the governing verb), not a marker-comparison -- so unlike
# every self-loss/probe-based method, it is NOT blocked by mbwa/kijana
# sharing one Swahili subject-marker class (this project's documented
# "structural ceiling").
#
# Evidence tiers, stated honestly per this project's own rules (do not
# round these up to the same confidence):
#   - Rule 3 (perception-verb object control): validated against this
#     project's EXISTING trusted evidence -- the 12-item round-trip-
#     corrected battery (11/12 confirmed) and the live self-loss-
#     validated mvulana/kalamu case all match this rule's prediction
#     exactly, and it is the first mechanism to resolve mbwa/kijana at
#     all (previously an acknowledged, unresolved structural ceiling).
#   - Rules 1/2 (same-subject for reported-speech/periphrastic-aux): a
#     standard, well-motivated Bantu-grammar default (same-subject/
#     switch-reference marking for converbal forms), spot-checked via
#     this project's own round-trip-correction methodology on a real
#     corpus sample below -- NOT yet checked against a native speaker or
#     gold treebank; report accordingly.
SPEECH_ACT_ROOTS = ("sem", "amb", "ulize", "uliz", "arifu", "jibu")
PERCEPTION_ROOTS = ("on", "kut", "siki", "kumbuk")
# 2026-10-10: rule 4's closed list -- the GOVERNOR here is a conjunction,
# not a verb, so there is no root to strip (decompose_verb("wakati")/
# decompose_verb("huku") both return None, confirmed directly). Matched
# against the governor's raw surface FORM instead of a root, since these
# are invariant function words with no SM/TAM/root/FV structure at all.
TEMPORAL_CONCESSIVE_CONJUNCTIONS = ("wakati", "huku")


# 2026-10-23: direct user request -- "build our evaluation metrics...
# agreement accuracy... wrong rewrite / missed slot / spurious rewrite".
# A "slot" here is specifically a CONCORD/AGREEMENT decision point --
# subject marker, adjective/possessive/associative/demonstrative/-ote
# concord -- NOT plain noun/verb-root vocabulary choice (that's the
# separate "vocabulary choice" metric, kitaita_eval_metrics.py).
# `record_slot()` is the single place every concord mechanism in this
# file reports a decision (success, known-failure-with-no-data, or
# "not needed"), so the eval module has one real structured ledger to
# read instead of parsing free-text flags.
@dataclass
class SlotRecord:
    token_idx: int
    slot_type: str  # "subject_sm" | "adjective" | "possessive" | "associative" | "demonstrative" | "ote"
    swahili_surface: str
    target_idx: int | None       # the noun/verb this slot concords WITH, if any
    needs_rewrite: bool          # does Kitaita concord actually differ from a bare passthrough here
    predicted_form: str | None   # the Kitaita form actually produced; None if no rewrite was attempted
    confidence: str              # "verified" | "statistical" | "guessed" | "none" (matches this
                                   # project's existing 3-tier language used throughout its own flags)


def record_slot(d: "Decomposition", token_idx: int, slot_type: str, swahili_surface: str,
                 target_idx: int | None, needs_rewrite: bool, predicted_form: str | None,
                 confidence: str) -> None:
    d.slots.append(SlotRecord(token_idx=token_idx, slot_type=slot_type, swahili_surface=swahili_surface,
                               target_idx=target_idx, needs_rewrite=needs_rewrite,
                               predicted_form=predicted_form, confidence=confidence))


@dataclass
class Decomposition:
    tokens: list[str]
    subject: str | None = None
    subject_idx: int | None = None
    subject_class: str | None = None
    subject_class_confidence: str | None = None
    slots: list = field(default_factory=list)  # list[SlotRecord] -- see record_slot() above
    verb: dict | None = None
    verb_idx: int | None = None
    verb_surface: str | None = None
    object: str | None = None
    object_idx: int | None = None
    adjective: dict | None = None
    adjective_idx: int | None = None
    secondary_nouns: list[tuple] = field(default_factory=list)  # (idx, swahili_lemma, kitaita_or_None)
    passthrough: list[tuple] = field(default_factory=list)      # (idx, original_token_text)
    flags: list[str] = field(default_factory=list)
    coordinated_subject: tuple | None = None  # (primary_lemma, secondary_lemma) when detected -- see
                                                # the "na"-adjacency check in decompose_sentence() below
    # 2026-10-15: set when there is NO noun subject at all but the verb's own
    # subject marker is a recognized PERSON prefix (1sg/2sg/1pl/2pl -- see
    # extract_swahili_grammar.PERSON_SM_TABLE) -- feeds translate(person=...)
    # instead of refusing outright. None for every ordinary noun-subject
    # sentence (unchanged behavior) and for a genuinely unresolvable 3rd-
    # person pro-drop (no coreference mechanism exists here, see the "NO
    # NOUN found" flag below).
    person: str | None = None
    # 2026-10-15: set when the verb itself was recognized as Swahili
    # NEGATIVE-present (extract_swahili_grammar.decompose_negative_verb) --
    # deliberately NEVER paired with a produced kitaita_sentence: no Kitaita
    # negation marker exists in this project's data at all (CLAUDE.md open
    # decisions), so this only drives a clear refusal/flag, never a silently
    # affirmative-looking mistranslation.
    negated: bool = False
    # 2026-10-09, direct user directive: "remove the one verb restriction...
    # we can handle multiple verbs." ONE clause per verb found, each with its
    # OWN subject/object/adjective (a noun CAN be the shared, pro-dropped
    # subject of more than one clause -- not a conflict, just one token
    # position substituted once and reused across clauses that share it).
    # subject/verb/object/adjective above remain the FIRST clause's values,
    # kept for backward compatibility with code that only ever expected one
    # clause (translate_sentence()'s heuristic path, which still has a real
    # single-verb limitation -- see decompose_sentence_with_probes()).
    clauses: list[dict] = field(default_factory=list)


def lookup_noun_kitaita(lemma: str, noun_kitaita: dict, plural_context: bool = False) -> str | None:
    """Plain noun -> Kitaita lookup, same trust order as the subject/object
    path (VERIFIED_NOUN_TRANSLATIONS first, then the statistical table), but
    with NO class/agreement info -- for nouns that don't control any verb
    agreement in this pipeline (a second/coordinated noun, a possessor).
    Returns None if genuinely untranslatable -- the caller passes the
    original Swahili word through unchanged rather than dropping it.

    plural_context: 2026-10-20, direct user report -- "siku" (day) is
    Swahili number-invariant but Kitaita uses a DIFFERENT WORD for plural
    ("matuku") vs singular ("ituku"), not just a different subject
    marker. Callers that know the immediate context marks "siku" as
    plural (currently: the token right after it is "zote" -- see
    SIKU_PLURAL_TRIGGER's own comment in kitaita_verified_corrections.py
    for the evidence and scope) pass True here to get "matuku" instead
    of the unconditional "ituku" default."""
    if lemma == "siku" and plural_context:
        return SIKU_PLURAL_TRIGGER
    if lemma in VERIFIED_NOUN_TRANSLATIONS:
        return VERIFIED_NOUN_TRANSLATIONS[lemma]
    cand = noun_kitaita.get(lemma)
    return cand["kitaita_word"] if cand else None


def _siku_plural_context(tokens: list[str], idx: int) -> bool:
    """See lookup_noun_kitaita's plural_context docstring -- factored out
    so every call site that has (tokens, idx) in scope checks the SAME
    one trigger, consistently."""
    return tokens[idx] == "siku" and idx + 1 < len(tokens) and tokens[idx + 1] == "zote"


def resolve_class_marker(kitaita_noun: str, noun_class_of: dict) -> str | None:
    """Same resolution order kitaita_sentence_generator.translate() already
    uses for the SUBJECT's own marker (VERIFIED_SUBJECT_MARKERS, then
    DEVERBAL_SM_HYPOTHESIS, then the corrected-class table) -- factored out
    here so it can be applied to ANY noun, not just the subject. Needed for
    2026-10-09's probe-identified adjective targets: the amod probe can
    correctly say an adjective modifies the OBJECT (or another noun), not
    just the subject, so computing that adjective's Kitaita form needs
    THAT noun's own class marker, which translate() itself never computes
    for anything but the subject."""
    if kitaita_noun in VERIFIED_SUBJECT_MARKERS:
        return VERIFIED_SUBJECT_MARKERS[kitaita_noun]
    if kitaita_noun in DEVERBAL_SM_HYPOTHESIS:
        return DEVERBAL_SM_HYPOTHESIS[kitaita_noun]
    class_info = noun_class_of.get(kitaita_noun)
    return class_info["subject_marker"] if class_info else None


def predict_kitaita_class_from_prefix(kitaita_noun: str, noun_class_of: dict,
                                        min_count: int = 4, min_purity: float = 0.75
                                        ) -> tuple[str | None, str | None]:
    """2026-10-23, direct user suggestion ("umbali" is the same class as
    "wughoma"/mercy; "we can make simple class prediction based off of the
    first few letters (2-3) of a word if no farther information is
    known") -- the LOWEST-confidence tier in this project's class-
    resolution chain, tried only after VERIFIED_SUBJECT_MARKERS/
    DEVERBAL_SM_HYPOTHESIS/noun_class_of AND the Swahili-side
    SWAHILI_CLASS_DEFAULT_SM_GUESS fallback have all failed (i.e. there is
    truly no grammatical information about this specific noun anywhere).

    Checked directly before building this (counted across all 503
    classified entries in noun_class_of): Kitaita's own noun-class
    prefixes are REAL word-shape signals, not noise, though strength
    varies by prefix and granularity -- the bare 2-letter "wa-" (class-2
    plural-human shape, matching Swahili's own "wa-") is 57/82 (70%)
    class 'wi' on its own, below this function's default purity bar, but
    its 3-letter sub-groups are much cleaner ('wak-' 10/11, 'wad-' 5/5,
    'wal-' 7/8, all class 'wi'); same story for "ma-" (62/86 = 72% class
    'gh' at 2 letters; 'maw-' 6/6, 'mak-' 8/9 at 3). "wu-" (Kitaita's
    abstract/quality-noun shape -- wukelu/kindness, wuzuri/goodness,
    wuwiwi/evil, wulacha/length) is the one 2-letter prefix clean enough
    to clear the bar on its own, 32/41 (78%) class 'gh'; its 3-letter
    sub-group "wuk-" (wukelu/wukiwa/wukomboli/wukongo) is cleaner still,
    4/4. Tries the 3-letter prefix FIRST (more specific, checked when
    there's enough data), falls back to 2-letter. Returns (None, None)
    rather than a weak guess when neither prefix clears `min_count`/
    `min_purity` -- an UNCONFIRMED GUESS from word shape alone,
    explicitly not the same confidence tier as any of the real
    grammatical-evidence tiers above it; the caller must flag it as
    such, this function does not do that itself."""
    for plen in (3, 2):
        if len(kitaita_noun) < plen:
            continue
        prefix = kitaita_noun[:plen]
        counter = Counter(info["subject_marker"] for word, info in noun_class_of.items()
                           if word != kitaita_noun and word.startswith(prefix) and info.get("subject_marker"))
        total = sum(counter.values())
        if total < min_count:
            continue
        top_sm, top_n = counter.most_common(1)[0]
        purity = top_n / total
        if purity >= min_purity:
            return top_sm, (f"word-shape GUESS: prefix '{prefix}-' -> class '{top_sm}-' in {top_n}/{total} "
                             f"({purity:.0%}) of this project's OWN already-classified Kitaita nouns sharing "
                             f"that prefix -- NOT a confirmed class for this specific noun")
    return None, None


# 2026-10-15: Swahili associative/possessive particles ("ya"/"wa"/"cha"/
# "la"/"za"/"vya" -- "X ya Y" = "Y's X"/"X of Y"). Module-level (was a local
# inside decompose_sentence()) so the new associative-concord logic below
# can share it with both decompose_sentence() and decompose_sentence_
# with_probes().
ASSOC_PARTICLES = {"cha", "wa", "ya", "la", "za", "vya"}


def decompose_verb_or_copula(tok: str) -> VerbFormEntry | None:
    """decompose_verb() plus one special case: Swahili "ni" (the invariant
    present-tense copula, "is/are/am" -- "X ni Y" = "X is Y"). 2026-10-15,
    direct user report: "ni" was simply dropped as an unrecognized
    passthrough token, even though this project already has a verified
    Kitaita copula mechanism -- VERIFIED_KUWA_ROOT_COPULA ("dek"), used
    whenever swahili_verb['root']=='kuw' and an adjective follows
    (kitaita_sentence_generator.translate()'s own existing branch, ZERO
    TAM always, confirmed on odeka/edeka/redeka/jideka/ghodeka/kodeka).
    "ni" is a DIFFERENT lexeme from "kuwa" (not one of its inflected
    forms -- Swahili's affirmative present copula is itself irregular,
    carrying no subject agreement of its own, unlike every "kuwa"-based
    tense), but semantically it fills the exact same present-tense copula
    slot -- so it's remapped onto the SAME root='kuw' mechanism rather
    than inventing a parallel one. tam_label is nominally 'present' (it
    doesn't actually matter which: the copula+adjective branch fires on
    root=='kuw' alone, regardless of tam) and subject_marker is None
    (true to Swahili: "ni" itself never carries agreement; the REAL
    subject-marker comes from whichever noun the probe/adjacency
    identifies as "ni"'s subject, same as any other verb)."""
    if tok == "ni":
        return VerbFormEntry(surface=tok, subject_marker=None, tam_marker="na", tam_label="present",
                              root_guess="kuw", final_vowel="a", confidence="reference-grammar")
    if tok in NA_HAVE_FORMS:
        return _na_have_verb_form_entry(tok)
    if tok in EXISTENTIAL_LOCATIVE_FORMS:
        return VerbFormEntry(surface=tok, subject_marker=None, tam_marker="na", tam_label="present",
                              root_guess="oko_exist", final_vowel="a", confidence="reference-grammar")
    if tok in PLURAL_HAVE_FORMS:
        return VerbFormEntry(surface=tok, subject_marker="wa", tam_marker="na", tam_label="present",
                              root_guess="ko_have", final_vowel="a", confidence="reference-grammar")
    return decompose_verb(tok)


# 2026-10-22: bug #2 from the African Storybook error report -- Swahili's
# locative-existential "yuko"/"yupo"/"yumo" ("X is [located] at Y") was
# simply unrecognized as a verb at all ("Yuko wapi paka wangu?" -> "NO
# VERB found -- cannot translate."), the same shape of gap "ni" had
# before its own fix above. Reverse-inference search of the full parallel
# Bible corpus (grep for "yuko"/"yupo"/"yumo" + nearby SUV/TAITA verse
# pairs, `bible collection/data/parallel_full_suv/parallel.jsonl`) found
# an extremely consistent 3rd-person-singular equivalent: Kitaita "oko"
# (n>=15 independent verses, zero exceptions). IMPORTANT CAVEAT, found
# only on re-reading the evidence afterward: every single tested subject
# -- Mungu/BWANA, Elisha, Petro, nabii, mjakazi, kahaba, mtu, proper
# names -- happens to be Swahili class 1/ANIMATE-SG. This project's own
# durable finding that Swahili agreement follows ANIMACY (kipepeo/paka/
# nyoka all take class-1 concord) means "paka" (the subject this fix was
# actually needed for, "Yuko wapi paka wangu?") is ALSO class 1 -- so
# "oko" is correct for every sentence this fix targets. But "oko" is
# treated here as a fully INVARIANT literal form (root_is_full_form),
# which is a STRONGER claim than the evidence supports: it has never
# been tested against a genuinely non-animate-class subject (e.g. "iko"
# for a class-'i' noun?), so do NOT reuse "oko" literally for a
# different noun class without checking first -- this may really be
# "u-" (class 1's own SM) + a root "ko", coincidentally always surfacing
# the same way because every tested case shares that one class. 2nd-
# person "Uko wapi?" -> "Koko hao?" shows a DIFFERENT, SM-bearing shape
# ("k-oko"), consistent with that reading -- deliberately not
# generalized to here from a single data point.
EXISTENTIAL_LOCATIVE_FORMS = {"yuko", "yupo", "yumo"}


# 2026-10-22: vocabulary-gap fix, direct user request ("solve #2... as
# well as the vocabulary gaps") -- NA_HAVE_FORMS's own comment already
# flagged that 1pl/2pl/3pl "have" ("tuna"/"mna"/"wana") were deliberately
# left unhandled ("flag, don't guess the rest of a paradigm from 3
# cells"). Reverse-inference search of the full parallel Bible corpus
# for "tuna"/"wana" + a clean abstract-noun object (amani/nguvu/akili/
# haki/sehemu/njaa) found this is NOT a simple extension of the singular
# "n"-root paradigm: plural "have" uses the EXISTENTIAL root "ko" (same
# root as "oko" above) with a REGULAR class subject marker, e.g. 1pl
# "Tuna sehemu gani" -> "Deko na ifunguki" ('di'+'ko'+'na'), 3pl "Wana
# amani nyingi" -> "weko na sere mbaa" (PSA.119.165), "wana nguvu" ->
# "wiko na ndighi" (EXO.1.9) -- 'we'/'wi' spelling variation attested in
# different verses (same noise this project already documents for the
# 'wi'/'wa' present-tense shift elsewhere), not resolved further. ONLY
# the AFFIRMATIVE 3rd-person-plural form ("wana") is wired in here --
# negative "hawana"/1pl "tuna"/2pl "mna" are NOT (no clean corpus
# evidence found for a negative plural-have form at all: the one 1pl
# negative example found, "hatuna", showed a THIRD, unrelated root
# ("wadie"/"dine", inconsistent between two near-duplicate verses) --
# flagged, not guessed). "wana" is ALSO this project's extremely common
# NOUN for "children" -- the existing pos_tags[i]=="NOUN" veto (same
# gate protecting "ukimwi"/"wakimbizi") and noun_hits claiming "wana"
# first whenever the noun lexicon recognizes it are this fix's ONLY
# protection against that collision, same risk tier NA_HAVE_FORMS
# already accepted for its own cells.
PLURAL_HAVE_FORMS = {"wana"}


# 2026-10-22: direct native-speaker correction -- Swahili "kuwa na" (to
# have) is a SECOND irregular/defective verb, same general shape as "ni"
# above: "ana"/"una"/"nina" (has/have you/I have) fuse the subject
# marker directly onto the invariant particle "na", with NO separate TAM
# syllable and NO further root+FV split the way a regular verb has.
# Crucially, the NEGATIVE forms ("hana"/"huna"/"sina") are ALSO
# irregular in a way decompose_negative_verb() can't catch: that
# function requires the final vowel to be literally '-i' (the one
# reliable signal that a word is negative-present at all, e.g.
# "ha-pend-i") -- but "hana" etc. keep the SAME '-a' ending as the
# affirmative, exactly mirroring Swahili's own irregularity here. Kept
# as its own small closed-set table (not a generalized pattern-match)
# for exactly the reason decompose_negative_verb() itself gives: don't
# pattern-match an ending this short and this ambiguous. 3rd person
# ("ana"/"hana") deliberately gets NO `person` value -- unlike 1st/2nd,
# 3rd person in this project's scheme is never a generic pro-drop slot,
# it is resolved from a real noun/pronoun subject (here, "papa",
# continued via the existing clause-chain fallback) -- see translate()'s
# own "na_have" branch (kitaita_sentence_generator.py) for how the
# Kitaita side assembles this (root "n", STAYS at final vowel '-a' even
# when negated -- the Swahili-mirroring irregularity carries over).
# 1pl/2pl/3pl ("tuna"/"mna"/"wana", "hatuna"/"hamna"/"hawana") were NOT
# given directly and are deliberately NOT added here -- flag, don't
# guess the rest of a paradigm from 3 cells.
NA_HAVE_FORMS = {
    "nina": {"person": "1sg", "negated": False},
    "una": {"person": "2sg", "negated": False},
    "ana": {"person": None, "negated": False},
    "sina": {"person": "1sg", "negated": True},
    "huna": {"person": "2sg", "negated": True},
    "hana": {"person": None, "negated": True},
}


def _na_have_verb_form_entry(tok: str) -> VerbFormEntry:
    info = NA_HAVE_FORMS[tok]
    return VerbFormEntry(surface=tok, subject_marker=None, tam_marker="", tam_label="present",
                          root_guess="na_have", final_vowel="a", confidence="reference-grammar",
                          person=info["person"], negated=info["negated"])


def associative_particle_for_class(kit_class: str | None) -> tuple[str | None, str | None]:
    """Kitaita associative ('of') concord for the GOVERNING noun's class --
    2026-10-15, direct native-speaker corrections: "mayai ya kuku" ->
    "maghi gha nguku" ('ghi' class) and "upendo wa Mungu" -> "lukundo lwa
    Mlungu" ('lu' class). Until that point this project's pipeline had NO
    Kitaita associative-concord mechanism at all (kept the Swahili particle
    literally, by design -- see the old comment this replaces).

    Two-tier lookup, checked in order:
    1. KITAITA_ASSOCIATIVE_VERIFIED (kitaita_verified_corrections.py) --
       directly-submitted answers, trusted at n=1 like every other
       VERIFIED_* table. Currently 'ghi'->'gha', 'lu'->'lwa'. 'lu'->'lwa' is
       NOT predictable from tier 2 below (that would give 'la', not 'lwa' --
       see the table's own comment for why) -- direct confirmation was
       necessary, not just nice to have.
    2. PRESENT_SM_VOWEL_CONFIRMED (this class's own already-verified
       PRESENT-TENSE-shifted subject marker -- "consonant stays fixed,
       vowel becomes 'a'") as a HYPOTHESIS for classes with no direct
       associative data of their own. Confirmed to coincide with tier 1
       for 'ghi' ('gha' both ways) but NOT for 'lu' (see above) -- so this
       tier's answers for any class NOT in tier 1 remain unconfirmed
       extrapolations, not assumed correct just because 'ghi' matched.

    Returns (particle, tier) -- tier is "verified" (tier 1), "hypothesis"
    (tier 2), or (None, None) when NEITHER has data for this class (flagged
    by the caller, never guessed)."""
    if kit_class in KITAITA_ASSOCIATIVE_VERIFIED:
        return KITAITA_ASSOCIATIVE_VERIFIED[kit_class], "verified"
    normalized = TAM_SM_ALIAS.get(kit_class, kit_class) if kit_class else None
    particle = PRESENT_SM_VOWEL_CONFIRMED.get(normalized) if normalized else None
    if particle is None:
        return None, None
    return particle, "hypothesis"


def associative_descriptor_particle_for_class(kit_class: str | None) -> tuple[str | None, str | None]:
    """Kitaita associative concord for the DESCRIPTIVE/PURPOSIVE "[noun] -a
    ku-VERB" construction ("water FOR DRINKING"), NOT the genitive/
    possessive one `associative_particle_for_class()` handles ("water OF
    [someone]") -- 2026-10-19, direct native-speaker correction: these take
    a DIFFERENT particle allomorph (confirmed class 'ghi': descriptive
    'ghe' vs. genitive 'gha' -- "maji ya kunywa" -> "machi ghenywa" but
    "maji ya Bill" stays "machi gha Bill", same noun/class, same Swahili
    particle, different Kitaita answer because the COMPLEMENT differs).
    See KITAITA_ASSOCIATIVE_DESCRIPTOR_VERIFIED's own comment (kitaita_
    verified_corrections.py) for why this is NOT extrapolated from the
    genitive table or from any formula -- n=1, one class only, no tier-2
    hypothesis fallback (unlike associative_particle_for_class(), which
    does have one: there is no independently-verified RELATED table to
    extrapolate this specific allomorph from for an untested class).

    Returns (particle, "verified") or (None, None) when this class has no
    data at all (flagged by the caller, never guessed)."""
    particle = KITAITA_ASSOCIATIVE_DESCRIPTOR_VERIFIED.get(kit_class) if kit_class else None
    return (particle, "verified") if particle else (None, None)


def apply_associative_descriptor_concord(d: "Decomposition", noun_kitaita: dict, noun_class_of: dict,
                                          root_tables, assigned: list) -> None:
    """Scans for "NOUN PARTICLE ku-VERB" (associative particle immediately
    followed by a Swahili INFINITIVE, e.g. "maji ya kunywa") and, when the
    governing NOUN's class has descriptive-associative data
    (associative_descriptor_particle_for_class()), fuses the particle and
    the verb's ROOT into ONE Kitaita word -- the infinitive's own "ku-" is
    DROPPED, not kept or translated separately: "ya" + "kunywa" -> Kitaita
    particle "ghe" + root "nywa" (resolved via resolve_verb_root(), the
    SAME root-translation chain translate() itself uses for any other
    verb, so an untranslated root still gets this project's Taita-fy
    fallback rather than silently failing) = "ghenywa", ONE token, not two.

    MUST run BEFORE apply_associative_concord() (the plain genitive case)
    and claim BOTH the particle's index AND the infinitive's index in
    `assigned` on a real match, so that function's own guard skips this
    particle instead of also emitting its (wrong, genitive) answer for the
    same index -- see that function's docstring for the matching guard.

    A particle not immediately followed by something shaped like an
    infinitive (e.g. "maji ya Bill") is left completely untouched here
    (falls through to apply_associative_concord exactly as before) --
    this function only ever fires on the specific "-a ku-VERB" shape,
    never on ownership/possession.

    2026-10-20, direct user examples ("mbwa wa kulinda" -> "koshi
    elindia") caught a real bug in an earlier version of this check:
    it used extract_swahili_grammar.decompose_verb() to validate the
    infinitive, which also tries to match a TAM-marker-shaped substring
    even for 'ku-' (that function's own general-purpose ambiguity --
    'ku' is EITHER tenseless infinitive OR a class-15 noun's finite
    SUBJECT marker, and only the latter can combine with a TAM). THIS
    syntactic position -- immediately governed by an associative
    particle -- can ONLY be the bare infinitive reading; there is no
    class-15-subject ambiguity here at all (that reading needs its OWN
    independent finite clause, not a particle's complement). So
    decompose_verb() wrongly matched 'li' as the past-TAM inside
    "kulinda" (a real, separate root-vs-TAM-prefix collision, same
    class of bug as 'ka' inside 'kabili-' -- see 2026-10-19's verb-root
    Taita-fy fix), giving root 'nd' instead of the correct 'lind'. Fixed
    by NOT calling decompose_verb() here at all -- this position's
    grammar already disambiguates it, so just strip 'ku-' and the final
    vowel directly, a strictly simpler and MORE correct check for this
    one fixed construction than the general-purpose decomposer."""
    for i, tok in enumerate(d.tokens):
        if tok not in ASSOC_PARTICLES or i == 0 or i + 1 >= len(d.tokens) or assigned[i]:
            continue
        next_tok = d.tokens[i + 1]
        if assigned[i + 1] or not next_tok.startswith("ku") or len(next_tok) < 5:
            continue
        rest = next_tok[2:]
        if rest[-1] not in "aeiou":
            continue
        swahili_root, final_vowel = rest[:-1], rest[-1]
        governor_sw = d.tokens[i - 1]
        governor_kit = lookup_noun_kitaita(governor_sw, noun_kitaita)
        governor_class = resolve_class_marker(governor_kit, noun_class_of) if governor_kit else None
        particle, tier = associative_descriptor_particle_for_class(governor_class)
        if particle is None:
            if governor_class:
                d.flags.append(f"'{tok} {next_tok}' (descriptive associative, after '{governor_sw}', "
                                f"class '{governor_class}-'): no Kitaita descriptive-associative-concord "
                                f"data for this class -- FLAGGED, kept as the original Swahili words.")
            continue
        kit_root, root_tier, is_full_form = resolve_verb_root(swahili_root, root_tables)
        if is_full_form:
            # 2026-10-19: the "per-form-fallback" tier returns an ALREADY-
            # INFLECTED full Kitaita word built around some OTHER, unrelated
            # subject+TAM combination (translate()'s own root_is_full_form
            # case) -- categorically incompatible with this construction's
            # "particle + BARE root + FV" template, not just lower-confidence.
            # Splicing it in would produce a word with no coherent internal
            # structure, worse than just keeping the Swahili root. Treated the
            # same as "no candidate at all" here, NOT used.
            kit_root, root_tier = swahili_root, "untranslated"
        fused = f"{particle}{kit_root}{final_vowel}"
        d.secondary_nouns.append((i, tok, fused))
        assigned[i] = True
        assigned[i + 1] = True
        gap_note = ", NO Kitaita translation -- vocabulary gap, not a parsing failure" \
            if root_tier == "untranslated" else ""
        d.flags.append(f"'{tok} {next_tok}' (descriptive associative, after '{governor_sw}') -> Kitaita "
                        f"'{fused}' (particle '{particle}-' [{tier}, class '{governor_class}-'] fused with "
                        f"verb root '{swahili_root}' [{root_tier}{gap_note}]; see kitaita_e2e_pipeline."
                        f"associative_descriptor_particle_for_class)")
        if root_tier == "untranslated":
            d.flags.append(f"Verb root '{swahili_root}' (in '{tok} {next_tok}') has NO Kitaita "
                            f"translation -- the descriptive-associative word above keeps the SWAHILI "
                            f"root, 'Taita-fied' only with the Kitaita particle. Insufficient vocabulary "
                            f"coverage, not a parsing failure.")


# 2026-10-20: Swahili "-ote" (all/whole) -- wote/yote/vyote/chote/kote/
# lote/pote/zote, a closed set (see OTE_STEM's own comment in kitaita_
# verified_corrections.py for the full evidence trail). Recognized by
# surface form alone (no prefix/stem split needed -- the KITAITA output
# comes entirely from the GOVERNING noun's own class, same as the
# associative particle just below, never from the Swahili word's own
# prefix letter).
OTE_TOKENS = {"wote", "yote", "vyote", "chote", "kote", "lote", "pote", "zote"}


def ote_form_for_class(kit_class: str | None) -> tuple[str | None, str | None]:
    """Kitaita '-ote' (all/whole) concord for the GOVERNING noun's class --
    see OTE_STEM's own comment (kitaita_verified_corrections.py) for the
    evidence. Formula: the noun's own class marker with its FINAL VOWEL
    ELIDED if it has one (kit_sm.rstrip('aeiou'), falling back to the
    unreduced marker if that would empty it out entirely -- a purely
    vocalic class like 'u' has nothing left to reduce) + OTE_STEM.
    Returns (form_or_None, tier) like associative_particle_for_class."""
    if not kit_class:
        return None, None
    reduced = kit_class.rstrip("aeiou") or kit_class
    confirmed = kit_class in OTE_STEM_CONFIRMED_CLASSES
    return f"{reduced}{OTE_STEM}", ("CONFIRMED class" if confirmed else "EXTRAPOLATED to an untested class")


def apply_ote_concord(d: "Decomposition", noun_kitaita: dict, noun_class_of: dict,
                       assigned: list) -> None:
    """Scans for "NOUN -OTE" (e.g. "siku zote", "vilambo vyote") the same
    way apply_associative_concord scans for "NOUN1 PARTICLE NOUN2" --
    GOVERNING noun immediately before the -ote token, verb-independent,
    called BEFORE apply_associative_concord (which needs to know an -ote
    token right before an associative particle is NOT itself the real
    governor -- see that function's governor-lookup skip)."""
    for i, tok in enumerate(d.tokens):
        if tok not in OTE_TOKENS or i == 0 or assigned[i]:
            continue
        governor_sw = d.tokens[i - 1]
        plural_ctx = _siku_plural_context(d.tokens, i - 1)
        governor_kit = lookup_noun_kitaita(governor_sw, noun_kitaita, plural_context=plural_ctx)
        governor_class = resolve_class_marker(governor_kit, noun_class_of) if governor_kit else None
        form, tier = ote_form_for_class(governor_class)
        if form:
            d.secondary_nouns.append((i, tok, form))
            assigned[i] = True
            d.flags.append(f"'{tok}' (-ote/all, after '{governor_sw}') -> Kitaita '{form}' "
                            f"({tier} -- '{governor_sw}' is class '{governor_class}-'; see "
                            f"kitaita_e2e_pipeline.ote_form_for_class)")
        elif governor_class:
            d.flags.append(f"'{tok}' (-ote/all, after '{governor_sw}', class '{governor_class}-'): "
                            f"no Kitaita -ote-concord data for this class -- FLAGGED, kept as the "
                            f"original Swahili word.")
        record_slot(d, i, "ote", tok, i - 1, needs_rewrite=True, predicted_form=form,
                    confidence=tier or ("known-class-no-data" if governor_class else "class-unresolved"))


def apply_associative_concord(d: "Decomposition", noun_kitaita: dict, noun_class_of: dict,
                               assigned: list, tokens: list[str] | None = None) -> None:
    """Scans the whole sentence for "NOUN1 PARTICLE NOUN2" associative/
    possessive constructions (PARTICLE in ASSOC_PARTICLES) and, when the
    GOVERNING noun (immediately before the particle) resolves to a known
    Kitaita class, replaces the particle with the correct Kitaita
    associative form (associative_particle_for_class()) instead of leaving
    it as the literal Swahili word. Appends to `d.secondary_nouns` (same
    placement mechanism every other independently-substituted word already
    uses) rather than touching `d.subject`/`d.object`/clause structure --
    this runs ONCE, generically, over the raw token sequence, independent
    of whether the sentence has a recognized verb at all (an associative
    noun phrase like "mayai ya kuku" needs no verb to be meaningful) and
    independent of which of the two decompose_* functions is calling it.
    Called BEFORE the passthrough sweep in both (which it also marks
    `assigned[i]=True` for on a real substitution, so the particle doesn't
    ALSO land in `d.passthrough` and produce a stale, contradictory
    "no mechanism" flag alongside the correct substitution) -- an
    unresolved particle is left unmarked and still correctly falls through
    to passthrough afterward, unchanged.

    `tokens` (ORIGINAL-CASE, optional): 2026-10-22, direct user report --
    "Sifuna wa sita" ('wa' needing the GOVERNING noun's class) failed
    entirely because "Sifuna" (a proper name, not in any noun table) was
    never resolvable at all. When `tokens` is supplied, an unresolved
    governor falls back to resolve_noun_kitaita_and_class()'s proper-
    name handling (capitalized, no Bantu prefix shape -> Kitaita class
    'u', same as mundu) before giving up; omitted (None), behavior is
    IDENTICAL to before this fix, for any caller not yet updated to
    pass it."""
    for i, tok in enumerate(d.tokens):
        if tok not in ASSOC_PARTICLES or i == 0 or i + 1 >= len(d.tokens):
            continue
        if assigned[i]:
            # 2026-10-19: already claimed by apply_associative_descriptor_concord()
            # (the "-a ku-VERB" descriptive/purposive construction, a DIFFERENT
            # particle allomorph fused with the verb root, not this plain
            # genitive one) -- see that function's docstring. Without this guard
            # the two functions would both emit a secondary_nouns entry at the
            # SAME index i, producing a doubled/contradictory particle in the
            # output. That function runs first and claims `i` only on an actual
            # match, so skipping here never drops a genuine genitive case.
            continue
        # 2026-10-20: direct user report -- "siku zote za maisha" needs 'za's
        # governor to be "siku" (the noun 'zote' itself modifies), not "zote"
        # (an -ote quantifier, not a noun at all). apply_ote_concord() runs
        # BEFORE this function and claims 'zote' (assigned[i-1] True) whenever
        # IT resolves a governor -- when that's the case, the real governor is
        # one token further back.
        governor_idx = i - 1
        if d.tokens[governor_idx] in OTE_TOKENS and governor_idx > 0 and assigned[governor_idx]:
            governor_idx -= 1
        governor_sw = d.tokens[governor_idx]
        plural_ctx = _siku_plural_context(d.tokens, governor_idx)
        if tokens is not None:
            governor_kit, governor_class = resolve_noun_kitaita_and_class(
                governor_sw, governor_idx, tokens, noun_kitaita, noun_class_of, plural_context=plural_ctx)
        else:
            governor_kit = lookup_noun_kitaita(governor_sw, noun_kitaita, plural_context=plural_ctx)
            governor_class = resolve_class_marker(governor_kit, noun_class_of) if governor_kit else None
        particle, tier = associative_particle_for_class(governor_class)
        name_fallback = tokens is not None and governor_kit == tokens[governor_idx] \
            and is_likely_proper_name(tokens[governor_idx], governor_sw)
        if particle:
            d.secondary_nouns.append((i, tok, particle))
            assigned[i] = True
            d.flags.append(f"'{tok}' (associative, after '{governor_sw}') -> Kitaita '{particle}-' "
                            f"({tier} -- '{governor_sw}' is class '{governor_class}-'"
                            + (", PROPER NAME fallback (capitalized, no Bantu prefix shape, treated as "
                               "mundu's class)" if name_fallback else "")
                            + f"; see kitaita_e2e_pipeline.associative_particle_for_class)")
        elif governor_class:
            d.flags.append(f"'{tok}' (associative, after '{governor_sw}', class '{governor_class}-'): "
                            f"no Kitaita associative-concord data for this class -- FLAGGED, kept as "
                            f"the original Swahili word.")
        record_slot(d, i, "associative", tok, governor_idx, needs_rewrite=True,
                    predicted_form=particle,
                    confidence=tier or ("known-class-no-data" if governor_class else "class-unresolved"))


# 2026-10-15: Swahili demonstratives ("this"/"that") -- closed-set token
# detection, SAME token sets nllb_concord_rewrite_pipeline's DEM_PROXIMAL_
# TABLE/DEM_REFERENTIAL_TABLE/DEM_DISTAL_TABLE already define (Swahili-
# side, reused rather than re-listed), so a word is recognized here iff
# it's a demonstrative that table already knows about. THREE degrees, not
# two -- direct correction: proximal ("hii"), referential ("hiyo"), distal
# ("ile").
SWAHILI_DEM_PROXIMAL = set(DEM_PROXIMAL_TABLE.values())
SWAHILI_DEM_REFERENTIAL = set(DEM_REFERENTIAL_TABLE.values())
SWAHILI_DEM_DISTAL = set(DEM_DISTAL_TABLE.values())
_DEM_VERIFIED_BY_DEGREE = {"proximal": KITAITA_DEM_PROXIMAL_VERIFIED,
                            "referential": KITAITA_DEM_REFERENTIAL_VERIFIED,
                            "distal": KITAITA_DEM_DISTAL_VERIFIED}


def kitaita_demonstrative_for_class(kit_class: str | None, degree: str) -> tuple[str | None, str | None]:
    """Kitaita demonstrative ('this'/'that') concord for a noun's class --
    2026-10-15, direct native-speaker correction: "hii kazi" (this work,
    'kazi' = class 'i') -> Kitaita "Ihi kazi". Until now this project had
    NO Kitaita demonstrative-concord mechanism at all -- see
    KITAITA_DEM_PROXIMAL_VERIFIED's own comment (kitaita_verified_
    corrections.py) for why this is the same class of gap as associative
    concord and object markers. Simple direct-table lookup (no present-
    tense-shift-style extrapolation attempted here -- unlike associative
    concord, there's no independently-verified related table to
    extrapolate FROM yet). `degree` is "proximal"/"referential"/"distal".
    Returns (word, "verified") or (None, None) -- flagged by the caller,
    never guessed."""
    table = _DEM_VERIFIED_BY_DEGREE[degree]
    word = table.get(kit_class) if kit_class else None
    return (word, "verified") if word else (None, None)


def apply_demonstrative_concord(d: "Decomposition", noun_kitaita: dict, noun_class_of: dict,
                                 assigned: list) -> None:
    """Scans the whole sentence for a Swahili demonstrative token adjacent
    to a recognized noun -- Swahili allows BOTH "DEM NOUN" ("hii kazi")
    and "NOUN DEM" ("kazi hii") order, so the noun right AFTER the
    demonstrative is tried first (the more common order), the noun right
    BEFORE it second -- and, when that noun's Kitaita class resolves,
    replaces the demonstrative with the correct Kitaita form instead of
    leaving it as literal Swahili. Same `d.secondary_nouns`-placement /
    `assigned[i]=True` mechanism as apply_associative_concord() (see its
    own docstring for why) -- deliberately a separate function rather
    than generalizing both into one, since demonstratives and associative
    particles differ in which side the governing noun sits on."""
    for i, tok in enumerate(d.tokens):
        if tok in SWAHILI_DEM_PROXIMAL:
            degree = "proximal"
        elif tok in SWAHILI_DEM_REFERENTIAL:
            degree = "referential"
        elif tok in SWAHILI_DEM_DISTAL:
            degree = "distal"
        else:
            continue
        neighbor_sw = None
        if i + 1 < len(d.tokens):
            neighbor_sw = d.tokens[i + 1]
        if (neighbor_sw is None or lookup_noun_kitaita(neighbor_sw, noun_kitaita) is None) and i > 0:
            neighbor_sw = d.tokens[i - 1]
        neighbor_kit = lookup_noun_kitaita(neighbor_sw, noun_kitaita) if neighbor_sw else None
        neighbor_class = resolve_class_marker(neighbor_kit, noun_class_of) if neighbor_kit else None
        word, tier = kitaita_demonstrative_for_class(neighbor_class, degree)
        if word:
            d.secondary_nouns.append((i, tok, word))
            assigned[i] = True
            d.flags.append(f"'{tok}' (demonstrative, near '{neighbor_sw}') -> Kitaita '{word}' "
                            f"({tier} -- '{neighbor_sw}' is class '{neighbor_class}-'; see "
                            f"kitaita_e2e_pipeline.kitaita_demonstrative_for_class)")
        elif neighbor_class:
            d.flags.append(f"'{tok}' (demonstrative, near '{neighbor_sw}', class '{neighbor_class}-'): "
                            f"no Kitaita demonstrative-concord data for this class/series -- FLAGGED, "
                            f"kept as the original Swahili word.")
        neighbor_idx = (i + 1 if neighbor_sw == (d.tokens[i + 1] if i + 1 < len(d.tokens) else None) else i - 1)
        record_slot(d, i, "demonstrative", tok, neighbor_idx, needs_rewrite=True, predicted_form=word,
                    confidence=tier or ("known-class-no-data" if neighbor_class else "class-unresolved"))


def compute_adjective_concord(kitaita_noun: str, swahili_noun: str, stem: str, noun_class_of: dict,
                               noun_kitaita: dict | None = None,
                               override_class: str | None = None) -> tuple[str | None, str]:
    """Generalized version of translate()'s 'baya'/'kubwa' adjective-concord
    branches, for a target noun that is NOT the sentence's primary subject
    (translate() itself only ever computes this for the subject). Reuses
    the EXACT SAME rules (same tables, same generalized-rule conditions),
    just keyed by whichever noun the amod probe says the adjective
    actually modifies. Returns (kitaita_form_or_None, explanation).

    `override_class` (2026-10-22): when the caller has ALREADY resolved
    the target's class through some mechanism other than noun_class_of
    (e.g. resolve_noun_kitaita_and_class()'s proper-name fallback --
    "Sifuna" has no noun_class_of entry at all, but should still concord
    as class 'u'/mundu), it's used in place of the internal resolve_
    class_marker() lookup. None (default): identical behavior to before
    this parameter existed."""
    # 2026-10-13 (cont. 3): `kit_class` is the noun's REAL class (may be
    # "i_nasal") -- used for the DICT-based "special paradigm" adjectives
    # (kubwa/refu/dogo/geni/fupi/zito/chafu), which genuinely see the mi-/
    # nd- split. `kit_sm` is TAM_SM_ALIAS-normalized (i_nasal -> i) -- used
    # for 'baya'/'zuri', which use the BARE subject marker uniformly and
    # do NOT see the split (see TAM_SM_ALIAS's comment in kitaita_verified_
    # corrections.py for why nyumba's native-speaker-verified baya,
    # "izamie", is correct despite nyumba being class 2/"i_nasal").
    kit_class = override_class if override_class is not None else resolve_class_marker(kitaita_noun, noun_class_of)
    kit_sm = TAM_SM_ALIAS.get(kit_class, kit_class) if kit_class else kit_class
    adj_class = kit_class
    if stem == "baya" and kitaita_noun in NUMBER_INVARIANT_NOUNS:
        return (f"{NUMBER_INVARIANT_SM_SINGULAR['default']}{BAYA_STEM}",
                "number-invariant noun, defaulted to singular prefix (same rule as the subject case)")
    if stem == "baya" and kit_sm:
        return (f"{kit_sm}{BAYA_STEM}", f"generalized 'baya' rule, class marker '{kit_sm}-'")
    adj_key = (stem, swahili_noun)
    if adj_key in VERIFIED_ADJECTIVES:
        return (VERIFIED_ADJECTIVES[adj_key], "native-speaker-VERIFIED flat per-class lookup")
    if stem == "kubwa" and adj_class:
        if adj_class in KUBWA_FORM_BY_SM:
            return (KUBWA_FORM_BY_SM[adj_class], f"generalized 'kubwa' rule, CONFIRMED class marker '{adj_class}-'")
        return (KUBWA_DEFAULT_FORM, f"'kubwa' rule, UNCONFIRMED default for marker '{adj_class}-' -- flag for review")
    if stem in INGI_SURFACE_FORMS and adj_class:
        # 2026-10-20: own irregular/archaic concord series, same shape as
        # kubwa (not a kit_sm+stem formula) -- see INGI_FORM_BY_SM's own
        # comment in kitaita_verified_corrections.py for the evidence.
        if adj_class in INGI_FORM_BY_SM:
            return (INGI_FORM_BY_SM[adj_class], f"'-ingi' rule, CONFIRMED class marker '{adj_class}-'")
        return (INGI_DEFAULT_FORM, f"'-ingi' rule, UNCONFIRMED default for marker '{adj_class}-' -- flag for review")
    # 2026-10-11: same tables as translate()'s own adjective branch (kitaita_
    # verified_corrections.py) -- kept in sync deliberately, not re-derived,
    # after this exact "same rule forked into 2 copies" shape caused a real
    # bug this morning (the object-marker suppress_tam fix).
    if stem == "zuri" and kit_sm:
        # 2026-10-12: Kitaita Word Bank round confirmed every remaining class
        # (chi/ghi/ghu/ji/ku/lu/ri/u/va) exactly matches this formula -- only
        # 'i'/'wi' predate this round; kept the distinction in the trace for
        # the record, but there is no longer an "extrapolated" case in practice.
        confirmed = kit_sm in ("i", "wi", "chi", "ghi", "ghu", "ji", "ku", "lu", "ri", "u", "va")
        return (f"{kit_sm}{ZURI_ROOT}",
                f"corpus-mined 'zuri' rule ({'CONFIRMED' if confirmed else 'EXTRAPOLATED'} class "
                f"marker '{kit_sm}-'), statistical candidate, NOT native-speaker-verified")
    if stem == "safi" and kit_sm:
        # 2026-10-20: same mechanism/evidence tier as 'zuri' above -- see
        # kitaita_verified_corrections.py SAFI_STEM for the full verse trail.
        confirmed = kit_sm in ("i", "ri", "vi", "wi", "u", "gha")
        return (f"{kit_sm}{SAFI_STEM}",
                f"corpus-mined 'safi' rule ({'CONFIRMED' if confirmed else 'EXTRAPOLATED'} class "
                f"marker '{kit_sm}-'), statistical candidate, NOT native-speaker-verified")
    if stem == "moja" and kit_sm:
        # 2026-10-22: DIRECT native-speaker correction (not corpus-mined
        # like zuri/safi -- see kitaita_verified_corrections.py MOJA_ROOT),
        # same formula (full kit_sm + shared stem). Confirmed on 2
        # independent classes: 'ji' (ituku jimweri/one day) and 'chi'
        # (chadu chimweri/one shoe).
        confirmed = kit_sm in ("ji", "chi")
        return (f"{kit_sm}{MOJA_ROOT}",
                f"'moja' rule ({'CONFIRMED, direct native-speaker correction' if confirmed else 'EXTRAPOLATED to an untested class'} "
                f"marker '{kit_sm}-')")
    if stem == "refu" and adj_class:
        # 2026-10-11 (cont. 8): same tables/logic as translate()'s own 'refu'
        # branch (kitaita_verified_corrections.py LACHA_FORM_BY_SM/LACHA_
        # DEFAULT_FORM) -- kept in sync deliberately, not re-derived.
        if adj_class in LACHA_FORM_BY_SM:
            return (LACHA_FORM_BY_SM[adj_class], f"'refu' rule, CONFIRMED class marker '{adj_class}-' "
                    f"(Dice 0.86, 3 versions agree), statistical candidate, NOT native-speaker-verified")
        return (LACHA_DEFAULT_FORM, f"'refu' rule, UNCONFIRMED default for marker '{adj_class}-' -- flag for review")
    if stem == "dogo" and adj_class:
        if adj_class in DOGO_FORM_BY_SM:
            return (DOGO_FORM_BY_SM[adj_class], f"'dogo' rule (2026-10-12 Kitaita Word Bank), CONFIRMED "
                    f"class marker '{adj_class}-', direct submission, NOT native-speaker-judgment-verified")
        return (DOGO_DEFAULT_FORM, f"'dogo' rule, UNCONFIRMED default for marker '{adj_class}-' -- flag for review")
    if stem == "geni" and adj_class:
        if adj_class in GENI_FORM_BY_SM:
            return (GENI_FORM_BY_SM[adj_class], f"'geni' rule (2026-10-12 Kitaita Word Bank), CONFIRMED "
                    f"class marker '{adj_class}-', direct submission, NOT native-speaker-judgment-verified")
        return (GENI_DEFAULT_FORM, f"'geni' rule, UNCONFIRMED default for marker '{adj_class}-' -- flag for review")
    if stem == "fupi" and adj_class:
        if adj_class in FUPI_FORM_BY_SM:
            return (FUPI_FORM_BY_SM[adj_class], f"'fupi' rule (2026-10-12 Kitaita Word Bank), CONFIRMED "
                    f"class marker '{adj_class}-', direct submission, NOT native-speaker-judgment-verified")
        return (FUPI_DEFAULT_FORM, f"'fupi' rule, UNCONFIRMED default for marker '{adj_class}-' -- flag for review")
    if stem == "zito" and adj_class:
        if adj_class in ZITO_FORM_BY_SM:
            return (ZITO_FORM_BY_SM[adj_class], f"'zito' rule (2026-10-12 Kitaita Word Bank), CONFIRMED "
                    f"class marker '{adj_class}-', direct submission, NOT native-speaker-judgment-verified")
        return (ZITO_DEFAULT_FORM, f"'zito' rule, UNCONFIRMED default for marker '{adj_class}-' -- flag for review")
    if stem == "chafu" and adj_class:
        if adj_class in CHAFU_FORM_BY_SM:
            return (CHAFU_FORM_BY_SM[adj_class], f"'chafu' rule (2026-10-12 Kitaita Word Bank), CONFIRMED "
                    f"class marker '{adj_class}-', direct submission, NOT native-speaker-judgment-verified")
        return (CHAFU_DEFAULT_FORM, f"'chafu' rule, UNCONFIRMED default for marker '{adj_class}-' -- flag for review")
    if stem in ADJ_SINGLE_ROOT_GENERALIZED and kit_sm:
        # 2026-10-11 (cont.): direct correction -- route via the KITAITA class
        # (kit_sm -> reduced prefix, SAME mechanism as kubwa and the SAME helper
        # translate()'s own adjective branch uses, kept in sync deliberately),
        # not the Swahili class label the ADJ_SINGLE_CLASS_CANDIDATE path below
        # still checks. This is the exact fix for the "kijana/mdawana" case: the
        # Swahili label resolves to '7/8', but mdawana's own KITAITA class ('u',
        # default) is exactly what the candidate root is itself shaped for.
        root = ADJ_SINGLE_ROOT_GENERALIZED[stem]
        prefix = kitaita_adj_prefix(kit_sm)
        return (f"{prefix}{root}", f"single corpus-mined data point, generalized across Kitaita "
                f"classes via the reduced-prefix series (kit_sm '{kit_sm}-' -> '-{prefix}-') -- "
                f"WEAKER confidence than kubwa (1 confirming class, not 3+), NOT speaker-verified")
    if stem in ADJ_SINGLE_CLASS_CANDIDATE:
        cand = ADJ_SINGLE_CLASS_CANDIDATE[stem]
        swahili_class = None
        if noun_kitaita is not None:
            hit = _noun_lookup(swahili_noun, noun_kitaita)
            if hit:
                swahili_class = hit[0]
        if swahili_class == cand["swahili_class_seen"]:
            return (cand["kitaita_word"], f"single corpus-mined data point for class "
                    f"{cand['swahili_class_seen']} only, NOT native-speaker-verified -- {cand['note']}")
        return (None, f"single-class corpus candidate for '{stem}' exists only for Swahili class "
                       f"{cand['swahili_class_seen']!r}; '{swahili_noun}' resolved to "
                       f"{swahili_class!r}, and this stem doesn't decompose against the reduced "
                       f"Kitaita prefix series either -- not applied, flagged rather than guessed")
    return (None, "no confirmed Kitaita adjective-concord table for this noun's class")


def _noun_lookup(lemma: str, noun_kitaita: dict) -> tuple[str, str] | None:
    """Returns (swahili_class_or_None, evidence_label) if `lemma` is a
    noun this project has SOME reasonable basis for treating as known,
    else None. Three sources, in trust order: the 319-word hand-built
    lexicon (exact match, highest precision); VERIFIED_NOUN_TRANSLATIONS
    (native-speaker-corrected, no class info but definitely a real noun
    this project tracks); page1_vocabulary.json's statistical candidates
    at ANY confidence level.

    2026-10-09, direct user correction: this used to require 'high'/
    'medium' confidence here, excluding 'low'/'prefix-pattern' candidates
    as likely coincidences. Removed deliberately, for now: "for now we
    are testing, let's see the full limit and identify the actual bugs
    instead of having the stop rule being used as a possible source of
    barriers." A confidence filter is a reasonable PRECISION choice for a
    finished product, but it was acting as an undisclosed stop rule
    during exploration -- hiding how far this pipeline's RECALL actually
    reaches and which low-confidence candidates are real bugs versus
    genuinely fine. The evidence label still reports the real confidence
    level in every trace, so nothing is hidden, just no longer excluded.

    2026-10-09, found immediately after removing the confidence filter:
    "kubwa" (an adjective) matched a LOW-confidence statistical NOUN
    candidate mapping it to "ibaa" -- which is actually kubwa's OWN
    Kitaita adjective concord form, not a noun translation at all. The
    statistical table is built from raw word co-occurrence and has no
    concept of part-of-speech, so "kubwa" and its own adjective form
    "ibaa" co-occur constantly and get mistaken for a noun pair. This is
    exactly the class of error a POS-aware identification step (the
    user's requested NLLB probe front end) prevents by construction;
    until that lands, excluding the small, closed set of known adjective
    roots from noun candidacy is a correct, narrow guard, not a
    reintroduced stop rule -- it does not hide anything or cap recall,
    it just stops a word from being mis-tagged as the wrong part of
    speech when we already know with certainty which part of speech it
    is."""
    if lemma in ADJ_ROOTS:
        return None
    if lemma in LEXICON_BY_SURFACE:
        cls, _ = LEXICON_BY_SURFACE[lemma]
        return cls, "319-word lexicon exact match"
    if lemma in VERIFIED_NOUN_TRANSLATIONS:
        return None, "VERIFIED_NOUN_TRANSLATIONS (native-speaker-corrected)"
    cand = noun_kitaita.get(lemma)
    if cand:
        return cand.get("propagated_class"), f"statistical candidate ({cand['confidence']})"
    return None


def decompose_sentence(text: str, noun_kitaita: dict, noun_class_of: dict | None = None,
                        root_tables=None) -> Decomposition:
    # root_tables: 2026-10-19, optional -- the full build_lookup_tables() tuple, needed
    # only by apply_associative_descriptor_concord() (resolve_verb_root()). Defaults to
    # None so any OTHER existing caller that doesn't pass it keeps working unchanged;
    # that one rule is simply skipped (no descriptive-associative fusion attempted) when
    # it is absent, exactly like noun_class_of's own optional-default already works.
    tokens = tokenize(text)
    d = Decomposition(tokens=[t.lower() for t in tokens])
    assigned = [False] * len(tokens)

    # 2026-10-19: descriptive/purposive associative concord ("maji ya kunywa"
    # -> "machi ghenywa") MUST claim its two tokens (particle + infinitive)
    # HERE, before the noun_hits scan below even runs -- that scan's own
    # _noun_lookup() treats ANY "ku-"+2-more-letters token as a candidate
    # class-15 DEVERBAL-NOUN reading too (ku- is genuinely both the
    # infinitive marker and that noun prefix in Swahili, a known, already-
    # documented ambiguity -- see _noun_lookup()'s own docstring), and
    # claims it outright (assigned[i]=True, added to noun_hits) with no
    # chance for this more specific 3-token pattern to intervene afterward.
    # No-op (claims nothing) when root_tables wasn't passed in, or when the
    # pattern/class data doesn't match -- see the function's own docstring.
    if root_tables is not None:
        apply_associative_descriptor_concord(d, noun_kitaita, noun_class_of or {}, root_tables, assigned)

    noun_hits = []  # (idx, lemma, class_or_None, evidence)
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            # 2026-10-19: already claimed above (descriptive-associative fusion) --
            # must not ALSO be considered as an independent noun candidate here.
            continue
        # 2026-10-15: same known-adjective-root veto as the probes path --
        # see its own comment (decompose_sentence_with_probes) for why.
        if any(tok == root or any(tok == p + root for p in ADJ_PREFIXES if p) for root in ADJ_ROOTS):
            continue
        hit = _noun_lookup(tok, noun_kitaita)
        if hit is not None:
            cls, evidence = hit
            noun_hits.append((i, tok, cls, evidence))
            assigned[i] = True

    # 2026-10-22: proper-name noun registration -- see decompose_sentence_
    # with_probes()'s matching pass (is_likely_proper_name()'s own
    # docstring) for the full reasoning; mirrored here so this legacy
    # heuristic decomposer doesn't silently diverge from the probe-based
    # one on the same fix.
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue
        # 2026-10-22 (same-day fix): direct user bug report -- a sentence-
        # initial capitalized VERB ("Ninawapenda ndege...") was being
        # wrongly swallowed as a "proper name" here, BEFORE the verb_hits
        # loop below ever got a chance to see it (is_likely_proper_name()
        # alone has no way to tell "Sifuna" from "Ninawapenda" -- both are
        # capitalized with no Bantu NOUN-class prefix shape). Checking
        # decomposability as a real verb FIRST is the same "known-root/
        # known-pattern wins regardless of position" discipline the
        # ADJ_ROOTS veto just above already uses -- a token that
        # genuinely decomposes as SM+TAM+root+FV is never a name.
        if is_likely_proper_name(tokens[i], tok) and decompose_verb_or_copula(tok) is None:
            noun_hits.append((i, tokens[i], "ANIMATE-SG",
                               "proper name (capitalized, no Bantu class-prefix shape) -- "
                               "Kitaita class 'u' (mundu), per PROPER_NAME_KITAITA_CLASS"))
            assigned[i] = True

    verb_hits = []  # (idx, VerbFormEntry)
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue
        v = decompose_verb_or_copula(tok)
        if v is not None and len(v.root_guess) >= 2:
            verb_hits.append((i, v))
            assigned[i] = True

    if not verb_hits:
        d.flags.append("NO VERB found -- cannot translate (kitaita_sentence_generator.translate() "
                        "requires a decomposed verb). Possible causes: an irregular/contracted verb "
                        "form, a verbless construction, or a TAM/SM combination not in SM_TABLE/"
                        "TAM_MARKERS.")
    else:
        vi, v = verb_hits[0]
        d.verb_idx, d.verb_surface = vi, tokens[vi]
        d.verb = {"sm": v.subject_marker, "tam": v.tam_marker, "root": v.root_guess, "fv": v.final_vowel,
                  "rel_candidates": v.relative_marker_candidates, "om_candidates": v.object_marker_candidates,
                  "negative_subjunctive_candidate": v.negative_subjunctive_candidate,
                  "ku_infinitive_alternate": v.ku_infinitive_alternate}
        if len(verb_hits) > 1:
            extra = ", ".join(tokens[i] for i, _ in verb_hits[1:])
            d.flags.append(f"Found {len(verb_hits)} verb-shaped tokens; used the FIRST ('{tokens[vi]}') "
                            f"and kept the rest ({extra}) as Swahili -- not guessing which is the real "
                            f"verb for a second clause (likely a coordinated/compound/relative-clause "
                            f"sentence, out of this decomposer's single-verb scope).")
            for ei, _ in verb_hits[1:]:
                d.passthrough.append((ei, tokens[ei]))

    # 2026-10-09, direct user correction: "why aren't we able to translate
    # sentences with multiple nouns... NLLB is coordinating sentences for us
    # already... in case you miss a word, use the Swahili word -- it would be
    # better and would show us the true limits of the capability." This
    # replaced an earlier version that capped at exactly one subject + one
    # object and DROPPED every other noun with a flag -- too strict, and it
    # hid the fact that every piece of e.g. "Kitabu cha mwalimu kilikuwa
    # kikubwa." is ALREADY computable from this project's own tables (kitabu
    # -> chuo, mwalimu -> mwalimu unchanged, kilikuwa+kikubwa -> an EXACT
    # verified sentence "Chuo chedeka kibaa."). Now: only the PRIMARY subject
    # (first noun before the verb) and PRIMARY object (first noun after the
    # verb) feed translate()'s agreement computation, exactly as before --
    # that part genuinely does need a fixed single noun per role, since
    # translate() only has one agreement-controlling subject slot. But every
    # OTHER noun (a possessor, a coordinated subject/object, anything else)
    # is still independently substituted to its own Kitaita word via a plain
    # lookup (no agreement involved for these -- object substitution was
    # always marker-free in this project anyway, and a possessor/coordinated
    # noun doesn't control the verb), kept at its OWN original position, and
    # falls back to the original Swahili word rather than being dropped when
    # no Kitaita translation exists -- same principle as every other
    # untranslatable piece below. The ONE real caveat (not swept under the
    # rug): when a secondary noun sits BEFORE the verb (a coordinated
    # subject, "X na Y verb"), the verb's agreement still only reflects the
    # PRIMARY (first) noun -- closest-conjunct agreement. This is not a
    # guess invented here: it is one of three candidate answers this
    # project's own research already found for coordinated-subject agreement
    # (CLAUDE.md: gold text used human-plural wa-, a naive rule predicted
    # vi-, NLLB's own generation produced singular/closest-conjunct) with no
    # native-speaker resolution yet -- flagged plainly, not hidden.
    # ASSOC_PARTICLES is now module-level (see apply_associative_concord()).
    # 2026-10-09: found by testing with the noun-confidence filter removed
    # (see _noun_lookup) -- "Askari alimwona mwizi." (soldier saw thief)
    # has NO recognized noun before the verb ("askari" isn't in any table),
    # but "mwizi" (thief), the real OBJECT, IS recognized -- and the old
    # code blindly took noun_hits[0] (whichever noun was found FIRST,
    # regardless of position) as the subject, producing a sentence with
    # the agreement computed as if the OBJECT were the subject. The
    # primary subject must come from a noun hit BEFORE the verb specifically
    # -- if none exists, there is no reliable subject at all (not "whichever
    # noun happened to be recognized"), and every noun found is treated as
    # an independent substitution instead (same as the no-noun-at-all case).
    pre_verbal_hits = [h for h in noun_hits if d.verb_idx is None or h[0] < d.verb_idx]
    if not pre_verbal_hits:
        # 2026-10-15: checked BEFORE the noun_hits/no-noun_hits messaging below --
        # a PRO-DROPPED person subject or a negated verb is possible regardless of
        # whether some OTHER, non-subject noun exists in the sentence (e.g. "kusoma"
        # in "Ninapenda kusoma." is a real noun_hit, as an infinitive-as-noun class-15
        # object, but that doesn't change that "Ninapenda" has no pre-verbal noun
        # subject and its own marker is a recognized 1st-person prefix). Checking
        # `noun_hits` first here would silently swallow every such case behind the
        # generic "no noun before the verb" message and never reach this branch.
        if verb_hits and verb_hits[0][1].negated and verb_hits[0][1].person is not None:
            v0 = verb_hits[0][1]
            d.negated = True
            d.person = v0.person
            d.flags.append(f"NEGATION detected ('{tokens[verb_hits[0][0]]}', subject marker "
                            f"'{v0.subject_marker}-', Swahili negative-present, person '{d.person}') -- "
                            f"Kitaita negation marker native-speaker-submitted 2026-10-15 (Kitaita Word "
                            f"Bank), translating.")
        elif verb_hits and verb_hits[0][1].negated:
            d.negated = True
            d.flags.append(f"NEGATION detected ('{tokens[verb_hits[0][0]]}'), but no noun subject and "
                            f"no person marker either -- same coreference limitation as affirmative "
                            f"3rd-person pro-drop (out of scope); cannot translate.")
        elif verb_hits and verb_hits[0][1].person is not None:
            d.person = verb_hits[0][1].person
            d.flags.append(f"No pre-verbal noun subject -- but subject marker "
                            f"'{verb_hits[0][1].subject_marker}-' is a recognized PERSON prefix "
                            f"('{d.person}') -- translating as a pro-dropped pronoun subject "
                            f"(kitaita_person_marker_investigation.py) instead of refusing outright.")
        elif noun_hits:
            d.flags.append("No noun recognized BEFORE the verb -- no reliable subject, so no subject-"
                            "verb agreement can be computed. Every noun found is substituted "
                            "independently instead (see below), not forced into the subject role.")
        else:
            d.flags.append("NO NOUN found at all -- no subject, so translation cannot proceed. (3rd-"
                            "person pro-drop with no antecedent in this sentence is a coreference "
                            "problem, out of scope; see kitaita_person_marker_investigation.py for the "
                            "1st/2nd-person pronoun-subject track, used above when applicable.)")
        for oi, olemma, _, _ in noun_hits:
            plural_ctx = _siku_plural_context(d.tokens, oi) if olemma == "siku" else False
            kit = lookup_noun_kitaita(olemma, noun_kitaita, plural_context=plural_ctx)
            d.secondary_nouns.append((oi, olemma, kit))
            d.flags.append(f"'{olemma}' -> '{kit}': substituted independently (no subject available)."
                            if kit else f"'{olemma}': no Kitaita translation available -- kept as the "
                                        f"original Swahili word (no subject available).")
    else:
        si, slemma, scls, sevidence = pre_verbal_hits[0]
        d.subject, d.subject_idx, d.subject_class = slemma, si, scls
        d.subject_class_confidence = sevidence
        # 2026-10-15: a real pre-verbal noun subject doesn't mean the verb is
        # affirmative -- "Mwalimu hapendi" has one. Checked here too, not just
        # in the no-pre-verbal-noun branch above.
        if verb_hits and verb_hits[0][1].negated:
            d.negated = True
        for oi, olemma, _, _ in noun_hits:
            if oi == si:
                continue
            if d.object is None and d.verb_idx is not None and oi > d.verb_idx:
                d.object, d.object_idx = olemma, oi
                continue
            plural_ctx = _siku_plural_context(d.tokens, oi) if olemma == "siku" else False
            kit = lookup_noun_kitaita(olemma, noun_kitaita, plural_context=plural_ctx)
            d.secondary_nouns.append((oi, olemma, kit))
            if oi > 0 and tokens[oi - 1] in ASSOC_PARTICLES:
                # 2026-10-15: the particle itself is now handled generically by
                # apply_associative_concord() (called once, separately, before
                # the passthrough sweep) -- this message just labels the
                # RELATION for this secondary noun, not the particle's fate.
                relation = f"possessive/associative noun (follows '{tokens[oi - 1]}', see Kitaita " \
                           f"associative concord above/below for that particle)"
            elif d.verb_idx is not None and oi < d.verb_idx:
                relation = "coordinated subject (pre-verbal) -- substituted independently, but the " \
                           "verb's agreement still reflects only the PRIMARY subject (closest-conjunct; " \
                           "see note above, this project has no settled rule for coordinated-subject " \
                           "agreement)"
                # 2026-10-13: cheap, zero-probe DETECTION per direct user request --
                # the literal adjacency "Subject na Noun2 Verb" (si+1 == "na" token,
                # oi == si+2, oi immediately precedes the verb) is read straight off
                # tokenization, no NLLB/probe needed. Deliberately detection-ONLY:
                # this does NOT choose a plural Kitaita form -- CLAUDE.md already
                # flags which plural rule applies (closest-conjunct vs shared-class
                # vs human-plural-override) as unresolved with no native-speaker
                # ruling, so guessing one here would just hide the ambiguity instead
                # of flagging it. Only fires on this exact adjacency, not any
                # pre-verbal secondary noun (e.g. a relative-clause subject would
                # wrongly fire the coordinated_subject label without "na" present).
                if (si + 1 < len(tokens) and tokens[si + 1] == "na" and oi == si + 2
                        and oi + 1 == d.verb_idx):
                    d.coordinated_subject = (slemma, olemma)
                    d.flags.append(f"COORDINATED SUBJECT detected ('{slemma} na {olemma}' immediately "
                                    f"before the verb) -- Swahili agreement here is almost certainly "
                                    f"PLURAL, but this project has NO settled rule for which Kitaita "
                                    f"plural form applies (three candidate rules, no native-speaker "
                                    f"ruling -- see CLAUDE.md). The verb output below uses only the "
                                    f"PRIMARY subject's (singular) marker and is likely WRONG for this "
                                    f"sentence -- flagged, not corrected.")
            else:
                relation = "additional/coordinated noun beyond the primary subject and object"
            if kit:
                d.flags.append(f"'{olemma}' -> '{kit}': {relation}.")
            else:
                d.flags.append(f"'{olemma}': no Kitaita translation available -- kept as the original "
                                f"Swahili word ({relation}).")

    # Adjective: recognized by ADJACENCY to the noun it modifies -- standard
    # Swahili/Bantu attributive structure, a noun and its adjective form one
    # unit. Two valid positions for the SUBJECT's own adjective: directly after
    # the subject (attributive, "chui mkubwa aliona...") or directly after the
    # verb when there is no object (predicate-complement, "mwalimu alikuwa
    # mkubwa" / "mwalimu akafika kubwa" -- matches s86's real judged sentence
    # exactly, see below). A kubwa/baya-shaped token directly after the OBJECT
    # concords with the OBJECT by the same adjacency logic, but this project's
    # generator has no object-adjective mechanism at all (translate()'s
    # adjective branches only ever use the subject's own marker) -- flagged,
    # not guessed onto the subject.
    #
    # 2026-10-09: this replaced an earlier, WRONG attempt to decide "where
    # Kitaita puts adjectives" as if it were a grammar fact to verify. It
    # isn't one -- checking kitaita_sentence_batch.py's build_swahili_sentence()
    # directly showed every one of this project's test prompts (s86 included)
    # was built with adjectives trailing by a fixed DISPLAY CONVENTION for
    # isolating one spec variable at a time, not because that is how Kitaita
    # naturally orders attributive adjectives. The actual project methodology
    # (CLAUDE.md, memory: marker-only substitution, never rearrangement) means
    # this decomposer's only job is to recognize which noun an adjective token
    # is adjacent to; assemble_in_order() below then places the SUBSTITUTED
    # Kitaita word at that SAME position in the output, never a fixed template
    # position. See that function's docstring for the full fix.
    if d.subject is not None:
        subj_class_guess = d.subject_class
        if subj_class_guess is None:
            guess = guess_noun_class_by_prefix(d.subject)
            if guess:
                subj_class_guess = guess[0]
        adj_i = d.subject_idx + 1 if d.subject_idx is not None else None
        verb_adj_i = (d.verb_idx + 1 if d.object is None and d.verb_idx is not None else None)
        obj_adj_i = d.object_idx + 1 if d.object_idx is not None else None
        for candidate_i in (adj_i, verb_adj_i):
            if d.adjective is not None or candidate_i is None:
                continue
            if candidate_i >= len(d.tokens) or assigned[candidate_i]:
                continue
            tok = d.tokens[candidate_i]
            for root in ADJ_ROOTS:
                if tok == root or any(tok == p + root for p in ADJ_PREFIXES if p):
                    d.adjective = {"stem": root, "class": subj_class_guess}
                    d.adjective_idx = candidate_i
                    assigned[candidate_i] = True
                    break
        if obj_adj_i is not None and obj_adj_i < len(d.tokens) and not assigned[obj_adj_i]:
            tok = d.tokens[obj_adj_i]
            for root in ADJ_ROOTS:
                if tok == root or any(tok == p + root for p in ADJ_PREFIXES if p):
                    d.flags.append(f"'{tokens[obj_adj_i]}' directly follows the OBJECT ('{d.object}'), "
                                    f"so it concords with the object by normal Swahili adjacency -- but "
                                    f"this project's generator has no object-adjective mechanism at all. "
                                    f"Kept as the original Swahili word rather than wrongly attached to "
                                    f"the subject.")
                    d.passthrough.append((obj_adj_i, tokens[obj_adj_i]))
                    assigned[obj_adj_i] = True
                    break

    # 2026-10-19: descriptive/purposive associative concord already ran at
    # the TOP of this function (before noun_hits), not here -- see that
    # call's own comment for why it has to run that early.
    # 2026-10-20: "-ote" (all/whole) concord -- see apply_ote_concord()'s own
    # docstring. Must run BEFORE apply_associative_concord (which needs to
    # skip back over an already-claimed -ote token to find the real governor).
    apply_ote_concord(d, noun_kitaita, noun_class_of or {}, assigned)
    # 2026-10-15: associative/possessive concord ("mayai ya kuku" ->
    # "maghi gha nguku") -- see apply_associative_concord()'s own docstring.
    # Runs over the whole token sequence, not tied to subject/object/verb
    # structure, so it must run before the passthrough sweep below but
    # doesn't need to be threaded through the subject/object logic above.
    apply_associative_concord(d, noun_kitaita, noun_class_of or {}, assigned, tokens=tokens)
    # 2026-10-15: demonstrative concord ("hii kazi" -> "iri kazi") -- see
    # apply_demonstrative_concord()'s own docstring. Same reasoning as above.
    apply_demonstrative_concord(d, noun_kitaita, noun_class_of or {}, assigned)

    # 2026-10-09, direct user correction: never drop a token for lack of
    # Kitaita data -- substitute what can be substituted, and keep the
    # ORIGINAL SWAHILI WORD for anything that can't, so the output always
    # covers the whole input and visibly shows exactly what this project
    # does and doesn't have data for, rather than silently refusing.
    for i, is_assigned in enumerate(assigned):
        if not is_assigned:
            d.passthrough.append((i, tokens[i]))
    if d.passthrough:
        words = ", ".join(tokens[i] for i, _ in sorted(set(d.passthrough)))
        d.flags.append(f"No Kitaita translation/mechanism for: {words} -- kept as the original Swahili "
                        f"word(s) in place. Could be a particle, conjunction, negation marker, object "
                        f"marker, relative clause, a different adjective, or simply a noun/verb not yet "
                        f"in this project's data.")
    return d


OBJECT_MARKER_RISK_ROOT_DICE_THRESHOLD = 0.4


def _root_evidence_score(root: str, root_kitaita: dict, root_kitaita_fallback: dict) -> float:
    """Best available Dice evidence for a Swahili VERB ROOT -- ROOT-LEVEL
    Dice (root_kitaita) when it exists, else the WHOLE-SURFACE-FORM
    fallback table's Dice, else -1.0 (no evidence at all). 2026-10-15: the
    existing object-marker presence-detection below only ever compared
    root_kitaita, which is fine there (object-marker roots usually DO have
    root-level data); relative-marker presence-detection needs this wider
    comparison because the unstripped root and the stripped one can BOTH
    be absent from root_kitaita (e.g. 'yofany' vs 'fany' -- neither has a
    root-level entry, so a root_kitaita-only comparison sees a tie, -1.0
    vs -1.0, and never switches), in which case the fallback table is the
    only evidence available to tell them apart at all."""
    c = root_kitaita.get(root)
    if c:
        return c["score"]
    fb = root_kitaita_fallback.get(root)
    return fb["best_dice"] if fb else -1.0


def _root_evidence_score_verified_first(root: str, root_kitaita: dict, root_kitaita_fallback: dict) -> float:
    """Same as _root_evidence_score() above, but a HAND-VERIFIED root
    (VERIFIED_ROOTS) or confirmed periphrasis (VERIFIED_IDIOMS) is treated
    as maximal-confidence evidence (1.0) -- strictly better than any Dice
    score, same reasoning as the object-marker presence-detection check's
    own VERIFIED_ROOTS special case further down (2026-10-20). 2026-10-21:
    factored out and shared with the relative-marker presence-detection
    check too, found while wiring up "alipoangusha"/"angush" and
    "alimkasirisha"/"kasirish" -- without this, a verified root with no
    root-level Dice entry of its own scores -1.0, below even a weak but
    PRESENT statistical competitor (e.g. the unstripped 'poangush'/
    'mkasirish'), and the verified root never wins the comparison."""
    if root in VERIFIED_ROOTS or root in VERIFIED_IDIOMS:
        return 1.0
    return _root_evidence_score(root, root_kitaita, root_kitaita_fallback)


# 2026-10-20: which Swahili noun class(es) a RELATIVE-marker string can
# imply, reference grammar (same source as REL_TABLE itself, extract_
# swahili_grammar.py) -- needed for the headless-subject case (direct user
# example: "Kile kinachonisikitisha", no concrete noun anywhere, only the
# class-7 demonstrative "kile" + the class-7 relative marker "-cho-" on the
# verb; SWAHILI_CLASS_DEFAULT_SM_GUESS's own per-lemma-fallback mechanism
# still needs SOME subject_class to key off of, which this supplies when
# there is no noun at all to get it from). Several REL_TABLE strings are
# genuinely ambiguous between multiple classes (Swahili itself doesn't
# disambiguate them without a visible antecedent) -- listed as multiple
# candidates here, resolved only when they happen to agree on the SAME
# Kitaita default (see resolve_headless_subject_class() below), never
# guessed when they don't. This project's own ANIMATE-SG/PL split (not
# Swahili's numbered 1/2) is used for "ye"/the animate share of "o", same
# scheme SWAHILI_CLASS_DEFAULT_SM_GUESS itself already uses.
REL_MARKER_TO_SWAHILI_CLASS = {
    "ye": ["ANIMATE-SG"], "o": ["ANIMATE-PL", "3", "11"], "yo": ["4", "6", "9/10"],
    "lo": ["5"], "cho": ["7"], "vyo": ["8"], "zo": ["9/10"], "ko": ["15"],
}


def resolve_headless_subject_class(rel_candidates: list) -> tuple[str | None, str | None, str | None, str | None]:
    """For a verb with NO noun subject at all (relative_marker_candidates
    is the only class signal available), tries every enumerated relative-
    marker candidate and returns (swahili_class, kitaita_default_sm, rel,
    remainder) for the FIRST one that resolves UNAMBIGUOUSLY -- either
    because REL_MARKER_TO_SWAHILI_CLASS lists only one candidate class for
    that marker string, or because every candidate class that actually HAS
    a default (SWAHILI_CLASS_DEFAULT_SM_GUESS) agrees on the same Kitaita
    marker. Returns (None, None, None, None), never a guess, when every
    candidate is either unlisted or genuinely conflicting (e.g. '-o-'
    could be class ANIMATE-PL='wi' or class 3='ghu' -- these disagree, so
    '-o-' alone resolves nothing).

    `rel`/`remainder` (the SPECIFIC candidate that resolved the class) are
    returned too so the caller can strip EXACTLY that relative marker
    directly, grammatically motivated by the class resolution itself --
    NOT re-decided by Dice-evidence comparison against the other
    candidates (translate_sentence_with_probes()'s own evidence-based
    correction, reused for the ORDINARY case where a real noun subject
    already pins the class down, is the wrong tool here: it only switches
    on a STRICT evidence improvement, but a brand-new/rare root and its
    relative-marker-contaminated version are both equally likely to have
    ZERO Dice evidence either way -- a tie that heuristic correctly never
    breaks. Here, resolving the class via REL_MARKER_TO_SWAHILI_CLASS IS
    the independent evidence that this candidate's marker is real, so
    stripping it isn't a guess)."""
    for rel, remainder in rel_candidates:
        sw_classes = REL_MARKER_TO_SWAHILI_CLASS.get(rel)
        if not sw_classes:
            continue
        defaults = {SWAHILI_CLASS_DEFAULT_SM_GUESS[c] for c in sw_classes if c in SWAHILI_CLASS_DEFAULT_SM_GUESS}
        if len(defaults) == 1:
            resolved_class = next(c for c in sw_classes if c in SWAHILI_CLASS_DEFAULT_SM_GUESS)
            return resolved_class, defaults.pop(), rel, remainder
    return None, None, None, None


def translate_sentence(text: str, tables) -> dict:
    noun_kitaita, root_kitaita, noun_class_of = tables[0], tables[1], tables[3]
    root_kitaita_fallback = tables[2]
    d = decompose_sentence(text, noun_kitaita, noun_class_of, root_tables=tables)
    report = {"input": text, "decomposition": d, "kitaita_sentence": None, "trace": [], "error": None}
    if d.verb is not None and d.subject is None and d.person is None:
        # 2026-10-20, direct user request: a HEADLESS subject (no noun at
        # all, e.g. "Kile kinachonisikitisha" -- "kile" is a demonstrative/
        # relative-clause pronoun, not a noun this decomposer recognizes as
        # one) previously fell straight through to the refusal below even
        # though the verb's OWN relative marker already names its class.
        # Tried BEFORE the refusal check, not after -- resolving the class
        # here is exactly what lets this sentence avoid that refusal.
        sw_class, kit_sm_default, resolved_rel, resolved_remainder = resolve_headless_subject_class(
            d.verb.get("rel_candidates") or [])
        if kit_sm_default is not None:
            d.subject_class = sw_class
            # Strip EXACTLY the relative marker that resolved the class above
            # (grammatically motivated by that resolution itself, not a
            # Dice-evidence guess -- see resolve_headless_subject_class()'s
            # own docstring for why the probes path's evidence-based switch
            # is the wrong tool here). Then re-enumerate OBJECT markers
            # against the NEWLY-stripped remainder (the ORIGINAL om_
            # candidates were computed against the unstripped stem and are
            # stale, same staleness the probes path's own 2026-10-15 comment
            # already flags) and apply this project's single-unambiguous-
            # candidate heuristic (2026-10-19 Taita-fy fix) to also drop a
            # person/reflexive object marker when exactly one fits.
            old_root = d.verb["root"]
            new_root, new_fv = resolved_remainder[:-1], resolved_remainder[-1]
            om_candidates = find_object_marker_candidates(resolved_remainder)
            if len(om_candidates) == 1 and len(om_candidates[0][1]) >= 3:
                om, alt_root, alt_fv, _ = om_candidates[0]
                new_root, new_fv = alt_root, alt_fv
            if new_root != old_root:
                d.flags.append(f"'{d.verb_surface}': relative marker '-{resolved_rel}-' (class '{sw_class}', "
                                f"resolved above) + object-marker check stripped '{old_root}' down to "
                                f"'{new_root}'.")
                d.verb["root"], d.verb["fv"] = new_root, new_fv
                d.verb["om_candidates"] = []
            d.flags.append(f"No noun subject found, but the verb's relative marker implies Swahili class "
                            f"'{sw_class}' -- used this project's CORPUS-COMPUTED per-class default Kitaita "
                            f"subject marker '{kit_sm_default}-' (SWAHILI_CLASS_DEFAULT_SM_GUESS, 2026-10-20) "
                            f"rather than refusing outright. This is a STATISTICAL DEFAULT, not a specific "
                            f"answer for this sentence's real (absent) noun -- flagged, not a confirmed "
                            f"agreement choice.")
    if d.verb is None or (d.subject is None and d.person is None and d.subject_class is None):
        # 2026-10-09, direct user correction: still produce a best-effort
        # sentence (every noun substituted independently, everything else
        # kept as Swahili) rather than refusing outright -- no agreement can
        # be computed without a subject+verb pair, but that is no reason to
        # withhold what IS translatable.
        # 2026-10-15: d.person (a recognized 1st/2nd-person pro-dropped
        # subject marker) is a second way past this guard, alongside the
        # original noun-subject path -- d.negated sentences never get a
        # person set (see decompose_sentence()), so they still fall through
        # here and refuse, exactly as intended. 2026-10-20: d.subject_class
        # (set just above, headless-relative-marker case) is a THIRD way
        # past this guard.
        report["error"] = "; ".join(d.flags) or "decomposition failed"
        report["kitaita_sentence"] = assemble_in_order(d, None, noun_kitaita)
        return report

    root = d.verb["root"]
    if root != "kuw" and root not in VERIFIED_ROOTS:
        cand = root_kitaita.get(root)
        if cand and cand["score"] < OBJECT_MARKER_RISK_ROOT_DICE_THRESHOLD:
            d.flags.append(f"Verb root '{root}' only matched a WEAK statistical candidate "
                            f"(Dice {cand['score']:.2f}) -- this decomposer does NOT strip infixed "
                            f"object markers (e.g. '-mw-'/'-wa-'/'-ki-' between TAM and root, as in "
                            f"'a-li-mw-on-a'), so a weak match here often means the 'root' actually "
                            f"still contains one, corrupting the result. Treat the output below as "
                            f"UNRELIABLE, not just low-confidence.")

    result = translate(d.subject, d.verb, d.object, d.adjective, tables, person=d.person,
                        subject_class=d.subject_class, negated=d.negated)
    if "error" in result:
        report["error"] = result["error"]
        report["trace"] = result.get("trace", [])
        report["kitaita_sentence"] = assemble_in_order(d, None, noun_kitaita)
        return report
    report["kitaita_sentence"] = assemble_in_order(d, result, noun_kitaita)
    report["trace"] = result["trace"]
    report["components"] = result["components"]
    if result["components"].get("root_untranslated"):
        # 2026-10-19: surfaced into d.flags (not just components) so every existing
        # consumer that scans report["decomposition"].flags for human-readable
        # warnings (e.g. this file's own __main__ demo loop) sees this gap too --
        # see translate()'s matching trace entry for the full reasoning.
        d.flags.append(f"Verb root '{d.verb['root']}' has NO Kitaita translation at all -- the "
                        f"sentence above keeps the SWAHILI root with Kitaita subject/TAM agreement "
                        f"('Taita-fied' hybrid, a readability stopgap), NOT a real translation. "
                        f"Insufficient vocabulary coverage, not a parsing failure.")
    if result["components"].get("adjective_untranslated"):
        d.flags.append(f"Adjective '{result['components']['adjective_stem']}' has NO Kitaita concord "
                        f"data at all -- kept as a Swahili stem with a GUESSED reduced concord prefix "
                        f"(double guess, weaker than the verb-root stopgap above; see translate()'s trace).")
    report["coordinated_subject"] = d.coordinated_subject  # see Decomposition field docstring -- detection-only
    return report


def assemble_in_order(d: Decomposition, result: dict | None, noun_kitaita: dict) -> str | None:
    """Builds the Kitaita output by substituting each identified piece's
    Kitaita word AT THE SAME POSITION it held in the original Swahili input --
    never a fixed subject-verb-object-adjective template, and never dropping
    a token just because one piece (most often verb-subject agreement) could
    not be resolved.

    2026-10-09, direct user correction #1: "the sentence order of words was
    not supposed to change... substitution should be algorithmic." This
    project's whole methodology (CLAUDE.md, memory: marker-only substitution)
    never calls for rearranging a sentence, only for substituting each word/
    morpheme-slot with its Kitaita equivalent in place.

    2026-10-09, direct user correction #2 (same session): "in case you miss
    a word in kitaita use the swahili word... would show us the true limits
    of the capability." `result` is None whenever translate() could not be
    called at all (no verb found, no subject found) or itself errored (e.g.
    no Kitaita candidate for the verb root) -- in that case this function
    still assembles everything it CAN: every noun (subject/object/
    secondary) via a plain independent lookup, and the verb/adjective kept
    as their ORIGINAL SWAHILI surface form (no agreement could be computed
    for them). `d.secondary_nouns` and `d.passthrough` are always included
    regardless of whether `result` succeeded -- a possessor, a coordinated
    noun, or a particle with no Kitaita data is never dropped, only the
    primary subject-verb(-object)(-adjective) agreement depends on
    translate() succeeding.

    EXCEPTION: when translate() returned an EXACT native-speaker-verified
    sentence (VERIFIED_SENTENCES), that string IS the real, whole, already-
    correct native-speaker answer for the (subject, verb, adjective) block --
    split into [subject-word, rest-of-string] so OTHER material (a
    possessive phrase, a coordinated noun) can still be correctly
    interleaved around it without touching its own internal order."""
    comps = (result or {}).get("components", {})
    pieces = []

    if comps.get("verified_exact_sentence"):
        verified = result["kitaita_sentence"].rstrip(".")
        first_word, _, rest = verified.partition(" ")
        if d.subject_idx is not None:
            pieces.append((d.subject_idx, first_word))
        if rest and d.verb_idx is not None:
            pieces.append((d.verb_idx, rest))
    else:
        if d.subject_idx is not None:
            plural_ctx = _siku_plural_context(d.tokens, d.subject_idx) if d.subject == "siku" else False
            kit = comps.get("subject_noun") if result else lookup_noun_kitaita(d.subject, noun_kitaita,
                                                                                 plural_context=plural_ctx)
            pieces.append((d.subject_idx, kit or d.subject))
        if d.verb_idx is not None:
            kit = comps.get("kitaita_verb_word") if result else None
            pieces.append((d.verb_idx, kit or d.verb_surface or d.tokens[d.verb_idx]))
        if d.object_idx is not None:
            kit = comps.get("object_noun") if result else lookup_noun_kitaita(d.object, noun_kitaita)
            pieces.append((d.object_idx, kit or d.object))
            if comps.get("idiom_object_trailing"):
                # 2026-10-21: a VERIFIED_IDIOMS fixed object that must come AFTER the
                # real object noun (see translate()'s own comment) -- no original token
                # to anchor to, so a fractional index places it right after the real
                # object's slot without disturbing any other token's position.
                pieces.append((d.object_idx + 0.5, comps["idiom_object_trailing"]))
        elif comps.get("idiom_object_trailing") and d.verb_idx is not None:
            # defensive fallback: translate() only sets this when it saw a real object
            # noun, so this should not normally trigger -- if it ever does (e.g. the
            # object noun itself had no Kitaita translation and was dropped upstream),
            # don't silently lose the idiom's fixed object, anchor it to the verb instead.
            pieces.append((d.verb_idx + 0.5, comps["idiom_object_trailing"]))
        if d.adjective_idx is not None:
            kit = comps.get("adjective_kitaita") if result else None
            pieces.append((d.adjective_idx, kit or d.tokens[d.adjective_idx]))

    for idx, lemma, kit in d.secondary_nouns:
        pieces.append((idx, kit or lemma))
    for idx, tok in d.passthrough:
        if not any(i == idx for i, _ in pieces):
            pieces.append((idx, tok))

    if not pieces:
        return None
    pieces.sort(key=lambda p: p[0])
    words = [w.capitalize() if i == 0 else w for i, (_, w) in enumerate(pieces)]
    return " ".join(words) + "."


def find_fallback_proper_name_subject(tokens: list[str], before_idx: int, used_idxs) -> int | None:
    """2026-10-20, direct user bug report: "Basi Hezekia akajigeuza, ..." --
    EVERY clause's verb failed to find a subject at all ("no noun won as
    nsubj against every competitor"), because "Hezekia" (a Biblical name,
    not in any noun table) was never even a NOUN CANDIDATE in the first
    place, let alone a winning one -- a real, high-impact gap: a human
    proper-name subject is extremely common in real text, and Swahili
    grammar requires ALL such names to take class-1/ANIMATE-SG verb
    agreement regardless of which specific name it is (a near-universal
    grammatical fact, not a statistical guess like most of this project's
    other defaults). Every verb in that sentence ended up keeping its
    RAW, UNCONVERTED Swahili subject marker ('a-'/'aka-') because
    translate() was never even called for any of them.

    Heuristic: the nearest token STRICTLY BEFORE `before_idx` (the verb)
    that (a) is NOT the sentence's first token (sentence-initial capitals
    say nothing about being a name -- "Basi" here is a discourse particle,
    not one), (b) is capitalized in the ORIGINAL-CASE `tokens` list (NOT
    `d.tokens`, which is already lowercased by this point), and (c) is not
    in `used_idxs` (already claimed as some OTHER clause's subject/object
    -- confirmed directly why this check matters: without it, "BWANA"
    [class-marker-free, also capitalized, correctly claimed as 'akamwomba'
    (he begged HIM)'s OBJECT a few tokens earlier] would be wrongly grabbed
    as 'akasema' (he said)'s SUBJECT instead of letting the real subject,
    "Hezekia", carry over across the whole coordinated clause chain via
    pro-drop, exactly as Swahili itself does here).

    Ordinary verb tokens are never capitalized unless sentence-initial, so
    no separate "is this a verb" exclusion is needed beyond (a). Returns
    the matched index, or None (flag, don't guess) if nothing fits --
    deliberately does NOT fall back to an uncapitalized guess; a name this
    heuristic can't find should surface as the ordinary "no subject"
    flag, not a confident wrong one."""
    for idx in range(before_idx - 1, 0, -1):
        if idx in used_idxs:
            continue
        tok = tokens[idx]
        if tok and tok[0].isupper():
            return idx
    return None


def find_fallback_common_noun_subject(d_tokens: list[str], before_idx: int, used_idxs,
                                        assigned: list) -> tuple[int, str] | None:
    """2026-10-20, direct user bug report: "Kila kikundi kitapewa pesa
    nyingi" ("kitapewa": no noun won as nsubj) -- "kikundi" (group) has
    ZERO occurrences anywhere in this project's Bible-corpus source data
    (checked directly: not in the 319-word lexicon, not VERIFIED, not in
    the statistical table), so it was never even a NOUN CANDIDATE at all,
    the exact same shape of gap find_fallback_proper_name_subject()
    fixed for unknown proper names -- just for an unknown COMMON noun
    instead, which can't use capitalization as its signal.

    Heuristic: the nearest token STRICTLY BEFORE `before_idx` (the verb)
    that (a) is not already claimed by anything else (`used_idxs`/
    `assigned`, same discipline as the proper-name version), and (b) has
    a recognizable Swahili noun-class PREFIX SHAPE (guess_noun_class_by_
    prefix -- morphology only, no vocabulary needed, e.g. "ki-" -> class
    "7/8"). Unlike the proper-name version, sentence-initial position is
    NOT excluded here -- that exclusion exists there specifically because
    capitalization at sentence-start is uninformative (any word is
    capitalized there), but a noun-class PREFIX shape is just as
    informative whether or not the word starts the sentence. Returns
    (index, guessed_swahili_class) so the caller can apply this project's
    existing SWAHILI_CLASS_DEFAULT_SM_GUESS fallback for VERB AGREEMENT
    ONLY -- the noun itself still has no Kitaita translation and stays in
    the original Swahili, same as the proper-name case."""
    for idx in range(before_idx - 1, -1, -1):
        if idx in used_idxs or assigned[idx]:
            continue
        guess = guess_noun_class_by_prefix(d_tokens[idx])
        if guess:
            # 2026-10-20: guess_noun_class_by_prefix returns a combined "N/M"
            # pair key (e.g. "7/8" for ki-/vi-); SWAHILI_CLASS_DEFAULT_SM_GUESS
            # is keyed by the single class number -- same split-and-take-first
            # convention this file's own _sm_for() already uses elsewhere.
            return idx, guess[0].split("/")[0]
    return None


# 2026-10-22: direct user instruction -- "have another handling for
# names put them in the same class as mundu." Until now, proper-name
# handling only covered a verb's own SUBJECT (find_fallback_proper_
# name_subject, used for VERB agreement when no noun won nsubj); a name
# that needs to GOVERN some other concord -- an associative particle
# ("Sifuna wa sita", 'wa' needing Sifuna's class), an adjective, a
# possessive -- had no mechanism at all and fell through to plain,
# unconcorded passthrough (see docs/status.md 2026-10-22 for the
# "Sifuna wa sita" bug report this closes). 'u' is mundu's own Kitaita
# class marker (kitaita_verified_corrections.py: "'u' -- 'mundu' (mtu/
# person)"); Swahili/Kitaita proper names take class-1/ANIMATE-SG
# agreement regardless of which specific name it is, the same
# grammatical fact find_fallback_proper_name_subject() already relies
# on, just generalized here to ANY noun role, not only "subject of a
# verb".
PROPER_NAME_KITAITA_CLASS = "u"


def is_likely_proper_name(tok_original: str, tok_lower: str) -> bool:
    """A token capitalized in the ORIGINAL-CASE source text with no
    recognizable Bantu noun-class prefix shape (guess_noun_class_by_
    prefix) is almost certainly a proper name -- Swahili/Kitaita proper
    names don't carry noun-class morphology, so this is the
    complementary case to find_fallback_common_noun_subject()'s own
    "no vocabulary match, but a real prefix shape" signal: no
    vocabulary match AND no prefix shape AND capitalized. Deliberately
    does NOT exclude sentence-initial position the way find_fallback_
    proper_name_subject() does -- that exclusion exists there because
    it's scanning BACKWARD from a later verb, where capitalization at
    index 0 is uninformative on its own; here the prefix-shape check
    already does that job (an unknown COMMON noun at index 0 still
    usually has a recognizable shape, e.g. 'Kikundi'), so position
    doesn't need to be considered separately."""
    return bool(tok_original) and tok_original[0].isupper() and guess_noun_class_by_prefix(tok_lower) is None


def resolve_noun_kitaita_and_class(lemma_lower: str, idx: int, tokens: list[str], noun_kitaita: dict,
                                     noun_class_of: dict, plural_context: bool = False) -> tuple[str | None, str | None]:
    """lookup_noun_kitaita() + resolve_class_marker(), with a proper-name
    fallback when NEITHER has anything: an unrecognized, capitalized,
    non-Bantu-shaped token (is_likely_proper_name) is passed through
    UNCHANGED, in its ORIGINAL case (not the lowercased lemma -- a name
    must never be lowercased), and assigned PROPER_NAME_KITAITA_CLASS,
    instead of silently failing concord for the whole phrase it
    governs. The single choke point every noun-governed concord
    mechanism (associative, adjective, possessive) should resolve a
    GOVERNING/TARGET noun's Kitaita form+class through, so the name
    fallback only needs to be written once."""
    kit = lookup_noun_kitaita(lemma_lower, noun_kitaita, plural_context=plural_context)
    if kit:
        return kit, resolve_class_marker(kit, noun_class_of)
    original = tokens[idx] if 0 <= idx < len(tokens) else lemma_lower
    if is_likely_proper_name(original, lemma_lower):
        return original, PROPER_NAME_KITAITA_CLASS
    return None, None


def decompose_sentence_with_probes(text: str, noun_kitaita: dict, noun_class_of: dict,
                                    root_tables=None) -> Decomposition:
    """root_tables: see decompose_sentence()'s matching parameter -- same
    optional full tables tuple, same default-None no-op behavior.

    2026-10-09, direct user directive: "we don't need to guess for verbs
    and adjectives etc. We already have probes to do that for us." Replaces
    decompose_sentence()'s adjacency heuristic for SUBJECT/OBJECT/adjective-
    target identification with the trained NLLB relation probes (nllb_probe_
    identification.py).

    2026-10-09 UPDATE: candidate detection now ALSO uses the trained POS
    probe (pos_probe.py), not just the lexicon/decompose_verb()/ADJ_ROOTS
    lookup tables. Found directly: "mdogo" (a real Swahili adjective, just
    not one of the 2 -- kubwa/baya -- this project has a Kitaita concord
    table for) was silently swallowed as an unrelated NOUN, because the
    old candidate detection ran noun-lookup first and had no way to know
    "mdogo" was grammatically an adjective in this sentence at all; it
    just happened to also have a real (but here, wrong-ROLE) noun-table
    entry from its own separate substantive usage elsewhere in the corpus.
    The POS probe tags every token from NLLB's own representations BEFORE
    noun/adjective candidacy is decided: a token tagged ADJ is excluded
    from noun candidacy outright (even if it has a noun-table entry) and
    becomes an adjective candidate regardless of whether it's in ADJ_ROOTS
    -- so an adjective this project has no Kitaita concord data for is now
    correctly flagged as "identified as modifying X, no concord rule yet"
    instead of being confidently mistranslated as an unrelated noun. If
    the POS probe isn't available (not yet trained, or NLLB alignment
    fails for this sentence), falls back to the lookup-only detection,
    flagged explicitly, never silently."""
    from nllb_probe_identification import identify_relations, tag_pos

    tokens = tokenize(text)
    d = Decomposition(tokens=[t.lower() for t in tokens])
    assigned = [False] * len(tokens)

    # 2026-10-19: descriptive/purposive associative concord ("maji ya kunywa"
    # -> "machi ghenywa") -- see decompose_sentence()'s identical call (top
    # of that function) for the full reasoning: this MUST run before ANY
    # candidate-detection loop below, including the noun_hits one a few
    # lines down, whose _noun_lookup() treats "ku-"+infinitive tokens as a
    # class-15 deverbal-noun candidate too (a real, pre-existing Swahili
    # ambiguity, not new) and would otherwise claim the infinitive first.
    if root_tables is not None:
        apply_associative_descriptor_concord(d, noun_kitaita, noun_class_of, root_tables, assigned)

    # 2026-10-11 (cont. 8): direct user instruction -- "Na should be a
    # dictionary look up since it does not have concord attached to it."
    # Closed-class function words (currently just the conjunction 'na',
    # "and"/"with") carry no agreement at all, so they don't belong in
    # the noun/verb/adjective candidate race -- checked FIRST and removed
    # from candidacy entirely, rather than relying on them losing every
    # race by chance. 'na' -> 'na' is not a guess: it appears repeatedly
    # as the literal conjunction in the real Kitaita Bible corpus text
    # already quoted elsewhere in this project (e.g. "isanga wulacha na
    # wushapa ghwaro", "meta inya na pointi inya", "wulacha meta ikumi na
    # iwi, na wushapa meta iwi") -- a shared Bantu cognate, not aligned
    # via Dice/statistics like the noun/adjective tables.
    for i, tok in enumerate(d.tokens):
        if tok in FUNCTION_WORD_TRANSLATIONS:
            d.secondary_nouns.append((i, tok, FUNCTION_WORD_TRANSLATIONS[tok]))
            d.flags.append(f"'{tok}': closed-class function word, plain dictionary substitution "
                            f"-> '{FUNCTION_WORD_TRANSLATIONS[tok]}' (no concord attached).")
            assigned[i] = True

    pos_tags = tag_pos(tokens)
    if pos_tags is None:
        d.flags.append("POS probe unavailable for this sentence (not trained yet, or NLLB alignment "
                        "failed) -- falling back to lookup-only noun/adjective candidate detection.")

    noun_hits = []
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue  # 2026-10-19: already claimed (descriptive-associative fusion, or a
            # closed-class function word above) -- must not also become a noun candidate.
        if any(tok == root or any(tok == p + root for p in ADJ_PREFIXES if p) for root in ADJ_ROOTS):
            continue  # 2026-10-15: found via "Mayai ya kuku ni mazuri." -- "mazuri" (zuri/good,
            # a KNOWN adjective root this project has a verified concord table for) was getting
            # claimed here FIRST via a near-noise statistical noun-table match (Dice 0.26, n=1,
            # "ghiboie") before the adjective_hits loop below -- which WOULD have correctly
            # resolved it via ZURI_ROOT -- ever got a chance to see it (that loop only looks at
            # tokens not already `assigned`). Same "known-root check wins regardless of POS tag"
            # discipline the adjective_hits loop already applies to itself, just checked earlier
            # so it actually has an effect.
        if pos_tags is not None and pos_tags[i] == "ADJ":
            continue  # a token the POS probe confidently tags ADJ is never a noun candidate,
            # even if it happens to also have a (different-usage) noun-table entry.
        if pos_tags is not None and pos_tags[i] == "VERB":
            # 2026-10-15: found while testing the new person-marker recognition --
            # "mnapenda" (we like) is correctly POS-tagged VERB here, but
            # _noun_lookup's confidence filter was removed project-wide (2026-10-09,
            # see its own docstring) so it ALSO matched a near-noise statistical
            # noun candidate (Dice 0.29, 1 attestation) with no class at all. The
            # noun_hits loop claimed it FIRST (it runs before verb_hits below),
            # removing it from verb candidacy entirely and silently breaking
            # recognition of a real, newly-supported 2pl pro-drop sentence. Same
            # veto the ADJ check above already applies, mirrored for VERB: only a
            # CONFIDENT VERB tag excludes noun candidacy (not "anything other than
            # NOUN"), same asymmetric-trust reasoning the verb_hits loop below
            # already uses for its own NOUN veto (POS probe's class recall isn't
            # perfect in either direction).
            continue
        hit = _noun_lookup(tok, noun_kitaita)
        if hit is not None:
            cls, evidence = hit
            noun_hits.append((i, tok, cls, evidence))
            assigned[i] = True

    # 2026-10-22: direct user instruction -- "have another handling for
    # names put them in the same class as mundu." A proper name (no
    # vocabulary entry, capitalized, no Bantu noun-class prefix shape --
    # see is_likely_proper_name()) is registered as a noun candidate too,
    # not just as the verb-subject fallback (find_fallback_proper_name_
    # subject) -- otherwise it can never be the TARGET of an adjective or
    # possessive in a verbless NP ("Sifuna wa sita" needed this for the
    # associative particle specifically, fixed separately in apply_
    # associative_concord(); this closes the matching gap for adjective/
    # possessive concord, e.g. a hypothetical "Sifuna mzuri"). Kept in its
    # own pass, after the main lookup loop, so a real vocabulary match
    # (assigned[i] already True) always wins first -- this only catches
    # what nothing else claimed. Original case preserved as the lemma
    # (NOT lowercased `tok`) since a name must round-trip unchanged.
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue
        # 2026-10-22 (same-day fix): direct user bug report -- a sentence-
        # initial capitalized VERB ("Ninawapenda ndege...") was being
        # wrongly swallowed as a "proper name" here, BEFORE the verb_hits
        # loop below ever got a chance to see it. Checking real
        # decomposability (not the POS probe's own VERB/ADJ tag -- tried
        # first, but found directly to be UNSAFE here: it also tags the
        # genuine name "Sifuna" as VERB, a false positive that would have
        # wrongly re-excluded the ORIGINAL "Sifuna wa sita" fix this same
        # session; morphological decomposability is the more precise,
        # grounded signal) is the same "known pattern wins regardless of
        # position" discipline the noun_hits loop above already applies
        # to ADJ_ROOTS -- a token that genuinely decomposes as SM+TAM+
        # root+FV is never a name, checked BEFORE is_likely_proper_name()
        # gets to claim it.
        if decompose_verb_or_copula(tok) is not None:
            continue
        if is_likely_proper_name(tokens[i], tok):
            noun_hits.append((i, tokens[i], "ANIMATE-SG",
                               "proper name (capitalized, no Bantu class-prefix shape) -- "
                               "Kitaita class 'u' (mundu), per PROPER_NAME_KITAITA_CLASS"))
            assigned[i] = True

    # 2026-10-11 (cont. 8): direct user instruction -- "wire in possessive-
    # pronoun detection and why wasn't it wired together with the
    # adjectives since it is adjective shaped." Checked directly: the POS
    # probe tags a possessive like 'vyako' as 'OTHER', not 'ADJ' (gold UD/
    # SUD treat possessives as DET/PRON, a different category from true
    # adjectives even though the surface morphology -- concord prefix +
    # stem -- looks the same), so the adjective sweep below (which gates
    # on pos_tags[i]=="ADJ") structurally cannot catch these regardless of
    # detection logic -- this needs its own morphological check, run here
    # (before verb_hits, so a short possessive token is never mistaken for
    # a verb) via find_possessive_pronoun() (POSS_TABLE prefix + a closed
    # set of 6 person-stems).
    possessive_hits = []  # (idx, prefix, person_stem)
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue
        hit = find_possessive_pronoun(tok)
        if hit is not None:
            prefix, stem = hit
            possessive_hits.append((i, prefix, stem))
            assigned[i] = True

    verb_hits = []
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue
        # 2026-10-10: direct user-prompted check found decompose_verb's
        # surface-pattern match alone is NOT verb detection -- it spuriously
        # parses real nouns that happen to collide with the SM+TAM+root+FV
        # shape (e.g. 'ukimwi'='AIDS' as u-ki-mw-i, 'wakimbizi'='refugees'
        # as wa-ki-mbiz-i; confirmed against gold UPOS, 50/319 = 15.7% of
        # decompose_verb's own '-ki-' hits across SWH_UDT+AfriSUD are
        # gold-tagged NOUN/PROPN, not VERB). 'ukimwi' specifically is NOT
        # in this project's noun lexicon, so the noun_hits loop above never
        # claims it first -- without this check it would reach here and be
        # wrongly treated as a live '-ki-' verb needing subject resolution.
        # Same veto pattern as the ADJ check above: only a CONFIDENT NOUN
        # tag vetoes candidacy (not "anything other than VERB"), since the
        # POS probe's own VERB recall is 91.4%, not perfect -- requiring
        # positive VERB agreement would risk new false negatives on real
        # verbs the probe is merely unsure about.
        if pos_tags is not None and pos_tags[i] == "NOUN":
            continue
        v = decompose_verb_or_copula(tok)
        if v is not None and len(v.root_guess) >= 2:
            verb_hits.append((i, v))
            assigned[i] = True

    adjective_hits = []  # (idx, stem) -- no adjacency restriction; the probe resolves the target
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue
        if tok in OTE_TOKENS:
            continue  # 2026-10-20: found via "Siku zote za maisha yangu." -- same "known
            # closed-set wins over POS tag" veto as the demonstrative one just below
            # ("zote" is a quantifier the POS probe plausibly tags ADJ, which would let it
            # win position-1 here with NO real adjective-concord table and silently block
            # apply_ote_concord()'s correct "ghose"/"rose"/etc. substitution for the SAME
            # token -- the exact silent-vanishing bug this fix addresses).
        if tok in SWAHILI_DEM_PROXIMAL or tok in SWAHILI_DEM_REFERENTIAL or tok in SWAHILI_DEM_DISTAL:
            continue  # 2026-10-15: found via "Kazi hii ni nzuri." -- the POS probe tags
            # demonstratives ("hii"/"ile"/etc.) as ADJ (a reasonable confusion, demonstratives
            # DO modify nouns the way adjectives do), which let "hii" win a spot in
            # adjective_hits and then, because clauses are assembled BEFORE secondary_nouns,
            # win the position-1 slot with NO real adjective concord (there's no table for
            # "hii" as an adjective) -- silently blocking apply_demonstrative_concord()'s
            # correct "iri" substitution for the SAME token. Same "known closed-set wins
            # over POS tag" discipline as the ADJ_ROOT-vs-noun_hits veto earlier this
            # session, just the other direction (demonstratives vetoed FROM adjective
            # candidacy, not an adjective veto INTO it).
        # Known-root check FIRST regardless of POS tag, so a prefixed form like
        # "mkubwa" still resolves to its real stem "kubwa" (needed for translate()'s
        # concord rules) rather than being stored as the raw, unstripped token.
        matched_root = next((root for root in ADJ_ROOTS
                              if tok == root or any(tok == p + root for p in ADJ_PREFIXES if p)), None)
        if matched_root is not None:
            adjective_hits.append((i, matched_root))
            assigned[i] = True
        elif pos_tags is not None and pos_tags[i] == "ADJ":
            # POS probe confidently says ADJ, but it's not one of the 2 roots this
            # project has a concord table for -- kept as its own stem (the real
            # word) so translate()'s concord lookup correctly flags it as
            # unsupported rather than silently matching nothing and vanishing.
            adjective_hits.append((i, tok))
            assigned[i] = True

    if not verb_hits:
        d.flags.append("NO VERB found -- cannot translate.")
    else:
        # Kept as the FIRST clause's values for backward compatibility with
        # code that only ever expected one clause (the "d.verb is None"
        # early-exit checks). The real, per-verb work happens below.
        vi0, v0 = verb_hits[0]
        d.verb_idx, d.verb_surface = vi0, tokens[vi0]
        d.verb = {"sm": v0.subject_marker, "tam": v0.tam_marker, "root": v0.root_guess, "fv": v0.final_vowel}

    if not noun_hits:
        if not verb_hits:
            # 2026-10-21: direct crash found while testing ("Tulipenda." -> IndexError,
            # `verb_hits[0]` below unconditionally dereferenced) -- every branch below
            # this one needs at least one verb, so when verb_hits is ALSO empty (not
            # just noun_hits), none of them fire and control falls through here
            # regardless. Same honest "flag, don't guess" outcome `decompose_sentence()`'s
            # own top-level `if not verb_hits:` check already gives (line ~897 above) --
            # this path just didn't have an equivalent guard. The underlying "why did
            # verb_hits come back empty for a verb decompose_verb_or_copula() recognizes
            # fine in isolation" question (POS-probe mistagging a short/unusual sentence,
            # same risk class as the noun-table false-positive bug elsewhere) is NOT
            # investigated further here -- this only stops the crash.
            d.flags.append("NO NOUN found and NO VERB found either -- cannot translate.")
        else:
            # 2026-10-22: GENERALIZED from a single `verb_hits[0]`-only special
            # case to a loop over EVERY verb -- direct user bug report:
            # "Ninaweza kuruka kamba nao." silently dropped "kuruka" (to jump)
            # entirely, keeping only "Ninaweza" (I can). Root cause, confirmed
            # directly (not guessed): `verb_hits` genuinely contained BOTH
            # verbs, correctly decomposed -- this whole branch (reached because
            # "kamba"/rope has no noun-table entry at all, so noun_hits is
            # empty) was written assuming at most one verb and simply never
            # looked past `verb_hits[0]`, so every verb after the first was
            # silently discarded whenever a sentence had NO recognized noun
            # anywhere in it. Found while testing simple non-Bible text
            # (African Storybook corpus, 2026-10-22) -- the Bible corpus
            # itself rarely has a noun-less multi-verb clause like "weza +
            # infinitive + [an object this project's vocabulary doesn't have]",
            # so this gap never surfaced there. Same per-verb fallback tiers
            # as before (negated+person / negated / person / proper-name /
            # give up), just applied to each verb in turn instead of only the
            # first. `chain_subject_idx` reused across iterations for the
            # SAME reason the main (noun_hits-non-empty) branch below already
            # does -- a fallback proper name found for one clause should
            # carry over to the rest of the chain, not be re-scanned fresh.
            chain_subject_idx = None
            for vi0, v0 in verb_hits:
                verb0 = {"sm": v0.subject_marker, "tam": v0.tam_marker, "root": v0.root_guess,
                         "fv": v0.final_vowel}
                if v0.negated and v0.person is not None:
                    d.negated = True
                    d.person = v0.person
                    d.flags.append(f"NEGATION detected ('{tokens[vi0]}', subject marker '{v0.subject_marker}-', "
                                    f"Swahili negative-present, person '{d.person}') -- Kitaita negation marker "
                                    f"native-speaker-submitted 2026-10-15 (Kitaita Word Bank), translating.")
                    d.clauses.append({"verb_idx": vi0, "verb_surface": tokens[vi0], "verb": verb0,
                                       "subject": None, "subject_idx": None, "subject_class": None,
                                       "object": None, "object_idx": None, "adjective": None, "adjective_idx": None,
                                       "person": d.person, "negated": True})
                elif v0.negated:
                    d.negated = True
                    d.flags.append(f"NEGATION detected ('{tokens[vi0]}'), but no noun subject and "
                                    f"no person marker either -- same coreference limitation as affirmative "
                                    f"3rd-person pro-drop (out of scope); cannot translate this verb.")
                elif v0.person is not None:
                    d.person = v0.person
                    d.flags.append(f"'{tokens[vi0]}': NO NOUN found -- but subject marker '{v0.subject_marker}-' "
                                    f"is a recognized PERSON prefix ('{d.person}') -- translating as a "
                                    f"pro-dropped pronoun subject instead of refusing outright.")
                    d.clauses.append({"verb_idx": vi0, "verb_surface": tokens[vi0], "verb": verb0,
                                       "subject": None, "subject_idx": None, "subject_class": None,
                                       "object": None, "object_idx": None, "adjective": None, "adjective_idx": None,
                                       "person": d.person})
                else:
                    # 2026-10-20: see find_fallback_proper_name_subject()'s own docstring.
                    # `used_noun_idxs` doesn't exist in this branch (there are NO noun
                    # hits at all here) -- nothing to exclude, pass an empty set.
                    reused = chain_subject_idx is not None
                    name_idx = chain_subject_idx if reused else find_fallback_proper_name_subject(tokens, vi0, set())
                    if name_idx is not None:
                        chain_subject_idx = name_idx
                        d.subject_class = "ANIMATE-SG"
                        how = "established earlier in this clause chain" if reused else "capitalized, not sentence-initial"
                        d.flags.append(f"'{tokens[vi0]}': NO NOUN found -- but '{tokens[name_idx]}' ({how}) is "
                                        f"almost certainly an unrecognized proper name acting as the subject -- "
                                        f"using this project's class-1/ANIMATE-SG default instead of leaving the "
                                        f"verb's Swahili subject marker unconverted (see "
                                        f"find_fallback_proper_name_subject()).")
                        d.clauses.append({"verb_idx": vi0, "verb_surface": tokens[vi0], "verb": verb0,
                                           "subject": None, "subject_idx": None, "subject_class": "ANIMATE-SG",
                                           "object": None, "object_idx": None, "adjective": None, "adjective_idx": None,
                                           "person": None})
                    elif v0.subject_marker == "ku" and root_tables is not None:
                        # 2026-10-22: direct user bug report, SAME sentence as the
                        # multi-clause-drop fix above ("Ninaweza kuruka kamba
                        # nao.") -- even once "kuruka" got its own loop iteration,
                        # it still fell all the way to "no subject" here, because
                        # an INFINITIVE has no subject to resolve in the first
                        # place -- "ku-" here is the bare infinitive marker
                        # (decompose_verb_or_copula's own "ku"+root+fv, tam_label
                        # 'infinitive/bare'), NOT a finite class-15 noun subject;
                        # it shares its subject with the MATRIX verb ("Ninaweza"),
                        # exactly like "maji ya kunywa"'s infinitive complement
                        # already handled by apply_associative_descriptor_concord()
                        # (SAME resolve_verb_root() reuse, same reasoning: no
                        # subject to resolve agreement from, but the ROOT still
                        # gets this project's best available translation). Unlike
                        # that construction, the "ku-" marker ITSELF is KEPT here
                        # (not dropped/fused) -- this is a plain dependent
                        # infinitive, not a particle+verb fusion.
                        # 2026-10-23: direct user bug report ("kutafuta" ->
                        # "nderiwuye") -- decompose_verb() greedily matches a
                        # TAM string even after "ku-", so a root beginning
                        # with a string that's ALSO a real TAM marker (e.g.
                        # "ta"/"na"/"ki"/"ka"/"li"/"da") gets wrongly split
                        # ("kutafuta" -> ku+TAM('ta')+root('fut'), not
                        # ku+root('tafut')) -- see VerbFormEntry.ku_
                        # infinitive_alternate's own comment for the full
                        # ambiguity (Swahili "ku-" is ALSO class-17/locative,
                        # which genuinely does take a TAM, e.g. "Kutakuwa na
                        # njaa" -- so this can't just be disabled outright).
                        # Reaching THIS branch means "ku-" has already been
                        # read as the infinitive marker (that's this
                        # branch's own condition), so there is nothing left
                        # to weigh -- the alternate (whole `rest`, no TAM
                        # stripped) is simply the correct root for that
                        # reading, used whenever decompose_verb() found one.
                        root_guess, final_vowel = (v0.ku_infinitive_alternate
                                                     if v0.ku_infinitive_alternate else
                                                     (v0.root_guess, v0.final_vowel))
                        kit_root, root_tier, is_full_form = resolve_verb_root(root_guess, root_tables)
                        inf_word = kit_root if is_full_form else f"ku{kit_root}{final_vowel}"
                        d.secondary_nouns.append((vi0, tokens[vi0], inf_word))
                        corrected_note = (f" (corrected from the greedy TAM-consuming parse, root "
                                            f"'{v0.root_guess}' -- see VerbFormEntry.ku_infinitive_alternate)"
                                            if v0.ku_infinitive_alternate else "")
                        d.flags.append(f"'{tokens[vi0]}': bare infinitive complement (shares its subject "
                                        f"with the matrix verb, nothing to resolve here){corrected_note} -> "
                                        f"'{inf_word}' (root tier: {root_tier})")
                    elif v0.root_guess == "oko_exist":
                        # 2026-10-22: bug #2 fix -- the invariant locative-
                        # existential "oko" (see EXISTENTIAL_LOCATIVE_FORMS)
                        # needs NO subject at all (translate()'s own
                        # 'oko_exist' bypass), so it must not fall through
                        # to the "cannot be translated" catch-all just
                        # because this branch (no noun anywhere in the
                        # sentence) has nothing to default a CLASS to --
                        # unlike the "a"/"wa" case below, there is no class
                        # defaulting decision being made here at all.
                        d.clauses.append({"verb_idx": vi0, "verb_surface": tokens[vi0], "verb": verb0,
                                           "subject": None, "subject_idx": None, "subject_class": None,
                                           "object": None, "object_idx": None, "adjective": None, "adjective_idx": None,
                                           "person": None})
                    elif v0.subject_marker in ("a", "wa"):
                        # 2026-10-22: direct user instruction -- handle 3rd-
                        # person pro-drop with no antecedent "similar to how
                        # we handled... 1st and 2nd person markers" (the
                        # personal-pronoun-defaults-to-mundu's-class fix,
                        # earlier this session). NOT a coreference-resolution
                        # mechanism (finding the REAL referent is still out
                        # of scope, and not attempted here) -- this uses the
                        # SWAHILI MARKER'S OWN MEANING as the default: "a-" IS
                        # Swahili's class-1/ANIMATE-SG marker and "wa-" IS its
                        # class-2/ANIMATE-PL marker (SM_TABLE), so a referent-
                        # less "a-"/"wa-" verb overwhelmingly means an
                        # unspecified human subject (he/she or they) -- the
                        # SAME class mundu/wandu already use, not a guess
                        # across unrelated classes. Found via the African
                        # Storybook test corpus ("Anajinyoosha."/"Anaita."/
                        # "Anasikiliza." -- dialogue-style text pro-drops 3rd
                        # person far more often than Bible narrative, where a
                        # noun or "-ka-" chain-continuation usually carries
                        # the subject across clauses instead).
                        guessed_class = "u" if v0.subject_marker == "a" else "wi"
                        d.subject_class = "ANIMATE-SG" if v0.subject_marker == "a" else "ANIMATE-PL"
                        d.flags.append(f"'{tokens[vi0]}': NO NOUN found and no antecedent for this 3rd-"
                                        f"person pro-drop (he/she/they) -- defaulting to class "
                                        f"'{guessed_class}-' (mundu/wandu), the class Swahili's OWN "
                                        f"'{v0.subject_marker}-' marker itself encodes -- an UNCONFIRMED "
                                        f"GUESS about WHICH noun, not about which class, flagged for "
                                        f"native-speaker review, not a real coreference resolution.")
                        d.clauses.append({"verb_idx": vi0, "verb_surface": tokens[vi0], "verb": verb0,
                                           "subject": None, "subject_idx": None, "subject_class": d.subject_class,
                                           "object": None, "object_idx": None, "adjective": None, "adjective_idx": None,
                                           "person": None})
                    else:
                        d.flags.append(f"'{tokens[vi0]}': NO NOUN found at all -- no subject, so this verb "
                                        f"cannot be translated. (3rd-person pro-drop with no antecedent in "
                                        f"this sentence is a coreference problem, out of scope.)")
        # 2026-10-22: direct user audit request found "Mdogo wangu amelala."
        # -> "Udalala." -- 'Mdogo'/'wangu' vanished with NO flag at all, not
        # even "no mechanism". SAME root cause the "elif not verb_hits:"
        # branch below already documents for adjective_hits/possessive_hits
        # (both get `assigned[i]=True` unconditionally up above, so the
        # generic "no mechanism" passthrough sweep at the end of this
        # function never sees them either) -- just never fixed for THIS
        # branch (verb_hits non-empty, noun_hits empty) too. Simpler than
        # that branch's fix: there is no noun at all here by definition
        # (that's this whole branch's condition), so there is no "nearest
        # preceding noun" to even try finding a concord target against --
        # straight passthrough. Runs once per SENTENCE (outside the per-verb
        # loop above), not once per verb.
        for ai, stem in adjective_hits:
            d.passthrough.append((ai, tokens[ai]))
            d.flags.append(f"'{tokens[ai]}': NO NOUN found anywhere in this sentence for this "
                            f"adjective to concord against -- kept as the original Swahili word.")
            record_slot(d, ai, "adjective", tokens[ai], None, needs_rewrite=True,
                        predicted_form=None, confidence="no-noun-in-sentence")
        for pi, prefix, stem in possessive_hits:
            d.passthrough.append((pi, tokens[pi]))
            d.flags.append(f"'{tokens[pi]}': NO NOUN found anywhere in this sentence for this "
                            f"possessive to concord against -- kept as the original Swahili word.")
            record_slot(d, pi, "possessive", tokens[pi], None, needs_rewrite=True,
                        predicted_form=None, confidence="no-noun-in-sentence")
    elif not verb_hits:
        for oi, olemma, _, _ in noun_hits:
            plural_ctx = _siku_plural_context(d.tokens, oi) if olemma == "siku" else False
            kit = lookup_noun_kitaita(olemma, noun_kitaita, plural_context=plural_ctx)
            d.secondary_nouns.append((oi, olemma, kit))
        # 2026-10-20: direct user report -- a verbless NP (e.g. "Siku zote za
        # maisha yangu.") was silently DROPPING every adjective_hits/
        # possessive_hits token instead of translating or passing it through:
        # both lists get `assigned[i]=True` unconditionally up above (so they
        # never reach the generic "no mechanism" passthrough sweep either),
        # but this branch never visited them. No probe.adjective_targets/
        # possessive_targets exist without a verb clause, so target
        # resolution falls back to the NEAREST PRECEDING noun -- the same
        # "the noun right before it" heuristic apply_associative_concord/
        # apply_ote_concord already use for their own adjacent particle/
        # quantifier.
        for ai, stem in adjective_hits:
            target = max((h for h in noun_hits if h[0] < ai), key=lambda h: h[0], default=None)
            # 2026-10-22: resolve_noun_kitaita_and_class() (not a plain
            # lookup_noun_kitaita()) so a proper-name target ("Sifuna
            # mzuri") still gets a class via the mundu-class fallback --
            # see its own docstring / is_likely_proper_name().
            target_kitaita, target_class = ((None, None) if not target else
                resolve_noun_kitaita_and_class(
                    target[1], target[0], tokens, noun_kitaita, noun_class_of,
                    plural_context=_siku_plural_context(d.tokens, target[0]) if target[1] == "siku" else False))
            adj_part, note = ((None, None) if not target_kitaita else
                               compute_adjective_concord(target_kitaita, target[1], stem, noun_class_of, noun_kitaita,
                                                           override_class=target_class))
            if adj_part:
                d.secondary_nouns.append((ai, tokens[ai], adj_part))
                d.flags.append(f"'{tokens[ai]}' -> '{adj_part}': verbless NP, nearest-preceding-noun "
                                f"target '{target_kitaita}' ({note}).")
            else:
                d.passthrough.append((ai, tokens[ai]))
                d.flags.append(f"'{tokens[ai]}': verbless NP, no adjective-concord target/data -- kept "
                                f"as the original Swahili word.")
            record_slot(d, ai, "adjective", tokens[ai], target[0] if target else None, needs_rewrite=True,
                        predicted_form=adj_part, confidence=note if adj_part else "no-target-or-data")
        for pi, prefix, stem in possessive_hits:
            target = max((h for h in noun_hits if h[0] < pi), key=lambda h: h[0], default=None)
            target_kitaita, kit_sm = ((None, None) if not target else
                resolve_noun_kitaita_and_class(
                    target[1], target[0], tokens, noun_kitaita, noun_class_of,
                    plural_context=_siku_plural_context(d.tokens, target[0]) if target[1] == "siku" else False))
            has_poss_data = kit_sm in POSS_PERSON_FORM_BY_CLASS and stem in POSS_PERSON_FORM_BY_CLASS[kit_sm]
            poss_part = POSS_PERSON_FORM_BY_CLASS[kit_sm][stem] if has_poss_data else None
            if poss_part:
                d.secondary_nouns.append((pi, tokens[pi], poss_part))
                d.flags.append(f"'{tokens[pi]}' -> '{poss_part}': verbless NP, nearest-preceding-noun "
                                f"target '{target_kitaita}', class '{kit_sm}-' concord "
                                f"(POSS_PERSON_FORM_BY_CLASS).")
            else:
                d.passthrough.append((pi, tokens[pi]))
                d.flags.append(f"'{tokens[pi]}': verbless NP, possessive target's class "
                                + (f"'{kit_sm}-' has no confirmed concord data" if kit_sm else "is unresolved")
                                + " -- kept as the original Swahili word.")
            record_slot(d, pi, "possessive", tokens[pi], target[0] if target else None, needs_rewrite=True,
                        predicted_form=poss_part, confidence="verified" if poss_part else "no-target-or-data")
    else:
        probe = identify_relations(tokens, [h[0] for h in noun_hits], [h[0] for h in verb_hits],
                                    [h[0] for h in adjective_hits],
                                    [h[0] for h in possessive_hits])
        for note in probe.notes:
            d.flags.append(note)

        used_noun_idxs = set()
        adjective_claimed = set()
        # 2026-10-20: see find_fallback_proper_name_subject()'s own docstring --
        # once a fallback proper-name subject is found for ONE clause in this
        # sentence, REUSE that same index for every later clause that also
        # fails nsubj, rather than re-scanning fresh each time. Found directly
        # necessary, not a hypothetical: on "...akamwomba BWANA akasema", the
        # nsubj/obj probe itself mis-assigned "BWANA"'s own object slot to the
        # wrong noun (a separate, pre-existing probe-accuracy issue, not this
        # fix's to solve), leaving "BWANA" un-claimed in `used_noun_idxs` -- a
        # fresh re-scan for 'akasema' then wrongly preferred the NEARER
        # "BWANA" over the correct, chain-continuing "Hezekia". A Swahili
        # "-ka-" narrative chain's subject is established ONCE and carries
        # over across the whole chain unless a noun explicitly wins nsubj for
        # a later clause (that case is handled separately, above, and always
        # takes priority over this fallback).
        chain_subject_idx = None

        for vi, v in verb_hits:
            vr = probe.verb_results.get(vi, {"subject_idx": None, "object_idx": None})
            subj_i, obj_i = vr["subject_idx"], vr["object_idx"]
            clause = {"verb_idx": vi, "verb_surface": tokens[vi],
                      "verb": {"sm": v.subject_marker, "tam": v.tam_marker, "root": v.root_guess, "fv": v.final_vowel,
                               "om_candidates": v.object_marker_candidates,
                               "rel_candidates": v.relative_marker_candidates,
                               "negative_subjunctive_candidate": v.negative_subjunctive_candidate,
                               "ku_infinitive_alternate": v.ku_infinitive_alternate},
                      "subject": None, "subject_idx": None, "subject_class": None,
                      "object": None, "object_idx": None, "adjective": None, "adjective_idx": None,
                      "negated": v.negated}
            if v.negated:
                d.negated = True
            if subj_i is not None:
                si, slemma, scls, sevidence = next(h for h in noun_hits if h[0] == subj_i)
                clause["subject"], clause["subject_idx"], clause["subject_class"] = slemma, si, scls
                used_noun_idxs.add(si)
                # 2026-10-20: a REAL, probe-won subject always updates the fallback
                # "chain memory" too (not just the proper-name case setting it) --
                # if the topic genuinely shifts to a different, recognized noun
                # mid-chain, a LATER clause's fallback (if it ever needs one) should
                # continue from THIS most-recent subject, not an earlier one.
                chain_subject_idx = None
                d.flags.append(f"'{slemma}' -> probe-identified SUBJECT of '{tokens[vi]}' (beat every "
                                f"other noun as nsubj)." + (" NEGATED verb." if v.negated else ""))
            elif v.negated and v.person is not None:
                # 2026-10-15: no noun won nsubj for THIS verb, but it's a recognized
                # person-marked negative -- translate via the person path, same
                # mechanism as the plain (affirmative) person branch below.
                clause["person"] = v.person
                d.flags.append(f"NEGATION detected ('{tokens[vi]}', subject marker '{v.subject_marker}-', "
                                f"person '{v.person}') -- Kitaita negation marker native-speaker-submitted "
                                f"2026-10-15 (Kitaita Word Bank), translating.")
            elif v.negated:
                d.flags.append(f"NEGATION detected ('{tokens[vi]}'), but no noun won nsubj and no person "
                                f"marker either -- same coreference limitation as affirmative 3rd-person "
                                f"pro-drop (out of scope); cannot translate this verb.")
            elif v.person is not None:
                # 2026-10-15: no NOUN won nsubj (there may be other nouns in this
                # sentence entirely unrelated to this verb -- e.g. its own object),
                # but this verb's own subject marker is a recognized 1st/2nd-person
                # prefix -- translate it as a pro-dropped pronoun subject instead of
                # giving up, same mechanism as the "no noun hits at all" branch above.
                clause["person"] = v.person
                d.flags.append(f"'{tokens[vi]}': no noun won as nsubj, but subject marker "
                                f"'{v.subject_marker}-' is a recognized PERSON prefix ('{v.person}') -- "
                                f"translating as a pro-dropped pronoun subject instead of refusing.")
            else:
                # 2026-10-20: see find_fallback_proper_name_subject()'s own docstring
                # (direct bug report, "Hezekia akajigeuza...") -- tried only after
                # the probe genuinely found no winning noun, never overriding it.
                # `chain_subject_idx` (set above the verb_hits loop): reuse an
                # ALREADY-established fallback name across the rest of this
                # sentence's clause chain instead of re-scanning fresh each time
                # -- see its own comment for why a fresh re-scan can pick the
                # wrong, nearer name once the probe has already mis-claimed (or
                # failed to claim) something else nearby.
                reused = chain_subject_idx is not None
                name_idx = chain_subject_idx if reused else \
                    find_fallback_proper_name_subject(tokens, vi, used_noun_idxs)
                if name_idx is not None:
                    chain_subject_idx = name_idx
                    clause["subject_class"] = "ANIMATE-SG"
                    how = "established earlier in this clause chain" if reused else "capitalized, unclaimed, not sentence-initial"
                    d.flags.append(f"'{tokens[vi]}': no noun won as nsubj, but '{tokens[name_idx]}' "
                                    f"({how}) is almost certainly an unrecognized proper name acting as "
                                    f"the subject -- Swahili grammar requires ANY human name to take "
                                    f"class-1/ANIMATE-SG verb agreement regardless of which name it is, "
                                    f"so this project's existing per-class default ('u-'/SWAHILI_CLASS_"
                                    f"DEFAULT_SM_GUESS) is used instead of leaving the verb's Swahili "
                                    f"subject marker unconverted. The name itself stays in the output as "
                                    f"the original Swahili word (no translation exists or is attempted "
                                    f"for it).")
                elif v.subject_marker == "ku" and root_tables is not None:
                    # 2026-10-22: direct user audit request found this is
                    # the SAME gap as the "a"/"wa" one just above, for the
                    # #3 infinitive-complement fix instead of #4's class
                    # default: "Lakini siwezi kupaa nao angani." ('kupaa'
                    # SKIPPED ENTIRELY) and "...kutafuta chakula." both have
                    # some OTHER noun in the sentence (making noun_hits
                    # non-empty), so they never reached the "if not
                    # noun_hits:" branch where the infinitive-complement
                    # case already lives -- same resolve_verb_root() reuse,
                    # same reasoning (shares its subject with the matrix
                    # verb, nothing to resolve here). Uses `continue` (skips
                    # this verb's own object-claim below, unlike the other
                    # branch which has no noun_hits to claim from at all) --
                    # acceptable here because an infinitive's object, if
                    # any, is typically ALSO the matrix verb's own object
                    # and gets claimed there instead (confirmed directly for
                    # both audited cases); a genuinely verb-specific object
                    # missed this way is a known, accepted gap, not silently
                    # assumed impossible.
                    # 2026-10-23: direct user bug report ("kutafuta" ->
                    # "nderiwuye") -- see the "if not noun_hits:" branch's
                    # matching fix (same shape) and VerbFormEntry.ku_
                    # infinitive_alternate's own comment for the full
                    # ambiguity. Reaching THIS branch means "ku-" has
                    # already been read as the infinitive marker, so the
                    # alternate (whole `rest`, no TAM stripped) is simply
                    # the correct root for that reading.
                    root_guess, final_vowel = (v.ku_infinitive_alternate if v.ku_infinitive_alternate
                                                 else (v.root_guess, v.final_vowel))
                    kit_root, root_tier, is_full_form = resolve_verb_root(root_guess, root_tables)
                    inf_word = kit_root if is_full_form else f"ku{kit_root}{final_vowel}"
                    d.secondary_nouns.append((vi, tokens[vi], inf_word))
                    corrected_note = (f" (corrected from the greedy TAM-consuming parse, root "
                                        f"'{v.root_guess}' -- see VerbFormEntry.ku_infinitive_alternate)"
                                        if v.ku_infinitive_alternate else "")
                    d.flags.append(f"'{tokens[vi]}': no noun won as nsubj, but subject marker 'ku-' is the "
                                    f"bare infinitive marker (shares its subject with the matrix verb, "
                                    f"nothing to resolve here){corrected_note} -> '{inf_word}' "
                                    f"(root tier: {root_tier})")
                    continue
                else:
                    # 2026-10-20: see find_fallback_common_noun_subject()'s own
                    # docstring (direct bug report, "Kila kikundi kitapewa pesa
                    # nyingi" -- "kikundi" has ZERO corpus occurrences, so it
                    # was never even a noun candidate) -- tried only after BOTH
                    # the probe and the proper-name fallback have failed.
                    common = find_fallback_common_noun_subject(d.tokens, vi, used_noun_idxs, assigned)
                    if common is not None:
                        common_idx, common_class = common
                        clause["subject_class"] = common_class
                        d.flags.append(f"'{tokens[vi]}': no noun won as nsubj, but '{tokens[common_idx]}' "
                                        f"(unclaimed, Swahili noun-class-PREFIX shape only -- no Kitaita "
                                        f"vocabulary at all, not even low-confidence) is almost certainly "
                                        f"the subject -- guessed Swahili class '{common_class}' from its "
                                        f"own prefix (find_fallback_common_noun_subject), then this "
                                        f"project's existing per-class default (SWAHILI_CLASS_DEFAULT_SM_"
                                        f"GUESS) for VERB AGREEMENT ONLY. The noun itself stays in the "
                                        f"output as the original Swahili word (no translation exists or "
                                        f"is attempted for it).")
                    elif v.subject_marker in ("a", "wa"):
                        # 2026-10-22: direct user audit request -- manually
                        # classifying every "no reliable subject" case in
                        # the African Storybook corpus found this project's
                        # OWN "a-"/"wa-" 3rd-person-pro-drop class default
                        # (added for the "if not noun_hits:" branch, bug #4)
                        # was NOT reachable here, in the much more common
                        # case where the sentence has SOME OTHER noun (its
                        # own object, or an unrelated noun elsewhere) that
                        # makes noun_hits non-empty -- e.g. "Anapunga
                        # mkono." ('mkono' is a real noun, so this branch
                        # runs instead of the empty-noun_hits one; "a-"
                        # still went completely unhandled). Roughly half of
                        # this audit's 32 "subject gap" cases were this
                        # exact shape -- the SAME already-confirmed rule
                        # was simply not wired into this second code path.
                        # Identical reasoning/caveats as bug #4's own
                        # comment: an UNCONFIRMED GUESS about WHICH noun,
                        # not about which class.
                        guessed_class = "u" if v.subject_marker == "a" else "wi"
                        clause["subject_class"] = "ANIMATE-SG" if v.subject_marker == "a" else "ANIMATE-PL"
                        d.flags.append(f"'{tokens[vi]}': no noun won as nsubj, and no proper-name/common-"
                                        f"noun fallback either, but subject marker '{v.subject_marker}-' "
                                        f"is Swahili's OWN class-1/class-2 marker -- defaulting to class "
                                        f"'{guessed_class}-' (mundu/wandu) for VERB AGREEMENT ONLY, same "
                                        f"reasoning as the 3rd-person pro-drop default elsewhere (bug #4) "
                                        f"-- an UNCONFIRMED GUESS about WHICH noun, not about which class.")
                    else:
                        d.flags.append(f"'{tokens[vi]}': no noun won as nsubj against every competitor -- no "
                                        f"reliable subject, so no agreement computed for this verb.")
            if obj_i is not None:
                oi, olemma, _, _ = next(h for h in noun_hits if h[0] == obj_i)
                clause["object"], clause["object_idx"] = olemma, oi
                used_noun_idxs.add(oi)
                d.flags.append(f"'{olemma}' -> probe-identified OBJECT of '{tokens[vi]}' (beat every "
                                f"other noun as obj).")

            # 2026-10-09, direct user directive: "Wire in the live self-loss
            # check." The trained nsubj/obj probe is confirmed UNRELIABLE for
            # '-ki-' (participial/conditional) verbs specifically -- only
            # 5/226 such verbs have a gold nsubj dependent across this
            # project's treebanks, too sparse to have ever taught the probe
            # anything real here (confirmed directly, see docs/status.md
            # 2026-10-09). For these verbs ONLY, override the probe's
            # subject guess with resolve_ki_subject_via_self_loss() --
            # NLLB's own teacher-forced self-translation likelihood,
            # validated 14/14 on hand-checked cases. This is a DIFFERENT
            # evidence tier (NLLB's generative self-consistency preference,
            # not independently verified grammar) and is flagged as such,
            # not presented as equivalent to the probe's own validated
            # win-condition.
            if v.tam_marker == "ki":
                override_idx, override_reason = None, None

                # 2026-10-09 (cont. 12), checked BEFORE the live self-loss
                # fallback -- see the module-level comment above SPEECH_ACT_
                # ROOTS/PERCEPTION_ROOTS for why these three rules exist and
                # their evidence tiers. preceding_clauses is every clause
                # already finalized, in textual order (verb_hits is itself
                # token-ordered, so d.clauses is too).
                preceding_clauses = [c for c in d.clauses if c["verb_idx"] < vi]

                # 2026-10-10, direct user directive: "Build the governor-
                # attachment probe" -- replaces the "nearest preceding verb-
                # shaped token" STRING heuristic below (previously the ONLY
                # mechanism feeding rules 2/3) with the trained governor_
                # attachment_probe, which finds the '-ki-' verb's real gold
                # head via structure instead of linear position. Validated
                # directly before this swap (docs/status.md 2026-10-10):
                # agrees with the real gold head more often than the old
                # heuristic (net +7 on 62 held-out cases), though not
                # overwhelmingly so -- reported honestly, not rounded up.
                # Deliberately NOT falling back to the old heuristic when
                # the probe's pick isn't usable (e.g. not an already-
                # resolved preceding verb clause) -- that would reintroduce
                # exactly the position-based failure modes this swap exists
                # to fix. In that case rules 2/3 simply don't fire, same
                # graceful degradation to the self-loss fallback as before.
                from nllb_probe_identification import find_ki_governor
                governor_idx = find_ki_governor(tokens, vi, [i for i in range(len(tokens)) if i != vi])
                nearest_pi, nearest_v = None, None
                if governor_idx is not None:
                    match = next(((pi, pv) for pi, pv in verb_hits if pi == governor_idx and pi < vi), None)
                    if match is not None:
                        nearest_pi, nearest_v = match
                nearest_clause = (next((c for c in preceding_clauses if c["verb_idx"] == nearest_pi), None)
                                   if nearest_pi is not None else None)

                # Rule 1: reported-speech '-ki-' verb (sema-type) -- same
                # subject as the nearest preceding clause with a resolved
                # subject (same-subject narrative chaining).
                if any(v.root_guess == r or v.root_guess.endswith(r) for r in SPEECH_ACT_ROOTS):
                    prev = next((c for c in reversed(preceding_clauses) if c["subject_idx"] is not None), None)
                    if prev is not None:
                        override_idx = prev["subject_idx"]
                        override_reason = (f"'-ki-' verb's own root ('{v.root_guess}') is a reported-speech "
                                            f"verb (sema-type) -- predicted SAME SUBJECT as the preceding "
                                            f"clause ('{prev['subject']}'), standard Swahili same-subject "
                                            f"narrative chaining; a free syntactic rule, no NLLB call needed "
                                            f"(not yet checked against a native speaker for this specific "
                                            f"sentence).")

                # Rule 2: periphrastic 'kuwa ... -ki-' continuous/progressive
                # -- the immediately preceding verb is a form of 'kuwa' (be);
                # the '-ki-' verb IS the real predicate and must share
                # 'kuwa's own subject by construction.
                if override_idx is None and nearest_v is not None:
                    if nearest_v.root_guess.endswith("kuw"):
                        prev = nearest_clause
                        if prev is not None and prev["subject_idx"] is not None:
                            override_idx = prev["subject_idx"]
                            override_reason = (f"'-ki-' verb is governed by an immediately preceding 'kuwa' "
                                                f"(be) auxiliary ('{tokens[nearest_pi]}') -- periphrastic "
                                                f"continuous/progressive construction, subject is shared by "
                                                f"construction, not inferred; a free syntactic rule, no NLLB "
                                                f"call needed (not yet checked against a native speaker for "
                                                f"this specific sentence).")

                # Rule 3: perception/cognition-verb object control -- the
                # nearest preceding verb is a closed-set perception/
                # cognition predicate (ona/kuta/sikia/kumbuka, "saw/found/
                # heard/remembered X doing Y") with a resolved OBJECT --
                # standard object-control for perception-verb small-clause
                # complements. This is the construction every one of this
                # project's hand-built '-ki-' validation sentences (mvulana/
                # kalamu; the 12-item battery in validate_roundtrip_and_
                # attention.py) actually tests -- matches the validated
                # self-loss/round-trip answer on all of them, and is the
                # first mechanism able to resolve mbwa/kijana at all, since
                # it is a SYNTACTIC rule (not a marker comparison) and so is
                # not blocked by the two nouns sharing one subject-marker
                # class.
                if override_idx is None and nearest_v is not None:
                    if any(nearest_v.root_guess == r or nearest_v.root_guess.endswith(r)
                           for r in PERCEPTION_ROOTS):
                        prev = nearest_clause
                        if prev is not None and prev["object_idx"] is not None:
                            override_idx = prev["object_idx"]
                            override_reason = (f"'-ki-' verb is governed by a perception/cognition verb "
                                                f"('{tokens[nearest_pi]}') with an identified object -- "
                                                f"predicted OBJECT CONTROL ('{prev['object']}' is understood "
                                                f"as doing the '-ki-' action), the standard 'saw/found X "
                                                f"doing Y' pattern; a free syntactic rule, no NLLB call "
                                                f"needed, and NOT subject to the shared-subject-marker "
                                                f"ceiling that blocks self-loss/probe-based methods "
                                                f"(validated against this project's existing round-trip-"
                                                f"corrected battery and the mvulana/kalamu case).")

                # Rule 4: temporal/concessive conjunction ('wakati'/'huku')
                # governing the '-ki-' clause. Checked directly BEFORE
                # building this (docs/status.md 2026-10-10): of 31 real
                # gold wakati/huku-governed '-ki-' verbs, 20/31 (64.5%)
                # have their OWN overt local subject right there in the
                # clause (e.g. "wakati kampeni zikiendelea" = "while the
                # CAMPAIGN was continuing" -- kampeni, not the matrix
                # subject). An unconditional same-subject default would
                # therefore be WRONG on the MAJORITY of real cases -- this
                # rule deliberately only fires when there is NO fresh noun
                # candidate between the conjunction and the '-ki-' verb
                # (the same "local window" signal that correlated exactly
                # with gold local-subject presence in the direct check),
                # i.e. only when the regular probe genuinely has nothing
                # local to find; otherwise its own (correctly local)
                # answer is left untouched, same conservative discipline
                # as the cross-clause-leakage fix (cont. 14).
                if override_idx is None and governor_idx is not None:
                    gov_word = tokens[governor_idx].lower()
                    if gov_word in TEMPORAL_CONCESSIVE_CONJUNCTIONS:
                        local_fresh_subject = [h for h in noun_hits if governor_idx < h[0] < vi]
                        if not local_fresh_subject:
                            prev = next((c for c in reversed(preceding_clauses)
                                         if c["subject_idx"] is not None), None)
                            if prev is not None:
                                override_idx = prev["subject_idx"]
                                override_reason = (f"'-ki-' verb is governed by a temporal/concessive "
                                                    f"conjunction ('{tokens[governor_idx]}') with NO local "
                                                    f"subject candidate in between -- predicted SAME SUBJECT "
                                                    f"as the preceding clause ('{prev['subject']}'); a free "
                                                    f"syntactic rule, no NLLB call needed. Checked directly: "
                                                    f"the majority (20/31) of real wakati/huku-governed cases "
                                                    f"have their OWN local subject instead -- this rule "
                                                    f"deliberately does not fire when one is available.")

                from nllb_probe_identification import resolve_ki_subject_via_self_loss

                def _sm_for(cls, lemma):
                    if cls:
                        head = cls.split("/")[0]
                        if head.isdigit() and int(head) in SM_TABLE:
                            return SM_TABLE[int(head)]
                        if cls in SM_TABLE:
                            return SM_TABLE[cls]
                    guess = guess_noun_class_by_prefix(lemma)
                    if guess:
                        head = guess[0].split("/")[0]
                        if head.isdigit() and int(head) in SM_TABLE:
                            return SM_TABLE[int(head)]
                    return None

                if override_idx is None:
                    # No free syntactic rule applied -- fall back to the
                    # live, now-batched self-loss check (2026-10-09 cont. 11).
                    candidates = [(h[0], _sm_for(h[2], h[1])) for h in noun_hits]
                    best_idx, scores = resolve_ki_subject_via_self_loss(
                        tokens, vi, v.root_guess, v.final_vowel, candidates)
                    if best_idx is not None:
                        override_idx = best_idx
                        override_reason = (f"NLLB self-consistency check (participial '-ki-' verb; the "
                                            f"trained nsubj/obj probe is unreliable here, see docs/status.md "
                                            f"2026-10-09 -- this reflects NLLB's own generative preference, "
                                            f"not verified grammar).")

                if override_idx is not None and override_idx != clause["subject_idx"]:
                    old_idx = clause["subject_idx"]
                    # Only free the OLD subject index if no EARLIER, already-
                    # finalized clause still depends on it -- a noun can be the
                    # shared, pro-dropped subject of multiple clauses (the whole
                    # point of tracking used_noun_idxs across clauses at all);
                    # discarding it unconditionally here caused exactly that case
                    # to break (caught directly: "Mvulana aliona kalamu
                    # ikivunjika." lost 'mvulana' as clause 0's own subject when
                    # clause 1's subject was overridden away from it).
                    if old_idx is not None and not any(
                            c["subject_idx"] == old_idx or c["object_idx"] == old_idx for c in d.clauses):
                        used_noun_idxs.discard(old_idx)
                    si, slemma, scls, sevidence = next(h for h in noun_hits if h[0] == override_idx)
                    clause["subject"], clause["subject_idx"], clause["subject_class"] = slemma, si, scls
                    used_noun_idxs.add(si)
                    d.flags.append(f"'{slemma}' -> OVERRIDING to SUBJECT of '{tokens[vi]}': {override_reason}")
                elif override_idx is not None:
                    d.flags.append(f"CONFIRMED '{clause['subject']}' as subject of '{tokens[vi]}': "
                                    f"{override_reason}")

            if clause["subject_idx"] is not None:
                subj_adj_i = next((ai for ai, _ in adjective_hits if ai not in adjective_claimed
                                    and probe.adjective_targets.get(ai) == clause["subject_idx"]), None)
                if subj_adj_i is not None:
                    stem = next(s for i, s in adjective_hits if i == subj_adj_i)
                    clause["adjective"] = {"stem": stem, "class": clause["subject_class"]}
                    clause["adjective_idx"] = subj_adj_i
                    adjective_claimed.add(subj_adj_i)
            d.clauses.append(clause)

        # 2026-10-26: direct crash found while crash-testing the new
        # storybook-sentence pool ("Hakujua kusoma maandishi." ->
        # IndexError) -- same shape as the 2026-10-21 "Tulipenda."
        # fix (an unconditional `[0]` index into a list this branch
        # assumed always gets at least one entry), but a DIFFERENT
        # cause: when EVERY verb in verb_hits is a "ku-" infinitive
        # complement (cont. 18's generalization of the #3 fix into
        # THIS branch), each one hits `continue` before ever reaching
        # `d.clauses.append(clause)` above -- if there is no OTHER,
        # finite verb in the sentence (here: "hakujua" isn't
        # recognized as a verb AT ALL, a vocabulary gap, leaving
        # "kusoma" as the only verb_hits entry), `d.clauses` stays
        # completely empty. The infinitive's own translation is NOT
        # lost (the `continue` branch already pushed it to
        # `d.secondary_nouns`) -- only this "first clause" subject/
        # object aggregation needs to tolerate having nothing to
        # aggregate; the caller (translate_sentence_with_probes)
        # already handles an empty `d.clauses` gracefully (same
        # "if not d.clauses:" guard the 2026-10-21 fix relies on).
        if d.clauses:
            first = d.clauses[0]
            d.subject, d.subject_idx, d.subject_class = first["subject"], first["subject_idx"], first["subject_class"]
            d.object, d.object_idx = first["object"], first["object_idx"]
            d.adjective, d.adjective_idx = first["adjective"], first["adjective_idx"]

        for oi, olemma, _, _ in noun_hits:
            if oi in used_noun_idxs:
                continue
            kit = lookup_noun_kitaita(olemma, noun_kitaita)
            d.secondary_nouns.append((oi, olemma, kit))
            d.flags.append((f"'{olemma}' -> '{kit}': not claimed as any clause's subject/object; "
                             f"substituted independently." if kit else
                             f"'{olemma}': not claimed as any clause's subject/object, and no Kitaita "
                             f"translation available -- kept as the original Swahili word."))

        for ai, stem in adjective_hits:
            if ai in adjective_claimed:
                continue
            # 2026-10-22: direct user request ("extend the adjective-concord
            # mechanism to non-subject targets") -- investigating this found
            # the amod probe/compute_adjective_concord() ALREADY consider
            # object/secondary-noun targets (target_i below can and does
            # resolve to either), so this was NOT actually a subject-only
            # restriction; every failure audited in cont. 18 turned out to
            # be a genuine vocabulary gap (target noun untranslatable, or
            # its class unknown anywhere in this project's data, or the
            # ADJECTIVE STEM itself has no concord table for any class --
            # "mrefu"/"umbali" is the first kind; "wengine"/"taslimu"/
            # "mnono"/"pekee"/"kwanza" are the third). What THIS code path
            # genuinely lacked, that the SUBJECT-linked path (inside the
            # per-verb loop above) and the "elif not verb_hits:" branch's
            # own adjective loop both already have: the proper-name
            # fallback (resolve_noun_kitaita_and_class, e.g. "Sifuna" has
            # no noun_class_of entry but should still concord as class
            # 'u') and the SWAHILI_CLASS_DEFAULT_SM_GUESS majority-default
            # fallback (when the TARGET's own Swahili class is known from
            # noun_hits but its KITAITA translation's class never got
            # independently confirmed) -- a real, if narrower, asymmetry.
            # Closed here for consistency even though no sentence in the
            # audited corpus happens to be rescued by it (their Swahili
            # classes are ALSO unknown, not just their Kitaita ones).
            target_i = probe.adjective_targets.get(ai)
            target_kitaita, target_swahili, override_class, guess_source = None, None, None, None
            if target_i is not None:
                hit = next((h for h in noun_hits if h[0] == target_i), None)
                swahili_class_hint = hit[2] if hit is not None else None
                target_swahili = (hit[1] if hit is not None else
                                   next((l for i, l, k in d.secondary_nouns if i == target_i), None))
                if target_swahili is not None:
                    target_kitaita, resolved_class = resolve_noun_kitaita_and_class(
                        target_swahili, target_i, tokens, noun_kitaita, noun_class_of)
                    if resolved_class is not None:
                        override_class = resolved_class
                    elif swahili_class_hint and (guess := SWAHILI_CLASS_DEFAULT_SM_GUESS.get(swahili_class_hint)):
                        override_class, guess_source = guess, "SWAHILI_CLASS_DEFAULT_SM_GUESS"
                    elif target_kitaita:
                        # 2026-10-23: direct user suggestion -- last-resort
                        # word-shape guess (predict_kitaita_class_from_
                        # prefix) when NEITHER the Kitaita word's own class
                        # NOR its Swahili source's class is known at all
                        # (exactly the "umbali"/"mrefu" situation this was
                        # built for).
                        override_class, prefix_note = predict_kitaita_class_from_prefix(
                            target_kitaita, noun_class_of)
                        if override_class is not None:
                            guess_source = prefix_note
            if target_kitaita:
                adj_part, note = compute_adjective_concord(target_kitaita, target_swahili, stem, noun_class_of,
                                                             noun_kitaita, override_class=override_class)
                if adj_part:
                    d.secondary_nouns.append((ai, tokens[ai], adj_part))
                    guess_note = f" -- {guess_source}" if guess_source else ""
                    d.flags.append(f"'{tokens[ai]}' -> '{adj_part}': probe-identified as modifying "
                                    f"'{target_kitaita}' ({note}){guess_note}.")
                    record_slot(d, ai, "adjective", tokens[ai], target_i, needs_rewrite=True,
                                predicted_form=adj_part, confidence=guess_source or note)
                    continue
                d.passthrough.append((ai, tokens[ai]))
                d.flags.append(f"'{tokens[ai]}': probe-identified as modifying '{target_kitaita}', but "
                                f"{note} -- kept as the original Swahili word.")
                record_slot(d, ai, "adjective", tokens[ai], target_i, needs_rewrite=True,
                            predicted_form=None, confidence=note)
                continue
            d.passthrough.append((ai, tokens[ai]))
            d.flags.append(f"'{tokens[ai]}': " + (
                f"probe-identified as modifying '{target_swahili}', but it has NO Kitaita translation "
                f"and no proper-name/class-default fallback applies"
                if target_swahili is not None else
                "no target noun identified at all (or none exists)"
            ) + " -- kept as the original Swahili word.")
            record_slot(d, ai, "adjective", tokens[ai], target_i, needs_rewrite=True,
                        predicted_form=None,
                        confidence="no-kitaita-translation" if target_swahili is not None else "no-target")

        # 2026-10-11 (cont. 8): possessive-pronoun targeting, same target-
        # resolution shape as the adjective loop just above (subject / object /
        # secondary-noun, in that order) -- reused rather than re-derived.
        for pi, prefix, stem in possessive_hits:
            target_i = probe.possessive_targets.get(pi)
            target_kitaita, target_swahili = None, None
            target_clause = (next((c for c in d.clauses
                                    if target_i in (c["subject_idx"], c["object_idx"])), None)
                              if target_i is not None else None)
            if target_clause is not None:
                target_swahili = (target_clause["subject"] if target_clause["subject_idx"] == target_i
                                   else target_clause["object"])
                target_kitaita, kit_sm = resolve_noun_kitaita_and_class(
                    target_swahili, target_i, tokens, noun_kitaita, noun_class_of,
                    plural_context=_siku_plural_context(d.tokens, target_i) if target_swahili == "siku" else False)
            elif target_i is not None:
                sec = next((k for i, l, k in d.secondary_nouns if i == target_i), None)
                target_kitaita = sec
                kit_sm = resolve_class_marker(target_kitaita, noun_class_of) if target_kitaita else None
            else:
                kit_sm = None
            # 2026-10-22: same fallback this entry adds to the adjective
            # sweep just above -- when the target's KITAITA class never
            # resolved but its SWAHILI class is known (from noun_hits),
            # fall back to SWAHILI_CLASS_DEFAULT_SM_GUESS rather than
            # giving up (the same tier the SUBJECT-linked path already
            # gets for free via translate()'s own resolution).
            if kit_sm is None and target_i is not None:
                hit = next((h for h in noun_hits if h[0] == target_i), None)
                if hit is not None and hit[2]:
                    kit_sm = SWAHILI_CLASS_DEFAULT_SM_GUESS.get(hit[2])
            # 2026-10-23: direct user suggestion -- last-resort word-shape
            # guess (predict_kitaita_class_from_prefix) when neither the
            # Kitaita word's own class nor its Swahili source's class is
            # known at all; see the adjective sweep above for the full
            # evidence this is based on.
            if kit_sm is None and target_kitaita:
                kit_sm, _ = predict_kitaita_class_from_prefix(target_kitaita, noun_class_of)
            # 2026-10-20: generalized from class-1-only to POSS_PERSON_FORM_BY_
            # CLASS (now 2 confirmed classes: 'u' and 'ji' -- see that table's
            # own comment in kitaita_verified_corrections.py).
            if kit_sm in POSS_PERSON_FORM_BY_CLASS and stem in POSS_PERSON_FORM_BY_CLASS[kit_sm]:
                poss_part = POSS_PERSON_FORM_BY_CLASS[kit_sm][stem]
                d.secondary_nouns.append((pi, tokens[pi], poss_part))
                d.flags.append(f"'{tokens[pi]}' -> '{poss_part}': probe-identified as possessing "
                                f"'{target_kitaita or '(unresolved target)'}' (class '{kit_sm}-', "
                                f"statistical candidate, NOT native-speaker-verified -- see kitaita_"
                                f"verified_corrections.py POSS_PERSON_FORM_BY_CLASS).")
                record_slot(d, pi, "possessive", tokens[pi], target_i, needs_rewrite=True,
                            predicted_form=poss_part, confidence="statistical")
            else:
                d.passthrough.append((pi, tokens[pi]))
                d.flags.append(f"'{tokens[pi]}': possessive pronoun"
                                + (f", target '{target_kitaita}' resolved to Kitaita class '{kit_sm}'"
                                   if kit_sm else ", target's Kitaita class is unresolved")
                                + " -- no confirmed concord data for this class (see "
                                  "POSS_PERSON_FORM_BY_CLASS) -- kept as the original Swahili word, "
                                  "flagged rather than guessed.")
                record_slot(d, pi, "possessive", tokens[pi], target_i, needs_rewrite=True,
                            predicted_form=None,
                            confidence="known-class-no-data" if kit_sm else "class-unresolved")

    # 2026-10-19: descriptive/purposive associative concord already ran at
    # the TOP of this function (before noun_hits), not here.
    # 2026-10-20: "-ote" (all/whole) concord -- must run BEFORE associative
    # concord (which skips back over an already-claimed -ote token).
    apply_ote_concord(d, noun_kitaita, noun_class_of, assigned)
    # 2026-10-15: associative/possessive concord ("mayai ya kuku" ->
    # "maghi gha nguku") -- see apply_associative_concord()'s own docstring.
    apply_associative_concord(d, noun_kitaita, noun_class_of, assigned, tokens=tokens)
    # 2026-10-15: demonstrative concord ("hii kazi" -> "iri kazi") -- see
    # apply_demonstrative_concord()'s own docstring.
    apply_demonstrative_concord(d, noun_kitaita, noun_class_of, assigned)

    for i, is_assigned in enumerate(assigned):
        if not is_assigned:
            d.passthrough.append((i, tokens[i]))
    leftover = [tokens[i] for i, a in enumerate(assigned) if not a]
    if leftover:
        d.flags.append(f"No Kitaita translation/mechanism for: {', '.join(leftover)} -- kept as the "
                        f"original Swahili word(s) in place.")
    return d


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation (. ! ?), punctuation kept
    attached to the sentence it ends. Needed because decompose_sentence_
    with_probes() has NO sentence-boundary concept at all -- tokenize()
    (extract_swahili_grammar.py) strips '.'/'!'/'?' outright, so every
    noun/verb in whatever text it's handed gets pooled into one flat
    candidate set. Found directly 2026-10-09 (user report: "we do not
    identify the correct subject for lia either"): "Mbwa anakunywa maji.
    mtoto analia." fed as one blob wrongly scored 'mbwa' (sentence 1's
    subject) as ALSO winning subject of 'analia' (sentence 2's verb),
    while 'mtoto' -- the correct, and only real, candidate inside its own
    sentence -- was never even offered to that verb and got dropped
    entirely untranslated. Each sentence decomposed alone was already
    correct (confirmed directly before writing this fix). Matches the
    project's own open-decisions note: probe-based relation identification
    is only ever valid within one sentence; cross-sentence linking is a
    separate, out-of-scope coreference problem."""
    parts = re.split(r"([.!?]+)", text)
    sentences = []
    buf = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:
            buf = part
        else:
            buf += part
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def translate_sentence_with_probes(text: str, tables) -> dict:
    """Probe-based counterpart to translate_sentence() -- same substitution
    (translate(), called ONCE PER CLAUSE now) and position-preserving
    reassembly, different identification source. Requires NLLB + the
    saved probes (train_and_save_identification_probes.py must have been
    run already); raises FileNotFoundError with a clear message otherwise
    (nllb_probe_identification.load_probes()).

    2026-10-09, direct user directive: "remove the one verb restriction...
    we can handle multiple verbs" -- loops over EVERY clause decompose_
    sentence_with_probes() found (one per verb), calling translate()
    independently for each (its own subject's class drives its own SM/
    TAM, exactly as kitaita_sentence_generator.translate() already does --
    nothing about that function needed to change, it's just called more
    than once now).

    2026-10-09 UPDATE: multi-sentence input is now split via split_sentences()
    FIRST, each sentence decomposed/translated/assembled completely
    independently, then the Kitaita outputs joined in order. See
    split_sentences() docstring for why this was necessary -- it is a
    SEPARATE bug from the multi-verb-per-sentence work above (multiple
    verbs sharing one real sentence is fine and was already handled
    correctly; multiple SENTENCES sharing one candidate pool was not)."""
    sentences = split_sentences(text)
    if len(sentences) > 1:
        sub_reports = [translate_sentence_with_probes(s, tables) for s in sentences]
        merged_flags, merged_trace, out_parts = [], [], []
        for sr in sub_reports:
            merged_flags.extend(sr.get("flags") or [])
            merged_trace.extend(sr.get("trace") or [])
            out_parts.append(sr.get("kitaita_sentence") or f"[untranslated: {sr.get('error') or 'unknown error'}]")
        errors = [sr["error"] for sr in sub_reports if sr.get("error")]
        return {
            "input": text,
            "decomposition": [sr["decomposition"] for sr in sub_reports],
            "kitaita_sentence": " ".join(out_parts) or None,
            "trace": merged_trace,
            "flags": merged_flags,
            "error": "; ".join(errors) if errors else None,
            "per_sentence": sub_reports,
        }
    noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of = tables[0], tables[1], tables[2], tables[3]
    d = decompose_sentence_with_probes(text, noun_kitaita, noun_class_of, root_tables=tables)
    report = {"input": text, "decomposition": d, "kitaita_sentence": None, "trace": [], "error": None, "flags": d.flags}
    if not d.clauses:
        report["error"] = "; ".join(d.flags) or "decomposition failed"
        report["kitaita_sentence"] = assemble_multi_clause(d, [], noun_kitaita)
        return report

    clause_results = []
    all_trace = []
    for clause in d.clauses:
        # 2026-10-15: subject_idx is None both for a genuinely unresolved
        # clause (skip, as before) AND for a recognized pro-dropped PERSON
        # subject (clause["person"] set by decompose_sentence_with_probes) --
        # only the first case should actually skip translate(). 2026-10-20:
        # clause["subject_class"] (set by find_fallback_proper_name_subject(),
        # an unrecognized proper-name subject) is a THIRD way past this guard,
        # same shape as the person case -- translate() already knows what to
        # do with subject_class alone (swahili_subject_noun=None), see its
        # own ANIMATE_SG_EXCLUDED_FROM_DEFAULT check.
        # 2026-10-22: bug #2 fix -- the invariant locative-existential
        # "oko" (clause["verb"]["root"] == "oko_exist") needs no subject
        # info of its own (see translate()'s own bypass for this same
        # root), so it must not be caught by this "genuinely unresolved,
        # skip translate() entirely" guard just because the relation
        # probe found no subject for it. Same reasoning for the plural-
        # have "ko_have" root (PLURAL_HAVE_FORMS) -- it defaults to
        # ANIMATE-PL ('wi') when unresolved, same as "oko_exist" defaults
        # to literal, rather than needing a real subject noun.
        if (clause["subject_idx"] is None and clause.get("person") is None
                and clause.get("subject_class") is None
                and clause["verb"]["root"] not in ("oko_exist", "ko_have")):
            # 2026-10-22: direct user report -- some verbs ("waliishi"/
            # "ilitokea") were silently left untranslated with NO flag
            # about the verb itself at all (only the separate, already-
            # existing "no reliable subject" note, which says nothing
            # about whether the VERB's own root translation was even
            # attempted -- it wasn't, translate() is never called here).
            # This doesn't fix the underlying gap (the relation probe
            # genuinely found no subject for these, a parsing-accuracy
            # limitation, not something guessed around) but makes it
            # visible instead of indistinguishable from "translated
            # successfully but happened to equal the Swahili surface
            # form".
            d.flags.append(f"Verb '{clause['verb_surface']}': root translation SKIPPED ENTIRELY "
                            f"(no subject/person/class resolved for it, so there was nothing to drive "
                            f"SM/TAM substitution) -- kept as the ORIGINAL SWAHILI WORD, unlike a "
                            f"'Taita-fied hybrid' (that tier at least translates the root). NOT the "
                            f"same situation as 'root has no Kitaita translation' below.")
            clause_results.append((clause, None))
            continue
        root = clause["verb"]["root"]
        # 2026-10-15: relative-marker PRESENCE DETECTION, direct user report
        # ("ninayofanya" -> "mafozi" instead of "naweribonya"). decompose_
        # verb() only ENUMERATES (relative_marker, remainder) candidates
        # (extract_swahili_grammar.find_relative_marker_candidates's own
        # docstring: same enumerate-don't-strip discipline as object
        # markers) -- nothing downstream ever consulted them before this,
        # so the UNSTRIPPED root ("yofany", relative marker 'yo' + real
        # root 'fany' glued together) was always what got looked up, and
        # it happened to have ITS OWN spurious high-Dice fallback match
        # ("mafozi") -- not a guess that failed, a question that was never
        # asked. Unlike the object-marker check below (which only compares
        # root_kitaita, the ROOT-LEVEL Dice table), this must ALSO compare
        # the FALLBACK (whole-surface-form) table: "fany" and "yofany" both
        # have NO root-level entry at all, so a root_kitaita-only
        # comparison would see a tie (-1.0 vs -1.0) and never switch --
        # confirmed directly before writing this ("fany" fallback Dice
        # 0.45 vs "yofany" fallback Dice 0.33, so 'fany' genuinely has
        # better evidence once fallback is included, not just structurally
        # "more correct").
        rel_candidates = clause["verb"].get("rel_candidates") or []
        if rel_candidates:
            current_score = _root_evidence_score_verified_first(root, root_kitaita, root_kitaita_fallback)
            best_rel, best_root, best_fv, best_score = None, None, None, current_score
            for rel, remainder in rel_candidates:
                cand_root, cand_fv = remainder[:-1], remainder[-1]
                cand_score = _root_evidence_score_verified_first(cand_root, root_kitaita, root_kitaita_fallback)
                if cand_score > best_score:
                    best_rel, best_root, best_fv, best_score = rel, cand_root, cand_fv, cand_score
            if best_rel is not None:
                d.flags.append(f"'{clause['verb_surface']}': detected likely relative marker '-{best_rel}-' "
                                f"(unstripped root '{root}' evidence={current_score:.2f} -> corrected root "
                                f"'{best_root}' evidence={best_score:.2f}) -- using the corrected root. This "
                                f"is a presence-detection HEURISTIC (root-plausibility comparison, root-level "
                                f"Dice or whole-form fallback Dice, whichever exists), not verified Swahili "
                                f"grammar for this specific sentence. Kitaita relative-marker REWRITE itself "
                                f"remains unsolved (README/status.md) -- this only fixes the ROOT, the "
                                f"relative marker '{best_rel}' itself is simply dropped, same as this "
                                f"pipeline already drops object markers rather than rendering them.")
                clause["verb"]["root"], clause["verb"]["fv"] = best_root, best_fv
                if clause["verb"].get("om_candidates"):
                    # 2026-10-15: om_candidates were enumerated against the
                    # UNSTRIPPED stem (e.g. "yofanya") -- stale and
                    # potentially misleading once the root just changed to
                    # "fany"/"fanya"; re-checking for an object marker INSIDE
                    # the now-corrected root is explicitly out of scope here
                    # (find_relative_marker_candidates's own docstring already
                    # flags this as a separate, deliberately-deferred
                    # question) -- dropped rather than acted on with stale data.
                    clause["verb"]["om_candidates"] = []
                root = best_root
        # 2026-10-11: object-marker PRESENCE DETECTION. decompose_verb()
        # only ENUMERATES candidate (object_marker, alternate_root) pairs
        # (extract_swahili_grammar.OM_TABLE's docstring explains why it
        # doesn't just strip blindly -- same surface-collision risk as the
        # '-ki-' verb-detection bug fixed 2026-10-10). This is the one
        # place with the EVIDENCE to decide: only look at candidates when
        # the UNSTRIPPED root is already weak (below the existing risk
        # threshold -- this exact scenario is what that threshold's name
        # anticipated), and only switch when a candidate's root scores
        # clearly better against the Kitaita root dictionary. Never
        # switches away from an already-good match.
        om_candidates = clause["verb"].get("om_candidates") or []
        current_cand = None if (root == "kuw" or root in VERIFIED_ROOTS) else root_kitaita.get(root)
        current_score = current_cand["score"] if current_cand else -1.0
        if om_candidates and current_score < OBJECT_MARKER_RISK_ROOT_DICE_THRESHOLD:
            best_om, best_root, best_fv, best_suppress_tam, best_score = None, None, None, False, current_score
            for om, cand_root, cand_fv, suppress_tam in om_candidates:
                # 2026-10-20: an OM-stripped candidate that's a hand-VERIFIED
                # root (VERIFIED_ROOTS) is checked FIRST and treated as an
                # automatic, maximal-confidence win -- found directly while
                # testing 'uipate' (OM 'i' + root 'pat', freshly added to
                # VERIFIED_ROOTS this session): the Dice-only check below
                # never looks at VERIFIED_ROOTS at all, so a verified root
                # with (correctly) NO statistical root_kitaita entry of its
                # own would score -1.0 here and never win, even though it is
                # STRICTLY better evidence than any Dice score. 2026-10-21:
                # extended to VERIFIED_IDIOMS too -- found while wiring up
                # "alimkasirisha" (OM 'm' + root 'kasirish', a confirmed
                # PERIPHRASIS, not a single-word VERIFIED_ROOTS entry); same
                # reasoning applies identically.
                if cand_root in VERIFIED_ROOTS or cand_root in VERIFIED_IDIOMS:
                    cand_score = 1.0
                else:
                    c = root_kitaita.get(cand_root)
                    cand_score = c["score"] if c else -1.0
                if cand_score > best_score:
                    best_om, best_root, best_fv, best_suppress_tam, best_score = (
                        om, cand_root, cand_fv, suppress_tam, cand_score)
            if best_om is not None and best_score >= OBJECT_MARKER_RISK_ROOT_DICE_THRESHOLD:
                d.flags.append(f"'{clause['verb_surface']}': detected likely object marker '-{best_om}-' "
                                f"(unstripped root '{root}' Dice={current_score:.2f} -> corrected root "
                                f"'{best_root}' Dice={best_score:.2f}) -- using the corrected root. This is "
                                f"a presence-detection HEURISTIC (root-plausibility comparison against the "
                                f"Kitaita dictionary), not verified Swahili grammar for this specific "
                                f"sentence.")
                clause["verb"]["root"], clause["verb"]["fv"] = best_root, best_fv
                clause["verb"]["object_marker"] = best_om
                if best_suppress_tam:
                    # the matched "tam" string was actually this candidate's
                    # object marker (infinitives take no tense marker at all,
                    # e.g. "kulifuatilia" -- "li" is class-5 OM, not past-TAM);
                    # clearing it here prevents it being double-counted if
                    # anything downstream reassembles sm+tam+root+fv.
                    clause["verb"]["tam"] = ""
                root = best_root
        # 2026-10-21: NEGATIVE SUBJUNCTIVE/IMPERATIVE presence detection
        # ("usiogope" -> root "siogop", wrong -- direct user bug report).
        # Same enumerate-then-compare-evidence pattern as the object-marker
        # check just above: `negative_subjunctive_candidate` (extract_
        # swahili_grammar.decompose_verb) is the "si"-stripped alternate
        # reading, NOT a decision -- "usikilize" (root "sikiliz", genuinely
        # NOT negated) has the identical string shape, so this only
        # switches when the stripped candidate has STRICTLY BETTER root
        # evidence than the unstripped one, never on string shape alone.
        neg_candidate = clause["verb"].get("negative_subjunctive_candidate")
        if neg_candidate:
            unstripped_score = _root_evidence_score_verified_first(root, root_kitaita, root_kitaita_fallback)
            candidate_score = _root_evidence_score_verified_first(neg_candidate, root_kitaita, root_kitaita_fallback)
            if candidate_score > unstripped_score:
                d.flags.append(f"'{clause['verb_surface']}': detected likely NEGATIVE subjunctive/"
                                f"imperative '-si-' (unstripped root '{root}' evidence={unstripped_score:.2f} "
                                f"-> corrected root '{neg_candidate}' evidence={candidate_score:.2f}) -- "
                                f"using the corrected root AND marking this clause NEGATED "
                                f"(subjunctive_imperative tense). Swahili-side recognition is reference "
                                f"grammar (Ashton 1944); the Kitaita rendering is native-speaker-"
                                f"submitted, Kitaita Word Bank 2026-10-21 (see NEGATIVE_PERSON_PREFIX_"
                                f"BY_TENSE / NEGATIVE_CLASS_* in kitaita_verified_corrections.py) -- "
                                f"this used to deliberately stay affirmative-shaped because no Kitaita "
                                f"subjunctive-negative data existed; that gap is now closed.")
                clause["verb"]["root"] = neg_candidate
                clause["negated"] = True
                clause["negated_tense"] = "subjunctive_imperative"
                root = neg_candidate
        if root != "kuw" and root not in VERIFIED_ROOTS:
            cand = root_kitaita.get(root)
            if cand and cand["score"] < OBJECT_MARKER_RISK_ROOT_DICE_THRESHOLD:
                d.flags.append(f"Verb root '{root}' (for '{clause['verb_surface']}') only matched a WEAK "
                                f"statistical candidate (Dice {cand['score']:.2f}) -- treat as UNRELIABLE.")
        result = translate(clause["subject"], clause["verb"], clause["object"], clause["adjective"], tables,
                            person=clause.get("person"), subject_class=clause["subject_class"],
                            negated=clause.get("negated", False),
                            negated_tense=clause.get("negated_tense", "present"))
        if "error" in result:
            d.flags.append(f"Verb '{clause['verb_surface']}': {result['error']}")
            clause_results.append((clause, None))
            continue
        if result["components"].get("root_untranslated"):
            # 2026-10-19: same gap-flagging as translate_sentence() above, per clause.
            d.flags.append(f"Verb '{clause['verb_surface']}' (root '{clause['verb']['root']}') has NO "
                            f"Kitaita translation at all -- kept as a 'Taita-fied' Swahili-root/"
                            f"Kitaita-agreement hybrid, NOT a real translation.")
        if result["components"].get("adjective_untranslated"):
            d.flags.append(f"Adjective '{result['components']['adjective_stem']}' (for "
                            f"'{clause['verb_surface']}') has NO Kitaita concord data -- kept as a "
                            f"Swahili stem with a GUESSED concord prefix (double guess).")
        all_trace.extend(result["trace"])
        clause_results.append((clause, result))

        # 2026-10-23: subject-marker agreement is itself a concord slot --
        # record it the same way the standalone concord mechanisms above do.
        # root_is_full_form roots (oko_exist/na_have/kasirik's passive/etc.)
        # genuinely carry no separate SM morpheme -- not a failure, so not
        # recorded as needing one.
        if not result["components"].get("root_is_full_form"):
            record_slot(d, clause["verb_idx"], "subject_sm", clause["verb_surface"],
                        clause["subject_idx"], needs_rewrite=True,
                        predicted_form=result["components"].get("subject_marker"),
                        confidence="root_untranslated" if result["components"].get("root_untranslated")
                        else "resolved")
        if clause.get("adjective") is not None:
            record_slot(d, clause["adjective_idx"], "adjective", d.tokens[clause["adjective_idx"]],
                        clause["subject_idx"], needs_rewrite=True,
                        predicted_form=result["components"].get("adjective_kitaita"),
                        confidence="guessed-double" if result["components"].get("adjective_untranslated")
                        else "resolved")

    report["kitaita_sentence"] = assemble_multi_clause(d, clause_results, noun_kitaita)
    report["trace"] = all_trace
    first_result = next((r for _, r in clause_results if r is not None), None)
    if first_result is not None:
        report["components"] = first_result["components"]
    if report["kitaita_sentence"] is None:
        report["error"] = "; ".join(d.flags) or "decomposition failed"
    return report


def assemble_multi_clause(d: Decomposition, clause_results: list, noun_kitaita: dict) -> str | None:
    """Multi-verb generalization of assemble_in_order(): merges every
    clause's substituted pieces (subject/verb/object/adjective) with
    secondary_nouns/passthrough, all placed at their ORIGINAL token
    position, never a fixed template -- same "never rearrange" principle,
    just across however many clauses were found. A noun shared (pro-
    dropped) across clauses is placed only ONCE, at its single real token
    position -- later clauses trying to re-place it are a no-op, not a
    duplicate."""
    pieces = []
    placed = set()

    def add(idx, word):
        if idx is None or word is None or idx in placed:
            return
        pieces.append((idx, word))
        placed.add(idx)

    for clause, result in clause_results:
        comps = (result or {}).get("components", {})
        if comps.get("verified_exact_sentence"):
            verified = result["kitaita_sentence"].rstrip(".")
            first_word, _, rest = verified.partition(" ")
            add(clause["subject_idx"], first_word)
            if rest:
                add(clause["verb_idx"], rest)
            continue
        if clause["subject_idx"] is not None:
            kit = comps.get("subject_noun") if result else lookup_noun_kitaita(clause["subject"], noun_kitaita)
            add(clause["subject_idx"], kit or clause["subject"])
        if clause["verb_idx"] is not None:
            kit = comps.get("kitaita_verb_word") if result else None
            add(clause["verb_idx"], kit or clause["verb_surface"] or d.tokens[clause["verb_idx"]])
        if clause["object_idx"] is not None:
            kit = comps.get("object_noun") if result else lookup_noun_kitaita(clause["object"], noun_kitaita)
            add(clause["object_idx"], kit or clause["object"])
            if comps.get("idiom_object_trailing"):
                # 2026-10-21: see assemble_in_order()'s matching comment -- a
                # VERIFIED_IDIOMS fixed object with no original token of its own,
                # placed right after the real object noun via a fractional index.
                add(clause["object_idx"] + 0.5, comps["idiom_object_trailing"])
        elif comps.get("idiom_object_trailing") and clause["verb_idx"] is not None:
            add(clause["verb_idx"] + 0.5, comps["idiom_object_trailing"])
        if clause["adjective_idx"] is not None:
            kit = comps.get("adjective_kitaita") if result else None
            add(clause["adjective_idx"], kit or d.tokens[clause["adjective_idx"]])

    for idx, lemma, kit in d.secondary_nouns:
        add(idx, kit or lemma)
    for idx, tok in d.passthrough:
        add(idx, tok)

    if not pieces:
        return None
    pieces.sort(key=lambda p: p[0])
    words = [w.capitalize() if i == 0 else w for i, (_, w) in enumerate(pieces)]
    return " ".join(words) + "."


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def translate_text(text: str, tables) -> list[dict]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return [translate_sentence(s, tables) for s in sentences]


# ---------------------------------------------------------------------------
# Optional English -> Swahili leg via real NLLB (facebook/nllb-200-distilled-
# 1.3B, CPU-only -- no GPU available on this machine, confirmed directly via
# nvidia-smi). Lazy-loaded: importing this module never touches torch/
# transformers unless english_to_swahili() is actually called.
# ---------------------------------------------------------------------------
_nllb_tok = None
_nllb_model = None


def _load_nllb(model_name: str = "facebook/nllb-200-distilled-1.3B"):
    global _nllb_tok, _nllb_model
    if _nllb_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _nllb_tok = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _nllb_tok, _nllb_model


def english_to_swahili(text: str, model_name: str = "facebook/nllb-200-distilled-1.3B") -> str:
    tok, model = _load_nllb(model_name)
    # 2026-10-21: set explicitly every call, not just relied on at load time
    # (_load_nllb's src_lang="eng_Latn" was previously only ever set ONCE,
    # when the tokenizer is first constructed) -- now that swahili_to_
    # swahili() below reuses the SAME cached tokenizer and mutates its
    # src_lang to "swh_Latn", a later english_to_swahili() call in the same
    # process would otherwise silently keep encoding as Swahili input.
    tok.src_lang = "eng_Latn"
    inputs = tok(text, return_tensors="pt")
    bos = tok.convert_tokens_to_ids("swh_Latn")
    out = model.generate(**inputs, forced_bos_token_id=bos, max_length=64)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def swahili_to_swahili(text: str, model_name: str = "facebook/nllb-200-distilled-1.3B") -> str:
    """Monolingual SW->SW NLLB regeneration (src_lang=tgt_lang="swh_Latn") --
    the same capability directly verified on facebook/nllb-200-3.3B in
    test_nllb_swahili.py ("SW -> SW (monolingual regeneration)"), NOT
    separately re-verified here on the distilled-1.3B default (CLAUDE.md:
    "spot-check distilled models the same way before relying on them" --
    see this session's own spot-check in docs/status.md before trusting
    this for anything beyond the REPHRASING fallback below, which only
    ever uses the result as a candidate to re-translate and compare, never
    as a claim about Kitaita or Swahili on its own)."""
    tok, model = _load_nllb(model_name)
    tok.src_lang = "swh_Latn"
    inputs = tok(text, return_tensors="pt")
    bos = tok.convert_tokens_to_ids("swh_Latn")
    out = model.generate(**inputs, forced_bos_token_id=bos, max_length=64, num_beams=5)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def _unknown_word_flag_count(flags: list[str]) -> int:
    """Counts flags that mark a genuine vocabulary gap -- the 3 canonical
    phrasings translate_sentence()/translate_sentence_with_probes() both
    already use consistently for 'no mechanism exists for this word at
    all' (noun/particle passthrough, verb root_untranslated, adjective_
    untranslated -- see their own d.flags.append() call sites). Does NOT
    count presence-detection/heuristic notices (object-marker, relative-
    marker correction, WEAK-Dice warnings) -- those are about CONFIDENCE
    in an existing answer, not the absence of one. Case-INsensitive on
    purpose: the noun/particle passthrough flag ("No Kitaita translation/
    mechanism for: ...") capitalizes only the leading "No", while the
    verb-root/adjective flags capitalize "NO" mid-sentence -- checked
    directly, a case-sensitive "NO Kitaita" check silently missed the
    noun passthrough flag entirely."""
    return sum(1 for f in flags if "no kitaita" in f.lower())


def translate_sentence_with_rephrase_fallback(text: str, tables, use_probes: bool = True,
                                               model_name: str = "facebook/nllb-200-distilled-1.3B") -> dict:
    """2026-10-21, direct user request: "implement the rephrasing step with
    NLLB for the case where a word does not exist." Runs the ordinary
    pipeline (translate_sentence_with_probes() by default, translate_
    sentence() if use_probes=False) first; if the result has at least one
    genuine vocabulary gap (_unknown_word_flag_count above), tries ONE
    NLLB SW->SW monolingual-regeneration rephrase of the WHOLE sentence
    (swahili_to_swahili() above) and re-runs the SAME pipeline on that
    rephrased text. ONLY switches to the rephrased translation if it has
    STRICTLY FEWER unknown-word flags than the original -- never picked
    just because it differs, and the original is always preserved on the
    returned dict (original_kitaita_sentence/original_sentence) so nothing
    is silently discarded either way. This is a best-effort READABILITY
    fallback, same spirit as the existing "Taita-fy" root_untranslated
    tier -- an NLLB paraphrase swapping a rare word for a common synonym
    is a plausible mechanism, not a guaranteed one (it may rephrase the
    WHOLE sentence, including parts that were already fine, so the
    comparison is on the FULL flag count, not just the originally-unknown
    word), and this never claims the paraphrase preserves meaning exactly
    -- only that it was tried, and the swahili_to_swahili() CALL is a
    directly-verified capability (see that function's own docstring)."""
    fn = translate_sentence_with_probes if use_probes else translate_sentence
    original = fn(text, tables)
    original_flags = list(original.get("flags") or [])
    original_unknown = _unknown_word_flag_count(original_flags)

    result = dict(original)
    result["flags"] = original_flags
    result["rephrased"] = False
    result["rephrase_attempted"] = False
    if original_unknown == 0:
        return result

    result["rephrase_attempted"] = True
    try:
        rephrased_text = swahili_to_swahili(text, model_name)
    except Exception as e:
        result["flags"] = original_flags + [
            f"Rephrase fallback: NLLB SW->SW call failed ({type(e).__name__}: {e}) -- kept the original "
            f"translation unchanged."]
        return result

    if rephrased_text.strip().lower() == text.strip().lower():
        result["flags"] = original_flags + [
            "Rephrase fallback: NLLB SW->SW regeneration returned the input sentence UNCHANGED -- no "
            "alternate wording available, kept the original translation."]
        result["rephrased_swahili_sentence"] = rephrased_text
        return result

    rephrased_result = fn(rephrased_text, tables)
    rephrased_flags = list(rephrased_result.get("flags") or [])
    rephrased_unknown = _unknown_word_flag_count(rephrased_flags)
    note = (f"Rephrase fallback: original '{text}' had {original_unknown} vocabulary-gap flag(s); NLLB "
            f"SW->SW regenerated it as '{rephrased_text}' ({rephrased_unknown} vocabulary-gap flag(s)).")

    if rephrased_unknown < original_unknown:
        out = dict(rephrased_result)
        out["flags"] = [note + " SWITCHED to the rephrased translation (fewer vocabulary gaps)."] + rephrased_flags
        out["rephrased"] = True
        out["rephrase_attempted"] = True
        out["original_sentence"] = text
        out["original_kitaita_sentence"] = original.get("kitaita_sentence")
        out["rephrased_swahili_sentence"] = rephrased_text
        return out

    result["flags"] = original_flags + [
        note + " KEPT the original translation (rephrasing did not reduce vocabulary gaps)."]
    result["rephrased_swahili_sentence"] = rephrased_text
    return result


def pipeline(text: str, source_lang: str, tables, model_name: str = "facebook/nllb-200-distilled-1.3B") -> dict:
    """source_lang: 'sw' (Swahili text directly) or 'en' (run real NLLB
    first). Returns the Swahili text actually used plus the per-sentence
    Kitaita translation reports."""
    if source_lang == "en":
        swahili_text = english_to_swahili(text, model_name)
    else:
        swahili_text = text
    reports = translate_text(swahili_text, tables)
    return {"input": text, "source_lang": source_lang, "swahili_text": swahili_text, "sentences": reports}


if __name__ == "__main__":
    import sys

    p1, p2, p3 = load_pages()
    tables = build_lookup_tables(p1, p2, p3)

    # 1) Sanity check: sentences reconstructed from already-translate()-tested
    # specs should decompose correctly AND reproduce the exact same output
    # translate() gives when called directly with hand-built args -- this is
    # the regression check for the DECOMPOSER itself, not the generator.
    KNOWN_GOOD = [
        "Mungu akawona.",          # s-type: mungu/kuw-ish? use a real simple one instead below
        "Nyumba itakuwa.",
        "Mtu mkubwa akaona.",
        "Mbwa mkubwa aliona mwanafunzi.",
    ]
    print("=== Known-structure sentences ===")
    for s in KNOWN_GOOD:
        r = translate_sentence(s, tables)
        print(f"\n> {s}")
        if r["error"]:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  -> {r['kitaita_sentence']}")
        for f in r["decomposition"].flags:
            print(f"  [flag] {f}")

    # 2) Genuinely novel free-text sentences, never hand-specified before.
    NOVEL = [
        "Simba mkubwa alikimbia.",
        "Mwalimu mzuri alisoma kitabu.",
        "Kuku mkubwa aliweobua inge.",
        "Chui mkubwa alimwona mbwa.",
    ]
    print("\n=== Novel free-text sentences ===")
    for s in NOVEL:
        r = translate_sentence(s, tables)
        print(f"\n> {s}")
        if r["error"]:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  -> {r['kitaita_sentence']}")
        for f in r["decomposition"].flags:
            print(f"  [flag] {f}")

    if "--with-nllb" in sys.argv:
        print("\n=== English -> (real NLLB) -> Swahili -> Kitaita ===")
        EN = [
            "The big dog saw the student.",
            "The leopard ran.",
        ]
        for e in EN:
            rep = pipeline(e, "en", tables)
            print(f"\n> EN: {e}")
            print(f"  SW (NLLB): {rep['swahili_text']}")
            for sr in rep["sentences"]:
                if sr["error"]:
                    print(f"  ERROR: {sr['error']}")
                else:
                    print(f"  -> Kitaita: {sr['kitaita_sentence']}")
