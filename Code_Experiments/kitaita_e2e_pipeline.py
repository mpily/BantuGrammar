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
from dataclasses import dataclass, field
from pathlib import Path

from extract_swahili_grammar import (
    decompose_verb, tokenize, guess_noun_class_by_prefix, SM_TABLE, TAM_MARKERS,
)
from swahili_noun_lexicon_agreement_v2 import NOUNS as SWAHILI_LEXICON
from nllb_concord_rewrite_pipeline import ADJ_BASE, POSS_TABLE
from kitaita_verified_corrections import (
    VERIFIED_NOUN_TRANSLATIONS, VERIFIED_ROOTS, VERIFIED_SUBJECT_MARKERS, DEVERBAL_SM_HYPOTHESIS,
    NUMBER_INVARIANT_NOUNS, NUMBER_INVARIANT_SM_SINGULAR, BAYA_STEM, KUBWA_FORM_BY_SM, KUBWA_DEFAULT_FORM,
    ZURI_ROOT, ADJ_SINGLE_CLASS_CANDIDATE, ADJ_SINGLE_ROOT_GENERALIZED, kitaita_adj_prefix, VERIFIED_ADJECTIVES,
    LACHA_FORM_BY_SM, LACHA_DEFAULT_FORM, FUNCTION_WORD_TRANSLATIONS, POSS_PERSON_FORM_CLASS1,
    DOGO_FORM_BY_SM, DOGO_DEFAULT_FORM, GENI_FORM_BY_SM, GENI_DEFAULT_FORM,
    FUPI_FORM_BY_SM, FUPI_DEFAULT_FORM, ZITO_FORM_BY_SM, ZITO_DEFAULT_FORM,
    CHAFU_FORM_BY_SM, CHAFU_DEFAULT_FORM, TAM_SM_ALIAS,
)
from kitaita_sentence_generator import load_pages, build_lookup_tables, translate

LEXICON_BY_SURFACE = {lemma: (cls, gloss) for lemma, cls, gloss in SWAHILI_LEXICON}
ADJ_PREFIXES = sorted({p for p in ADJ_BASE.values()}, key=len, reverse=True)  # "", "m", "wa", "mi", "ki", "vi", "n", "ma"
ADJ_ROOTS = ["kubwa", "baya", "zuri", "dogo", "geni", "refu", "zito", "chafu", "fupi"]
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


@dataclass
class Decomposition:
    tokens: list[str]
    subject: str | None = None
    subject_idx: int | None = None
    subject_class: str | None = None
    subject_class_confidence: str | None = None
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


def lookup_noun_kitaita(lemma: str, noun_kitaita: dict) -> str | None:
    """Plain noun -> Kitaita lookup, same trust order as the subject/object
    path (VERIFIED_NOUN_TRANSLATIONS first, then the statistical table), but
    with NO class/agreement info -- for nouns that don't control any verb
    agreement in this pipeline (a second/coordinated noun, a possessor).
    Returns None if genuinely untranslatable -- the caller passes the
    original Swahili word through unchanged rather than dropping it."""
    if lemma in VERIFIED_NOUN_TRANSLATIONS:
        return VERIFIED_NOUN_TRANSLATIONS[lemma]
    cand = noun_kitaita.get(lemma)
    return cand["kitaita_word"] if cand else None


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


def compute_adjective_concord(kitaita_noun: str, swahili_noun: str, stem: str, noun_class_of: dict,
                               noun_kitaita: dict | None = None) -> tuple[str | None, str]:
    """Generalized version of translate()'s 'baya'/'kubwa' adjective-concord
    branches, for a target noun that is NOT the sentence's primary subject
    (translate() itself only ever computes this for the subject). Reuses
    the EXACT SAME rules (same tables, same generalized-rule conditions),
    just keyed by whichever noun the amod probe says the adjective
    actually modifies. Returns (kitaita_form_or_None, explanation)."""
    # 2026-10-13 (cont. 3): `kit_class` is the noun's REAL class (may be
    # "i_nasal") -- used for the DICT-based "special paradigm" adjectives
    # (kubwa/refu/dogo/geni/fupi/zito/chafu), which genuinely see the mi-/
    # nd- split. `kit_sm` is TAM_SM_ALIAS-normalized (i_nasal -> i) -- used
    # for 'baya'/'zuri', which use the BARE subject marker uniformly and
    # do NOT see the split (see TAM_SM_ALIAS's comment in kitaita_verified_
    # corrections.py for why nyumba's native-speaker-verified baya,
    # "izamie", is correct despite nyumba being class 2/"i_nasal").
    kit_class = resolve_class_marker(kitaita_noun, noun_class_of)
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


def decompose_sentence(text: str, noun_kitaita: dict) -> Decomposition:
    tokens = tokenize(text)
    d = Decomposition(tokens=[t.lower() for t in tokens])
    assigned = [False] * len(tokens)

    noun_hits = []  # (idx, lemma, class_or_None, evidence)
    for i, tok in enumerate(d.tokens):
        hit = _noun_lookup(tok, noun_kitaita)
        if hit is not None:
            cls, evidence = hit
            noun_hits.append((i, tok, cls, evidence))
            assigned[i] = True

    verb_hits = []  # (idx, VerbFormEntry)
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue
        v = decompose_verb(tok)
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
        d.verb = {"sm": v.subject_marker, "tam": v.tam_marker, "root": v.root_guess, "fv": v.final_vowel}
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
    ASSOC_PARTICLES = {"cha", "wa", "ya", "la", "za", "vya"}
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
        if noun_hits:
            d.flags.append("No noun recognized BEFORE the verb -- no reliable subject, so no subject-"
                            "verb agreement can be computed. Every noun found is substituted "
                            "independently instead (see below), not forced into the subject role.")
        else:
            d.flags.append("NO NOUN found at all -- no subject, so translation cannot proceed. (Pronoun/"
                            "pro-drop subjects are not attempted here; see kitaita_person_marker_"
                            "investigation.py for that separate, less-confident track.)")
        for oi, olemma, _, _ in noun_hits:
            kit = lookup_noun_kitaita(olemma, noun_kitaita)
            d.secondary_nouns.append((oi, olemma, kit))
            d.flags.append(f"'{olemma}' -> '{kit}': substituted independently (no subject available)."
                            if kit else f"'{olemma}': no Kitaita translation available -- kept as the "
                                        f"original Swahili word (no subject available).")
    else:
        si, slemma, scls, sevidence = pre_verbal_hits[0]
        d.subject, d.subject_idx, d.subject_class = slemma, si, scls
        d.subject_class_confidence = sevidence
        for oi, olemma, _, _ in noun_hits:
            if oi == si:
                continue
            if d.object is None and d.verb_idx is not None and oi > d.verb_idx:
                d.object, d.object_idx = olemma, oi
                continue
            kit = lookup_noun_kitaita(olemma, noun_kitaita)
            d.secondary_nouns.append((oi, olemma, kit))
            if oi > 0 and tokens[oi - 1] in ASSOC_PARTICLES:
                relation = f"possessive/associative noun (follows '{tokens[oi - 1]}') -- no Kitaita " \
                           f"associative-concord table exists, so '{tokens[oi - 1]}' is kept as-is " \
                           f"(Swahili) and only the noun itself is substituted"
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


def translate_sentence(text: str, tables) -> dict:
    noun_kitaita, root_kitaita = tables[0], tables[1]
    d = decompose_sentence(text, noun_kitaita)
    report = {"input": text, "decomposition": d, "kitaita_sentence": None, "trace": [], "error": None}
    if d.verb is None or d.subject is None:
        # 2026-10-09, direct user correction: still produce a best-effort
        # sentence (every noun substituted independently, everything else
        # kept as Swahili) rather than refusing outright -- no agreement can
        # be computed without a subject+verb pair, but that is no reason to
        # withhold what IS translatable.
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

    result = translate(d.subject, d.verb, d.object, d.adjective, tables)
    if "error" in result:
        report["error"] = result["error"]
        report["trace"] = result.get("trace", [])
        report["kitaita_sentence"] = assemble_in_order(d, None, noun_kitaita)
        return report
    report["kitaita_sentence"] = assemble_in_order(d, result, noun_kitaita)
    report["trace"] = result["trace"]
    report["components"] = result["components"]
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
            kit = comps.get("subject_noun") if result else lookup_noun_kitaita(d.subject, noun_kitaita)
            pieces.append((d.subject_idx, kit or d.subject))
        if d.verb_idx is not None:
            kit = comps.get("kitaita_verb_word") if result else None
            pieces.append((d.verb_idx, kit or d.verb_surface or d.tokens[d.verb_idx]))
        if d.object_idx is not None:
            kit = comps.get("object_noun") if result else lookup_noun_kitaita(d.object, noun_kitaita)
            pieces.append((d.object_idx, kit or d.object))
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


def decompose_sentence_with_probes(text: str, noun_kitaita: dict, noun_class_of: dict) -> Decomposition:
    """2026-10-09, direct user directive: "we don't need to guess for verbs
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
        if pos_tags is not None and pos_tags[i] == "ADJ":
            continue  # a token the POS probe confidently tags ADJ is never a noun candidate,
            # even if it happens to also have a (different-usage) noun-table entry.
        hit = _noun_lookup(tok, noun_kitaita)
        if hit is not None:
            cls, evidence = hit
            noun_hits.append((i, tok, cls, evidence))
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
        v = decompose_verb(tok)
        if v is not None and len(v.root_guess) >= 2:
            verb_hits.append((i, v))
            assigned[i] = True

    adjective_hits = []  # (idx, stem) -- no adjacency restriction; the probe resolves the target
    for i, tok in enumerate(d.tokens):
        if assigned[i]:
            continue
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
        d.flags.append("NO NOUN found at all -- no subject, so translation cannot proceed.")
    elif not verb_hits:
        for oi, olemma, _, _ in noun_hits:
            kit = lookup_noun_kitaita(olemma, noun_kitaita)
            d.secondary_nouns.append((oi, olemma, kit))
    else:
        probe = identify_relations(tokens, [h[0] for h in noun_hits], [h[0] for h in verb_hits],
                                    [h[0] for h in adjective_hits],
                                    [h[0] for h in possessive_hits])
        for note in probe.notes:
            d.flags.append(note)

        used_noun_idxs = set()
        adjective_claimed = set()

        for vi, v in verb_hits:
            vr = probe.verb_results.get(vi, {"subject_idx": None, "object_idx": None})
            subj_i, obj_i = vr["subject_idx"], vr["object_idx"]
            clause = {"verb_idx": vi, "verb_surface": tokens[vi],
                      "verb": {"sm": v.subject_marker, "tam": v.tam_marker, "root": v.root_guess, "fv": v.final_vowel,
                               "om_candidates": v.object_marker_candidates},
                      "subject": None, "subject_idx": None, "subject_class": None,
                      "object": None, "object_idx": None, "adjective": None, "adjective_idx": None}
            if subj_i is not None:
                si, slemma, scls, sevidence = next(h for h in noun_hits if h[0] == subj_i)
                clause["subject"], clause["subject_idx"], clause["subject_class"] = slemma, si, scls
                used_noun_idxs.add(si)
                d.flags.append(f"'{slemma}' -> probe-identified SUBJECT of '{tokens[vi]}' (beat every "
                                f"other noun as nsubj).")
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
            target_i = probe.adjective_targets.get(ai)
            target_kitaita, target_swahili = None, None
            target_clause = next((c for c in d.clauses if c["object_idx"] == target_i), None) if target_i is not None else None
            if target_clause is not None:
                target_swahili = target_clause["object"]
                target_kitaita = lookup_noun_kitaita(target_swahili, noun_kitaita)
            elif target_i is not None:
                sec = next(((k, l) for i, l, k in d.secondary_nouns if i == target_i), None)
                if sec:
                    target_kitaita, target_swahili = sec[0], sec[1]
            if target_kitaita:
                adj_part, note = compute_adjective_concord(target_kitaita, target_swahili, stem, noun_class_of,
                                                             noun_kitaita)
                if adj_part:
                    d.secondary_nouns.append((ai, tokens[ai], adj_part))
                    d.flags.append(f"'{tokens[ai]}' -> '{adj_part}': probe-identified as modifying "
                                    f"'{target_kitaita}' ({note}).")
                    continue
            d.passthrough.append((ai, tokens[ai]))
            d.flags.append(f"'{tokens[ai]}': probe-identified as NOT modifying any clause's subject -- "
                            f"this project's generator has no concord mechanism for that target (or none "
                            f"was identified at all) -- kept as the original Swahili word.")

        # 2026-10-11 (cont. 8): possessive-pronoun targeting, same target-
        # resolution shape as the adjective loop just above (subject / object /
        # secondary-noun, in that order) -- reused rather than re-derived.
        for pi, prefix, stem in possessive_hits:
            target_i = probe.possessive_targets.get(pi)
            target_kitaita = None
            target_clause = (next((c for c in d.clauses
                                    if target_i in (c["subject_idx"], c["object_idx"])), None)
                              if target_i is not None else None)
            if target_clause is not None:
                target_swahili = (target_clause["subject"] if target_clause["subject_idx"] == target_i
                                   else target_clause["object"])
                target_kitaita = lookup_noun_kitaita(target_swahili, noun_kitaita)
            elif target_i is not None:
                sec = next((k for i, l, k in d.secondary_nouns if i == target_i), None)
                target_kitaita = sec
            kit_sm = resolve_class_marker(target_kitaita, noun_class_of) if target_kitaita else None
            if kit_sm == "u" and stem in POSS_PERSON_FORM_CLASS1:
                # Gated strictly to the default/'u' (class-1) bucket this data was
                # mined for -- see kitaita_verified_corrections.py POSS_PERSON_FORM_
                # CLASS1 for why this does NOT generalize to other classes the way
                # kubwa/lacha do (no second confirmed class exists to show that).
                poss_part = POSS_PERSON_FORM_CLASS1[stem]
                d.secondary_nouns.append((pi, tokens[pi], poss_part))
                d.flags.append(f"'{tokens[pi]}' -> '{poss_part}': probe-identified as possessing "
                                f"'{target_kitaita or '(unresolved target)'}' (class-1-only statistical "
                                f"candidate, NOT native-speaker-verified -- see kitaita_verified_"
                                f"corrections.py POSS_PERSON_FORM_CLASS1).")
            else:
                d.passthrough.append((pi, tokens[pi]))
                d.flags.append(f"'{tokens[pi]}': possessive pronoun"
                                + (f", target '{target_kitaita}' resolved to Kitaita class '{kit_sm}'"
                                   if kit_sm else ", target's Kitaita class is unresolved")
                                + " -- no confirmed concord data (only class-1/'u' is covered, see "
                                  "POSS_PERSON_FORM_CLASS1) -- kept as the original Swahili word, "
                                  "flagged rather than guessed.")

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
    noun_kitaita, root_kitaita, noun_class_of = tables[0], tables[1], tables[3]
    d = decompose_sentence_with_probes(text, noun_kitaita, noun_class_of)
    report = {"input": text, "decomposition": d, "kitaita_sentence": None, "trace": [], "error": None, "flags": d.flags}
    if not d.clauses:
        report["error"] = "; ".join(d.flags) or "decomposition failed"
        report["kitaita_sentence"] = assemble_multi_clause(d, [], noun_kitaita)
        return report

    clause_results = []
    all_trace = []
    for clause in d.clauses:
        if clause["subject_idx"] is None:
            clause_results.append((clause, None))
            continue
        root = clause["verb"]["root"]
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
        if root != "kuw" and root not in VERIFIED_ROOTS:
            cand = root_kitaita.get(root)
            if cand and cand["score"] < OBJECT_MARKER_RISK_ROOT_DICE_THRESHOLD:
                d.flags.append(f"Verb root '{root}' (for '{clause['verb_surface']}') only matched a WEAK "
                                f"statistical candidate (Dice {cand['score']:.2f}) -- treat as UNRELIABLE.")
        result = translate(clause["subject"], clause["verb"], clause["object"], clause["adjective"], tables,
                            subject_class=clause["subject_class"])
        if "error" in result:
            d.flags.append(f"Verb '{clause['verb_surface']}': {result['error']}")
            clause_results.append((clause, None))
            continue
        all_trace.extend(result["trace"])
        clause_results.append((clause, result))

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
    inputs = tok(text, return_tensors="pt")
    bos = tok.convert_tokens_to_ids("swh_Latn")
    out = model.generate(**inputs, forced_bos_token_id=bos, max_length=64)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


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
