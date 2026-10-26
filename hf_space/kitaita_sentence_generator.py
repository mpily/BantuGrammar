# -*- coding: utf-8 -*-
"""
kitaita_sentence_generator.py

2026-10-07. Direct user request: take a simple Swahili sentence, map
its noun(s) and verb root to Kitaita, substitute Kitaita subject/TAM
markers (and adjective concord where we have evidence for it), and
output the resulting Kitaita sentence -- using ONLY the evidence this
project has actually built (reliable noun translations, root-level
verb alignment, the native-speaker-corrected TAM table, the from-
scratch noun classes), never inventing a form the data doesn't
support.

This is NOT a general Swahili parser -- the Swahili sentence is
hand-decomposed (SM + TAM + root + FV for the verb; class for the
noun) since this project has no Swahili morphological analyser beyond
extract_swahili_grammar.py's own greedy decomposer, which this script
reuses directly rather than re-implementing. Each substitution step
records its OWN evidence (which data file, what Dice score, how many
nouns' worth of subject-marker data) so a human can see exactly why
each piece of the output was chosen, and where it is a guess.
"""
from __future__ import annotations

import json
from pathlib import Path

from kitaita_verified_corrections import (
    VERIFIED_ROOTS, VERIFIED_KUWA_ROOT_EXISTENTIAL, VERIFIED_KUWA_ROOT_COPULA,
    VERIFIED_SUBJECT_MARKERS, VERIFIED_ADJECTIVES, VERIFIED_SENTENCES,
    PRESENT_SM_VOWEL_CONFIRMED, PRESENT_TAM_DEFAULT, KOGH_RE_WCLASS_VOWEL_CORPUS,
    KOGH_RE_FRONT_VOWEL_SHIFTS,
    BAYA_STEM, KUWA_PAST_ALTERNATES, NUMBER_INVARIANT_SM_SINGULAR, NUMBER_INVARIANT_SM_PLURAL,
    NUMBER_INVARIANT_NOUNS,
    KUBWA_FORM_BY_SM, KUBWA_DEFAULT_FORM, ZURI_ROOT, SAFI_STEM, MOJA_ROOT, INGI_FORM_BY_SM, INGI_DEFAULT_FORM,
    INGI_SURFACE_FORMS, ADJ_SINGLE_CLASS_CANDIDATE,
    ADJ_SINGLE_ROOT_GENERALIZED, LACHA_FORM_BY_SM, LACHA_DEFAULT_FORM,
    DOGO_FORM_BY_SM, DOGO_DEFAULT_FORM, GENI_FORM_BY_SM, GENI_DEFAULT_FORM,
    FUPI_FORM_BY_SM, FUPI_DEFAULT_FORM, ZITO_FORM_BY_SM, ZITO_DEFAULT_FORM,
    CHAFU_FORM_BY_SM, CHAFU_DEFAULT_FORM, TAM_SM_ALIAS,
    kitaita_adj_prefix, kogh_re_o_shift,
    SWAHILI_CLASS_DEFAULT_SM_GUESS, ANIMATE_SG_EXCLUDED_FROM_DEFAULT,
    SWAHILI_PERSONAL_PRONOUNS, PERSONAL_PRONOUN_KITAITA_CLASS, SWAHILI_PRONOUN_TO_PERSON,
    PERSON_SUBJECT_MARKERS_BY_TAM, PERSON_SM_DEFAULT, VERIFIED_PAST_TAM_BY_SM,
    VERIFIED_NOUN_TRANSLATIONS, FUTURE_KUWA_FULL_ROOT_MARKERS, DEVERBAL_SM_HYPOTHESIS,
    CONDITIONAL_TAM_HYPOTHESIS,
    NEGATIVE_MARKER_KITAITA, NEGATIVE_FINAL_VOWEL, NEGATIVE_SM_BY_CLASS_OVERRIDE,
    VERIFIED_IDIOMS,
    NEGATIVE_PERSON_PREFIX_BY_TENSE, NEGATIVE_PERSON_FINAL_VOWEL,
    NEGATIVE_CLASS_FUTURE_FINAL_VOWEL, NEGATIVE_CLASS_PAST_SM_FORM,
    NEGATIVE_CLASS_PAST_SUFFIX, NEGATIVE_CLASS_PAST_VA_OVERRIDE,
    NEGATIVE_ANIMATE_CLASS_PREFIX_BY_TENSE,
    swahili_passive_base_candidates, kitaita_passive_form,
)

PERSON_LABELS = {"1sg": "mimi (I)", "2sg": "wewe (you-sg)", "1pl": "sisi (we)", "2pl": "ninyi (you-pl)"}

OUT_DIR = Path(__file__).parent


def load_pages():
    p1 = json.loads((OUT_DIR / "page1_vocabulary.json").read_text(encoding="utf-8"))
    p2 = json.loads((OUT_DIR / "page2_noun_classes.json").read_text(encoding="utf-8"))
    p3 = json.loads((OUT_DIR / "page3_verb_rules.json").read_text(encoding="utf-8"))
    return p1, p2, p3


def _derive_fallback_root(forms, min_suffix_len=4):
    """2026-10-15. The per-root fallback (root_kitaita_fallback below) was
    picking the single globally-best-Dice SURFACE FORM across every attested
    person/tense of a Swahili root, with no regard for whether that form's
    person/tense matches the sentence actually being translated (confirmed
    directly on 'pend'/love: the stored fallback was 'unapenda'->'kwadakunda',
    a 2sg form, Dice 0.57, even though a 3sg candidate 'anapenda'->'kiwachenyi'
    existed at Dice 0.44 -- picked purely because it scored higher, not because
    it matched any sentence's actual subject). That means a correctly-identified
    3rd-person noun subject got a verb form carrying someone else's person
    marker baked in, with no way to re-inflect it for the real subject.

    Rather than trust any single mined surface form, this looks for a root
    hiding inside >=2 INDEPENDENTLY-aligned fallback forms (longest common
    suffix across all pairs), the same 'agreement across independent evidence'
    bar this project uses everywhere else (e.g. the n>=3 correction bar,
    cont. 8's cross-cluster confirmations). If found, that shared material is
    treated as the real root (final vowel stripped per this file's kit_root
    convention, so normal SM+TAM substitution runs on it like any other root)
    instead of reusing one person-specific whole word verbatim. Confirmed by
    hand on 'pend': ukipenda->'kukakunda' and unapenda->'kwadakunda' share the
    suffix 'akunda' (6 chars) -- anapenda->'kiwachenyi' shares nothing with
    either and is correctly left out as the non-agreeing one. Explicitly NOT
    native-speaker-verified -- flagged as DERIVED in the trace, lowest
    priority of the three root-resolution tiers (VERIFIED_ROOTS > root-level
    Dice > this), used only when neither of those exists."""
    candidates = []
    for f in forms:
        cands = f.get("kitaita_candidates")
        if cands:
            candidates.append((f["surface"], cands[0]["kitaita_word"], cands[0]["best_dice"]))
    if len(candidates) < 2:
        return None
    best_suffix, best_pair = "", None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            w1, w2 = candidates[i][1], candidates[j][1]
            k = 0
            while k < min(len(w1), len(w2)) and w1[-(k + 1)] == w2[-(k + 1)]:
                k += 1
            suffix = w1[-k:] if k else ""
            if len(suffix) > len(best_suffix):
                best_suffix, best_pair = suffix, (candidates[i], candidates[j])
    if len(best_suffix) < min_suffix_len:
        return None
    # Bantu verb roots are canonically consonant-initial. A matched suffix that
    # STARTS with a vowel most likely overshot into the preceding TAM syllable's
    # own final vowel (e.g. 'pend''s two fallback forms both happen to have a
    # TAM ending in '-a' -- 'ka'/'da' -- so the match extends one spurious
    # character left of the real root boundary: 'akunda' instead of 'kunda').
    # Trim leading vowels while enough material remains, so the match collapses
    # to its consonant-initial core rather than carrying that coincidence into
    # the derived root.
    core = best_suffix
    while len(core) > min_suffix_len and core[0] in "aeiou":
        core = core[1:]
    root = core[:-1] if core[-1] in "aeiou" else core
    return {"kitaita_root": root, "suffix": core, "support": best_pair}


_SWAHILI_WORD_FREQ: dict | None = None


def swahili_word_freq() -> dict:
    """Raw corpus frequency of every Swahili word TYPE, pooled across all
    3 Bible versions (multi_version_pooled_swahili_counts.json, already
    built by this project's own corpus tooling -- no new data collected
    here). 2026-10-22, direct user request: a general mechanism for
    "where multiple words map to the same word", using basic statistics
    (frequency) as the signal -- see candidate_frequency_score() below
    for how it's actually used. Cached at module level since this file
    is read once and build_lookup_tables() may be called more than once
    per process."""
    global _SWAHILI_WORD_FREQ
    if _SWAHILI_WORD_FREQ is None:
        path = OUT_DIR / "multi_version_pooled_swahili_counts.json"
        _SWAHILI_WORD_FREQ = dict(json.loads(path.read_text(encoding="utf-8")))
    return _SWAHILI_WORD_FREQ


def candidate_frequency_score(candidate: dict, lemma: str, freq: dict) -> tuple[float, str]:
    """Re-scores one statistical kitaita_candidates/root_level_kitaita_
    candidates entry, combining its existing Dice score with TWO cheap,
    already-available frequency-shaped signals, instead of trusting Dice
    alone (what build_lookup_tables() did before this function existed --
    just `candidates[0]`, already Dice-sorted, no further check).

    Direct motivation: the "papa" (Pope) bug, 2026-10-22 (see docs/
    status.md) -- papa's own candidate "kukanye" had a plausible-looking
    Dice score (0.33) but was actually noise: reading its `per_version`
    breakdown, THREE independently-translated Bible versions each gave a
    DIFFERENT "best partner word" for the same Kitaita candidate (SUV:
    "papa" itself; NEN: "kutana"; BHN: "umeondoka") -- i.e. "papa" only
    won its OWN best-alignment slot in 1 of its 3 attestations. A
    genuinely reliable alignment should see the SAME source word win
    repeatedly, independent versions corroborating each other (exactly
    this project's own existing "high" confidence tier already requires,
    e.g. "nchi"/"siku" above -- this just makes that same judgment
    available as a per-candidate SCORE, not only a hand-read one-off).

    1. SELF-CONSISTENCY: fraction of `per_version` entries where the
       winning partner word IS this lemma/root itself. A candidate with
       no `per_version` breakdown at all (root-level candidates often
       don't carry one) is not penalized for a check it has no data for
       (consistency defaults to 1.0).
    2. FREQUENCY PLAUSIBILITY: how much of the LEMMA's own total corpus
       frequency the alignment's cooccurrence count actually represents
       (`coverage`). A thin cooccur_count=1 backing a word that occurs
       21 times in the corpus (papa's case) means the alignment explains
       essentially none of that word's real occurrences -- more likely
       coincidence than a genuine recurring correspondence. A lemma with
       NO frequency entry at all (freq=0, e.g. a rare proper name) isn't
       penalized on this axis either, same "don't penalize for missing
       data" discipline as (1).

    Returns (score, note) -- note is a short, printable explanation for
    the trace/flag text, same spirit as this project's existing
    "confidence" strings. Deliberately NOT a confident probability or a
    hard accept/reject cutoff -- it only RE-RANKS competing candidates
    and flags low scores; this is a basic-statistics heuristic, not a
    learned or validated model.

    Works for BOTH the noun-level candidate shape ("best_dice",
    "per_version") and the root-level candidate shape ("score", no
    "per_version" -- root-level candidates don't carry a per-version
    breakdown at all, so self-consistency defaults to 1.0 for them,
    same "don't penalize for missing data" rule as an unlisted lemma_
    freq)."""
    dice = candidate.get("best_dice", candidate.get("score", 0.0))
    per_version = candidate.get("per_version") or []
    if per_version:
        matches = sum(1 for pv in per_version if pv.get("swahili_word") == lemma)
        consistency = matches / len(per_version)
    else:
        matches, consistency = None, 1.0
    lemma_freq = freq.get(lemma, 0)
    cooccur = max((pv.get("cooccur_count", 0) for pv in per_version), default=candidate.get("cooccur_count", 0) or 0)
    coverage = min(1.0, cooccur / lemma_freq) if lemma_freq else 1.0
    score = dice * (0.25 + 0.75 * consistency) * (0.5 + 0.5 * coverage)
    note = (f"dice={dice:.2f}, self-consistency={consistency:.2f}"
            + (f" ({matches}/{len(per_version)} versions)" if per_version else " (no per-version data)")
            + f", freq-coverage={coverage:.2f} (lemma seen {lemma_freq}x, cooccur {cooccur}x)")
    return score, note


def build_lookup_tables(p1, p2, p3):
    freq = swahili_word_freq()
    noun_kitaita = {}
    for n in p1["nouns"]:
        cands = n.get("kitaita_candidates")
        if not cands:
            continue
        lemma = n["lemma"]
        if len(cands) > 1:
            # 2026-10-22: multiple competing Kitaita candidates for the SAME
            # Swahili lemma (e.g. "siku"->"ituku"/"matuku") -- re-rank by the
            # frequency-aware score instead of trusting Dice-only ordering.
            scored = [(c, *candidate_frequency_score(c, lemma, freq)) for c in cands]
            scored.sort(key=lambda t: -t[1])
            best, best_score, best_note = scored[0]
        else:
            best = cands[0]
            best_score, best_note = candidate_frequency_score(best, lemma, freq)
        best = dict(best)
        best["frequency_reliability"] = {"score": round(best_score, 4), "note": best_note}
        noun_kitaita[lemma] = best
    root_kitaita, root_kitaita_fallback, root_kitaita_derived = {}, {}, {}
    for vr in p1["verb_roots"]:
        cands = vr.get("root_level_kitaita_candidates")
        if cands:
            root_lemma = vr["root_guess"]
            if len(cands) > 1:
                # 2026-10-22: same frequency-aware reranking as noun_kitaita
                # above -- multiple competing Kitaita ROOT candidates for the
                # same Swahili root (e.g. "va" -> "rwa"/"rw"/"joh").
                scored = [(c, *candidate_frequency_score(c, root_lemma, freq)) for c in cands]
                scored.sort(key=lambda t: -t[1])
                best, best_score, best_note = scored[0]
            else:
                best = cands[0]
                best_score, best_note = candidate_frequency_score(best, root_lemma, freq)
            best = dict(best)
            best["frequency_reliability"] = {"score": round(best_score, 4), "note": best_note}
            root_kitaita[root_lemma] = best
        old = vr.get("root_kitaita_candidate")
        if old:
            root_kitaita_fallback[vr["root_guess"]] = old
        derived = _derive_fallback_root(vr.get("forms", []))
        if derived:
            root_kitaita_derived[vr["root_guess"]] = derived
    corrected = p2["kitaita_corrected_classes"]
    noun_class_of = {}  # kitaita noun -> accepted-class follower prefix + its subject-marker subgroups
    for cls in corrected["accepted_classes"]:
        for sm, members in cls["subject_marker_subgroups"].items():
            if sm == "(no subject-marker data)":  # sentinel bucket, not a real marker
                continue
            for m in members:
                noun_class_of[m] = {"class": cls["follower_marker"], "subject_marker": sm}
    allomorphy = p3["allomorphy_investigation"]
    return (noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy,
            root_kitaita_derived)


TAM_SWAHILI_TO_LABEL = {"na": "present", "li": "past", "ta": "future", "me": "perfect", "ki": "conditional"}


def past_variant_for_sm(sm: str, allomorphy: dict) -> dict:
    """Returns {'variant':..., 'support':..., 'confident': bool} using
    the re/de-by-subject-marker breakdown already computed (project
    rule: n<8 -> no confident claim, report the number instead)."""
    if sm in VERIFIED_PAST_TAM_BY_SM:
        # Direct native-speaker correction, trusted even at n=1 like every other
        # VERIFIED_* table -- takes priority over the statistical allomorphy table,
        # which has NO entry at all for some markers (e.g. 'ji': the from-scratch
        # induction's allomorphy pass never saw 'ji' as a marker, since nchi/ituku/
        # isanga/indi were only hand-corrected to 'ji' afterward -- 2026-10-08
        # cont. 4, sanduku/s90 is the first-ever test of 'ji' + plain past tense).
        return {"variant": VERIFIED_PAST_TAM_BY_SM[sm], "support": None, "confident": True}
    sm_by_variant = allomorphy["past_re_de"]["subject_marker_by_variant"]
    best_variant, best_support = None, 0
    for variant, counts in sm_by_variant.items():
        for s, n in counts:
            if s == sm and n > best_support:
                best_variant, best_support = variant, n
    if best_variant is None:
        return {"variant": None, "support": 0, "confident": False}
    return {"variant": best_variant, "support": best_support, "confident": best_support >= 8}


def resolve_verb_root(swahili_root: str, tables) -> tuple[str, str, bool]:
    """Mirrors translate()'s own non-'kuw' verb-ROOT resolution chain
    (VERIFIED_ROOTS -> root-level Dice candidate -> derived fallback ->
    per-SURFACE-FORM fallback -> keep-the-Swahili-root 'Taita-fy' tier,
    2026-10-19) -- factored out here, SAME precedent as kitaita_e2e_
    pipeline.resolve_class_marker() (which duplicates a piece of
    translate()'s NOUN-side resolution order for reuse outside the
    subject slot, for the identical reason: a caller that has a root/
    noun with NO subject, no SM, no TAM of its own -- translate() cannot
    be called for it at all). Needed for the associative-descriptor
    construction ("maji ya kunywa" -> "machi ghenywa"): the infinitive
    complement after an associative particle is not a finite clause, it
    has no subject to resolve agreement from, but its ROOT should still
    get this project's best available Kitaita translation (or the
    Taita-fy fallback) exactly like any other verb root does.

    Kept in sync with translate() DELIBERATELY, not re-derived
    independently -- if that chain's order or tiers change, update
    both; this is a narrower, read-only reuse of the SAME tables, not a
    second, independently-evolving root-translation policy.

    Returns (kit_root, tier, is_full_form). `tier` is one of "verified"/
    "root-level"/"derived"/"per-form-fallback"/"untranslated". `is_full_
    form` mirrors translate()'s root_is_full_form -- True only for the
    "per-form-fallback" tier, where kit_root is already a whole inflected
    Kitaita word, not a bare root (the caller should use it as-is, not
    try to further prefix/fuse it)."""
    noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy, root_kitaita_derived = tables
    if swahili_root in VERIFIED_ROOTS:
        return VERIFIED_ROOTS[swahili_root], "verified", False
    passive_base = next((b for b in swahili_passive_base_candidates(swahili_root) if b in VERIFIED_ROOTS), None)
    if passive_base is not None:
        # 2026-10-20: kept in sync with translate()'s own passive-extension
        # branch, same evidence -- see kitaita_verified_corrections.py.
        return kitaita_passive_form(VERIFIED_ROOTS[passive_base]), "verified", False
    cand = root_kitaita.get(swahili_root)
    if cand is not None:
        return cand["kitaita_root"], "root-level", False
    derived = root_kitaita_derived.get(swahili_root)
    if derived is not None:
        return derived["kitaita_root"], "derived", False
    fb = root_kitaita_fallback.get(swahili_root)
    if fb is not None:
        return fb["kitaita_word"], "per-form-fallback", True
    return swahili_root, "untranslated", False


def translate(swahili_subject_noun: str | None, swahili_verb: dict, swahili_object_noun: str | None,
              swahili_adjective: dict | None, tables, person: str | None = None,
              subject_class: str | None = None, negated: bool = False,
              negated_tense: str = "present") -> dict:
    """swahili_verb: {'sm':.., 'tam':.., 'root':.., 'fv':..} (already
    decomposed -- see module docstring on why this isn't auto-parsed).
    swahili_adjective: {'stem':.., 'class':..} or None.

    person: one of "1sg"/"2sg"/"1pl"/"2pl" for a PRONOUN subject
    ("I"/"you"/"we"/"you-pl") instead of a noun -- pass
    swahili_subject_noun=None when using this. 2026-10-08 addition (see
    kitaita_person_marker_investigation.py); Kitaita is pro-drop in the
    corpus data this was built from (no independent pronoun word ever
    appears alongside the verb prefix in the clean examples), so the
    output is the inflected verb alone, flagged as such rather than
    inventing an independent-pronoun word with zero evidence.

    negated: 2026-10-15 addition. When True, the normal SM+TAM+root+FV
    assembly is replaced entirely by the native-speaker-submitted
    negative-present pattern (see NEGATIVE_* in kitaita_verified_
    corrections.py) -- root resolution (VERIFIED_ROOTS/root-level Dice/
    derived-fallback) is UNCHANGED, only how the subject marker and verb
    are assembled at the end differs. swahili_verb['tam']/['sm'] are
    ignored in this mode (negation has its own, separate paradigm, same
    as the Swahili side's decompose_negative_verb).

    negated_tense: 2026-10-21 addition, one of "present"(default)/"past"/
    "future"/"subjunctive_imperative". Selects which of the 4 negative
    templates native speakers submitted (Kitaita Word Bank, 2026-10-21)
    applies -- see NEGATIVE_PERSON_PREFIX_BY_TENSE / NEGATIVE_CLASS_*
    in kitaita_verified_corrections.py for the full evidence. The 4
    tenses are NOT simple variations of one rule (person vs. class
    subjects use genuinely different templates outside present tense,
    and past uses a dedicated negative-only TAM, not each class's own
    affirmative past marker) -- callers must pass the right one, there
    is no safe default beyond "present" (the only tense this parameter
    existed for before this addition)."""
    noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy, root_kitaita_derived = tables
    trace = []

    if person is not None:
        kit_subject = None
        trace.append(f"Pronoun subject '{PERSON_LABELS[person]}' -- Kitaita is pro-drop in every "
                     f"clean corpus example found (no separate pronoun word, just the verb prefix); "
                     f"output is the inflected verb alone (kitaita_person_marker_investigation.py)")
        if negated:
            # 2026-10-15 (extended 2026-10-21 to all 4 tenses): negation has
            # its own separate person-marker set (NEGATIVE_PERSON_PREFIX_BY_
            # TENSE), applied at the final assembly step below -- this
            # affirmative-TAM-keyed lookup doesn't apply and would just emit
            # a confusing "no data" line.
            kit_sm = None
        else:
            # Person markers are keyed by (person, TAM) -- 2026-10-08 direct correction: the
            # marker's surface form itself depends on the following TAM (same KIND of
            # relationship PRESENT_SM_VOWEL_CONFIRMED/kogh_re_o_shift already established for
            # noun-class markers), not one fixed consonant per person with tense as an
            # afterthought. "ka" is looked up directly (TAM_SWAHILI_TO_LABEL has no entry for
            # Swahili's narrative -ka-, handled by its own branch below).
            person_tam_key = "ka" if swahili_verb["tam"] == "ka" else TAM_SWAHILI_TO_LABEL.get(swahili_verb["tam"])
            if (person, person_tam_key) in PERSON_SUBJECT_MARKERS_BY_TAM:
                kit_sm = PERSON_SUBJECT_MARKERS_BY_TAM[(person, person_tam_key)]
                trace.append(f"Person '{person}' subject marker before Kitaita TAM-category "
                             f"'{person_tam_key}' -> CORPUS-INDUCED HYPOTHESIS '{kit_sm}-' (NOT "
                             f"native-speaker-verified, n<8 -- see kitaita_person_marker_"
                             f"investigation.py and kitaita_verified_corrections.py for the evidence)")
            elif person in PERSON_SM_DEFAULT:
                # 2026-10-20, direct user request ("add default person markers
                # for sentences that don't have explicit nouns") -- no entry
                # for THIS specific TAM category, but see PERSON_SM_DEFAULT's
                # own comment for why this person's FUTURE-tense marker is a
                # reasonable fallback rather than refusing outright (the exact
                # gap "Umesongeza kiti." hit: 2sg + perfect had no entry at
                # all above). Explicitly weaker than a real per-TAM entry --
                # flagged as a default, not presented as equally confirmed.
                kit_sm = PERSON_SM_DEFAULT[person]
                trace.append(f"Person '{person}' subject marker for TAM-category '{person_tam_key}': "
                             f"no entry for this specific TAM, used this person's DEFAULT (future-"
                             f"tense-matching) marker '{kit_sm}-' instead of refusing outright -- "
                             f"PERSON_SM_DEFAULT, weaker than a real (person, TAM) entry, not the "
                             f"same confidence as PERSON_SUBJECT_MARKERS_BY_TAM.")
            else:
                kit_sm = None
                trace.append(f"Person '{person}' subject marker for TAM-category '{person_tam_key}': "
                             f"NO corpus evidence at all -- FLAGGED, not guessed")
        sentence_key = None
    else:
        sentence_key = (swahili_subject_noun, swahili_verb["root"], swahili_verb["tam"],
                         swahili_adjective["stem"] if swahili_adjective else None)
    if sentence_key is not None and sentence_key in VERIFIED_SENTENCES:
        verified = VERIFIED_SENTENCES[sentence_key]
        trace.append(f"EXACT native-speaker-verified sentence for this (noun, root, TAM"
                     f"{', adjective' if swahili_adjective else ''}) combination -- used as-is "
                     f"rather than recomposed from components (several interacting unknowns here "
                     f"haven't been generalized into separate rules yet; see kitaita_verified_"
                     f"corrections.py).")
        return {"kitaita_sentence": verified, "trace": trace, "object": None,
                "adjective_flagged": False, "components": {"verified_exact_sentence": True}}

    if person is None:
        if swahili_subject_noun in VERIFIED_NOUN_TRANSLATIONS:
            kit_subject = VERIFIED_NOUN_TRANSLATIONS[swahili_subject_noun]
            trace.append(f"Noun '{swahili_subject_noun}' -> Kitaita '{kit_subject}' "
                         f"(native-speaker-VERIFIED translation -- overrides the statistical "
                         f"candidate, which was wrong; see VERIFIED_NOUN_TRANSLATIONS in "
                         f"kitaita_verified_corrections.py)")
        else:
            subj = noun_kitaita.get(swahili_subject_noun)
            if subj is None:
                kit_subject = None
            else:
                kit_subject = subj["kitaita_word"]
                trace.append(f"Noun '{swahili_subject_noun}' -> Kitaita '{kit_subject}' "
                             f"(Dice {subj['best_dice']:.2f}, {subj['n_versions_attested']}/3 versions, "
                             f"{subj['confidence']})")

        if swahili_subject_noun in SWAHILI_PERSONAL_PRONOUNS:
            kit_sm = PERSONAL_PRONOUN_KITAITA_CLASS
            trace.append(f"Subject '{swahili_subject_noun}' is a personal pronoun -- defaults to "
                         f"mundu's class '{kit_sm}-' regardless of which Kitaita word it translates "
                         f"to (direct user instruction, 2026-10-22; see SWAHILI_PERSONAL_PRONOUNS in "
                         f"kitaita_verified_corrections.py)")
        elif kit_subject is None and swahili_verb["root"] == "oko_exist":
            # 2026-10-22: bug #2 fix -- the invariant locative-existential
            # "oko" (see the 'oko_exist' branch below) needs NO subject
            # marker of its own (root_is_full_form bypasses kit_sm
            # entirely), so it should never hit the "no reliable
            # translation" hard error below just because the relation
            # probe couldn't find a subject for it (the test sentences
            # this verb is needed for are existential-inversion word
            # order -- "Yuko wapi paka wangu?" -- which the probe was
            # never trained to recognize; a genuine parsing gap, not a
            # reason to drop the whole clause when the verb itself
            # doesn't need the answer).
            kit_sm = None
            trace.append("Verb 'yuko'/'yupo'/'yumo' needs no subject marker of its own (invariant) -- "
                         "proceeding without one even though no subject was resolved")
        elif kit_subject is None and swahili_verb["root"] == "ko_have" and subject_class is None:
            # 2026-10-22: vocabulary-gap fix (plural "have", see
            # PLURAL_HAVE_FORMS in kitaita_e2e_pipeline.py) -- "wana" IS
            # Swahili's own class-2/ANIMATE-PL marker, so when no real
            # subject noun was resolved (the common case: this verb's
            # object, e.g. "kazi", is usually the only recognized noun
            # nearby, and coordinated non-lexicon subjects like "Mama na
            # bibi" have no settled class-resolution rule in this project
            # -- see kitaita-project/CLAUDE.md's open-decisions section),
            # default to class 'wi' (wandu) for the SAME reason bug #4's
            # 3rd-person pro-drop fix does: using the Swahili marker's
            # OWN meaning, not a guess across unrelated classes.
            kit_sm = "wi"
            trace.append("Verb 'wana' (they have) needs a class subject marker but none was resolved -- "
                         "defaulting to class 'wi-' (wandu), the class Swahili's OWN 'wa-' marker itself "
                         "encodes, same reasoning as the 3rd-person pro-drop default")
        elif kit_subject is None:
            # 2026-10-11 (cont.): class-level fallback instead of an immediate hard
            # stop -- see SWAHILI_CLASS_DEFAULT_SM_GUESS's own comment for scope and
            # the exclusion list. Only reached when person is None AND the noun had
            # no VERIFIED translation AND no statistical candidate at all.
            excluded = subject_class == "ANIMATE-SG" and swahili_subject_noun in ANIMATE_SG_EXCLUDED_FROM_DEFAULT
            fallback_sm = None if excluded else SWAHILI_CLASS_DEFAULT_SM_GUESS.get(subject_class)
            if fallback_sm is not None:
                kit_sm = fallback_sm
                trace.append(f"Noun '{swahili_subject_noun}': no Kitaita translation at all, but its "
                             f"Swahili class '{subject_class}' has a well-evidenced MAJORITY default "
                             f"subject marker '{kit_sm}-' (8+ other {subject_class} nouns confirmed) -- "
                             f"used as an UNCONFIRMED GUESS for the VERB's agreement only; the noun "
                             f"itself stays in the original Swahili, flagged for native-speaker review "
                             f"(see SWAHILI_CLASS_DEFAULT_SM_GUESS in kitaita_verified_corrections.py)")
            else:
                return {"error": f"no reliable Kitaita translation for '{swahili_subject_noun}'", "trace": trace}
        elif kit_subject in VERIFIED_SUBJECT_MARKERS:
            kit_sm = VERIFIED_SUBJECT_MARKERS[kit_subject]
            trace.append(f"Subject marker for '{kit_subject}' -> native-speaker-VERIFIED '{kit_sm}-' "
                         f"(overrides the statistical class guess; see kitaita_verified_corrections.py)")
            if kit_subject in NUMBER_INVARIANT_NOUNS:
                trace.append(f"'{kit_subject}' is NUMBER-INVARIANT (its Swahili source word doesn't "
                             f"distinguish singular/plural) -- '{kit_sm}-' is the PLURAL-reading "
                             f"default; the singular alternative is '{NUMBER_INVARIANT_SM_SINGULAR['default']}-' "
                             f"(see NUMBER_INVARIANT_NOUNS in kitaita_verified_corrections.py)")
        elif kit_subject in DEVERBAL_SM_HYPOTHESIS:
            kit_sm = DEVERBAL_SM_HYPOTHESIS[kit_subject]
            trace.append(f"Subject marker for '{kit_subject}' -> '{kit_sm}-' -- UNCONFIRMED HYPOTHESIS, "
                         f"NOT native-speaker-verified and NOT even this project's own corpus induction "
                         f"(borrowed from a SIBLING research track's own from-scratch class-15 candidate; "
                         f"see DEVERBAL_SM_HYPOTHESIS in kitaita_verified_corrections.py). This is a "
                         f"class-15-style infinitive-as-noun subject, never tested before.")
        else:
            class_info = noun_class_of.get(kit_subject)
            if class_info:
                kit_sm = class_info["subject_marker"]
                trace.append(f"'{kit_subject}' is in corrected class \"{class_info['class']}-\", "
                             f"own dominant subject marker '{kit_sm}-'")
            else:
                kit_sm = None
                trace.append(f"'{kit_subject}' has no corrected-class subject-marker data -- FLAGGED, not guessed")

    # 2026-10-13 (cont. 2): `kit_class` is the noun's REAL class as stored
    # (may be "i_nasal", a genuine class distinct from "i" -- see TAM_SM_
    # ALIAS in kitaita_verified_corrections.py, direct user instruction to
    # model this the way Swahili treats class 4 vs 9/10: one subject
    # marker, two different adjective paradigms). `kit_sm` from here on is
    # normalized through TAM_SM_ALIAS ONCE, before any TAM-conditioned
    # vowel-shift logic below runs, so every existing shift rule keeps
    # working unmodified on "i_nasal" nouns (they share "i"'s whole TAM
    # paradigm by construction) -- only the ADJECTIVE block further down
    # uses `kit_class` (the un-normalized value) instead of `kit_sm`.
    kit_class = kit_sm
    kit_sm = TAM_SM_ALIAS.get(kit_sm, kit_sm) if kit_sm else kit_sm

    # 2026-10-22: direct user correction -- for a personal pronoun
    # subject, the VERB's own prefix uses PERSON agreement (NOT mundu's
    # class, which is for the adjective/concord side only -- kit_class
    # above is already "u" from the pronoun branch and is untouched by
    # this). Re-overrides kit_sm AFTER the kit_class copy on purpose, so
    # the two deliberately diverge for this one case. Routes through
    # this project's EXISTING "bare citation form" person-marker tier
    # (PERSON_SM_DEFAULT, 2026-10-20) rather than inventing a new one --
    # same tier that already gives 1sg="ni", matching the user's own
    # worked example ("Ini nideka mlacha", not "Ini udeka mlacha").
    # "yeye"/"wao" (3rd person, no entry in SWAHILI_PRONOUN_TO_PERSON)
    # are deliberately left unresolved here -- flagged, not guessed.
    if person is None and swahili_subject_noun in SWAHILI_PRONOUN_TO_PERSON:
        pronoun_person = SWAHILI_PRONOUN_TO_PERSON[swahili_subject_noun]
        kit_sm = PERSON_SM_DEFAULT.get(pronoun_person)
        trace.append(f"Subject '{swahili_subject_noun}': VERB subject marker uses PERSON "
                     f"agreement ('{pronoun_person}' -> '{kit_sm}-', PERSON_SM_DEFAULT bare-"
                     f"citation tier), NOT mundu's class -- that applies to the adjective/"
                     f"concord side only (kit_class stays 'u')")

    root_is_full_form = False
    root_untranslated = False  # True iff kit_root is the ORIGINAL SWAHILI root, kept because no
    # Kitaita translation exists at any tier (VERIFIED/root-level/derived/per-form-fallback) --
    # see the final `else` branch below. Distinct from root_is_full_form: that flag means "this
    # root resolved to an already-inflected KITAITA surface form, do not re-inflect it"; this one
    # means "this root is SWAHILI, DO run it through the normal Kitaita SM+TAM substitution below."
    idiom_object = None  # set below (VERIFIED_IDIOMS) when the Swahili root has no SINGLE-WORD
    # Kitaita root at all but IS a confirmed [verb root + fixed object noun] periphrasis, e.g.
    # "tawal" (rule) = "bony" (do) + "nguma" (authority) -- a real, compositional Swahili
    # paraphrase ("tawala" = "fanya mamlaka", BOTH independently confirmed: VERIFIED_ROOTS
    # 'fany'->'bony', VERIFIED_NOUN_TRANSLATIONS 'mamlaka'->'nguma'), not an arbitrary idiom.
    # Tried as the LAST tier before the Taita-fy fallback (2026-10-19) -- direct user request:
    # "before using Taita-fied Swahili words" try the known phrase first. Appended as a separate
    # word after the inflected verb (components['idiom_object'] below), never fused into kit_root
    # -- see VERIFIED_IDIOMS's own comment for why a single fused string would misplace the FV.
    kuwa_past_alt = None  # (root, tam, final_vowel) when the "alternate, no rule yet" path is used
    is_kuwa_future = swahili_verb["root"] == "kuw" and swahili_verb["tam"] == "ta"
    if is_kuwa_future and person is None and kit_sm in FUTURE_KUWA_FULL_ROOT_MARKERS:
        kit_root = "kai"
        trace.append(f"Verb root 'kuw' (be) in the FUTURE tense: subject marker '{kit_sm}-' is one "
                     f"of the classes that does NOT take the bare-'k' reduction below -- 2026-10-08 "
                     f"cont. 12 (batch 10): both tested 'wi'-class nouns (watu, wazee) and both "
                     f"tested 'chi'-class nouns (kikombe, kiti) need the FULLER root 'kai' instead "
                     f"('wichakaia'/'chichakaia', not 'wichaka'/'chichaka') -- 2/2 clean on each "
                     f"class (cont. 14 briefly misread kikombe's own stored judgment as a split here "
                     f"due to a mistyped correction; cont. 15 confirmed directly the class is "
                     f"uniform and fixed the stored text). See FUTURE_KUWA_FULL_ROOT_MARKERS in "
                     f"kitaita_verified_corrections.py.")
    elif is_kuwa_future:
        kit_root = "k"
        trace.append("Verb root 'kuw' (be) in the FUTURE tense: native-speaker-CONFIRMED the root "
                     "collapses to a bare 'k' here, NOT 'kogh' or 'dek' ('Nyumba itakuwa' -> "
                     "'Nyumba ichaka', s53) -- overrides the statistical root-level candidate 'kai'. "
                     "GENERALIZED, confirmed across 7 more nouns with zero exceptions (batch 4, "
                     "2026-10-07 cont. 16: mungu/mti/samaki/siku/mwana/mtu/nchi all took this form; "
                     "nchi's OLD pre-finding answer 'jichakoghe' was specifically re-tested and "
                     "corrected to 'jichaka'). NOT universal, though -- the 'wi'/'chi' classes need "
                     "the fuller root 'kai' instead (see above), and the 'i' and 'ghi' classes each "
                     "genuinely SPLIT per noun (i: nyumba/shimba take bare 'k', miti/nyoka need "
                     "'kai'; ghi: maji takes neither shape cleanly, mayai needs 'kai' -- kept as "
                     "exact per-noun overrides, not generalized).")
    elif swahili_verb["root"] == "kuw" and swahili_adjective:
        kit_root = VERIFIED_KUWA_ROOT_COPULA
        trace.append(f"Verb root 'kuw' (be) -> native-speaker-VERIFIED '{kit_root}' "
                     f"(copula, predicate complement follows -- overrides the statistical "
                     f"root-level candidate 'kai', which was wrong; see kitaita_verified_corrections.py)")
    elif swahili_verb["root"] == "kuw" and swahili_verb["sm"] is None:
        # 2026-10-22: direct native-speaker correction -- the INVARIANT
        # present copula "ni" ("X ni Y") is ALSO just "ni" in Kitaita, kept
        # as a literal loanword/shared form, NOT conjugated at all. `sm is
        # None` is the signal that uniquely identifies the bare word "ni"
        # here (decompose_verb_or_copula() sets subject_marker=None only
        # for "ni" itself -- every OTHER inflected 'kuwa' form, e.g.
        # "alikuwa"/"anakuwa", carries a real subject marker and so does
        # NOT match this branch, leaving the existing dek-copula/bare-
        # existential findings below untouched). Checked AFTER the dek
        # branch above on purpose -- "ni" + an ADJECTIVE complement still
        # takes the dek root (4/4 native-speaker-confirmed, unaffected by
        # this addition); this branch only catches "ni" with NO adjective
        # (a noun/pronoun predicate, or no complement at all).
        kit_root = "ni"
        root_is_full_form = True
        trace.append("Verb 'ni' (invariant present copula, no adjective complement) -> "
                     "native-speaker-CONFIRMED Kitaita ALSO uses 'ni' literally here, unconjugated "
                     "(not the dek/kogh roots, which are for an ADJECTIVE complement or a "
                     "conjugated 'kuwa' form respectively)")
    elif swahili_verb["root"] == "na_have":
        # 2026-10-22: direct native-speaker correction -- Swahili "kuwa
        # na" (to have) is literally the SAME root in Kitaita, "na" --
        # see kitaita_e2e_pipeline.py's NA_HAVE_FORMS/_na_have_verb_
        # form_entry() for the Swahili-side recognition (a 2nd defective
        # verb, same shape as "ni"). Stored here as bare consonant "n"
        # (NOT "na") so the normal SM+TAM+root+FINAL-VOWEL assembly below
        # supplies the "-a" itself, exactly like every other root in this
        # file -- storing "na" would double it ("na"+"a" -> "naa").
        kit_root = "n"
        trace.append("Verb root 'na_have' (kuwa na/to have) -> native-speaker-CONFIRMED Kitaita "
                     "root 'n' (-> '...na' once the final vowel is appended below) -- a shared "
                     "cognate, same as the Swahili root itself")
    elif swahili_verb["root"] == "oko_exist":
        # 2026-10-22: bug #2 fix -- see kitaita_e2e_pipeline.py's
        # EXISTENTIAL_LOCATIVE_FORMS for the reverse-inference evidence
        # ("yuko"/"yupo"/"yumo" -> Kitaita "oko", invariant, n>=15 corpus
        # verses, zero exceptions for 3rd-person singular). Same shape as
        # "ni" above: a literal, unconjugated form, not re-inflected.
        kit_root = "oko"
        root_is_full_form = True
        trace.append("Verb 'yuko'/'yupo'/'yumo' (locative-existential, 'X is at Y') -> "
                     "reverse-inference-CONFIRMED Kitaita 'oko', literal and invariant for "
                     "3rd-person singular (no SM/TAM of its own)")
    elif swahili_verb["root"] == "ko_have":
        # 2026-10-22: vocabulary-gap fix (plural "have") -- see
        # PLURAL_HAVE_FORMS in kitaita_e2e_pipeline.py for the reverse-
        # inference evidence ("wana X" -> "[class SM]+ko (na) X", e.g.
        # "wiko na ndighi"/"weko na sere"). Built directly as [SM]+"ko"
        # here (bypassing the normal TAM/final-vowel assembly, which
        # would wrongly insert the regular '-we-' present marker and
        # '-a' final vowel this construction doesn't take) -- same
        # "already a real surface form, don't re-inflect further"
        # reasoning as every other root_is_full_form case.
        if not kit_sm:
            # 2026-10-22: a REAL subject noun can still resolve to an
            # unknown/unconfirmed class (kit_sm left None by the generic
            # noun-class lookup, e.g. 'mama'/'baba' -- neither has a
            # VERIFIED_SUBJECT_MARKERS or noun_class_of entry) even though
            # the "no subject at all" bypass above didn't fire (a subject
            # WAS found, just not a confidently-classed one) -- found
            # directly: this previously produced literal "Noneko"
            # (f"{None}ko"). Same ANIMATE-PL default as that bypass,
            # applied at the value level instead of the presence level.
            kit_sm = "wi"
            trace.append("Verb 'wana' (they have): subject was found but its class marker is "
                         "unresolved -- defaulting to 'wi-' (wandu) rather than emitting a literal "
                         "'None' into the output")
        kit_root = f"{kit_sm}ko"
        root_is_full_form = True
        trace.append(f"Verb 'wana' (they have) -> reverse-inference-CONFIRMED Kitaita '{kit_sm}-' + "
                     f"'ko' (existential root, same as 'oko' above) -> '{kit_root}', no separate TAM "
                     f"syllable or final vowel (corpus: 'wiko'/'weko', not 'wawekoa')")
    elif swahili_verb["root"] == "kuw":
        kuwa_past_alt = KUWA_PAST_ALTERNATES[0]
        kit_root = kuwa_past_alt[0]
        trace.append(f"Verb root 'kuw' (be), past/bare existential: '{kit_root}' is a STRONG "
                     f"default (7 of 8 bare-existential-tested nouns confirmed -- mungu/mti/siku/"
                     f"mwana/samaki/mtu/nchi -- vs 1 exception, nyumba, which takes "
                     f"'{KUWA_PAST_ALTERNATES[1][0]}' instead). Not a coin-flip; used with real "
                     f"confidence here, with nyumba-like exceptions still possible on an untested "
                     f"noun and flagged as such if one turns up.")
    elif swahili_verb["root"] == "furahi":
        # 2026-10-22: direct user instruction, confirmed by reverse-
        # inference over 620 real corpus tokens (see VERIFIED_ROOTS
        # ["furahi"]'s own comment for the full paradigm) -- this verb
        # has NO active transitive form in the corpus at all; it is
        # ALWAYS passive-shaped ("boi" + this project's own regular
        # passive infix, kitaita_passive_form() -- "boi" ends in a
        # vowel, so "-lw-": "boilw") regardless of Swahili's own voice
        # ("ninafurahia X" is Swahili ACTIVE, but renders as Kitaita
        # PASSIVE). Checked BEFORE the generic VERIFIED_ROOTS branch so
        # the bare active root never gets used un-passivized. The
        # object/stimulus becomes a "ni"-agent phrase, not a direct
        # object -- see the object-append step below, gated on this
        # same root check.
        kit_root = kitaita_passive_form(VERIFIED_ROOTS["furahi"])
        trace.append(f"Verb root 'furahi' (enjoy/rejoice) -> 'boi' + passive infix = '{kit_root}' -- "
                     f"this verb is INHERENTLY PASSIVE in Kitaita (confirmed directly: 620 corpus "
                     f"tokens, zero active-transitive examples found) -- see kitaita_verified_"
                     f"corrections.py VERIFIED_ROOTS['furahi'] for the full paradigm")
    elif swahili_verb["root"] == "kasirik":
        # 2026-10-22: direct user report ("kukasirika... we are only
        # using udejokwa instead of the full phrase") -- "kasirika" (get
        # angry, intransitive) is the PASSIVE counterpart of "kasirisha"
        # (VERIFIED_IDIOMS["kasirish"]); same "jok"/"jos" root
        # alternation, same fixed "machu" (anger) object -- see
        # VERIFIED_ROOTS["kasirik"]'s own comment for the n=8 corpus
        # evidence. Checked BEFORE the generic VERIFIED_ROOTS branch so
        # the bare root never surfaces un-passivized, same reasoning as
        # 'furahi' above. The fixed agent phrase is attached via
        # `idiom_object` below (this verb is intransitive -- no real
        # Swahili object ever competes for that slot), not the obj_part
        # mutation 'furahi' uses (that path only runs when a real object
        # noun exists).
        kit_root = kitaita_passive_form(VERIFIED_ROOTS["kasirik"])
        idiom_object = "ni machu"
        trace.append(f"Verb root 'kasirik' (get angry) -> 'jok' + passive infix = '{kit_root}' -- "
                     f"PASSIVE counterpart of the causative 'kasirisha'/'jos', fixed agent phrase "
                     f"'ni machu' (anger) -- reverse-inference-CONFIRMED, n=8 corpus verses, zero "
                     f"exceptions (see VERIFIED_ROOTS['kasirik'])")
    elif swahili_verb["root"] in VERIFIED_ROOTS:
        kit_root = VERIFIED_ROOTS[swahili_verb["root"]]
        trace.append(f"Verb root '{swahili_verb['root']}' -> native-speaker-VERIFIED '{kit_root}' "
                     f"(overrides the statistical candidate; see kitaita_verified_corrections.py)")
    elif (idiom := VERIFIED_IDIOMS.get(swahili_verb["root"])) is not None:
        # 2026-10-20: checked BEFORE root_kitaita/derived/fallback below on purpose --
        # this is a HAND-CONFIRMED answer (same trust tier as VERIFIED_ROOTS just
        # above), and for 'tawal' specifically, root_kitaita_derived ALREADY has its
        # own statistical candidate ('bony', found independently via 2 fallback-form
        # alignments -- a reassuring cross-confirmation of the SAME root, checked
        # directly before writing this) that would otherwise fire first and silently
        # drop the fixed object 'nguma' this periphrasis needs -- "Mfalme anatawala"
        # would become the incomplete "Mzuri anabonya" (he DOES, not "he RULES") if
        # the statistical tier were allowed to win here.
        kit_root, idiom_object = idiom["root"], idiom["object"]
        trace.append(f"Verb root '{swahili_verb['root']}' has NO single-word Kitaita root, but IS a "
                     f"confirmed PERIPHRASIS: '{kit_root}' (do/make) + fixed object '{idiom_object}' "
                     f"-- {idiom['note']}. Runs through normal subject-marker/TAM substitution below "
                     f"like any other verb root; '{idiom_object}' is appended afterward, unchanged.")
    elif (passive_base := next((b for b in swahili_passive_base_candidates(swahili_verb["root"])
                                 if b in VERIFIED_ROOTS), None)) is not None:
        # 2026-10-20, direct user request ("get the various verb forms ...
        # so we don't have to duplicate effort for every single root") --
        # GENERALIZED passive rule, checked before the statistical tiers
        # below (same reasoning as VERIFIED_IDIOMS just above: a confirmed
        # GRAMMATICAL RULE outranks a raw per-root Dice statistic). See
        # kitaita_verified_corrections.py's "verb EXTENSIONS" section for
        # the full evidence (2 independent roots, pend->kund/kundwa and
        # ambi->ghore/ghorelwa, zero exceptions).
        kit_root = kitaita_passive_form(VERIFIED_ROOTS[passive_base])
        trace.append(f"Verb root '{swahili_verb['root']}': PASSIVE of verified root '{passive_base}' "
                     f"(-> '{VERIFIED_ROOTS[passive_base]}') -- GENERALIZED passive rule applied: "
                     f"'{kit_root}' (see kitaita_verified_corrections.py's passive-extension evidence, "
                     f"NOT itself a separately-verified whole form)")
    elif (root_cand := root_kitaita.get(swahili_verb["root"])) is not None:
        kit_root = root_cand["kitaita_root"]
        trace.append(f"Verb root '{swahili_verb['root']}' -> Kitaita root '{kit_root}' "
                     f"(root-level Dice {root_cand['score']:.2f}, {root_cand['cooccur_count']} shared verses, "
                     f"{root_cand['n_competitors']} competitors)")
    elif (derived := root_kitaita_derived.get(swahili_verb["root"])) is not None:
        kit_root = derived["kitaita_root"]
        (s1, w1, d1), (s2, w2, d2) = derived["support"]
        trace.append(f"Verb root '{swahili_verb['root']}': root-level alignment found NO candidate, "
                     f"but DERIVED root '{kit_root}' from 2 independently-aligned fallback forms "
                     f"agreeing on the shared material '-{derived['suffix']}' ('{s1}'->'{w1}' Dice "
                     f"{d1:.2f}, '{s2}'->'{w2}' Dice {d2:.2f}) -- NOT native-speaker-verified, lower "
                     f"confidence than a root-level Dice match, but runs through normal subject-"
                     f"marker/TAM substitution below rather than reusing either whole form as-is")
    else:
        fb = root_kitaita_fallback.get(swahili_verb["root"])
        if fb is None:
            # 2026-10-19, direct user request: previously returned a hard error here,
            # which forced the caller (kitaita_e2e_pipeline.assemble_in_order) all the
            # way back to the ORIGINAL, UNINFLECTED SWAHILI SURFACE FORM (e.g. a bare
            # "anapeleka" sitting untouched inside an otherwise-Kitaita sentence) --
            # confirmed directly as the single worst-sounding case: Swahili SM+Swahili
            # TAM+Swahili root, no Kitaita morphology anywhere in the word at all.
            # Keeping the Swahili ROOT but running it through the SAME Kitaita SM+TAM
            # substitution every other verb below already gets ("wawepeleka" =
            # Kitaita 'wa-' + Kitaita present '-we-' + Swahili root 'peleka') is a
            # strictly smaller, more defensible guess: it reuses this project's own
            # well-evidenced SM/TAM tables (the actual object of this whole pipeline),
            # the ONLY thing left unresolved is the lexical root itself -- which is
            # exactly the gap the Kitaita Word Bank's verb-roots tab already exists to
            # close, and this is now flagged (components['root_untranslated']) for
            # root coverage/testing to single out, not silently presented as a real
            # translation. root_is_full_form stays False on purpose: that flag means
            # "already a real Kitaita surface form, don't re-inflect"; this is the
            # opposite -- a bare Swahili STEM that SHOULD run through the normal
            # SM+TAM dispatch below.
            kit_root = swahili_verb["root"]
            root_untranslated = True
            trace.append(f"Verb root '{swahili_verb['root']}': NO Kitaita candidate at any tier "
                         f"(not VERIFIED, no root-level or derived or per-form-fallback match) -- "
                         f"NOT a Kitaita translation. Kept the ORIGINAL SWAHILI ROOT and will run it "
                         f"through the normal Kitaita subject-marker/TAM substitution below, "
                         f"'Taita-fying' only the grammar around it (FLAGGED as "
                         f"root_untranslated -- this is a vocabulary gap, not a translation; "
                         f"native-speaker/Word-Bank coverage is the real fix, this is a readability "
                         f"stopgap for testing, not a substitute for it)")
        else:
            # fb['kitaita_word'] is a full inflected surface form (e.g. 'wichawona'); strip the
            # known SM it was built with to recover a bare root is not reliable here, so this
            # fallback path is reported but FLAGGED as the weaker of the two methods, with the
            # reason spelled out (a very frequent competing root can statistically outrank the
            # correct one -- checked directly for 'ona'/'won': Kitaita root 'won' co-occurs with
            # Swahili 'kuw' (be, 32 verses) MORE than with 'on' (see, 18 verses) simply because
            # 'kuw' is near-ubiquitous, so root-level alignment picks the wrong winner here).
            kit_root = fb["kitaita_word"]
            root_is_full_form = True
            trace.append(f"Verb root '{swahili_verb['root']}': root-level alignment found NO candidate "
                         f"(a very frequent unrelated root likely won the comparison by sheer frequency -- "
                         f"FLAGGED, not silently trusted). Falling back to the weaker per-SURFACE-FORM "
                         f"method: '{kit_root}' (aligned to '{fb['best_swahili_word']}', Dice "
                         f"{fb['best_dice']:.2f}, {fb['n_competitors']} competitors) -- this is already a "
                         f"full inflected form, not just a root, so it is used AS-IS below rather than "
                         f"re-inflected.")

    is_kuwa_pastlike = swahili_verb["root"] == "kuw" and not swahili_adjective and not is_kuwa_future
    final_vowel = swahili_verb["fv"]

    kit_tam = None
    if negated:
        # 2026-10-15: negation has its own, completely separate paradigm
        # (native-speaker-submitted, see NEGATIVE_* in kitaita_verified_
        # corrections.py) -- none of the affirmative TAM-dispatch branches
        # below apply or are even consulted; kit_sm/kit_tam/final_vowel
        # computed here are simply unused (the real assembly happens at the
        # final kit_verb step, after this chain, using NEGATIVE_* instead).
        # Skipped entirely rather than falling through to "No mapping rule"
        # so the trace doesn't show a spurious contradiction.
        pass
    elif root_is_full_form:
        trace.append(f"Using the fallback form '{kit_root}' as-is -- subject marker and TAM "
                     f"substitution SKIPPED (it is already a real inflected word; re-inflecting it "
                     f"would require knowing its own internal morpheme boundaries, which this "
                     f"fallback path does not determine)")
    elif is_kuwa_future:
        kit_tam = "cha"
        final_vowel = "a"
        trace.append("Swahili future (-ta-) + 'kuwa' -> Kitaita TAM '-cha-' + bare root 'k' + "
                     "final vowel 'a' (native-speaker-confirmed, s53: 'ichaka' -- NOT the 'e' "
                     "the 'kogh'/'dek' past roots take)")
    elif swahili_verb["root"] == "kuw" and swahili_adjective:
        # Copula 'dek': ZERO TAM always, no exceptions found (odeka/edeka/redeka/jideka/
        # ghodeka/kodeka all lack a separate TAM syllable) -- must NOT fall through to the
        # generic statistical past-TAM lookup below (that lookup was built for the wrong,
        # discarded 'kai' root and returns None for several subject markers, e.g. 'ji',
        # which previously surfaced as an unresolved "?" in the output instead of zero).
        kit_tam = ""
        trace.append("'kuw' (be), copula ('dek' root): ZERO TAM always (odeka/edeka/redeka/"
                     "jideka/ghodeka/kodeka all confirmed with no separate TAM syllable) -- "
                     "does NOT use the generic statistical past-TAM lookup, which was built "
                     "for a different, discarded root")
    elif is_kuwa_pastlike:
        # kuwa_past_alt = (root, tam, final_vowel), chosen above with no confirmed rule --
        # see KUWA_PAST_ALTERNATES. TAM and final vowel are COUPLED to which root was picked,
        # not independent (every real 'kogh' answer takes '-re-'+'e'; every real 'dek' answer
        # takes zero-TAM+'a').
        kit_tam = kuwa_past_alt[1]
        final_vowel = kuwa_past_alt[2]
        if kit_tam == "re" and kit_sm and (kit_sm.endswith("u") or kit_sm in KOGH_RE_FRONT_VOWEL_SHIFTS):
            shifted = kogh_re_o_shift(kit_sm)
            if kit_sm.endswith("u"):
                trace.append(f"Subject marker '{kit_sm}-' + existential 'kogh' + TAM '-re-': any "
                             f"marker ending in 'u' shifts its final vowel to 'o' here (confirmed on "
                             f"2 independent markers -- bare 'u': orekoghe 1455x vs urekoghe 0x in "
                             f"the corpus; 'ghu': msalaba s71 -- CONFLICTS with one earlier human "
                             f"correction for mwana that used 'u-' unshifted, flagged not silently "
                             f"resolved) -> '{shifted}-'")
            else:
                trace.append(f"Subject marker '{kit_sm}-' + existential 'kogh' + TAM '-re-': the "
                             f"front-vowel mirror of the u->o shift above -- this specific marker is "
                             f"confirmed to shift (NOT a general 'ends in i' rule -- 'chi' stays "
                             f"unshifted; see KOGH_RE_FRONT_VOWEL_SHIFTS in kitaita_verified_"
                             f"corrections.py for the per-marker evidence) -> '{shifted}-'")
            kit_sm = shifted
        trace.append(f"'kuw' (be), past/bare existential -> TAM '-{kit_tam or '(zero)'}-' + "
                     f"final vowel '-{final_vowel}' (coupled to the root choice above, not "
                     f"independently verified for this specific noun)")
    elif swahili_verb["root"] == "furahi" and swahili_verb["tam"] == "na":
        # 2026-10-22: PRESENT tense only -- other tenses for this root
        # (ka/cha/li/me) matched the generic dispatch below CLEANLY in
        # the corpus (see VERIFIED_ROOTS['furahi']'s paradigm comment),
        # so only present is special-cased here. The generic '-we-'
        # present marker this project uses for every OTHER verb has
        # ZERO corpus attestation for "furahi" -- real present/
        # habitual-sense examples instead use the stative/perfect '-da-'
        # marker (PSA.16.9 "nadaboilwa"<->"moyo wangu unafurahi", PSA.
        # 68.3 "wadaboilwa"<->"wenye haki hufurahi") -- a real aspect
        # blur for this stative/emotion verb, not an error.
        kit_tam = "da"
        trace.append("Verb 'furahi' (enjoy), PRESENT tense -> Kitaita stative/perfect marker "
                     "'-da-' (NOT the generic '-we-' present marker, which has ZERO corpus "
                     "attestation for this verb -- see VERIFIED_ROOTS['furahi'] for the citations)")
    elif swahili_verb["root"] == "kasirik" and swahili_verb["tam"] == "na":
        # 2026-10-22: same stative/perfect PRESENT override as 'furahi'
        # above, applied by analogy (both are passive-shaped emotion
        # verbs) but ALSO independently corpus-attested here, if thin:
        # "nadajokwa"/"wadajokwa" (n=2) use '-da-', not the generic
        # '-we-' present marker ('najokwa'/'ngajokwa' also occur, but
        # without an aligned Swahili source to confirm which Swahili
        # tense they answer -- not used to override this).
        kit_tam = "da"
        trace.append("Verb 'kasirik' (get angry), PRESENT tense -> Kitaita stative/perfect marker "
                     "'-da-' (thin but real corpus support, n=2: 'nadajokwa'/'wadajokwa', by analogy "
                     "with 'furahi' above -- same passive-shaped emotion-verb pattern)")
    else:
        tam_label = TAM_SWAHILI_TO_LABEL.get(swahili_verb["tam"])
        if tam_label == "past" and kit_sm:
            pv = past_variant_for_sm(kit_sm, allomorphy)
            if pv["variant"]:
                kit_tam = pv["variant"]
                if pv["support"] is None:
                    trace.append(f"Swahili past (-li-) + subject marker '{kit_sm}-' -> Kitaita "
                                 f"'-{kit_tam}-' (native-speaker-VERIFIED override, n=1 -- the "
                                 f"statistical allomorphy table has no data at all for this marker; "
                                 f"see VERIFIED_PAST_TAM_BY_SM in kitaita_verified_corrections.py)")
                else:
                    trace.append(f"Swahili past (-li-) + subject marker '{kit_sm}-' -> Kitaita '-{kit_tam}-' "
                                 f"(support {pv['support']} occurrences{'' if pv['confident'] else ', BELOW n=8 -- weak, flagged'})")
            else:
                trace.append(f"Swahili past (-li-): no re/de evidence at all for subject marker "
                             f"'{kit_sm}-' -- FLAGGED")
        elif swahili_verb["tam"] == "ka":
            kit_tam = "ka"
            trace.append("Swahili narrative/consecutive -ka- -> Kitaita -ka- (attested identical retained form)")
        elif tam_label == "future":
            kit_tam = "cha"
            trace.append("Swahili future -ta- -> Kitaita -cha- (native-speaker-corrected table)")
        elif tam_label == "perfect":
            # 2026-10-20, direct user question ("doesn't -me- have a Kitaita
            # equivalent?") -- it does, and this project already had it on
            # file (KITAITA_TAM_VARIANTS['perfect'] = ['da'], kitaita_tam_
            # allomorphy.py, native-speaker-corrected same session as past
            # re/de and conditional ki/ka) -- it had simply never been wired
            # into THIS function's own TAM dispatch, so every perfect-tense
            # sentence ("amefanya"/"umesongeza"/...) fell through to the
            # generic "No mapping rule" flag and left a literal '?' in the
            # output, same class of gap as the tenseless-verb '?' fixed
            # earlier today. No vowel-shift or person/class split is
            # recorded for this TAM anywhere in this project's data, unlike
            # present/past -- used as a flat constant until evidence says
            # otherwise.
            kit_tam = "da"
            trace.append("Swahili perfect -me- -> Kitaita -da- (native-speaker-corrected table, "
                         "kitaita_tam_allomorphy.KITAITA_TAM_VARIANTS -- previously recorded but never "
                         "wired into this function)")
        elif tam_label == "present":
            kit_tam = PRESENT_TAM_DEFAULT
            trace.append("Swahili present -na- -> Kitaita zero-marked present (corpus-dominant "
                         "form, confirmed directly against the full Bible corpus: 580+ tokens "
                         "across 3 classes vs ~0 for the '-we-' variant the speaker's own hand "
                         "corrections happened to use -- see kitaita_verified_corrections.py)")
            if person is not None:
                # 2026-10-08 cont. 4 direct correction (s82, "Include TAM marker"): person
                # present tense ALSO takes the explicit '-we-' default, same as nouns -- an
                # earlier version of this code zero-marked person present instead, based on
                # a single thin corpus hit; that guess was wrong, native-speaker-verified. The
                # subject marker itself is already fully resolved (e.g. 'na'/'kwa') from
                # PERSON_SUBJECT_MARKERS_BY_TAM above -- it is NOT a bare noun-class consonant,
                # so it must NOT be run through PRESENT_SM_VOWEL_CONFIRMED (that table's keys
                # are noun-class markers like 'u'/'gh'; feeding it 'na' or 'kwa' would just
                # miss and wrongly flag "no confirmed vowel-alternation data").
                pass
            elif kit_sm in PRESENT_SM_VOWEL_CONFIRMED:
                new_sm = PRESENT_SM_VOWEL_CONFIRMED[kit_sm]
                trace.append(f"Subject marker '{kit_sm}-' before present tense -> '{new_sm}-' "
                             f"(consonant stays fixed, vowel becomes 'a' -- corpus-confirmed, "
                             f"see kitaita_verified_corrections.py for the token counts)")
                kit_sm = new_sm
            elif kit_sm:
                trace.append(f"Subject marker '{kit_sm}-' before present tense: no confirmed "
                             f"vowel-alternation data for this class (neither speaker-corrected nor "
                             f"corpus-attested) -- FLAGGED, not guessed; see batch 3")
        elif tam_label == "conditional":
            # 2026-10-09 (cont. 13): Swahili '-ki-' (conditional/participial/
            # converbal) previously had NO branch here at all -- fell through
            # to the generic "no mapping rule" flag below and emitted a
            # literal "?" placeholder TAM syllable, even once the probe-based
            # pipeline correctly identified WHO the subject was (a separate,
            # earlier fix, cont. 12-13). Uses CONDITIONAL_TAM_HYPOTHESIS (see
            # kitaita_verified_corrections.py for the full reasoning) --
            # EXPLICITLY a hypothesis, not a native-speaker-confirmed mapping
            # for this specific Swahili source TAM, unlike every other branch
            # above.
            kit_tam = CONDITIONAL_TAM_HYPOTHESIS
            trace.append(f"Swahili conditional/participial -ki- -> Kitaita -{kit_tam}- "
                         f"(HYPOTHESIS, NOT native-speaker-confirmed for this mapping: Kitaita's own "
                         f"native-speaker-verified 'conditional' TAM category has two attested variants, "
                         f"'ki'/'ka' [kitaita_tam_allomorphy.py], with an unresolved standing hypothesis "
                         f"that they split the same way Swahili's own -ka-[narrative]/-ki-[conditional] "
                         f"does rather than being free variation; 'ki' chosen here to parallel Swahili's "
                         f"own -ki- rather than reusing 'ka', which this generator already maps from "
                         f"Swahili's SEPARATE narrative -ka-. See CONDITIONAL_TAM_HYPOTHESIS in "
                         f"kitaita_verified_corrections.py.)")
        elif swahili_verb["tam"] == "":
            # 2026-10-20: found while fixing decompose_verb()'s new subjunctive/
            # suffixed-relative branches ("uipate"/"aletaye") -- but NOT new
            # itself: the PRE-EXISTING bare-infinitive case (sm=='ku', tam='')
            # had this SAME gap already whenever it was used as a FINITE verb
            # with a real subject (confirmed directly: "mwalimu" + "kuisaidia"
            # -> "Mwalimu u?isaidia.", the same literal '?' this fixes, before
            # today's subjunctive/relative work ever touched this function) --
            # every one of this function's other TAM branches sets kit_tam
            # explicitly; a genuinely TENSELESS Swahili form (infinitive used
            # finitely, subjunctive, or the suffixed "general relative") had no
            # branch here at all and fell through to the generic "no mapping
            # rule" flag, leaving kit_tam=None -> the literal placeholder '?'.
            # Zero-marks the Kitaita TAM slot too -- a GUESS, not a confirmed
            # answer (Kitaita's OWN zero-marked present, PRESENT_TAM_DEFAULT, is
            # a DIFFERENT, separately-confirmed category; this is deliberately
            # NOT assumed to be the same thing, just the least-wrong default
            # available when nothing else is known).
            kit_tam = ""
            trace.append("This Swahili verb form has no separate tense/aspect marker at all "
                         "(infinitive-used-finitely, subjunctive, or the suffixed general-relative "
                         "construction) -- zero-marked the Kitaita TAM slot too as the least-wrong "
                         "default (UNCONFIRMED guess, NOT the same as Kitaita's own separately-"
                         "verified zero-marked PRESENT tense) instead of leaving a literal '?' "
                         "placeholder in the output.")
        else:
            trace.append(f"No mapping rule for Swahili TAM '{swahili_verb['tam']}' -- FLAGGED")

    if not root_is_full_form:
        # Standard Bantu glide alternation, confirmed directly against real corpus
        # attestation rather than assumed: Kitaita class-"w" subject marker surfaces
        # as 'u-' before a CONSONANT-initial TAM (e.g. real text "Mlungu ukaghamba",
        # never "wkaghamba") and stays 'w-' before a vowel-initial one. Applies
        # regardless of which branch above set kit_sm/kit_tam (future/past-alternate/
        # copula/general) -- moved out of the general-TAM branch alone after it was
        # found to silently skip the 'kuw'-root branches (cont. 16: "Mudi ghchaka"
        # missing its linking vowel entirely because this check never ran for them).
        if kit_sm == "w" and kit_tam and kit_tam[0] not in "aeiou":
            trace.append("Subject marker 'w-' + consonant-initial TAM -> surfaces as 'u-' "
                         "(glide alternation, confirmed against real attested text: 'Mlungu "
                         "ukaghamba', never 'wkaghamba')")
            kit_sm = "u"

        # The full-Bible subject-marker induction often returns a BARE
        # CONSONANT (e.g. 'k', 'gh') rather than a consonant+vowel form,
        # which cannot concatenate directly onto a consonant-initial TAM
        # without an illegal consonant cluster ('k'+'ka'='kka'). This is
        # a genuine, UNSOLVED morphophonology gap (which vowel actually
        # links them is not in the data this project has), not guessed
        # away silently: a default linking vowel is inserted and the
        # whole word is flagged as a GUESS specifically needing native-
        # speaker correction, which is exactly what should drive the
        # next table update.
        # The next sound after the subject marker is the TAM if there is one,
        # otherwise the root directly (a ZERO TAM, e.g. every copula 'dek' answer) --
        # checking only kit_tam missed the zero-TAM case entirely (cont. 18: a bare
        # consonant marker 'j' on a brand-new noun, 'indi', directly abutting the
        # consonant-initial root 'dek' produced the illegal cluster "jdeka").
        next_sound = kit_tam if kit_tam else kit_root
        if person is not None and kit_sm and kit_sm[-1] not in "aeiou" and next_sound and next_sound[0] not in "aeiou":
            trace.append(f"Subject marker '{kit_sm}-' is a bare consonant before a consonant-initial "
                         f"'-{next_sound}-', but this is a PERSON marker with its own direct corpus "
                         f"attestation of exactly this combination (e.g. 'mcha-', 2pl future) -- "
                         f"NOT run through the generic noun-class linking-vowel guess, which would "
                         f"wrongly insert an unattested vowel here")
        elif kit_sm and kit_sm[-1] not in "aeiou" and next_sound and next_sound[0] not in "aeiou":
            trace.append(f"Subject marker '{kit_sm}-' is a bare consonant and the next sound "
                         f"('-{next_sound}-') is consonant-initial -- concatenating them directly "
                         f"would give an unattested consonant cluster. Inserted a default "
                         f"linking vowel 'u' as a GUESS (flagged, not confirmed) -- this is "
                         f"exactly the kind of gap a native-speaker correction should resolve.")
            kit_sm = kit_sm + "u"

    if negated and not root_is_full_form:
        # 2026-10-21: extended from present-only (2026-10-15) to all 4
        # tenses -- see NEGATIVE_PERSON_PREFIX_BY_TENSE/NEGATIVE_CLASS_*
        # in kitaita_verified_corrections.py for the full evidence and
        # this function's own docstring (negated_tense param) for why
        # there is no safe cross-tense default.
        neg_tense = negated_tense or "present"
        # 2026-10-22: "na_have" (kuwa na/to have) is IRREGULAR here too --
        # it keeps its OWN final vowel '-a' even when negated ("hana",
        # not "hane"), unlike every regular negative verb this project
        # has tested (which all take '-e', NEGATIVE_PERSON_FINAL_VOWEL).
        # Mirrors the exact SAME irregularity on the Swahili side ("hana"
        # ends in '-a', not the regular negative-present '-i').
        neg_person_fv = "a" if swahili_verb["root"] == "na_have" else NEGATIVE_PERSON_FINAL_VOWEL
        if person is not None:
            prefix = NEGATIVE_PERSON_PREFIX_BY_TENSE.get((person, neg_tense))
            if prefix is None:
                kit_verb = f"?{kit_root}{neg_person_fv}"
                trace.append(f"NEGATIVE {neg_tense}: no negative-marker data for person '{person}' "
                             f"at this tense -- FLAGGED, not guessed")
            else:
                kit_verb = f"{prefix}{kit_root}{neg_person_fv}"
                trace.append(f"NEGATIVE {neg_tense}, person '{person}': prefix '{prefix}-' + root + "
                             f"'-{neg_person_fv}' -> '{kit_verb}' (native-speaker-"
                             f"submitted, Kitaita Word Bank 2026-10-21, n=1 per cell)")
        elif (animate_prefix := NEGATIVE_ANIMATE_CLASS_PREFIX_BY_TENSE.get((kit_class, neg_tense))) is not None:
            # 2026-10-21: an explicit ANIMATE noun subject (classes 'u'/'wi',
            # e.g. mwalimu/watu) follows the PERSON template, NOT the
            # ordinary-noun-class one below -- see NEGATIVE_ANIMATE_CLASS_
            # PREFIX_BY_TENSE's own comment for the evidence. Checked BEFORE
            # the ordinary class dispatch on purpose.
            kit_verb = f"{animate_prefix}{kit_root}{neg_person_fv}"
            trace.append(f"NEGATIVE {neg_tense}, ANIMATE noun class '{kit_class}': prefix "
                         f"'{animate_prefix}-' + root + '-{neg_person_fv}' -> "
                         f"'{kit_verb}' (follows the PERSON template, not the ordinary noun-class "
                         f"one -- native-speaker-submitted, Kitaita Word Bank 2026-10-21, n=1 per cell)")
        elif neg_tense in ("present", "subjunctive_imperative"):
            # 2026-10-21: checked directly -- a noun-CLASS subject's negative
            # subjunctive/imperative is NOT marked separately from its negative
            # present (unlike a PERSON subject, which gets a completely
            # different "-se-"-infix template for this tense); same template
            # either way.
            neg_sm = NEGATIVE_SM_BY_CLASS_OVERRIDE.get(kit_class, kit_sm)
            if neg_sm and neg_sm[-1] not in "aeiou" and kit_root and kit_root[0] not in "aeiou":
                neg_sm = neg_sm + "u"
            if neg_sm is None:
                kit_verb = f"?{kit_root}{NEGATIVE_FINAL_VOWEL}"
                trace.append(f"NEGATIVE {neg_tense}: no negative subject-marker data for this "
                             f"subject -- FLAGGED, not guessed")
            else:
                kit_verb = f"{NEGATIVE_MARKER_KITAITA}{neg_sm}{kit_root}{NEGATIVE_FINAL_VOWEL}"
                trace.append(f"NEGATIVE {neg_tense}: '{NEGATIVE_MARKER_KITAITA}-' + '{neg_sm}-' "
                             f"(bare, unshifted class marker) + root + '-{NEGATIVE_FINAL_VOWEL}' -> "
                             f"'{kit_verb}' (native-speaker-submitted, Kitaita Word Bank 2026-10-{'15' if neg_tense == 'present' else '21'}, n=1 per cell)")
        elif neg_tense == "future":
            # 2026-10-21: class-level negative future RETAINS the affirmative
            # future TAM "-cha-" and keeps FV "-a-" (NOT "-e-" -- checked
            # directly, every other negative form in this file shifts to "-e-",
            # this one alone does not).
            neg_sm = NEGATIVE_SM_BY_CLASS_OVERRIDE.get(kit_class, kit_sm)
            if neg_sm is None:
                kit_verb = f"?cha{kit_root}{NEGATIVE_CLASS_FUTURE_FINAL_VOWEL}"
                trace.append("NEGATIVE future: no negative subject-marker data for this subject -- "
                             "FLAGGED, not guessed")
            else:
                kit_verb = f"{NEGATIVE_MARKER_KITAITA}{neg_sm}cha{kit_root}{NEGATIVE_CLASS_FUTURE_FINAL_VOWEL}"
                trace.append(f"NEGATIVE future: '{NEGATIVE_MARKER_KITAITA}-' + '{neg_sm}-' + '-cha-' "
                             f"(affirmative future TAM retained) + root + "
                             f"'-{NEGATIVE_CLASS_FUTURE_FINAL_VOWEL}' -> '{kit_verb}' (native-speaker-"
                             f"submitted, Kitaita Word Bank 2026-10-21, n=1 per cell)")
        else:  # past
            # 2026-10-21: a DEDICATED negative-past TAM ("-re-...-eghe"), NOT
            # each class's own affirmative past-TAM allomorph (VERIFIED_PAST_
            # TAM_BY_SM) -- checked directly, "ji" is "de" affirmatively but
            # "-re-" negatively. 'va' is a flagged exception with a different
            # shape entirely (see NEGATIVE_CLASS_PAST_VA_OVERRIDE's own comment).
            if kit_class == "va":
                sm_form, suffix = (NEGATIVE_CLASS_PAST_VA_OVERRIDE["sm_form"],
                                    NEGATIVE_CLASS_PAST_VA_OVERRIDE["suffix"])
                note = "FLAGGED EXCEPTION (own shape, not the 9-class '-re-...-eghe' pattern)"
            else:
                sm_form = NEGATIVE_CLASS_PAST_SM_FORM.get(kit_class)
                suffix = NEGATIVE_CLASS_PAST_SUFFIX
                note = "dedicated negative-past TAM '-re-...-eghe', independent of the affirmative one"
            if sm_form is None:
                kit_verb = f"?re{kit_root}{NEGATIVE_CLASS_PAST_SUFFIX}"
                trace.append(f"NEGATIVE past: no negative past-tense data for class '{kit_class}' -- "
                             f"FLAGGED, not guessed")
            else:
                kit_verb = f"{NEGATIVE_MARKER_KITAITA}{sm_form}{kit_root}{suffix}"
                trace.append(f"NEGATIVE past: '{NEGATIVE_MARKER_KITAITA}-' + '{sm_form}-' ({note}) + "
                             f"root + '-{suffix}' -> '{kit_verb}' (native-speaker-submitted, Kitaita "
                             f"Word Bank 2026-10-21, n=1 per cell)")
    elif root_is_full_form:
        kit_verb = kit_root
        if negated:
            trace.append("NEGATIVE present requested, but the verb root resolved to an opaque FULL "
                         "FALLBACK FORM (no known morpheme boundary) -- cannot safely apply the negative "
                         "template to it, FLAGGED rather than guessed; kept as the (wrong, affirmative) "
                         "fallback form.")
    else:
        kit_verb = f"{kit_sm or '?'}{kit_tam if kit_tam is not None else '?'}{kit_root}{final_vowel}"
    idiom_object_trailing = None  # set below when idiom_object must be placed AFTER a real object noun
    if idiom_object and not swahili_object_noun:
        # 2026-10-20: the periphrasis's fixed object noun -- appended AFTER the fully
        # inflected verb, unchanged/never re-inflected (VERIFIED_IDIOMS's own comment),
        # same position Swahili's own "tawala kwa mamlaka" puts it in. Only correct when
        # there is NO separate real object noun competing for that slot (true for every
        # VERIFIED_IDIOMS entry until "kasirish" below).
        kit_verb = f"{kit_verb} {idiom_object}"
    elif idiom_object:
        # 2026-10-21: direct user correction/example ("kasirisha" = "josh" + a REAL
        # object noun [the one made angry] + the fixed anger-object "machu") --
        # the fixed idiom object must come AFTER the real object noun's own token
        # position, not fused onto the verb (which would wrongly print it BEFORE the
        # real object: e.g. Swahili source token order has "mzazi" right after
        # "alimkasirisha", so marker-only/never-rearrange placement puts the real
        # object there -- the idiom's own fixed word has no original token to anchor
        # to, so it goes in the one sensible leftover slot, right after the real
        # object, mirroring the Swahili paraphrase's own "verb obj1 obj2" word order
        # the user's own worked example used to explain this). The free-text e2e
        # pipeline (kitaita_e2e_pipeline.assemble_in_order/assemble_multi_clause)
        # places this at `object_idx + 0.5`; a caller that ignores this component
        # (e.g. a hand-built-spec sentence batch using `sentence` below directly)
        # still sees it correctly below since `sentence`'s own object-append happens
        # before this value is even read.
        idiom_object_trailing = idiom_object
    if kit_subject is None:
        sentence = kit_verb[:1].upper() + kit_verb[1:]
    else:
        sentence = f"{kit_subject.capitalize()} {kit_verb}"

    obj_part = None
    if swahili_object_noun:
        # 2026-10-22: "furahi" (enjoy/rejoice) is INHERENTLY PASSIVE in
        # Kitaita (see the verb-root branch above) -- its stimulus is a
        # "ni"-agent PHRASE ("pleased BY X"), not a plain direct object,
        # matching Swahili's own ACTIVE "furahia X" to Kitaita PASSIVE
        # "boilwa ni X" (PSA.1.2 "wadaboilwa ni Sharia ya BWANA" <->
        # "sheria ya BWANA ndiyo impendezayo", read directly). Single-
        # verb special case (not a general object-marking rule) --
        # flagged in the trace, not silently inserted.
        # 2026-10-12: found while testing the new Kitaita Word Bank noun
        # entries -- this branch only ever consulted the statistical
        # `noun_kitaita` table, never VERIFIED_NOUN_TRANSLATIONS, unlike the
        # SUBJECT side above (which checks VERIFIED_NOUN_TRANSLATIONS first).
        # Any native-speaker/Word-Bank-submitted noun used as an OBJECT was
        # silently falling through to "no reliable Kitaita translation" even
        # though a real answer existed -- e.g. 'mzee'->'mghosi'. Checked
        # first, same priority order as the subject side.
        if swahili_object_noun in VERIFIED_NOUN_TRANSLATIONS:
            obj_part = VERIFIED_NOUN_TRANSLATIONS[swahili_object_noun]
            trace.append(f"Object noun '{swahili_object_noun}' -> Kitaita '{obj_part}' "
                         f"(VERIFIED_NOUN_TRANSLATIONS -- direct submission/correction, overrides the "
                         f"statistical table) -- no object marker inserted (marker-only rewrite rule: "
                         f"object agreement optional, matches Swahili source)")
        else:
            obj = noun_kitaita.get(swahili_object_noun)
            if obj:
                obj_part = obj["kitaita_word"]
                trace.append(f"Object noun '{swahili_object_noun}' -> Kitaita '{obj_part}' "
                             f"(Dice {obj['best_dice']:.2f}, {obj['confidence']}) -- no object marker inserted "
                             f"(marker-only rewrite rule: object agreement optional, matches Swahili source)")
            else:
                trace.append(f"Object noun '{swahili_object_noun}': no reliable Kitaita translation -- FLAGGED")
        if obj_part is not None and swahili_verb["root"] == "furahi":
            # 2026-10-22: "furahi" (enjoy/rejoice) is INHERENTLY PASSIVE in
            # Kitaita (see the verb-root branch above) -- its stimulus is a
            # "ni"-agent PHRASE ("pleased BY X"), not a plain direct object,
            # matching Swahili's own ACTIVE "furahia X" to Kitaita PASSIVE
            # "boilwa ni X" (PSA.1.2 "wadaboilwa ni Sharia ya BWANA" <->
            # "sheria ya BWANA ndiyo impendezayo", read directly). Fused
            # directly into `obj_part` itself (not appended separately in
            # the local `sentence` string below) because the free-text e2e
            # pipeline's own reassembly (kitaita_e2e_pipeline.assemble_
            # multi_clause) places THIS field by token position, not the
            # local `sentence` string -- a separate append here would be
            # silently invisible in that pipeline, exactly as first found
            # when this produced "chumbo rawo" with no "ni" in testing.
            # Single-verb special case (not a general object-marking
            # rule), flagged in the trace, not silently inserted.
            obj_part = f"ni {obj_part}"
            trace.append("'furahi' object -> rendered as a \"ni\"-agent PHRASE (\"pleased BY X\"), "
                         "not a plain object -- this verb's own argument structure, not a general rule "
                         "(see VERIFIED_ROOTS['furahi']'s own comment)")
        if obj_part is not None and swahili_verb["root"] == "ko_have":
            # 2026-10-22: vocabulary-gap fix (plural "have") -- the "ko"
            # existential needs the comitative "na" ("with") before its
            # object, same construction as the 1pl/3pl corpus evidence
            # ("deko na ifunguki"/"weko na sere"/"wiko na ndighi") --
            # fused directly into `obj_part` for the same reason as
            # 'furahi' above (assemble_multi_clause places this field by
            # position, not a local string).
            obj_part = f"na {obj_part}"
            trace.append("'wana' (have) object -> prefixed with the comitative \"na\" (\"with\"), "
                         "matching the corpus construction ('[SM]ko na X'), not a plain object")
        if obj_part is not None:
            sentence += f" {obj_part}"
        if idiom_object_trailing:
            sentence += f" {idiom_object_trailing}"

    adj_part = None
    adjective_untranslated = False  # see the matching flag on the verb root above; set in the
    # final `else` branch of the adjective dispatch below when NO concord table/rule applies.
    if swahili_adjective:
        adj_key = (swahili_adjective["stem"], swahili_subject_noun)  # keyed by NOUN, not Swahili
        # class -- nchi/nyumba share a Swahili class label but take different 'kubwa' forms.
        # 2026-10-13 (cont. 2): adjective-concord lookups/formulas below use
        # `kit_class` (the noun's REAL, possibly-"i_nasal" class, computed
        # near the top of this function), not the TAM-normalized `kit_sm` --
        # kept in sync with kitaita_e2e_pipeline.compute_adjective_concord()'s
        # identical substitution.
        adj_class = kit_class
        if swahili_adjective["stem"] == "baya" and kit_subject in NUMBER_INVARIANT_NOUNS:
            # Every NUMBER_INVARIANT_NOUNS member is EXCLUDED from the generalized BAYA_STEM
            # rule below: each has a singular/plural subject-marker split (NUMBER_INVARIANT_
            # SM_SINGULAR/PLURAL, kitaita_verified_corrections.py) that Swahili's number-
            # invariant source word never disambiguates, and the one confirmed data point
            # (samaki, s48, "Samaki edeka izamie") used the SINGULAR 'i-' prefix, not the
            # 'ri-'/'ra-' plural default -- applying the general rule here would have
            # silently produced the wrong (plural) answer. Flagged, not guessed.
            adj_part = f"{NUMBER_INVARIANT_SM_SINGULAR['default']}{BAYA_STEM}"
            trace.append(f"Adjective 'baya' (bad) on '{kit_subject}': UNRESOLVED singular/plural "
                         f"ambiguity (this noun's Swahili source doesn't mark number) -- defaulting "
                         f"to the SINGULAR prefix '{NUMBER_INVARIANT_SM_SINGULAR['default']}-' (the "
                         f"one confirmed case, samaki s48, used singular) to get '{adj_part}', but "
                         f"the plural reading '{NUMBER_INVARIANT_SM_PLURAL['default']}{BAYA_STEM}' is "
                         f"equally plausible and unconfirmed either way -- see kitaita_verified_"
                         f"corrections.py)")
        elif swahili_adjective["stem"] == "baya" and adj_class:
            # Generalized rule (corpus-confirmed across all 5 classes tried, zero
            # counterexamples -- see kitaita_verified_corrections.py BAYA_STEM):
            # 'baya' (bad) = the noun's OWN subject-marker-class prefix + 'zamie'.
            # Uses the noun's base class marker (kit_sm as already resolved above),
            # not a present-tense-conditioned variant -- adjectives aren't verbs/TAM.
            # 2026-10-13 (cont. 3): 'baya' uses the BARE subject marker
            # (kit_sm, TAM-normalized -- NOT kit_class/adj_class) -- it does
            # NOT follow the mi-/nd- split (see TAM_SM_ALIAS's comment in
            # kitaita_verified_corrections.py for why nyumba's verified
            # "izamie" is correct despite nyumba being class 2/"i_nasal").
            adj_part = f"{kit_sm}{BAYA_STEM}"
            trace.append(f"Adjective 'baya' (bad): Kitaita concord = subject-marker-class prefix "
                         f"'{kit_sm}-' + stem '-{BAYA_STEM}' = '{adj_part}' (GENERALIZED rule, "
                         f"corpus-confirmed for every class tried: uzamie 12, ghuzamie 13, "
                         f"jizamie 36, rizamie 66, izamie 70 -- see kitaita_verified_corrections.py)")
        elif adj_key in VERIFIED_ADJECTIVES:
            adj_part = VERIFIED_ADJECTIVES[adj_key]
            trace.append(f"Adjective '{swahili_adjective['stem']}' (class {swahili_adjective['class']}): "
                         f"native-speaker-VERIFIED Kitaita form '{adj_part}' (first-pass seed data, "
                         f"not a full paradigm, uses a DIFFERENT concord strategy than 'baya' -- "
                         f"flat per-class lookup, not the subject-marker prefix; see "
                         f"kitaita_verified_corrections.py)")
        elif swahili_adjective["stem"] == "kubwa" and adj_class:
            # Generalized rule (cont. 16, extended cont. 2026-10-11): 'ji' (nchi, siku)
            # confirmed 'ibaa'; 'chi' (kikombe, kiti, kitabu) confirmed 'kibaa'; every other
            # marker tried (u/ghu/ri/i) confirmed 'mbaa' -- tracks the CONSONANT, not the
            # Swahili class label. Only used for a noun not already in the flat table above.
            if adj_class in KUBWA_FORM_BY_SM:
                adj_part = KUBWA_FORM_BY_SM[adj_class]
                trace.append(f"Adjective 'kubwa' (big), no exact record for this noun: subject "
                             f"marker '{adj_class}-' -> '{adj_part}' (GENERALIZED, CONFIRMED marker "
                             f"-- see kitaita_verified_corrections.py)")
            else:
                adj_part = KUBWA_DEFAULT_FORM
                trace.append(f"Adjective 'kubwa' (big), no exact record for this noun AND no "
                             f"confirmed data for marker '{adj_class}-': defaulting to the majority "
                             f"form '{adj_part}' (mungu/mti/mwana/samaki/nyumba's marker) as an "
                             f"UNCONFIRMED GUESS, not a verified answer for this marker -- flag "
                             f"for native-speaker review, see kitaita_verified_corrections.py)")
        elif swahili_adjective["stem"] in INGI_SURFACE_FORMS and adj_class:
            # 2026-10-20: own irregular/archaic concord series, same shape as
            # kubwa (not a kit_sm+stem formula like baya/zuri/safi) -- see
            # INGI_FORM_BY_SM's own comment in kitaita_verified_corrections.py.
            if adj_class in INGI_FORM_BY_SM:
                adj_part = INGI_FORM_BY_SM[adj_class]
                trace.append(f"Adjective '-ingi' (many/much), no exact record for this noun: subject "
                             f"marker '{adj_class}-' -> '{adj_part}' (GENERALIZED, CONFIRMED marker "
                             f"-- see kitaita_verified_corrections.py)")
            else:
                adj_part = INGI_DEFAULT_FORM
                trace.append(f"Adjective '-ingi' (many/much), no confirmed data for marker "
                             f"'{adj_class}-': defaulting to the majority form '{adj_part}' as an "
                             f"UNCONFIRMED GUESS, not a verified answer for this marker -- flag "
                             f"for native-speaker review, see kitaita_verified_corrections.py)")
        elif swahili_adjective["stem"] == "zuri" and kit_sm:
            # 2026-10-11: corpus-mined (concord_families.py), NOT speaker-verified like baya.
            # 2 independently-confirmed classes (i-, wi-) both use kit_sm + 'ghokie' -- same
            # STRATEGY as baya, but this is a statistical Dice candidate, a weaker evidence
            # tier than baya's native-speaker-confirmed answer. Extrapolating to a class
            # neither of the 2 confirmed covers is a further, unconfirmed step -- flagged.
            # 2026-10-13 (cont. 3): uses kit_sm (TAM-normalized), NOT kit_class/
            # adj_class -- zuri does NOT follow the mi-/nd- split (same as baya).
            adj_part = f"{kit_sm}{ZURI_ROOT}"
            # 2026-10-12: Kitaita Word Bank round confirmed every remaining
            # class -- only 'i'/'wi' predate it; no "extrapolated" case left
            # in practice, kept distinct in the trace for the historical record.
            confirmed = kit_sm in ("i", "wi", "chi", "ghi", "ghu", "ji", "ku", "lu", "ri", "u", "va")
            trace.append(f"Adjective 'zuri' (good): subject marker '{kit_sm}-' + corpus-mined root "
                         f"'-{ZURI_ROOT}' = '{adj_part}' ({'CONFIRMED class, Dice 0.40-0.58' if confirmed else 'EXTRAPOLATED to an untested class'}, "
                         f"statistical candidate from Bible-corpus mining, NOT native-speaker-verified "
                         f"like baya -- see kitaita_verified_corrections.py ZURI_ROOT)")
        elif swahili_adjective["stem"] == "safi" and kit_sm:
            # 2026-10-20: corpus-mined via reverse inference (26 short SUV/TAITA
            # verses containing "safi", read directly) -- SAME strategy/evidence
            # tier as 'zuri' (kit_sm + shared stem, NOT native-speaker-verified).
            # Stem "elie" confirmed across 6 independent classes with ZERO
            # counterexamples in direct noun+adjective or relative-clause use:
            # i- (dhahabu/nguwo/nia, e.g. MAT.27.59 "nguwo ielie"), ri- (nyonyi/
            # nyerinyeri, e.g. JOB.25.5 "nderielie"), vi- (vilambo, DEU.14.20 +
            # LUK.11.41 "vielie", 2 independent verses), wi- (human plural, awo/
            # wenye-moyo, PSA.73.1 + MAT.5.8 "wielie", 2 independent verses), u-
            # (mundu, PRO.15.26 "uelie"), gha- (malombi, JOB.16.17 "ghaelie").
            # Apparent non-matches checked and are semantically-divergent, not
            # counterexamples: "kitani ndicha" (PRO.31.22, fabric-quality sense
            # of "fine", not "clean"), "wuchafu" litotes (ISA.28.8, antonym +
            # negation), "madedo mecha" (PRO.15.4, free paraphrase), "kuela"/
            # "ieleshero"/"wueli" (PSA.51.7, JHN.15.3, PRO.8.10, ISA.13.12,
            # 1PE.3.2 -- a separate VERB/NOUN derivational family sharing the
            # same root, not needed for adjective concord). See docs/status.md.
            adj_part = f"{kit_sm}{SAFI_STEM}"
            confirmed = kit_sm in ("i", "ri", "vi", "wi", "u", "gha")
            trace.append(f"Adjective 'safi' (clean/pure): subject marker '{kit_sm}-' + corpus-mined "
                         f"root '-{SAFI_STEM}' = '{adj_part}' ({'CONFIRMED class' if confirmed else 'EXTRAPOLATED to an untested class'}, "
                         f"statistical candidate from Bible-corpus reverse inference, NOT native-speaker-"
                         f"verified -- see kitaita_verified_corrections.py SAFI_STEM)")
        elif swahili_adjective["stem"] == "moja" and kit_sm:
            # 2026-10-22: DIRECT native-speaker correction (stronger evidence
            # tier than zuri/safi above, which are corpus-mined) -- same
            # formula (full kit_sm + shared stem). Confirmed on 2 independent
            # classes: "ituku jimweri" (siku moja/one day, 'ji') and "chadu
            # chimweri" (kiatu kimoja/one shoe, 'chi') -- see kitaita_
            # verified_corrections.py MOJA_ROOT.
            adj_part = f"{kit_sm}{MOJA_ROOT}"
            confirmed = kit_sm in ("ji", "chi")
            trace.append(f"Adjective 'moja' (one): subject marker '{kit_sm}-' + root '-{MOJA_ROOT}' = "
                         f"'{adj_part}' ({'CONFIRMED, direct native-speaker correction' if confirmed else 'EXTRAPOLATED to an untested class'} "
                         f"-- see kitaita_verified_corrections.py MOJA_ROOT)")
        elif swahili_adjective["stem"] == "refu" and adj_class:
            # 2026-10-11 (cont. 8): direct user instruction -- use the Kitaita-
            # side 'lacha' root data (mlacha/walacha/milacha) directly, keyed by
            # KITAITA kit_sm, not by Swahili class label (same discipline as the
            # kubwa/ADJ_SINGLE_ROOT_GENERALIZED fix above -- "classes don't
            # necessarily line up"). Same get()-with-default shape as kubwa.
            if adj_class in LACHA_FORM_BY_SM:
                adj_part = LACHA_FORM_BY_SM[adj_class]
                trace.append(f"Adjective 'refu' (tall/long): subject marker '{adj_class}-' -> '{adj_part}' "
                             f"(CONFIRMED class, Dice 0.86, 3 versions agree -- statistical candidate, "
                             f"NOT native-speaker-verified; see kitaita_verified_corrections.py LACHA_FORM_BY_SM)")
            else:
                adj_part = LACHA_DEFAULT_FORM
                trace.append(f"Adjective 'refu' (tall/long): no confirmed data for marker '{adj_class}-': "
                             f"defaulting to '{adj_part}' (mrefu's own class, Dice 0.53) as an UNCONFIRMED "
                             f"GUESS, not a verified answer for this marker -- flag for native-speaker "
                             f"review (see kitaita_verified_corrections.py LACHA_DEFAULT_FORM)")
        elif swahili_adjective["stem"] in ("dogo", "geni", "fupi", "zito", "chafu") and adj_class:
            # 2026-10-12: Kitaita Word Bank round (same shape as kubwa/refu above,
            # kept in sync with kitaita_e2e_pipeline.compute_adjective_concord()
            # deliberately, not re-derived) -- see kitaita_verified_corrections.py
            # DOGO_FORM_BY_SM/GENI_FORM_BY_SM/FUPI_FORM_BY_SM/ZITO_FORM_BY_SM/
            # CHAFU_FORM_BY_SM for the submitted data and anomaly notes.
            stem = swahili_adjective["stem"]
            table, default = {
                "dogo": (DOGO_FORM_BY_SM, DOGO_DEFAULT_FORM),
                "geni": (GENI_FORM_BY_SM, GENI_DEFAULT_FORM),
                "fupi": (FUPI_FORM_BY_SM, FUPI_DEFAULT_FORM),
                "zito": (ZITO_FORM_BY_SM, ZITO_DEFAULT_FORM),
                "chafu": (CHAFU_FORM_BY_SM, CHAFU_DEFAULT_FORM),
            }[stem]
            if adj_class in table:
                adj_part = table[adj_class]
                trace.append(f"Adjective '{stem}': subject marker '{adj_class}-' -> '{adj_part}' "
                             f"(2026-10-12 Kitaita Word Bank, direct submission, NOT native-speaker-"
                             f"judgment-verified -- see kitaita_verified_corrections.py)")
            else:
                adj_part = default
                trace.append(f"Adjective '{stem}': no confirmed data for marker '{adj_class}-': defaulting "
                             f"to '{adj_part}' as an UNCONFIRMED GUESS -- flag for native-speaker review "
                             f"(see kitaita_verified_corrections.py)")
        elif swahili_adjective["stem"] in ADJ_SINGLE_ROOT_GENERALIZED and kit_sm:
            # 2026-10-11 (cont.): direct correction -- route via the KITAITA class
            # (kit_sm -> reduced prefix, same mechanism kubwa uses), not the Swahili
            # class label the ADJ_SINGLE_CLASS_CANDIDATE branch below still checks.
            # Confirmed case this fixes: 'dogo'/'mtini' failed to apply to kijana/
            # mdawana purely because kijana's SWAHILI class resolves to '7/8', even
            # though mdawana's own KITAITA class ('u', default) is exactly what
            # 'mtini' is itself shaped for -- Swahili class labels are not the
            # reliable signal here, Kitaita class groupings are.
            root = ADJ_SINGLE_ROOT_GENERALIZED[swahili_adjective["stem"]]
            prefix = kitaita_adj_prefix(kit_sm)
            adj_part = f"{prefix}{root}"
            trace.append(f"Adjective '{swahili_adjective['stem']}': single corpus-mined data point, "
                         f"generalized across Kitaita classes via the reduced-prefix series (kit_sm "
                         f"'{kit_sm}-' -> '-{prefix}-') + root '-{root}' = '{adj_part}' -- WEAKER "
                         f"confidence than kubwa (only 1 confirming class here, not 3+), NOT native-"
                         f"speaker-verified (see kitaita_verified_corrections.py ADJ_SINGLE_ROOT_GENERALIZED)")
        elif swahili_adjective["stem"] in ADJ_SINGLE_CLASS_CANDIDATE:
            cand = ADJ_SINGLE_CLASS_CANDIDATE[swahili_adjective["stem"]]
            if swahili_adjective["class"] == cand["swahili_class_seen"]:
                adj_part = cand["kitaita_word"]
                trace.append(f"Adjective '{swahili_adjective['stem']}': single corpus-mined data "
                             f"point for this EXACT class only (Dice {cand['dice']:.2f}) -> "
                             f"'{adj_part}' -- {cand['note']}; NOT shown to generalize to any other "
                             f"class, NOT native-speaker-verified (see kitaita_verified_corrections.py)")
            else:
                trace.append(f"Adjective '{swahili_adjective['stem']}' (class {swahili_adjective['class']}): "
                              f"only have a single, class-{cand['swahili_class_seen']}-specific corpus "
                              f"candidate for a DIFFERENT class, and this stem doesn't decompose against "
                              f"the reduced Kitaita prefix series either -- NOT applied, flagged rather "
                              f"than guessed (see kitaita_verified_corrections.py)")
        elif adj_class:
            # 2026-10-19, direct user request, same "Taita-fy the grammar, flag the
            # vocabulary gap" shape as the verb-root fallback above. UNLIKE the verb
            # case, this is a DOUBLE guess, not a single one: for verbs, SM+TAM
            # substitution is this whole project's one well-evidenced, uniformly-
            # applied mechanism (every translated verb already uses it); for
            # adjectives, the project has found the CONCORD STRATEGY ITSELF differs
            # per stem and is not settled even among the 9 it has data for -- 'baya'/
            # 'zuri' prefix the noun's FULL subject marker onto a shared root, while
            # 'kubwa'/'refu'/'dogo'/'geni'/'fupi'/'zito'/'chafu' use a DIFFERENT,
            # reduced prefix series (kitaita_adj_prefix/KITAITA_ADJ_BASE_HYPOTHESIS)
            # onto a stem-specific Kitaita root (ZURI_ROOT itself only has n=2 classes
            # confirmed, pointing a DIFFERENT direction than baya -- see its own
            # comment). For a stem with NO data at all, there is no basis to prefer
            # either strategy -- the reduced-prefix series is used here only because
            # it is the one explicitly built as a cross-stem GENERALIZATION (not a
            # single stem's own fitted table), not because it is confirmed to be
            # right for an unseen adjective. Flagged accordingly -- weaker evidence
            # than the verb fallback, not a parallel claim of equal confidence.
            prefix = kitaita_adj_prefix(adj_class)
            adj_part = f"{prefix}{swahili_adjective['stem']}"
            adjective_untranslated = True
            trace.append(f"Adjective '{swahili_adjective['stem']}' (class {swahili_adjective['class']}): "
                          f"NO Kitaita concord table/translation at all for this stem -- kept the "
                          f"ORIGINAL SWAHILI STEM and prefixed it with the reduced Kitaita concord "
                          f"series for marker '{adj_class}-' ('-{prefix}-') -> '{adj_part}' as a "
                          f"readability stopgap, FLAGGED as adjective_untranslated (DOUBLE guess: "
                          f"stem untranslated AND this project has NOT confirmed the reduced-prefix "
                          f"strategy generalizes to stems it has no data for at all -- 'baya'/'zuri' "
                          f"use a different strategy entirely; see notes)")
        else:
            trace.append(f"Adjective '{swahili_adjective['stem']}' (class {swahili_adjective['class']}): "
                          "no confirmed Kitaita adjective-concord table exists for this class, AND no "
                          "Kitaita class marker resolved for the noun either -- cannot even apply the "
                          "readability-stopgap prefix rule -- NOT constructed, flagged rather than "
                          "guessed (see notes)")

    # This function's OWN assembly always trails the adjective at the end --
    # correct and unchanged for its actual use case: kitaita_sentence_batch.py's
    # hand-built specs have no real "input word order" to preserve (build_
    # swahili_sentence() always displays noun+verb+[object]+"-adjective" by its
    # own fixed convention, purely to isolate one variable per spec), and every
    # VERIFIED_SENTENCES entry (including s86, s48) was judged against exactly
    # that convention, so this trailing order is what those judgments actually
    # verify. 2026-10-09: a free-text caller (kitaita_e2e_pipeline.py) that DOES
    # have a real input sentence to preserve order from should NOT rely on this
    # function's assembled `kitaita_sentence` string for that -- it should use
    # `components` below (including `kitaita_verb_word`) to re-assemble the
    # sentence in the ORIGINAL input's word order instead. See that module's
    # docstring for why: this project's whole methodology is marker-only
    # substitution, never rearrangement -- the adjective's output POSITION
    # should mirror wherever it sat in the real input, not a fixed template.
    if adj_part is not None:
        sentence += f" {adj_part}"

    sentence += "."
    components = {
        "subject_noun": kit_subject, "subject_marker": None if root_is_full_form else kit_sm,
        "tam": None if root_is_full_form else kit_tam, "root": kit_root,
        "root_is_full_form": root_is_full_form, "final_vowel": final_vowel,
        # 2026-10-19: True iff kit_root above is the untranslated SWAHILI root (no Kitaita
        # candidate at any tier) run through normal Kitaita SM+TAM substitution anyway --
        # a vocabulary gap, not a translation; callers doing coverage/testing should treat
        # any sentence with this flag set as "insufficient Kitaita vocabulary", not as a
        # confirmed answer (see the final `else` branch of the root-resolution chain above).
        "root_untranslated": root_untranslated,
        # 2026-10-20: set iff the periphrasis tier (VERIFIED_IDIOMS) fired -- already
        # folded into kitaita_verb_word below (the FULL "root + fixed object" string),
        # kept here too just for transparency/flagging, not something a caller needs
        # to place separately.
        "idiom_object": idiom_object,
        "idiom_object_trailing": idiom_object_trailing,
        "object_noun": obj_part, "adjective_stem": swahili_adjective["stem"] if swahili_adjective else None,
        "adjective_kitaita": adj_part, "adjective_untranslated": adjective_untranslated,
        "kitaita_verb_word": kit_verb,
    }
    return {"kitaita_sentence": sentence, "trace": trace, "object": obj_part,
            "adjective_flagged": bool(swahili_adjective), "components": components}
