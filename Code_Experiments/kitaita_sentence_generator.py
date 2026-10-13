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
    KUBWA_FORM_BY_SM, KUBWA_DEFAULT_FORM, ZURI_ROOT, ADJ_SINGLE_CLASS_CANDIDATE,
    ADJ_SINGLE_ROOT_GENERALIZED, LACHA_FORM_BY_SM, LACHA_DEFAULT_FORM,
    DOGO_FORM_BY_SM, DOGO_DEFAULT_FORM, GENI_FORM_BY_SM, GENI_DEFAULT_FORM,
    FUPI_FORM_BY_SM, FUPI_DEFAULT_FORM, ZITO_FORM_BY_SM, ZITO_DEFAULT_FORM,
    CHAFU_FORM_BY_SM, CHAFU_DEFAULT_FORM, TAM_SM_ALIAS,
    kitaita_adj_prefix, kogh_re_o_shift,
    SWAHILI_CLASS_DEFAULT_SM_GUESS, ANIMATE_SG_EXCLUDED_FROM_DEFAULT,
    PERSON_SUBJECT_MARKERS_BY_TAM, VERIFIED_PAST_TAM_BY_SM,
    VERIFIED_NOUN_TRANSLATIONS, FUTURE_KUWA_FULL_ROOT_MARKERS, DEVERBAL_SM_HYPOTHESIS,
    CONDITIONAL_TAM_HYPOTHESIS,
)

PERSON_LABELS = {"1sg": "mimi (I)", "2sg": "wewe (you-sg)", "1pl": "sisi (we)", "2pl": "ninyi (you-pl)"}

OUT_DIR = Path(__file__).parent


def load_pages():
    p1 = json.loads((OUT_DIR / "page1_vocabulary.json").read_text(encoding="utf-8"))
    p2 = json.loads((OUT_DIR / "page2_noun_classes.json").read_text(encoding="utf-8"))
    p3 = json.loads((OUT_DIR / "page3_verb_rules.json").read_text(encoding="utf-8"))
    return p1, p2, p3


def build_lookup_tables(p1, p2, p3):
    noun_kitaita = {n["lemma"]: n["kitaita_candidates"][0] for n in p1["nouns"] if n["kitaita_candidates"]}
    root_kitaita, root_kitaita_fallback = {}, {}
    for vr in p1["verb_roots"]:
        cands = vr.get("root_level_kitaita_candidates")
        if cands:
            root_kitaita[vr["root_guess"]] = cands[0]
        old = vr.get("root_kitaita_candidate")
        if old:
            root_kitaita_fallback[vr["root_guess"]] = old
    corrected = p2["kitaita_corrected_classes"]
    noun_class_of = {}  # kitaita noun -> accepted-class follower prefix + its subject-marker subgroups
    for cls in corrected["accepted_classes"]:
        for sm, members in cls["subject_marker_subgroups"].items():
            if sm == "(no subject-marker data)":  # sentinel bucket, not a real marker
                continue
            for m in members:
                noun_class_of[m] = {"class": cls["follower_marker"], "subject_marker": sm}
    allomorphy = p3["allomorphy_investigation"]
    return noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy


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


def translate(swahili_subject_noun: str | None, swahili_verb: dict, swahili_object_noun: str | None,
              swahili_adjective: dict | None, tables, person: str | None = None,
              subject_class: str | None = None) -> dict:
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
    inventing an independent-pronoun word with zero evidence."""
    noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy = tables
    trace = []

    if person is not None:
        kit_subject = None
        trace.append(f"Pronoun subject '{PERSON_LABELS[person]}' -- Kitaita is pro-drop in every "
                     f"clean corpus example found (no separate pronoun word, just the verb prefix); "
                     f"output is the inflected verb alone (kitaita_person_marker_investigation.py)")
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

        if kit_subject is None:
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

    root_is_full_form = False
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
    elif swahili_verb["root"] == "kuw":
        kuwa_past_alt = KUWA_PAST_ALTERNATES[0]
        kit_root = kuwa_past_alt[0]
        trace.append(f"Verb root 'kuw' (be), past/bare existential: '{kit_root}' is a STRONG "
                     f"default (7 of 8 bare-existential-tested nouns confirmed -- mungu/mti/siku/"
                     f"mwana/samaki/mtu/nchi -- vs 1 exception, nyumba, which takes "
                     f"'{KUWA_PAST_ALTERNATES[1][0]}' instead). Not a coin-flip; used with real "
                     f"confidence here, with nyumba-like exceptions still possible on an untested "
                     f"noun and flagged as such if one turns up.")
    elif swahili_verb["root"] in VERIFIED_ROOTS:
        kit_root = VERIFIED_ROOTS[swahili_verb["root"]]
        trace.append(f"Verb root '{swahili_verb['root']}' -> native-speaker-VERIFIED '{kit_root}' "
                     f"(overrides the statistical candidate; see kitaita_verified_corrections.py)")
    elif (root_cand := root_kitaita.get(swahili_verb["root"])) is not None:
        kit_root = root_cand["kitaita_root"]
        trace.append(f"Verb root '{swahili_verb['root']}' -> Kitaita root '{kit_root}' "
                     f"(root-level Dice {root_cand['score']:.2f}, {root_cand['cooccur_count']} shared verses, "
                     f"{root_cand['n_competitors']} competitors)")
    else:
        fb = root_kitaita_fallback.get(swahili_verb["root"])
        if fb is None:
            return {"error": f"no Kitaita candidate (root-level or per-form) for Swahili root "
                              f"'{swahili_verb['root']}'", "trace": trace}
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
    if root_is_full_form:
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

    if root_is_full_form:
        kit_verb = kit_root
    else:
        kit_verb = f"{kit_sm or '?'}{kit_tam if kit_tam is not None else '?'}{kit_root}{final_vowel}"
    if kit_subject is None:
        sentence = kit_verb[:1].upper() + kit_verb[1:]
    else:
        sentence = f"{kit_subject.capitalize()} {kit_verb}"

    obj_part = None
    if swahili_object_noun:
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
            sentence += f" {obj_part}"
        else:
            obj = noun_kitaita.get(swahili_object_noun)
            if obj:
                obj_part = obj["kitaita_word"]
                trace.append(f"Object noun '{swahili_object_noun}' -> Kitaita '{obj_part}' "
                             f"(Dice {obj['best_dice']:.2f}, {obj['confidence']}) -- no object marker inserted "
                             f"(marker-only rewrite rule: object agreement optional, matches Swahili source)")
                sentence += f" {obj_part}"
            else:
                trace.append(f"Object noun '{swahili_object_noun}': no reliable Kitaita translation -- FLAGGED")

    adj_part = None
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
        else:
            trace.append(f"Adjective '{swahili_adjective['stem']}' (class {swahili_adjective['class']}): "
                          "no confirmed Kitaita adjective-concord table exists for this class -- "
                          "NOT constructed, flagged rather than guessed (see notes)")

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
        "object_noun": obj_part, "adjective_stem": swahili_adjective["stem"] if swahili_adjective else None,
        "adjective_kitaita": adj_part, "kitaita_verb_word": kit_verb,
    }
    return {"kitaita_sentence": sentence, "trace": trace, "object": obj_part,
            "adjective_flagged": bool(swahili_adjective), "components": components}
