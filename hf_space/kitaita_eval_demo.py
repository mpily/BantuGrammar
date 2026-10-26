# -*- coding: utf-8 -*-
"""
kitaita_eval_demo.py

2026-10-23. Validates kitaita_eval_metrics.py end-to-end -- NOT a
claim about the real pipeline's overall accuracy (this project has
no native-speaker gold translations for the African Storybook corpus
yet, see kitaita_eval_metrics.py's own docstring). Two parts:

PART 1 (synthetic): hand-built predicted-vs-gold slot pairs that
exercise each of the 3 error types the user named plus "correct", to
prove `compute_agreement_accuracy()` classifies each one right. These
are NOT real sentences -- fabricating a "pipeline got this wrong"
example from real Kitaita data would misrepresent actual native-
speaker evidence; the classifier logic itself needs checking
independent of whether any real sentence happens to hit every case.

PART 2 (real): the pipeline's regression battery already carries a
handful of native-speaker-reviewed, zero-known-errors sentences
(kitaita_e2e_pipeline.py's own KNOWN_GOOD list, +2 from the "novel
free-text" list that are ALSO clean) -- hand-annotated as gold here
(by reading d.tokens/d.slots directly, not guessed) and run through
the real pipeline + real `Decomposition.slots` ledger. Expected: 100%
agreement accuracy, confirming the ledger integrates correctly with
real pipeline output, not just the metric math in isolation.
"""
import json
from pathlib import Path

from kitaita_e2e_pipeline import translate_sentence_with_probes, load_pages, build_lookup_tables
from kitaita_eval_metrics import (
    GoldSlot, GoldSentence, compute_agreement_accuracy, aggregate_agreement_accuracy,
    compute_vocabulary_choice, aggregate_vocabulary_choice, compute_stability, save_report,
)

OUT_DIR = Path(__file__).parent


def part1_synthetic_classifier_check():
    print("=== PART 1: synthetic classifier check (not real sentences) ===")
    # slot 0: pipeline correctly rewrote -- CORRECT
    # slot 1: pipeline rewrote, but to the wrong form -- WRONG_REWRITE
    # slot 2: gold says this needs a rewrite, pipeline produced none -- MISSED_SLOT
    # slot 3: gold says NO rewrite needed here, pipeline produced one anyway -- SPURIOUS_REWRITE
    # slot 4: both agree no rewrite needed -- CORRECT
    gold = [
        GoldSlot(token_idx=0, slot_type="adjective", needs_rewrite=True, correct_form="mbaa"),
        GoldSlot(token_idx=1, slot_type="adjective", needs_rewrite=True, correct_form="ighokie"),
        GoldSlot(token_idx=2, slot_type="subject_sm", needs_rewrite=True, correct_form="u"),
        GoldSlot(token_idx=3, slot_type="adjective", needs_rewrite=False, correct_form=None),
        GoldSlot(token_idx=4, slot_type="subject_sm", needs_rewrite=False, correct_form=None),
    ]

    class FakeSlot:
        def __init__(self, token_idx, slot_type, predicted_form):
            self.token_idx, self.slot_type, self.predicted_form = token_idx, slot_type, predicted_form

    predicted = [
        FakeSlot(0, "adjective", "mbaa"),      # matches gold -> correct
        FakeSlot(1, "adjective", "ghokie"),    # wrong form -> wrong_rewrite
        # slot 2 has NO predicted entry at all -> missed_slot
        FakeSlot(3, "adjective", "mbaa"),      # gold says none needed -> spurious_rewrite
        # slot 4 has no predicted entry either, and gold also needed none -> correct
    ]
    result = compute_agreement_accuracy(predicted, gold)
    print(json.dumps(result["counts"], indent=2))
    expected = {"correct": 2, "wrong_rewrite": 1, "missed_slot": 1, "spurious_rewrite": 1}
    assert result["counts"] == expected, f"classifier mismatch: {result['counts']} != {expected}"
    print("PASS: all 4 labels (correct/wrong_rewrite/missed_slot/spurious_rewrite) classified as expected.\n")


# PART 2: hand-annotated gold for real, native-speaker-reviewed sentences.
# token_idx values read directly from each sentence's own d.tokens (confirmed
# by running the pipeline first, not guessed) -- see each comment.
REAL_GOLD = [
    # tokens: [mungu, akawona] -- regression battery KNOWN_GOOD, Kitaita "Mlungu ukawona."
    GoldSentence("Mungu akawona.", "Mlungu ukawona.",
                 [GoldSlot(1, "subject_sm", True, "u")]),
    # tokens: [mtu, mkubwa, akaona] -- Kitaita "Mundu mbaa ukawona."
    GoldSentence("Mtu mkubwa akaona.", "Mundu mbaa ukawona.",
                 [GoldSlot(1, "adjective", True, "mbaa"), GoldSlot(2, "subject_sm", True, "u")]),
    # tokens: [mbwa, mkubwa, aliona, mwanafunzi] -- Kitaita "Koshi mbaa idewona mnughi."
    GoldSentence("Mbwa mkubwa aliona mwanafunzi.", "Koshi mbaa idewona mnughi.",
                 [GoldSlot(1, "adjective", True, "mbaa"), GoldSlot(2, "subject_sm", True, "i")]),
    # tokens: [simba, mkubwa, alikimbia] -- Kitaita "Shimba mbaa idekimbia."
    GoldSentence("Simba mkubwa alikimbia.", "Shimba mbaa idekimbia.",
                 [GoldSlot(1, "adjective", True, "mbaa"), GoldSlot(2, "subject_sm", True, "i")]),
    # tokens: [mwalimu, mzuri, alisoma, kitabu] -- Kitaita "Mwalimu ughokie udeshoma chuo."
    GoldSentence("Mwalimu mzuri alisoma kitabu.", "Mwalimu ughokie udeshoma chuo.",
                 [GoldSlot(1, "adjective", True, "ughokie"), GoldSlot(2, "subject_sm", True, "u")]),
    # tokens: [chui, mkubwa, alimwona, mbwa] -- Kitaita "Inge mbaa idewona koshi."
    GoldSentence("Chui mkubwa alimwona mbwa.", "Inge mbaa idewona koshi.",
                 [GoldSlot(1, "adjective", True, "mbaa"), GoldSlot(2, "subject_sm", True, "i")]),
]


def part2_real_pipeline_validation():
    print("=== PART 2: real pipeline, real native-speaker-reviewed gold (n=6 sentences) ===")
    p1, p2, p3 = load_pages()
    tables = build_lookup_tables(p1, p2, p3)

    per_sentence = []
    for gold in REAL_GOLD:
        report = translate_sentence_with_probes(gold.sentence, tables)
        d = report["decomposition"]
        agr = compute_agreement_accuracy(d.slots, gold.slots)
        per_sentence.append(agr)
        print(f"{gold.sentence:<45} -> {report['kitaita_sentence']:<30} "
              f"accuracy={agr['accuracy']} counts={agr['counts']}")
        if agr["accuracy"] != 1.0:
            for det in agr["details"]:
                if det["label"] != "correct":
                    print("   MISMATCH:", det)

    agg = aggregate_agreement_accuracy(per_sentence)
    print("\nAggregate:", json.dumps(agg, indent=2))
    assert agg["accuracy"] == 1.0, (
        "Expected 100% on a hand-picked, already-regression-clean gold set -- a miss here means "
        "the LEDGER or this demo's gold annotation disagrees with the real pipeline, not that the "
        "pipeline itself regressed (the regression battery already guards that separately)."
    )
    print("PASS: ledger + gold format integrate correctly with the real pipeline.\n")
    return per_sentence


def part3_vocabulary_and_stability_demo(agreement_per_sentence):
    print("=== PART 3: vocabulary-choice and stability math (synthetic native-choice data) ===")
    # No real multi-speaker word-choice data exists yet either (same gap as the
    # agreement metric) -- this demonstrates compute_vocabulary_choice()'s math only.
    predicted_words = {0: "mlungu", 3: "mundu"}
    native_choices = {0: ["mlungu", "mlungu", "mulungu"], 3: ["mundu", "mutu", "mundu"]}
    vocab = compute_vocabulary_choice(predicted_words, native_choices)
    print("vocabulary_choice (synthetic):", json.dumps(vocab, indent=2))
    # idx 0: "mlungu" said by 2/3 speakers; idx 3: "mundu" said by 2/3 speakers -> mean 2/3
    assert abs(vocab["mean_support"] - (2 / 3)) < 1e-9

    stability_rows = [{"sentence": "x", "n_slots": a["scored_slots"], "accuracy": a["accuracy"]}
                        for a in agreement_per_sentence]
    stability = compute_stability(stability_rows)
    print("stability curve (from PART 2's real sentences, n too small to mean much yet):")
    print(json.dumps(stability, indent=2))
    print("PASS: vocabulary-choice and stability computation both run correctly.\n")


def main():
    part1_synthetic_classifier_check()
    agreement_per_sentence = part2_real_pipeline_validation()
    part3_vocabulary_and_stability_demo(agreement_per_sentence)
    save_report({"note": "validation run, not a real-corpus evaluation -- see module docstrings",
                 "part1": "PASS", "part2_aggregate": aggregate_agreement_accuracy(agreement_per_sentence)},
                OUT_DIR / "kitaita_eval_demo_report.json")
    print("All checks passed. Report: kitaita_eval_demo_report.json")


if __name__ == "__main__":
    main()
