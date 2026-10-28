# -*- coding: utf-8 -*-
"""
dav_swa_silver_agreement_metrics.py

2026-10-28 cont. Direct user request, after being told metric 1 (agreement
accuracy) and metric 2 (vocabulary choice) can't be computed on dav_swa at
all (both need per-slot/multi-speaker GOLD data this sentence-pair corpus
doesn't have): "build the agreement positions and determine whether our
output matches the expected output in agreement locations" for sentences
where our output "relatively" matches the real reference.

This is the SAME technique the live Space's feedback pipeline already uses
to turn a speaker's holistic corrected sentence into slot-level judgments
(docs/status.md, 2026-10-26 cont. 5: "sound specifically because assemble_
in_order/assemble_multi_clause are position-preserving"): kitaita_e2e_
pipeline.translate_sentence() never reorders/inserts/deletes relative to
its own tokenized input, so a SlotRecord's token_idx indexes the OUTPUT
sentence's tokens at that same position, not just the input's.

Difference from the Space's own use of the trick: there, the "correction"
is a native speaker's deliberate edit of OUR OWN output (so position
alignment is true by construction). Here, the "correction" is an
INDEPENDENT real translation (the dav_swa test reference) that happens to
come out close to our output -- position alignment is NOT guaranteed, so
it is only TRUSTED when (a) the reference tokenizes to the exact same
token count as our output (necessary for index correspondence to mean
anything) and (b) overall per-position token agreement clears a stated
threshold. Sentences that don't clear this bar are EXCLUDED, not guessed.

This produces a SILVER (automatically-derived, NOT native-speaker-
confirmed) version of metric 1 (agreement accuracy) and, as a direct
byproduct, metric 3 (stability) -- both via the EXISTING, UNMODIFIED
kitaita_eval_metrics.evaluate_corpus()/compute_agreement_accuracy()/
compute_stability(), not reimplemented. Metric 2 (vocabulary choice)
genuinely CANNOT be derived this way even in principle: it measures
agreement across SEVERAL independent native speakers' choices at a
position, and one reference sentence gives exactly one data point, not a
panel -- not attempted here, and not faked as if it were the same thing.

Every number below should be reported with "(silver, n=...)" -- this is a
statistical-tier result, not the native-speaker-verified gold the metrics
module's own docstring says the real metric 1/2 numbers require.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))

from extract_swahili_grammar import tokenize as grammar_tokenize
from kitaita_e2e_pipeline import translate_sentence
from kitaita_eval_metrics import GoldSlot, GoldSentence, evaluate_corpus

from dav_swa_corpus_eval import load_split, mine_noun_candidates, build_condition_tables

THRESHOLDS = [0.5, 0.7, 0.9]


def gtok(text: str) -> list[str]:
    return [t.lower() for t in grammar_tokenize(text or "")]


def build_silver_pairs(test_rows: list[dict], tables, threshold: float):
    """Returns (predicted_per_sentence, gold_sentences, counts) for sentences
    passing the position-alignment trust bar at this threshold."""
    predicted, gold = [], []
    counts = {"total": len(test_rows), "no_slots": 0, "token_count_mismatch": 0,
              "below_threshold": 0, "eligible": 0, "output_tokens_ne_d_tokens": 0}
    for row in test_rows:
        out = translate_sentence(row["swa"], tables)
        d = out.get("decomposition")
        if d is None or not d.slots:
            counts["no_slots"] += 1
            continue
        out_tokens = gtok(out.get("kitaita_sentence") or "")
        if len(out_tokens) != len(d.tokens):
            # REAL finding, not a tautology: assemble_in_order is NOT always position-
            # preserving in the token-COUNT sense -- some concord rewrites (e.g. a
            # demonstrative/associative expanding into two Kitaita words for one Swahili
            # word) insert a token. Confirmed directly (e.g. "maji"->"agho machi",
            # d.tokens len 4 vs out_tokens len 5) before trusting anything downstream.
            # SlotRecord.token_idx would misalign against out_tokens for these --
            # excluded from the silver set entirely, not guessed at.
            counts["output_tokens_ne_d_tokens"] += 1
            continue
        gold_tokens = gtok(row["dav"])
        if len(gold_tokens) != len(d.tokens):
            counts["token_count_mismatch"] += 1
            continue
        agree = sum(1 for a, b in zip(out_tokens, gold_tokens) if a == b) / len(gold_tokens)
        if agree < threshold:
            counts["below_threshold"] += 1
            continue
        counts["eligible"] += 1
        gold_slots = []
        for slot in d.slots:
            if slot.token_idx >= len(gold_tokens):
                continue
            gold_tok = gold_tokens[slot.token_idx]
            bare_swahili = d.tokens[slot.token_idx]
            gold_needs = gold_tok != bare_swahili
            gold_slots.append(GoldSlot(token_idx=slot.token_idx, slot_type=slot.slot_type,
                                        needs_rewrite=gold_needs,
                                        correct_form=gold_tok if gold_needs else None))
        predicted.append({"sentence": row["swa"], "slots": d.slots})
        gold.append(GoldSentence(sentence=row["swa"], gold_translation=row["dav"], slots=gold_slots))
    return predicted, gold, counts


def main():
    train_rows, _ = load_split("train")
    test_rows, _ = load_split("test")
    mined = mine_noun_candidates(train_rows)
    tables_a, tables_b, tables_c, added = build_condition_tables(mined)

    report = {}
    for cond_name, tables in [("A_current_production", tables_a),
                               ("B_current_plus_train_mined", tables_b)]:
        report[cond_name] = {}
        for th in THRESHOLDS:
            predicted, gold, counts = build_silver_pairs(test_rows, tables, th)
            metrics = evaluate_corpus(predicted, gold)
            report[cond_name][f"threshold_{th}"] = {
                "counts": counts,
                "agreement_accuracy": metrics["agreement_accuracy"],
                "stability": metrics["stability"],
            }

    out_path = HERE / "dav_swa_silver_agreement_results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda o: o.__dict__),
                         encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=lambda o: o.__dict__))


if __name__ == "__main__":
    main()
