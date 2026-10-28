# -*- coding: utf-8 -*-
"""
dav_swa_vocab_tolerant_agreement_metrics.py

2026-10-28 cont. "Approach 2" -- direct user request after seeing how few
sentences cont. 2/3's whole-sentence-threshold approach ("Approach 1") can
use: treat most sentence-level differences as VOCABULARY mismatches
(expected noise between two independent translations) rather than reasons
to exclude the whole sentence, and still score the agreement SLOT itself
whenever it can be aligned at all -- using ALL sentences, not a threshold-
gated subset. User's own framing: "if there is a correct word that is
still used for some agreement check whether it matches" -- i.e. separate
"is this the same underlying word" (vocabulary) from "is the agreement
marking on it right" (what metric 1 actually measures), instead of letting
the first contaminate the second.

Mechanically: a per-sentence whole-sentence SIMILARITY score is no longer
the gate. Instead, `difflib.SequenceMatcher` aligns `out_tokens` against
`gold_tokens` directly (handles different lengths from insertions/
deletions elsewhere in the sentence -- Approach 1 excluded ALL of those,
681/1,480 sentences at every threshold, cont. 3). Each slot's own aligned
counterpart is then classified into ONE of:
  - "correct"        -- matches (exact, or the subject_sm startswith rule
                         this project's app.py already uses, cont. 5 fix)
  - "wrong_rewrite"   -- doesn't match, but shares enough of a stem with
                         the gold word to say it's the SAME underlying
                         word with the WRONG agreement marking
  - "vocab_mismatch"  -- doesn't match and shares no real stem -- a
                         different word was used; cannot speak to
                         agreement correctness at all, scored separately,
                         not folded into correct/wrong
  - "missed_slot" / "spurious_rewrite" -- same meaning as Approach 1
  - "unalignable"     -- the slot's position falls in a `delete` block or
                         an unequal-size `replace` block with no clean
                         1:1 counterpart -- excluded, not guessed

This is a genuinely different method from Approach 1 (kitaita_eval_
metrics.compute_agreement_accuracy has no "vocab_mismatch" category, so
this does NOT reuse that function unchanged -- a 5th outcome category
needs its own classifier; Approach 1's script/numbers are left as they
are for comparison, per "one change per experiment" / not forking what
already exists). Which of the two is "the most honest" metric is an open
question this script does not resolve -- see docs/status.md.
"""
from __future__ import annotations

import difflib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))

from extract_swahili_grammar import tokenize as grammar_tokenize
from kitaita_e2e_pipeline import translate_sentence

from dav_swa_corpus_eval import load_split, mine_noun_candidates, build_condition_tables
from dav_swa_eval_common import gtok as _gtok_raw, normalize_ortho

MIN_SHARED_SUFFIX = 3  # stem-overlap bar deciding vocab_mismatch vs wrong_rewrite -- a chosen
                        # parameter, not derived; same spirit as this project's existing
                        # longest-common-suffix root checks (kitaita_sentence_generator.
                        # _derive_fallback_root), not reused verbatim since that function
                        # trims leading vowels for a different purpose (root isolation, not
                        # a same/different-word test).


def gtok(text: str) -> list[str]:
    """2026-10-28 cont. 6 fix (external-review feedback): apostrophe-
    normalized at the tokenization boundary, so every downstream string
    comparison in this file (the difflib alignment, startswith/equality
    checks, the stem-overlap heuristic) is apostrophe-insensitive for free,
    without needing a parallel normalized/raw pair of lists to keep in sync."""
    return [normalize_ortho(t) for t in _gtok_raw(text)]


def longest_common_suffix_len(a: str, b: str) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[-(k + 1)] == b[-(k + 1)]:
        k += 1
    return k


def shares_stem(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return longest_common_suffix_len(a, b) >= MIN_SHARED_SUFFIX


def align_out_to_gold(out_tokens: list[str], gold_tokens: list[str]) -> dict[int, str | None]:
    """Maps each out_tokens index to its aligned gold token, or None when no
    clean 1:1 counterpart exists (a `delete` block, or a `replace` block
    whose two sides have different lengths -- ambiguous which gold word a
    given out-token corresponds to, so left unmapped rather than guessed)."""
    sm = difflib.SequenceMatcher(None, out_tokens, gold_tokens, autojunk=False)
    mapping: dict[int, str | None] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace") and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                mapping[i1 + k] = gold_tokens[j1 + k]
        else:
            for i in range(i1, i2):
                mapping[i] = None
    return mapping


def classify_slot(slot, aligned_gold: str | None, bare_swahili: str, out_tokens: list[str]) -> dict:
    """2026-10-28 cont. 7 fix, found during the 50-case hand audit: normalizing
    `aligned_gold` HERE (idempotent if the caller already did) rather than
    trusting every caller to pre-normalize it -- dav_swa_source_aligned_
    agreement_metrics.py (Approach 3) imports the UNNORMALIZED shared `gtok`
    and never normalized its own gold-token list, so apostrophe variants
    (e.g. bare "wa" vs gold "w'a") were still reading as a mismatch there, a
    partial application of the cont. 6 orthography fix that only this direct
    audit surfaced -- 1/50 sampled missed_slot cases were purely this."""
    if aligned_gold is None:
        return {"label": "unalignable", "gold_form": None}
    aligned_gold = normalize_ortho(aligned_gold)
    bare_swahili = normalize_ortho(bare_swahili)
    gold_needs = aligned_gold != bare_swahili
    pf = normalize_ortho(slot.predicted_form) if slot.predicted_form else slot.predicted_form
    if not gold_needs:
        return {"label": "spurious_rewrite" if pf is not None else "correct", "gold_form": None}
    if pf is None:
        return {"label": "missed_slot", "gold_form": aligned_gold}
    if slot.slot_type == "subject_sm":
        matches = aligned_gold.startswith(pf)
        compare_word = out_tokens[slot.token_idx]  # whole word, for the stem check below
    else:
        matches = pf == aligned_gold
        compare_word = pf
    if matches:
        return {"label": "correct", "gold_form": aligned_gold}
    if shares_stem(compare_word, aligned_gold):
        return {"label": "wrong_rewrite", "gold_form": aligned_gold}
    return {"label": "vocab_mismatch", "gold_form": aligned_gold}


def run_condition(test_rows: list[dict], tables) -> dict:
    slot_counts = Counter()
    sentence_counts = Counter()
    non_slot_vocab_mismatches = 0
    non_slot_positions_compared = 0
    per_sentence_for_stability = []

    for row in test_rows:
        out = translate_sentence(row["swa"], tables)
        d = out.get("decomposition")
        if d is None or not d.slots:
            sentence_counts["no_slots"] += 1
            continue
        out_tokens = gtok(out.get("kitaita_sentence") or "")
        if len(out_tokens) != len(d.tokens):
            sentence_counts["output_tokens_ne_d_tokens"] += 1
            continue
        gold_tokens = gtok(row["dav"])
        mapping = align_out_to_gold(out_tokens, gold_tokens)
        sentence_counts["used"] += 1

        slot_idxs = {s.token_idx for s in d.slots}
        sentence_labels = []
        for slot in d.slots:
            aligned = mapping.get(slot.token_idx)
            bare = d.tokens[slot.token_idx]
            result = classify_slot(slot, aligned, bare, out_tokens)
            slot_counts[result["label"]] += 1
            sentence_labels.append(result["label"])

        # Descriptive only, not used to gate anything: non-slot positions that
        # DID get a clean 1:1 aligned counterpart, and whether that counterpart
        # differs (a "vocabulary mismatch" in the sense this whole script uses).
        for i, out_tok in enumerate(out_tokens):
            if i in slot_idxs:
                continue
            aligned = mapping.get(i)
            if aligned is None:
                continue
            non_slot_positions_compared += 1
            if aligned != out_tok:
                non_slot_vocab_mismatches += 1

        scorable = [l for l in sentence_labels if l not in ("unalignable",)]
        if scorable:
            correct = sum(1 for l in scorable if l == "correct")
            judgeable = [l for l in scorable if l != "vocab_mismatch"]
            acc = (sum(1 for l in judgeable if l == "correct") / len(judgeable)) if judgeable else None
            per_sentence_for_stability.append({"n_slots": len(judgeable), "accuracy": acc})

    scorable_total = sum(v for k, v in slot_counts.items() if k not in ("unalignable", "vocab_mismatch"))
    correct_total = slot_counts.get("correct", 0)

    stability = Counter()
    stability_sum = {}
    for r in per_sentence_for_stability:
        if r["accuracy"] is None or r["n_slots"] == 0:
            continue
        stability_sum.setdefault(r["n_slots"], []).append(r["accuracy"])

    return {
        "sentence_counts": dict(sentence_counts),
        "slot_counts": dict(slot_counts),
        "scorable_slots_excl_vocab_mismatch": scorable_total,
        "accuracy_excl_vocab_mismatch": (correct_total / scorable_total) if scorable_total else None,
        "non_slot_positions_compared": non_slot_positions_compared,
        "non_slot_vocab_mismatch_rate": (non_slot_vocab_mismatches / non_slot_positions_compared)
                                          if non_slot_positions_compared else None,
        "stability_by_n_judgeable_slots": {
            str(k): {"n_sentences": len(v), "mean_accuracy": sum(v) / len(v)}
            for k, v in sorted(stability_sum.items())
        },
    }


def main():
    train_rows, _ = load_split("train")
    test_rows, _ = load_split("test")
    mined = mine_noun_candidates(train_rows)
    tables_a, tables_b, tables_c, added = build_condition_tables(mined)

    report = {
        "A_current_production": run_condition(test_rows, tables_a),
        "B_current_plus_train_mined": run_condition(test_rows, tables_b),
    }
    out_path = HERE / "dav_swa_vocab_tolerant_agreement_results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
