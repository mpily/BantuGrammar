# -*- coding: utf-8 -*-
"""
dav_swa_source_aligned_agreement_metrics.py

2026-10-28 cont. External-review feedback, item 4 ("Approach 3"): "Align
source to reference, not output to reference. Train eflomal or fast_align
on the 21k training pairs. Map each Swahili slot position to its gold
Kitaita word. This removes the dependence on your own output's token
count, including the expansion bug. It should cut unalignable
substantially, and it makes the scoring independent of what the pipeline
produced."

Approach 1 (silver_agreement) and Approach 2 (vocab_tolerant) both aligned
OUR OWN rendered OUTPUT against the gold reference -- meaning both
inherited the cont. 2 token-expansion bug's exclusion zone, and both
depend on the pipeline having produced SOMETHING reasonable to align
against in the first place (circular: a badly-translated sentence is both
the thing being measured and a reason it gets excluded from measurement).

This script instead trains `eflomal` (pip-installable; `fast_align`
wasn't -- would need a from-source C++ build with no clear benefit here)
on the 21,329 TRAIN pairs (Swahili source, real Kitaita target) and asks
it directly: for THIS Swahili test sentence, which word in the REAL gold
Kitaita reference does Swahili position i align to? That mapping exists
whether or not our pipeline ever ran, let alone what it produced -- the
expansion bug, and the whole "does our output's token count match"
question, simply don't apply to it.

eflomal is run over TRAIN+TEST sentences in one corpus (standard practice
for a statistical aligner used as a measurement tool, not a leakage
concern for THIS project's own pipeline -- the aligner never sees or
uses our pipeline's output, only the real Swahili/Kitaita pairs, and nothing
about our pipeline is fit to this data). Viterbi alignment, source->target
direction ("fwd"); a source token eflomal could not align to anything
(dropped from the fwd link list, effectively NULL-aligned) is EXCLUDED,
not guessed.

Reuses dav_swa_vocab_tolerant_agreement_metrics.classify_slot() UNCHANGED
for the actual outcome taxonomy (correct/wrong_rewrite/vocab_mismatch/
missed_slot/spurious_rewrite/unalignable) -- only how "aligned_gold" is
obtained changes.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))

import eflomal

from kitaita_e2e_pipeline import translate_sentence
from dav_swa_corpus_eval import load_split, mine_noun_candidates, build_condition_tables
from dav_swa_eval_common import gtok
from dav_swa_vocab_tolerant_agreement_metrics import classify_slot


def build_fwd_alignments(train_rows: list[dict], test_rows: list[dict]):
    """Returns (test_swa_tokens, test_dav_tokens, fwd_maps) -- fwd_maps[k] is
    {swa_idx: dav_idx} for test_rows[k], Viterbi source->target, from eflomal
    trained on train_rows+test_rows' tokenized text (see module docstring for
    why combining them is fine here)."""
    all_rows = train_rows + test_rows
    src_lines = [" ".join(gtok(r["swa"])) for r in all_rows]
    trg_lines = [" ".join(gtok(r["dav"])) for r in all_rows]

    fwd_path = HERE / "_eflomal_fwd.tmp"
    rev_path = HERE / "_eflomal_rev.tmp"
    aligner = eflomal.Aligner()
    aligner.align(src_lines, trg_lines, links_filename_fwd=str(fwd_path), links_filename_rev=str(rev_path))
    fwd_lines = fwd_path.read_text(encoding="utf-8").splitlines()
    fwd_path.unlink(missing_ok=True)
    rev_path.unlink(missing_ok=True)

    n_train = len(train_rows)
    test_src_tokens = [gtok(r["swa"]) for r in test_rows]
    test_trg_tokens = [gtok(r["dav"]) for r in test_rows]
    fwd_maps = []
    for k in range(len(test_rows)):
        line = fwd_lines[n_train + k]
        m: dict[int, int] = {}
        for pair in line.split():
            i_str, j_str = pair.split("-")
            i, j = int(i_str), int(j_str)
            if i not in m:  # first link wins if a source position has >1 (rare at model 3) -- documented choice
                m[i] = j
        fwd_maps.append(m)
    return test_src_tokens, test_trg_tokens, fwd_maps


def run_condition(test_rows: list[dict], tables, fwd_maps, test_dav_tokens) -> dict:
    slot_counts = Counter()
    sentence_counts = Counter()
    missingness = defaultdict(lambda: Counter())  # (slot_type, length_bucket) -> label counter

    def length_bucket(n_tokens: int) -> str:
        if n_tokens <= 4:
            return "1-4"
        if n_tokens <= 8:
            return "5-8"
        if n_tokens <= 12:
            return "9-12"
        return "13+"

    for k, row in enumerate(test_rows):
        out = translate_sentence(row["swa"], tables)
        d = out.get("decomposition")
        if d is None or not d.slots:
            sentence_counts["no_slots"] += 1
            continue
        out_tokens = gtok(out.get("kitaita_sentence") or "")
        sentence_counts["used"] += 1
        fwd = fwd_maps[k]
        gold_tokens = test_dav_tokens[k]
        bucket = length_bucket(len(d.tokens))
        for slot in d.slots:
            j = fwd.get(slot.token_idx)
            aligned = gold_tokens[j] if (j is not None and j < len(gold_tokens)) else None
            bare = d.tokens[slot.token_idx]
            result = classify_slot(slot, aligned, bare, out_tokens)
            slot_counts[result["label"]] += 1
            missingness[(slot.slot_type, bucket)][result["label"]] += 1

    scorable_total = sum(v for k2, v in slot_counts.items() if k2 not in ("unalignable", "vocab_mismatch"))
    correct_total = slot_counts.get("correct", 0)
    return {
        "sentence_counts": dict(sentence_counts),
        "slot_counts": dict(slot_counts),
        "scorable_slots_excl_vocab_mismatch": scorable_total,
        "accuracy_excl_vocab_mismatch": (correct_total / scorable_total) if scorable_total else None,
        "missingness_by_slot_type_and_length": {
            f"{st}|{bk}": dict(c) for (st, bk), c in sorted(missingness.items())
        },
    }


def main():
    train_rows, _ = load_split("train")
    test_rows, _ = load_split("test")
    mined = mine_noun_candidates(train_rows)
    tables_a, tables_b, tables_c, added = build_condition_tables(mined)

    print("Training eflomal on", len(train_rows), "pairs + aligning", len(test_rows), "test pairs...",
          file=sys.stderr)
    _src, test_dav_tokens, fwd_maps = build_fwd_alignments(train_rows, test_rows)

    report = {
        "A_current_production": run_condition(test_rows, tables_a, fwd_maps, test_dav_tokens),
        "B_current_plus_train_mined": run_condition(test_rows, tables_b, fwd_maps, test_dav_tokens),
    }
    out_path = HERE / "dav_swa_source_aligned_agreement_results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
