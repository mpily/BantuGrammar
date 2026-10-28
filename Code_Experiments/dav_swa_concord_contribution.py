# -*- coding: utf-8 -*-
"""
dav_swa_concord_contribution.py

2026-10-28 cont. External-review feedback, relayed by the user, item 1:
"Run the naive-dictionary condition through both chrF and Approach 2.
Concord contribution = full pipeline - dictionary-only."

A TRUE naive baseline, not `assemble_in_order(d, None, noun_kitaita)`:
checked that function directly before reusing it and found it is NOT pure
dictionary substitution -- `apply_ote_concord`/`apply_associative_concord`/
`apply_demonstrative_concord` all run UNCONDITIONALLY inside `decompose_
sentence()` itself (confirmed by reading that function's body, not
assumed), independent of whether `translate()` (subject/adjective
agreement) ever runs or succeeds. Using it as "dictionary-only" would
silently keep most of one concord mechanism while dropping another,
understating concord's true contribution.

This script instead calls `decompose_sentence()` only for STRUCTURE
(where the nouns/slots are -- a property of the sentence's grammar, not
of translation choice) and renders its own output: every recognized noun
occurrence (subject/object/secondary) substituted via the SAME dictionary
lookup chain production uses (`lookup_noun_kitaita`), every other token
(verb, adjective, associative/demonstrative/ote particles, everything
else) kept as the untouched Swahili surface form. Zero agreement
computation of any kind -- the actual "naive dictionary substitution"
baseline this project's own CLAUDE.md Stage F plan names.

Scored two ways, both reusing existing code unchanged:
- chrF/exact-match vs. the real test reference (dav_swa_corpus_eval.py's
  own metric functions).
- Approach 2's classifier (dav_swa_vocab_tolerant_agreement_metrics.py),
  fed the SAME slot positions `d.slots` already identifies, but with every
  slot's `predicted_form` forced to `None` -- the honest statement of what
  a system that never attempts a rewrite anywhere actually produces at
  each one. This reuses classify_slot()/align_out_to_gold() unmodified.

Concord contribution = full-pipeline score - dictionary-only score, for
each metric. Not computed by rerunning a diff tool -- read directly off
the two conditions' numbers in the printed/saved report.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))

from extract_swahili_grammar import tokenize as grammar_tokenize_raw
from kitaita_e2e_pipeline import decompose_sentence, lookup_noun_kitaita, _siku_plural_context
from api.stats import chrf, mean_ci, rate_ci
from api.util import normalize_kitaita

from dav_swa_corpus_eval import load_split, mine_noun_candidates, build_condition_tables
from dav_swa_eval_common import gtok, normalize_ortho, tokens_match
from dav_swa_vocab_tolerant_agreement_metrics import align_out_to_gold, classify_slot
from collections import Counter


def naive_dictionary_translate(text: str, tables) -> tuple[str | None, object]:
    noun_kitaita = tables[0]
    noun_class_of = tables[3]
    d = decompose_sentence(text, noun_kitaita, noun_class_of, root_tables=tables)
    orig_tokens = grammar_tokenize_raw(text)
    if len(orig_tokens) != len(d.tokens):
        # Same real integrity risk flagged in cont. 2/3 -- extract_swahili_grammar.
        # tokenize() should be deterministic on the same string called twice, so
        # this should never actually fire; kept as a checked assumption, not an
        # unchecked one.
        return None, d
    out = list(orig_tokens)
    if d.subject_idx is not None:
        plural_ctx = _siku_plural_context(d.tokens, d.subject_idx) if d.subject == "siku" else False
        kit = lookup_noun_kitaita(d.subject, noun_kitaita, plural_context=plural_ctx)
        if kit:
            out[d.subject_idx] = kit
    if d.object_idx is not None:
        kit = lookup_noun_kitaita(d.object, noun_kitaita)
        if kit:
            out[d.object_idx] = kit
    for idx, _lemma, kit in d.secondary_nouns:
        if kit and idx < len(out):
            out[idx] = kit
    if not out:
        return None, d
    out[0] = out[0].capitalize()
    return " ".join(out) + ".", d


def run_chrf(test_rows: list[dict], tables) -> dict:
    exact_k = exact_n = 0
    chrf_scores = []
    for row in test_rows:
        hyp, _d = naive_dictionary_translate(row["swa"], tables)
        hyp = hyp or ""
        ref = row["dav"]
        exact_n += 1
        exact_k += int(tokens_match(normalize_kitaita(hyp), normalize_kitaita(ref)))
        chrf_scores.append(chrf(normalize_kitaita(hyp), normalize_kitaita(ref)))
    return {"exact_match": rate_ci(exact_k, exact_n), "chrf": mean_ci(chrf_scores), "n_scored": exact_n}


def run_approach2(test_rows: list[dict], tables) -> dict:
    slot_counts = Counter()
    sentence_counts = Counter()
    for row in test_rows:
        hyp, d = naive_dictionary_translate(row["swa"], tables)
        if d is None or not d.slots:
            sentence_counts["no_slots"] += 1
            continue
        out_tokens = gtok(hyp or "")
        if len(out_tokens) != len(d.tokens):
            sentence_counts["output_tokens_ne_d_tokens"] += 1
            continue
        gold_tokens = gtok(row["dav"])
        mapping = align_out_to_gold(out_tokens, gold_tokens)
        sentence_counts["used"] += 1
        # The honest statement of "no concord attempted anywhere": every slot's
        # predicted_form forced to None, everything else about the slot (its
        # position, type) left as decompose_sentence() found it -- that part is
        # a structural fact about the SENTENCE, not a translation choice.
        zeroed_slots = [dataclasses.replace(s, predicted_form=None) for s in d.slots]
        for slot in zeroed_slots:
            aligned = mapping.get(slot.token_idx)
            bare = d.tokens[slot.token_idx]
            result = classify_slot(slot, aligned, bare, out_tokens)
            slot_counts[result["label"]] += 1
    scorable_total = sum(v for k, v in slot_counts.items() if k not in ("unalignable", "vocab_mismatch"))
    correct_total = slot_counts.get("correct", 0)
    return {
        "sentence_counts": dict(sentence_counts),
        "slot_counts": dict(slot_counts),
        "scorable_slots_excl_vocab_mismatch": scorable_total,
        "accuracy_excl_vocab_mismatch": (correct_total / scorable_total) if scorable_total else None,
    }


def main():
    train_rows, _ = load_split("train")
    test_rows, _ = load_split("test")
    mined = mine_noun_candidates(train_rows)
    tables_a, tables_b, tables_c, added = build_condition_tables(mined)

    report = {
        "dictionary_only_A_tables": {"chrf": run_chrf(test_rows, tables_a),
                                       "approach2": run_approach2(test_rows, tables_a)},
        "dictionary_only_B_tables": {"chrf": run_chrf(test_rows, tables_b),
                                       "approach2": run_approach2(test_rows, tables_b)},
    }
    out_path = HERE / "dav_swa_concord_contribution_results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
