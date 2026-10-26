# -*- coding: utf-8 -*-
"""
kitaita_eval_metrics.py

2026-10-23. Direct user request: "now let's build our evaluation
metrics... 1. Agreement accuracy (wrong rewrite / missed slot /
spurious rewrite) 2. Vocabulary choice (% of native speakers who
chose the words the pipeline produced) 3. Stability (how accuracy
degrades as the number of agreement slots in a sentence increases).
Kindly implement the checkpoints and wire them in to the pipeline."

"Wired in" means: `kitaita_e2e_pipeline.py`'s concord mechanisms
(subject-marker agreement, adjective/possessive/associative/
demonstrative/-ote concord) now populate a real structured ledger,
`Decomposition.slots` (list[SlotRecord], see that file's own
`record_slot()`), at the exact point each decision is made -- NOT
reconstructed after the fact by parsing free-text flags. This module
is the CONSUMER of that ledger: it compares it against a GOLD
annotation (native-speaker-confirmed) for the same sentence and
computes the three metrics below.

IMPORTANT SCOPE NOTE, stated plainly rather than discovered later:
metrics 1 and 2 both require a GOLD reference (what a native speaker
says the correct Kitaita slot/word actually is) for the SAME
sentence the pipeline translated. This project does NOT yet have
native-speaker gold translations for the 71-sentence African
Storybook corpus (that collection is explicitly future work per the
user's own framing, "we will build a more descriptive pipeline that
involves native speakers later"). What exists TODAY and is used to
validate this module end-to-end: VERIFIED_SENTENCES (54 native-
speaker-confirmed (noun,root,tam,adjective)->sentence tuples,
kitaita_verified_corrections.py) and a handful of KNOWN_GOOD Bible
sentences (kitaita_e2e_pipeline.py's own regression battery). See
kitaita_eval_demo.py for that validation run. Metric 3 (stability)
needs per-sentence accuracy from metric 1, so it inherits the same
gold-data dependency.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path


# --------------------------------------------------------------------------- gold annotation format
@dataclass
class GoldSlot:
    """One native-speaker-confirmed concord decision for a sentence.
    `token_idx` must refer to the SAME tokenization the pipeline itself
    uses (whitespace-split, lowercased -- Decomposition.tokens) for the
    two ledgers to line up at all; kitaita_eval_demo.py shows how to
    build this by hand against a sentence's own d.tokens."""
    token_idx: int
    slot_type: str          # same vocabulary as SlotRecord: "subject_sm" | "adjective" | "possessive" |
                               # "associative" | "demonstrative" | "ote"
    needs_rewrite: bool      # does Kitaita concord differ from a bare passthrough here at all
    correct_form: str | None = None  # the native-speaker-confirmed Kitaita form, if needs_rewrite


@dataclass
class GoldSentence:
    sentence: str
    gold_translation: str | None
    slots: list = field(default_factory=list)  # list[GoldSlot]


def load_gold_jsonl(path) -> list[GoldSentence]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        slots = [GoldSlot(**s) for s in row.get("slots", [])]
        out.append(GoldSentence(sentence=row["sentence"], gold_translation=row.get("gold_translation"),
                                 slots=slots))
    return out


# --------------------------------------------------------------------------- metric 1: agreement accuracy
def compute_agreement_accuracy(predicted_slots: list, gold_slots: list[GoldSlot]) -> dict:
    """predicted_slots: Decomposition.slots (list[SlotRecord], kitaita_
    e2e_pipeline.py) for ONE sentence. gold_slots: the matching GoldSlot
    list for that SAME sentence. Matched by token_idx -- a slot present
    in only one side is exactly as informative as one present in both
    (gold-only -> missed or correct-absence; predicted-only -> spurious
    or a slot type gold didn't annotate at all, kept separate below).

    Returns per-slot classification into the user's own 3 error types
    plus "correct", with the full per-slot detail list for auditing
    (never just a bare number -- this project's own CLAUDE.md rule:
    "report n and confidence intervals; no selection... on n < 8" means
    the detail list matters at least as much as the aggregate here)."""
    gold_by_idx = {g.token_idx: g for g in gold_slots}
    pred_by_idx = {p.token_idx: p for p in predicted_slots}
    all_idxs = sorted(set(gold_by_idx) | set(pred_by_idx))

    counts = Counter()
    details = []
    for idx in all_idxs:
        g = gold_by_idx.get(idx)
        p = pred_by_idx.get(idx)
        gold_needs = g.needs_rewrite if g is not None else False
        gold_form = g.correct_form if g is not None else None
        pred_form = p.predicted_form if p is not None else None

        if g is None and p is not None:
            # gold never annotated this position at all -- cannot judge whether a
            # rewrite was needed, so this is EXCLUDED from scoring (not silently
            # counted as either correct or spurious) rather than guessed.
            label = "unannotated_by_gold"
        elif gold_needs and pred_form is None:
            label = "missed_slot"
        elif gold_needs and pred_form is not None and pred_form != gold_form:
            label = "wrong_rewrite"
        elif not gold_needs and pred_form is not None:
            label = "spurious_rewrite"
        else:
            label = "correct"
        counts[label] += 1
        details.append({"token_idx": idx, "slot_type": (g.slot_type if g else (p.slot_type if p else None)),
                         "gold_needs_rewrite": gold_needs, "gold_form": gold_form,
                         "predicted_form": pred_form, "label": label})

    scored = sum(v for k, v in counts.items() if k != "unannotated_by_gold")
    accuracy = counts["correct"] / scored if scored else None
    return {"counts": dict(counts), "scored_slots": scored, "accuracy": accuracy, "details": details}


def aggregate_agreement_accuracy(per_sentence: list[dict]) -> dict:
    """Sums counts across sentences (NOT an average of per-sentence
    accuracies, which would silently weight a 1-slot sentence equal to
    a 10-slot one) -- same pooling discipline as every other n-reporting
    rule this project has applied all session."""
    total = Counter()
    for r in per_sentence:
        total.update(r["counts"])
    scored = sum(v for k, v in total.items() if k != "unannotated_by_gold")
    accuracy = total["correct"] / scored if scored else None
    return {"counts": dict(total), "scored_slots": scored, "accuracy": accuracy, "n_sentences": len(per_sentence)}


# --------------------------------------------------------------------------- metric 2: vocabulary choice
def compute_vocabulary_choice(predicted_words: dict, native_choices: dict) -> dict:
    """predicted_words: {token_idx: kitaita_word_the_pipeline_produced}
    for ONE sentence (content words -- noun/verb-root lexical choices,
    NOT concord morphemes; those are metric 1's job). native_choices:
    {token_idx: [word_speaker_1_picked, word_speaker_2_picked, ...]} --
    one independent choice per native speaker who answered that
    position (this is the Kitaita Word Bank's own per-row submission
    shape: several speakers' answers for the same slot).

    Per the user's own framing ("check what percentage of the native
    speakers chose the words produced by the pipeline"): for each
    position, `support` = the fraction of THOSE SPEAKERS who
    independently picked the exact word the pipeline did. Overall
    accuracy is the mean support across every scored position
    (positions with zero native responses are excluded, not scored as
    0 -- no data is not the same as disagreement)."""
    details = []
    supports = []
    for idx, pred in predicted_words.items():
        choices = native_choices.get(idx)
        if not choices:
            continue
        support = sum(1 for c in choices if c == pred) / len(choices)
        supports.append(support)
        details.append({"token_idx": idx, "predicted": pred, "native_choices": list(choices),
                         "n_speakers": len(choices), "support": support})
    accuracy = sum(supports) / len(supports) if supports else None
    return {"scored_positions": len(supports), "mean_support": accuracy, "details": details}


def aggregate_vocabulary_choice(per_sentence: list[dict]) -> dict:
    all_supports = [d["support"] for r in per_sentence for d in r["details"]]
    return {"scored_positions": len(all_supports),
            "mean_support": sum(all_supports) / len(all_supports) if all_supports else None}


# --------------------------------------------------------------------------- metric 3: stability
def compute_stability(per_sentence_results: list[dict]) -> dict:
    """per_sentence_results: one dict per sentence, each already carrying
    "n_slots" (total agreement slots the GOLD annotation says this
    sentence requires -- i.e. `compute_agreement_accuracy()`'s own
    `scored_slots`, NOT the pipeline's own slot count, so a sentence
    the pipeline totally failed to find any slots in still counts its
    TRUE slot burden) and "accuracy" (that sentence's own agreement
    accuracy). Buckets by exact slot count rather than pre-binned
    ranges -- with gold data this scarce (see module docstring), a
    bucket boundary choice would hide more than it reveals; the caller
    can re-bin the returned curve once there's enough data for ranges
    to mean something (this project's own n>=8 rule)."""
    buckets = defaultdict(list)
    for r in per_sentence_results:
        if r["accuracy"] is not None:
            buckets[r["n_slots"]].append(r["accuracy"])
    curve = [{"n_slots": k, "mean_accuracy": sum(v) / len(v), "n_sentences": len(v)}
              for k, v in sorted(buckets.items())]
    flagged = [c for c in curve if c["n_sentences"] < 8]
    return {"curve": curve,
            "note": ("buckets with n_sentences < 8 are shown but NOT a confident estimate -- "
                      "this project's own n>=8 rule for claims") if flagged else None}


# --------------------------------------------------------------------------- shared runner
def evaluate_corpus(predicted_per_sentence: list[dict], gold_sentences: list[GoldSentence],
                     vocab_native_choices: dict | None = None) -> dict:
    """predicted_per_sentence: [{"sentence":..., "slots": list[SlotRecord],
    "predicted_words": {idx: word}}] aligned 1:1 with gold_sentences by
    list position (same order the gold file was built in -- see
    kitaita_eval_demo.py). vocab_native_choices: {sentence_idx:
    {token_idx: [choices]}}, optional (metric 2 is skipped per-sentence
    if omitted for that sentence)."""
    agreement_per_sentence, vocab_per_sentence, stability_rows = [], [], []
    for i, (pred, gold) in enumerate(zip(predicted_per_sentence, gold_sentences)):
        agr = compute_agreement_accuracy(pred["slots"], gold.slots)
        agreement_per_sentence.append(agr)
        stability_rows.append({"sentence": pred["sentence"], "n_slots": agr["scored_slots"],
                                "accuracy": agr["accuracy"]})
        if vocab_native_choices and i in vocab_native_choices:
            vocab_per_sentence.append(compute_vocabulary_choice(pred.get("predicted_words", {}),
                                                                   vocab_native_choices[i]))
    return {
        "agreement_accuracy": aggregate_agreement_accuracy(agreement_per_sentence),
        "agreement_accuracy_per_sentence": agreement_per_sentence,
        "vocabulary_choice": aggregate_vocabulary_choice(vocab_per_sentence) if vocab_per_sentence else None,
        "vocabulary_choice_per_sentence": vocab_per_sentence,
        "stability": compute_stability(stability_rows),
    }


def save_report(report: dict, path: str | Path) -> None:
    def default(o):
        if hasattr(o, "__dict__"):
            return asdict(o) if hasattr(o, "__dataclass_fields__") else o.__dict__
        return str(o)
    json.dump(report, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=default)
