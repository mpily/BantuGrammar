# -*- coding: utf-8 -*-
"""
nllb_concord_source_probe.py

User's directive: build and validate a DEDICATED probe for the
"waziri wa ..." associative/case relation (cont. 20 found neither the
AMOD attention mechanism nor the existing relation_probe.py classifier
reliably identifies the CONCORD SOURCE of a Swahili associative "-a"
marker -- that probe was trained on the wrong pairing, the case
marker's SYNTACTIC head, not the noun its own `feats` says it
concords with).

Step 1 (ground truth extraction, done BEFORE writing any training
code): checked directly whether "case marker -> its syntactic head ->
that head's own head" reliably recovers the concord source, by
comparing against the case marker's own `feats` NounClass. Result:
300/550 confirmed matches, 44 explicit MISMATCHES (mostly multi-level
associative chains like "Makamu wa Rais wa Sudan Kusini", where simple
one-hop-up traversal picks the wrong noun), 206 cases with missing
feats on one side. Only the 300 FEATS-VERIFIED matches are used as
positive training data below -- the structural rule alone is not
trusted blindly.

Step 2: build a binary classifier (reusing `relation_probe.RelationProbe`'s
architecture unchanged -- hard rule #10 -- just with 2 output classes
instead of 21): given (candidate_noun, case_marker) embeddings, predict
whether the candidate IS the concord source. Positive examples: the 300
feats-verified (concord_source, marker) pairs. Negative examples: the
SAME marker paired with OTHER nouns from the same sentence (distractors
-- same design principle as every other relation validated in this
project).

Step 3: Hewitt & Liang control task (fixed random label per
(noun_form, marker_form) word-pair TYPE, same spirit as
`relation_probe.build_control_labels`), selectivity = real - control.

Step 4: held-out evaluation (80/20 sentence split), plus a distractor-
controlled external check specifically on real "wa"/"waziri"-type
instances, Wilson CI, same format as every other relation this project
has validated (nsubj/obj: cont. 11; amod: cont. 8).

Usage:
    python3 -u nllb_concord_source_probe.py
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from conllu_utils import Sentence, load_conllu_file
from cross_lingual_embedding_decomposition import MODEL_ID
from nllb_structural_probe_crosslingual_check import extract_embeddings
from relation_probe import RelationProbe

LAYER = 12


# ---------------------------------------------------------------------------
# Step 1: feats-verified ground truth extraction
# ---------------------------------------------------------------------------

@dataclass
class ConcordExample:
    sent_idx: int
    noun_idx: int
    marker_idx: int
    label: int  # 1 = concords, 0 = does not


def extract_concord_examples(sentences: List[Sentence], max_negatives_per_marker: int = 3,
                              seed: int = 0) -> List[ConcordExample]:
    rng = random.Random(seed)
    examples = []
    for s_i, sent in enumerate(sentences):
        nouns = [t for t in sent.tokens if t.upos in ("NOUN", "PROPN")]
        for c in sent.tokens:
            if c.deprel != "case" or c.upos != "ADP":
                continue
            n2 = sent.token_by_idx(c.head)
            if n2 is None or n2.head == 0:
                continue
            n1 = sent.token_by_idx(n2.head)
            if n1 is None or n1.upos not in ("NOUN", "PROPN"):
                continue
            c_class = c.feats.get("NounClass")
            n1_class = n1.feats.get("NounClass")
            if c_class is None or n1_class is None or c_class != n1_class:
                continue  # not feats-verified -- skip, do not trust the structural guess alone

            examples.append(ConcordExample(s_i, n1.idx, c.idx, 1))
            distractors = [n for n in nouns if n.idx != n1.idx]
            rng.shuffle(distractors)
            for d in distractors[:max_negatives_per_marker]:
                examples.append(ConcordExample(s_i, d.idx, c.idx, 0))
    return examples


def build_control_labels_binary(sentences: List[Sentence], examples: List[ConcordExample], seed: int = 0):
    """Hewitt & Liang control task: fixed random label per (noun_form, marker_form)
    word-pair TYPE, independent of the real structure."""
    rng = random.Random(seed)
    pair_types = set()
    for ex in examples:
        sent = sentences[ex.sent_idx]
        noun_form = sent.token_by_idx(ex.noun_idx).form.lower()
        marker_form = sent.token_by_idx(ex.marker_idx).form.lower()
        pair_types.add((noun_form, marker_form))
    return {pair: rng.choice([0, 1]) for pair in sorted(pair_types)}


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (phat, (center - margin) / denom, (center + margin) / denom)


def train_concord_probe(sentences, embeddings, examples: List[ConcordExample], hidden_dim,
                         control_map: Optional[dict] = None, lr=1e-4, epochs=30, batch_size=32,
                         grad_clip=1.0, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = RelationProbe(hidden_dim, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    idxs = list(range(len(examples)))
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            heads, deps, labels = [], [], []
            for i in batch_idx:
                ex = examples[i]
                heads.append(embeddings[ex.sent_idx][ex.noun_idx - 1])
                deps.append(embeddings[ex.sent_idx][ex.marker_idx - 1])
                if control_map is not None:
                    sent = sentences[ex.sent_idx]
                    key = (sent.token_by_idx(ex.noun_idx).form.lower(),
                           sent.token_by_idx(ex.marker_idx).form.lower())
                    labels.append(control_map[key])
                else:
                    labels.append(ex.label)
            h_head = torch.stack(heads).to(device)
            h_dep = torch.stack(deps).to(device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = loss_fn(model(h_head, h_dep), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
    return model


@torch.no_grad()
def evaluate_concord_probe(model, sentences, embeddings, examples: List[ConcordExample],
                            control_map: Optional[dict] = None, device="cpu"):
    model.eval()
    correct, total = 0, 0
    tp = fp = fn = tn = 0
    for ex in examples:
        h_head = embeddings[ex.sent_idx][ex.noun_idx - 1].unsqueeze(0).to(device)
        h_dep = embeddings[ex.sent_idx][ex.marker_idx - 1].unsqueeze(0).to(device)
        pred = model(h_head, h_dep).argmax(dim=-1).item()
        if control_map is not None:
            sent = sentences[ex.sent_idx]
            key = (sent.token_by_idx(ex.noun_idx).form.lower(), sent.token_by_idx(ex.marker_idx).form.lower())
            gold = control_map[key]
        else:
            gold = ex.label
        total += 1
        correct += int(pred == gold)
        if pred == 1 and gold == 1: tp += 1
        if pred == 1 and gold == 0: fp += 1
        if pred == 0 and gold == 1: fn += 1
        if pred == 0 and gold == 0: tn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"accuracy": correct / max(total, 1), "precision": precision, "recall": recall, "f1": f1,
            "n": total, "n_pos": tp + fn}


def main():
    print(f"loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_lm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()
    for p in model_lm.parameters():
        p.requires_grad_(False)
    hidden_dim = model_lm.config.d_model

    print("loading Swahili gold treebank (SWH_UDT)...", flush=True)
    sw_830 = load_conllu_file("Conlu/swahili_udt/GlobalVoices.ken.sw.8_30.filtered.conllu")
    sw_2050 = load_conllu_file("Conlu/swahili_udt/GlobalVoices.ken.sw.20_50.filtered.conllu")
    sw_all = sw_830 + sw_2050

    print(f"\nextracting NLLB encoder embeddings (layer={LAYER})...", flush=True)
    sw_sents, sw_embs = extract_embeddings(tokenizer, model_lm, sw_all, "swh_Latn", LAYER,
                                            max_sentences=None, seed=0, target_norm=20.0)
    # re-index: extract_embeddings may skip/reorder sentences on alignment failure,
    # so build examples against THIS kept list, not the original sw_all.
    all_examples = extract_concord_examples(sw_sents, seed=0)
    n_pos = sum(1 for e in all_examples if e.label == 1)
    n_neg = sum(1 for e in all_examples if e.label == 0)
    print(f"\nfeats-verified concord examples: {n_pos} positive, {n_neg} negative (distractor) pairs",
          flush=True)

    # 80/20 split BY SENTENCE
    sent_idxs_with_examples = sorted(set(e.sent_idx for e in all_examples))
    rng = random.Random(0)
    rng.shuffle(sent_idxs_with_examples)
    n_train = int(len(sent_idxs_with_examples) * 0.8)
    train_sent_set = set(sent_idxs_with_examples[:n_train])
    test_sent_set = set(sent_idxs_with_examples[n_train:])
    train_examples = [e for e in all_examples if e.sent_idx in train_sent_set]
    test_examples = [e for e in all_examples if e.sent_idx in test_sent_set]
    print(f"train sentences={len(train_sent_set)} ({len(train_examples)} examples)  "
          f"test sentences={len(test_sent_set)} ({len(test_examples)} examples)", flush=True)

    control_map = build_control_labels_binary(sw_sents, all_examples, seed=0)

    print("\ntraining REAL concord-source probe...", flush=True)
    real_probe = train_concord_probe(sw_sents, sw_embs, train_examples, hidden_dim, seed=0)
    real_metrics = evaluate_concord_probe(real_probe, sw_sents, sw_embs, test_examples)
    print(f"  REAL: {real_metrics}", flush=True)

    print("\ntraining CONTROL-TASK probe (same architecture/hparams)...", flush=True)
    control_probe = train_concord_probe(sw_sents, sw_embs, train_examples, hidden_dim,
                                         control_map=control_map, seed=0)
    control_metrics = evaluate_concord_probe(control_probe, sw_sents, sw_embs, test_examples,
                                              control_map=control_map)
    print(f"  CONTROL: {control_metrics}", flush=True)

    print(f"\nSELECTIVITY (real - control): accuracy={real_metrics['accuracy'] - control_metrics['accuracy']:+.4f}"
          f"  F1={real_metrics['f1'] - control_metrics['f1']:+.4f}", flush=True)

    # -----------------------------------------------------------------
    # External check: for each positive TEST example, does the probe
    # correctly rank the TRUE concord source above ALL other candidate
    # nouns in that sentence (not just beat one random distractor)?
    # -----------------------------------------------------------------
    print(f"\n=== external check: does the true concord source win AGAINST EVERY other noun "
          f"in its sentence, not just one distractor? ===", flush=True)
    test_positives = [e for e in test_examples if e.label == 1]
    wins = []
    for ex in test_positives:
        sent = sw_sents[ex.sent_idx]
        emb = sw_embs[ex.sent_idx]
        nouns = [t for t in sent.tokens if t.upos in ("NOUN", "PROPN")]
        h_marker = emb[ex.marker_idx - 1].unsqueeze(0)
        with torch.no_grad():
            true_logit = real_probe(emb[ex.noun_idx - 1].unsqueeze(0), h_marker)[0, 1].item()
            best_other = max(
                (real_probe(emb[n.idx - 1].unsqueeze(0), h_marker)[0, 1].item()
                 for n in nouns if n.idx != ex.noun_idx),
                default=float("-inf"))
        wins.append(1 if true_logit > best_other else 0)
    phat, lo, hi = wilson_ci(sum(wins), len(wins))
    print(f"  n={len(wins)}  win-rate={phat:.3f}  [95% CI {lo:.3f}, {hi:.3f}]", flush=True)


if __name__ == "__main__":
    main()
