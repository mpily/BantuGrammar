# -*- coding: utf-8 -*-
"""
nllb_swahili_embedding_decodability_probe.py

Claim 1a (Swahili half), redone with better guarantees per the
2026-10-01 audit. Multi-class (11-way noun class, not binary gender)
extension of `nllb_french_embedding_decodability_probe.py`'s ported
methodology, so both languages are finally tested with the SAME
statistical tool (trained 5-fold-CV classifier + N-shuffle control +
selectivity), not the mismatched pair this project had before (French:
trained logistic-regression probe, 82.2% vs 51% control; Swahili: a
naive centroid cohesion/separation metric with no trained classifier at
all). Uses `swahili_noun_lexicon_agreement_v2.py` (319 nouns, 11 classes
labeled by OBSERVED NLLB concord, not morphological shape -- the
project's own current-best lexicon per the 2026-10-01 infrastructure
survey) and the same whole-word mean-pooling fix applied to the French
script (status.md, 2026-09-25: first-subword-only pooling collapsed
125/319 Swahili words onto 35 identical vectors under this exact
lexicon -- checked again here on NLLB's own tokenizer, not assumed to
be the same collision count).

Multi-class logistic regression (linear layer, num_classes outputs,
cross-entropy) replaces the binary version; the shuffle control
permutes the full label vector exactly as the French script does (same
convention, not a different one invented for this case).

Usage:
    python3 -u nllb_swahili_embedding_decodability_probe.py
"""
from __future__ import annotations

import random
from collections import Counter

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import mbart_agreement_lib as lib
from swahili_noun_lexicon_agreement_v2 import NOUNS

MODEL_ID = "facebook/nllb-200-distilled-1.3B"
N_FOLDS = 5
N_CONTROL_SHUFFLES = 20
TRAIN_STEPS = 300
LR = 0.05
WEIGHT_DECAY = 1e-2
SEED = 0


def whole_word_embedding(tokenizer, weight, word):
    ids = lib.full_word_subword_ids(tokenizer, word)
    return weight.data[ids].mean(dim=0)


def kfold_indices(n, k, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


def train_and_eval_fold(X, y, num_classes, train_idx, test_idx, return_model=False):
    d = X.shape[1]
    clf = nn.Linear(d, num_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    for _ in range(TRAIN_STEPS):
        opt.zero_grad()
        loss = loss_fn(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = clf(Xte).argmax(dim=-1)
        acc = (preds == yte).float().mean().item()
    return (acc, clf) if return_model else acc


def cross_val_accuracy(X, y, num_classes, k, seed):
    n = X.shape[0]
    folds = kfold_indices(n, k, seed)
    accs = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = [j for f in folds for j in f if f is not folds[i]]
        accs.append(train_and_eval_fold(X, y, num_classes, train_idx, test_idx))
    return sum(accs) / len(accs)


def build_X_y(tokenizer, weight, verbose=True):
    """Assembles the 319-noun/11-class (X, y) matrix. Factored out of
    `main` (2026-10-01) for reuse by
    `nllb_swahili_class_signal_location_check.py`, which needs the exact
    same data this script's validated +0.65-selectivity result was
    computed on (CLAUDE.md hard rule #10 -- extend, don't refork)."""
    classes = sorted(set(c for _, c, _ in NOUNS))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if verbose:
        print(f"classes ({len(classes)}): {classes}", flush=True)

    embs, labels, collided = [], [], 0
    seen_id_tuples = {}
    for noun, cls, gloss in NOUNS:
        ids = tuple(lib.full_word_subword_ids(tokenizer, noun))
        if ids in seen_id_tuples:
            collided += 1
        seen_id_tuples[ids] = noun
        embs.append(whole_word_embedding(tokenizer, weight, noun))
        labels.append(class_to_idx[cls])
    X = torch.stack(embs)
    y = torch.tensor(labels, dtype=torch.long)

    if verbose:
        counts = Counter(labels)
        n_total = len(y)
        majority_baseline = max(counts.values()) / n_total
        print(f"n={n_total}, per-class counts: "
              f"{ {classes[i]: counts[i] for i in range(len(classes))} }", flush=True)
        print(f"majority-class baseline={majority_baseline:.4f}", flush=True)
        print(f"whole-word-tokenization collisions: {collided}/{n_total} "
              f"(distinct words mapping to the identical full subword-id sequence, "
              f"on {MODEL_ID}'s own tokenizer)", flush=True)
        print(f"embedding dim: {X.shape[1]}", flush=True)
    return X, y, classes, class_to_idx


def main():
    print(f"loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()
    weight = model.model.shared.weight
    print(f"embedding matrix shape: {tuple(weight.shape)}", flush=True)

    X, y, classes, class_to_idx = build_X_y(tokenizer, weight)
    num_classes = len(classes)
    real_acc = cross_val_accuracy(X, y, num_classes, N_FOLDS, SEED)
    print(f"\nREAL labels, {N_FOLDS}-fold CV accuracy: {real_acc:.4f}", flush=True)

    control_accs = []
    for i in range(N_CONTROL_SHUFFLES):
        y_shuffled = y[torch.randperm(n_total, generator=torch.Generator().manual_seed(1000 + i))]
        acc = cross_val_accuracy(X, y_shuffled, num_classes, N_FOLDS, SEED)
        control_accs.append(acc)
        print(f"  control shuffle {i:2d}: {acc:.4f}", flush=True)
    control_mean = sum(control_accs) / len(control_accs)
    control_std = (sum((a - control_mean) ** 2 for a in control_accs) / len(control_accs)) ** 0.5

    selectivity = real_acc - control_mean
    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL:              {MODEL_ID}", flush=True)
    print(f"REAL accuracy:      {real_acc:.4f}", flush=True)
    print(f"CONTROL accuracy:   {control_mean:.4f} (std {control_std:.4f}, n_shuffles={N_CONTROL_SHUFFLES})", flush=True)
    print(f"Majority baseline:  {majority_baseline:.4f}", flush=True)
    print(f"SELECTIVITY (real - control): {selectivity:.4f}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
