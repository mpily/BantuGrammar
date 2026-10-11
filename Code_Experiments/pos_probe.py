# -*- coding: utf-8 -*-
"""
pos_probe.py

2026-10-09. Direct user directive, after "mdogo" (a real Swahili
adjective) was silently swallowed as a noun because this project's
candidate detection only ever recognized the 2 adjectives it happens to
have a hardcoded Kitaita concord table for (ADJ_ROOTS = kubwa/baya):
"Do #2, build the POS probe properly." A real, NLLB-representation-based
part-of-speech classifier -- NOUN/VERB/ADJ/etc. identified from the
encoder's own hidden states, not a closed lookup list -- so a token's
candidacy for noun/verb/adjective role doesn't depend on this project
having pre-enumerated every Swahili word of that kind.

Per-TOKEN classification (unlike relation_probe.py's RelationProbe,
which classifies a RELATION between a head/dependent PAIR) -- a single
pooled hidden vector in, one UPOS label out. Linear only, no hidden
layer, same "no additional learned nonlinearity" spirit as RelationProbe,
and the SAME Hewitt & Liang (2019) control task this project requires
for every probe: a fixed random label per WORD-FORM TYPE (not per
token instance), so real_accuracy - control_accuracy is the number to
trust, not raw accuracy alone.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from conllu_utils import Sentence

# UPOS tags actually useful to this pipeline; everything else collapses to
# OTHER (determiners, conjunctions, particles, punctuation, ...) -- this
# project only needs to distinguish noun-like/verb-like/adjective-like
# tokens from everything else, not the full 17-tag UD inventory.
CORE_UPOS = ["NOUN", "PROPN", "VERB", "ADJ", "PRON"]
OTHER_LABEL = "OTHER"


def canonicalize_upos(upos: str) -> str:
    return upos if upos in CORE_UPOS else OTHER_LABEL


@dataclass
class PosExample:
    sent_idx: int
    tok_idx: int  # 1-indexed, matches CoNLL-U ID column (same convention as RelationExample)
    label: str


def build_pos_examples(sentences: Sequence[Sentence]) -> List[PosExample]:
    examples = []
    for s_i, sent in enumerate(sentences):
        for tok in sent.tokens:
            examples.append(PosExample(s_i, tok.idx, canonicalize_upos(tok.upos)))
    return examples


def build_control_labels(
    *sentence_sets: Sequence[Sentence],
    label_source_examples: Sequence[PosExample],
    seed: int = 0,
) -> Dict[str, str]:
    """Same discipline as relation_probe.build_control_labels: build ONCE
    over every split that will be evaluated, keyed by WORD-FORM TYPE (not
    token instance) so the same word gets a consistent control label
    everywhere it appears."""
    label_counts = Counter(ex.label for ex in label_source_examples)
    labels, weights = zip(*label_counts.items())
    rng = random.Random(seed)

    word_types = set()
    for sentences in sentence_sets:
        for sent in sentences:
            for tok in sent.tokens:
                word_types.add(tok.form.lower())

    return {w: rng.choices(labels, weights=weights, k=1)[0] for w in sorted(word_types)}


class PosProbe(nn.Module):
    """Pure linear map: pooled hidden vector -> UPOS-class logits."""

    def __init__(self, hidden_dim: int, n_labels: int):
        super().__init__()
        self.net = nn.Linear(hidden_dim, n_labels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def make_label_vocab(examples: Sequence[PosExample]) -> Tuple[List[str], Dict[str, int]]:
    labels = sorted(set(ex.label for ex in examples))
    return labels, {l: i for i, l in enumerate(labels)}


def train_pos_probe(
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    examples: Sequence[PosExample],
    label_to_id: Dict[str, int],
    hidden_dim: int,
    control_map: Optional[Dict[str, str]] = None,
    lr: float = 1e-4,
    epochs: int = 15,
    batch_size: int = 64,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[PosProbe, List[float]]:
    torch.manual_seed(seed)
    model = PosProbe(hidden_dim, len(label_to_id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    idxs = list(range(len(examples)))
    losses = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            hs, labels = [], []
            for i in batch_idx:
                ex = examples[i]
                hs.append(embeddings[ex.sent_idx][ex.tok_idx - 1])
                if control_map is not None:
                    word = sentences[ex.sent_idx].token_by_idx(ex.tok_idx).form.lower()
                    labels.append(label_to_id[control_map[word]])
                else:
                    labels.append(label_to_id[ex.label])
            h = torch.stack(hs).to(device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = loss_fn(model(h), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(loss.item())
    return model, losses


@torch.no_grad()
def evaluate_pos_probe(model: PosProbe, sentences, embeddings, examples: Sequence[PosExample],
                        label_to_id: Dict[str, int], control_map: Optional[Dict[str, str]] = None,
                        device: str = "cpu"):
    model.eval()
    id_to_label = {v: k for k, v in label_to_id.items()}
    correct, total = 0, 0
    per_label_correct, per_label_total = Counter(), Counter()
    for ex in examples:
        h = embeddings[ex.sent_idx][ex.tok_idx - 1].unsqueeze(0).to(device)
        pred_id = model(h).argmax(dim=-1).item()
        pred = id_to_label[pred_id]
        if control_map is not None:
            word = sentences[ex.sent_idx].token_by_idx(ex.tok_idx).form.lower()
            true_label = control_map[word]
        else:
            true_label = ex.label
        total += 1
        per_label_total[true_label] += 1
        if pred == true_label:
            correct += 1
            per_label_correct[true_label] += 1
    accuracy = correct / total if total else float("nan")
    per_label_acc = {l: per_label_correct[l] / per_label_total[l] for l in per_label_total}
    return accuracy, per_label_acc
