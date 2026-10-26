"""
order_probe.py

Probe #4: relative linear order. This is the ONLY probe among 2-4 that
isn't a pairwise classifier -- it's a single scalar functional

    s(w) = c^T w

trained so that comparing s(head) vs s(dep) alone predicts which word
comes first in the sentence (MarginRankingLoss). This is the same SHAPE
of claim as Hewitt & Manning's depth-as-norm probe: one linear read of a
word's own vector reproduces a global, sentence-wide ordering property.

Why this one, and not a pairwise order classifier: if this fits well, c
IS the steering direction directly -- pass model.direction() straight
into steering.AdditiveSteering. No larger weight matrix to extract a
direction from.

NOTE on defaults: MarginRankingLoss needs a higher lr / more epochs than
the CrossEntropyLoss-based probes (relation, morph) to converge -- its
gradient goes to zero for any pair that already clears the margin.
Confirmed empirically during development: lr=1e-4/epochs=15 reached only
~0.70 accuracy on a task where a perfect global ranking demonstrably
exists; lr=1e-3/epochs=40 reached ~1.00. Don't reuse the other probes'
lr=1e-4 default here.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from conllu_utils import Sentence

PRECEDES = "dep_before_head"
FOLLOWS = "dep_after_head"
LABEL_TO_ID = {PRECEDES: 0, FOLLOWS: 1}


@dataclass
class OrderExample:
    sent_idx: int
    head_idx: int
    dep_idx: int
    label: str
    signed_offset: int


def build_order_examples(sentences: Sequence[Sentence]) -> List[OrderExample]:
    examples = []
    for s_i, sent in enumerate(sentences):
        for tok in sent.tokens:
            if tok.head == 0:
                continue
            head_tok = sent.token_by_idx(tok.head)
            if head_tok is None:
                continue
            offset = tok.idx - head_tok.idx
            label = PRECEDES if offset < 0 else FOLLOWS
            examples.append(OrderExample(s_i, head_tok.idx, tok.idx, label, offset))
    return examples


def build_control_labels(
    *sentence_sets: Sequence[Sentence],
    label_source_examples: Sequence[OrderExample],
    seed: int = 0,
) -> Dict[Tuple[str, str], str]:
    """Build ONCE over every split you'll evaluate on -- see
    relation_probe.build_control_labels for why per-split construction is
    a bug, not just a style choice."""
    from collections import Counter
    label_counts = Counter(ex.label for ex in label_source_examples)
    labels, weights = zip(*label_counts.items())
    rng = random.Random(seed)

    pair_types = set()
    for sentences in sentence_sets:
        for sent in sentences:
            for tok in sent.tokens:
                if tok.head == 0:
                    continue
                head_tok = sent.token_by_idx(tok.head)
                if head_tok is None:
                    continue
                pair_types.add((head_tok.form.lower(), tok.form.lower()))

    return {pair: rng.choices(labels, weights=weights, k=1)[0] for pair in sorted(pair_types)}


class OrderProbe(nn.Module):
    """s(w) = c^T w + b. The row vector c is the candidate steering direction."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)

    def direction(self) -> torch.Tensor:
        """Trained c, L2-normalized -- feed straight into
        steering.AdditiveSteering(direction=...)."""
        with torch.no_grad():
            c = self.linear.weight.squeeze(0).clone()
            return c / (c.norm() + 1e-8)


def _label_to_y(label: str) -> int:
    return 1 if label == PRECEDES else -1


def train_order_probe(
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    examples: Sequence[OrderExample],
    hidden_dim: int,
    control_map: Optional[Dict[Tuple[str, str], str]] = None,
    lr: float = 1e-3,
    epochs: int = 40,
    batch_size: int = 64,
    grad_clip: float = 1.0,
    margin: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[OrderProbe, List[float]]:
    torch.manual_seed(seed)
    model = OrderProbe(hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MarginRankingLoss(margin=margin)

    idxs = list(range(len(examples)))
    losses = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(idxs)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            heads, deps, ys = [], [], []
            for i in batch_idx:
                ex = examples[i]
                heads.append(embeddings[ex.sent_idx][ex.head_idx - 1])
                deps.append(embeddings[ex.sent_idx][ex.dep_idx - 1])
                if control_map is not None:
                    sent = sentences[ex.sent_idx]
                    lbl = control_map[(sent.token_by_idx(ex.head_idx).form.lower(),
                                        sent.token_by_idx(ex.dep_idx).form.lower())]
                else:
                    lbl = ex.label
                ys.append(_label_to_y(lbl))

            h_head = torch.stack(heads).to(device)
            h_dep = torch.stack(deps).to(device)
            y = torch.tensor(ys, dtype=torch.float32, device=device)

            opt.zero_grad()
            loss = loss_fn(model(h_head), model(h_dep), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return model, losses


@torch.no_grad()
def evaluate_order_probe(
    model: OrderProbe,
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    examples: Sequence[OrderExample],
    control_map: Optional[Dict[Tuple[str, str], str]] = None,
    device: str = "cpu",
) -> float:
    model.eval()
    correct, total = 0, 0
    for ex in examples:
        h_head = embeddings[ex.sent_idx][ex.head_idx - 1].unsqueeze(0).to(device)
        h_dep = embeddings[ex.sent_idx][ex.dep_idx - 1].unsqueeze(0).to(device)
        pred = PRECEDES if model(h_head).item() > model(h_dep).item() else FOLLOWS

        if control_map is not None:
            sent = sentences[ex.sent_idx]
            gold = control_map[(sent.token_by_idx(ex.head_idx).form.lower(),
                                 sent.token_by_idx(ex.dep_idx).form.lower())]
        else:
            gold = ex.label
        correct += int(pred == gold)
        total += 1
    return correct / max(total, 1)


def relation_order_profile_with_support(
    sentences: Sequence[Sentence],
    examples: Sequence[OrderExample],
    relation_filter: Optional[str] = None,
    min_support: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Gold (no model needed) precedes-rate per relation, with support
    counts -- filter min_support before trusting extreme (0.0/1.0) values
    on rare relations. Use on a target-language treebank to get real
    calibration targets before steering."""
    by_sent: Dict[int, List[OrderExample]] = defaultdict(list)
    for ex in examples:
        by_sent[ex.sent_idx].append(ex)

    counts = defaultdict(lambda: [0, 0])
    for s_i, sent in enumerate(sentences):
        for ex in by_sent.get(s_i, []):
            dep = sent.token_by_idx(ex.dep_idx)
            rel = dep.deprel.split(":")[0]
            if relation_filter and rel != relation_filter:
                continue
            counts[rel][1] += 1
            if ex.label == PRECEDES:
                counts[rel][0] += 1

    return {
        rel: {"precedes_rate": n_pre / max(n_tot, 1), "support": n_tot}
        for rel, (n_pre, n_tot) in counts.items()
        if n_tot >= min_support
    }


def cross_lingual_steering_direction(model_a: OrderProbe, model_b: OrderProbe) -> torch.Tensor:
    """direction = c_B - c_A, normalized. Both models MUST have been
    trained on activations from the SAME underlying encoder/layer -- a
    direction from one model means nothing in a different model's space."""
    with torch.no_grad():
        d = model_b.linear.weight.squeeze(0) - model_a.linear.weight.squeeze(0)
        return d / (d.norm() + 1e-8)
