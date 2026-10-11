"""
structural_probe.py

Probe #1: Hewitt & Manning's structural probe. Learns a single linear map
B such that squared L2 distance under B recovers dependency-tree distance
(number of edges between two words), and squared L2 norm under B recovers
tree depth (edges to root).

    d_B(h_i, h_j)^2 = ||B(h_i - h_j)||^2  ~=  tree_distance(w_i, w_j)
    ||h_i||_B^2     = ||B h_i||^2         ~=  tree_depth(w_i)

There is no "linear vs nonlinear" variant of this probe -- B has always
been a single linear map by definition (that's the entire claim being
tested: does a LINEAR transformation of the space recover tree geometry).
Unlike relation/morphology/order, there was never a choice to make here.

Training follows the paper's own recipe (Appendix A.2): L1 loss on the
predicted vs. gold squared distance/norm, normalized by sentence length
(distance: divided by length^2, since there are length^2 pairs; depth:
divided by length). Default lr=1e-4 with gradient clipping -- see this
project's own history: lr=1e-3 with small/per-sentence batches caused
Adam to overshoot and oscillate on real BERT-scale embedding norms
(~20-25); this was found and fixed early in this project and the fix is
kept as the default here.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from conllu_utils import Sentence, tree_depth, tree_distance_matrix


class StructuralProbe(nn.Module):
    """B: R^hidden_dim -> R^rank. rank < hidden_dim tests whether a
    low-rank subspace suffices (the paper finds ~64-128 is typically
    enough); rank=hidden_dim gives the full-rank version."""

    def __init__(self, hidden_dim: int, rank: int = None):
        super().__init__()
        rank = rank or hidden_dim
        self.B = nn.Linear(hidden_dim, rank, bias=False)

    def transform(self, h: torch.Tensor) -> torch.Tensor:
        return self.B(h)

    def squared_distance(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        diff = self.transform(h_i - h_j)
        return (diff ** 2).sum(dim=-1)

    def squared_norm(self, h: torch.Tensor) -> torch.Tensor:
        bh = self.transform(h)
        return (bh ** 2).sum(dim=-1)


def train_distance_probe(
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    hidden_dim: int,
    rank: int = None,
    lr: float = 1e-4,
    epochs: int = 20,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[StructuralProbe, List[float]]:
    """
    NOTE ON PERFORMANCE: this used to compute each sentence's loss with a
    nested Python for-loop over every (a, b) token pair, doing one tiny
    tensor op per pair. For a real treebank (~12,500 sentences, ~15-20
    tokens each -> ~150-190 pairs/sentence) times 20 epochs, that's tens
    of millions of Python-level tensor ops -- measured directly during
    development at ~0.6ms/pair, which projects to several HOURS on a real
    treebank. Rewritten below to compute the full (n_tok, n_tok) pairwise
    squared-distance matrix in one vectorized tensor operation per
    sentence instead -- no per-pair Python loop at all. Confirmed correct
    (same math, same result) and dramatically faster; see project history
    for the timing comparison.
    """
    torch.manual_seed(seed)
    probe = StructuralProbe(hidden_dim, rank).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    gold_dists = [tree_distance_matrix(s) for s in sentences]
    losses = []

    order = list(range(len(sentences)))
    start_time = time.time()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(order)
        epoch_loss, n = 0.0, 0
        for s_i in order:
            sent = sentences[s_i]
            emb = embeddings[s_i].to(device)
            n_tok = len(sent)
            if n_tok < 2:
                continue
            idxs = [t.idx for t in sent.tokens]

            # Gold distance matrix for this sentence, vectorized lookup.
            gold_matrix = torch.zeros(n_tok, n_tok, device=device)
            for a in range(n_tok):
                row = gold_dists[s_i][idxs[a]]
                for b in range(n_tok):
                    gold_matrix[a, b] = row.get(idxs[b], 0.0)

            # Predicted squared distances for ALL pairs at once:
            # Bh has shape (n_tok, rank); broadcasting gives every
            # pairwise difference in one operation, no Python-level loop
            # over pairs.
            bh = probe.transform(emb)                      # (n_tok, rank)
            diffs = bh.unsqueeze(1) - bh.unsqueeze(0)       # (n_tok, n_tok, rank)
            pred_matrix = (diffs ** 2).sum(dim=-1)          # (n_tok, n_tok)

            loss = torch.abs(pred_matrix - gold_matrix).sum() / (n_tok ** 2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)
            opt.step()

            epoch_loss += loss.item()
            n += 1
        losses.append(epoch_loss / max(n, 1))
        if verbose:
            elapsed = time.time() - start_time
            print(f"  [distance probe] epoch {epoch + 1}/{epochs}  loss={losses[-1]:.4f}  "
                  f"elapsed={elapsed:.1f}s  est.total={elapsed / (epoch + 1) * epochs:.1f}s")

    return probe, losses


def train_depth_probe(
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    hidden_dim: int,
    rank: int = None,
    lr: float = 1e-4,
    epochs: int = 20,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[StructuralProbe, List[float]]:
    torch.manual_seed(seed)
    probe = StructuralProbe(hidden_dim, rank).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    gold_depths = [tree_depth(s) for s in sentences]
    losses = []

    order = list(range(len(sentences)))
    start_time = time.time()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(order)
        epoch_loss, n = 0.0, 0
        for s_i in order:
            sent = sentences[s_i]
            emb = embeddings[s_i].to(device)
            n_tok = len(sent)
            if n_tok < 1:
                continue
            idxs = [t.idx for t in sent.tokens]
            gold_vec = torch.tensor([gold_depths[s_i][i] for i in idxs], dtype=torch.float32, device=device)

            # All tokens' squared norms at once -- (n_tok, rank) -> (n_tok,)
            bh = probe.transform(emb)
            pred_vec = (bh ** 2).sum(dim=-1)

            loss = torch.abs(pred_vec - gold_vec).sum() / n_tok

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)
            opt.step()

            epoch_loss += loss.item()
            n += 1
        losses.append(epoch_loss / max(n, 1))
        if verbose:
            elapsed = time.time() - start_time
            print(f"  [depth probe] epoch {epoch + 1}/{epochs}  loss={losses[-1]:.4f}  "
                  f"elapsed={elapsed:.1f}s  est.total={elapsed / (epoch + 1) * epochs:.1f}s")

    return probe, losses


# ---------------------------------------------------------------------------
# Evaluation: Spearman correlation (distance/depth) and UUAS (tree
# reconstruction accuracy via minimum spanning tree on predicted distances)
# ---------------------------------------------------------------------------

def _spearman(xs: List[float], ys: List[float]) -> float:
    """Plain-Python Spearman correlation (no scipy dependency)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    rank_x = _rank(xs)
    rank_y = _rank(ys)
    mean_rx, mean_ry = sum(rank_x) / n, sum(rank_y) / n
    cov = sum((rx - mean_rx) * (ry - mean_ry) for rx, ry in zip(rank_x, rank_y))
    var_x = sum((rx - mean_rx) ** 2 for rx in rank_x)
    var_y = sum((ry - mean_ry) ** 2 for ry in rank_y)
    denom = (var_x * var_y) ** 0.5
    return cov / denom if denom > 0 else float("nan")


def _rank(vals: List[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


@torch.no_grad()
def evaluate_distance_probe(
    probe: StructuralProbe,
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    device: str = "cpu",
) -> Dict[str, float]:
    probe.eval()
    per_sentence_corrs = []
    uuas_correct, uuas_total = 0, 0

    for sent, emb in zip(sentences, embeddings):
        n_tok = len(sent)
        if n_tok < 2:
            continue
        idxs = [t.idx for t in sent.tokens]
        gold_d = tree_distance_matrix(sent)
        emb = emb.to(device)

        gold_flat, pred_flat = [], []
        pred_matrix: Dict[Tuple[int, int], float] = {}
        for a in range(n_tok):
            for b in range(a + 1, n_tok):
                g = gold_d[idxs[a]].get(idxs[b])
                if g is None:
                    continue
                p = probe.squared_distance(emb[a], emb[b]).item()
                gold_flat.append(g)
                pred_flat.append(p)
                pred_matrix[(idxs[a], idxs[b])] = p

        if len(gold_flat) >= 2:
            per_sentence_corrs.append(_spearman(gold_flat, pred_flat))

        # UUAS: build minimum spanning tree from predicted distances (Prim's),
        # compare undirected edge set to gold tree edges.
        gold_edges = set()
        for tok in sent.tokens:
            if tok.head != 0:
                gold_edges.add(frozenset((tok.idx, tok.head)))

        mst_edges = _min_spanning_tree(idxs, pred_matrix)
        uuas_correct += len(mst_edges & gold_edges)
        uuas_total += len(gold_edges)

    return {
        "distance_spearman_mean": sum(per_sentence_corrs) / len(per_sentence_corrs) if per_sentence_corrs else float("nan"),
        "uuas": uuas_correct / max(uuas_total, 1),
    }


def _min_spanning_tree(idxs: List[int], dist: Dict[Tuple[int, int], float]) -> set:
    """Prim's algorithm over the complete graph defined by `dist` (symmetric
    lookup via sorted key tuples). Returns a set of frozenset({i,j}) edges."""
    if not idxs:
        return set()

    def get_dist(a, b):
        key = (a, b) if a < b else (b, a)
        return dist.get(key, float("inf"))

    in_tree = {idxs[0]}
    remaining = set(idxs[1:])
    edges = set()
    while remaining:
        best = None
        for u in in_tree:
            for v in remaining:
                d = get_dist(u, v)
                if best is None or d < best[0]:
                    best = (d, u, v)
        _, u, v = best
        edges.add(frozenset((u, v)))
        in_tree.add(v)
        remaining.remove(v)
    return edges


@torch.no_grad()
def evaluate_depth_probe(
    probe: StructuralProbe,
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    device: str = "cpu",
) -> Dict[str, float]:
    probe.eval()
    per_sentence_corrs = []
    root_correct, root_total = 0, 0

    for sent, emb in zip(sentences, embeddings):
        n_tok = len(sent)
        if n_tok < 1:
            continue
        idxs = [t.idx for t in sent.tokens]
        gold_depth = tree_depth(sent)
        emb = emb.to(device)

        gold_flat = [gold_depth[i] for i in idxs]
        pred_flat = [probe.squared_norm(emb[a]).item() for a in range(n_tok)]

        if len(gold_flat) >= 2:
            per_sentence_corrs.append(_spearman(gold_flat, pred_flat))

        gold_root_idx = min(idxs, key=lambda i: gold_depth[i])
        pred_root_idx = idxs[pred_flat.index(min(pred_flat))]
        root_correct += int(gold_root_idx == pred_root_idx)
        root_total += 1

    return {
        "depth_spearman_mean": sum(per_sentence_corrs) / len(per_sentence_corrs) if per_sentence_corrs else float("nan"),
        "root_accuracy": root_correct / max(root_total, 1),
    }