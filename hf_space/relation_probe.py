"""
relation_probe.py

Probe #2: labeled dependency relation (nsubj, obj, det, amod, ...) between
a head and dependent. LINEAR ONLY -- a single linear map over the pair
feature [h_head; h_dep; h_head-h_dep; h_head*h_dep], no hidden layer.

Note the "linear" here is linear ON TOP OF that fixed feature map -- the
elementwise product term is itself a fixed nonlinearity in the raw
embeddings, so this is "no additional LEARNED nonlinearity," not "linear
in the raw BERT vectors." Worth stating explicitly in any writeup.

Includes the Hewitt & Liang (2019) control task: a fixed random label per
(head_form, dep_form) word-pair TYPE, sampled proportional to the
empirical label distribution. selectivity = real_accuracy - control_accuracy
is the number to trust, not raw accuracy alone.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from conllu_utils import Sentence

CORE_RELATIONS = [
    "nsubj", "obj", "iobj", "obl", "root", "amod", "advmod",
    "det", "aux", "case", "nmod", "conj", "cc", "mark", "cop",
    "punct", "compound", "acl", "xcomp", "ccomp", "advcl",
]
OTHER_LABEL = "other"


def canonicalize_deprel(deprel: str) -> str:
    base = deprel.split(":")[0]
    return base if base in CORE_RELATIONS else OTHER_LABEL


@dataclass
class RelationExample:
    sent_idx: int
    head_idx: int
    dep_idx: int
    label: str


def build_relation_examples(sentences: Sequence[Sentence]) -> List[RelationExample]:
    examples = []
    for s_i, sent in enumerate(sentences):
        for tok in sent.tokens:
            if tok.head == 0:
                continue
            head_tok = sent.token_by_idx(tok.head)
            if head_tok is None:
                continue
            examples.append(RelationExample(s_i, head_tok.idx, tok.idx, canonicalize_deprel(tok.deprel)))
    return examples


def build_relation_examples_with_negatives(sentences: Sequence[Sentence], max_negatives: int = 3,
                                            seed: int = 0) -> List[RelationExample]:
    """2026-10-09, direct user directive, after confirming directly (logits:
    5.91 vs 3.06, not a close call) that the nsubj/obj probe repeats the
    documented Wanaharakati/Kudai failure mode on a NEW real sentence
    ("Mbwa mkubwa aliona Kijana mdogo akitembea.", the dog's own, genuinely
    strong subject-likeness for the MAIN clause wins the nsubj comparison
    for an unrelated embedded verb "akitembea" too, where "kijana" -- the
    one actually walking -- should win instead).

    Root cause, found by reading build_relation_examples() before changing
    anything: it only ever builds POSITIVE examples from REAL gold edges --
    the probe has NEVER seen a labeled (verb, unrelated-noun) NON-edge
    during training at all, unlike the amod/poss/acl probes (nllb_concord_
    rewrite_pipeline.extract_direct_examples), which already sample
    max_negatives distractors per positive example for exactly this reason.
    This function brings the SAME negative-sampling discipline to the main
    relation probe: for every VERB token, samples up to max_negatives OTHER
    noun-like tokens in the SAME sentence that are NOT its real dependent,
    labeled OTHER_LABEL -- teaching the model "subject-like in general" is
    not the same as "THIS verb's subject", directly targeting the failure
    mode above rather than patching it with a position-based heuristic
    (which the margin here -- 5.91 vs 3.06, not close -- would not even
    catch)."""
    rng = random.Random(seed)
    examples = list(build_relation_examples(sentences))
    for s_i, sent in enumerate(sentences):
        noun_toks = [t for t in sent.tokens if t.upos in ("NOUN", "PROPN", "PRON")]
        for verb_tok in (t for t in sent.tokens if t.upos == "VERB"):
            real_dep_idxs = {t.idx for t in sent.tokens if t.head == verb_tok.idx}
            candidates = [n for n in noun_toks if n.idx not in real_dep_idxs and n.idx != verb_tok.idx]
            if not candidates:
                continue
            rng.shuffle(candidates)
            for n in candidates[:max_negatives]:
                examples.append(RelationExample(s_i, verb_tok.idx, n.idx, OTHER_LABEL))
    return examples


def build_control_labels(
    *sentence_sets: Sequence[Sentence],
    label_source_examples: Sequence[RelationExample],
    seed: int = 0,
) -> Dict[Tuple[str, str], str]:
    """Build ONCE over every split you'll evaluate on (pass all splits as
    positional args) -- calling this separately per split lets the same
    word-pair TYPE get inconsistent control labels across splits, which
    breaks the control task's premise. See project history for the bug
    this fixes."""
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


class RelationProbe(nn.Module):
    """Pure linear map: [h_head; h_dep; h_head-h_dep; h_head*h_dep] -> logits."""

    def __init__(self, hidden_dim: int, n_labels: int):
        super().__init__()
        self.net = nn.Linear(hidden_dim * 4, n_labels)

    def forward(self, h_head: torch.Tensor, h_dep: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([h_head, h_dep, h_head - h_dep, h_head * h_dep], dim=-1)
        return self.net(feats)


def make_label_vocab(examples: Sequence[RelationExample]) -> Tuple[List[str], Dict[str, int]]:
    labels = sorted(set(ex.label for ex in examples))
    return labels, {l: i for i, l in enumerate(labels)}


def train_relation_probe(
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    examples: Sequence[RelationExample],
    label_to_id: Dict[str, int],
    hidden_dim: int,
    control_map: Optional[Dict[Tuple[str, str], str]] = None,
    lr: float = 1e-4,
    epochs: int = 15,
    batch_size: int = 64,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[RelationProbe, List[float]]:
    torch.manual_seed(seed)
    model = RelationProbe(hidden_dim, len(label_to_id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    idxs = list(range(len(examples)))
    losses = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(idxs)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            heads, deps, labels = [], [], []
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
                labels.append(label_to_id[lbl])

            h_head = torch.stack(heads).to(device)
            h_dep = torch.stack(deps).to(device)
            y = torch.tensor(labels, dtype=torch.long, device=device)

            opt.zero_grad()
            loss = loss_fn(model(h_head, h_dep), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return model, losses


@torch.no_grad()
def evaluate_relation_probe(
    model: RelationProbe,
    sentences: Sequence[Sentence],
    embeddings: Sequence[torch.Tensor],
    examples: Sequence[RelationExample],
    label_to_id: Dict[str, int],
    control_map: Optional[Dict[Tuple[str, str], str]] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    """Returns accuracy, macro-F1, and per-class precision/recall/F1."""
    model.eval()
    id_to_label = {v: k for k, v in label_to_id.items()}
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    correct, total = 0, 0

    for ex in examples:
        h_head = embeddings[ex.sent_idx][ex.head_idx - 1].unsqueeze(0).to(device)
        h_dep = embeddings[ex.sent_idx][ex.dep_idx - 1].unsqueeze(0).to(device)
        pred = id_to_label[model(h_head, h_dep).argmax(dim=-1).item()]

        if control_map is not None:
            sent = sentences[ex.sent_idx]
            gold = control_map[(sent.token_by_idx(ex.head_idx).form.lower(),
                                 sent.token_by_idx(ex.dep_idx).form.lower())]
        else:
            gold = ex.label

        support[gold] += 1
        total += 1
        if pred == gold:
            tp[gold] += 1
            correct += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    f1s = []
    per_class = {}
    for label in sorted(support):
        p = tp[label] / max(tp[label] + fp[label], 1)
        r = tp[label] / max(tp[label] + fn[label], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        per_class[label] = {"precision": p, "recall": r, "f1": f1, "support": support[label]}
        f1s.append(f1)

    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": sum(f1s) / max(len(f1s), 1),
        "per_class": per_class,
    }


def find_argmax_candidates(
    verb_idx: int,
    sentence: Sentence,
    d_b: Dict[int, Dict[int, int]],
    candidate_upos: Sequence[str] = ("NOUN", "PROPN", "PRON"),
    top_k: int = 3,
) -> List[int]:
    """Use the structural probe's distance predictions (d_b) to find
    candidate head-dependent pairs when gold heads aren't available --
    the real-application bridge from probe #1 to probe #2."""
    candidates = [t.idx for t in sentence.tokens if t.upos in candidate_upos and t.idx != verb_idx]
    candidates.sort(key=lambda i: (d_b[verb_idx].get(i, float("inf")), abs(i - verb_idx)))
    return candidates[:top_k]
