"""
conllu_utils.py

Shared CoNLL-U parsing for the grammar_probe pipeline. Used by:
  - probes.py              (Hewitt & Manning structural probe: distance/depth)
  - relation_probe.py       (labeled dependency relation probe)
  - morph_probe.py          (morphological feature probe)

A CoNLL-U sentence is parsed into a list of Token objects. Multiword tokens
(e.g. French "du" -> "de le") and empty nodes (id like "8.1") are skipped —
we only keep single, integer-indexed surface tokens, since those are what
get one embedding each from the encoder's tokenizer-to-word alignment.

Notes on alignment:
  - This module does NOT do subword alignment. That happens in the
    embedding-extraction step (extract_grammar.py / extract_relations_and_morph.py),
    which is expected to hand back one contextual vector per Token, already
    averaged/pooled over subwords per the paper's footnote 4 convention
    (align subword vectors to gold tokens, average).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class Token:
    idx: int  # 1-indexed position in the sentence, matches CoNLL-U ID column
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: Dict[str, str]  # e.g. {"Number": "Sing", "Case": "Nom", ...}
    head: int  # 1-indexed idx of syntactic head; 0 means root
    deprel: str  # e.g. "nsubj", "obj", "root", "det", "amod", ...
    # 2026-10-09 (cont. 16): the ENHANCED dependencies column (CoNLL-U col 9,
    # "_" when absent/not annotated) -- list of (head_idx, rel) pairs beyond
    # the single basic tree edge above. Added specifically to recover control-
    # structure annotations like "nsubj:xsubj" (e.g. English UD EWT marks
    # ~3,020 real controller-of-embedded-predicate edges this way) that the
    # basic HEAD/DEPREL columns never carry -- Swahili's own treebank has
    # only 5/226 real nsubj/csubj edges for its structurally analogous '-ki-'
    # participial construction (docs/status.md 2026-10-09 cont. 9), so this
    # is the concrete data source for testing cross-lingual transfer of a
    # "controls" relation into that gap. A head_idx pointing to an empty
    # node (e.g. "8.1") is skipped, not crashed on -- this project only keeps
    # single, integer-indexed surface tokens elsewhere in this file too.
    deps: List["tuple[int, str]"] = field(default_factory=list)


@dataclass
class Sentence:
    sent_id: Optional[str]
    text: Optional[str]
    tokens: List[Token] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)

    def token_by_idx(self, idx: int) -> Optional[Token]:
        """idx is 1-indexed CoNLL-U style; idx==0 means the artificial root."""
        if idx == 0:
            return None
        for t in self.tokens:
            if t.idx == idx:
                return t
        return None

    def head_of(self, token: Token) -> Optional[Token]:
        return self.token_by_idx(token.head)

    def dependents_of(self, token: Token) -> List[Token]:
        return [t for t in self.tokens if t.head == token.idx]

    def root(self) -> Optional[Token]:
        for t in self.tokens:
            if t.deprel == "root" or t.head == 0:
                return t
        return None


_FEATS_RE = re.compile(r"([^=|]+)=([^|]+)")


def _parse_feats(raw: str) -> Dict[str, str]:
    if raw in ("_", "", None):
        return {}
    out = {}
    for m in _FEATS_RE.finditer(raw):
        key, val = m.group(1), m.group(2)
        # UD sometimes stacks multiple values with commas (rare); keep first
        # for a clean single-label classification target.
        val = val.split(",")[0]
        out[key] = val
    return out


def _is_integer_id(field_val: str) -> bool:
    return bool(re.fullmatch(r"\d+", field_val))


def _parse_deps(raw: str) -> List["tuple[int, str]"]:
    """CoNLL-U enhanced DEPS column: 'head:rel|head:rel|...', '_' if absent.
    rel itself can contain colons (e.g. 'nsubj:xsubj'), so split on the
    FIRST colon only. A head pointing to an empty node ('8.1') or otherwise
    non-integer is skipped, not crashed on."""
    if raw in ("_", "", None):
        return []
    out = []
    for piece in raw.split("|"):
        if ":" not in piece:
            continue
        head_str, rel = piece.split(":", 1)
        if _is_integer_id(head_str):
            out.append((int(head_str), rel))
    return out


def parse_conllu(text: str) -> List[Sentence]:
    """Parse a full CoNLL-U file's contents into a list of Sentence objects."""
    sentences: List[Sentence] = []
    sent_id = None
    sent_text = None
    tokens: List[Token] = []

    def _flush():
        nonlocal sent_id, sent_text, tokens
        if tokens:
            sentences.append(Sentence(sent_id=sent_id, text=sent_text, tokens=tokens))
        sent_id, sent_text, tokens = None, None, []

    for raw_line in io.StringIO(text):
        line = raw_line.rstrip("\n")
        if not line.strip():
            _flush()
            continue
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text"):
                sent_text = line.split("=", 1)[1].strip()
            continue

        cols = line.split("\t")
        if len(cols) < 10:
            continue
        tok_id = cols[0]
        if not _is_integer_id(tok_id):
            # skip multiword-token ranges ("1-2") and empty nodes ("8.1")
            continue

        idx = int(tok_id)
        form, lemma, upos, xpos, feats_raw, head_raw, deprel, deps_raw = (
            cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8]
        )
        try:
            head = int(head_raw)
        except ValueError:
            # e.g. head == '_' for some malformed/ellided rows; treat as root-less
            head = 0

        tokens.append(
            Token(
                idx=idx,
                form=form,
                lemma=lemma,
                upos=upos,
                xpos=xpos,
                feats=_parse_feats(feats_raw),
                head=head,
                deprel=deprel,
                deps=_parse_deps(deps_raw),
            )
        )

    _flush()
    return sentences


def load_conllu_file(path: str) -> List[Sentence]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_conllu(f.read())


def iter_conllu_file(path: str) -> Iterator[Sentence]:
    """Streaming variant for large treebanks."""
    for sent in load_conllu_file(path):
        yield sent


def tree_distance_matrix(sentence: Sentence) -> Dict[int, Dict[int, int]]:
    """
    Unweighted shortest-path (= edge-count) distance between every pair of
    tokens in the dependency tree, via BFS from each node. O(n^2), fine for
    sentence-length graphs.

    Keyed by CoNLL-U idx (1-indexed), matching Token.idx.
    """
    idxs = [t.idx for t in sentence.tokens]
    adj: Dict[int, List[int]] = {i: [] for i in idxs}
    for t in sentence.tokens:
        if t.head != 0 and t.head in adj:
            adj[t.idx].append(t.head)
            adj[t.head].append(t.idx)

    dist: Dict[int, Dict[int, int]] = {}
    for src in idxs:
        d = {src: 0}
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        nxt.append(v)
            frontier = nxt
        dist[src] = d
    return dist


def tree_depth(sentence: Sentence) -> Dict[int, int]:
    """Depth (edges to root) of every token, keyed by idx."""
    depth: Dict[int, int] = {}

    def _depth_of(t: Token) -> int:
        if t.idx in depth:
            return depth[t.idx]
        if t.head == 0:
            depth[t.idx] = 0
            return 0
        head_tok = sentence.token_by_idx(t.head)
        if head_tok is None:
            depth[t.idx] = 0
            return 0
        d = 1 + _depth_of(head_tok)
        depth[t.idx] = d
        return d

    for t in sentence.tokens:
        _depth_of(t)
    return depth
