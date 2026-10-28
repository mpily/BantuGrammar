# -*- coding: utf-8 -*-
"""
dav_swa_eval_common.py

2026-10-28 cont. 6. Shared helpers for the dav_swa_* evaluation scripts --
factored out because `gtok`/apostrophe-handling had been copy-pasted into
three scripts (dav_swa_corpus_eval.py, dav_swa_silver_agreement_metrics.py,
dav_swa_vocab_tolerant_agreement_metrics.py) with the SAME logic; extending
one and not the others would silently desync them.

Direct user-relayed feedback (external review, 2026-10-28): "normalise
orthography (w' apostrophes, spelling variants) on both sides before any
comparison. Three of the 26 dictionary 'mismatches' were just apostrophes."
Confirmed directly before fixing: the observed pattern is a word-initial
"w" + apostrophe + vowel (w'undu/wundu, w'alimu/walimu, w'omi/womi) that
varies freely in this corpus/project's own data, not a phonemic contrast --
contrasted with "ng'" (e.g. "ng'ombe"), which appeared WITH the apostrophe
in every attestation checked this session and is a standard Bantu-
orthography digraph for a real phoneme, not spelling noise. Rather than
hand-list which apostrophes are "real" vs "noise" (no linguistic basis
available here to do that reliably), `normalize_ortho` strips ALL
apostrophes for COMPARISON purposes only -- never for displayed/stored
text. Risk, stated plainly: this can only ever INFLATE apparent matches
(two genuinely different words that happen to collide once apostrophes are
stripped), never deflate them -- a known, flaggable direction of error,
not a silent one.
"""
from __future__ import annotations

from extract_swahili_grammar import tokenize as _grammar_tokenize


def gtok(text: str) -> list[str]:
    """The SAME tokenizer kitaita_e2e_pipeline.decompose_sentence() itself
    uses to build Decomposition.tokens (extract_swahili_grammar.tokenize,
    lowercased) -- required for SlotRecord.token_idx to mean anything when
    applied to an independently-tokenized string."""
    return [t.lower() for t in _grammar_tokenize(text or "")]


def normalize_ortho(token: str) -> str:
    """Apostrophe-insensitive form of one token, for comparison only."""
    return (token or "").replace("'", "")


def tokens_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is b
    return normalize_ortho(a) == normalize_ortho(b)
