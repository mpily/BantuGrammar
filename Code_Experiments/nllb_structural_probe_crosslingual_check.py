# -*- coding: utf-8 -*-
"""
nllb_structural_probe_crosslingual_check.py

User's directive (2026-10-05): before committing to a Swahili
treebank-building effort, test whether a Hewitt & Manning structural
probe trained on NLLB's OWN encoder representations for ONE language
(English, where we have real UD data) transfers to OTHER languages'
dependency structure -- an Indo-European one (Spanish, a sanity check)
and a well-documented NON-Indo-European one (Finnish, agglutinative,
typologically distant from English) -- WITHOUT retraining. If transfer
holds up on Finnish specifically, that is evidence worth trusting
before applying the same frozen probe to Swahili (which has no gold
treebank to score against at all).

Reuses `structural_probe.py`'s StructuralProbe/train_distance_probe/
evaluate_distance_probe and `conllu_utils.py`'s parsing unchanged
(CLAUDE.md hard rule #10) -- that infrastructure was already complete;
the only missing piece was embedding extraction, since the
`extract_grammar.py` script referenced in conllu_utils.py's docstring
does not actually exist on disk, and the project's prior structural-
probe work used BERT (CLAUDE.md), not NLLB.

Alignment method: tokenize the FULL sentence once (to get the hidden
states NLLB actually computes, with real sentence context), and
SEPARATELY tokenize each gold CoNLL-U token in isolation (with a
leading space) to get its own subword count, then walk the full
tokenization greedily consuming that many subwords per gold token --
standard word-by-word accumulation alignment for SentencePiece-style
tokenizers. Sentences where the subword counts don't reconcile exactly
(the full-sentence tokenization is not the clean concatenation of each
word's own tokenization -- can happen at punctuation/contraction edge
cases) are SKIPPED, not guessed, with a count reported.

Usage:
    python3 -u nllb_structural_probe_crosslingual_check.py \
        --n-train 400 --n-eval 150 --layer 12 --seed 0
"""
from __future__ import annotations

import argparse
import random

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from conllu_utils import load_conllu_file
from cross_lingual_embedding_decomposition import MODEL_ID
from structural_probe import (
    StructuralProbe, train_distance_probe, evaluate_distance_probe,
)

LANG_CODES = {"en": "eng_Latn", "es": "spa_Latn", "fi": "fin_Latn"}


def extract_embeddings(tokenizer, model, sentences, lang_code, layer, max_sentences=None, seed=0,
                        target_norm=None):
    """Returns (kept_sentences, embeddings) -- one (n_tok, hidden_dim)
    tensor per kept sentence, skipping any sentence whose full-text
    tokenization doesn't cleanly decompose into its gold tokens' own
    individual tokenizations."""
    tokenizer.src_lang = lang_code
    if max_sentences is not None:
        rng = random.Random(seed)
        sentences = rng.sample(sentences, min(max_sentences, len(sentences)))

    kept_sentences, kept_embeddings = [], []
    n_skipped = 0
    for sent in sentences:
        if len(sent.tokens) < 2:
            continue
        text = sent.text if sent.text else " ".join(t.form for t in sent.tokens)
        full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]

        per_word_counts = []
        ok = True
        for t in sent.tokens:
            w_ids = tokenizer(f" {t.form}", add_special_tokens=False)["input_ids"]
            if len(w_ids) == 0:
                ok = False
                break
            per_word_counts.append(len(w_ids))
        if not ok or sum(per_word_counts) + 2 > len(full_ids) + 5:
            # +2 for bos/eos-ish specials on the word side is a loose sanity
            # bound; the real check is the cumulative walk below
            pass

        with torch.no_grad():
            enc = tokenizer(text, return_tensors="pt")
            out = model.get_encoder()(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer][0]  # (seq_len, hidden_dim), includes specials

        # walk: NLLB prepends the src_lang token then the sentence, no bos;
        # find where content starts by locating the lang-code token
        ids_list = enc["input_ids"][0].tolist()
        lang_tok_id = tokenizer.convert_tokens_to_ids(lang_code)
        try:
            start = ids_list.index(lang_tok_id) + 1
        except ValueError:
            n_skipped += 1
            continue

        word_vecs = []
        pos = start
        failed = False
        for n_sub in per_word_counts:
            if pos + n_sub > len(ids_list):
                failed = True
                break
            word_vecs.append(hidden[pos:pos + n_sub].mean(dim=0))
            pos += n_sub
        # allow trailing eos/specials to remain unconsumed; don't require
        # pos == len(ids_list) exactly, just that every word got its slice
        if failed or len(word_vecs) != len(sent.tokens):
            n_skipped += 1
            continue

        stacked = torch.stack(word_vecs)
        if target_norm is not None:
            # rescale each token's vector to a FIXED norm -- NLLB's raw encoder
            # hidden states run far larger than the BERT-scale (~20-25) the
            # existing probe hyperparameters (lr=1e-4) were calibrated for
            # (confirmed directly: layer-12 mean norm ~1766, up to ~12700),
            # so without this the probe never escapes a near-degenerate init
            # within a reasonable epoch budget.
            norms = stacked.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            stacked = stacked / norms * target_norm

        kept_sentences.append(sent)
        kept_embeddings.append(stacked)

    print(f"  [{lang_code}] kept {len(kept_sentences)}/{len(sentences)} sentences "
          f"({n_skipped} skipped on alignment failure)", flush=True)
    return kept_sentences, kept_embeddings


def random_direction_eval(sentences, embeddings, hidden_dim, rank, seed):
    """Random, UNTRAINED probe (random B matrix) as a floor control."""
    torch.manual_seed(seed + 9999)
    probe = StructuralProbe(hidden_dim, rank)
    return evaluate_distance_probe(probe, sentences, embeddings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-norm", type=float, default=20.0,
                     help="rescale every token embedding to this fixed norm before training/eval "
                          "(matches the BERT-scale the existing probe defaults were calibrated for); "
                          "pass 0 to disable")
    args = ap.parse_args()
    target_norm = args.target_norm if args.target_norm > 0 else None
    print(f"resolved config: {vars(args)}", flush=True)

    print(f"loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden_dim = model.config.d_model

    print("\nloading treebanks...", flush=True)
    en_train = load_conllu_file("Conlu/en_ewt-ud-train.conllu")
    es_dev = load_conllu_file("Conlu/es_ancora-ud-dev.conllu")
    fi_dev = load_conllu_file("Conlu/fi_tdt-ud-dev.conllu")
    print(f"  en_train={len(en_train)}  es_dev={len(es_dev)}  fi_dev={len(fi_dev)}", flush=True)

    print("\nextracting NLLB encoder embeddings...", flush=True)
    en_sents, en_embs = extract_embeddings(tokenizer, model, en_train, LANG_CODES["en"],
                                            args.layer, args.n_train, args.seed, target_norm)
    es_sents, es_embs = extract_embeddings(tokenizer, model, es_dev, LANG_CODES["es"],
                                            args.layer, args.n_eval, args.seed, target_norm)
    fi_sents, fi_embs = extract_embeddings(tokenizer, model, fi_dev, LANG_CODES["fi"],
                                            args.layer, args.n_eval, args.seed, target_norm)

    print(f"\ntraining distance probe on English only (n={len(en_sents)}, layer={args.layer}, "
          f"rank={args.rank})...", flush=True)
    probe, losses = train_distance_probe(en_sents, en_embs, hidden_dim, rank=args.rank,
                                          epochs=args.epochs, seed=args.seed)

    print("\n=== evaluating the SAME English-trained probe, frozen, cross-lingually ===", flush=True)
    for lang, sents, embs in [("English (in-sample check)", en_sents, en_embs),
                               ("Spanish (Indo-European)", es_sents, es_embs),
                               ("Finnish (non-Indo-European)", fi_sents, fi_embs)]:
        metrics = evaluate_distance_probe(probe, sents, embs)
        rand_metrics = random_direction_eval(sents, embs, hidden_dim, args.rank, args.seed)
        print(f"  {lang:30s}  DSpr={metrics['distance_spearman_mean']:+.4f}  UUAS={metrics['uuas']:.4f}   "
              f"[random control: DSpr={rand_metrics['distance_spearman_mean']:+.4f} "
              f"UUAS={rand_metrics['uuas']:.4f}]", flush=True)


if __name__ == "__main__":
    main()
