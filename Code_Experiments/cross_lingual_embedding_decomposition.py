# -*- coding: utf-8 -*-
"""
cross_lingual_embedding_decomposition.py

Much more basic hypothesis than the Claim 2 steering benchmarks (user's
request, 2026-10-01, after flagging the steering benchmarks as too
complex/compute-heavy and at risk of just re-confirming what's already
known): for a concept with translations in several languages, does the
vector common to all of them (their mean) carry the MAJORITY of each
individual embedding's magnitude? And does the REMAINING component
(after removing that shared direction) still encode the word's
LANGUAGE-SPECIFIC grammatical property (Swahili noun class), i.e. is
grammar cleanly separable from cross-lingual "meaning" as a geometric
decomposition, with no gradient fitting or generation at all?

Method, entirely linear algebra on cached STATIC embeddings (no
generation needed for the actual test -- only cheap, unsteered
translation calls to build the multilingual concept set):
  1. For each Swahili noun (word, class, gloss) in `swahili_noun_
     lexicon_agreement_v2.py`, translate the English GLOSS into 4 other
     well-resourced NLLB languages (French, Spanish, German,
     Portuguese) -- cheap, short, cached, unsteered generate() calls.
  2. Collect the 5-language embedding set per concept (the Swahili word
     itself + the 4 translations), each whole-word mean-pooled (the
     Claim-1a pooling fix, reused).
  3. m_c = mean of the 5 vectors; m_hat = m_c / ||m_c||.
  4. MAGNITUDE TEST: for the Swahili embedding, what fraction of its
     squared norm projects onto m_hat? Reported as a distribution
     (mean/std), not a single number -- it will vary by concept.
  5. RESIDUAL TEST: residual = swahili_embedding - (swahili_embedding
     . m_hat) * m_hat. Run the EXACT SAME classifier + 20-shuffle
     control methodology as `nllb_swahili_embedding_decodability_
     probe.py` (Claim 1a), reusing its functions unchanged, on the
     residual instead of the raw embedding, and separately on the raw
     embedding again in this same run as a same-sample baseline for a
     fair, direct before/after comparison (not just citing the
     original run's number from a different concept sample).

Usage:
    python3 -u cross_lingual_embedding_decomposition.py
"""
from __future__ import annotations

import string

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import mbart_agreement_lib as lib
from swahili_noun_lexicon_agreement_v2 import NOUNS
from nllb_swahili_embedding_decodability_probe import (
    kfold_indices, train_and_eval_fold, cross_val_accuracy, N_FOLDS, N_CONTROL_SHUFFLES, SEED,
)

MODEL_ID = "facebook/nllb-200-distilled-1.3B"
EN = "eng_Latn"
SW = "swh_Latn"
OTHER_LANGS = {"fra_Latn": "French", "spa_Latn": "Spanish", "deu_Latn": "German", "por_Latn": "Portuguese"}
MAX_NEW_TOKENS = 8
MIN_LANGS_FOR_MEAN = 4  # Swahili + at least 3 of the 4 others


def whole_word_embedding(tokenizer, weight, word):
    ids = lib.full_word_subword_ids(tokenizer, word)
    return weight.data[ids].mean(dim=0)


def first_word(text):
    for w in text.split():
        w = w.strip(string.punctuation)
        if w:
            return w
    return None


def build_concepts(tokenizer, model, weight, nouns=None):
    """Builds the 5-language (Swahili + 4 translations) embedding set per
    noun. Factored out of `main` (2026-10-01, for reuse by
    `nllb_swahili_global_language_residual_check.py` per CLAUDE.md hard
    rule #10 -- extend, don't refork) since it was the only part of this
    script's logic another script needed verbatim.

    2026-10-02 (cont.): two additive, backward-compatible extensions for
    `nllb_swahili_decoder_language_neuron_check.py`: (1) an optional
    `nouns` override so a caller can pass a subsample instead of the
    full 319-noun lexicon (that script needs a much smaller pool -- its
    site is a decoder forward pass per language per concept, not a
    cheap embedding lookup); (2) each concept dict also stores the
    literal surface word used per language (`lang_words`), which that
    script needs to look up first-subword ids for its causal margin
    metric -- `lang_vecs` alone (just the pooled embedding) doesn't
    expose the word. No existing field changed or removed."""
    nouns = NOUNS if nouns is None else nouns
    lang_ids = {code: tokenizer.convert_tokens_to_ids(code) for code in OTHER_LANGS}

    print("\n=== building the multilingual concept set (cheap, unsteered translations) ===", flush=True)
    concepts = []
    for i, (noun, cls, gloss) in enumerate(nouns):
        sw_vec = whole_word_embedding(tokenizer, weight, noun)
        lang_vecs = {"swh_Latn": sw_vec}
        lang_words = {"swh_Latn": noun}
        tokenizer.src_lang = EN
        enc = tokenizer(gloss, return_tensors="pt")
        for code in OTHER_LANGS:
            with torch.no_grad():
                out = model.generate(**enc, forced_bos_token_id=lang_ids[code], max_new_tokens=MAX_NEW_TOKENS,
                                      num_beams=1, do_sample=False)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            w = first_word(text)
            if w is None:
                continue
            try:
                lang_vecs[code] = whole_word_embedding(tokenizer, weight, w)
                lang_words[code] = w
            except Exception:
                continue
        if len(lang_vecs) >= MIN_LANGS_FOR_MEAN:
            concepts.append({"noun": noun, "class": cls, "gloss": gloss, "lang_vecs": lang_vecs,
                              "lang_words": lang_words})
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(nouns)}  usable so far: {len(concepts)}", flush=True)
    print(f"\nusable concepts (>= {MIN_LANGS_FOR_MEAN} languages): {len(concepts)}/{len(nouns)}", flush=True)
    return concepts


def decompose_concept(item):
    """Projects the Swahili embedding onto this concept's cross-lingual
    mean direction m_hat and returns (ratio, residual, raw_vec, m_hat).
    Factored out of `main` (2026-10-01) for reuse by
    `nllb_swahili_global_language_residual_check.py`, which needs the
    per-concept m_hat too (not just the residual) to later build a
    GLOBAL (across-concept) residual direction."""
    vecs = list(item["lang_vecs"].values())
    m_c = torch.stack(vecs).mean(dim=0)
    m_hat = m_c / m_c.norm()
    sw_e = item["lang_vecs"]["swh_Latn"]
    proj_scalar = torch.dot(sw_e, m_hat)
    proj_vec = proj_scalar * m_hat
    ratio = (proj_vec.norm() / sw_e.norm()).item()
    residual = sw_e - proj_vec
    return ratio, residual, sw_e, m_hat


def main():
    print(f"loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    weight = model.model.shared.weight

    concepts = build_concepts(tokenizer, model, weight)

    print("\n=== magnitude test: does the cross-lingual mean direction dominate each embedding? ===", flush=True)
    ratios = []
    residuals, raw_vecs, labels = [], [], []
    classes = sorted(set(c["class"] for c in concepts))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for item in concepts:
        ratio, residual, sw_e, _ = decompose_concept(item)
        ratios.append(ratio)
        residuals.append(residual)
        raw_vecs.append(sw_e)
        labels.append(class_to_idx[item["class"]])
    mean_ratio = sum(ratios) / len(ratios)
    std_ratio = (sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)) ** 0.5
    print(f"fraction of ||embedding|| along the cross-lingual mean direction: "
          f"mean={mean_ratio:.4f}  std={std_ratio:.4f}  (n={len(ratios)})", flush=True)
    print(f"(1.0 would mean the embedding points EXACTLY along the shared cross-lingual direction; "
          f"0.0 would mean fully orthogonal to it)", flush=True)

    X_raw = torch.stack(raw_vecs)
    X_resid = torch.stack(residuals)
    y = torch.tensor(labels, dtype=torch.long)
    num_classes = len(classes)
    print(f"\nclasses ({num_classes}): {classes}", flush=True)

    def selectivity_report(X, label):
        real_acc = cross_val_accuracy(X, y, num_classes, N_FOLDS, SEED)
        control_accs = []
        for i in range(N_CONTROL_SHUFFLES):
            y_shuffled = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(1000 + i))]
            control_accs.append(cross_val_accuracy(X, y_shuffled, num_classes, N_FOLDS, SEED))
        control_mean = sum(control_accs) / len(control_accs)
        control_std = (sum((a - control_mean) ** 2 for a in control_accs) / len(control_accs)) ** 0.5
        selectivity = real_acc - control_mean
        print(f"[{label}] REAL={real_acc:.4f}  CONTROL={control_mean:.4f} (std {control_std:.4f})  "
              f"SELECTIVITY={selectivity:+.4f}", flush=True)
        return real_acc, control_mean, selectivity

    print("\n=== RAW embedding (same concept sample, same-run baseline) ===", flush=True)
    selectivity_report(X_raw, "RAW")

    print("\n=== RESIDUAL (cross-lingual mean direction removed) ===", flush=True)
    selectivity_report(X_resid, "RESIDUAL")

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
