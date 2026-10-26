"""
mbart_agreement_lib.py

Shared fit/test harness for Phase 2 and Phase 3 of next_step.md, built by
generalizing gradient_steering_mbart.py's single-word/single-sentence
recipe (full-word joint-log-prob objective, GradientPatch/FixedPatch,
first_real_subword_id/full_word_subword_ids gotchas) to:
  - an ARBITRARY, ORIGIN-AGNOSTIC (masc->fem or fem->masc) flip, since
    mined real sentences attest either gender;
  - MULTIPLE simultaneous patch positions sharing one direction (the toy
    model's "controller + every upstream agreement position" ingredient),
    not just the target word's own position;
  - random-direction AND orthogonal-null controls at the same positions
    and alpha, per project convention (CLAUDE.md "always include a
    matched random-direction control").

Kept deliberately close to gradient_steering_mbart.py's function shapes
so anyone diffing the two can see exactly what was generalized and why.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import steering  # get_subword_indices_for_word only -- read-only, no gradient risk


def load_model(model_path: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def build_inputs(tokenizer, prompt: str, lang_code: str, device: str):
    tokenizer.src_lang = lang_code
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    # mBART's tokenizer exposes lang_code_to_id; NLLB's (NllbTokenizer) does
    # not -- confirmed directly (2026-10-01: "NllbTokenizer has no attribute
    # lang_code_to_id"). convert_tokens_to_ids works for both, so prefer
    # lang_code_to_id only where present rather than branching on model type.
    if hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id[lang_code]
    else:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(lang_code)
    return enc, forced_bos_token_id


def first_real_subword_id(tokenizer, word: str):
    ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    for tid in ids:
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        if tok.replace("▁", "").strip():
            return tid, tok
    raise ValueError(f"could not find a non-degenerate subword id for {word!r}")


def full_word_subword_ids(tokenizer, word: str) -> List[int]:
    ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    start = 0
    for i, tid in enumerate(ids):
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        if tok.replace("▁", "").strip():
            start = i
            break
    return ids[start:]


class GradientPatch:
    def __init__(self, positions: List[int], delta: torch.Tensor):
        self.positions = positions
        self.delta = delta

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        for pos in self.positions:
            if pos < hidden.shape[1]:
                hidden[:, pos, :] = hidden[:, pos, :] + self.delta
        if is_tuple:
            return (hidden,) + output[1:]
        return hidden


class FixedPatch:
    def __init__(self, positions: List[int], direction: torch.Tensor, alpha: float):
        self.positions = positions
        self.push = alpha * direction

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        for pos in self.positions:
            if pos < hidden.shape[1]:
                hidden[:, pos, :] = hidden[:, pos, :] + self.push
        if is_tuple:
            return (hidden,) + output[1:]
        return hidden


def measure_class_margins(model, layer_module, enc, decoder_input_ids, position, direction, alpha, candidate_ids):
    """One forward pass (no `generate()`): returns the logits for each
    {name: token_id} pair in `candidate_ids` at the final decoder
    position, with `alpha * direction`
    (or no patch at all if `direction` is None) added at `position` in
    `layer_module`'s output. Added 2026-09-25 per the external-critique
    synthesis (all three reviewers independently flagged that greedy
    decoded-text pass/fail hides whether a direction moved the relevant
    logits at all) -- every steering attempt on this track should log
    these margins across its alpha grid instead of only the decoded string.
    """
    handle = None
    if direction is not None:
        hook = FixedPatch([position], direction, alpha)
        handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"),
                        decoder_input_ids=decoder_input_ids)
        logits = out.logits[0, -1, :]
    finally:
        if handle:
            handle.remove()
    return {name: logits[cid].item() for name, cid in candidate_ids.items()}


def get_baseline_ids(model, tokenizer, enc, forced_bos_token_id, max_new_tokens, device):
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=forced_bos_token_id,
                              max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    return out


def word_positions_to_subword_positions(tokenizer, words: List[str], word_idxs: List[int]) -> List[int]:
    idx_pos = []
    for w in word_idxs:
        idx_pos.extend(steering.get_subword_indices_for_word(tokenizer, words, w))
    return sorted(set(idx_pos))


def fit_direction(model, tokenizer, layer_module, words: List[str], patch_word_idxs: List[int],
                   orig_form: str, flip_form: str, lang_code: str, device: str,
                   max_new_tokens: int = 30) -> Optional[dict]:
    """Fits a direction (gradient of log P(flip_form), whole word, jointly
    over all its subwords) at the given patch positions. Returns None if
    the ORIGINAL form's first subword isn't found in the model's own
    unpatched reconstruction of the sentence (the "baseline reconstructs
    the input" assumption gradient_steering_mbart.py's approach relies
    on) -- callers should count/report these as skipped, not silently
    drop them."""
    idx_pos = word_positions_to_subword_positions(tokenizer, words, patch_word_idxs)
    prompt = " ".join(words)
    enc, forced_bos = build_inputs(tokenizer, prompt, lang_code, device)
    baseline_ids = get_baseline_ids(model, tokenizer, enc, forced_bos, max_new_tokens, device)

    orig_ids_full = full_word_subword_ids(tokenizer, orig_form)
    flip_ids_full = full_word_subword_ids(tokenizer, flip_form)

    seq = baseline_ids[0].tolist()
    if orig_ids_full[0] not in seq:
        return None
    target_step = seq.index(orig_ids_full[0])
    if target_step == 0:
        return None

    prefix = baseline_ids[:, :target_step]
    flip_prefix_tensor = torch.tensor([flip_ids_full[:-1]], dtype=torch.long, device=device)
    decoder_input_ids = torch.cat([prefix, flip_prefix_tensor], dim=1)

    d_model = model.config.d_model if hasattr(model.config, "d_model") else model.config.hidden_size
    delta = torch.zeros(d_model, device=device, requires_grad=True)
    hook = GradientPatch(idx_pos, delta)
    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            outputs = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"),
                             decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            k = len(flip_ids_full)
            step_positions = torch.arange(target_step - 1, target_step - 1 + k, device=device)
            token_targets = torch.tensor(flip_ids_full, dtype=torch.long, device=device)
            per_token_logprob = log_probs[0, step_positions, token_targets]
            objective = per_token_logprob.sum()
            model.zero_grad(set_to_none=True)
            objective.backward()
    finally:
        handle.remove()

    grad = delta.grad
    if grad is None or grad.norm().item() == 0.0:
        return None
    direction = grad.detach().clone() / grad.norm()
    return {
        "direction": direction, "idx_pos": idx_pos,
        "orig_ids_full": orig_ids_full, "flip_ids_full": flip_ids_full,
        "target_step": target_step,
    }


def first_divergence(orig_ids_full: List[int], flip_ids_full: List[int]) -> Optional[int]:
    """Index of the first subword position where the two full-word token
    sequences actually differ. Comparing p_orig/p_flip at index 0
    unconditionally (the original approach) is wrong whenever the two
    surface forms share a subword-tokenized prefix -- common in French,
    where the regular fem pattern is masc-stem+suffix (`royal`/`royale`,
    `majeur`/`majeure`, `algerien`/`algerienne`, ...): mBART's tokenizer
    then gives both forms the SAME first subword id, so
    probs[orig_ids_full[0]] and probs[flip_ids_full[0]] read the
    identical tensor entry and are bit-identical by construction,
    regardless of any steering effect (found via
    phase2_failure_analysis.py / phase3_failure_analysis.py, ~50% of
    tested lemmas). Returns None if one sequence is a strict prefix of
    the other (no position where they differ at all) -- a genuine edge
    case with no natural single-token probability to compare; callers
    treat this as unscoreable, matching the "skip and report" pattern
    used elsewhere in this module (e.g. fit_direction's baseline-
    reconstruction-failure case)."""
    n = min(len(orig_ids_full), len(flip_ids_full))
    for i in range(n):
        if orig_ids_full[i] != flip_ids_full[i]:
            return i
    return None


@torch.no_grad()
def test_direction(model, tokenizer, layer_module, words: List[str], idx_pos: List[int],
                    direction: torch.Tensor, alpha: float, lang_code: str, device: str,
                    orig_ids_full: List[int], flip_ids_full: List[int], target_step: int,
                    max_new_tokens: int = 30) -> dict:
    div_idx = first_divergence(orig_ids_full, flip_ids_full)
    if div_idx is None:
        raise ValueError(f"orig_ids_full={orig_ids_full} and flip_ids_full={flip_ids_full} "
                          f"never diverge (one is a token-level prefix of the other) -- unscoreable")
    prompt = " ".join(words)
    enc, forced_bos = build_inputs(tokenizer, prompt, lang_code, device)
    hook = FixedPatch(idx_pos, direction, alpha)
    handle = layer_module.register_forward_hook(hook)
    try:
        out = model.generate(**enc, forced_bos_token_id=forced_bos, max_new_tokens=max_new_tokens,
                              num_beams=1, do_sample=False)
        decoder_input_ids = out[:, :target_step] if target_step < out.shape[1] else out[:, :-1]
        shared_prefix = orig_ids_full[:div_idx]  # == flip_ids_full[:div_idx] by construction
        probe_input_ids = decoder_input_ids
        if shared_prefix:
            shared_tensor = torch.tensor([shared_prefix], dtype=torch.long, device=device)
            probe_input_ids = torch.cat([decoder_input_ids, shared_tensor], dim=1)
        probe_out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"),
                           decoder_input_ids=probe_input_ids)
        probs = torch.softmax(probe_out.logits[0, -1, :], dim=-1)
        p_orig, p_flip = probs[orig_ids_full[div_idx]].item(), probs[flip_ids_full[div_idx]].item()
    finally:
        handle.remove()
    generated_ids = out[0].tolist()

    # continuation success: teacher-force the ENTIRE correct flip word
    # (all subwords but the last, exactly mirroring fit_direction's own
    # decoder_input_ids construction -- feeding only ONE extra token here
    # while checking predictions across the WHOLE flip word, as an
    # earlier version of this function did, reads past the logits tensor
    # for any 3+-subword flip word: caught via a real IndexError on
    # 'irakien'->'irakienne', not a theoretical concern), then check
    # whether every subword's argmax matches, under the patched encoder
    # representation (active for the whole decode).
    cont_ok = None
    if len(flip_ids_full) > 1:
        flip_prefix_tensor = torch.tensor([flip_ids_full[:-1]], dtype=torch.long, device=device)
        with_prefix = torch.cat([decoder_input_ids, flip_prefix_tensor], dim=1)
        hook2 = FixedPatch(idx_pos, direction, alpha)
        handle2 = layer_module.register_forward_hook(hook2)
        try:
            cont_out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"),
                              decoder_input_ids=with_prefix)
            cont_log_probs = torch.log_softmax(cont_out.logits, dim=-1)
            k = len(flip_ids_full)
            # predicts flip_ids_full[0..k-1] at positions
            # decoder_input_ids.shape[1]-1 .. decoder_input_ids.shape[1]-1+k-1
            positions = torch.arange(decoder_input_ids.shape[1] - 1, decoder_input_ids.shape[1] - 1 + k, device=device)
            targets = torch.tensor(flip_ids_full, dtype=torch.long, device=device)
            argmax_ids = cont_log_probs[0, positions, :].argmax(dim=-1)
            cont_ok = bool((argmax_ids == targets).all().item())
        finally:
            handle2.remove()
    whole_word_ok = any(generated_ids[i:i + len(flip_ids_full)] == flip_ids_full
                         for i in range(max(0, len(generated_ids) - len(flip_ids_full) + 1)))

    return {
        "text": tokenizer.decode(out[0], skip_special_tokens=False),
        "p_orig": p_orig, "p_flip": p_flip, "divergence_idx": div_idx,
        "prefix_success": bool(p_flip > p_orig),
        "continuation_success": cont_ok,
        "whole_word_success": whole_word_ok,
    }


def random_direction_like(direction: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    v = torch.randn(direction.shape, generator=gen)
    return v / (v.norm() + 1e-8)


def orthogonal_null_direction_like(direction: torch.Tensor, seed: int) -> torch.Tensor:
    """A random direction with the component ALONG `direction` removed via
    Gram-Schmidt, then renormalized -- guaranteed orthogonal (cos~0) to
    the real fitted direction, unlike an unconstrained random draw which
    is only orthogonal in expectation. Distinguishes 'this exact direction
    matters' from 'any large enough push in a generic direction would do
    the same thing', per next_step.md's non-negotiables."""
    gen = torch.Generator()
    gen.manual_seed(seed + 999983)
    v = torch.randn(direction.shape, generator=gen)
    v = v - (v @ direction) * direction
    return v / (v.norm() + 1e-8)
