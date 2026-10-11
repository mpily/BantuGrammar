"""
steering.py

Everything needed to patch a decoder LM's residual stream at a chosen
layer and observe the effect on generation. Consumes directions saved by
coordinator.py (e.g. an order probe's direction, or a cross-lingual
difference direction).

Restricted by design to decoder (autoregressive) models, patched and
generated from as the SAME model instance -- BERT-shaped encoders have no
generation to observe, and you cannot inject one model's direction into a
DIFFERENT model's layer (unrelated training runs, unrelated geometry).
See project history for the full reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Steering mechanisms
# ---------------------------------------------------------------------------

class ClassReflectionSteering:
    """
    Reflects EVERY token's representation across a linear classifier's
    decision boundary, flipping whichever class each token currently
    belongs to -- Masc tokens become Fem, Fem tokens become Masc,
    SIMULTANEOUSLY, for the whole sequence, self-determined per token
    (no need to specify in advance which words are which class).

    Fixes a real limitation of AdditiveSteering for this use case (see
    chat): pushing every position by the SAME fixed vector can only ever
    move things toward ONE target class -- it would push an
    already-Fem-classified token even FURTHER toward Fem, not flip it
    back to Masc. A genuine bidirectional swap needs a sign-aware
    operation, not a uniform translation.

    Math: for signed classifier margin g(h) = w.h + c (g>0 means the
    classifier prefers `to_label` over `from_label`), reflecting h across
    the decision hyperplane {h : g(h)=0} via
        h' = h - strength * 2 * g(h) / ||w||^2 * w
    EXACTLY negates g at strength=1.0 (verify: g(h') = w.h' + c =
    g(h) - 2*g(h) = -g(h)), so every token flips to the opposite side of
    the boundary. strength<1 is a partial move (doesn't fully cross the
    boundary for weakly-classified tokens); strength>1 overshoots further
    past it.

    w, c: use reflection_params_from_probe() to extract these from a
    trained MorphProbe, or random_reflection_params_like() for a control.
    """

    def __init__(self, w: torch.Tensor, c: float, strength: float = 1.0):
        self.w = w
        self.c = c
        self.strength = strength

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        w = self.w.to(h.dtype).to(h.device)
        w_sq = (w * w).sum() + 1e-8
        g = (h * w).sum(dim=-1, keepdim=True) + self.c
        w_broadcast = w.view(*([1] * (h.dim() - 1)), -1)
        return h - self.strength * 2 * g / w_sq * w_broadcast


def reflection_params_from_probe(probe, label_to_id: dict, from_label: str, to_label: str):
    """Extracts (w, c) for ClassReflectionSteering from a trained
    morph_probe.MorphProbe -- duck-typed on probe.net.weight/.net.bias,
    no import of morph_probe needed here."""
    with torch.no_grad():
        w_from = probe.net.weight[label_to_id[from_label]]
        w_to = probe.net.weight[label_to_id[to_label]]
        b_from = probe.net.bias[label_to_id[from_label]]
        b_to = probe.net.bias[label_to_id[to_label]]
        w = (w_to - w_from).clone()
        c = (b_to - b_from).item()
    return w, c


def random_reflection_params_like(w: torch.Tensor, seed: Optional[int] = None):
    """Control condition for ClassReflectionSteering: a random decision
    boundary through the origin (c=0), same norm-scale as the real w
    (norm isn't forced equal, matching random_direction_like's philosophy
    of a comparably-sized but arbitrary alternative)."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    rand_w = torch.randn(w.shape, generator=gen) * w.norm()
    return rand_w, 0.0


class AdditiveSteering:
    """h' = h + alpha * direction, broadcast over the sequence dimension."""

    def __init__(self, direction: torch.Tensor, alpha: float = 1.0):
        self.direction = direction
        self.alpha = alpha

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.alpha * self.direction.to(h.dtype).to(h.device)


class SmallAdapter(nn.Module):
    """
    A genuinely small, bottlenecked MLP (hidden_dim -> bottleneck -> hidden_dim)
    meant to be trained with an explicit fidelity term so it learns the
    MINIMAL edit needed to achieve some goal (here: flip the order probe's
    prediction for a specific token), rather than an unconstrained,
    hand-picked-magnitude push. See train_adapter.py for the training loop.

    Final layer initialized near-zero, so an UNTRAINED SmallAdapter starts
    as an approximate identity function when wrapped in AdapterSteering's
    residual connection (adapter(h) + h ~= h at initialization) -- training
    then learns how far from identity it actually needs to move, rather
    than starting from an arbitrary random transform.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        # near-zero init on the output layer -- see class docstring
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class TargetedAdapterSteering:
    """
    Like TargetedPairSteering, but replaces the fixed alpha*direction push
    with a LEARNED, per-role adapter transform, applied ONLY at the
    specified token positions -- everything else in the sequence is left
    untouched. adapter_positive is applied at idx_push_positive positions
    (intended role: "push later" in the order-probe's sense), adapter_negative
    at idx_push_negative positions ("push earlier"). Two separate small
    adapters are used because the two roles need to move in OPPOSITE
    directions along the order axis, and a single shared adapter has no
    way to know which role it's being asked to play for a given call.
    """

    def __init__(
        self,
        adapter_positive: nn.Module,
        adapter_negative: nn.Module,
        idx_push_positive: List[int],
        idx_push_negative: List[int],
    ):
        self.adapter_positive = adapter_positive
        self.adapter_negative = adapter_negative
        self.idx_push_positive = idx_push_positive
        self.idx_push_negative = idx_push_negative

    @torch.no_grad()
    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        h = h.clone()
        for i in self.idx_push_positive:
            h[:, i, :] = self.adapter_positive(h[:, i, :]) + h[:, i, :]
        for i in self.idx_push_negative:
            h[:, i, :] = self.adapter_negative(h[:, i, :]) + h[:, i, :]
        return h


class AdapterSteering:
    """h' = adapter(h) + h (residual applied here, not inside the adapter
    -- forgetting it is the 'discards all content' degenerate-solution
    failure mode). blend interpolates between unpatched and fully-patched."""

    def __init__(self, adapter: nn.Module, blend: float = 1.0):
        self.adapter = adapter
        self.blend = blend

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            edited = self.adapter(h) + h
        return (1 - self.blend) * h + self.blend * edited


class TargetedPairSteering:
    """
    Fixes a real problem found via chat: PositionalGradientSteering's ramp
    is tied to raw sequence position (-1 at the first token, +1 at the
    last). If the direction being pushed is the model's OWN order axis
    (higher score = later), and the two words you're testing already sit
    in their "natural" left-to-right order, that ramp AMPLIFIES the
    existing order rather than ever being able to reverse it for a
    specific pair -- confirmed directly: pushing "can"/"see" (which
    already sit earlier/later respectively) widened their score gap
    roughly 5x instead of closing it.

    This class instead pushes EXACTLY two sets of token positions in
    OPPOSITE directions along `direction`, leaving every other position
    untouched -- a direct, surgical test of "can this specific pair's
    relative order be flipped," independent of where they sit in the
    sequence. idx_push_positive tokens are pushed toward HIGHER
    order-score (later); idx_push_negative tokens are pushed toward LOWER
    (earlier). To try to flip a pair where the probe currently predicts
    word_a before word_b, push word_a's token(s) positive and word_b's
    token(s) negative.

    Indices are SUBWORD token positions in whatever tokenization the
    patched forward pass actually uses -- NOT word indices. A short common
    word is often (not always) a single BPE token, but don't assume it;
    get the real subword index list from the tokenizer's word_ids() for
    the exact call being patched (see
    get_subword_indices_for_word in this module).
    """

    def __init__(
        self,
        direction: torch.Tensor,
        idx_push_positive: List[int],
        idx_push_negative: List[int],
        alpha: float = 1.0,
    ):
        self.direction = direction
        self.idx_push_positive = idx_push_positive
        self.idx_push_negative = idx_push_negative
        self.alpha = alpha

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        seq_len = h.shape[1]
        push = torch.zeros(seq_len, device=h.device, dtype=h.dtype)
        for i in self.idx_push_positive:
            push[i] = self.alpha
        for i in self.idx_push_negative:
            push[i] = -self.alpha
        push = push.view(1, seq_len, 1)
        direction = self.direction.to(h.dtype).to(h.device)
        return h + push * direction


def get_subword_indices_for_word(tokenizer, words: List[str], word_idx: int) -> List[int]:
    """Returns the list of SUBWORD token positions corresponding to
    words[word_idx], for the SAME is_split_into_words=True tokenization
    used elsewhere in this module (_encode_words_at_layer,
    measure_pairwise_order_shift) -- call this with the identical
    `words` list before building a TargetedPairSteering, so the indices
    line up with what will actually be patched. Deterministic given the
    same tokenizer + words, so pre-computing like this is safe."""
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    word_ids = enc.word_ids(batch_index=0)
    return [i for i, w in enumerate(word_ids) if w == word_idx]


class PositionalGradientSteering:
    """
    Fixes a mathematical dead end in AdditiveSteering, found via chat
    discussion: for a linear order-score probe s(h) = c^T h, adding the
    SAME vector to every token position leaves every pairwise difference
    s(h_i) - s(h_j) UNCHANGED, for any direction and any alpha --
    the shared additive term cancels exactly. So AdditiveSteering, no
    matter what direction you feed it (English's own order axis, a
    cross-lingual difference, random), is mathematically incapable of
    moving relative word order as measured by the very probe that
    produced the direction. Any effect it has on generation (e.g. the
    representational collapse seen at higher alpha in chat) is NOT
    evidence of a word-order change -- it's a fluency/stability casualty.

    This class instead scales the push by a per-position weight that
    varies across the sequence, so different tokens get different amounts
    of shift and pairwise order-score differences CAN move. Default
    weighting: linear ramp from -1 (first token) to +1 (last token),
    i.e. push early tokens one way along `direction` and late tokens the
    opposite way -- directly manipulating the spread of order-scores
    across the sentence rather than translating everything together.

    IMPORTANT LIMITATION found via chat: because the ramp is tied to raw
    SEQUENCE POSITION, and `direction` is typically the model's own order
    axis (higher score = later), this AMPLIFIES whatever order the model
    already perceives between any two tokens rather than ever being able
    to REVERSE it for a specific pair -- confirmed directly: steering
    "can"/"see" (which already sit earlier/later respectively) widened
    their score gap roughly 5x instead of closing it. Use
    TargetedPairSteering instead if you want to test whether a SPECIFIC
    pair's relative order can be flipped, independent of their position.
    """

    def __init__(self, direction: torch.Tensor, alpha: float = 1.0, reverse: bool = False):
        self.direction = direction
        self.alpha = alpha
        self.reverse = reverse

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq_len, hidden_dim)
        seq_len = h.shape[1]
        if seq_len <= 1:
            return h
        weights = torch.linspace(-1.0, 1.0, seq_len, device=h.device, dtype=h.dtype)
        if self.reverse:
            weights = -weights
        # (seq_len,) -> (1, seq_len, 1) so it broadcasts against (batch, seq_len, hidden_dim)
        weights = weights.view(1, seq_len, 1)
        direction = self.direction.to(h.dtype).to(h.device)
        return h + self.alpha * weights * direction


def load_direction_from_coordinator_output(path: str, lang_tag: str) -> torch.Tensor:
    """Load an order-probe direction saved by coordinator.py, ready for
    AdditiveSteering(direction=..., alpha=...)."""
    all_results = torch.load(path, weights_only=False)
    return all_results[lang_tag]["order"]["direction"]


def cross_lingual_direction_from_coordinator_output(path: str, lang_a: str, lang_b: str) -> torch.Tensor:
    """direction = c_B - c_A, from two lang_tags saved in the SAME
    coordinator.py output file (same model/layer, required)."""
    import order_probe as ordp

    all_results = torch.load(path, weights_only=False)
    hidden_dim = all_results[lang_a]["hidden_dim"]
    model_a = ordp.OrderProbe(hidden_dim)
    model_a.load_state_dict(all_results[lang_a]["order"]["state_dict"])
    model_b = ordp.OrderProbe(hidden_dim)
    model_b.load_state_dict(all_results[lang_b]["order"]["state_dict"])
    return ordp.cross_lingual_steering_direction(model_a, model_b)


# ---------------------------------------------------------------------------
# Layer patching via forward hooks
# ---------------------------------------------------------------------------

@dataclass
class PatchHandle:
    hook_handle: "torch.utils.hooks.RemovableHandle"

    def remove(self):
        self.hook_handle.remove()


def patch_layer(
    layer_module: nn.Module,
    steering: Callable[[torch.Tensor], torch.Tensor],
    output_index: Optional[int] = 0,
) -> PatchHandle:
    """Registers a forward hook replacing `layer_module`'s output (or
    output[output_index] if it returns a tuple, as GPT-2-style blocks do)
    with steering(output)."""

    def hook(module, inputs, output):
        if output_index is None:
            return steering(output)
        if isinstance(output, tuple):
            new_hidden = steering(output[output_index])
            return (new_hidden,) + output[output_index + 1:]
        return steering(output)

    handle = layer_module.register_forward_hook(hook)
    return PatchHandle(hook_handle=handle)


class patched:
    """Context manager: guarantees hook removal even if generation raises."""

    def __init__(self, layer_module: nn.Module, steering: Callable, output_index: Optional[int] = 0):
        self.layer_module = layer_module
        self.steering = steering
        self.output_index = output_index
        self._handle = None

    def __enter__(self):
        self._handle = patch_layer(self.layer_module, self.steering, self.output_index)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(
    model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 20,
    device: str = "cpu", do_sample: bool = False, temperature: float = 1.0,
    forced_bos_token_id: Optional[int] = None,
) -> str:
    """
    Minimal greedy/sampling generation, deliberately not using HF's
    .generate() so behavior stays transparent while debugging whether a
    patch does anything. Dispatches to the correct loop based on whether
    the model is encoder-decoder (BART/mBART/T5) or decoder-only (GPT-2).

    forced_bos_token_id: REQUIRED for coherent generation from mBART-family
    models specifically (found via real mbart-large-50 output during
    development, not a hypothetical concern): mBART is trained with
    explicit language-ID tokens and its own documented usage forces the
    decoder's first REAL generated token to be the target language code
    (e.g. tokenizer.lang_code_to_id["fr_XX"]) via HF's forced_bos_token_id
    mechanism. Without it, the model has no idea what language it's
    supposed to produce -- observed symptom: degenerate repetition of a
    single word, occurring even in the UNPATCHED baseline, before any
    steering is involved at all. facebook/bart-base doesn't need this (no
    language-ID tokens in its vocabulary), which is why earlier
    experiments on it worked without setting this. Ignored for
    decoder-only models.

    BUG THIS FIXES (found via real BART output during development): the
    original version only implemented the decoder-only loop and reused it
    for BART too -- but a seq2seq model's encoder input is supposed to
    stay FIXED across decoding steps; only the decoder's own sequence
    grows. Feeding the whole growing sequence back in as `input_ids` every
    step made HF silently auto-construct decoder inputs as a right-shifted
    copy of that growing sequence (BART's own denoising-pretraining
    default when no decoder_input_ids is given) -- which produces
    "reconstruct an increasingly strange pseudo-input", not "continue the
    prompt". Symptom: output collapses to echoing the prompt followed by
    EOS repeated for the rest of the budget. If you see that pattern
    again, it's this bug, not the model or the steering direction.
    """
    is_seq2seq = getattr(model.config, "is_encoder_decoder", False)
    if is_seq2seq:
        return _generate_text_encoder_decoder(model, tokenizer, prompt, max_new_tokens, device,
                                               do_sample, temperature, forced_bos_token_id)
    return _generate_text_decoder_only(model, tokenizer, prompt, max_new_tokens, device, do_sample, temperature)


@torch.no_grad()
def _generate_text_decoder_only(
    model, tokenizer, prompt, max_new_tokens, device, do_sample, temperature,
) -> str:
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(input_ids=input_ids).logits[:, -1, :]
        if do_sample:
            next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return tokenizer.decode(input_ids[0].tolist())


@torch.no_grad()
def _generate_text_encoder_decoder(
    model, tokenizer, prompt, max_new_tokens, device, do_sample, temperature,
    forced_bos_token_id: Optional[int] = None,
) -> str:
    """Encode the prompt ONCE (any active patch fires during this single
    call), then decode autoregressively against those FIXED encoder
    outputs -- the encoder is never re-run, only the decoder's own
    sequence grows. This is what makes patching the ENCODER meaningful:
    the steered representation is computed once and then conditions every
    decoding step via cross-attention, matching how the real pipeline is
    supposed to work (see project history on encoder-decoder design)."""
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    encoder_outputs = model.get_encoder()(
        input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask")
    )

    decoder_start_id = model.config.decoder_start_token_id
    if decoder_start_id is None:
        decoder_start_id = model.config.bos_token_id or tokenizer.bos_token_id or tokenizer.eos_token_id
    decoder_input_ids = torch.tensor([[decoder_start_id]], dtype=torch.long, device=device)

    eos_id = model.config.eos_token_id
    for step in range(max_new_tokens):
        out = model(
            encoder_outputs=encoder_outputs,
            attention_mask=enc.get("attention_mask"),
            decoder_input_ids=decoder_input_ids,
        )
        logits = out.logits[:, -1, :]

        # Force the FIRST real generated token to be the target language
        # code, matching HF's documented mBART usage -- see generate_text's
        # docstring. Only applies at step 0; every subsequent step is
        # chosen normally.
        if step == 0 and forced_bos_token_id is not None:
            next_id = torch.tensor([[forced_bos_token_id]], dtype=torch.long, device=device)
        elif do_sample:
            next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return tokenizer.decode(decoder_input_ids[0].tolist())


@torch.no_grad()
def generate_text_with_topk(
    model, tokenizer, prompt: str, max_new_tokens: int = 20, top_k: int = 5,
    device: str = "cpu", forced_bos_token_id: Optional[int] = None,
) -> Tuple[str, List[dict]]:
    """
    Same generation as generate_text, but records the top_k candidate
    tokens and their softmax probabilities at EVERY decoding step, instead
    of only returning the final greedy string. Built specifically to test
    hypotheses like "once the model skips a disrupted token, is the next
    most likely word just whatever the ORIGINAL sentence would have said"
    with actual numbers, rather than inferring it from the output string
    alone -- greedy decoding only shows you the winner at each step and
    discards how close the race was.

    forced_bos_token_id: see generate_text's docstring -- required for
    coherent mBART-family output.

    Returns (generated_text, steps), where steps is a list of dicts:
        {"step": i, "chosen": str, "chosen_prob": float,
         "topk": [(token_str, prob), ...]}
    Currently only implements the encoder-decoder (seq2seq/BART) path,
    since that's what this project is steering. Raises NotImplementedError
    for decoder-only models -- extend _decoder_only variant analogously if
    you need it there too.
    """
    is_seq2seq = getattr(model.config, "is_encoder_decoder", False)
    if not is_seq2seq:
        raise NotImplementedError("generate_text_with_topk currently only supports encoder-decoder models")

    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    encoder_outputs = model.get_encoder()(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))

    decoder_start_id = model.config.decoder_start_token_id
    if decoder_start_id is None:
        decoder_start_id = model.config.bos_token_id or tokenizer.bos_token_id or tokenizer.eos_token_id
    decoder_input_ids = torch.tensor([[decoder_start_id]], dtype=torch.long, device=device)

    eos_id = model.config.eos_token_id
    steps = []
    for step in range(max_new_tokens):
        out = model(encoder_outputs=encoder_outputs, attention_mask=enc.get("attention_mask"),
                     decoder_input_ids=decoder_input_ids)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]
        top_probs, top_ids = probs.topk(top_k)

        if step == 0 and forced_bos_token_id is not None:
            chosen_id = forced_bos_token_id
            chosen_prob = probs[chosen_id].item()
        else:
            chosen_id = top_ids[0].item()
            chosen_prob = top_probs[0].item()

        steps.append({
            "step": step,
            "chosen": tokenizer.decode([chosen_id]),
            "chosen_prob": chosen_prob,
            "topk": [(tokenizer.decode([tid.item()]), p.item()) for tid, p in zip(top_ids, top_probs)],
        })

        next_id = torch.tensor([[chosen_id]], dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=1)
        if eos_id is not None and chosen_id == eos_id:
            break

    return tokenizer.decode(decoder_input_ids[0].tolist()), steps


def print_topk_comparison(baseline_steps: List[dict], patched_steps: List[dict], top_k: int = 5):
    """Side-by-side step-by-step printout: at each decoding position, what
    was baseline's top-k vs patched's top-k. Makes it visible exactly
    where and how much a specific token's probability moved, rather than
    only seeing whether the greedy argmax happened to flip."""
    n = max(len(baseline_steps), len(patched_steps))
    for i in range(n):
        print(f"--- step {i} ---")
        if i < len(baseline_steps):
            b = baseline_steps[i]
            b_str = "  ".join(f"{tok!r}:{p:.3f}" for tok, p in b["topk"][:top_k])
            print(f"  baseline: chose {b['chosen']!r} (p={b['chosen_prob']:.3f})  |  top-{top_k}: {b_str}")
        if i < len(patched_steps):
            p_ = patched_steps[i]
            p_str = "  ".join(f"{tok!r}:{p:.3f}" for tok, p in p_["topk"][:top_k])
            print(f"  patched:  chose {p_['chosen']!r} (p={p_['chosen_prob']:.3f})  |  top-{top_k}: {p_str}")


@torch.no_grad()
def measure_feature_flip(
    model, tokenizer, layer_module, steering_fn, morph_probe, label_to_id: dict,
    words: List[str], target_word_indices: List[int], layer_index: int,
    device: str = "cpu", output_index: Optional[int] = 0,
) -> dict:
    """
    The classification-probe analogue of measure_pairwise_order_shift: for
    each target word position, report the morph probe's predicted class
    before vs. after steering. Supports MULTIPLE target positions at once
    -- e.g. a full noun+adjective+article agreement chain -- since a real
    gender flip requires moving every agreeing word together, not just one.

    To push several positions toward the SAME target class (unlike the
    order probe's opposite-direction pair), use TargetedPairSteering with
    ALL target indices in idx_push_positive and idx_push_negative=[] --
    it already supports this without any new steering class.
    """
    id_to_label = {v: k for k, v in label_to_id.items()}

    baseline_pooled = _encode_words_at_layer(model, tokenizer, words, layer_index, device)
    baseline_preds = {}
    for idx in target_word_indices:
        logits = morph_probe(baseline_pooled[idx].unsqueeze(0))
        baseline_preds[words[idx]] = id_to_label[logits.argmax(dim=-1).item()]

    with patched(layer_module, steering_fn, output_index):
        patched_pooled = _encode_words_at_layer(model, tokenizer, words, layer_index, device)
    patched_preds = {}
    for idx in target_word_indices:
        logits = morph_probe(patched_pooled[idx].unsqueeze(0))
        patched_preds[words[idx]] = id_to_label[logits.argmax(dim=-1).item()]

    all_flipped = all(baseline_preds[w] != patched_preds[w] for w in baseline_preds)
    return {
        "baseline_predictions": baseline_preds,
        "patched_predictions": patched_preds,
        "all_flipped": all_flipped,
    }


def compare_reflection_vs_random_control(
    model, tokenizer, layer_module, w: torch.Tensor, c: float, prompt: str,
    strength: float = 1.0, max_new_tokens: int = 20, device: str = "cpu",
    output_index: Optional[int] = 0, n_random_controls: int = 3, seed: int = 0,
    forced_bos_token_id: Optional[int] = None,
) -> dict:
    """The ClassReflectionSteering analogue of compare_direction_vs_random_control:
    unpatched baseline, patched with the REAL classifier's decision
    boundary reflected, patched with n_random_controls reflections across
    RANDOM decision boundaries of comparable scale. Same causal-attribution
    logic as everywhere else in this project -- if the real reflection
    reliably differs from random reflections, that's evidence the ACTUAL
    trained gender boundary matters, not just that reflecting across any
    old hyperplane perturbs output."""
    baseline = generate_text(model, tokenizer, prompt, max_new_tokens, device, forced_bos_token_id=forced_bos_token_id)

    real_steering = ClassReflectionSteering(w, c, strength)
    with patched(layer_module, real_steering, output_index):
        real_patched = generate_text(model, tokenizer, prompt, max_new_tokens, device, forced_bos_token_id=forced_bos_token_id)

    random_outputs = []
    for i in range(n_random_controls):
        rand_w, rand_c = random_reflection_params_like(w, seed=seed + i)
        rand_steering = ClassReflectionSteering(rand_w, rand_c, strength)
        with patched(layer_module, rand_steering, output_index):
            random_outputs.append(generate_text(model, tokenizer, prompt, max_new_tokens, device, forced_bos_token_id=forced_bos_token_id))

    return {
        "prompt": prompt,
        "baseline": baseline,
        "real_reflection_patched": real_patched,
        "random_reflection_patched": random_outputs,
    }


def compare_patched_vs_unpatched(
    model, tokenizer, layer_module, steering, prompt, max_new_tokens=20, device="cpu", output_index=0,
    forced_bos_token_id: Optional[int] = None,
) -> dict:
    baseline = generate_text(model, tokenizer, prompt, max_new_tokens, device, forced_bos_token_id=forced_bos_token_id)
    with patched(layer_module, steering, output_index):
        patched_out = generate_text(model, tokenizer, prompt, max_new_tokens, device, forced_bos_token_id=forced_bos_token_id)
    return {"prompt": prompt, "baseline": baseline, "patched": patched_out}


def random_direction_like(direction: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """Control condition: is THIS learned direction responsible for the
    effect, or would any comparably-sized perturbation do the same thing?"""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    v = torch.randn(direction.shape, generator=gen)
    return v / (v.norm() + 1e-8)


def compare_direction_vs_random_control(
    model, tokenizer, layer_module, real_direction: torch.Tensor, prompt: str,
    alpha: float = 1.0, max_new_tokens: int = 20, device: str = "cpu",
    output_index: Optional[int] = 0, n_random_controls: int = 3, seed: int = 0,
    steering_cls=None, forced_bos_token_id: Optional[int] = None,
) -> dict:
    """Three-way comparison: unpatched baseline, patched with the REAL
    direction, patched with n_random_controls random directions of the
    same norm/alpha. If real reliably differs from random (which should
    differ minimally from itself across seeds), that's evidence the
    specific direction matters -- not just that perturbing this layer by
    this much changes output, which is true of almost any large enough
    vector. This function only returns text; scoring it against a real
    metric (e.g. order_probe accuracy, or relation_order_profile) is a
    separate step requiring an actually-trained model.

    steering_cls: which steering mechanism to wrap each direction in.
    Defaults to AdditiveSteering for backward compatibility, but see its
    docstring for why it's mathematically incapable of moving RELATIVE
    order between tokens (the same vector added everywhere cancels out of
    every pairwise comparison). Pass PositionalGradientSteering to
    actually test order-relevant hypotheses instead.

    forced_bos_token_id: see generate_text's docstring -- REQUIRED for
    coherent output from mBART-family models (found directly: without it,
    even the UNPATCHED baseline degenerates into repeating a single word,
    which would otherwise be mistaken for a steering effect)."""
    steering_cls = steering_cls or AdditiveSteering
    baseline = generate_text(model, tokenizer, prompt, max_new_tokens, device, forced_bos_token_id=forced_bos_token_id)

    with patched(layer_module, steering_cls(real_direction, alpha), output_index):
        real_patched = generate_text(model, tokenizer, prompt, max_new_tokens, device, forced_bos_token_id=forced_bos_token_id)

    random_outputs = []
    for i in range(n_random_controls):
        rand_dir = random_direction_like(real_direction, seed=seed + i)
        with patched(layer_module, steering_cls(rand_dir, alpha), output_index):
            random_outputs.append(generate_text(model, tokenizer, prompt, max_new_tokens, device, forced_bos_token_id=forced_bos_token_id))

    return {
        "prompt": prompt,
        "baseline": baseline,
        "real_direction_patched": real_patched,
        "random_direction_patched": random_outputs,
    }


# ---------------------------------------------------------------------------
# Direct order-probe diagnostic: measure whether steering moves the TRAINED
# ORDER PROBE'S OWN predicted ordering between two specific words, rather
# than inferring an order effect by reading generated text (which conflates
# order, lexical choice, and fluency -- see chat discussion of the "dog"/
# "cat" substitution that looked order-relevant but wasn't).
# ---------------------------------------------------------------------------

def _pool_hidden_states_by_word(hidden: torch.Tensor, word_ids: List[Optional[int]], n_words: int) -> torch.Tensor:
    """Average subword vectors into one embedding per word -- the SAME
    convention coordinator.py used to train the probes, so this stays
    apples-to-apples with what they were fit on."""
    hidden_dim = hidden.shape[-1]
    pooled = torch.zeros(n_words, hidden_dim)
    counts = torch.zeros(n_words)
    for sub_i, w_i in enumerate(word_ids):
        if w_i is None:
            continue
        pooled[w_i] += hidden[0, sub_i]
        counts[w_i] += 1
    return pooled / counts.clamp(min=1).unsqueeze(-1)


@torch.no_grad()
def _encode_words_at_layer(
    model, tokenizer, words: List[str], layer_index: int, device: str = "cpu"
) -> torch.Tensor:
    """Tokenize a pre-split list of words (matching coordinator.py's
    is_split_into_words=True training convention) and return one pooled
    embedding per word at the given hidden_states index. Any steering hook
    already registered on the model's encoder fires normally, since this
    is just a regular forward pass -- call this INSIDE a `with patched(...)`
    block to get the steered version."""
    is_seq2seq = getattr(model.config, "is_encoder_decoder", False)
    encoder = model.get_encoder() if is_seq2seq else model
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True).to(device)
    out = encoder(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"), output_hidden_states=True)
    hidden = out.hidden_states[layer_index]
    return _pool_hidden_states_by_word(hidden, enc.word_ids(batch_index=0), len(words))


@torch.no_grad()
def measure_pairwise_order_shift(
    model, tokenizer, layer_module, steering_fn, order_probe,
    words: List[str], word_a_idx: int, word_b_idx: int, layer_index: int,
    device: str = "cpu", output_index: Optional[int] = 0,
) -> dict:
    """
    The direct diagnostic: run the SAME trained order probe on word_a and
    word_b's embeddings, once unpatched and once with the given steering
    function active, and report both raw scores and which word the probe
    predicts comes first each time -- no generated text, no eyeballing.

    order_probe: an order_probe.OrderProbe instance with trained weights
    loaded (see run_steering_example.py for how to reconstruct one from a
    coordinator.py output file).

    Convention (matches order_probe.py's training target): HIGHER score =
    LATER position. So the predicted-earlier word is whichever of word_a/
    word_b has the LOWER score -- this holds regardless of whether they
    have any head/dependent relationship; it's just "which of these two
    does the probe think comes first."
    """
    order_probe.eval()

    baseline_pooled = _encode_words_at_layer(model, tokenizer, words, layer_index, device)
    s_a_base = order_probe(baseline_pooled[word_a_idx].unsqueeze(0)).item()
    s_b_base = order_probe(baseline_pooled[word_b_idx].unsqueeze(0)).item()

    with patched(layer_module, steering_fn, output_index):
        patched_pooled = _encode_words_at_layer(model, tokenizer, words, layer_index, device)
    s_a_patch = order_probe(patched_pooled[word_a_idx].unsqueeze(0)).item()
    s_b_patch = order_probe(patched_pooled[word_b_idx].unsqueeze(0)).item()

    def _predicted_first(s_a, s_b):
        return words[word_a_idx] if s_a < s_b else words[word_b_idx]

    baseline_first = _predicted_first(s_a_base, s_b_base)
    patched_first = _predicted_first(s_a_patch, s_b_patch)

    return {
        "word_a": words[word_a_idx], "word_b": words[word_b_idx],
        "baseline_score_a": s_a_base, "baseline_score_b": s_b_base,
        "baseline_predicted_first": baseline_first,
        "patched_score_a": s_a_patch, "patched_score_b": s_b_patch,
        "patched_predicted_first": patched_first,
        "order_flipped": baseline_first != patched_first,
        "baseline_gap": s_a_base - s_b_base,
        "patched_gap": s_a_patch - s_b_patch,
    }