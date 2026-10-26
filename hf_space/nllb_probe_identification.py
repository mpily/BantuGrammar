# -*- coding: utf-8 -*-
"""
nllb_probe_identification.py

2026-10-09. Direct user directive: replace the heuristic, token-adjacency-
based relation identification in kitaita_e2e_pipeline.py with the already-
validated NLLB relation probes (relation_probe.py / nllb_concord_source_
probe.py, trained+saved by train_and_save_identification_probes.py) --
"we don't need to guess for verbs and adjectives etc. We already have
probes to do that for us."

ARCHITECTURE (matches the user's spec exactly):
  1. Tokenize the sentence into words (reused: extract_swahili_grammar.
     tokenize()).
  2. Candidate nouns/verbs/adjectives are found with this project's
     EXISTING, already-effective part-of-speech-level machinery (the
     319-word lexicon + statistical noun table, decompose_verb()'s SM+TAM
     segmenter, the closed ADJ_ROOTS set) -- these are not probabilistic
     guesses about STRUCTURE, just closed-vocabulary/pattern lookups
     about which WORD is which kind of word, something this project
     already does reliably. No separate "POS probe" is built for this;
     see the note below for why.
  3. For the sentence's verb, the NLLB nsubj/obj relation probe scores
     EVERY candidate noun as subject and object (not just "whichever
     noun sits nearest") -- this is the real upgrade over the old
     adjacency heuristic, and is exactly how this project's own earlier
     research caught the Samak/waziri long-distance case that adjacency
     got wrong.
  4. For every adjective candidate, the NLLB amod probe scores EVERY
     candidate noun as its concord target -- also no longer restricted
     to physical adjacency.
  5. The result is the same (subject_idx, verb_idx, object_idx,
     adjective_idx-with-its-target-noun) structure kitaita_e2e_pipeline.
     py's Decomposition already uses -- substitution and reassembly are
     UNCHANGED (direct noun substitution; verb subject-marker from the
     subject noun's class + TAM from the rule table + root from
     vocabulary; adjective prefix from its target noun's class + root
     from the table; reassembled in the original word order). Only the
     IDENTIFICATION step changes.

WHY NO SEPARATE "POS PROBE": the user's spec names one, but checking
this project's own code first (hard rule: verify before building) found
no POS-tagging probe anywhere, and training one from scratch would mean
a SIXTH probe (another full corpus encoding + training pass) for a job
this project's lexicon/decompose_verb/ADJ_ROOTS already do deterministically
and well -- stated here explicitly as a scoping choice, not an oversight,
open to revisiting if the lexicon-based candidate detection itself turns
out to be the bottleneck once this is tested on real sentences.

WHY 'obj' NEEDED NO SEPARATE PROBE: checked directly in relation_probe.py
before writing train_and_save_identification_probes.py -- CORE_RELATIONS
already includes "obj" as one of the 21 labels the SAME multiclass model
trained for nsubj classifies. nsubj and obj identification both read off
one saved model (nsubj_obj_probe.pt), not two.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from conllu_utils import Sentence, Token
from cross_lingual_embedding_decomposition import MODEL_ID
from relation_probe import RelationProbe
from pos_probe import PosProbe
from nllb_concord_rewrite_pipeline import pool_span, get_hidden_and_spans, LAYER

PROBE_DIR = Path(__file__).parent / "saved_identification_probes"

_tokenizer = None
_model = None
_probes = None  # dict: nsubj, nsubj_label_to_id, amod, case, poss, acl, hidden_dim


def load_probes():
    """Lazy-loads NLLB + every saved probe. Raises FileNotFoundError with a
    clear message if train_and_save_identification_probes.py hasn't been
    run yet (or hasn't finished) -- never silently falls back to a guess."""
    global _tokenizer, _model, _probes
    if _probes is not None:
        return _tokenizer, _model, _probes
    if not (PROBE_DIR / "config.pt").exists():
        raise FileNotFoundError(
            f"No saved probes at {PROBE_DIR} -- run train_and_save_identification_probes.py "
            f"first (one-time, ~20-30 min on CPU, no GPU available on this machine).")
    config = torch.load(PROBE_DIR / "config.pt")
    hidden_dim = config["hidden_dim"]
    _tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    _model = AutoModelForSeq2SeqLM.from_pretrained(config["model_id"]).eval()
    for p in _model.parameters():
        p.requires_grad_(False)

    nsubj_label_to_id = torch.load(PROBE_DIR / "nsubj_obj_label_to_id.pt")
    nsubj_probe = RelationProbe(hidden_dim, len(nsubj_label_to_id))
    nsubj_probe.load_state_dict(torch.load(PROBE_DIR / "nsubj_obj_probe.pt"))
    nsubj_probe.eval()

    def _load_binary(name):
        m = RelationProbe(hidden_dim, 2)
        m.load_state_dict(torch.load(PROBE_DIR / f"{name}.pt"))
        m.eval()
        return m

    pos_path = PROBE_DIR / "pos_probe.pt"
    pos_probe, pos_label_to_id = None, None
    if pos_path.exists():
        pos_label_to_id = torch.load(PROBE_DIR / "pos_label_to_id.pt")
        pos_probe = PosProbe(hidden_dim, len(pos_label_to_id))
        pos_probe.load_state_dict(torch.load(pos_path))
        pos_probe.eval()

    governor_path = PROBE_DIR / "governor_attachment_probe.pt"
    governor_probe = _load_binary("governor_attachment_probe") if governor_path.exists() else None

    _probes = {
        "nsubj": nsubj_probe, "nsubj_label_to_id": nsubj_label_to_id,
        "amod": _load_binary("amod_probe"), "case": _load_binary("case_probe"),
        "poss": _load_binary("poss_probe"), "acl": _load_binary("acl_probe"),
        "pos": pos_probe, "pos_label_to_id": pos_label_to_id,
        "governor": governor_probe,
        "hidden_dim": hidden_dim,
    }
    return _tokenizer, _model, _probes


def find_ki_governor(tokens_raw: list[str], ki_verb_idx: int, candidate_idxs: list[int]) -> int | None:
    """2026-10-10. Finds the real governing head of a '-ki-' (participial)
    verb via the trained governor_attachment_probe, replacing the "nearest
    preceding verb-shaped token" STRING heuristic previously used by rules
    2/3 in kitaita_e2e_pipeline.py's decompose_sentence_with_probes().
    Validated directly before wiring in (docs/status.md 2026-10-10): on
    313/319 real gold '-ki-' verbs this project's current code detects
    across SWH_UDT+AfriSUD, the production heuristic agreed with the real
    gold head only 52.0% of the time (n=127, VERB-headed subset); this
    probe nets +7 cases over the heuristic on 62 held-out gold cases (13
    fixed, 6 regressed -- a real but not yet overwhelming improvement,
    reported honestly). tokens_raw/ki_verb_idx/candidate_idxs are all
    0-indexed, matching identify_relations()'s own convention. Scores
    EVERY candidate (not just preceding ones) -- same unrestricted pool
    the probe was trained on -- so callers must still check the winning
    index is something usable (e.g. an already-resolved preceding verb
    clause) before trusting it; this function only answers "which token
    looks most like the real governor," not "is that governor usable
    here," same "flag, don't guess" split as every other probe in this
    project. Returns None if the probe isn't trained yet, there are no
    candidates, or NLLB alignment fails for this sentence."""
    tokenizer, model, probes = load_probes()
    governor_probe = probes.get("governor")
    if governor_probe is None or not candidate_idxs:
        return None
    sent = _make_sentence(tokens_raw)
    hidden, spans = get_hidden_and_spans(tokenizer, model, sent)
    if hidden is None:
        return None
    with torch.no_grad():
        h_ki = pool_span(hidden, spans[ki_verb_idx])
        scores = {c: governor_probe(pool_span(hidden, spans[c]), h_ki)[0, 1].item() for c in candidate_idxs}
    return max(scores, key=scores.get)


def tag_pos(tokens_raw: list[str]) -> list[str] | None:
    """2026-10-09, direct user directive after 'mdogo' (a real adjective)
    was silently swallowed as a noun because candidate detection only ever
    recognized the 2 adjectives this project has a hardcoded Kitaita
    concord table for: "build the POS probe properly." Returns one UPOS-
    style tag per token (NOUN/PROPN/VERB/ADJ/PRON/OTHER), from NLLB's own
    representations via the trained pos_probe.py model -- not a lookup
    list, so a token's candidacy for a role doesn't depend on this project
    having pre-enumerated every Swahili word of that kind. Returns None if
    the POS probe hasn't been trained yet (train_and_save_pos_probe.py) or
    NLLB's subword tokenization didn't align with the word count -- the
    caller falls back to the lookup-based candidate detection, never
    silently guesses a tag."""
    tokenizer, model, probes = load_probes()
    if probes["pos"] is None:
        return None
    sent = _make_sentence(tokens_raw)
    hidden, spans = get_hidden_and_spans(tokenizer, model, sent)
    if hidden is None:
        return None
    id_to_label = {v: k for k, v in probes["pos_label_to_id"].items()}
    tags = []
    with torch.no_grad():
        for i in range(len(tokens_raw)):
            pred_id = probes["pos"](pool_span(hidden, spans[i])).argmax(dim=-1).item()
            tags.append(id_to_label[pred_id])
    return tags


def _make_sentence(tokens_raw: list[str]) -> Sentence:
    """Minimal Sentence/Token wrapper for FREE TEXT -- get_hidden_and_spans
    only ever reads .text, .tokens, and each token's .idx/.form, so no gold
    upos/deprel/head annotation is needed for identification to work."""
    toks = [Token(idx=i + 1, form=w, lemma=w, upos="", xpos="", feats={}, head=0, deprel="")
            for i, w in enumerate(tokens_raw)]
    return Sentence(sent_id=None, text=" ".join(tokens_raw), tokens=toks)


@dataclass
class ProbeIdentification:
    verb_results: dict = None  # verb_idx (0-indexed) -> {"subject_idx":..., "object_idx":...}
    adjective_targets: dict = None  # adjective_token_idx -> target_noun_idx
    possessive_targets: dict = None  # possessive_token_idx -> target_noun_idx
    notes: list = None


def identify_relations(tokens_raw: list[str], noun_idxs: list[int], verb_idxs: list[int],
                        adjective_idxs: list[int], possessive_idxs: list[int] | None = None) -> ProbeIdentification:
    """tokens_raw: the ORIGINAL (not lowercased) tokenized sentence.
    noun_idxs/adjective_idxs: 0-indexed positions already found by this
    project's existing lexicon/decompose_verb/ADJ_ROOTS/POS-probe candidate
    detection. verb_idxs: 0-indexed positions of EVERY verb found -- NOT
    just a chosen "primary" one (2026-10-09, direct user directive: "remove
    the one verb restriction. we don't need it. we can handle multiple
    verbs"). Each verb gets its OWN subject/object identified independently
    against the SAME candidate noun pool (a noun can legitimately be the
    shared, pro-dropped subject of more than one verb in a sequence, e.g.
    "Mwalimu alisoma kitabu kisha akalala" -- "mwalimu" wins as nsubj for
    BOTH verbs, which is correct, not a conflict to resolve).

    Returns, per verb, which noun wins as nsubj and which wins as obj
    (excluding whichever won nsubj for THAT verb), and for each adjective
    candidate (sentence-wide, not per-verb), which noun wins as its amod
    concord source -- using cross-candidate comparison (wins_as_nsubj/
    wins_as_concord_source's own win-condition: beats EVERY other
    candidate, not just a bare argmax), the same validated method this
    project's own research used throughout. One NLLB encoding pass for
    the whole sentence, reused across every verb/adjective query."""
    result = ProbeIdentification(verb_results={}, adjective_targets={}, possessive_targets={}, notes=[])
    if not noun_idxs or not verb_idxs:
        return result

    tokenizer, model, probes = load_probes()
    sent = _make_sentence(tokens_raw)
    hidden, spans = get_hidden_and_spans(tokenizer, model, sent)
    if hidden is None:
        result.notes.append("NLLB subword tokenization didn't align with word count -- "
                             "probe identification skipped for this sentence.")
        return result

    # 1-indexed Token stand-ins for wins_as_nsubj/wins_as_concord_source's
    # own expected interface (.idx), built fresh from the 0-indexed inputs.
    class _N:
        __slots__ = ("idx",)
        def __init__(self, idx): self.idx = idx
    all_noun_toks = [_N(i + 1) for i in noun_idxs]

    nsubj_probe, nsubj_label_to_id = probes["nsubj"], probes["nsubj_label_to_id"]
    claimed_idxs = set()  # 1-indexed tok idx of nouns already used as an EARLIER verb's subject/object
    prev_subject_tok_idx = None  # the immediately preceding verb's own resolved subject (if any)
    for vi_pos, verb_idx in enumerate(verb_idxs):
        verb_tok_idx = verb_idx + 1
        subject_idx, object_idx = None, None

        # 2026-10-09 (cont. 14), direct user report: "Mtoto alipika mayai,
        # mbwa alipokuwa akilala." wrongly re-selected 'mtoto' (the FIRST
        # clause's already-claimed subject) as subject of 'alipokuwa'/
        # 'akilala' too, even though 'mbwa' sits immediately before and is
        # the obviously correct local subject -- the probe's own cross-
        # candidate win-condition has NO concept of clause locality, so a
        # strong, already-claimed subject from an EARLIER clause can beat
        # a fresh, genuinely-correct local candidate from a LATER one just
        # by scoring higher in absolute terms (this is the SAME underlying
        # failure family as the original Kudai/Wanaharakati false positive,
        # just triggered by comma-juxtaposed clauses with no sentence-
        # ending punctuation, so split_sentences() never applies).
        #
        # Fix, for every verb EXCEPT the first (the first verb's own
        # subject/object resolution is UNCHANGED -- this is the scope that
        # cont. 11's position-independence validation actually covers: one
        # real clause, no competing unrelated clause in the pool yet):
        # restrict subject candidacy to nouns NOT already claimed by an
        # earlier clause, positioned strictly between the PREVIOUS verb and
        # this one (the structurally plausible "new, local subject" window).
        # If that set is empty (no fresh local noun -- e.g. "kisha akalala",
        # where the only noun in the window is the previous clause's own,
        # already-claimed OBJECT), fall back to the single carried-forward
        # previous-clause subject, preserving legitimate pro-dropped-subject
        # continuation (e.g. "Mwalimu alisoma kitabu kisha akalala" --
        # 'mwalimu' correctly wins again). Object resolution is UNCHANGED
        # (still the full candidate pool) -- this fix is scoped to the
        # reported subject-leakage bug only, not a general rework.
        if vi_pos == 0:
            subj_candidates = all_noun_toks
        else:
            prev_verb_idx = verb_idxs[vi_pos - 1]
            local_fresh = [n for n in all_noun_toks
                           if n.idx not in claimed_idxs and prev_verb_idx < n.idx - 1 < verb_idx]
            if local_fresh:
                subj_candidates = local_fresh
            elif prev_subject_tok_idx is not None:
                subj_candidates = [n for n in all_noun_toks if n.idx == prev_subject_tok_idx]
            else:
                subj_candidates = all_noun_toks  # safety net; should not normally trigger

        if "nsubj" in nsubj_label_to_id:
            nsubj_id = nsubj_label_to_id["nsubj"]
            for n in subj_candidates:
                if _wins_as_label(nsubj_probe, nsubj_id, hidden, spans, verb_tok_idx, n.idx, subj_candidates):
                    subject_idx = n.idx - 1
                    break
        if "obj" in nsubj_label_to_id:
            obj_id = nsubj_label_to_id["obj"]
            remaining = [n for n in all_noun_toks if subject_idx is None or n.idx - 1 != subject_idx]
            for n in remaining:
                if _wins_as_label(nsubj_probe, obj_id, hidden, spans, verb_tok_idx, n.idx, remaining):
                    object_idx = n.idx - 1
                    break
        result.verb_results[verb_idx] = {"subject_idx": subject_idx, "object_idx": object_idx}

        if subject_idx is not None:
            claimed_idxs.add(subject_idx + 1)
            prev_subject_tok_idx = subject_idx + 1
        if object_idx is not None:
            claimed_idxs.add(object_idx + 1)

    amod_probe = probes["amod"]
    for adj_i in adjective_idxs:
        adj_tok_idx = adj_i + 1
        best_noun, best_score = None, float("-inf")
        with torch.no_grad():
            h_adj = pool_span(hidden, spans[adj_tok_idx - 1])
            for n in all_noun_toks:
                score = amod_probe(pool_span(hidden, spans[n.idx - 1]), h_adj)[0, 1].item()
                if score > best_score:
                    best_score, best_noun = score, n.idx - 1
        if best_noun is not None:
            result.adjective_targets[adj_i] = best_noun

    # 2026-10-11: possessive-pronoun targeting ("vyako" -> which noun does
    # "your" attach to?), using the ALREADY-TRAINED, ALREADY-VALIDATED
    # poss_probe (win-rate 1.000, n=15, nmod:poss) -- loaded since this
    # module's very first version but never actually called anywhere
    # until now. Same win-condition pattern as amod immediately above
    # (hard rule #10: reuse the pattern, don't re-derive it).
    poss_probe = probes.get("poss")
    if poss_probe is not None:
        for poss_i in (possessive_idxs or []):
            poss_tok_idx = poss_i + 1
            best_noun, best_score = None, float("-inf")
            with torch.no_grad():
                h_poss = pool_span(hidden, spans[poss_tok_idx - 1])
                for n in all_noun_toks:
                    score = poss_probe(pool_span(hidden, spans[n.idx - 1]), h_poss)[0, 1].item()
                    if score > best_score:
                        best_score, best_noun = score, n.idx - 1
            if best_noun is not None:
                result.possessive_targets[poss_i] = best_noun

    return result


def _wins_as_label(probe, label_id, hidden, spans, head_idx, target_idx, all_candidates):
    """Generalized wins_as_nsubj: does target_idx win AS label_id (argmax
    for the pair is label_id, AND its label_id-logit beats every other
    candidate's) against EVERY other candidate for the same head. Same
    win-condition as nllb_concord_rewrite_pipeline.wins_as_nsubj, just
    parameterized over which label (nsubj vs obj) instead of hardcoding
    nsubj -- obj is just another id in the SAME label vocabulary."""
    h_head = pool_span(hidden, spans[head_idx - 1])
    with torch.no_grad():
        logits_target = probe(h_head, pool_span(hidden, spans[target_idx - 1]))
        if logits_target.argmax(dim=-1).item() != label_id:
            return False
        target_score = logits_target[0, label_id].item()
        best_other = max(
            (probe(h_head, pool_span(hidden, spans[c.idx - 1]))[0, label_id].item()
             for c in all_candidates if c.idx != target_idx),
            default=float("-inf"))
    return target_score > best_other


def _batched_self_loss(tokenizer, model, bos_sw: int, texts: list[str]) -> list[float]:
    """2026-10-09 (cont. 11): the per-candidate self-loss check used to run
    ONE full tokenizer+model forward pass per candidate, sequentially. Since
    every candidate's variant text is the same sentence with only the verb's
    subject-marker prefix changed, this batches all variants into a SINGLE
    padded forward pass instead -- the model's own forward signature already
    supports a batch dimension, this is not an architecture change, just no
    longer throwing that capability away. Confirmed losses are numerically
    identical (matched to <1e-4) to the old sequential per-text computation
    on both the mbwa/kijana and mvulana/kalamu motivating sentences before
    this replaced the sequential loop; measured ~2x speedup at n=3 candidates,
    growing to ~2.4x at n=6 (docs/status.md 2026-10-09 cont. 11) -- fixed
    per-call tokenizer/forward overhead is amortized across the batch instead
    of paid once per candidate. Padding is right-side (confirmed from the
    tokenizer directly, not assumed) so labels[:, 0] = bos_sw is safe for
    every row; per-example loss is computed by hand from the logits (rather
    than relying on the model's own aggregate `.loss`, which pools ALL
    rows' token losses into one mean and would hide which candidate is
    better) with padded positions excluded via -100 the same way the
    sequential version implicitly excluded them by never including them."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    labels = tokenizer(texts, return_tensors="pt", padding=True)["input_ids"].clone()
    labels[:, 0] = bos_sw
    pad_id = tokenizer.pad_token_id
    labels_masked = labels.clone()
    labels_masked[labels == pad_id] = -100
    with torch.no_grad():
        logits = model(**inputs, labels=labels_masked).logits
    logp = torch.log_softmax(logits, dim=-1)
    gather_labels = labels_masked.clone()
    gather_labels[gather_labels == -100] = 0
    token_logp = logp.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels_masked != -100).float()
    return (-(token_logp * mask).sum(-1) / mask.sum(-1)).tolist()


def resolve_ki_subject_via_self_loss(tokens_lower: list[str], verb_idx: int, root: str, fv: str,
                                      noun_candidates: list) -> tuple:
    """2026-10-09, direct user directive: "Wire in the live self-loss
    check." For '-ki-' (participial/conditional) verbs specifically --
    confirmed to have a gold nsubj dependent in only 5/226 cases across
    this project's available treebanks, too sparse for the trained
    nsubj/obj probe to ever learn from, and confirmed separately that a
    probe DISTILLED from this same signal does not generalize either
    (selectivity +0.054, regressed on a case the raw signal got right)
    -- uses NLLB's own teacher-forced SELF-TRANSLATION LIKELIHOOD
    directly, at inference time, instead of any trained classifier.
    Validated 14/14 on hand-checked cases (2 original + 12 systematic,
    see docs/status.md 2026-10-09) before being wired in here.

    For each candidate noun, builds a variant of the sentence with the
    verb's subject marker set to THAT noun's own class marker, and
    scores how "natural"/self-consistent NLLB finds regenerating that
    exact variant as its own SW->SW translation target (lower loss =
    more natural). Returns (best_noun_idx, scores_dict) picking
    whichever candidate's variant NLLB finds most self-consistent, or
    (None, {}) if fewer than 2 candidates have distinct markers (no
    real contrast to resolve).

    EVIDENCE TIER, stated here so callers repeat it rather than assert
    this as settled fact: this reflects NLLB's own generative self-
    consistency preference, not independently verified Swahili or
    Kitaita grammar. Report it as such.

    EFFICIENCY, 2026-10-09 (cont. 11): all candidate variants are now
    scored in ONE batched forward pass (_batched_self_loss) instead of
    one sequential pass per candidate -- see that function's docstring
    for the validation this change was checked against before replacing
    the sequential loop."""
    tokenizer, model, probes = load_probes()
    bos_sw = tokenizer.convert_tokens_to_ids("swh_Latn")

    distinct_sms = {sm for _, sm in noun_candidates if sm}
    if len(distinct_sms) < 2:
        return None, {}

    idxs, texts = [], []
    for idx, sm in noun_candidates:
        if not sm:
            continue
        variant_tokens = list(tokens_lower)
        variant_tokens[verb_idx] = f"{sm}ki{root}{fv}"
        idxs.append(idx)
        texts.append(" ".join(variant_tokens) + ".")

    if not texts:
        return None, {}
    losses = _batched_self_loss(tokenizer, model, bos_sw, texts)
    scores = dict(zip(idxs, losses))
    best_idx = min(scores, key=scores.get)
    return best_idx, scores
