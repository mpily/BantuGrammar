# -*- coding: utf-8 -*-
"""
nllb_concord_rewrite_pipeline.py

The full pipeline, combining all five validated relations from this
session: nsubj (relation_probe, cont. 11), amod/det (AfriSUD `mod`+ADJ
probe, cont. 25), associative/case (concord-source probe, cont. 21),
acl:relcl (AfriSUD `mod@relcl` probe, cont. 24), nmod:poss (AfriSUD
`mod:poss` probe, cont. 24). obj is explicitly NOT included -- object
marking is optional in Swahili and this project never built an
identification mechanism for "does an object marker exist at all"
(cont. 17), so it stays an open gap, not silently assumed solved.

Two genuinely separate halves, kept separate on purpose:

  IDENTIFICATION -- fully general, NLLB-representation-based, the part
  validated all session: for a chosen TARGET noun and a sentence, scan
  every candidate word of each relation type and ask the relevant
  trained probe "does the target noun win as concord source, against
  every other noun in the sentence" (same win-condition as every
  external check this session used).

  REWRITE -- static, hand-verified Swahili concord tables (standard
  reference-grammar facts, not NLLB-derived, not a morpheme segmenter
  guessing at NLLB token boundaries -- the thing the user explicitly
  said to avoid, cont. "use the tokens of NLLB" directive). For
  nsubj/amod/det/case/poss, the OLD class is known (from gold feats,
  since every test case here is real gold-corpus text), so the OLD
  concord prefix STRING is known exactly from the table -- stripping a
  known string and prepending a new one is simple substitution, not
  segmentation. For acl:relcl, the relative marker sits INSIDE the verb
  AFTER the subject+tense prefixes (not at a word boundary), and this
  project never solved general morpheme-position-finding (cont. 12-16)
  -- so acl:relcl's rewrite is explicitly NOT attempted here; only its
  IDENTIFICATION is, and the gap is flagged in the output, the same
  honest treatment cont. 19 gave the "wa"/"mtiifu" cascade gaps.

Usage:
    python3 -u nllb_concord_rewrite_pipeline.py
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from conllu_utils import Sentence, load_conllu_file
from cross_lingual_embedding_decomposition import MODEL_ID
from nllb_concord_source_probe import (
    ConcordExample, train_concord_probe, LAYER,
)
from nllb_structural_probe_crosslingual_check import extract_embeddings
from relation_probe import build_relation_examples, make_label_vocab, train_relation_probe

# ---------------------------------------------------------------------------
# Static Swahili concord tables (standard reference-grammar facts, classes
# 1-10 -- the classes this project's gold data actually exercises).
# ---------------------------------------------------------------------------

VOWELS = set("aeiou")

SM_TABLE = {1: "a", 2: "wa", 3: "u", 4: "i", 5: "li", 6: "ya", 7: "ki", 8: "vi", 9: "i", 10: "zi"}
ADJ_BASE = {1: "m", 2: "wa", 3: "m", 4: "mi", 5: "", 6: "ma", 7: "ki", 8: "vi", 9: "n", 10: "n"}
ASSOC_TABLE = {1: "wa", 2: "wa", 3: "wa", 4: "ya", 5: "la", 6: "ya", 7: "cha", 8: "vya", 9: "ya", 10: "za"}

# Possessive-pronoun concord (2026-10-06): the prefix before the person/
# number stems -angu/-ako/-ake/-etu/-enu/-ao. This is NOT the same series
# as ADJ_BASE -- confirmed directly by the 2026-10-06 generalization test,
# where ADJ_BASE's class-9/10 "n-" wrongly failed to match a real "wako".
# It IS the same consonantal concord as ASSOC_TABLE minus the trailing
# vowel (wa->w, ya->y, la->l, cha->ch, vya->vy, za->z) -- standard
# reference-grammar fact (mti wangu cl.3, nchi yangu cl.9, kitabu changu
# cl.7, vitabu vyangu cl.8, nchi zangu cl.10, jicho langu cl.5, mtoto/
# watoto wangu cl.1/2). Every stem is vowel-initial, so -- unlike
# ADJ_BASE's adjective roots, which can be consonant-initial -- no
# vowel-alternation branch is needed here; the bare consonant/glide
# prefix is always the correct surface form.
POSS_TABLE = {1: "w", 2: "w", 3: "w", 4: "y", 5: "l", 6: "y", 7: "ch", 8: "vy", 9: "y", 10: "z"}

# Demonstrative concord (2026-10-06): a small CLOSED set of whole-word
# forms ("this"/proximal and "that"/distal series), standard reference-
# grammar facts. These do NOT decompose into prefix+root productively
# (e.g. "huyu"/"yule" have no shared stem with the other classes'
# forms) -- confirmed directly: ADJ_BASE's prefix-strip wrongly failed
# to match a real "hiki". Table lookup + substitution only, never a
# strip-and-reconstruct.
DEM_PROXIMAL_TABLE = {1: "huyu", 2: "hawa", 3: "huu", 4: "hii", 5: "hili",
                       6: "haya", 7: "hiki", 8: "hivi", 9: "hii", 10: "hizi"}
DEM_DISTAL_TABLE = {1: "yule", 2: "wale", 3: "ule", 4: "ile", 5: "lile",
                     6: "yale", 7: "kile", 8: "vile", 9: "ile", 10: "zile"}


def apply_adj_prefix(root: str, cls: int) -> str:
    """Adjective/possessive concord: prefix + root, with the standard
    vowel-initial-root alternations (ki->ch, m->mw, vi->vy, zero->j,
    n->ny)."""
    base = ADJ_BASE[cls]
    vowel_initial = root[:1].lower() in VOWELS
    if vowel_initial:
        alt = {"m": "mw", "ki": "ch", "vi": "vy", "": "j", "n": "ny"}.get(base, base)
        return alt + root
    return base + root if base else root


def rewrite_sm(word: str, old_cls: int, new_cls: int) -> Optional[str]:
    """Verb subject-prefix rewrite: strip the KNOWN old SM string (known
    because old_cls is known from gold feats, not guessed), prepend the
    new one. Simple substitution, not segmentation."""
    old_sm = SM_TABLE[old_cls]
    if not word.lower().startswith(old_sm):
        return None  # doesn't match expectation -- flag, don't guess
    return SM_TABLE[new_cls] + word[len(old_sm):]


def rewrite_adj_like(word: str, old_cls: int, new_cls: int) -> Optional[str]:
    """Adjective concord rewrite ONLY -- strip the KNOWN old prefix (with
    its vowel-alternation form), reconstruct with the new class's
    prefix. Despite this function's former docstring, it does NOT cover
    det or possessive concord -- those are different series (see
    rewrite_demonstrative, rewrite_poss below), confirmed wrong by the
    2026-10-06 generalization test (ADJ_BASE's prefixes failed to match
    real "hiki"/"wako"). Still used as the amod/det branch's FALLBACK
    for non-demonstrative determiners (e.g. "nyingine"/"kila"), which
    this project has not separately verified -- see rewrite_det_or_dem."""
    old_base = ADJ_BASE[old_cls]
    old_vowel_alt = {"m": "mw", "ki": "ch", "vi": "vy", "": "j", "n": "ny"}.get(old_base, old_base)
    for candidate_prefix in sorted({old_base, old_vowel_alt}, key=len, reverse=True):
        if candidate_prefix and word.lower().startswith(candidate_prefix):
            root = word[len(candidate_prefix):]
            return apply_adj_prefix(root, new_cls)
    if not old_base:  # class 5 consonant-initial: zero prefix, root == word
        return apply_adj_prefix(word, new_cls)
    return None


def rewrite_demonstrative(word: str, old_cls: int, new_cls: int) -> Optional[str]:
    """Demonstrative concord rewrite: exact-match the word against
    whichever of the two closed-set series (proximal "this" / distal
    "that") it belongs to for old_cls, then substitute that SAME
    series' form for new_cls. Table lookup, not prefix-stripping --
    these forms don't decompose productively (see DEM_*_TABLE comment)."""
    word_l = word.lower()
    if DEM_PROXIMAL_TABLE.get(old_cls) == word_l:
        return DEM_PROXIMAL_TABLE[new_cls]
    if DEM_DISTAL_TABLE.get(old_cls) == word_l:
        return DEM_DISTAL_TABLE[new_cls]
    return None  # neither known form for this class matched -- flag, don't guess


def rewrite_det_or_adj(word: str, old_cls: int, new_cls: int) -> Optional[str]:
    """DET-branch dispatcher: try the demonstrative closed-set lookup
    first (safe, exact-match, no guessing); fall back to the adjective-
    concord strip-and-reconstruct for other determiner types (e.g.
    "nyingine"/"kila") that this project has not separately verified."""
    dem = rewrite_demonstrative(word, old_cls, new_cls)
    if dem is not None:
        return dem
    return rewrite_adj_like(word, old_cls, new_cls)


def rewrite_poss(word: str, old_cls: int, new_cls: int) -> Optional[str]:
    """Possessive-pronoun concord rewrite: strip the KNOWN old
    POSS_TABLE prefix, keep the person/number stem (-angu/-ako/-ake/
    -etu/-enu/-ao) unchanged, prepend the new class's prefix. No
    vowel-alternation branch needed (see POSS_TABLE comment)."""
    old_prefix = POSS_TABLE[old_cls]
    if not word.lower().startswith(old_prefix):
        return None  # doesn't match expectation (e.g. an animacy-override
                      # noun using a different concord series) -- flag, don't guess
    return POSS_TABLE[new_cls] + word[len(old_prefix):]


def rewrite_assoc(old_cls: int, new_cls: int) -> str:
    """The associative marker IS the whole word (wa/ya/cha/...) -- no
    stem to strip, just substitute the whole token."""
    return ASSOC_TABLE[new_cls]


def rewrite_object_marker(word: str, old_cls: int, new_cls: int) -> Optional[str]:
    """2026-10-11, obj REWRITE (the original Stage 4/5 gap -- obj is the
    only relation excluded from this pipeline since 2026-10-09, when
    object-marker presence detection didn't exist yet; it does now,
    docs/status.md 2026-10-11).

    The object marker is INFIXED (SM-TAM-OM-ROOT-FV), not a simple
    prefix, so this can't reuse rewrite_sm's strip-and-prepend directly.
    It locates the KNOWN old OM among decompose_verb()'s own ENUMERATED
    candidates (extract_swahili_grammar.OM_TABLE, built from real gold
    NounClass[obj] data) and only proceeds if that EXACT candidate is
    actually present -- otherwise flags rather than guesses. "Currently
    carries the noun's concord" (CLAUDE.md's own filter condition) is a
    real, load-bearing check here, not a formality: confirmed directly
    that many real verbs with a gold NounClass[obj] feature carry NO
    object-marker string at all (object marking is optional in
    Swahili) -- this must come back None for those, not force a guess.

    Local import to avoid a circular dependency: extract_swahili_
    grammar.py already imports SM_TABLE FROM this module."""
    from extract_swahili_grammar import decompose_verb, OM_TABLE
    v = decompose_verb(word.lower())
    if v is None:
        return None
    old_om_set = {OM_TABLE[1], OM_TABLE["1_vowel"]} if old_cls == 1 else {OM_TABLE.get(old_cls)}
    match = next((c for c in v.object_marker_candidates if c[0] in old_om_set), None)
    if match is None:
        return None  # doesn't currently carry this concord -- flag, don't guess
    _, root, fv, suppress_tam = match
    if new_cls == 1:
        new_om = OM_TABLE["1_vowel"] if root[:1] in VOWELS else OM_TABLE[1]
    else:
        new_om = OM_TABLE.get(new_cls)
    if new_om is None:
        return None
    # suppress_tam=True means the matched "tam_marker" string was actually
    # THIS candidate's object marker (infinitives take no tense marker at
    # all) -- confirmed real collision, e.g. "kulifuatilia" ("li" = both
    # past-TAM and class-5 OM); the tense slot is empty in that reading,
    # not "tam_marker and then also the OM" (that double-counted "li" and
    # produced a real bug, caught before trusting this function).
    tam = "" if suppress_tam else v.tam_marker
    return v.subject_marker + tam + new_om + root + fv


def rewrite_relative_marker(word: str, old_cls: int, new_cls: int) -> Optional[str]:
    """2026-10-11, acl:relcl REWRITE -- the original "rewrite deliberately
    not attempted" gap (relative marker position inside the verb was
    unsolved, cont. 12-16 of 2026-10-05). Solved the same way as obj:
    locate the KNOWN old relative-marker string among decompose_verb()'s
    own enumerated candidates (extract_swahili_grammar.REL_TABLE, built
    from real gold NounClass[rel] data), swap it for the new class's
    string, leave everything else (subject marker, tense marker, and
    whatever comes after -- including any object marker further inside
    the word) UNCHANGED. Only the REL slot itself needs to change; no
    need to also decompose the remainder into OM+root+FV for this task.
    Returns None (flag, don't guess) if the expected old REL string
    isn't actually present -- confirmed directly that this happens for
    real (e.g. gold 'ilishawaambia' carries no detectable REL string
    despite being tagged NounClass[rel]).

    Local import to avoid a circular dependency: extract_swahili_
    grammar.py already imports SM_TABLE FROM this module."""
    from extract_swahili_grammar import decompose_verb, REL_TABLE
    v = decompose_verb(word.lower())
    if v is None:
        return None
    old_rel = REL_TABLE.get(old_cls)
    match = next((c for c in v.relative_marker_candidates if c[0] == old_rel), None)
    if match is None:
        return None  # doesn't currently carry this concord -- flag, don't guess
    _, remainder = match
    new_rel = REL_TABLE.get(new_cls)
    if new_rel is None:
        return None
    return v.subject_marker + v.tam_marker + new_rel + remainder


# ---------------------------------------------------------------------------
# Train the four rewrite-capable probes + the identification-only acl probe.
# ---------------------------------------------------------------------------

@dataclass
class Probes:
    nsubj_probe: object
    nsubj_label_to_id: dict
    amod_probe: object
    case_probe: object
    poss_probe: object
    acl_probe: object


def extract_direct_examples(sentences: List[Sentence], deprel: str, dep_upos=None,
                             max_negatives=3, seed=0) -> List[ConcordExample]:
    rng = random.Random(seed)
    examples = []
    for s_i, sent in enumerate(sentences):
        nouns = [t for t in sent.tokens if t.upos in ("NOUN", "PROPN")]
        for tok in sent.tokens:
            if tok.deprel != deprel:
                continue
            if dep_upos is not None and tok.upos not in dep_upos:
                continue
            head = sent.token_by_idx(tok.head)
            if head is None or head.upos not in ("NOUN", "PROPN"):
                continue
            examples.append(ConcordExample(s_i, head.idx, tok.idx, 1))
            distractors = [n for n in nouns if n.idx != head.idx]
            rng.shuffle(distractors)
            for d in distractors[:max_negatives]:
                examples.append(ConcordExample(s_i, d.idx, tok.idx, 0))
    return examples


def extract_concord_source_examples_case(sentences: List[Sentence], max_negatives=3, seed=0):
    """Same grandparent-traversal + feats-verification as cont. 21 --
    the associative/case relation genuinely needs this, unlike the
    others."""
    rng = random.Random(seed)
    examples = []
    for s_i, sent in enumerate(sentences):
        nouns = [t for t in sent.tokens if t.upos in ("NOUN", "PROPN")]
        for c in sent.tokens:
            if c.deprel != "case" or c.upos != "ADP":
                continue
            n2 = sent.token_by_idx(c.head)
            if n2 is None or n2.head == 0:
                continue
            n1 = sent.token_by_idx(n2.head)
            if n1 is None or n1.upos not in ("NOUN", "PROPN"):
                continue
            c_class, n1_class = c.feats.get("NounClass"), n1.feats.get("NounClass")
            if c_class is None or n1_class is None or c_class != n1_class:
                continue
            examples.append(ConcordExample(s_i, n1.idx, c.idx, 1))
            distractors = [n for n in nouns if n.idx != n1.idx]
            rng.shuffle(distractors)
            for d in distractors[:max_negatives]:
                examples.append(ConcordExample(s_i, d.idx, c.idx, 0))
    return examples


def train_all_probes(tokenizer, model, hidden_dim) -> Probes:
    print("=== training nsubj probe (SWH_UDT) ===", flush=True)
    sw_830 = load_conllu_file("Conlu/swahili_udt/GlobalVoices.ken.sw.8_30.filtered.conllu")
    sw_2050 = load_conllu_file("Conlu/swahili_udt/GlobalVoices.ken.sw.20_50.filtered.conllu")
    sw_sample = load_conllu_file("Conlu/swahili_udt/GlobalVoices.ken.sw.sample.conllu")
    sw_all = sw_830 + sw_2050 + sw_sample
    sw_sents, sw_embs = extract_embeddings(tokenizer, model, sw_all, "swh_Latn", LAYER,
                                            max_sentences=None, seed=0, target_norm=20.0)
    rel_examples = build_relation_examples(sw_sents)
    labels, label_to_id = make_label_vocab(rel_examples)
    nsubj_probe = train_relation_probe(sw_sents, sw_embs, rel_examples, label_to_id, hidden_dim, seed=0)[0]

    print("=== training associative/case probe (SWH_UDT, feats-verified) ===", flush=True)
    case_examples = extract_concord_source_examples_case(sw_sents, seed=0)
    case_probe = train_concord_probe(sw_sents, sw_embs, case_examples, hidden_dim, seed=0)

    print("=== training amod/det probe (AfriSUD mod+ADJ) ===", flush=True)
    afri_train = load_conllu_file("Conlu/afrisud_swahili/swa/swa_train.conllu")
    afri_dev = load_conllu_file("Conlu/afrisud_swahili/swa/swa_dev.conllu")
    afri_all = afri_train + afri_dev
    afri_sents, afri_embs = extract_embeddings(tokenizer, model, afri_all, "swh_Latn", LAYER,
                                                max_sentences=None, seed=0, target_norm=20.0)
    amod_examples = extract_direct_examples(afri_sents, "mod", dep_upos={"ADJ"}, seed=0)
    amod_probe = train_concord_probe(afri_sents, afri_embs, amod_examples, hidden_dim, seed=0)

    print("=== training nmod:poss probe (AfriSUD mod:poss) ===", flush=True)
    poss_examples = extract_direct_examples(afri_sents, "mod:poss", seed=0)
    poss_probe = train_concord_probe(afri_sents, afri_embs, poss_examples, hidden_dim, seed=0)

    print("=== training acl:relcl probe (AfriSUD mod@relcl, identification only) ===", flush=True)
    acl_examples = extract_direct_examples(afri_sents, "mod@relcl", seed=0)
    acl_probe = train_concord_probe(afri_sents, afri_embs, acl_examples, hidden_dim, seed=0)

    return Probes(nsubj_probe, label_to_id, amod_probe, case_probe, poss_probe, acl_probe)


# ---------------------------------------------------------------------------
# Identification + rewrite on a fresh sentence.
# ---------------------------------------------------------------------------

def pool_span(hidden, span, target_norm=20.0):
    s, e = span
    v = hidden[s:e].mean(dim=0, keepdim=True)
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6) * target_norm


def get_hidden_and_spans(tokenizer, model, sent: Sentence, lang_code="swh_Latn"):
    tokenizer.src_lang = lang_code
    text = sent.text if sent.text else " ".join(t.form for t in sent.tokens)
    per_word_counts = [len(tokenizer(f" {t.form}", add_special_tokens=False)["input_ids"])
                        for t in sent.tokens]
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt")
        out = model.get_encoder()(**enc, output_hidden_states=True)
    hidden = out.hidden_states[LAYER][0]
    hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp(min=1e-6) * 20.0
    ids_list = enc["input_ids"][0].tolist()
    start = ids_list.index(tokenizer.convert_tokens_to_ids(lang_code)) + 1
    spans, pos = [], start
    for n_sub in per_word_counts:
        if pos + n_sub > len(ids_list):
            return None, None
        spans.append((pos, pos + n_sub))
        pos += n_sub
    return hidden, spans


def get_hidden_and_spans_multilayer(tokenizer, model, sent: Sentence, layers: List[int],
                                     lang_code: str = "swh_Latn"):
    """2026-10-09 (cont. 11): multi-layer generalization of get_hidden_and_
    spans -- pools and concatenates SEVERAL encoder layers (each
    individually renormalized to norm=20 BEFORE concatenation, so no one
    layer dominates purely from a larger raw norm, same "measure the norm
    first" discipline as every other rescaling in this project) into one
    per-token feature vector of size hidden_dim * len(layers), instead of
    a single layer's hidden_dim. Still exactly ONE encoder forward pass
    (output_hidden_states=True already returns every layer) -- no decoder
    involved, so this stays in the "efficient, training-free-at-inference"
    tier, unlike the live self-loss check.

    Built to directly test the standing hypothesis (docs/status.md
    2026-10-09 cont. 9, attempt 3 / silver nsubj probe): that NLLB's
    teacher-forced self-loss signal for '-ki-' subject resolution is a
    full-decoder computation that may not be linearly present in a SINGLE
    pooled layer-12 encoder vector at all. If that's right, a probe trained
    on concatenated features from several depths may recover more of it
    than any one layer can -- if it's still absent even with 6 layers
    concatenated, that's real evidence for the "wrong representation
    entirely" reading over the "which single layer" reading."""
    tokenizer.src_lang = lang_code
    text = sent.text if sent.text else " ".join(t.form for t in sent.tokens)
    per_word_counts = [len(tokenizer(f" {t.form}", add_special_tokens=False)["input_ids"])
                        for t in sent.tokens]
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt")
        out = model.get_encoder()(**enc, output_hidden_states=True)
    layer_hiddens = []
    for L in layers:
        h = out.hidden_states[L][0]
        h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-6) * 20.0
        layer_hiddens.append(h)
    hidden = torch.cat(layer_hiddens, dim=-1)
    ids_list = enc["input_ids"][0].tolist()
    try:
        start = ids_list.index(tokenizer.convert_tokens_to_ids(lang_code)) + 1
    except ValueError:
        return None, None
    spans, pos = [], start
    for n_sub in per_word_counts:
        if pos + n_sub > len(ids_list):
            return None, None
        spans.append((pos, pos + n_sub))
        pos += n_sub
    return hidden, spans


def wins_as_concord_source(probe, hidden, spans, target_idx, marker_idx, all_nouns):
    """Same win-condition as every external check this session: does the
    target noun beat every OTHER candidate noun as this marker's concord
    source?"""
    h_marker = pool_span(hidden, spans[marker_idx - 1])
    with torch.no_grad():
        target_logit = probe(pool_span(hidden, spans[target_idx - 1]), h_marker)[0, 1].item()
        best_other = max(
            (probe(pool_span(hidden, spans[n.idx - 1]), h_marker)[0, 1].item()
             for n in all_nouns if n.idx != target_idx),
            default=float("-inf"))
    return target_logit > best_other


def wins_as_nsubj(nsubj_probe, nsubj_id, hidden, spans, verb_idx, target_idx, all_nouns):
    """2026-10-06 fix: nsubj previously had NO win-condition at all (see
    status.md 2026-10-06, the Kudai false positive) -- it took
    `relation_probe`'s argmax on the single (verb, target) pair in
    isolation, never comparing against competing candidate nouns the
    way every other relation branch does via wins_as_concord_source.
    `relation_probe` is trained only on REAL edges (build_relation_
    examples); a (verb, distant-unrelated-noun) pair like
    (Kudai, Wanaharakati) -- six dependency edges apart, no real nsubj
    edge exists for Kudai at all -- is out of that training
    distribution, and the classifier extrapolated to a confident but
    wrong "nsubj" in isolation.

    Two conditions now required, both necessary: (1) nsubj must be
    target_idx's own argmax-predicted label for this pair (kept from
    the original check), AND (2) target_idx's nsubj-class logit must
    beat every OTHER noun's nsubj-class logit for this same verb --
    the same relative, cross-candidate comparison every other relation
    branch already uses and that this project's external checks
    validated. This does not fix the underlying OOD-pair issue (the
    probe is still never trained on non-edge pairs) -- it makes the
    nsubj branch no more exposed to it than the other four branches,
    which is exactly the asymmetry the Kudai case exposed."""
    h_verb = pool_span(hidden, spans[verb_idx - 1])
    with torch.no_grad():
        logits_target = nsubj_probe(h_verb, pool_span(hidden, spans[target_idx - 1]))
        if logits_target.argmax(dim=-1).item() != nsubj_id:
            return False
        target_score = logits_target[0, nsubj_id].item()
        best_other = max(
            (nsubj_probe(h_verb, pool_span(hidden, spans[n.idx - 1]))[0, nsubj_id].item()
             for n in all_nouns if n.idx != target_idx),
            default=float("-inf"))
    return target_score > best_other


def run_pipeline(tokenizer, model, probes: Probes, sent: Sentence, target_idx: int, old_cls: int,
                  new_cls: int):
    disp_text = sent.text if sent.text else " ".join(t.form for t in sent.tokens)
    target_tok = sent.token_by_idx(target_idx)
    print(f"\n{'=' * 95}\nsentence: {disp_text!r}\ntarget noun: {target_tok.form!r} "
          f"(class {old_cls} -> hypothetically class {new_cls})\n{'=' * 95}", flush=True)

    hidden, spans = get_hidden_and_spans(tokenizer, model, sent)
    if hidden is None:
        print("  alignment failed -- skipping", flush=True)
        return
    all_nouns = [t for t in sent.tokens if t.upos in ("NOUN", "PROPN")]

    # edits keyed by TOKEN INDEX, not string content -- a blind text.replace(old, new)
    # is unsafe whenever old_form is a short substring that can also occur INSIDE other
    # words (e.g. replacing the associative "wa" as a substring corrupts "waziri" into
    # "chaziri" -- caught by inspecting the first run's output before trusting it).
    edits: Dict[int, tuple] = {}  # token_idx -> (description, old_form, new_form_or_None)

    # nsubj: does target_idx win as subject of any verb? (2026-10-06: now a
    # real win-condition, see wins_as_nsubj -- previously this had none,
    # see the Kudai false positive, status.md 2026-10-06)
    nsubj_id = probes.nsubj_label_to_id.get("nsubj")
    for verb in sent.tokens:
        if verb.upos != "VERB":
            continue
        if wins_as_nsubj(probes.nsubj_probe, nsubj_id, hidden, spans, verb.idx, target_idx, all_nouns):
            new_form = rewrite_sm(verb.form, old_cls, new_cls)
            edits[verb.idx] = (f"nsubj: {target_tok.form!r} -> {verb.form!r}", verb.form, new_form)

    # obj: does target_idx win as OBJECT of any verb, AND does that verb
    # CURRENTLY carry an object marker for old_cls? (2026-10-11: the
    # original Stage 4/5 gap -- excluded since 2026-10-09 for lack of a
    # presence-detection mechanism, which now exists, docs/status.md
    # 2026-10-11. Same "obj" label from the SAME nsubj_probe -- it was
    # already trained on this label, see train_and_save_identification_
    # probes.py's own note.) rewrite_object_marker legitimately returns
    # None about half the real time (object marking is optional) --
    # that's correctly displayed as FLAGGED/not-applied below, same as
    # acl:relcl, not an error.
    obj_id = probes.nsubj_label_to_id.get("obj")
    for verb in sent.tokens:
        if verb.upos != "VERB":
            continue
        if wins_as_nsubj(probes.nsubj_probe, obj_id, hidden, spans, verb.idx, target_idx, all_nouns):
            new_form = rewrite_object_marker(verb.form, old_cls, new_cls)
            edits[verb.idx] = (f"obj: {target_tok.form!r} -> {verb.form!r}", verb.form,
                                new_form if new_form else
                                "<IDENTIFIED but no object marker currently present -- nothing to rewrite>")

    # amod/det: ADJ/DET words headed by target_idx
    for tok in sent.tokens:
        if tok.upos not in ("ADJ", "DET") or tok.head != target_idx:
            continue
        if wins_as_concord_source(probes.amod_probe, hidden, spans, target_idx, tok.idx, all_nouns):
            new_form = rewrite_det_or_adj(tok.form, old_cls, new_cls)
            edits[tok.idx] = (f"amod/det: {target_tok.form!r} -> {tok.form!r}", tok.form, new_form)

    # nmod:poss: possessive words headed by target_idx
    for tok in sent.tokens:
        if tok.deprel != "nmod:poss" or tok.head != target_idx:
            continue
        if wins_as_concord_source(probes.poss_probe, hidden, spans, target_idx, tok.idx, all_nouns):
            new_form = rewrite_poss(tok.form, old_cls, new_cls)
            edits[tok.idx] = (f"nmod:poss: {target_tok.form!r} -> {tok.form!r}", tok.form, new_form)

    # associative/case: ADP "case" markers whose grandparent is target_idx
    for c in sent.tokens:
        if c.deprel != "case" or c.upos != "ADP":
            continue
        n2 = sent.token_by_idx(c.head)
        if n2 is None:
            continue
        if n2.head != target_idx:
            continue
        if wins_as_concord_source(probes.case_probe, hidden, spans, target_idx, c.idx, all_nouns):
            new_form = rewrite_assoc(old_cls, new_cls)
            edits[c.idx] = (f"case/assoc: {target_tok.form!r} -> {c.form!r}", c.form, new_form)

    # acl:relcl: 2026-10-11 -- REWRITE now solved (was "identification
    # only" since cont. 12-16 of 2026-10-05; see rewrite_relative_marker
    # and docs/status.md 2026-10-11 for the real gold NounClass[rel]
    # validation, 0/67 mismatches, 38/67 clean round-trips). Still
    # correctly returns None (flagged below, not a bug) for verbs that
    # genuinely don't carry a detectable REL string -- confirmed real,
    # several distinct causes (negation forms, the "-ja-...-zo" suffix
    # pattern, manner-"vyo" idiom, vowel elision before vowel-initial
    # roots, conjoined verbs that don't repeat the marker).
    for tok in sent.tokens:
        if tok.upos != "VERB" or tok.head != target_idx:
            continue
        if wins_as_concord_source(probes.acl_probe, hidden, spans, target_idx, tok.idx, all_nouns):
            new_form = rewrite_relative_marker(tok.form, old_cls, new_cls)
            edits[tok.idx] = (f"acl:relcl: {target_tok.form!r} -> {tok.form!r}", tok.form,
                               new_form if new_form else
                               "<IDENTIFIED but no relative marker currently detectable -- nothing "
                               "to rewrite (see docs/status.md 2026-10-11 for known causes)>")

    print("\nidentified + rewritten:", flush=True)
    for desc, old_form, new_form in edits.values():
        flag = "" if (new_form and not new_form.startswith("<")) else "  <-- FLAGGED, not applied"
        print(f"  {desc}: {old_form!r} -> {new_form!r}{flag}", flush=True)

    # Rebuild the sentence from its OWN token list, replacing only the token at the
    # exact identified index -- never a substring search over the full text.
    out_tokens = [t.form for t in sent.tokens]
    for idx, (desc, old_form, new_form) in edits.items():
        if new_form and not new_form.startswith("<"):
            out_tokens[idx - 1] = new_form
    rewritten_text = " ".join(out_tokens)
    print(f"\n  ORIGINAL:  {disp_text}", flush=True)
    print(f"  REWRITTEN: {rewritten_text}", flush=True)


def main():
    print(f"loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden_dim = model.config.d_model

    probes = train_all_probes(tokenizer, model, hidden_dim)

    # --- test case 1: nsubj + amod, the jaji/Madaktari-style sentence ---
    sw_830 = load_conllu_file("Conlu/swahili_udt/GlobalVoices.ken.sw.8_30.filtered.conllu")
    sw_2050 = load_conllu_file("Conlu/swahili_udt/GlobalVoices.ken.sw.20_50.filtered.conllu")
    sw_sample = load_conllu_file("Conlu/swahili_udt/GlobalVoices.ken.sw.sample.conllu")
    sw_all = sw_830 + sw_2050 + sw_sample

    sent1 = next(s for s in sw_all if "watapata" in [t.form for t in s.tokens])
    watu_idx = next(t.idx for t in sent1.tokens if t.form == "watu")
    run_pipeline(tokenizer, model, probes, sent1, watu_idx, old_cls=2, new_cls=1)

    sent2 = next(s for s in sw_all if "amefariki" in [t.form for t in s.tokens])
    waziri_idx = next(t.idx for t in sent2.tokens if t.form == "waziri")
    run_pipeline(tokenizer, model, probes, sent2, waziri_idx, old_cls=1, new_cls=7)

    # --- test case: relcl + poss, if findable ---
    sent3 = next((s for s in sw_all if "zilizotajwa" in [t.form for t in s.tokens]), None)
    if sent3 is not None:
        zile_idx = next(t.idx for t in sent3.tokens if t.form == "zile")
        run_pipeline(tokenizer, model, probes, sent3, zile_idx, old_cls=10, new_cls=7)

    # --- generalization check (2026-10-06): three NEW sentences, not seen by
    # cont. 26's hand-picked validation, chosen to exercise nsubj, amod/det
    # (including the DET pathway specifically -- never exercised before,
    # since cont. 26's amod example used an ADJ "mkuu", not a DET), and
    # nmod:poss (the one relation cont. 26 never ran at all).
    sent4 = sw_all[3]  # "Wanaharakati pia wanaziomba nchi nyingine ..." -- nsubj, 3 verbs in scope
    wanaharakati_idx = next(t.idx for t in sent4.tokens if t.form == "Wanaharakati")
    run_pipeline(tokenizer, model, probes, sent4, wanaharakati_idx, old_cls=2, new_cls=7)

    sent5 = sw_all[8]  # "Brita alitembelea kituo hiki na kugundua wajibu wake ..." -- det "hiki"
    kituo_idx = next(t.idx for t in sent5.tokens if t.form == "kituo")
    run_pipeline(tokenizer, model, probes, sent5, kituo_idx, old_cls=7, new_cls=9)

    sent6 = sw_all[271]  # "Usisahau mbwa wako , ni lazima waondoke ..." -- nmod:poss "wako"
    mbwa_idx = next(t.idx for t in sent6.tokens if t.form == "mbwa")
    run_pipeline(tokenizer, model, probes, sent6, mbwa_idx, old_cls=10, new_cls=6)


if __name__ == "__main__":
    main()
