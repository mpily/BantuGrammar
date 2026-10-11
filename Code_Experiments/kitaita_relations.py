# -*- coding: utf-8 -*-
"""
kitaita_relations.py

2026-10-11 (cont. 9). Direct user instruction: "the e2e pipeline code
is too complicated. Let's do it step by step. Step 1: write a function
that determines the relations from the probes."

This is STEP 1 ONLY, deliberately narrow: given a sentence and which
token positions are already known to be nouns/verbs, find each verb's
SUBJECT and OBJECT using the trained nsubj/obj relation probe. Nothing
else -- no candidate detection (that's a separate, earlier step), no
adjective/possessive targeting, no '-ki-' override rules, no
translation. Those are later steps, added only once this one is
confirmed correct.

Reuses the already-trained, already-validated probe-loading/encoding/
win-condition machinery from nllb_probe_identification.py (load_probes,
_make_sentence, _wins_as_label) rather than re-deriving it -- hard rule
#10 (extend, don't fork). What's NEW here is just the function below:
a single, isolated entry point with a small, readable body, instead of
being one piece embedded inside a much bigger function.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from nllb_probe_identification import load_probes, _make_sentence, _wins_as_label, find_ki_governor
from nllb_concord_rewrite_pipeline import pool_span, get_hidden_and_spans
from extract_swahili_grammar import decompose_verb
from kitaita_e2e_pipeline import (
    lookup_noun_kitaita, resolve_class_marker, _noun_lookup, compute_adjective_concord,
    ADJ_ROOTS, ADJ_PREFIXES, SPEECH_ACT_ROOTS, PERCEPTION_ROOTS, TEMPORAL_CONCESSIVE_CONJUNCTIONS,
)
from kitaita_sentence_generator import translate


@dataclass
class VerbRelations:
    subject_idx: int | None = None  # 0-indexed token position, or None if no noun won
    object_idx: int | None = None
    override_reason: str | None = None  # set when a '-ki-' rule overrode the plain probe's subject


class _N:
    """Minimal (.idx) stand-in _wins_as_label expects -- idx is 1-indexed."""
    __slots__ = ("idx",)

    def __init__(self, idx):
        self.idx = idx


def _ki_override_subject(tokens: list[str], verb_idx: int, noun_idxs: list[int],
                          resolved: dict[int, VerbRelations]) -> tuple[int | None, str | None]:
    """The 4 free syntactic rules this project validated for '-ki-'
    (participial) verbs specifically -- the ONE category the plain
    nsubj/obj probe is documented-unreliable for (only 5/226 such verbs
    have a gold nsubj dependent at all, too sparse to have taught it
    anything real; see docs/status.md 2026-10-09/10). Ported from
    kitaita_e2e_pipeline.py's decompose_sentence_with_probes() rather
    than re-derived (hard rule #10) -- same rules, same order, same
    governor probe. Does NOT include the self-loss fallback that
    function also had: self-loss needs each noun's KITAITA subject-
    marker to build its variant sentences, which isn't available yet
    at this step (that's step 2's job) -- left for later if needed.

    `resolved`: every EARLIER verb's already-finalized VerbRelations
    (verb_idxs are processed in order, so this is complete for every
    verb_idx' < verb_idx by the time this runs).

    Returns (override_idx, reason) -- both None if no rule fires (the
    plain probe's own answer is then left untouched)."""
    governor_idx = find_ki_governor(tokens, verb_idx, [i for i in range(len(tokens)) if i != verb_idx])
    preceding = [resolved[i] for i in sorted(resolved) if i < verb_idx]

    governor_verb = decompose_verb(tokens[governor_idx].lower()) if governor_idx is not None else None
    governor_is_resolved_clause = governor_idx is not None and governor_idx in resolved and governor_idx < verb_idx
    governor_clause = resolved[governor_idx] if governor_is_resolved_clause else None

    this_verb = decompose_verb(tokens[verb_idx].lower())
    this_root = this_verb.root_guess if this_verb else ""

    # Rule 1: reported-speech verb (sema-type) -- same subject as the
    # nearest preceding clause with a resolved subject.
    if any(this_root == r or this_root.endswith(r) for r in SPEECH_ACT_ROOTS):
        prev = next((c for c in reversed(preceding) if c.subject_idx is not None), None)
        if prev is not None:
            return prev.subject_idx, "reported-speech verb -- same subject as preceding clause"

    # Rule 2: periphrastic 'kuwa ... -ki-' -- the '-ki-' verb shares its
    # governing 'kuwa' auxiliary's own subject.
    if governor_clause is not None and governor_verb is not None and governor_verb.root_guess.endswith("kuw"):
        if governor_clause.subject_idx is not None:
            return governor_clause.subject_idx, "periphrastic 'kuwa' auxiliary -- subject shared by construction"

    # Rule 3: perception/cognition-verb object control -- "saw/found/
    # heard/remembered X doing Y": X (the governor's OBJECT) is the
    # '-ki-' verb's subject.
    if governor_clause is not None and governor_verb is not None:
        if any(governor_verb.root_guess == r or governor_verb.root_guess.endswith(r) for r in PERCEPTION_ROOTS):
            if governor_clause.object_idx is not None:
                return governor_clause.object_idx, "perception-verb object control -- governor's object is the subject"

    # Rule 4: temporal/concessive conjunction ('wakati'/'huku') with NO
    # fresh local noun between it and the '-ki-' verb -- default to the
    # same subject as the preceding clause (deliberately does NOT fire
    # when a local noun exists -- checked directly, most real cases have
    # their own local subject instead).
    if governor_idx is not None and tokens[governor_idx].lower() in TEMPORAL_CONCESSIVE_CONJUNCTIONS:
        local_fresh = [n for n in noun_idxs if governor_idx < n < verb_idx]
        if not local_fresh:
            prev = next((c for c in reversed(preceding) if c.subject_idx is not None), None)
            if prev is not None:
                return prev.subject_idx, "temporal/concessive conjunction, no local subject -- same as preceding clause"

    return None, None


def find_relations(tokens: list[str], noun_idxs: list[int], verb_idxs: list[int]) -> dict[int, VerbRelations]:
    """For each position in `verb_idxs`, finds which position in
    `noun_idxs` is its subject and which is its object.

    tokens: the full tokenized sentence (plain strings); noun_idxs/
        verb_idxs are 0-indexed positions into `tokens`, already
        identified as nouns/verbs by SOME EARLIER step -- not this
        function's job to find them.

    Every verb is scored independently against the SAME full noun
    pool (a noun can legitimately be the subject of more than one
    verb, e.g. a pro-dropped continuation -- "Mwalimu alisoma kitabu
    kisha akalala"). A noun wins a role only if it beats EVERY other
    candidate noun for that (verb, role) pair -- not just the highest
    score in isolation; this is the same win-condition validated
    throughout this project (plain argmax was never validated and is
    deliberately not used).

    2026-10-11 (cont. 10): direct user instruction -- "add the heuristic
    rules for -ki- participles to the find relations part of the
    process." For any verb whose TAM is '-ki-' specifically (the ONE
    category this project confirmed the plain probe is unreliable for),
    the plain probe's subject guess is checked against 4 free syntactic
    rules (_ki_override_subject, just above) and overridden if one
    fires -- same rules this project already validated in the old
    pipeline, moved to this step rather than re-derived.

    Returns {verb_idx: VerbRelations(subject_idx, object_idx,
    override_reason)}, one entry per verb in `verb_idxs`, PROCESSED IN
    ORDER (so a later verb's override rules can see an earlier verb's
    already-resolved clause). Returns {} if there are no nouns or no
    verbs, or if NLLB's subword tokenization didn't align with the word
    count for this sentence (checked, not assumed)."""
    result: dict[int, VerbRelations] = {}
    if not noun_idxs or not verb_idxs:
        return result

    tokenizer, model, probes = load_probes()
    sent = _make_sentence(tokens)
    hidden, spans = get_hidden_and_spans(tokenizer, model, sent)
    if hidden is None:
        return result

    nsubj_probe, label_to_id = probes["nsubj"], probes["nsubj_label_to_id"]
    if "nsubj" not in label_to_id or "obj" not in label_to_id:
        return result
    nsubj_id, obj_id = label_to_id["nsubj"], label_to_id["obj"]

    all_nouns = [_N(i + 1) for i in noun_idxs]  # 1-indexed, _wins_as_label's expected shape

    for verb_idx in sorted(verb_idxs):
        verb_tok_idx = verb_idx + 1
        rel = VerbRelations()

        for n in all_nouns:
            if _wins_as_label(nsubj_probe, nsubj_id, hidden, spans, verb_tok_idx, n.idx, all_nouns):
                rel.subject_idx = n.idx - 1
                break

        remaining = [n for n in all_nouns if n.idx - 1 != rel.subject_idx]
        for n in remaining:
            if _wins_as_label(nsubj_probe, obj_id, hidden, spans, verb_tok_idx, n.idx, remaining):
                rel.object_idx = n.idx - 1
                break

        v = decompose_verb(tokens[verb_idx].lower())
        if v is not None and v.tam_marker == "ki":
            override_idx, reason = _ki_override_subject(tokens, verb_idx, noun_idxs, result)
            if override_idx is not None:
                rel.subject_idx = override_idx
                rel.override_reason = reason

        result[verb_idx] = rel

    return result


def _adjective_stem(surface: str) -> str:
    """'mkubwa' -> 'kubwa', 'mdogo' -> 'dogo' -- strips a known concord
    prefix off a known adjective root. This is NOT a probe step (it's
    the same deterministic ADJ_ROOTS/ADJ_PREFIXES matching candidate-
    detection already uses elsewhere); reused here, not re-derived,
    because translate()/compute_adjective_concord() both expect the
    bare STEM ('kubwa'), not the inflected surface form ('mkubwa').
    Returns `surface` unchanged if it isn't a recognized inflected form
    of a known root."""
    matched = next((root for root in ADJ_ROOTS
                     if surface == root or any(surface == p + root for p in ADJ_PREFIXES if p)), None)
    return matched if matched is not None else surface


def find_adjective_targets(tokens: list[str], noun_idxs: list[int], adjective_idxs: list[int]) -> dict[int, int]:
    """STEP 2a. For each adjective position, which noun (0-indexed
    position) it modifies -- using the trained amod probe. Same shape
    as find_relations, just for a different relation (amod is a
    binary probe -- "does this noun govern this adjective's concord,
    yes/no" -- so the comparison is a plain best-score pick among the
    nouns, not the label-must-win-the-argmax check nsubj/obj needs).

    Returns {} if there are no nouns, no adjectives, or NLLB's subword
    tokenization didn't align with the word count."""
    result: dict[int, int] = {}
    if not noun_idxs or not adjective_idxs:
        return result

    tokenizer, model, probes = load_probes()
    sent = _make_sentence(tokens)
    hidden, spans = get_hidden_and_spans(tokenizer, model, sent)
    if hidden is None:
        return result

    amod_probe = probes["amod"]
    for adj_idx in adjective_idxs:
        adj_tok_idx = adj_idx + 1
        best_noun, best_score = None, float("-inf")
        with torch.no_grad():
            h_adj = pool_span(hidden, spans[adj_tok_idx - 1])
            for n_idx in noun_idxs:
                score = amod_probe(pool_span(hidden, spans[n_idx]), h_adj)[0, 1].item()
                if score > best_score:
                    best_score, best_noun = score, n_idx
        if best_noun is not None:
            result[adj_idx] = best_noun

    return result


def align_concord_and_tam(tokens: list[str], noun_idxs: list[int], verb_idxs: list[int],
                           adjective_idxs: list[int], tables) -> dict:
    """STEP 2. "Handle the concord alignment for every verb and
    adjective: first find the driver (subject and object) of the
    concord agreement, then translate it to Kitaita and find the class
    it belongs to, then substitute the concord agreement of that new
    class. Also substitute the TAM markers." (direct user spec)

    For each verb: the driver of its subject-marker concord is its
    SUBJECT (from find_relations, step 1). For each adjective: the
    driver of its concord is whichever noun -- subject OR object -- the
    amod probe says it modifies (find_adjective_targets, just above).
    Either driver is translated to Kitaita and its class resolved, then
    the actual class->concord/TAM substitution is done by calling
    kitaita_sentence_generator.translate() -- the EXISTING, native-
    speaker-validated engine that already has every verified SM/TAM
    table right (hard rule #10: this step's new contribution is
    correctly FINDING the driver via probes, not re-deriving
    translate()'s own substitution tables).

    Returns {verb_idx: {"pieces": {token_idx: kitaita_word_or_None, ...},
    "trace": [...]}}. DELIBERATELY NOT a reassembled sentence string --
    direct user correction (2026-10-11 cont. 9): an earlier version of
    this function returned translate()'s own "kitaita_sentence" field,
    which always assembles as subject+verb+[object]+adjective-TRAILING-
    AT-THE-END regardless of where the adjective actually sat in the
    input (that field is correct only for translate()'s OTHER caller,
    kitaita_sentence_batch.py's hand-built single-clause specs, which
    have no real word order to preserve) -- so "Mbwa mkubwa aliona..."
    came back with the adjective moved to the end, a bug this project
    had already solved once (kitaita_e2e_pipeline.py's assemble_in_
    order/assemble_multi_clause place each piece at its ORIGINAL token
    index instead). Reassembly in original word order is its own,
    separate step -- NOT done here. "pieces" is keyed by the ORIGINAL
    token index of the piece it replaces, exactly what a future
    reassembly step would need, without this function assembling
    anything itself. `pieces[idx]` is None where no Kitaita form could
    be computed (flagged in "trace", never silently guessed)."""
    noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy = tables

    # find_relations/find_adjective_targets run the sentence through NLLB
    # as given (original case matters for its own tokenization/alignment,
    # same convention decompose_sentence_with_probes uses -- tokens_raw
    # for the probes, lowercased only for table lookups below).
    relations = find_relations(tokens, noun_idxs, verb_idxs)
    adj_targets = find_adjective_targets(tokens, noun_idxs, adjective_idxs)
    lower = [t.lower() for t in tokens]

    results = {}
    for verb_idx in verb_idxs:
        rel = relations.get(verb_idx) or VerbRelations()
        verb_word = lower[verb_idx]
        v = decompose_verb(verb_word)
        if v is None:
            results[verb_idx] = {"error": f"could not decompose verb '{verb_word}'"}
            continue

        subject_word, subject_class = None, None
        if rel.subject_idx is not None:
            subject_word = lower[rel.subject_idx]
            hit = _noun_lookup(subject_word, noun_kitaita)
            subject_class = hit[0] if hit else None

        object_word = lower[rel.object_idx] if rel.object_idx is not None else None

        # An adjective driven by THIS verb's SUBJECT feeds translate()
        # directly (that's the only driver translate() itself knows how
        # to use). An adjective driven by the OBJECT is computed
        # separately below via compute_adjective_concord(), the same
        # generalized-target utility this project already built for
        # exactly this case.
        subject_adjective = None
        object_adjective = None
        for adj_idx, target_idx in adj_targets.items():
            if target_idx == rel.subject_idx:
                subject_adjective = {"stem": _adjective_stem(lower[adj_idx]), "class": subject_class}
            elif target_idx == rel.object_idx and object_word is not None:
                obj_kitaita = lookup_noun_kitaita(object_word, noun_kitaita)
                if obj_kitaita:
                    object_adjective = compute_adjective_concord(
                        obj_kitaita, object_word, _adjective_stem(lower[adj_idx]), noun_class_of, noun_kitaita)
                else:
                    object_adjective = (None, f"object '{object_word}' has no Kitaita translation -- "
                                               f"adjective concord cannot be computed")

        verb_dict = {"sm": v.subject_marker, "tam": v.tam_marker, "root": v.root_guess, "fv": v.final_vowel}
        translated = translate(subject_word, verb_dict, object_word, subject_adjective, tables,
                                subject_class=subject_class)
        comps = translated.get("components", {})

        pieces = {verb_idx: comps.get("kitaita_verb_word")}
        if rel.subject_idx is not None:
            pieces[rel.subject_idx] = comps.get("subject_noun") or subject_word
        if rel.object_idx is not None:
            pieces[rel.object_idx] = comps.get("object_noun") or object_word
        for adj_idx, target_idx in adj_targets.items():
            if target_idx == rel.subject_idx:
                pieces[adj_idx] = comps.get("adjective_kitaita")
            elif target_idx == rel.object_idx and object_adjective is not None:
                pieces[adj_idx] = object_adjective[0]

        results[verb_idx] = {"pieces": pieces, "trace": translated.get("trace", []),
                              "object_adjective_note": object_adjective[1] if object_adjective else None}

    return results


def collect_pieces(verb_results: dict) -> dict[int, str]:
    """Merges every verb clause's own "pieces" dict (align_concord_and_
    tam()'s return value, one entry per verb) into a single {token_idx:
    kitaita_word} mapping. Multiple clauses can each contribute a piece
    for the SAME position (e.g. a pro-dropped subject shared across two
    verbs) -- harmless, since they're expected to agree."""
    merged: dict[int, str] = {}
    for r in verb_results.values():
        for idx, word in r.get("pieces", {}).items():
            if word:
                merged[idx] = word
    return merged


def dictionary_substitute_remaining(tokens: list[str], covered_idxs, noun_kitaita: dict) -> dict[int, str]:
    """STEP 2.5 (between step 2 and step 3). Direct user instruction:
    "substitute all [other] components of the sentence to Kitaita
    using the dictionary lookup." Step 2 only produces pieces for
    positions it already reasoned about (a verb, its subject/object,
    an adjective's concord target) -- everything else (a noun that
    isn't anyone's subject or object, a possessor, any other word with
    no concord to compute) gets NO concord/class reasoning at all here,
    just a PLAIN dictionary lookup, same trust order as lookup_noun_
    kitaita() already uses (VERIFIED_NOUN_TRANSLATIONS first, then the
    statistical candidate table).

    covered_idxs: positions step 2 already produced a piece for --
    skipped here so this step never overrides a concord-driven answer
    with a plain lookup.

    Returns {token_idx: kitaita_word}, only for positions where a
    dictionary entry actually exists -- a position with no entry is
    simply absent (reassemble_sentence already falls back to the
    original Swahili word for anything missing; never guessed here)."""
    pieces = {}
    for i, tok in enumerate(tokens):
        if i in covered_idxs:
            continue
        kit = lookup_noun_kitaita(tok.lower(), noun_kitaita)
        if kit:
            pieces[i] = kit
    return pieces


def reassemble_sentence(tokens: list[str], pieces: dict[int, str]) -> str | None:
    """STEP 3. "Reassemble the pieces in original word order."

    Places every piece back at its ORIGINAL token position --- falling
    back to the original Swahili word wherever no piece exists for a
    position, never dropping a word. Purely positional placement, no
    reasoning of its own -- `pieces` is expected to already be the
    MERGED result of step 2 (collect_pieces) and step 2.5 (dictionary_
    substitute_remaining).

    Capitalizes the first word and appends a period, same convention
    kitaita_e2e_pipeline.py's own assemble_in_order() already uses
    (reused, not re-derived).

    Returns None if there are no tokens at all."""
    if not tokens:
        return None
    words = [pieces.get(i, tok) for i, tok in enumerate(tokens)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."
