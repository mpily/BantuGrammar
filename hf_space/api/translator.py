"""Adapter from the research pipeline's report dict to the API's shape.

Read-only with respect to the pipeline: it calls the same functions the
Gradio UI calls (translate_sentence_with_probes / translate_sentence) and
only re-labels what they return. No pipeline logic lives here.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from typing import Callable

from .util import normalize_input

# SlotRecord.slot_type -> UD-style relation between the slot and its controller
RELATION_OF_SLOT = {
    "subject_sm": "nsubj",
    "adjective": "amod",
    "possessive": "nmod:poss",
    "associative": "nmod",
    "demonstrative": "det",
    "ote": "det",
}
_SKIPPED_VERB_RE = re.compile(r"Verb '([^']+)': root translation SKIPPED ENTIRELY")
_PUNCT = ".,!?;:\"'“”"


def _decompositions(report: dict) -> list:
    d = report.get("decomposition")
    if d is None:
        return []
    return d if isinstance(d, list) else [d]


def _unknown_words(d) -> list[str]:
    """Words the pipeline itself says it has no Kitaita for, kept as Swahili:
    passthrough tokens, secondary nouns without a Kitaita form, and verbs
    whose translation was skipped entirely."""
    out = [tok for _, tok in getattr(d, "passthrough", [])]
    out += [lemma for _, lemma, kit in getattr(d, "secondary_nouns", []) if not kit]
    for f in getattr(d, "flags", []):
        m = _SKIPPED_VERB_RE.search(f)
        if m:
            out.append(m.group(1))
    return [w.strip(_PUNCT) for w in out if w and w.strip(_PUNCT)]


def mark_unknown(kitaita: str | None, unknown: list[str]) -> str | None:
    """Wraps each passed-through Swahili word in the output as [?word] so a
    reader never mistakes it for Kitaita."""
    if not kitaita or not unknown:
        return kitaita
    keys = {w.lower() for w in unknown}
    parts = []
    for word in kitaita.split(" "):
        core = word.strip(_PUNCT)
        if core and core.lower() in keys:
            lead = word[: len(word) - len(word.lstrip(_PUNCT))]
            trail = word[len(word.rstrip(_PUNCT)):]
            parts.append(f"{lead}[?{core}]{trail}")
        else:
            parts.append(word)
    return " ".join(parts)


def adapt_report(report: dict, vocabulary_gap_counter: Callable[[list[str]], int] | None = None) -> dict:
    """Pipeline report -> API fields (kitaita, slots, unknown_words, ...)."""
    decomps = _decompositions(report)
    slots, unknown = [], []
    for si, d in enumerate(decomps):
        tokens = getattr(d, "tokens", []) or []
        for s in getattr(d, "slots", []) or []:
            controller = tokens[s.target_idx] if s.target_idx is not None and s.target_idx < len(tokens) else None
            slots.append({
                "slot": s.swahili_surface,
                "controller": controller.strip(_PUNCT) if controller else None,
                "relation": RELATION_OF_SLOT.get(s.slot_type, s.slot_type),
                "rewritten_to": s.predicted_form,
                "verified_form": s.confidence == "verified",
                "slot_type": s.slot_type,
                "confidence": s.confidence,
                "token_index": s.token_idx,
                "sentence_index": si,
            })
        unknown += _unknown_words(d)
    flags = list(report.get("flags") or [])
    if not flags:
        for d in decomps:
            flags += list(getattr(d, "flags", []) or [])
    kitaita = report.get("kitaita_sentence") or None
    noun_class = None
    if len(decomps) == 1:
        noun_class = getattr(decomps[0], "subject_class", None)
    seen = set()
    unknown = [w for w in unknown if not (w.lower() in seen or seen.add(w.lower()))]
    return {
        "kitaita": kitaita,
        "kitaita_marked": mark_unknown(kitaita, unknown),
        "unknown_words": unknown,
        "vocabulary_gaps": vocabulary_gap_counter(flags) if vocabulary_gap_counter else len(unknown),
        "slots": slots,
        "flags": flags,
        "error": report.get("error"),
        "noun_class": str(noun_class) if noun_class is not None else None,
    }


class Translator:
    """Serialises pipeline calls (one loaded model, not thread-safe) and keeps
    an LRU cache keyed by (mode, normalized input)."""

    def __init__(self, fns: dict[str, Callable[[str], dict]], cache_size: int = 512,
                 vocabulary_gap_counter: Callable[[list[str]], int] | None = None,
                 pipeline_version: str = "unknown", loaded: bool = True):
        self.fns = fns
        self.cache_size = cache_size
        self.gap_counter = vocabulary_gap_counter
        self.pipeline_version = pipeline_version
        self.loaded = loaded
        self._cache: OrderedDict = OrderedDict()
        self._run_lock = threading.Lock()
        self._cache_lock = threading.Lock()

    def cached(self, text: str, mode: str) -> dict | None:
        key = (mode, normalize_input(text))
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def translate(self, text: str, mode: str) -> tuple[dict, bool, int]:
        """Returns (adapted result, cache_hit, latency_ms)."""
        t0 = time.perf_counter()
        hit = self.cached(text, mode)
        if hit is not None:
            return hit, True, int((time.perf_counter() - t0) * 1000)
        if mode not in self.fns:
            raise ValueError(f"unsupported mode {mode!r}")
        norm = normalize_input(text)
        with self._run_lock:
            try:
                report = self.fns[mode](norm)
                result = adapt_report(report, self.gap_counter)
            except Exception as e:  # one bad input must not take the API down
                result = {"kitaita": None, "kitaita_marked": None, "unknown_words": [], "vocabulary_gaps": 0,
                          "slots": [], "flags": [f"SERVER ERROR: {type(e).__name__}: {e}"],
                          "error": f"{type(e).__name__}: {e}", "noun_class": None}
                return result, False, int((time.perf_counter() - t0) * 1000)
        with self._cache_lock:
            self._cache[(mode, norm)] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result, False, int((time.perf_counter() - t0) * 1000)


def pipeline_translator(tables, cache_size: int = 512, pipeline_version: str = "unknown") -> Translator:
    """Real pipeline. Imported lazily so the api package (and its tests) don't
    require torch unless the real pipeline is used."""
    from kitaita_e2e_pipeline import (translate_sentence, translate_sentence_with_probes,
                                      _unknown_word_flag_count)
    return Translator(
        {"probes": lambda t: translate_sentence_with_probes(t, tables),
         "heuristic": lambda t: translate_sentence(t, tables)},
        cache_size=cache_size, vocabulary_gap_counter=_unknown_word_flag_count,
        pipeline_version=pipeline_version,
    )
