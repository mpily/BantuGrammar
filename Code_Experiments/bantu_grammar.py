# -*- coding: utf-8 -*-
"""
bantu_grammar.py

Data structure for "as much grammar as can be gotten from text" for a
Bantu language, populated once for Swahili (donor, higher confidence --
known reference grammar + existing lexicon) and once for Kitaita
(target, lower confidence -- no prior lexicon/parser exists; everything
here is either a distributional pattern found in the collected Genesis
text or a hypothesis propagated from the verse-aligned Swahili side).
2026-10-06, first pass -- for native-speaker review before any
downstream use, not a finished grammar.

Every Kitaita-side entry carries an explicit `confidence` and `source`
field (never silently presented as equally certain as the Swahili
side, which has its own extensive independent sourcing already in this
project -- see swahili_noun_lexicon_agreement_v2.py, nllb_concord_
rewrite_pipeline.py's SM_TABLE/ADJ_BASE/ASSOC_TABLE).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NounClassEntry:
    """One noun, one hypothesized/known class membership."""
    lemma: str
    noun_class: str          # e.g. "1", "2", "9/10", "ANIMATE-SG", or "?unlabeled-cluster-3"
    gloss: Optional[str] = None
    confidence: str = "known"  # "known" | "lexicon" | "prefix-pattern" | "aligned" | "distributional"
    source: str = ""           # free text: where this came from
    count_in_corpus: int = 0


@dataclass
class ConcordForm:
    """One concord marker (subject prefix, adjective prefix, associative,
    possessive, demonstrative) attested or hypothesized for a class."""
    noun_class: str
    relation: str             # "subject" | "adjective" | "associative" | "possessive" | "demonstrative"
    form: str
    confidence: str = "known"
    example_word: Optional[str] = None


@dataclass
class VerbFormEntry:
    """One decomposed verb token: subject marker + TAM marker + root (+
    extensions) + final vowel, when a decomposition could be guessed."""
    surface: str
    subject_marker: Optional[str] = None
    tam_marker: Optional[str] = None
    tam_label: Optional[str] = None   # "past" | "present" | "future" | "perfect" | ... or None
    root_guess: Optional[str] = None
    final_vowel: Optional[str] = None
    confidence: str = "known"
    count_in_corpus: int = 1
    # 2026-10-11: candidate (object_marker, alternate_root, alternate_fv)
    # readings if a known object-marker string matches the start of the
    # post-TAM remainder -- NOT a decision, just enumerated possibilities.
    # root_guess/final_vowel above are UNCHANGED by this (every existing
    # caller keeps seeing exactly what it always did); callers that have
    # the evidence to choose (root-plausibility scoring against the
    # Kitaita dictionary) decide which candidate, if any, is real. See
    # extract_swahili_grammar.decompose_verb's docstring for why blind
    # stripping was rejected (same surface-collision risk as the '-ki-'
    # verb-detection bug fixed 2026-10-10).
    object_marker_candidates: list = field(default_factory=list)
    # 2026-10-11: candidate (relative_marker, remainder_after_rel) readings
    # if a known relative-marker string matches the start of the post-TAM
    # remainder -- same enumerate-don't-decide discipline, see
    # extract_swahili_grammar.REL_TABLE. Only populated for FINITE verb
    # forms (relative clauses need a conjugated verb) -- empty for the
    # bare-infinitive branch.
    relative_marker_candidates: list = field(default_factory=list)


@dataclass
class AlignmentHypothesis:
    """A candidate word-level correspondence between the two languages,
    from co-occurrence statistics over the parallel verses -- the ONLY
    source of Kitaita-side information that isn't a within-language
    distributional pattern or a typological guess."""
    kitaita_word: str
    swahili_word: str
    score: float              # co-occurrence-based association score
    kitaita_count: int
    swahili_count: int
    cooccur_count: int


@dataclass
class BantuGrammar:
    language_name: str
    iso_code: str
    noun_classes: Dict[str, List[NounClassEntry]] = field(default_factory=dict)
    concord_forms: List[ConcordForm] = field(default_factory=list)
    verb_forms: List[VerbFormEntry] = field(default_factory=list)
    tam_marker_inventory: Dict[str, dict] = field(default_factory=dict)  # marker -> {label, count, confidence}
    alignment_hypotheses: List[AlignmentHypothesis] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"=== BantuGrammar: {self.language_name} ({self.iso_code}) ==="]
        n_nouns = sum(len(v) for v in self.noun_classes.values())
        lines.append(f"noun classes: {len(self.noun_classes)} groups, {n_nouns} noun entries")
        lines.append(f"concord forms: {len(self.concord_forms)}")
        lines.append(f"verb forms decomposed: {len(self.verb_forms)}")
        lines.append(f"TAM markers: {len(self.tam_marker_inventory)}")
        lines.append(f"alignment hypotheses: {len(self.alignment_hypotheses)}")
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)
