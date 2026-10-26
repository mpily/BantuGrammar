"""Promotion rule (config.yaml `promotion`).

A translation's Kitaita becomes `verified` when
  (a) >= min_distinct_speakers distinct speakers submit the identical
      normalized native_output, or
  (b) a trusted reviewer approves it (feedback with approve=True and a
      reviewer flag set server-side from a valid X-Reviewer-Key).
Otherwise any submitted native_output makes it `speaker_corrected`; with no
correction it stays `machine`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .util import normalize_kitaita


@dataclass
class Promotion:
    status: str                    # machine | speaker_corrected | verified
    kitaita: str | None = None     # the verified form (as first submitted), if verified
    method: str | None = None      # consensus | reviewer
    n_speakers: int = 0


def evaluate(feedback: list[dict], machine_kitaita: str | None, machine_has_unknown: bool,
             min_distinct_speakers: int = 2, trusted_reviewer_approval: bool = True) -> Promotion:
    """`feedback` = every stored feedback row for ONE translation, oldest first."""
    if trusted_reviewer_approval:
        for fb in feedback:
            if fb.get("approve") and fb.get("trusted_reviewer"):
                form = fb.get("native_output")
                if not form:
                    # approving the machine output as-is is only allowed when it
                    # contains no unknown (passed-through Swahili) words
                    if not machine_kitaita or machine_has_unknown:
                        continue
                    form = machine_kitaita
                return Promotion("verified", form, "reviewer", 1)

    speakers_by_form: dict[str, set[str]] = {}
    first_surface: dict[str, str] = {}
    for fb in feedback:
        out = fb.get("native_output")
        if not out:
            continue
        key = normalize_kitaita(out)
        if not key:
            continue
        speakers_by_form.setdefault(key, set()).add(fb["speaker_id"])
        first_surface.setdefault(key, out.strip())

    best = max(speakers_by_form.items(), key=lambda kv: len(kv[1]), default=None)
    if best and len(best[1]) >= min_distinct_speakers:
        return Promotion("verified", first_surface[best[0]], "consensus", len(best[1]))
    if speakers_by_form:
        return Promotion("speaker_corrected", None, None, 0)
    return Promotion("machine")


def pair_kind(swahili: str) -> str:
    return "word" if len(swahili.strip().rstrip(".!?").split()) == 1 else "sentence"


def pair_level(swahili: str) -> int:
    """Learner difficulty heuristic for the drills feed: 1 = single word,
    2 = sentence of <= 5 words, 3 = longer. Not a linguistic claim."""
    n = len(swahili.strip().split())
    return 1 if n <= 1 else (2 if n <= 5 else 3)
