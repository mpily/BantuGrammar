"""Pydantic request/response models. Contract: api/CONTRACT.md."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import API_VERSION

KitaitaStatus = Literal["machine", "speaker_corrected", "verified", "unknown"]
Channel = Literal["web", "sms", "ussd"]
MACHINE_LABEL = "machine translation — awaiting native-speaker review"


class _Versioned(BaseModel):
    api_version: str = API_VERSION


# --------------------------------------------------------------------- translate
class TranslateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    src: Literal["swh"] = "swh"
    tgt: Literal["dav"] = "dav"
    mode: Optional[Literal["probes", "heuristic"]] = None  # default from config.yaml

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class Slot(BaseModel):
    slot: str                      # Swahili surface carrying the concord
    controller: Optional[str]      # Swahili noun it agrees with (None = not resolved)
    relation: str                  # nsubj | amod | nmod:poss | nmod | det
    rewritten_to: Optional[str]    # Kitaita form produced (None = no rewrite attempted)
    verified_form: bool            # pipeline's own confidence == "verified"
    slot_type: str                 # pipeline SlotRecord.slot_type
    confidence: str                # pipeline SlotRecord.confidence
    token_index: int
    sentence_index: int = 0


class VerifiedAlternative(BaseModel):
    kitaita: str
    status: Literal["verified"] = "verified"
    verified_at: str


class TranslateResponse(_Versioned):
    id: str
    source: str
    swahili_mt: str
    kitaita: Optional[str]
    kitaita_marked: Optional[str]
    status: KitaitaStatus
    label: str
    unknown_words: list[str]
    vocabulary_gaps: int
    slots: list[Slot]
    flags: list[str]
    mode: str
    verified_alternative: Optional[VerifiedAlternative] = None
    pipeline_version: str
    latency_ms: int
    cached: bool


# --------------------------------------------------------------------- feedback
class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    translation_id: str = Field(min_length=3, max_length=64)
    stars: Optional[int] = Field(default=None, ge=1, le=5)
    native_output: Optional[str] = Field(default=None, max_length=1000)
    speaker_id: str = Field(min_length=1, max_length=128)
    comment: Optional[str] = Field(default=None, max_length=2000)
    channel: Channel = "web"
    speaker_name: Optional[str] = Field(default=None, max_length=120)
    name_consent: bool = False
    approve: bool = False  # honoured only with a valid X-Reviewer-Key header

    @field_validator("native_output", "comment", "speaker_name")
    @classmethod
    def _blank_to_none(cls, v):
        return v.strip() or None if isinstance(v, str) else v

    @model_validator(mode="after")
    def _something_to_say(self):
        if self.stars is None and self.native_output is None and not self.approve:
            raise ValueError("provide stars (1-5), native_output, or both")
        return self


class FeedbackResponse(_Versioned):
    id: str
    translation_id: str
    translation_status: KitaitaStatus
    promoted: bool


# --------------------------------------------------------------------- progress
class MeanCI(BaseModel):
    mean: Optional[float]
    ci95: Optional[tuple[float, float]]
    n: int
    low_n: bool


class RateCI(BaseModel):
    rate: Optional[float]
    ci95: Optional[tuple[float, float]]
    k: int
    n: int
    low_n: bool


class DayTrend(BaseModel):
    date: str
    translations: int
    feedback: int
    rated: int
    mean_stars: Optional[float]
    low_n: bool


class ProgressCounts(BaseModel):
    translations: int
    feedback: int
    unique_speakers: int
    verified_pairs: int


class ProgressResponse(_Versioned):
    counts: ProgressCounts
    mean_stars: MeanCI
    four_plus_rate: RateCI
    star_histogram: dict[str, int]
    exact_match: RateCI
    chrf: MeanCI
    trend: list[DayTrend]
    low_n_threshold: int
    generated_at: str


# --------------------------------------------------------------------- verified pairs
class VerifiedPair(BaseModel):
    id: str
    swahili: str
    kitaita: str
    kind: Literal["word", "sentence"]
    noun_class: Optional[str]
    level: int
    verified_at: str
    status: Literal["verified"] = "verified"


class VerifiedPairsResponse(_Versioned):
    pairs: list[VerifiedPair]


class HealthResponse(_Versioned):
    ok: bool
    pipeline_version: str
    pipeline_loaded: bool
    persistence: str
    at_configured: bool


class ErrorResponse(_Versioned):
    error: str
    detail: object = None
