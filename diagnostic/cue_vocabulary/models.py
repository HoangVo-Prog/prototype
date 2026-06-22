"""Typed models for atomic cue vocabulary construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class TrainingCaption:
    dataset: str
    identity_id: str
    caption: str
    image_ref: str | None = None
    annotation_file: str = ""
    split: str = "train"

    @property
    def namespaced_identity_id(self) -> str:
        return f"{self.dataset}::{self.identity_id}"


@dataclass(frozen=True)
class CueDefinition:
    cue_id: str
    family: str
    slot: str
    value: str
    display_name: str
    runtime_expression: str
    aliases: tuple[str, ...]
    matcher: Mapping[str, object]
    broader_than: tuple[str, ...] = ()
    narrower_than: tuple[str, ...] = ()
    incompatible_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class CueEvidence:
    cue_id: str
    dataset: str
    identity_id: str
    image_ref: str | None
    caption: str
    matched_alias: str
    matched_span: str


@dataclass(frozen=True)
class CueSupportDecision:
    cue_id: str
    counts_by_dataset: Mapping[str, int]
    datasets_meeting_threshold: tuple[str, ...]
    retained: bool
    reason: str


@dataclass(frozen=True)
class CuePair:
    case_id: str
    cue_a_id: str
    cue_b_id: str
    cue_a_runtime: str
    cue_b_runtime: str
    query_regex: str


@dataclass(frozen=True)
class PairExclusion:
    cue_a_id: str
    cue_b_id: str
    reason: str


@dataclass(frozen=True)
class DatasetLoadResult:
    dataset_name: str
    annotation_file: Path
    records: tuple[TrainingCaption, ...]
    num_training_identities: int
    num_training_captions: int
