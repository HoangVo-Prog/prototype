"""Atomic cue vocabulary construction utilities."""

from diagnostic.cue_vocabulary.audit import build_matcher_audit_rows
from diagnostic.cue_vocabulary.catalog import load_candidate_catalog
from diagnostic.cue_vocabulary.datasets import DATASET_ORDER, load_all_training_captions
from diagnostic.cue_vocabulary.match import build_case_regex, match_training_caption
from diagnostic.cue_vocabulary.models import (
    CueDefinition,
    CueEvidence,
    CuePair,
    CueSupportDecision,
    DatasetLoadResult,
    PairExclusion,
    TrainingCaption,
)
from diagnostic.cue_vocabulary.pairs import build_cue_pairs
from diagnostic.cue_vocabulary.serialize import (
    cue_support_csv_rows,
    cue_vocabulary_payload,
    markdown_summary,
    summary_payload,
    write_csv,
    write_json,
    write_json_as_yaml_subset,
    write_markdown,
    write_yaml,
)
from diagnostic.cue_vocabulary.support import collect_support

__all__ = [
    "DATASET_ORDER",
    "CueDefinition",
    "CueEvidence",
    "CuePair",
    "CueSupportDecision",
    "DatasetLoadResult",
    "PairExclusion",
    "TrainingCaption",
    "build_case_regex",
    "build_cue_pairs",
    "build_matcher_audit_rows",
    "collect_support",
    "cue_support_csv_rows",
    "cue_vocabulary_payload",
    "load_all_training_captions",
    "load_candidate_catalog",
    "markdown_summary",
    "match_training_caption",
    "summary_payload",
    "write_csv",
    "write_json",
    "write_json_as_yaml_subset",
    "write_markdown",
    "write_yaml",
]
