"""Candidate catalog loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from diagnostic.cue_vocabulary.models import CueDefinition
from diagnostic.cue_vocabulary.normalize import normalize_phrase

ALLOWED_FAMILIES = (
    "clothing_footwear_color",
    "garment_footwear_type",
    "sleeve_length",
    "carried_item",
    "worn_accessory",
    "clothing_pattern",
)


def _require_non_empty_string(raw: dict[str, Any], field: str, prefix: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix} must define non-empty string field '{field}'")
    return value.strip()


def _require_string_list(raw: dict[str, Any], field: str, prefix: str) -> tuple[str, ...]:
    value = raw.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{prefix} must define non-empty list field '{field}'")
    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{prefix} field '{field}' must contain non-empty strings")
        normalized.append(normalize_phrase(item))
    return tuple(sorted(dict.fromkeys(normalized)))


def load_candidate_catalog(path: Path) -> tuple[int, list[CueDefinition]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Candidate catalog must be a YAML mapping: {path}")
    version = payload.get("version")
    if not isinstance(version, int):
        raise ValueError(f"Candidate catalog must define integer version: {path}")
    raw_cues = payload.get("cues")
    if not isinstance(raw_cues, list) or not raw_cues:
        raise ValueError(f"Candidate catalog must define non-empty 'cues' list: {path}")

    seen_ids: set[str] = set()
    definitions: list[CueDefinition] = []
    for index, raw_cue in enumerate(raw_cues):
        prefix = f"Candidate cue at index {index} in {path}"
        if not isinstance(raw_cue, dict):
            raise ValueError(f"{prefix} must be a mapping")
        cue_id = _require_non_empty_string(raw_cue, "id", prefix)
        if cue_id in seen_ids:
            raise ValueError(f"Duplicate cue id '{cue_id}' in {path}")
        seen_ids.add(cue_id)
        family = _require_non_empty_string(raw_cue, "family", prefix)
        if family not in ALLOWED_FAMILIES:
            raise ValueError(f"{prefix} has unsupported family '{family}'")
        matcher = raw_cue.get("matcher")
        if not isinstance(matcher, dict) or not matcher.get("type"):
            raise ValueError(f"{prefix} must define matcher mapping with a 'type'")
        definitions.append(
            CueDefinition(
                cue_id=cue_id,
                family=family,
                slot=_require_non_empty_string(raw_cue, "slot", prefix),
                value=_require_non_empty_string(raw_cue, "value", prefix),
                display_name=_require_non_empty_string(raw_cue, "display_name", prefix),
                runtime_expression=_require_non_empty_string(raw_cue, "runtime_expression", prefix),
                aliases=_require_string_list(raw_cue, "aliases", prefix),
                matcher=dict(matcher),
                broader_than=tuple(sorted(raw_cue.get("broader_than", ()))),
                narrower_than=tuple(sorted(raw_cue.get("narrower_than", ()))),
                incompatible_with=tuple(sorted(raw_cue.get("incompatible_with", ()))),
            )
        )
    return version, sorted(definitions, key=lambda cue: cue.cue_id)
