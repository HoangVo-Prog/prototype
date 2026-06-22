"""Stable serialization helpers for the atomic cue builder."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from diagnostic.cue_vocabulary.datasets import DATASET_ORDER
from diagnostic.cue_vocabulary.models import CueDefinition, CuePair, CueSupportDecision, DatasetLoadResult, PairExclusion


def _stable_aliases(cue: CueDefinition) -> list[str]:
    return sorted(cue.aliases)


def cue_vocabulary_payload(
    *,
    catalog_version: int,
    retained_cues: list[CueDefinition],
    decisions: dict[str, CueSupportDecision],
    min_identities: int,
    min_datasets: int,
) -> dict[str, Any]:
    return {
        "version": catalog_version,
        "construction_rule": {
            "support_unit": "distinct_training_identity",
            "min_identities_per_dataset": min_identities,
            "min_datasets": min_datasets,
        },
        "cues": [
            {
                "id": cue.cue_id,
                "family": cue.family,
                "slot": cue.slot,
                "value": cue.value,
                "display_name": cue.display_name,
                "runtime_expression": cue.runtime_expression,
                "aliases": _stable_aliases(cue),
                "matcher": dict(cue.matcher),
                "broader_than": list(cue.broader_than),
                "narrower_than": list(cue.narrower_than),
                "incompatible_with": list(cue.incompatible_with),
                "support": {
                    dataset: int(decisions[cue.cue_id].counts_by_dataset[dataset])
                    for dataset in DATASET_ORDER
                },
                "datasets_meeting_threshold": list(decisions[cue.cue_id].datasets_meeting_threshold),
            }
            for cue in sorted(retained_cues, key=lambda item: item.cue_id)
        ],
    }


def cue_support_csv_rows(
    cues: list[CueDefinition],
    decisions: dict[str, CueSupportDecision],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue in sorted(cues, key=lambda item: (item.family, item.cue_id)):
        decision = decisions[cue.cue_id]
        rows.append(
            {
                "cue_id": cue.cue_id,
                "family": cue.family,
                "slot": cue.slot,
                "value": cue.value,
                "display_name": cue.display_name,
                "aliases": "; ".join(sorted(cue.aliases)),
                "cuhk_identity_count": int(decision.counts_by_dataset["CUHK-PEDES"]),
                "icfg_identity_count": int(decision.counts_by_dataset["ICFG-PEDES"]),
                "rstp_identity_count": int(decision.counts_by_dataset["RSTPReid"]),
                "datasets_meeting_threshold": "; ".join(decision.datasets_meeting_threshold),
                "num_datasets_meeting_threshold": len(decision.datasets_meeting_threshold),
                "retained": str(bool(decision.retained)).lower(),
                "decision_reason": decision.reason,
            }
        )
    return rows


def _cue_decision_rows(
    cues: list[CueDefinition],
    decisions: dict[str, CueSupportDecision],
) -> list[dict[str, Any]]:
    rows = []
    for cue in sorted(cues, key=lambda item: item.cue_id):
        decision = decisions[cue.cue_id]
        rows.append(
            {
                "cue_id": cue.cue_id,
                "family": cue.family,
                "slot": cue.slot,
                "value": cue.value,
                "display_name": cue.display_name,
                "aliases": list(sorted(cue.aliases)),
                "counts_by_dataset": {
                    dataset: int(decision.counts_by_dataset[dataset])
                    for dataset in DATASET_ORDER
                },
                "datasets_meeting_threshold": list(decision.datasets_meeting_threshold),
                "retained": bool(decision.retained),
                "decision_reason": decision.reason,
            }
        )
    return rows


def summary_payload(
    *,
    catalog_version: int,
    dataset_results: list[DatasetLoadResult],
    candidate_catalog_path: Path,
    candidate_catalog_sha256: str,
    cues: list[CueDefinition],
    retained_cues: list[CueDefinition],
    decisions: dict[str, CueSupportDecision],
    min_identities: int,
    min_datasets: int,
    seed: int,
    pairs: list[CuePair],
    exclusions: list[PairExclusion],
    output_hashes: dict[str, str],
    git_commit: str | None,
    git_dirty: bool | None,
) -> dict[str, Any]:
    retained_by_family: dict[str, int] = {}
    rejected_by_family: dict[str, int] = {}
    for cue in cues:
        bucket = retained_by_family if decisions[cue.cue_id].retained else rejected_by_family
        bucket[cue.family] = bucket.get(cue.family, 0) + 1
    return {
        "schema_version": catalog_version,
        "run_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "code_version_or_git_commit": git_commit,
        "git_worktree_dirty": git_dirty,
        "input_dataset_roots": {
            result.dataset_name: str(result.annotation_file.parent)
            for result in dataset_results
        },
        "resolved_training_annotation_files": {
            result.dataset_name: str(result.annotation_file)
            for result in dataset_results
        },
        "candidate_catalog_path": str(candidate_catalog_path),
        "candidate_catalog_sha256": candidate_catalog_sha256,
        "thresholds": {
            "support_unit": "distinct_training_identity",
            "min_identities_per_dataset": min_identities,
            "min_datasets": min_datasets,
        },
        "seed": seed,
        "per_dataset_num_training_identities": {
            result.dataset_name: result.num_training_identities
            for result in dataset_results
        },
        "per_dataset_num_training_captions": {
            result.dataset_name: result.num_training_captions
            for result in dataset_results
        },
        "num_candidate_cues": len(cues),
        "num_retained_cues": len(retained_cues),
        "num_rejected_cues": len(cues) - len(retained_cues),
        "retained_by_family": dict(sorted(retained_by_family.items())),
        "rejected_by_family": dict(sorted(rejected_by_family.items())),
        "num_all_possible_pairs": len(retained_cues) * max(len(retained_cues) - 1, 0) // 2,
        "num_semantically_excluded_pairs": len(exclusions),
        "num_generated_cases": len(pairs),
        "output_file_sha256_values": dict(sorted(output_hashes.items())),
        "all_cue_decisions": _cue_decision_rows(cues, decisions),
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    thresholds = summary["thresholds"]
    lines = [
        "# Atomic Cue Vocabulary Construction Summary",
        "",
        "The training-only construction rule and resulting vocabulary were frozen before the final reported diagnostic runs.",
        "",
        "## Rule",
        "",
        "- Training captions only: yes",
        "- Support unit: distinct training identity",
        f"- Rule: >={thresholds['min_identities_per_dataset']} identities in >={thresholds['min_datasets']} datasets",
        "- Test captions used: no",
        "- Retrieval outputs used: no",
        "",
        "## Inputs",
        "",
    ]
    for dataset in DATASET_ORDER:
        lines.append(f"- {dataset}: {summary['resolved_training_annotation_files'][dataset]}")
    lines.extend(
        [
            "",
            "## Corpus Sizes",
            "",
        ]
    )
    for dataset in DATASET_ORDER:
        lines.append(
            f"- {dataset}: identities={summary['per_dataset_num_training_identities'][dataset]}, "
            f"captions={summary['per_dataset_num_training_captions'][dataset]}"
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Candidate cues: {summary['num_candidate_cues']}",
            f"- Retained cues: {summary['num_retained_cues']}",
            f"- Rejected cues: {summary['num_rejected_cues']}",
            f"- All unordered retained pairs: {summary['num_all_possible_pairs']}",
            f"- Semantically excluded pairs: {summary['num_semantically_excluded_pairs']}",
            f"- Generated cases: {summary['num_generated_cases']}",
            "",
            "## Cue Decisions",
            "",
            "| cue_id | family | CUHK | ICFG | RSTP | meets threshold | retained | reason |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in summary["all_cue_decisions"]:
        counts = row["counts_by_dataset"]
        lines.append(
            f"| {row['cue_id']} | {row['family']} | {counts['CUHK-PEDES']} | {counts['ICFG-PEDES']} | "
            f"{counts['RSTPReid']} | {', '.join(row['datasets_meeting_threshold']) or '-'} | "
            f"{'yes' if row['retained'] else 'no'} | {row['decision_reason']} |"
        )
    return "\n".join(lines) + "\n"


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_json_as_yaml_subset(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
