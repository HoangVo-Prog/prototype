"""Support counting for candidate cues."""

from __future__ import annotations

from collections import defaultdict

from diagnostic.cue_vocabulary.datasets import DATASET_ORDER
from diagnostic.cue_vocabulary.match import match_training_caption
from diagnostic.cue_vocabulary.models import CueDefinition, CueEvidence, CueSupportDecision, DatasetLoadResult


def _evidence_sort_key(evidence: CueEvidence) -> tuple[str, str, str, str, str]:
    return (
        evidence.caption,
        evidence.image_ref or "",
        evidence.matched_alias,
        evidence.matched_span,
        evidence.identity_id,
    )


def _decision_reason(
    *,
    dataset_counts: dict[str, int],
    datasets_meeting_threshold: tuple[str, ...],
    min_identities: int,
) -> str:
    dataset_summary = ", ".join(f"{dataset}={dataset_counts[dataset]}" for dataset in DATASET_ORDER)
    meets = len(datasets_meeting_threshold)
    if meets == 0:
        return f"excluded_support: no dataset meets >={min_identities} identities ({dataset_summary})"
    if meets == 1:
        return f"excluded_support: only 1 dataset meets >={min_identities} identities ({dataset_summary})"
    return f"retained: {meets} datasets meet >={min_identities} identities ({dataset_summary})"


def collect_support(
    *,
    dataset_results: list[DatasetLoadResult],
    cues: list[CueDefinition],
    min_identities: int,
    min_datasets: int,
) -> tuple[dict[str, CueSupportDecision], dict[str, dict[str, dict[str, CueEvidence]]]]:
    cue_by_id = {cue.cue_id: cue for cue in cues}
    if set(DATASET_ORDER) != {result.dataset_name for result in dataset_results}:
        raise ValueError("Expected exactly the three benchmark datasets when collecting support")

    evidence_by_cue: dict[str, dict[str, dict[str, CueEvidence]]] = {
        cue.cue_id: {dataset: {} for dataset in DATASET_ORDER}
        for cue in cues
    }
    for dataset_result in dataset_results:
        for record in dataset_result.records:
            if record.split != "train":
                raise RuntimeError(f"Non-training record reached support counter: {record}")
            for evidence in match_training_caption(record, cues):
                current = evidence_by_cue[evidence.cue_id][dataset_result.dataset_name].get(evidence.identity_id)
                if current is None or _evidence_sort_key(evidence) < _evidence_sort_key(current):
                    evidence_by_cue[evidence.cue_id][dataset_result.dataset_name][evidence.identity_id] = evidence

    decisions: dict[str, CueSupportDecision] = {}
    for cue_id, cue in sorted(cue_by_id.items()):
        dataset_counts = {
            dataset_name: len(evidence_by_cue[cue_id][dataset_name])
            for dataset_name in DATASET_ORDER
        }
        datasets_meeting_threshold = tuple(
            dataset_name
            for dataset_name in DATASET_ORDER
            if dataset_counts[dataset_name] >= min_identities
        )
        retained = len(datasets_meeting_threshold) >= min_datasets
        decisions[cue_id] = CueSupportDecision(
            cue_id=cue_id,
            counts_by_dataset=dataset_counts,
            datasets_meeting_threshold=datasets_meeting_threshold,
            retained=retained,
            reason=_decision_reason(
                dataset_counts=dataset_counts,
                datasets_meeting_threshold=datasets_meeting_threshold,
                min_identities=min_identities,
            ),
        )
    return decisions, evidence_by_cue
