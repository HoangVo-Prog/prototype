"""Matcher audit sampling."""

from __future__ import annotations

import random

from diagnostic.cue_vocabulary.datasets import DATASET_ORDER
from diagnostic.cue_vocabulary.models import CueDefinition, CueEvidence


def build_matcher_audit_rows(
    *,
    retained_cues: list[CueDefinition],
    evidence_by_cue: dict[str, dict[str, dict[str, CueEvidence]]],
    audit_samples_per_cue: int,
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cue in sorted(retained_cues, key=lambda item: item.cue_id):
        rng = random.Random(f"{seed}:{cue.cue_id}")
        per_dataset = {
            dataset: list(evidence_by_cue[cue.cue_id][dataset].values())
            for dataset in DATASET_ORDER
        }
        for values in per_dataset.values():
            values.sort(key=lambda evidence: (evidence.identity_id, evidence.caption, evidence.image_ref or "", evidence.matched_span))
            rng.shuffle(values)
        selected: list[CueEvidence] = []
        used = {(evidence.dataset, evidence.identity_id) for evidence in selected}
        for dataset in DATASET_ORDER:
            for evidence in per_dataset[dataset]:
                key = (evidence.dataset, evidence.identity_id)
                if key in used:
                    continue
                selected.append(evidence)
                used.add(key)
                break
        remaining = []
        for dataset in DATASET_ORDER:
            remaining.extend(per_dataset[dataset])
        remaining.sort(key=lambda evidence: (evidence.dataset, evidence.identity_id, evidence.caption, evidence.image_ref or "", evidence.matched_span))
        rng.shuffle(remaining)
        for evidence in remaining:
            if len(selected) >= audit_samples_per_cue:
                break
            key = (evidence.dataset, evidence.identity_id)
            if key in used:
                continue
            selected.append(evidence)
            used.add(key)
        selected = sorted(selected[:audit_samples_per_cue], key=lambda evidence: (evidence.dataset, evidence.identity_id, evidence.caption, evidence.image_ref or "", evidence.matched_span))
        for evidence in selected:
            rows.append(
                {
                    "cue_id": cue.cue_id,
                    "family": cue.family,
                    "dataset": evidence.dataset,
                    "identity_id": evidence.identity_id,
                    "image_id_or_path": evidence.image_ref or "",
                    "caption": evidence.caption,
                    "matched_alias": evidence.matched_alias,
                    "matched_span": evidence.matched_span,
                    "review_is_correct": "",
                    "review_note": "",
                }
            )
    return rows
