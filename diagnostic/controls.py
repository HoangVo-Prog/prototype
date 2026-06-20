"""Hardness-matched control gallery construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class HardnessControlBuild:
    galleries: dict[str, np.ndarray]
    diagnostics: dict[str, float]


def _nearest_without_replacement(
    target_scores: np.ndarray,
    candidate_indices: np.ndarray,
    full_scores: np.ndarray,
) -> np.ndarray:
    unused = list(int(index) for index in candidate_indices.tolist())
    chosen: list[int] = []
    for target in target_scores:
        if not unused:
            raise ValueError("not_enough_hardness_candidates")
        best_pos = min(
            range(len(unused)),
            key=lambda pos: (abs(float(full_scores[unused[pos]]) - float(target)), int(unused[pos])),
        )
        chosen.append(unused.pop(best_pos))
    return np.asarray(chosen, dtype=np.int64)


def _build_one_control(
    gallery_type: str,
    pid: int,
    cue_gallery: np.ndarray,
    positive_indices: np.ndarray,
    gallery_pids: np.ndarray,
    full_scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    cue_distractors = cue_gallery[gallery_pids[cue_gallery] != int(pid)].astype(np.int64)
    if len(cue_distractors) == 0:
        raise ValueError("cue_gallery_has_no_distractors")
    cue_set = set(int(index) for index in cue_gallery.tolist())
    candidate_pool = np.asarray(
        [index for index, gallery_pid in enumerate(gallery_pids) if int(gallery_pid) != int(pid) and index not in cue_set],
        dtype=np.int64,
    )
    if len(candidate_pool) < len(cue_distractors):
        candidate_pool = np.asarray(
            [index for index, gallery_pid in enumerate(gallery_pids) if int(gallery_pid) != int(pid)],
            dtype=np.int64,
        )
    if len(candidate_pool) < len(cue_distractors):
        raise ValueError("not_enough_hardness_candidates")
    target_order = np.lexsort((cue_distractors, full_scores[cue_distractors]))
    sorted_targets = cue_distractors[target_order]
    matched = _nearest_without_replacement(full_scores[sorted_targets], candidate_pool, full_scores)
    gallery = np.concatenate([positive_indices, matched]).astype(np.int64)
    if len(np.unique(gallery)) != len(gallery):
        raise ValueError("hardness_control_contains_duplicate_image_ids")
    prefix = "hm_a" if gallery_type == "a_dense" else "hm_b"
    return gallery, {
        f"{prefix}_mean_score_cue_subset": float(np.mean(full_scores[cue_distractors])),
        f"{prefix}_mean_score_control_subset": float(np.mean(full_scores[matched])),
        f"{prefix}_std_score_cue_subset": float(np.std(full_scores[cue_distractors])),
        f"{prefix}_std_score_control_subset": float(np.std(full_scores[matched])),
    }


def construct_hardness_matched_controls(
    pid: int,
    cue_galleries: dict[str, np.ndarray],
    gallery_pids: np.ndarray,
    full_scores: np.ndarray,
) -> tuple[Optional[HardnessControlBuild], Optional[str]]:
    positive_indices = np.flatnonzero(gallery_pids == int(pid)).astype(np.int64)
    try:
        hm_a, diag_a = _build_one_control("a_dense", pid, cue_galleries["a_dense"], positive_indices, gallery_pids, full_scores)
        hm_b, diag_b = _build_one_control("b_dense", pid, cue_galleries["b_dense"], positive_indices, gallery_pids, full_scores)
    except ValueError as exc:
        return None, str(exc)
    diagnostics = {**diag_a, **diag_b}
    return HardnessControlBuild(galleries={"hm_a": hm_a, "hm_b": hm_b}, diagnostics=diagnostics), None

