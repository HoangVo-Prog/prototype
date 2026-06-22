"""Exhaustive pair generation over retained cues."""

from __future__ import annotations

from itertools import combinations

from diagnostic.cue_vocabulary.match import build_case_regex
from diagnostic.cue_vocabulary.models import CueDefinition, CuePair, PairExclusion


def _exclusion_reason(cue_a: CueDefinition, cue_b: CueDefinition) -> str | None:
    if cue_b.cue_id in cue_a.broader_than or cue_b.cue_id in cue_a.narrower_than:
        return "generic_specific_relation"
    if cue_a.cue_id in cue_b.broader_than or cue_a.cue_id in cue_b.narrower_than:
        return "generic_specific_relation"
    if cue_b.cue_id in cue_a.incompatible_with or cue_a.cue_id in cue_b.incompatible_with:
        return "explicit_semantic_incompatibility"
    return None


def build_cue_pairs(retained_cues: list[CueDefinition]) -> tuple[list[CuePair], list[PairExclusion]]:
    pairs: list[CuePair] = []
    exclusions: list[PairExclusion] = []
    for cue_a, cue_b in combinations(sorted(retained_cues, key=lambda cue: cue.cue_id), 2):
        reason = _exclusion_reason(cue_a, cue_b)
        if reason is not None:
            exclusions.append(PairExclusion(cue_a_id=cue_a.cue_id, cue_b_id=cue_b.cue_id, reason=reason))
            continue
        regex_a = build_case_regex(cue_a)
        regex_b = build_case_regex(cue_b)
        query_regex = rf"(?=.*{regex_a})(?=.*{regex_b})"
        pairs.append(
            CuePair(
                case_id=f"{cue_a.cue_id}__vs__{cue_b.cue_id}",
                cue_a_id=cue_a.cue_id,
                cue_b_id=cue_b.cue_id,
                cue_a_runtime=cue_a.runtime_expression,
                cue_b_runtime=cue_b.runtime_expression,
                query_regex=query_regex,
            )
        )
    return pairs, exclusions
