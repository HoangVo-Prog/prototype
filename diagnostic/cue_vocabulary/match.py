"""Deterministic lexical cue matching and query-regex generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from diagnostic.cue_vocabulary.models import CueDefinition, CueEvidence, TrainingCaption
from diagnostic.cue_vocabulary.normalize import alias_sort_key, normalize_phrase, span_text, tokenize

_UPPER_BODY_CONTEXT = (
    "shirt",
    "t shirt",
    "tee",
    "top",
    "jacket",
    "coat",
    "hoodie",
    "sweater",
    "sweatshirt",
    "vest",
    "blazer",
    "suit",
    "uniform",
    "blouse",
    "jersey",
)
_LOWER_BODY_CONTEXT = (
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "skirt",
    "dress",
    "leggings",
)
_FOOTWEAR_CONTEXT = (
    "shoes",
    "shoe",
    "sneakers",
    "sneaker",
    "trainers",
    "trainer",
    "boots",
    "boot",
    "sandals",
    "sandal",
    "heels",
    "heel",
)
_CONTEXT_GROUPS = {
    "upper_body": _UPPER_BODY_CONTEXT,
    "lower_body": _LOWER_BODY_CONTEXT,
    "footwear": _FOOTWEAR_CONTEXT,
}


@dataclass(frozen=True)
class _SpanMatch:
    cue_id: str
    slot: str
    matched_alias: str
    matched_span: str
    start: int
    end: int

    @property
    def span_length(self) -> int:
        return self.end - self.start


def _find_alias_spans(tokens: list[str], alias: str) -> list[tuple[int, int]]:
    alias_tokens = alias.split()
    width = len(alias_tokens)
    if width == 0 or width > len(tokens):
        return []
    spans: list[tuple[int, int]] = []
    for start in range(0, len(tokens) - width + 1):
        if tokens[start : start + width] == alias_tokens:
            spans.append((start, start + width))
    return spans


def _resolve_phrase_matches(tokens: list[str], cue: CueDefinition) -> list[_SpanMatch]:
    matches: list[_SpanMatch] = []
    for alias in sorted(cue.aliases, key=alias_sort_key):
        for start, end in _find_alias_spans(tokens, alias):
            matches.append(
                _SpanMatch(
                    cue_id=cue.cue_id,
                    slot=cue.slot,
                    matched_alias=alias,
                    matched_span=span_text(tokens, start, end),
                    start=start,
                    end=end,
                )
            )
    return matches


def _resolve_longest_specific(matches: Iterable[_SpanMatch]) -> list[_SpanMatch]:
    kept: list[_SpanMatch] = []
    occupied: list[tuple[int, int]] = []
    seen_cues: set[str] = set()
    for match in sorted(matches, key=lambda item: (-item.span_length, item.start, item.cue_id, item.matched_alias)):
        if match.cue_id in seen_cues:
            continue
        if any(not (match.end <= start or match.start >= end) for start, end in occupied):
            continue
        kept.append(match)
        occupied.append((match.start, match.end))
        seen_cues.add(match.cue_id)
    return sorted(kept, key=lambda item: (item.start, item.end, item.cue_id))


def _context_candidates(group_name: str, extra_aliases: Iterable[str] = ()) -> tuple[str, ...]:
    base = list(_CONTEXT_GROUPS.get(group_name, ()))
    for alias in extra_aliases:
        normalized = normalize_phrase(alias)
        if normalized:
            base.append(normalized)
    return tuple(sorted(dict.fromkeys(base), key=alias_sort_key))


def _distance(color_start: int, color_end: int, ctx_start: int, ctx_end: int) -> int:
    if color_end <= ctx_start:
        return ctx_start - color_end
    if ctx_end <= color_start:
        return color_start - ctx_end
    return 0


def _resolve_color_match(tokens: list[str], cue: CueDefinition) -> list[_SpanMatch]:
    context_group = str(cue.matcher.get("context_group", "")).strip()
    if context_group not in _CONTEXT_GROUPS:
        raise ValueError(f"Color cue {cue.cue_id} has unsupported context_group '{context_group}'")
    max_token_distance = int(cue.matcher.get("max_token_distance", 3))
    context_aliases = _context_candidates(context_group, cue.matcher.get("extra_context_aliases", ()))
    color_matches: list[_SpanMatch] = []
    for alias in sorted(cue.aliases, key=alias_sort_key):
        alias_spans = _find_alias_spans(tokens, alias)
        if not alias_spans:
            continue
        for color_start, color_end in alias_spans:
            best: tuple[int, int, int, int, str] | None = None
            for context_alias in context_aliases:
                for ctx_start, ctx_end in _find_alias_spans(tokens, context_alias):
                    gap = _distance(color_start, color_end, ctx_start, ctx_end)
                    if gap > max_token_distance:
                        continue
                    span_start = min(color_start, ctx_start)
                    span_end = max(color_end, ctx_end)
                    candidate = (gap, span_end - span_start, span_start, span_end, context_alias)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                continue
            _, _, span_start, span_end, _ = best
            color_matches.append(
                _SpanMatch(
                    cue_id=cue.cue_id,
                    slot=cue.slot,
                    matched_alias=alias,
                    matched_span=span_text(tokens, span_start, span_end),
                    start=span_start,
                    end=span_end,
                )
            )
    color_matches.sort(key=lambda item: (item.start, item.end, item.cue_id))
    deduped: list[_SpanMatch] = []
    seen: set[str] = set()
    for match in color_matches:
        if match.cue_id in seen:
            continue
        deduped.append(match)
        seen.add(match.cue_id)
    return deduped


def match_training_caption(caption: TrainingCaption, cues: list[CueDefinition]) -> list[CueEvidence]:
    tokens = tokenize(caption.caption)
    phrase_matches_by_slot: dict[str, list[_SpanMatch]] = {}
    evidences: list[CueEvidence] = []
    for cue in cues:
        matcher_type = str(cue.matcher.get("type", "")).strip()
        if matcher_type == "phrase":
            phrase_matches_by_slot.setdefault(cue.slot, []).extend(_resolve_phrase_matches(tokens, cue))
        elif matcher_type == "color_context":
            for match in _resolve_color_match(tokens, cue):
                evidences.append(
                    CueEvidence(
                        cue_id=cue.cue_id,
                        dataset=caption.dataset,
                        identity_id=caption.identity_id,
                        image_ref=caption.image_ref,
                        caption=caption.caption,
                        matched_alias=match.matched_alias,
                        matched_span=match.matched_span,
                    )
                )
        else:
            raise ValueError(f"Unsupported matcher type '{matcher_type}' for cue {cue.cue_id}")

    for slot_matches in phrase_matches_by_slot.values():
        for match in _resolve_longest_specific(slot_matches):
            evidences.append(
                CueEvidence(
                    cue_id=match.cue_id,
                    dataset=caption.dataset,
                    identity_id=caption.identity_id,
                    image_ref=caption.image_ref,
                    caption=caption.caption,
                    matched_alias=match.matched_alias,
                    matched_span=match.matched_span,
                )
            )
    evidences.sort(key=lambda item: (item.cue_id, item.dataset, item.identity_id, item.matched_span, item.matched_alias))
    return evidences


def _phrase_fragment(aliases: tuple[str, ...]) -> str:
    alternatives = "|".join(re.escape(alias) for alias in sorted(aliases, key=alias_sort_key))
    return rf"(?:^|\s)(?:{alternatives})(?:\s|$)"


def build_case_regex(cue: CueDefinition) -> str:
    matcher_type = str(cue.matcher.get("type", "")).strip()
    if matcher_type == "phrase":
        return _phrase_fragment(tuple(normalize_phrase(alias) for alias in cue.aliases))
    if matcher_type == "color_context":
        colors = "|".join(re.escape(normalize_phrase(alias)) for alias in sorted(cue.aliases, key=alias_sort_key))
        context_group = str(cue.matcher.get("context_group", "")).strip()
        context_aliases = _context_candidates(context_group, cue.matcher.get("extra_context_aliases", ()))
        contexts = "|".join(re.escape(alias) for alias in context_aliases)
        window = int(cue.matcher.get("max_token_distance", 3))
        gap = rf"(?:\s+\w+){{0,{window}}}\s+"
        return (
            rf"(?:"
            rf"(?:^|\s)(?:{colors}){gap}(?:{contexts})(?:\s|$)"
            rf"|"
            rf"(?:^|\s)(?:{contexts}){gap}(?:{colors})(?:\s|$)"
            rf")"
        )
    raise ValueError(f"Unsupported matcher type '{matcher_type}' for cue {cue.cue_id}")
