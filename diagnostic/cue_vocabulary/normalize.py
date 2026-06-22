"""Deterministic normalization helpers."""

from __future__ import annotations

import re
import unicodedata

HYPHEN_PATTERN = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\-]+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.lower()
    normalized = HYPHEN_PATTERN.sub(" ", normalized)
    normalized = NON_ALNUM_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def normalize_phrase(text: str) -> str:
    return normalize_text(text)


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def alias_sort_key(alias: str) -> tuple[int, int, str]:
    token_count = len(alias.split())
    return (-token_count, -len(alias), alias)


def span_text(tokens: list[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end])
