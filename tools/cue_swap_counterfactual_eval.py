"""Cue-swap counterfactual gallery evaluation for text-based person search.

This script implements the diagnostic protocol described in AGENTS.md without
touching the training pipeline. It reuses the repository dataset classes,
transforms, model builder, checkpoint loader, and encoder methods.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from itertools import combinations
import json
import logging
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_NAMES = ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid")

GALLERY_TYPES = ("a_dense", "b_dense")


def _color_aliases(color: str) -> List[str]:
    if color == "black":
        return ["black", "dark"]
    if color == "gray":
        return ["gray", "grey"]
    return [color]


def _garment_aliases(garment: str) -> List[str]:
    if garment == "t-shirt":
        return ["t-shirt", "tshirt", "t shirt"]
    if garment == "pants":
        return ["pants", "trousers"]
    if garment == "sneakers":
        return ["sneakers", "trainers"]
    return [garment]


def _build_colored_cues(
    colors: Sequence[str],
    garments: Sequence[str],
    category: str,
    region: str,
) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    seen = set()
    for color in colors:
        canonical_color = "gray" if color == "grey" else color
        for garment in garments:
            canonical_garment = "t-shirt" if garment in {"tshirt", "t-shirt"} else garment
            canonical = f"{canonical_color} {canonical_garment}"
            if canonical in seen:
                continue
            seen.add(canonical)
            aliases = sorted(
                {
                    f"{color_alias} {garment_alias}"
                    for color_alias in _color_aliases(canonical_color)
                    for garment_alias in _garment_aliases(canonical_garment)
                }
            )
            cues.append(
                {
                    "name": canonical,
                    "category": category,
                    "region": region,
                    "aliases": aliases,
                    "exclude_aliases": [],
                }
            )
    return cues


def _build_pre_registered_cue_ontology() -> List[Dict[str, Any]]:
    colors = [
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "gray",
        "grey",
        "brown",
        "pink",
        "purple",
        "orange",
    ]
    ontology: List[Dict[str, Any]] = []
    ontology.extend(
        _build_colored_cues(
            colors=colors,
            garments=[
                "jacket",
                "coat",
                "hoodie",
                "shirt",
                "t-shirt",
                "sweater",
                "vest",
                "uniform",
                "top",
            ],
            category="upper_color_garment",
            region="upper_body",
        )
    )
    ontology.extend(
        _build_colored_cues(
            colors=colors,
            garments=["pants", "trousers", "jeans", "shorts", "skirt", "dress"],
            category="lower_color_garment",
            region="lower_body",
        )
    )
    ontology.extend(
        _build_colored_cues(
            colors=colors,
            garments=["shoes", "sneakers", "boots", "sandals"],
            category="footwear_color",
            region="feet",
        )
    )
    ontology.extend(
        [
            {
                "name": "bag",
                "category": "accessory_object",
                "region": "carried",
                "aliases": ["bag", "bags", "carrying a bag", "with a bag"],
                "exclude_aliases": [],
            },
            {
                "name": "backpack",
                "category": "accessory_object",
                "region": "carried",
                "aliases": ["backpack", "back pack", "rucksack"],
                "exclude_aliases": [],
            },
            {
                "name": "handbag",
                "category": "accessory_object",
                "region": "carried",
                "aliases": ["handbag", "hand bag", "purse"],
                "exclude_aliases": [],
            },
            {
                "name": "shoulder bag",
                "category": "accessory_object",
                "region": "carried",
                "aliases": ["shoulder bag"],
                "exclude_aliases": [],
            },
            {
                "name": "hat",
                "category": "accessory_object",
                "region": "head",
                "aliases": ["hat", "wearing a hat"],
                "exclude_aliases": [],
            },
            {
                "name": "cap",
                "category": "accessory_object",
                "region": "head",
                "aliases": ["cap", "baseball cap"],
                "exclude_aliases": [],
            },
            {
                "name": "glasses",
                "category": "accessory_object",
                "region": "head",
                "aliases": ["glasses", "sunglasses", "wearing glasses"],
                "exclude_aliases": [],
            },
            {
                "name": "umbrella",
                "category": "accessory_object",
                "region": "carried",
                "aliases": ["umbrella"],
                "exclude_aliases": [],
            },
            {
                "name": "striped",
                "category": "pattern",
                "region": "clothing",
                "aliases": ["striped", "stripe", "stripes"],
                "exclude_aliases": [],
            },
            {
                "name": "plaid",
                "category": "pattern",
                "region": "clothing",
                "aliases": ["plaid"],
                "exclude_aliases": [],
            },
            {
                "name": "checked",
                "category": "pattern",
                "region": "clothing",
                "aliases": ["checked", "checkered"],
                "exclude_aliases": [],
            },
            {
                "name": "printed",
                "category": "pattern",
                "region": "clothing",
                "aliases": ["printed", "print"],
                "exclude_aliases": [],
            },
            {
                "name": "logo",
                "category": "pattern",
                "region": "clothing",
                "aliases": ["logo"],
                "exclude_aliases": [],
            },
        ]
    )
    return ontology


PRE_REGISTERED_CUE_ONTOLOGY = _build_pre_registered_cue_ontology()


@dataclass(frozen=True)
class QueryRecord:
    query_id: int
    query_text: str
    pid: int


@dataclass
class SplitData:
    query_records: List[QueryRecord]
    gallery_pids: np.ndarray
    gallery_paths: List[str]


@dataclass
class EmbeddingCache:
    query_global: torch.Tensor
    gallery_global: torch.Tensor
    query_grab: Optional[torch.Tensor]
    gallery_grab: Optional[torch.Tensor]


@dataclass
class CueFeatures:
    global_features: Dict[str, torch.Tensor]
    grab_features: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class CueSpec:
    name: str
    aliases: Tuple[str, ...]
    category: str = ""
    region: str = ""
    exclude_aliases: Tuple[str, ...] = ()


class CaseValidationError(ValueError):
    """Raised when a cue-case file is malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cue-Swap Counterfactual Gallery Evaluation for TBPS"
    )
    parser.add_argument("--dataset", default="RSTPReid", choices=sorted(DATASET_NAMES))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--checkpoint", type=Path, help="Path to an existing best checkpoint")
    parser.add_argument(
        "--config",
        "--config_file",
        dest="config",
        type=Path,
        help="Training config YAML used to build the checkpoint model",
    )
    parser.add_argument(
        "--cases_file",
        type=Path,
        default=None,
        help="Manual cue case JSON file. If omitted, automatic cue case generation is used.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=None,
        help="Dataset root override. If omitted, the config value or repo default is used.",
    )
    parser.add_argument("--gallery_size", type=int, default=500)
    parser.add_argument("--dense_ratio", type=float, default=0.5)
    parser.add_argument("--num_random_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda_contrast", type=float, default=0.5)
    parser.add_argument("--cue_threshold_quantile", type=float, default=0.75)
    parser.add_argument("--tau_density", type=float, default=0.02)
    parser.add_argument("--tau_crowding", type=float, default=0.07)
    parser.add_argument("--max_queries_per_case", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--auto_cases",
        action="store_true",
        help=(
            "Force automatic pre-registered cue case generation. If --cases_file is omitted, "
            "this is enabled automatically."
        ),
    )
    parser.add_argument(
        "--cue_vocab_file",
        type=Path,
        default=None,
        help="Optional JSON cue vocabulary file. If omitted, use the built-in pre-registered TBPS cue ontology.",
    )
    parser.add_argument(
        "--min_queries_per_auto_case",
        type=int,
        default=40,
        help="Minimum number of eligible queries required for an automatically generated cue pair.",
    )
    parser.add_argument(
        "--max_auto_cases",
        type=int,
        default=None,
        help="Optional cap on number of auto-generated cases after sorting by support. Do not sort by retrieval outcomes.",
    )
    parser.add_argument(
        "--min_constructible_query_trials",
        type=int,
        default=1,
        help="Minimum number of constructible query-trials required to keep an auto-generated cue case.",
    )
    parser.add_argument(
        "--min_case_swap_strength",
        type=float,
        default=0.4,
        help=(
            "Optional manipulation-validity threshold. If set, keep an auto-generated case only when "
            "its mean swap strength over constructible dry construction trials is at least this value. "
            "This filter must not use retrieval metrics."
        ),
    )
    parser.add_argument(
        "--auto_case_max_queries_for_constructibility",
        type=int,
        default=50,
        help="Maximum queries per case used for the constructibility/manipulation-validity check.",
    )
    parser.add_argument(
        "--write_auto_cases_file",
        type=Path,
        default=None,
        help="Optional path to save the generated auto cases JSON. If omitted, save to output_dir/auto_cases_generated.json.",
    )
    parser.add_argument(
        "--score_mode",
        default="auto",
        choices=["auto", "global", "grab", "fusion"],
        help="Retrieval score branch. auto uses fusion when GRAB is available, else global.",
    )
    parser.add_argument(
        "--cue_mode",
        default="global",
        choices=["global", "grab", "fusion", "score"],
        help="Feature branch for label-free cue-image affinity.",
    )
    parser.add_argument(
        "--lambda_global",
        "--alpha_global",
        dest="lambda_global",
        type=float,
        default=0.68,
        help=(
            "Global-branch weight for fusion scores: "
            "lambda_global * s_global + (1 - lambda_global) * s_grab. "
            "--alpha_global is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument(
        "--only_global",
        action="store_true",
        help="Force model construction/evaluation to skip GRAB features.",
    )
    parser.add_argument(
        "--neutral_strategy",
        default="low_affinity",
        choices=["low_affinity", "random"],
        help="How to choose non-dense filler distractors.",
    )
    parser.add_argument(
        "--neutral_pool_factor",
        type=int,
        default=5,
        help="For low_affinity neutral fill, sample from k * factor lowest-affinity negatives.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cue_swap")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "cue_swap_eval.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def is_auto_case_mode(args: argparse.Namespace) -> bool:
    return bool(args.auto_cases)


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.cases_file is not None and args.auto_cases:
        raise ValueError("Provide either --cases_file or --auto_cases, not both.")
    if args.cases_file is None:
        args.auto_cases = True
    if args.gallery_size <= 0:
        raise ValueError("--gallery_size must be positive")
    if not 0.0 <= args.dense_ratio <= 1.0:
        raise ValueError("--dense_ratio must be in [0, 1]")
    if args.num_random_trials <= 0:
        raise ValueError("--num_random_trials must be positive")
    if not 0.0 <= args.cue_threshold_quantile <= 1.0:
        raise ValueError("--cue_threshold_quantile must be in [0, 1]")
    if args.tau_density <= 0:
        raise ValueError("--tau_density must be positive")
    if args.tau_crowding <= 0:
        raise ValueError("--tau_crowding must be positive")
    if not 0.0 <= args.lambda_global <= 1.0:
        raise ValueError("--lambda_global must be in [0, 1]")
    if args.max_queries_per_case is not None and args.max_queries_per_case <= 0:
        raise ValueError("--max_queries_per_case must be positive when provided")
    if args.neutral_pool_factor <= 0:
        raise ValueError("--neutral_pool_factor must be positive")
    if args.min_queries_per_auto_case <= 0:
        raise ValueError("--min_queries_per_auto_case must be positive")
    if args.max_auto_cases is not None and args.max_auto_cases <= 0:
        raise ValueError("--max_auto_cases must be positive when provided")
    if args.min_constructible_query_trials <= 0:
        raise ValueError("--min_constructible_query_trials must be positive")
    if args.auto_case_max_queries_for_constructibility <= 0:
        raise ValueError("--auto_case_max_queries_for_constructibility must be positive")
    if args.min_case_swap_strength is not None and args.min_case_swap_strength < 0:
        raise ValueError("--min_case_swap_strength must be >= 0 when provided")
    if not args.dry_run and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --dry_run is set")
    if args.checkpoint is not None and not args.checkpoint.exists() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config is not None and not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if args.cases_file is not None and not args.cases_file.exists():
        raise FileNotFoundError(f"Cue case file not found: {args.cases_file}")
    if args.cue_vocab_file is not None and not args.cue_vocab_file.exists():
        raise FileNotFoundError(f"Cue vocabulary file not found: {args.cue_vocab_file}")


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for CSV output. Install pandas or add it to the environment."
        ) from exc
    return pd


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_normalized_phrase(haystack: str, needle: str) -> bool:
    if not needle:
        return True
    return f" {needle} " in f" {haystack} "


def _normalized_unique(values: Iterable[str]) -> Tuple[str, ...]:
    normalized = {normalize_text(value) for value in values if isinstance(value, str)}
    normalized.discard("")
    return tuple(sorted(normalized, key=lambda value: (-len(value.split()), -len(value), value)))


def _cue_spec_from_raw(raw: Mapping[str, Any], index: int, source: str) -> CueSpec:
    prefix = f"Cue spec at index {index} in {source}"
    name = raw.get("name")
    aliases = raw.get("aliases")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{prefix} must have a non-empty string 'name'")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"{prefix} must have a non-empty list 'aliases'")
    if not all(isinstance(alias, str) and alias.strip() for alias in aliases):
        raise ValueError(f"{prefix} aliases must be non-empty strings")
    exclude_aliases = raw.get("exclude_aliases", [])
    if not isinstance(exclude_aliases, list):
        raise ValueError(f"{prefix} optional 'exclude_aliases' must be a list")
    if not all(isinstance(alias, str) and alias.strip() for alias in exclude_aliases):
        raise ValueError(f"{prefix} exclude_aliases must be non-empty strings")
    category = raw.get("category", "")
    region = raw.get("region", "")
    if not isinstance(category, str) or not isinstance(region, str):
        raise ValueError(f"{prefix} optional 'category' and 'region' must be strings")

    alias_values = list(aliases)
    if name not in alias_values:
        alias_values.append(name)
    return CueSpec(
        name=name.strip(),
        aliases=tuple(alias.strip() for alias in alias_values),
        category=category.strip(),
        region=region.strip(),
        exclude_aliases=tuple(alias.strip() for alias in exclude_aliases),
    )


def load_cue_specs(cue_vocab_file: Optional[Path]) -> List[CueSpec]:
    if cue_vocab_file is None:
        raw_specs = PRE_REGISTERED_CUE_ONTOLOGY
        source = "builtin"
    else:
        with cue_vocab_file.open("r", encoding="utf-8") as handle:
            raw_specs = json.load(handle)
        source = str(cue_vocab_file)

    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"Cue vocabulary {source} must contain a non-empty JSON list")

    cue_specs: List[CueSpec] = []
    seen_names = set()
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Cue spec at index {index} in {source} must be an object")
        spec = _cue_spec_from_raw(raw, index, source)
        if spec.name in seen_names:
            raise ValueError(f"Duplicate cue name '{spec.name}' in {source}")
        seen_names.add(spec.name)
        cue_specs.append(spec)
    return sorted(cue_specs, key=lambda spec: spec.name)


def detect_cues_in_query(query_text: str, cue_specs: Sequence[CueSpec]) -> List[str]:
    """
    Return sorted canonical cue names detected in query_text.
    Use aliases. Apply exclude_aliases to avoid obvious false positives.
    Deterministic output.
    """
    normalized_query = normalize_text(query_text)
    detected = set()
    for spec in cue_specs:
        exclude_aliases = _normalized_unique(spec.exclude_aliases)
        if any(contains_normalized_phrase(normalized_query, alias) for alias in exclude_aliases):
            continue
        aliases = _normalized_unique(spec.aliases)
        if any(contains_normalized_phrase(normalized_query, alias) for alias in aliases):
            detected.add(spec.name)
    return sorted(detected)


def detect_cues_for_queries(
    query_records: Sequence[QueryRecord],
    cue_specs: Sequence[CueSpec],
) -> Dict[int, List[str]]:
    return {
        record.query_id: detect_cues_in_query(record.query_text, cue_specs)
        for record in query_records
    }


def cue_vocab_source(args: argparse.Namespace) -> str:
    return str(args.cue_vocab_file) if args.cue_vocab_file is not None else "builtin"


def auto_cases_output_path(args: argparse.Namespace) -> Path:
    return args.write_auto_cases_file or (args.output_dir / "auto_cases_generated.json")


def write_auto_query_cues(
    output_dir: Path,
    query_records: Sequence[QueryRecord],
    detected_by_query: Mapping[int, Sequence[str]],
) -> None:
    pd = import_pandas()
    rows = []
    for record in query_records:
        detected = sorted(set(detected_by_query.get(record.query_id, [])))
        rows.append(
            {
                "query_id": record.query_id,
                "pid": record.pid,
                "query_text": record.query_text,
                "detected_cues": json.dumps(detected),
                "num_detected_cues": len(detected),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "auto_query_cues.csv", index=False)


def slugify_case_part(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_text(value))
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "cue"


def auto_case_id(cue_a: str, cue_b: str) -> str:
    left, right = sorted([cue_a, cue_b])
    return f"{slugify_case_part(left)}__{slugify_case_part(right)}"


def generate_candidate_cases_from_cooccurrence(
    query_records: Sequence[QueryRecord],
    detected_by_query: Mapping[int, Sequence[str]],
    gallery_pids: np.ndarray,
    min_queries: int,
    max_cases: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate candidate cue cases from all unordered cue pairs that co-occur
    in at least min_queries eligible queries.

    Eligibility at this stage:
    - query has at least two detected cues
    - query contains both cues
    - query pid has at least one positive image in the gallery

    Do not use retrieval metrics.
    """
    pids_with_gallery = set(int(pid) for pid in gallery_pids.tolist())
    pair_to_query_ids: Dict[Tuple[str, str], List[int]] = {}
    for record in query_records:
        if int(record.pid) not in pids_with_gallery:
            continue
        cues = sorted(set(detected_by_query.get(record.query_id, [])))
        if len(cues) < 2:
            continue
        for cue_a, cue_b in combinations(cues, 2):
            pair = tuple(sorted((cue_a, cue_b)))
            pair_to_query_ids.setdefault(pair, []).append(record.query_id)

    all_rows: List[Dict[str, Any]] = []
    supported_cases: List[Dict[str, Any]] = []
    for (cue_a, cue_b), query_ids in pair_to_query_ids.items():
        query_ids = sorted(set(int(query_id) for query_id in query_ids))
        case_id = auto_case_id(cue_a, cue_b)
        supported = len(query_ids) >= min_queries
        row = {
            "case_id": case_id,
            "cue_a": cue_a,
            "cue_b": cue_b,
            "num_queries": len(query_ids),
            "query_ids": json.dumps(query_ids),
            "meets_min_queries": supported,
            "kept_after_support": supported,
            "reason": "" if supported else "below_min_queries_per_auto_case",
        }
        all_rows.append(row)
        if supported:
            supported_cases.append(
                {
                    "case_id": case_id,
                    "cue_a": cue_a,
                    "cue_b": cue_b,
                    "query_ids": query_ids,
                    "min_queries": min_queries,
                    "selection_method": "pre_registered_auto",
                    "auto_case_support": len(query_ids),
                }
            )

    supported_cases.sort(key=lambda case: (-int(case["auto_case_support"]), str(case["case_id"])))
    if max_cases is not None:
        kept_ids = {str(case["case_id"]) for case in supported_cases[:max_cases]}
        supported_cases = supported_cases[:max_cases]
        for row in all_rows:
            if row["kept_after_support"] and row["case_id"] not in kept_ids:
                row["kept_after_support"] = False
                row["reason"] = "capped_by_max_auto_cases"

    all_rows.sort(key=lambda row: (-int(row["num_queries"]), str(row["case_id"])))
    return supported_cases, all_rows


def write_auto_case_candidates(
    output_dir: Path,
    candidate_cases: Sequence[Mapping[str, Any]],
    auto_case_generation_rows: Sequence[Mapping[str, Any]],
) -> None:
    pd = import_pandas()
    pd.DataFrame(auto_case_generation_rows).to_csv(
        output_dir / "auto_case_candidates.csv", index=False
    )
    with (output_dir / "auto_case_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(list(candidate_cases)), handle, indent=2, sort_keys=True)


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, SimpleNamespace):
        return {key: to_jsonable(value) for key, value in vars(obj).items()}
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def load_cases(cases_file: Path) -> List[Dict[str, Any]]:
    if not cases_file.exists():
        raise FileNotFoundError(f"Cue case file not found: {cases_file}")
    with cases_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise CaseValidationError("Cue case file must contain a JSON list of case objects")
    if not data:
        raise CaseValidationError("Cue case file contains no cases")

    cases: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, raw_case in enumerate(data):
        if not isinstance(raw_case, dict):
            raise CaseValidationError(f"Case at index {index} must be a JSON object")
        case = dict(raw_case)
        prefix = f"Case at index {index}"
        for field in ("case_id", "cue_a", "cue_b"):
            if field not in case:
                raise CaseValidationError(f"{prefix} is missing required field '{field}'")
            if not isinstance(case[field], str) or not case[field].strip():
                raise CaseValidationError(f"{prefix} field '{field}' must be a non-empty string")
            case[field] = case[field].strip()

        if case["case_id"] in seen_ids:
            raise CaseValidationError(f"Duplicate case_id '{case['case_id']}'")
        seen_ids.add(case["case_id"])

        if "query_include_all" in case:
            values = case["query_include_all"]
            if not isinstance(values, list) or not values:
                raise CaseValidationError(
                    f"{prefix} field 'query_include_all' must be a non-empty list"
                )
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise CaseValidationError(
                    f"{prefix} field 'query_include_all' must contain only non-empty strings"
                )
            case["query_include_all"] = [value.strip() for value in values]

        if "query_ids" in case:
            values = case["query_ids"]
            if not isinstance(values, list) or not values:
                raise CaseValidationError(f"{prefix} field 'query_ids' must be a non-empty list")
            if not all(isinstance(value, int) for value in values):
                raise CaseValidationError(f"{prefix} field 'query_ids' must contain integers")

        if "query_regex" in case:
            if not isinstance(case["query_regex"], str) or not case["query_regex"].strip():
                raise CaseValidationError(f"{prefix} field 'query_regex' must be a string")
            try:
                re.compile(case["query_regex"], flags=re.IGNORECASE)
            except re.error as exc:
                raise CaseValidationError(
                    f"{prefix} has invalid query_regex: {exc}"
                ) from exc

        for field in ("max_queries", "min_queries"):
            if field in case:
                if not isinstance(case[field], int) or case[field] <= 0:
                    raise CaseValidationError(f"{prefix} field '{field}' must be a positive integer")

        cases.append(case)
    return cases


def case_needles(case: Mapping[str, Any]) -> List[str]:
    if "query_include_all" in case:
        needles = [normalize_text(value) for value in case["query_include_all"]]
    else:
        cue_text = f"{case['cue_a']} {case['cue_b']}"
        needles = normalize_text(cue_text).split()
    return [needle for needle in needles if needle]


def default_repo_args() -> Dict[str, Any]:
    # Mirrors utils.options defaults so the script can dry-run or run with an
    # explicit checkpoint even when no training config YAML is available.
    return {
        "tau": 0.015,
        "select_ratio": 0.4,
        "margin": 0.1,
        "lambda1_weight": 0.5,
        "lambda2_weight": 3.5,
        "local_rank": 0,
        "output_dir": "run_logs",
        "name": "ITSELF",
        "log_period": 20,
        "eval_period": 1,
        "val_dataset": "test",
        "resume": False,
        "resume_ckpt_file": "",
        "finetune": "",
        "pretrain": "",
        "pretrain_choice": "ViT-B/16",
        "temperature": 0.02,
        "img_aug": True,
        "txt_aug": True,
        "loss_names": "tal+cid",
        "img_size": (384, 128),
        "stride_size": 16,
        "text_length": 77,
        "vocab_size": 49408,
        "optimizer": "Adam",
        "lr": 1e-5,
        "bias_lr_factor": 2.0,
        "lr_factor": 5.0,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "weight_decay_bias": 0.0,
        "alpha": 0.9,
        "beta": 0.999,
        "num_epoch": 60,
        "milestones": (45, 50),
        "gamma": 0.1,
        "warmup_factor": 0.1,
        "warmup_epochs": 5,
        "warmup_method": "linear",
        "lrscheduler": "cosine",
        "target_lr": 0,
        "power": 0.9,
        "dataset_name": "CUHK-PEDES",
        "sampler": "identity",
        "num_instance": 2,
        "root_dir": "data",
        "batch_size": 256,
        "test_batch_size": 512,
        "num_workers": 4,
        "training": False,
        "only_global": False,
        "return_all": False,
        "topk_type": "mean",
        "layer_index": -1,
        "average_attn_weights": True,
        "modify_k": False,
        "distributed": False,
    }


def load_repo_args(args: argparse.Namespace) -> SimpleNamespace:
    cfg = default_repo_args()
    if args.config is not None:
        from utils.iotools import load_train_configs

        loaded = load_train_configs(str(args.config))
        cfg.update(dict(loaded))

    cfg["dataset_name"] = args.dataset
    if args.root_dir is not None:
        cfg["root_dir"] = str(args.root_dir)
    cfg["training"] = False
    cfg["output_dir"] = str(args.output_dir)
    cfg["distributed"] = False
    if args.test_batch_size is not None:
        cfg["test_batch_size"] = args.test_batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.only_global:
        cfg["only_global"] = True

    img_size = cfg.get("img_size", (384, 128))
    if isinstance(img_size, list):
        img_size = tuple(img_size)
    if isinstance(img_size, str):
        numbers = [int(value) for value in re.findall(r"\d+", img_size)]
        if len(numbers) != 2:
            raise ValueError(f"Could not parse img_size from config value: {img_size}")
        img_size = tuple(numbers)
    cfg["img_size"] = tuple(img_size)
    cfg["milestones"] = tuple(cfg.get("milestones", (45, 50)))
    return SimpleNamespace(**cfg)


def build_split_metadata(repo_args: SimpleNamespace, split: str) -> SplitData:
    dataset_name = repo_args.dataset_name
    root_dir = Path(repo_args.root_dir)
    if dataset_name == "RSTPReid":
        dataset_dir = root_dir / "RSTPReid"
        anno_path = dataset_dir / "data_captions.json"
        image_key = "img_path"
    elif dataset_name == "CUHK-PEDES":
        dataset_dir = root_dir / "CUHK-PEDES"
        anno_path = dataset_dir / "reid_raw.json"
        image_key = "file_path"
    elif dataset_name == "ICFG-PEDES":
        dataset_dir = root_dir / "ICFG-PEDES"
        anno_path = dataset_dir / "ICFG-PEDES.json"
        image_key = "file_path"
    else:
        raise ValueError(f"Unsupported dataset for metadata loading: {dataset_name}")
    if not anno_path.exists():
        raise FileNotFoundError(f"Dataset annotation file not found: {anno_path}")

    with anno_path.open("r", encoding="utf-8") as handle:
        annos = json.load(handle)
    query_records: List[QueryRecord] = []
    gallery_pids: List[int] = []
    gallery_paths: List[str] = []
    query_id = 0
    for anno in annos:
        if anno.get("split") != split:
            continue
        pid = int(anno["id"])
        gallery_pids.append(pid)
        gallery_paths.append(str(dataset_dir / "imgs" / anno[image_key]))
        for caption in anno.get("captions", []):
            query_records.append(QueryRecord(query_id=query_id, query_text=str(caption), pid=pid))
            query_id += 1
    if not query_records or not gallery_pids:
        raise RuntimeError(f"No {split} records found in {anno_path}")
    return SplitData(
        query_records=query_records,
        gallery_pids=np.asarray(gallery_pids, dtype=np.int64),
        gallery_paths=gallery_paths,
    )


def build_split_data(
    repo_args: SimpleNamespace,
    split: str,
) -> Tuple[Any, DataLoader, DataLoader, SplitData]:
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms
    from datasets.cuhkpedes import CUHKPEDES
    from datasets.icfgpedes import ICFGPEDES
    from datasets.rstpreid import RSTPReid

    dataset_factories = {
        "CUHK-PEDES": CUHKPEDES,
        "ICFG-PEDES": ICFGPEDES,
        "RSTPReid": RSTPReid,
    }
    dataset_cls = dataset_factories[repo_args.dataset_name]
    dataset = dataset_cls(root=repo_args.root_dir)
    split_ds = getattr(dataset, split)

    transform = build_transforms(img_size=repo_args.img_size, is_train=False)
    img_set = ImageDataset(split_ds["image_pids"], split_ds["img_paths"], transform)
    txt_set = TextDataset(
        split_ds["caption_pids"],
        split_ds["captions"],
        text_length=repo_args.text_length,
    )

    img_loader = DataLoader(
        img_set,
        batch_size=repo_args.test_batch_size,
        shuffle=False,
        num_workers=repo_args.num_workers,
    )
    txt_loader = DataLoader(
        txt_set,
        batch_size=repo_args.test_batch_size,
        shuffle=False,
        num_workers=repo_args.num_workers,
    )

    query_records = [
        QueryRecord(query_id=index, query_text=text, pid=int(pid))
        for index, (pid, text) in enumerate(zip(txt_set.caption_pids, txt_set.captions))
    ]
    split_data = SplitData(
        query_records=query_records,
        gallery_pids=np.asarray(img_set.image_pids, dtype=np.int64),
        gallery_paths=[str(path) for path in img_set.img_paths],
    )
    return dataset, img_loader, txt_loader, split_data


def select_queries_for_cases(
    cases: Sequence[Mapping[str, Any]],
    query_records: Sequence[QueryRecord],
    gallery_pids: np.ndarray,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    query_by_id = {record.query_id: record for record in query_records}
    pids_with_gallery = set(int(pid) for pid in gallery_pids.tolist())

    for case in cases:
        case_id = str(case["case_id"])
        regex = (
            re.compile(str(case["query_regex"]), flags=re.IGNORECASE)
            if "query_regex" in case
            else None
        )

        if "query_ids" in case:
            candidates: List[QueryRecord] = []
            for query_id in case["query_ids"]:
                record = query_by_id.get(int(query_id))
                if record is None:
                    skipped.append(
                        {
                            "case_id": case_id,
                            "query_id": int(query_id),
                            "reason": "query_id_not_in_split",
                        }
                    )
                    continue
                candidates.append(record)
            selection_method = str(case.get("selection_method", "query_ids"))
        else:
            needles = case_needles(case)
            candidates = []
            for record in query_records:
                normalized = normalize_text(record.query_text)
                if not all(contains_normalized_phrase(normalized, needle) for needle in needles):
                    continue
                if regex is not None and regex.search(normalized) is None:
                    continue
                candidates.append(record)
            selection_method = "automatic"

        validated: List[QueryRecord] = []
        for record in candidates:
            if record.pid not in pids_with_gallery:
                skipped.append(
                    {
                        "case_id": case_id,
                        "query_id": record.query_id,
                        "pid": record.pid,
                        "reason": "query_has_no_positive_gallery_image",
                    }
                )
                continue
            validated.append(record)

        max_queries = case.get("max_queries")
        if args.max_queries_per_case is not None:
            max_queries = (
                min(max_queries, args.max_queries_per_case)
                if max_queries is not None
                else args.max_queries_per_case
            )
        if max_queries is not None:
            validated = validated[: int(max_queries)]

        min_queries = case.get("min_queries")
        if min_queries is not None and len(validated) < int(min_queries):
            logger.warning(
                "case_id=%s selected %d queries, below min_queries=%d",
                case_id,
                len(validated),
                int(min_queries),
            )

        if not validated:
            skipped.append(
                {
                    "case_id": case_id,
                    "reason": "case_selected_no_eligible_queries",
                    "selection_method": selection_method,
                }
            )

        for record in validated:
            selected.append(
                {
                    "case_id": case_id,
                    "query_id": record.query_id,
                    "query_text": record.query_text,
                    "pid": record.pid,
                    "cue_a": case["cue_a"],
                    "cue_b": case["cue_b"],
                    "selection_method": selection_method,
                }
            )
    return selected, skipped


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return device


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, logger: logging.Logger) -> None:
    from utils.checkpoint import load_state_dict

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"Checkpoint does not contain a model state dict: {checkpoint_path}")
    load_state_dict(model, state_dict)


def model_has_grab(model: torch.nn.Module, repo_args: SimpleNamespace) -> bool:
    return (
        not bool(getattr(repo_args, "only_global", False))
        and hasattr(model, "visul_emb_layer")
        and hasattr(model, "texual_emb_layer")
    )


def resolve_mode(mode: str, has_grab: bool, lambda_global: float, kind: str) -> str:
    if mode == "auto":
        return "fusion" if has_grab else "global"
    if mode == "score":
        raise ValueError("Internal error: resolve 'score' before calling resolve_mode")
    if mode in {"grab", "fusion"} and not has_grab:
        raise ValueError(f"{kind} mode '{mode}' requires GRAB features, but the model has no GRAB branch")
    if mode == "fusion" and lambda_global in (0.0, 1.0):
        return "global" if lambda_global == 1.0 else "grab"
    return mode


def extract_text_embeddings(
    model: torch.nn.Module,
    txt_loader: DataLoader,
    device: torch.device,
    use_grab: bool,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    logger.info("Encoding %d text queries", len(txt_loader.dataset))
    global_features: List[torch.Tensor] = []
    grab_features: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _, captions in txt_loader:
            captions = captions.to(device)
            feats = model.encode_text(captions)
            global_features.append(F.normalize(feats.float(), p=2, dim=1).cpu())
            if use_grab:
                grab = model.encode_text_grab(captions)
                grab_features.append(F.normalize(grab.float(), p=2, dim=1).cpu())
    global_tensor = torch.cat(global_features, dim=0)
    grab_tensor = torch.cat(grab_features, dim=0) if use_grab else None
    return global_tensor, grab_tensor


def extract_image_embeddings(
    model: torch.nn.Module,
    img_loader: DataLoader,
    device: torch.device,
    use_grab: bool,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    logger.info("Encoding %d gallery images", len(img_loader.dataset))
    global_features: List[torch.Tensor] = []
    grab_features: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _, images in img_loader:
            images = images.to(device)
            feats = model.encode_image(images)
            global_features.append(F.normalize(feats.float(), p=2, dim=1).cpu())
            if use_grab:
                grab = model.encode_image_grab(images)
                grab_features.append(F.normalize(grab.float(), p=2, dim=1).cpu())
    global_tensor = torch.cat(global_features, dim=0)
    grab_tensor = torch.cat(grab_features, dim=0) if use_grab else None
    return global_tensor, grab_tensor


def encode_cue_features(
    model: torch.nn.Module,
    cues: Sequence[str],
    text_length: int,
    device: torch.device,
    use_grab: bool,
    logger: logging.Logger,
) -> CueFeatures:
    from datasets.bases import tokenize
    from utils.simple_tokenizer import SimpleTokenizer

    logger.info("Encoding %d unique cues", len(cues))
    tokenizer = SimpleTokenizer()
    tokens = torch.stack(
        [tokenize(cue, tokenizer=tokenizer, text_length=text_length) for cue in cues],
        dim=0,
    ).to(device)
    global_features: Dict[str, torch.Tensor] = {}
    grab_features: Dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        global_tensor = F.normalize(model.encode_text(tokens).float(), p=2, dim=1).cpu()
        grab_tensor = (
            F.normalize(model.encode_text_grab(tokens).float(), p=2, dim=1).cpu()
            if use_grab
            else None
        )
    for index, cue in enumerate(cues):
        global_features[cue] = global_tensor[index]
        if grab_tensor is not None:
            grab_features[cue] = grab_tensor[index]
    return CueFeatures(global_features=global_features, grab_features=grab_features)


def build_embedding_cache(
    model: torch.nn.Module,
    img_loader: DataLoader,
    txt_loader: DataLoader,
    device: torch.device,
    use_grab: bool,
    logger: logging.Logger,
) -> EmbeddingCache:
    query_global, query_grab = extract_text_embeddings(model, txt_loader, device, use_grab, logger)
    gallery_global, gallery_grab = extract_image_embeddings(model, img_loader, device, use_grab, logger)
    return EmbeddingCache(
        query_global=query_global,
        gallery_global=gallery_global,
        query_grab=query_grab,
        gallery_grab=gallery_grab,
    )


def query_gallery_scores(
    cache: EmbeddingCache,
    query_id: int,
    gallery_indices: Sequence[int],
    mode: str,
    lambda_global: float,
) -> np.ndarray:
    idx = torch.as_tensor(gallery_indices, dtype=torch.long)
    global_scores = (cache.query_global[query_id] @ cache.gallery_global[idx].T).cpu()
    if mode == "global":
        return global_scores.numpy()
    if cache.query_grab is None or cache.gallery_grab is None:
        raise RuntimeError("GRAB scores requested without cached GRAB features")
    grab_scores = (cache.query_grab[query_id] @ cache.gallery_grab[idx].T).cpu()
    if mode == "grab":
        return grab_scores.numpy()
    if mode == "fusion":
        return (lambda_global * global_scores + (1.0 - lambda_global) * grab_scores).numpy()
    raise ValueError(f"Unsupported score mode: {mode}")


def cue_gallery_affinity(
    cache: EmbeddingCache,
    cue_features: CueFeatures,
    cue: str,
    mode: str,
    lambda_global: float,
) -> np.ndarray:
    global_scores = (cache.gallery_global @ cue_features.global_features[cue]).cpu()
    if mode == "global":
        return global_scores.numpy()
    if cache.gallery_grab is None or cue not in cue_features.grab_features:
        raise RuntimeError("GRAB cue affinity requested without cached GRAB features")
    grab_scores = (cache.gallery_grab @ cue_features.grab_features[cue]).cpu()
    if mode == "grab":
        return grab_scores.numpy()
    if mode == "fusion":
        return (lambda_global * global_scores + (1.0 - lambda_global) * grab_scores).numpy()
    raise ValueError(f"Unsupported cue mode: {mode}")


def full_test_similarity_chunk(
    cache: EmbeddingCache,
    start: int,
    end: int,
    mode: str,
    lambda_global: float,
) -> torch.Tensor:
    global_scores = cache.query_global[start:end] @ cache.gallery_global.T
    if mode == "global":
        return global_scores
    if cache.query_grab is None or cache.gallery_grab is None:
        raise RuntimeError("GRAB full-test scores requested without cached GRAB features")
    grab_scores = cache.query_grab[start:end] @ cache.gallery_grab.T
    if mode == "grab":
        return grab_scores
    if mode == "fusion":
        return lambda_global * global_scores + (1.0 - lambda_global) * grab_scores
    raise ValueError(f"Unsupported full-test score mode: {mode}")


def compute_full_test_metrics(
    cache: EmbeddingCache,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    mode: str,
    lambda_global: float,
    chunk_size: int = 128,
) -> Dict[str, float]:
    g_pids = torch.as_tensor(gallery_pids, dtype=torch.long)
    q_pids = torch.as_tensor(query_pids, dtype=torch.long)
    num_queries = int(len(query_pids))
    num_gallery = int(len(gallery_pids))
    if num_queries == 0 or num_gallery == 0:
        raise ValueError("Full-test metrics require at least one query and one gallery image")

    ranks = torch.arange(1, num_gallery + 1, dtype=torch.float32).view(1, -1)
    total_valid = 0
    r1_total = 0.0
    r5_total = 0.0
    r10_total = 0.0
    ap_total = 0.0
    min_rank_total = 0.0
    skipped_no_positive = 0

    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        scores = full_test_similarity_chunk(cache, start, end, mode, lambda_global)
        indices = torch.argsort(scores, dim=1, descending=True)
        pred_pids = g_pids[indices]
        matches = pred_pids.eq(q_pids[start:end].view(-1, 1))
        num_rel = matches.sum(dim=1)
        valid = num_rel > 0
        skipped_no_positive += int((~valid).sum().item())
        if not bool(valid.any()):
            continue

        matches = matches[valid]
        num_rel = num_rel[valid].float()
        total_valid += int(matches.shape[0])

        r1_total += float(matches[:, :1].any(dim=1).float().sum().item())
        r5_total += float(matches[:, : min(5, num_gallery)].any(dim=1).float().sum().item())
        r10_total += float(matches[:, : min(10, num_gallery)].any(dim=1).float().sum().item())

        cumulative = matches.cumsum(dim=1).float()
        precision = cumulative / ranks[:, : matches.shape[1]]
        ap = (precision * matches.float()).sum(dim=1) / num_rel
        ap_total += float(ap.sum().item())
        min_ranks = matches.float().argmax(dim=1).float() + 1.0
        min_rank_total += float(min_ranks.sum().item())

    if total_valid == 0:
        raise ValueError("No full-test queries have positive gallery images")

    return {
        "num_queries": int(num_queries),
        "num_gallery": int(num_gallery),
        "num_valid_queries": int(total_valid),
        "num_queries_without_positive": int(skipped_no_positive),
        "R1": 100.0 * r1_total / total_valid,
        "R5": 100.0 * r5_total / total_valid,
        "R10": 100.0 * r10_total / total_valid,
        "mAP": 100.0 * ap_total / total_valid,
        "mean_min_positive_rank": min_rank_total / total_valid,
    }


def full_test_sanity_modes(score_mode: str, use_grab: bool) -> List[str]:
    modes = [score_mode]
    if use_grab:
        for mode in ("global", "grab", "fusion"):
            if mode not in modes:
                modes.append(mode)
    return modes


def write_full_test_metrics(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Any:
    pd = import_pandas()
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "whole_test_metrics.csv", index=False)
    with (output_dir / "whole_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(list(rows)), handle, indent=2, sort_keys=True)
    return df


def stable_topk(indices: np.ndarray, scores: np.ndarray, k: int, largest: bool) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    values = scores[indices]
    primary = -values if largest else values
    order = np.lexsort((indices, primary))
    return indices[order[:k]]


def choose_neutral_indices(
    remaining: np.ndarray,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    k: int,
    rng: np.random.Generator,
    strategy: str,
    pool_factor: int,
) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    if len(remaining) < k:
        raise ValueError("not_enough_remaining_distractors_for_neutral_fill")

    if strategy == "random":
        pool = remaining
    else:
        max_affinity = np.maximum(psi_a[remaining], psi_b[remaining])
        order = np.lexsort((remaining, max_affinity))
        pool_size = min(len(remaining), max(k, k * pool_factor))
        pool = remaining[order[:pool_size]]

    if len(pool) < k:
        pool = remaining
    selected = rng.choice(pool, size=k, replace=False)
    return np.asarray(selected, dtype=np.int64)


def make_gallery(
    positive_indices: np.ndarray,
    candidate_indices: np.ndarray,
    dense_scores: np.ndarray,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    num_dense: int,
    num_neutral: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    dense = stable_topk(candidate_indices, dense_scores, num_dense, largest=True)
    dense_set = set(int(index) for index in dense.tolist())
    remaining = np.asarray(
        [int(index) for index in candidate_indices.tolist() if int(index) not in dense_set],
        dtype=np.int64,
    )
    neutral = choose_neutral_indices(
        remaining=remaining,
        psi_a=psi_a,
        psi_b=psi_b,
        k=num_neutral,
        rng=rng,
        strategy=args.neutral_strategy,
        pool_factor=args.neutral_pool_factor,
    )
    gallery = np.concatenate([positive_indices, dense, neutral]).astype(np.int64)
    if len(np.unique(gallery)) != len(gallery):
        raise ValueError("gallery_contains_duplicate_image_ids")
    return gallery


def construct_counterfactual_galleries(
    pid: int,
    gallery_pids: np.ndarray,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    args: argparse.Namespace,
    case_id: str,
    query_id: int,
    trial_id: int,
) -> Tuple[Dict[str, np.ndarray], Optional[str]]:
    positive_indices = np.flatnonzero(gallery_pids == int(pid)).astype(np.int64)
    if len(positive_indices) == 0:
        return {}, "no_positive_gallery_images"
    if len(positive_indices) > args.gallery_size:
        return {}, "num_positives_exceeds_gallery_size"

    candidate_indices = np.flatnonzero(gallery_pids != int(pid)).astype(np.int64)
    num_distractors = args.gallery_size - len(positive_indices)
    if len(candidate_indices) < num_distractors:
        return {}, "not_enough_valid_distractors"

    num_dense = int(round(args.dense_ratio * num_distractors))
    num_dense = min(max(num_dense, 0), num_distractors)
    num_neutral = num_distractors - num_dense

    score_a_dense = psi_a - args.lambda_contrast * psi_b
    score_b_dense = psi_b - args.lambda_contrast * psi_a

    try:
        rng_a = np.random.default_rng(stable_seed(args.seed, case_id, query_id, trial_id, "a_dense"))
        rng_b = np.random.default_rng(stable_seed(args.seed, case_id, query_id, trial_id, "b_dense"))
        gallery_a = make_gallery(
            positive_indices=positive_indices,
            candidate_indices=candidate_indices,
            dense_scores=score_a_dense,
            psi_a=psi_a,
            psi_b=psi_b,
            num_dense=num_dense,
            num_neutral=num_neutral,
            rng=rng_a,
            args=args,
        )
        gallery_b = make_gallery(
            positive_indices=positive_indices,
            candidate_indices=candidate_indices,
            dense_scores=score_b_dense,
            psi_a=psi_a,
            psi_b=psi_b,
            num_dense=num_dense,
            num_neutral=num_neutral,
            rng=rng_b,
            args=args,
        )
    except ValueError as exc:
        return {}, str(exc)

    for gallery in (gallery_a, gallery_b):
        if len(gallery) != args.gallery_size:
            return {}, "constructed_gallery_size_mismatch"
        distractor_pids = gallery_pids[gallery][gallery_pids[gallery] != int(pid)]
        if np.any(distractor_pids == int(pid)):
            return {}, "same_identity_distractor_detected"

    return {"a_dense": gallery_a, "b_dense": gallery_b}, None


def compute_retrieval_metrics(scores: np.ndarray, is_positive: np.ndarray) -> Dict[str, float]:
    if int(is_positive.sum()) == 0:
        raise ValueError("metrics_requested_for_gallery_without_positives")
    order = np.lexsort((np.arange(len(scores)), -scores))
    matches = is_positive[order].astype(np.int64)
    positive_ranks = np.flatnonzero(matches) + 1
    cumulative = np.cumsum(matches)
    precisions = cumulative[positive_ranks - 1] / positive_ranks
    ap = float(precisions.mean() * 100.0)
    return {
        "R1": float(matches[:1].any() * 100.0),
        "R5": float(matches[:5].any() * 100.0),
        "R10": float(matches[:10].any() * 100.0),
        "AP": ap,
        "min_positive_rank": float(positive_ranks.min()),
    }


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def compute_density(
    psi: np.ndarray,
    gallery_indices: np.ndarray,
    threshold: float,
    tau_density: float,
) -> Tuple[float, float]:
    values = psi[gallery_indices]
    hard = float(np.mean(values > threshold))
    soft = float(np.mean(sigmoid_np((values - threshold) / tau_density)))
    return hard, soft


def estimate_swap_strength_for_galleries(
    gallery_a: np.ndarray,
    gallery_b: np.ndarray,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    tau_density: float,
) -> float:
    _, soft_a_in_a = compute_density(psi_a, gallery_a, threshold_a, tau_density)
    _, soft_a_in_b = compute_density(psi_a, gallery_b, threshold_a, tau_density)
    _, soft_b_in_a = compute_density(psi_b, gallery_a, threshold_b, tau_density)
    _, soft_b_in_b = compute_density(psi_b, gallery_b, threshold_b, tau_density)
    return float((soft_a_in_a - soft_a_in_b) + (soft_b_in_b - soft_b_in_a))


def filter_visually_constructible_cases(
    candidate_cases: Sequence[Mapping[str, Any]],
    query_records: Sequence[QueryRecord],
    gallery_pids: np.ndarray,
    cue_affinities: Mapping[str, np.ndarray],
    cue_thresholds: Mapping[str, float],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Keep only auto-generated cases for which paired cue-swap galleries can be
    constructed for at least args.min_constructible_query_trials query-trials.

    This is a manipulation/constructibility filter only. It must not compute
    or use retrieval scores, R1, AP, ranks, or any retrieval outcome.
    """
    query_by_id = {record.query_id: record for record in query_records}
    kept_cases: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for raw_case in candidate_cases:
        case = dict(raw_case)
        case_id = str(case["case_id"])
        cue_a = str(case["cue_a"])
        cue_b = str(case["cue_b"])
        checked_trials = 0
        constructible_trials = 0
        swap_strengths: List[float] = []
        skip_reasons: Counter[str] = Counter()

        if cue_a not in cue_affinities or cue_b not in cue_affinities:
            rows.append(
                {
                    "case_id": case_id,
                    "cue_a": cue_a,
                    "cue_b": cue_b,
                    "auto_case_support": int(case.get("auto_case_support", 0)),
                    "checked_query_trials": 0,
                    "constructible_query_trials": 0,
                    "mean_constructibility_swap_strength": float("nan"),
                    "kept": False,
                    "reason": "missing_cue_affinity",
                    "construction_skip_reasons": "{}",
                }
            )
            continue

        query_ids = [int(query_id) for query_id in case.get("query_ids", [])]
        query_ids = query_ids[: args.auto_case_max_queries_for_constructibility]
        psi_a = cue_affinities[cue_a]
        psi_b = cue_affinities[cue_b]
        threshold_a = cue_thresholds[cue_a]
        threshold_b = cue_thresholds[cue_b]

        for query_id in query_ids:
            record = query_by_id.get(query_id)
            if record is None:
                skip_reasons["query_id_not_in_split"] += args.num_random_trials
                checked_trials += args.num_random_trials
                continue
            for trial_id in range(args.num_random_trials):
                checked_trials += 1
                galleries, skip_reason = construct_counterfactual_galleries(
                    pid=record.pid,
                    gallery_pids=gallery_pids,
                    psi_a=psi_a,
                    psi_b=psi_b,
                    args=args,
                    case_id=case_id,
                    query_id=query_id,
                    trial_id=trial_id,
                )
                if skip_reason is not None:
                    skip_reasons[skip_reason] += 1
                    continue
                constructible_trials += 1
                swap_strengths.append(
                    estimate_swap_strength_for_galleries(
                        gallery_a=galleries["a_dense"],
                        gallery_b=galleries["b_dense"],
                        psi_a=psi_a,
                        psi_b=psi_b,
                        threshold_a=threshold_a,
                        threshold_b=threshold_b,
                        tau_density=args.tau_density,
                    )
                )

        mean_swap_strength = (
            float(np.mean(swap_strengths)) if swap_strengths else float("nan")
        )
        if constructible_trials == 0:
            kept = False
            reason = "no_constructible_trials"
        elif constructible_trials < args.min_constructible_query_trials:
            kept = False
            reason = "below_min_constructible_query_trials"
        elif (
            args.min_case_swap_strength is not None
            and mean_swap_strength < args.min_case_swap_strength
        ):
            kept = False
            reason = "below_min_case_swap_strength"
        else:
            kept = True
            reason = ""

        if kept:
            kept_case = dict(case)
            kept_case["constructible_query_trials"] = constructible_trials
            kept_case["mean_constructibility_swap_strength"] = mean_swap_strength
            kept_case["case_source"] = "pre_registered_auto"
            kept_cases.append(kept_case)

        rows.append(
            {
                "case_id": case_id,
                "cue_a": cue_a,
                "cue_b": cue_b,
                "auto_case_support": int(case.get("auto_case_support", 0)),
                "checked_query_trials": checked_trials,
                "constructible_query_trials": constructible_trials,
                "mean_constructibility_swap_strength": mean_swap_strength,
                "kept": kept,
                "reason": reason,
                "construction_skip_reasons": json.dumps(dict(sorted(skip_reasons.items()))),
            }
        )

    kept_cases.sort(
        key=lambda case: (
            -int(case.get("auto_case_support", 0)),
            str(case.get("case_id", "")),
        )
    )
    reason_counts = Counter(row["reason"] or "kept" for row in rows)
    logger.info(
        "Auto constructibility kept %d/%d cases; top reasons: %s",
        len(kept_cases),
        len(candidate_cases),
        dict(reason_counts.most_common(5)),
    )
    return kept_cases, rows


def write_auto_case_constructibility(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    pd = import_pandas()
    columns = [
        "case_id",
        "cue_a",
        "cue_b",
        "auto_case_support",
        "checked_query_trials",
        "constructible_query_trials",
        "mean_constructibility_swap_strength",
        "kept",
        "reason",
        "construction_skip_reasons",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        output_dir / "auto_case_constructibility.csv", index=False
    )


def write_auto_cases_json(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(list(cases)), handle, indent=2, sort_keys=True)


def compute_crowding(scores: np.ndarray, is_positive: np.ndarray, tau_crowding: float) -> float:
    non_positive_scores = scores[~is_positive]
    if len(non_positive_scores) == 0:
        return float("nan")
    scaled = non_positive_scores / tau_crowding
    max_value = float(np.max(scaled))
    return float(max_value + np.log(np.exp(scaled - max_value).sum()))


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")


def write_config_used(
    output_dir: Path,
    args: argparse.Namespace,
    cases: Sequence[Mapping[str, Any]],
    repo_args: Optional[SimpleNamespace] = None,
    score_mode: Optional[str] = None,
    cue_mode: Optional[str] = None,
    case_source: Optional[str] = None,
    cue_vocab_source_value: Optional[str] = None,
) -> None:
    payload = {
        "script_args": to_jsonable(vars(args)),
        "cases": to_jsonable(list(cases)),
        "repo_args": to_jsonable(repo_args) if repo_args is not None else None,
        "resolved_score_mode": score_mode,
        "resolved_cue_mode": cue_mode,
        "case_source": case_source,
        "cue_vocab_source": cue_vocab_source_value,
        "auto_case_parameters": {
            "min_queries_per_auto_case": args.min_queries_per_auto_case,
            "max_auto_cases": args.max_auto_cases,
            "min_constructible_query_trials": args.min_constructible_query_trials,
            "min_case_swap_strength": args.min_case_swap_strength,
            "auto_case_max_queries_for_constructibility": args.auto_case_max_queries_for_constructibility,
        },
    }
    with (output_dir / "config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_outputs(
    output_dir: Path,
    selected_queries: List[Dict[str, Any]],
    per_query_rows: List[Dict[str, Any]],
    paired_rows: List[Dict[str, Any]],
    gallery_rows: List[Dict[str, Any]],
    skipped_rows: List[Dict[str, Any]],
) -> Tuple[Any, Any, Any]:
    pd = import_pandas()

    selected_columns = [
        "case_id",
        "query_id",
        "query_text",
        "pid",
        "cue_a",
        "cue_b",
        "selection_method",
    ]
    per_query_columns = [
        "case_id",
        "query_id",
        "query_text",
        "pid",
        "cue_a",
        "cue_b",
        "gallery_type",
        "gallery_size",
        "num_positives",
        "R1",
        "R5",
        "R10",
        "AP",
        "min_positive_rank",
        "D_soft_a",
        "D_soft_b",
        "D_hard_a",
        "D_hard_b",
        "crowding",
        "swap_strength",
        "rank_volatility",
        "ap_delta",
        "r1_flip",
        "seed",
        "trial_id",
        "score_mode",
        "cue_mode",
    ]

    selected_df = pd.DataFrame(selected_queries, columns=selected_columns)
    per_query_df = pd.DataFrame(per_query_rows, columns=per_query_columns)
    selected_df.to_csv(output_dir / "selected_queries.csv", index=False)
    per_query_df.to_csv(output_dir / "per_query_results.csv", index=False)

    if per_query_df.empty:
        summary_by_case = pd.DataFrame(
            columns=[
                "case_id",
                "gallery_type",
                "num_queries",
                "num_query_trials",
                "R1",
                "R5",
                "R10",
                "mAP",
                "mean_min_positive_rank",
                "mean_crowding",
                "mean_D_soft_a",
                "mean_D_soft_b",
                "mean_swap_strength",
            ]
        )
    else:
        rows = []
        for (case_id, gallery_type), group in per_query_df.groupby(["case_id", "gallery_type"]):
            rows.append(
                {
                    "case_id": case_id,
                    "gallery_type": gallery_type,
                    "num_queries": int(group["query_id"].nunique()),
                    "num_query_trials": int(len(group)),
                    "R1": float(group["R1"].mean()),
                    "R5": float(group["R5"].mean()),
                    "R10": float(group["R10"].mean()),
                    "mAP": float(group["AP"].mean()),
                    "mean_min_positive_rank": float(group["min_positive_rank"].mean()),
                    "mean_crowding": float(group["crowding"].mean()),
                    "mean_D_soft_a": float(group["D_soft_a"].mean()),
                    "mean_D_soft_b": float(group["D_soft_b"].mean()),
                    "mean_swap_strength": float(group["swap_strength"].mean()),
                }
            )
        summary_by_case = pd.DataFrame(rows)

    summary_by_case.to_csv(output_dir / "summary_by_case.csv", index=False)

    paired_df = pd.DataFrame(paired_rows)
    if per_query_df.empty:
        summary_overall = pd.DataFrame(
            [
                {
                    "num_cases": 0,
                    "num_queries_total": 0,
                    "num_query_trials": 0,
                    "R1_a_dense": float("nan"),
                    "R1_b_dense": float("nan"),
                    "R5_a_dense": float("nan"),
                    "R5_b_dense": float("nan"),
                    "R10_a_dense": float("nan"),
                    "R10_b_dense": float("nan"),
                    "mAP_a_dense": float("nan"),
                    "mAP_b_dense": float("nan"),
                    "mean_rank_volatility": float("nan"),
                    "r1_flip_rate": float("nan"),
                    "mean_swap_strength": float("nan"),
                }
            ]
        )
    else:
        a_rows = per_query_df[per_query_df["gallery_type"] == "a_dense"]
        b_rows = per_query_df[per_query_df["gallery_type"] == "b_dense"]
        summary_overall = pd.DataFrame(
            [
                {
                    "num_cases": int(per_query_df["case_id"].nunique()),
                    "num_queries_total": int(
                        per_query_df[["case_id", "query_id"]].drop_duplicates().shape[0]
                    ),
                    "num_query_trials": int(len(paired_df)),
                    "R1_a_dense": float(a_rows["R1"].mean()),
                    "R1_b_dense": float(b_rows["R1"].mean()),
                    "R5_a_dense": float(a_rows["R5"].mean()),
                    "R5_b_dense": float(b_rows["R5"].mean()),
                    "R10_a_dense": float(a_rows["R10"].mean()),
                    "R10_b_dense": float(b_rows["R10"].mean()),
                    "mAP_a_dense": float(a_rows["AP"].mean()),
                    "mAP_b_dense": float(b_rows["AP"].mean()),
                    "mean_rank_volatility": float(paired_df["rank_volatility"].mean()),
                    "r1_flip_rate": float(paired_df["r1_flip"].mean()),
                    "mean_swap_strength": float(paired_df["swap_strength"].mean()),
                }
            ]
        )

    summary_overall.to_csv(output_dir / "summary_overall.csv", index=False)
    write_jsonl(output_dir / "galleries.jsonl", gallery_rows)
    write_jsonl(output_dir / "skipped_queries.jsonl", skipped_rows)
    return per_query_df, summary_by_case, summary_overall


def unique_cues_from_cases(cases: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted({str(case["cue_a"]) for case in cases} | {str(case["cue_b"]) for case in cases})


def build_case_cue_affinities(
    cases: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    cache: EmbeddingCache,
    repo_args: SimpleNamespace,
    device: torch.device,
    use_grab: bool,
    cue_mode: str,
    lambda_global: float,
    cue_threshold_quantile: float,
    logger: logging.Logger,
) -> Tuple[List[str], CueFeatures, Dict[str, np.ndarray], Dict[str, float]]:
    unique_cues = unique_cues_from_cases(cases)
    cue_features = encode_cue_features(
        model=model,
        cues=unique_cues,
        text_length=repo_args.text_length,
        device=device,
        use_grab=use_grab,
        logger=logger,
    )
    cue_affinities = {
        cue: cue_gallery_affinity(cache, cue_features, cue, cue_mode, lambda_global)
        for cue in unique_cues
    }
    cue_thresholds = {
        cue: float(np.quantile(scores, cue_threshold_quantile))
        for cue, scores in cue_affinities.items()
    }
    return unique_cues, cue_features, cue_affinities, cue_thresholds


def run_auto_text_case_generation(
    args: argparse.Namespace,
    split_data: SplitData,
    logger: logging.Logger,
) -> Tuple[List[CueSpec], Dict[int, List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cue_specs = load_cue_specs(args.cue_vocab_file)
    logger.info("Loaded %d cue specs from %s", len(cue_specs), cue_vocab_source(args))
    detected_by_query = detect_cues_for_queries(split_data.query_records, cue_specs)
    write_auto_query_cues(args.output_dir, split_data.query_records, detected_by_query)
    queries_with_two_cues = sum(1 for cues in detected_by_query.values() if len(cues) >= 2)
    logger.info("Detected >=2 cues in %d queries", queries_with_two_cues)
    candidate_cases, generation_rows = generate_candidate_cases_from_cooccurrence(
        query_records=split_data.query_records,
        detected_by_query=detected_by_query,
        gallery_pids=split_data.gallery_pids,
        min_queries=args.min_queries_per_auto_case,
        max_cases=args.max_auto_cases,
    )
    write_auto_case_candidates(args.output_dir, candidate_cases, generation_rows)
    supported_before_cap = sum(1 for row in generation_rows if row.get("meets_min_queries"))
    logger.info("Candidate cue pairs before support filter: %d", len(generation_rows))
    logger.info("Candidate cases after support/cap filter: %d", len(candidate_cases))
    logger.info("Candidate cases meeting support before cap: %d", supported_before_cap)
    return cue_specs, detected_by_query, candidate_cases, generation_rows


def print_auto_case_generation_block(
    cue_specs_count: int,
    detected_by_query: Mapping[int, Sequence[str]],
    candidate_rows: Sequence[Mapping[str, Any]],
    kept_cases_count: Optional[int],
    auto_cases_file: Optional[Path],
) -> None:
    queries_with_two_cues = sum(1 for cues in detected_by_query.values() if len(cues) >= 2)
    supported_after_filter = sum(1 for row in candidate_rows if row.get("kept_after_support"))
    print("\nAuto case generation:")
    print(f"  cue specs: {cue_specs_count}")
    print(f"  queries with >=2 cues: {queries_with_two_cues}")
    print(f"  candidate pairs after support filter: {supported_after_filter}")
    if kept_cases_count is not None:
        print(f"  constructible cases kept: {kept_cases_count}")
    if auto_cases_file is not None:
        print(f"  auto cases file: {auto_cases_file}")


def run_dry_run(args: argparse.Namespace, logger: logging.Logger) -> None:
    if not is_auto_case_mode(args):
        cases = load_cases(args.cases_file)
        write_config_used(
            args.output_dir,
            args,
            cases,
            repo_args=None,
            case_source="manual",
            cue_vocab_source_value=None,
        )
        logger.info("Manual dry run complete: validated %d cue cases", len(cases))
        print("Dry run complete")
        print(f"Mode: manual")
        print(f"Validated cases: {len(cases)}")
        print(f"Output directory: {args.output_dir}")
        return

    logger.info("Auto dry run: loading dataset split and running cue co-occurrence only")
    repo_args = load_repo_args(args)
    split_data = build_split_metadata(repo_args, args.split)
    cue_specs, detected_by_query, candidate_cases, generation_rows = run_auto_text_case_generation(
        args=args,
        split_data=split_data,
        logger=logger,
    )
    write_config_used(
        args.output_dir,
        args,
        candidate_cases,
        repo_args=repo_args,
        case_source="pre_registered_auto",
        cue_vocab_source_value=cue_vocab_source(args),
    )
    print_auto_case_generation_block(
        cue_specs_count=len(cue_specs),
        detected_by_query=detected_by_query,
        candidate_rows=generation_rows,
        kept_cases_count=None,
        auto_cases_file=None,
    )
    print("Dry run complete")
    print(f"Mode: pre_registered_auto")
    print(f"Candidate cases: {len(candidate_cases)}")
    print(f"Output directory: {args.output_dir}")


def run_evaluation(args: argparse.Namespace, logger: logging.Logger) -> None:
    from model import build_model

    repo_args = load_repo_args(args)

    dataset, img_loader, txt_loader, split_data = build_split_data(repo_args, args.split)

    device = resolve_device(args.device)
    num_classes = len(dataset.train_id_container)
    model = build_model(repo_args, num_classes)
    load_checkpoint(model, args.checkpoint, logger)
    if device.type == "cpu":
        model = model.float()
    model.to(device)
    model.eval()

    use_grab = model_has_grab(model, repo_args)
    score_mode = resolve_mode(args.score_mode, use_grab, args.lambda_global, "score")
    cue_mode_arg = score_mode if args.cue_mode == "score" else args.cue_mode
    cue_mode = resolve_mode(cue_mode_arg, use_grab, args.lambda_global, "cue")
    logger.info(
        "Resolved score_mode=%s cue_mode=%s use_grab=%s lambda_global=%.4f",
        score_mode,
        cue_mode,
        use_grab,
        args.lambda_global,
    )

    cache = build_embedding_cache(model, img_loader, txt_loader, device, use_grab, logger)
    query_pids = np.asarray([record.pid for record in split_data.query_records], dtype=np.int64)
    full_metric_rows: List[Dict[str, Any]] = []
    full_metric_chunk = max(1, min(256, int(getattr(repo_args, "test_batch_size", 128))))
    for mode in full_test_sanity_modes(score_mode, use_grab):
        metrics = compute_full_test_metrics(
            cache=cache,
            query_pids=query_pids,
            gallery_pids=split_data.gallery_pids,
            mode=mode,
            lambda_global=args.lambda_global,
            chunk_size=full_metric_chunk,
        )
        full_metric_rows.append(
            {
                "score_mode": mode,
                "selected_for_protocol": mode == score_mode,
                "lambda_global": args.lambda_global,
                "formula": (
                    "lambda_global*s_global + (1-lambda_global)*s_grab"
                    if mode == "fusion"
                    else mode
                ),
                **metrics,
            }
        )
    full_metric_df = write_full_test_metrics(args.output_dir, full_metric_rows)
    selected_full_metric = next(row for row in full_metric_rows if row["selected_for_protocol"])
    logger.info(
        "Whole-test sanity R1 for %s: %.2f",
        selected_full_metric["score_mode"],
        selected_full_metric["R1"],
    )
    print("\nWhole-test sanity metrics:")
    print(full_metric_df.to_string(index=False))

    if is_auto_case_mode(args):
        logger.info("Using pre-registered automatic cue cases")
        cue_specs, detected_by_query, candidate_cases, generation_rows = run_auto_text_case_generation(
            args=args,
            split_data=split_data,
            logger=logger,
        )
        auto_cases_file = auto_cases_output_path(args)
        if not candidate_cases:
            write_auto_case_constructibility(args.output_dir, [])
            write_auto_cases_json(auto_cases_file, [])
            write_config_used(
                args.output_dir,
                args,
                [],
                repo_args=repo_args,
                score_mode=score_mode,
                cue_mode=cue_mode,
                case_source="pre_registered_auto",
                cue_vocab_source_value=cue_vocab_source(args),
            )
            print_auto_case_generation_block(
                cue_specs_count=len(cue_specs),
                detected_by_query=detected_by_query,
                candidate_rows=generation_rows,
                kept_cases_count=0,
                auto_cases_file=auto_cases_file,
            )
            raise RuntimeError(
                "No auto-generated cue cases met the query-support threshold. "
                "See auto_case_candidates.csv."
            )

        _, cue_features, cue_affinities, cue_thresholds = build_case_cue_affinities(
            cases=candidate_cases,
            model=model,
            cache=cache,
            repo_args=repo_args,
            device=device,
            use_grab=use_grab,
            cue_mode=cue_mode,
            lambda_global=args.lambda_global,
            cue_threshold_quantile=args.cue_threshold_quantile,
            logger=logger,
        )
        cases, constructibility_rows = filter_visually_constructible_cases(
            candidate_cases=candidate_cases,
            query_records=split_data.query_records,
            gallery_pids=split_data.gallery_pids,
            cue_affinities=cue_affinities,
            cue_thresholds=cue_thresholds,
            args=args,
            logger=logger,
        )
        write_auto_case_constructibility(args.output_dir, constructibility_rows)
        write_auto_cases_json(auto_cases_file, cases)
        write_config_used(
            args.output_dir,
            args,
            cases,
            repo_args=repo_args,
            score_mode=score_mode,
            cue_mode=cue_mode,
            case_source="pre_registered_auto",
            cue_vocab_source_value=cue_vocab_source(args),
        )
        removed = len(candidate_cases) - len(cases)
        reason_counts = Counter(row["reason"] or "kept" for row in constructibility_rows)
        logger.info("Auto constructibility removed %d cases", removed)
        logger.info("Auto constructibility top reasons: %s", dict(reason_counts.most_common(5)))
        print_auto_case_generation_block(
            cue_specs_count=len(cue_specs),
            detected_by_query=detected_by_query,
            candidate_rows=generation_rows,
            kept_cases_count=len(cases),
            auto_cases_file=auto_cases_file,
        )
        if not cases:
            raise RuntimeError(
                "No auto-generated cue cases remained after constructibility filtering. "
                "See auto_case_candidates.csv and auto_case_constructibility.csv."
            )
    else:
        logger.info("Using manual cue cases from %s", args.cases_file)
        cases = load_cases(args.cases_file)
        _, cue_features, cue_affinities, cue_thresholds = build_case_cue_affinities(
            cases=cases,
            model=model,
            cache=cache,
            repo_args=repo_args,
            device=device,
            use_grab=use_grab,
            cue_mode=cue_mode,
            lambda_global=args.lambda_global,
            cue_threshold_quantile=args.cue_threshold_quantile,
            logger=logger,
        )
        write_config_used(
            args.output_dir,
            args,
            cases,
            repo_args=repo_args,
            score_mode=score_mode,
            cue_mode=cue_mode,
            case_source="manual",
            cue_vocab_source_value=None,
        )

    selected_queries, skipped_rows = select_queries_for_cases(
        cases=cases,
        query_records=split_data.query_records,
        gallery_pids=split_data.gallery_pids,
        args=args,
        logger=logger,
    )

    if not selected_queries:
        write_outputs(args.output_dir, selected_queries, [], [], [], skipped_rows)
        raise RuntimeError("No eligible queries were selected; see skipped_queries.jsonl")

    selected_by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in selected_queries:
        selected_by_case.setdefault(str(row["case_id"]), []).append(row)

    per_query_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    gallery_rows: List[Dict[str, Any]] = []
    query_lookup = {record.query_id: record for record in split_data.query_records}

    for case in cases:
        case_id = str(case["case_id"])
        cue_a = str(case["cue_a"])
        cue_b = str(case["cue_b"])
        psi_a = cue_affinities[cue_a]
        psi_b = cue_affinities[cue_b]
        threshold_a = cue_thresholds[cue_a]
        threshold_b = cue_thresholds[cue_b]
        case_queries = selected_by_case.get(case_id, [])
        logger.info("Evaluating case_id=%s with %d selected queries", case_id, len(case_queries))

        for selected in case_queries:
            query_id = int(selected["query_id"])
            record = query_lookup[query_id]
            for trial_id in range(args.num_random_trials):
                galleries, skip_reason = construct_counterfactual_galleries(
                    pid=record.pid,
                    gallery_pids=split_data.gallery_pids,
                    psi_a=psi_a,
                    psi_b=psi_b,
                    args=args,
                    case_id=case_id,
                    query_id=query_id,
                    trial_id=trial_id,
                )
                if skip_reason is not None:
                    skipped_rows.append(
                        {
                            "case_id": case_id,
                            "query_id": query_id,
                            "pid": record.pid,
                            "trial_id": trial_id,
                            "reason": skip_reason,
                        }
                    )
                    continue

                metrics_by_gallery: Dict[str, Dict[str, float]] = {}
                density_by_gallery: Dict[str, Dict[str, float]] = {}
                crowding_by_gallery: Dict[str, float] = {}
                positives_by_gallery: Dict[str, int] = {}

                for gallery_type in GALLERY_TYPES:
                    gallery_indices = galleries[gallery_type]
                    gallery_pids = split_data.gallery_pids[gallery_indices]
                    is_positive = gallery_pids == int(record.pid)
                    scores = query_gallery_scores(
                        cache=cache,
                        query_id=query_id,
                        gallery_indices=gallery_indices,
                        mode=score_mode,
                        lambda_global=args.lambda_global,
                    )
                    metrics_by_gallery[gallery_type] = compute_retrieval_metrics(scores, is_positive)
                    hard_a, soft_a = compute_density(psi_a, gallery_indices, threshold_a, args.tau_density)
                    hard_b, soft_b = compute_density(psi_b, gallery_indices, threshold_b, args.tau_density)
                    density_by_gallery[gallery_type] = {
                        "D_hard_a": hard_a,
                        "D_soft_a": soft_a,
                        "D_hard_b": hard_b,
                        "D_soft_b": soft_b,
                    }
                    crowding_by_gallery[gallery_type] = compute_crowding(
                        scores, is_positive, args.tau_crowding
                    )
                    positives_by_gallery[gallery_type] = int(is_positive.sum())

                    gallery_rows.append(
                        {
                            "case_id": case_id,
                            "query_id": query_id,
                            "pid": record.pid,
                            "trial_id": trial_id,
                            "gallery_type": gallery_type,
                            "image_ids": gallery_indices.astype(int).tolist(),
                            "image_paths": [split_data.gallery_paths[int(i)] for i in gallery_indices],
                            "positive_image_ids": gallery_indices[is_positive].astype(int).tolist(),
                            "cue_a": cue_a,
                            "cue_b": cue_b,
                        }
                    )

                swap_strength = (
                    density_by_gallery["a_dense"]["D_soft_a"]
                    - density_by_gallery["b_dense"]["D_soft_a"]
                    + density_by_gallery["b_dense"]["D_soft_b"]
                    - density_by_gallery["a_dense"]["D_soft_b"]
                )
                rank_volatility = abs(
                    metrics_by_gallery["a_dense"]["min_positive_rank"]
                    - metrics_by_gallery["b_dense"]["min_positive_rank"]
                )
                ap_delta = metrics_by_gallery["a_dense"]["AP"] - metrics_by_gallery["b_dense"]["AP"]
                r1_flip = float(
                    metrics_by_gallery["a_dense"]["R1"] != metrics_by_gallery["b_dense"]["R1"]
                )

                paired_rows.append(
                    {
                        "case_id": case_id,
                        "query_id": query_id,
                        "pid": record.pid,
                        "trial_id": trial_id,
                        "rank_volatility": rank_volatility,
                        "ap_delta": ap_delta,
                        "r1_flip": r1_flip,
                        "swap_strength": swap_strength,
                    }
                )

                for gallery_type in GALLERY_TYPES:
                    row = {
                        "case_id": case_id,
                        "query_id": query_id,
                        "query_text": record.query_text,
                        "pid": record.pid,
                        "cue_a": cue_a,
                        "cue_b": cue_b,
                        "gallery_type": gallery_type,
                        "gallery_size": int(len(galleries[gallery_type])),
                        "num_positives": positives_by_gallery[gallery_type],
                        **metrics_by_gallery[gallery_type],
                        **density_by_gallery[gallery_type],
                        "crowding": crowding_by_gallery[gallery_type],
                        "swap_strength": swap_strength,
                        "rank_volatility": rank_volatility,
                        "ap_delta": ap_delta,
                        "r1_flip": r1_flip,
                        "seed": args.seed,
                        "trial_id": trial_id,
                        "score_mode": score_mode,
                        "cue_mode": cue_mode,
                    }
                    per_query_rows.append(row)

    per_query_df, summary_by_case, summary_overall = write_outputs(
        output_dir=args.output_dir,
        selected_queries=selected_queries,
        per_query_rows=per_query_rows,
        paired_rows=paired_rows,
        gallery_rows=gallery_rows,
        skipped_rows=skipped_rows,
    )

    if per_query_df.empty:
        raise RuntimeError("All selected queries were skipped; see skipped_queries.jsonl")

    print("\nSummary overall:")
    print(summary_overall.to_string(index=False))
    print("\nSummary by case:")
    print(summary_by_case.to_string(index=False))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    validate_cli_args(args)
    set_deterministic(args.seed)
    if args.dry_run:
        run_dry_run(args, logger)
        return
    run_evaluation(args, logger)


if __name__ == "__main__":
    main()
