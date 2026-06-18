"""Cue-swap counterfactual gallery evaluation for text-based person search.

This script implements the diagnostic protocol described in AGENTS.md without
touching the training pipeline. It reuses the repository dataset classes,
transforms, model builder, checkpoint loader, and encoder methods.
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--cases_file", type=Path, required=True)
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
        "--alpha_global",
        type=float,
        default=0.68,
        help="Global-branch weight for fusion scores.",
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


def validate_cli_args(args: argparse.Namespace) -> None:
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
    if not 0.0 <= args.alpha_global <= 1.0:
        raise ValueError("--alpha_global must be in [0, 1]")
    if args.max_queries_per_case is not None and args.max_queries_per_case <= 0:
        raise ValueError("--max_queries_per_case must be positive when provided")
    if args.neutral_pool_factor <= 0:
        raise ValueError("--neutral_pool_factor must be positive")
    if not args.dry_run and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --dry_run is set")
    if args.checkpoint is not None and not args.checkpoint.exists() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config is not None and not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")


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
    from utils.iotools import load_train_configs

    cfg = default_repo_args()
    if args.config is not None:
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
            selection_method = "query_ids"
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


def resolve_mode(mode: str, has_grab: bool, alpha_global: float, kind: str) -> str:
    if mode == "auto":
        return "fusion" if has_grab else "global"
    if mode == "score":
        raise ValueError("Internal error: resolve 'score' before calling resolve_mode")
    if mode in {"grab", "fusion"} and not has_grab:
        raise ValueError(f"{kind} mode '{mode}' requires GRAB features, but the model has no GRAB branch")
    if mode == "fusion" and alpha_global in (0.0, 1.0):
        return "global" if alpha_global == 1.0 else "grab"
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
    alpha_global: float,
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
        return (alpha_global * global_scores + (1.0 - alpha_global) * grab_scores).numpy()
    raise ValueError(f"Unsupported score mode: {mode}")


def cue_gallery_affinity(
    cache: EmbeddingCache,
    cue_features: CueFeatures,
    cue: str,
    mode: str,
    alpha_global: float,
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
        return (alpha_global * global_scores + (1.0 - alpha_global) * grab_scores).numpy()
    raise ValueError(f"Unsupported cue mode: {mode}")


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
) -> None:
    payload = {
        "script_args": to_jsonable(vars(args)),
        "cases": to_jsonable(list(cases)),
        "repo_args": to_jsonable(repo_args) if repo_args is not None else None,
        "resolved_score_mode": score_mode,
        "resolved_cue_mode": cue_mode,
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


def run_dry_run(
    args: argparse.Namespace,
    cases: Sequence[Mapping[str, Any]],
    logger: logging.Logger,
) -> None:
    write_config_used(args.output_dir, args, cases)
    logger.info("Dry run complete: validated %d cue cases", len(cases))
    print("Dry run complete")
    print(f"Validated cases: {len(cases)}")
    print(f"Output directory: {args.output_dir}")


def run_evaluation(args: argparse.Namespace, logger: logging.Logger) -> None:
    from model import build_model

    cases = load_cases(args.cases_file)
    repo_args = load_repo_args(args)

    dataset, img_loader, txt_loader, split_data = build_split_data(repo_args, args.split)
    selected_queries, skipped_rows = select_queries_for_cases(
        cases=cases,
        query_records=split_data.query_records,
        gallery_pids=split_data.gallery_pids,
        args=args,
        logger=logger,
    )

    if not selected_queries:
        write_config_used(args.output_dir, args, cases, repo_args)
        write_outputs(args.output_dir, selected_queries, [], [], [], skipped_rows)
        raise RuntimeError("No eligible queries were selected; see skipped_queries.jsonl")

    device = resolve_device(args.device)
    num_classes = len(dataset.train_id_container)
    model = build_model(repo_args, num_classes)
    load_checkpoint(model, args.checkpoint, logger)
    if device.type == "cpu":
        model = model.float()
    model.to(device)
    model.eval()

    use_grab = model_has_grab(model, repo_args)
    score_mode = resolve_mode(args.score_mode, use_grab, args.alpha_global, "score")
    cue_mode_arg = score_mode if args.cue_mode == "score" else args.cue_mode
    cue_mode = resolve_mode(cue_mode_arg, use_grab, args.alpha_global, "cue")
    write_config_used(args.output_dir, args, cases, repo_args, score_mode, cue_mode)
    logger.info("Resolved score_mode=%s cue_mode=%s use_grab=%s", score_mode, cue_mode, use_grab)

    cache = build_embedding_cache(model, img_loader, txt_loader, device, use_grab, logger)
    unique_cues = sorted({str(case["cue_a"]) for case in cases} | {str(case["cue_b"]) for case in cases})
    cue_features = encode_cue_features(
        model=model,
        cues=unique_cues,
        text_length=repo_args.text_length,
        device=device,
        use_grab=use_grab,
        logger=logger,
    )
    cue_affinities = {
        cue: cue_gallery_affinity(cache, cue_features, cue, cue_mode, args.alpha_global)
        for cue in unique_cues
    }
    cue_thresholds = {
        cue: float(np.quantile(scores, args.cue_threshold_quantile))
        for cue, scores in cue_affinities.items()
    }

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
                        alpha_global=args.alpha_global,
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
    cases = load_cases(args.cases_file)
    if args.dry_run:
        run_dry_run(args, cases, logger)
        return
    run_evaluation(args, logger)


if __name__ == "__main__":
    main()
