"""Training-only dataset adapters for official benchmark files."""

from __future__ import annotations

import json
from pathlib import Path

from diagnostic.cue_vocabulary.models import DatasetLoadResult, TrainingCaption

DATASET_ORDER = ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid")

_DATASET_CONFIG = {
    "CUHK-PEDES": {"annotation_name": "reid_raw.json", "image_key": "file_path"},
    "ICFG-PEDES": {"annotation_name": "ICFG-PEDES.json", "image_key": "file_path"},
    "RSTPReid": {"annotation_name": "data_captions.json", "image_key": "img_path"},
}
_OVERRIDE_ARG_NAMES = {
    "CUHK-PEDES": "--cuhk_train_annotations",
    "ICFG-PEDES": "--icfg_train_annotations",
    "RSTPReid": "--rstp_train_annotations",
}


def _candidate_annotation_files(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.iterdir() if path.is_file() and path.suffix.lower() == ".json")


def resolve_annotation_file(dataset_name: str, dataset_root: Path, override: Path | None = None) -> Path:
    dataset_root = dataset_root.resolve()
    if override is not None:
        override = override.resolve()
        if not override.exists():
            raise FileNotFoundError(f"Explicit training annotation override does not exist: {override}")
        return override
    expected_name = _DATASET_CONFIG[dataset_name]["annotation_name"]
    expected_path = dataset_root / expected_name
    if expected_path.exists():
        return expected_path
    candidates = _candidate_annotation_files(dataset_root) if dataset_root.exists() else []
    candidate_list = ", ".join(str(path.name) for path in candidates) if candidates else "none"
    raise FileNotFoundError(
        f"Could not resolve official training annotation for {dataset_name}. "
        f"Expected {expected_path}. Direct JSON candidates under root: {candidate_list}. "
        f"Use the explicit {_OVERRIDE_ARG_NAMES[dataset_name]} override if needed."
    )


def _load_training_records(dataset_name: str, dataset_root: Path, annotation_file: Path) -> DatasetLoadResult:
    dataset_root = dataset_root.resolve()
    annotation_file = annotation_file.resolve()
    with annotation_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Official annotation file must contain a non-empty list: {annotation_file}")

    image_key = _DATASET_CONFIG[dataset_name]["image_key"]
    records: list[TrainingCaption] = []
    identities: set[str] = set()
    observed_splits: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Annotation record {index} in {annotation_file} must be a mapping")
        split = record.get("split")
        if not isinstance(split, str):
            raise ValueError(f"Annotation record {index} in {annotation_file} is missing string 'split'")
        observed_splits.add(split)
        if split != "train":
            continue
        if "id" not in record:
            raise ValueError(f"Training record {index} in {annotation_file} is missing 'id'")
        if "captions" not in record or not isinstance(record["captions"], list):
            raise ValueError(f"Training record {index} in {annotation_file} must define list 'captions'")
        identity_id = str(record["id"])
        image_value = record.get(image_key)
        image_ref = None if image_value in (None, "") else str((dataset_root / "imgs" / str(image_value)).resolve())
        identities.add(identity_id)
        for caption in record["captions"]:
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"Training caption for identity {identity_id} in {annotation_file} must be non-empty string")
            records.append(
                TrainingCaption(
                    dataset=dataset_name,
                    identity_id=identity_id,
                    caption=caption,
                    image_ref=image_ref,
                    annotation_file=str(annotation_file),
                    split="train",
                )
            )
    if "train" not in observed_splits:
        raise RuntimeError(f"Could not prove any record belongs to the training split in {annotation_file}")
    if not records:
        raise RuntimeError(f"No training captions found in official file {annotation_file}")
    return DatasetLoadResult(
        dataset_name=dataset_name,
        annotation_file=annotation_file,
        records=tuple(records),
        num_training_identities=len(identities),
        num_training_captions=len(records),
    )


def load_dataset_training_captions(
    dataset_name: str,
    dataset_root: Path,
    override: Path | None = None,
) -> DatasetLoadResult:
    if dataset_name not in _DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist for {dataset_name}: {dataset_root}")
    annotation_file = resolve_annotation_file(dataset_name, dataset_root, override)
    return _load_training_records(dataset_name, dataset_root, annotation_file)


def load_all_training_captions(
    *,
    cuhk_root: Path,
    icfg_root: Path,
    rstp_root: Path,
    cuhk_train_annotations: Path | None = None,
    icfg_train_annotations: Path | None = None,
    rstp_train_annotations: Path | None = None,
) -> list[DatasetLoadResult]:
    return [
        load_dataset_training_captions("CUHK-PEDES", cuhk_root, cuhk_train_annotations),
        load_dataset_training_captions("ICFG-PEDES", icfg_root, icfg_train_annotations),
        load_dataset_training_captions("RSTPReid", rstp_root, rstp_train_annotations),
    ]
