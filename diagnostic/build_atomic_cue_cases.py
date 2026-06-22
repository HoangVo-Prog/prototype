"""Build a frozen training-derived atomic cue vocabulary and cases file."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostic.cue_cases import load_cases
from diagnostic.cue_vocabulary import (
    CueDefinition,
    build_cue_pairs,
    build_matcher_audit_rows,
    collect_support,
    cue_support_csv_rows,
    cue_vocabulary_payload,
    load_all_training_captions,
    load_candidate_catalog,
    markdown_summary,
    summary_payload,
    write_csv,
    write_json,
    write_json_as_yaml_subset,
    write_markdown,
    write_yaml,
)
from diagnostic.cue_vocabulary.datasets import DATASET_ORDER
from diagnostic.cue_vocabulary.models import CuePair, DatasetLoadResult, PairExclusion
from diagnostic.cue_vocabulary.serialize import file_sha256
from diagnostic.cue_vocabulary.normalize import tokenize

DEFAULT_CATALOG = Path("diagnostic/cue_vocabulary/candidate_catalog.yaml")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a frozen atomic cue vocabulary from training captions only.")
    parser.add_argument("--cuhk_root", type=Path, required=True)
    parser.add_argument("--icfg_root", type=Path, required=True)
    parser.add_argument("--rstp_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--candidate_catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cuhk_train_annotations", type=Path, default=None)
    parser.add_argument("--icfg_train_annotations", type=Path, default=None)
    parser.add_argument("--rstp_train_annotations", type=Path, default=None)
    parser.add_argument("--min_identities", type=int, default=25)
    parser.add_argument("--min_datasets", type=int, default=2)
    parser.add_argument("--audit_samples_per_cue", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases_filename", default="cue_cases.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--emit_training_ngrams", action="store_true")
    parser.add_argument("--max_ngram_n", type=int, default=3)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_info(repo_root: Path) -> tuple[str | None, bool | None]:
    cmd_base = ["git", "-c", f"safe.directory={repo_root.as_posix()}"]
    try:
        commit = subprocess.run(
            [*cmd_base, "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except Exception:
        commit = None
    try:
        status = subprocess.run(
            [*cmd_base, "status", "--short"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return commit, dirty


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Refusing to overwrite non-empty output directory without --overwrite: {path}")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _catalog_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _validate_invariants(
    *,
    dataset_results: list[DatasetLoadResult],
    cues: list[CueDefinition],
    retained_cues: list[CueDefinition],
    decisions: dict[str, Any],
    pairs: list[CuePair],
    exclusions: list[PairExclusion],
    cases_path: Path,
) -> None:
    dataset_names = {result.dataset_name for result in dataset_results}
    if dataset_names != set(DATASET_ORDER):
        raise RuntimeError(f"Expected exactly three benchmark datasets, found {sorted(dataset_names)}")
    for result in dataset_results:
        if any(record.split != "train" for record in result.records):
            raise RuntimeError(f"Non-training record detected in {result.dataset_name}")
    retained_ids = {cue.cue_id for cue in retained_cues}
    for cue in retained_cues:
        if not decisions[cue.cue_id].retained:
            raise RuntimeError(f"Retained cue does not satisfy support rule: {cue.cue_id}")
    for cue in cues:
        if cue.cue_id not in retained_ids and decisions[cue.cue_id].retained:
            raise RuntimeError(f"Rejected cue unexpectedly satisfies support rule: {cue.cue_id}")
    all_pair_count = len(retained_cues) * max(len(retained_cues) - 1, 0) // 2
    if len(pairs) + len(exclusions) != all_pair_count:
        raise RuntimeError("Generated pairs plus exclusions do not cover all unordered retained pairs")
    seen_pairs: set[tuple[str, str]] = set()
    for pair in pairs:
        key = tuple(sorted((pair.cue_a_id, pair.cue_b_id)))
        if key in seen_pairs:
            raise RuntimeError(f"Duplicate generated pair detected: {key}")
        seen_pairs.add(key)
        if pair.cue_a_id not in retained_ids or pair.cue_b_id not in retained_ids:
            raise RuntimeError(f"Case references cue outside retained vocabulary: {pair.case_id}")
    for exclusion in exclusions:
        if not exclusion.reason:
            raise RuntimeError(f"Excluded pair missing explicit reason: {exclusion}")
    loaded_cases = load_cases(cases_path)
    if len(loaded_cases) != len(pairs):
        raise RuntimeError("Existing cases loader did not preserve generated cases")
    loaded_case_ids = {str(case["case_id"]) for case in loaded_cases}
    expected_case_ids = {pair.case_id for pair in pairs}
    if loaded_case_ids != expected_case_ids:
        raise RuntimeError("Existing cases loader changed the generated case identifiers")


def _ngrams(records: list[str], max_n: int) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, str]] = Counter()
    for caption in records:
        tokens = tokenize(caption)
        for n in range(1, max_n + 1):
            for start in range(0, max(len(tokens) - n + 1, 0)):
                counts[(n, " ".join(tokens[start : start + n]))] += 1
    rows = [
        {"n": n, "ngram": ngram, "count": count}
        for (n, ngram), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]
    return rows


def _pair_payload(pairs: list[CuePair]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": pair.case_id,
            "cue_a": pair.cue_a_runtime,
            "cue_b": pair.cue_b_runtime,
            "query_regex": pair.query_regex,
        }
        for pair in pairs
    ]


def run_builder(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_identities <= 0:
        raise ValueError("--min_identities must be positive")
    if args.min_datasets <= 0:
        raise ValueError("--min_datasets must be positive")
    if args.max_ngram_n <= 0:
        raise ValueError("--max_ngram_n must be positive")
    if args.audit_samples_per_cue <= 0:
        raise ValueError("--audit_samples_per_cue must be positive")

    repo_root = _repo_root()
    output_dir = args.output_dir.resolve()
    candidate_catalog = args.candidate_catalog.resolve()
    _ensure_output_dir(output_dir, args.overwrite)
    catalog_version, cues = load_candidate_catalog(candidate_catalog)
    dataset_results = load_all_training_captions(
        cuhk_root=args.cuhk_root,
        icfg_root=args.icfg_root,
        rstp_root=args.rstp_root,
        cuhk_train_annotations=args.cuhk_train_annotations,
        icfg_train_annotations=args.icfg_train_annotations,
        rstp_train_annotations=args.rstp_train_annotations,
    )
    decisions, evidence_by_cue = collect_support(
        dataset_results=dataset_results,
        cues=cues,
        min_identities=args.min_identities,
        min_datasets=args.min_datasets,
    )
    retained_cues = [cue for cue in cues if decisions[cue.cue_id].retained]
    if not retained_cues:
        raise RuntimeError(
            "No cues satisfied the frozen support rule. "
            "Inspect cue_support.csv after lowering thresholds only if that is methodologically intended."
        )
    pairs, exclusions = build_cue_pairs(retained_cues)
    if not pairs:
        raise RuntimeError(
            "Retained cues produced zero semantically valid pairs. "
            "Inspect pair exclusions and retained cues before running the diagnostic."
        )
    audit_rows = build_matcher_audit_rows(
        retained_cues=retained_cues,
        evidence_by_cue=evidence_by_cue,
        audit_samples_per_cue=args.audit_samples_per_cue,
        seed=args.seed,
    )

    cue_vocab_path = output_dir / "cue_vocabulary.yaml"
    cue_support_path = output_dir / "cue_support.csv"
    cases_path = output_dir / args.cases_filename
    summary_json_path = output_dir / "construction_summary.json"
    summary_md_path = output_dir / "construction_summary.md"
    matcher_audit_path = output_dir / "matcher_audit.csv"
    pair_exclusions_path = output_dir / "pair_exclusions.csv"
    ngrams_path = output_dir / "training_ngrams.csv"

    write_yaml(
        cue_vocab_path,
        cue_vocabulary_payload(
            catalog_version=catalog_version,
            retained_cues=retained_cues,
            decisions=decisions,
            min_identities=args.min_identities,
            min_datasets=args.min_datasets,
        ),
    )
    write_csv(cue_support_path, cue_support_csv_rows(cues, decisions))
    write_json_as_yaml_subset(cases_path, _pair_payload(pairs))
    write_csv(matcher_audit_path, audit_rows)
    if exclusions:
        write_csv(
            pair_exclusions_path,
            [
                {
                    "cue_a_id": exclusion.cue_a_id,
                    "cue_b_id": exclusion.cue_b_id,
                    "exclusion_reason": exclusion.reason,
                }
                for exclusion in exclusions
            ],
        )
    if args.emit_training_ngrams:
        all_captions = [record.caption for result in dataset_results for record in result.records]
        write_csv(ngrams_path, _ngrams(all_captions, args.max_ngram_n))

    output_hashes = {
        cue_vocab_path.name: file_sha256(cue_vocab_path),
        cue_support_path.name: file_sha256(cue_support_path),
        cases_path.name: file_sha256(cases_path),
        matcher_audit_path.name: file_sha256(matcher_audit_path),
    }
    if exclusions:
        output_hashes[pair_exclusions_path.name] = file_sha256(pair_exclusions_path)
    if args.emit_training_ngrams:
        output_hashes[ngrams_path.name] = file_sha256(ngrams_path)

    git_commit, git_dirty = _git_info(repo_root)
    summary = summary_payload(
        catalog_version=catalog_version,
        dataset_results=dataset_results,
        candidate_catalog_path=candidate_catalog,
        candidate_catalog_sha256=_catalog_sha256(candidate_catalog),
        cues=cues,
        retained_cues=retained_cues,
        decisions=decisions,
        min_identities=args.min_identities,
        min_datasets=args.min_datasets,
        seed=args.seed,
        pairs=pairs,
        exclusions=exclusions,
        output_hashes=output_hashes,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    write_json(summary_json_path, summary)
    write_markdown(summary_md_path, markdown_summary(summary))

    _validate_invariants(
        dataset_results=dataset_results,
        cues=cues,
        retained_cues=retained_cues,
        decisions=decisions,
        pairs=pairs,
        exclusions=exclusions,
        cases_path=cases_path,
    )
    return {
        "summary": summary,
        "retained_cues": retained_cues,
        "pairs": pairs,
        "output_dir": output_dir,
    }


def _print_console_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    retained_cues: list[CueDefinition] = result["retained_cues"]
    by_family: dict[str, list[str]] = {}
    for cue in retained_cues:
        by_family.setdefault(cue.family, []).append(cue.cue_id)
    print("Atomic cue vocabulary build complete")
    print("------------------------------------")
    print("Training captions only: yes")
    print("Support unit: distinct identity")
    print(
        f"Rule: >={summary['thresholds']['min_identities_per_dataset']} identities in "
        f">={summary['thresholds']['min_datasets']} datasets"
    )
    print(f"Candidates: {summary['num_candidate_cues']}")
    print(f"Retained cues: {summary['num_retained_cues']}")
    print(f"Rejected cues: {summary['num_rejected_cues']}")
    print(f"All unordered retained pairs: {summary['num_all_possible_pairs']}")
    print(f"Semantically excluded pairs: {summary['num_semantically_excluded_pairs']}")
    print(f"Generated cases: {summary['num_generated_cases']}")
    print("Cases loader validation: passed")
    print(f"Outputs: {result['output_dir']}")
    print("")
    print("Retained cue IDs by family:")
    for family in sorted(by_family):
        print(f"- {family}: {', '.join(sorted(by_family[family]))}")
    if summary.get("git_worktree_dirty"):
        print("")
        print("Warning: git working tree was dirty during this build.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_builder(args)
    _print_console_summary(result)


if __name__ == "__main__":
    main()
