"""Aggregate multiple cue-swap diagnostic output folders into paper tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostic.constants import OUTPUT_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate cue-swap diagnostic runs")
    parser.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_config(path: Path) -> dict[str, Any]:
    config_path = path / OUTPUT_FILES["config"]
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ci_lookup(summary_ci: pd.DataFrame) -> dict[str, dict[str, float]]:
    if summary_ci.empty or "metric" not in summary_ci.columns:
        return {}
    return {
        str(row["metric"]): {
            "mean": row.get("mean"),
            "ci_low": row.get("ci_low"),
            "ci_high": row.get("ci_high"),
        }
        for _, row in summary_ci.iterrows()
    }


def run_row(input_dir: Path) -> dict[str, Any]:
    summary = read_csv_optional(input_dir / OUTPUT_FILES["summary_overall"])
    validity = read_csv_optional(input_dir / OUTPUT_FILES["validity_counts"])
    summary_ci = read_csv_optional(input_dir / OUTPUT_FILES["summary_ci"])
    config = read_config(input_dir)
    args = config.get("args", {})
    base = summary.iloc[0].to_dict() if not summary.empty else {}
    validity_row = validity.iloc[0].to_dict() if not validity.empty else {}
    ci = ci_lookup(summary_ci)

    def ci_value(metric: str, field: str) -> Any:
        return ci.get(metric, {}).get(field)

    return {
        "input_dir": str(input_dir),
        "dataset": base.get("dataset", args.get("dataset")),
        "retriever_name": base.get("retriever_name", args.get("retriever_name")),
        "cue_scorer": base.get("cue_scorer", args.get("cue_scorer")),
        "ref_R1": base.get("ref_R1"),
        "num_cases": base.get("num_cases"),
        "num_queries": base.get("num_queries", validity_row.get("unique_queries")),
        "num_pairs": base.get("num_pairs", validity_row.get("valid_pairs")),
        "valid_pair_rate": base.get("valid_pair_rate", validity_row.get("valid_pair_rate")),
        "mean_cue_shift": base.get("mean_cue_shift"),
        "r1_flip": ci_value("r1_flip", "mean"),
        "r1_flip_ci_low": ci_value("r1_flip", "ci_low"),
        "r1_flip_ci_high": ci_value("r1_flip", "ci_high"),
        "rank_shift": ci_value("rank_shift", "mean"),
        "rank_shift_ci_low": ci_value("rank_shift", "ci_low"),
        "rank_shift_ci_high": ci_value("rank_shift", "ci_high"),
        "hm_r1_flip": ci_value("hm_r1_flip", "mean"),
        "hm_r1_flip_ci_low": ci_value("hm_r1_flip", "ci_low"),
        "hm_r1_flip_ci_high": ci_value("hm_r1_flip", "ci_high"),
        "delta_r1_flip": ci_value("delta_r1_flip", "mean"),
        "delta_r1_flip_ci_low": ci_value("delta_r1_flip", "ci_low"),
        "delta_r1_flip_ci_high": ci_value("delta_r1_flip", "ci_high"),
        "hm_rank_shift": ci_value("hm_rank_shift", "mean"),
        "hm_rank_shift_ci_low": ci_value("hm_rank_shift", "ci_low"),
        "hm_rank_shift_ci_high": ci_value("hm_rank_shift", "ci_high"),
        "delta_rank_shift": ci_value("delta_rank_shift", "mean"),
        "delta_rank_shift_ci_low": ci_value("delta_rank_shift", "ci_low"),
        "delta_rank_shift_ci_high": ci_value("delta_rank_shift", "ci_high"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_row(input_dir) for input_dir in args.input_dirs]
    combined = pd.DataFrame(rows)

    general_cols = [
        "dataset",
        "retriever_name",
        "cue_scorer",
        "ref_R1",
        "num_cases",
        "num_queries",
        "num_pairs",
        "valid_pair_rate",
        "mean_cue_shift",
        "r1_flip",
        "r1_flip_ci_low",
        "r1_flip_ci_high",
        "rank_shift",
        "rank_shift_ci_low",
        "rank_shift_ci_high",
    ]
    hardness_cols = [
        "dataset",
        "retriever_name",
        "cue_scorer",
        "hm_r1_flip",
        "hm_r1_flip_ci_low",
        "hm_r1_flip_ci_high",
        "r1_flip",
        "r1_flip_ci_low",
        "r1_flip_ci_high",
        "delta_r1_flip",
        "delta_r1_flip_ci_low",
        "delta_r1_flip_ci_high",
        "hm_rank_shift",
        "hm_rank_shift_ci_low",
        "hm_rank_shift_ci_high",
        "rank_shift",
        "rank_shift_ci_low",
        "rank_shift_ci_high",
        "delta_rank_shift",
        "delta_rank_shift_ci_low",
        "delta_rank_shift_ci_high",
    ]
    validity_cols = [
        "dataset",
        "retriever_name",
        "cue_scorer",
        "num_cases",
        "num_queries",
        "num_pairs",
        "valid_pair_rate",
        "mean_cue_shift",
    ]
    combined.reindex(columns=general_cols).to_csv(args.output_dir / "generalized_cue_swap_table.csv", index=False)
    combined.reindex(columns=hardness_cols).to_csv(args.output_dir / "hardness_control_table.csv", index=False)
    combined.reindex(columns=validity_cols).to_csv(args.output_dir / "validity_counts_table.csv", index=False)
    print(combined.reindex(columns=general_cols).to_string(index=False))


if __name__ == "__main__":
    main()

