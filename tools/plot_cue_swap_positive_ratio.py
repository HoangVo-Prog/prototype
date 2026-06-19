"""Plot positive-image fraction audits for existing cue-swap output folders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


TRIAL_KEYS = ["dataset", "case_id", "query_id", "trial_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit positive-image ratios in cue-swap runs")
    parser.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def ensure_positive_ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        for column in ("num_distractors", "positive_ratio", "positive_pct"):
            if column not in df.columns:
                df[column] = []
        return df
    required = {"gallery_size", "num_positives"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"per_query_results.csv is missing required columns: {sorted(missing)}")
    if "num_distractors" not in df.columns:
        df["num_distractors"] = df["gallery_size"] - df["num_positives"]
    if "positive_ratio" not in df.columns:
        df["positive_ratio"] = df["num_positives"] / df["gallery_size"]
    if "positive_pct" not in df.columns:
        df["positive_pct"] = 100.0 * df["positive_ratio"]
    return df


def read_run(input_dir: Path, label: str) -> pd.DataFrame:
    csv_path = input_dir / "per_query_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing per_query_results.csv: {csv_path}")
    df = pd.read_csv(csv_path)
    df = ensure_positive_ratio_columns(df)
    df["dataset"] = label
    df["input_dir"] = str(input_dir)
    return df


def unique_trial_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    required = {"dataset", "case_id", "query_id", "trial_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"per_query_results.csv is missing required columns: {sorted(missing)}")
    return (
        df.sort_values(["dataset", "case_id", "query_id", "trial_id", "gallery_type"])
        .drop_duplicates(TRIAL_KEYS)
        .reset_index(drop=True)
    )


def summary_row(prefix: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            **prefix,
            "num_cases": 0,
            "num_query_trials": 0,
            "mean_positive_pct": np.nan,
            "median_positive_pct": np.nan,
            "max_positive_pct": np.nan,
            "p95_positive_pct": np.nan,
        }
    return {
        **prefix,
        "num_cases": int(df[["dataset", "case_id"]].drop_duplicates().shape[0]),
        "num_query_trials": int(len(df)),
        "mean_positive_pct": float(df["positive_pct"].mean()),
        "median_positive_pct": float(df["positive_pct"].median()),
        "max_positive_pct": float(df["positive_pct"].max()),
        "p95_positive_pct": float(df["positive_pct"].quantile(0.95)),
    }


def write_empty_outputs(output_dir: Path) -> None:
    overall_cols = [
        "num_datasets",
        "num_cases",
        "num_query_trials",
        "mean_positive_pct",
        "median_positive_pct",
        "max_positive_pct",
        "p95_positive_pct",
    ]
    dataset_cols = [
        "dataset",
        "num_cases",
        "num_query_trials",
        "mean_positive_pct",
        "median_positive_pct",
        "max_positive_pct",
        "p95_positive_pct",
    ]
    case_cols = [
        "dataset",
        "case_id",
        "num_queries",
        "num_query_trials",
        "mean_num_positives",
        "mean_positive_pct",
        "median_positive_pct",
        "max_positive_pct",
        "p95_positive_pct",
    ]
    pd.DataFrame(columns=overall_cols).to_csv(
        output_dir / "combined_positive_ratio_overall.csv", index=False
    )
    pd.DataFrame(columns=dataset_cols).to_csv(
        output_dir / "combined_positive_ratio_by_dataset.csv", index=False
    )
    pd.DataFrame(columns=case_cols).to_csv(
        output_dir / "combined_positive_ratio_by_case.csv", index=False
    )


def write_summaries(output_dir: Path, trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trial_df.empty:
        write_empty_outputs(output_dir)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    overall = pd.DataFrame(
        [
            summary_row(
                {"num_datasets": int(trial_df["dataset"].nunique())},
                trial_df,
            )
        ]
    )
    by_dataset = pd.DataFrame(
        [
            summary_row({"dataset": dataset}, group)
            for dataset, group in trial_df.groupby("dataset")
        ]
    ).sort_values("dataset")

    case_rows = []
    for (dataset, case_id), group in trial_df.groupby(["dataset", "case_id"]):
        case_rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "num_queries": int(group["query_id"].nunique()),
                "num_query_trials": int(len(group)),
                "mean_num_positives": float(group["num_positives"].mean()),
                "mean_positive_pct": float(group["positive_pct"].mean()),
                "median_positive_pct": float(group["positive_pct"].median()),
                "max_positive_pct": float(group["positive_pct"].max()),
                "p95_positive_pct": float(group["positive_pct"].quantile(0.95)),
            }
        )
    by_case = pd.DataFrame(case_rows).sort_values(["dataset", "case_id"])

    overall.to_csv(output_dir / "combined_positive_ratio_overall.csv", index=False)
    by_dataset.to_csv(output_dir / "combined_positive_ratio_by_dataset.csv", index=False)
    by_case.to_csv(output_dir / "combined_positive_ratio_by_case.csv", index=False)
    return overall, by_dataset, by_case


def plot_outputs(output_dir: Path, trial_df: pd.DataFrame, by_dataset: pd.DataFrame) -> None:
    if trial_df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(by_dataset["dataset"]) if not by_dataset.empty else sorted(trial_df["dataset"].unique())

    plt.figure(figsize=(9, 5))
    for label in labels:
        values = trial_df.loc[trial_df["dataset"] == label, "positive_pct"]
        if values.empty:
            continue
        plt.hist(values, bins=30, alpha=0.45, label=label)
    plt.xlabel("Positive images in gallery (%)")
    plt.ylabel("Number of query-trials")
    plt.title("Positive fraction in constructed galleries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "combined_positive_pct_hist.png", dpi=200)
    plt.close()

    box_values = [trial_df.loc[trial_df["dataset"] == label, "positive_pct"].dropna() for label in labels]
    box_labels = [label for label, values in zip(labels, box_values) if not values.empty]
    box_values = [values for values in box_values if not values.empty]
    if box_values:
        plt.figure(figsize=(8, 5))
        plt.boxplot(box_values, labels=box_labels, showfliers=False)
        plt.ylabel("Positive images in gallery (%)")
        plt.xlabel("Dataset")
        plt.title("Positive fraction by dataset")
        plt.tight_layout()
        plt.savefig(output_dir / "combined_positive_pct_boxplot.png", dpi=200)
        plt.close()

    if not by_dataset.empty:
        plot_df = by_dataset.sort_values("mean_positive_pct", ascending=False)
        plt.figure(figsize=(8, 5))
        plt.bar(plot_df["dataset"], plot_df["mean_positive_pct"], color="#4C78A8")
        plt.ylabel("Mean positive images in gallery (%)")
        plt.xlabel("Dataset")
        plt.title("Mean positive fraction by dataset")
        plt.tight_layout()
        plt.savefig(output_dir / "combined_mean_positive_pct_by_dataset.png", dpi=200)
        plt.close()


def print_summary(by_dataset: pd.DataFrame) -> None:
    if by_dataset.empty:
        print("No positive-ratio rows found.")
        return
    print("Positive-ratio audit summary:")
    for _, row in by_dataset.iterrows():
        print(
            f"{row['dataset']}: cases={int(row['num_cases'])}, "
            f"query_trials={int(row['num_query_trials'])}, "
            f"mean={row['mean_positive_pct']:.3f}%, "
            f"median={row['median_positive_pct']:.3f}%, "
            f"max={row['max_positive_pct']:.3f}%, "
            f"p95={row['p95_positive_pct']:.3f}%"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.labels is not None and len(args.labels) != len(args.input_dirs):
        raise ValueError("--labels must have the same length as --input_dirs")
    labels = args.labels or [path.name for path in args.input_dirs]

    runs = [read_run(input_dir, label) for input_dir, label in zip(args.input_dirs, labels)]
    combined = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()
    trial_df = unique_trial_rows(combined)

    _, by_dataset, _ = write_summaries(args.output_dir, trial_df)
    plot_outputs(args.output_dir, trial_df, by_dataset)
    print_summary(by_dataset)


if __name__ == "__main__":
    main()
