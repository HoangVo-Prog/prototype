"""Positive-ratio audit for diagnostic output folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostic.constants import OUTPUT_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit positive-image ratios in diagnostic galleries")
    parser.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_run(input_dir: Path, label: str) -> pd.DataFrame:
    path = input_dir / OUTPUT_FILES["per_gallery"]
    if not path.exists():
        raise FileNotFoundError(f"Missing {OUTPUT_FILES['per_gallery']}: {path}")
    df = pd.read_csv(path)
    if "positive_ratio" not in df.columns:
        df["positive_ratio"] = df["num_positives"] / df["gallery_size"]
    df["positive_pct"] = 100.0 * df["positive_ratio"]
    df["run_label"] = label
    df["input_dir"] = str(input_dir)
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels or [path.name for path in args.input_dirs]
    if len(labels) != len(args.input_dirs):
        raise ValueError("--labels length must match --input_dirs length")
    frames = [read_run(path, label) for path, label in zip(args.input_dirs, labels)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(args.output_dir / "combined_positive_ratio_rows.csv", index=False)
    if df.empty:
        pd.DataFrame().to_csv(args.output_dir / "combined_positive_ratio_overall.csv", index=False)
        return

    group_cols = ["run_label", "dataset", "retriever_name", "construction_type", "gallery_type"]
    summary = (
        df.groupby(group_cols)
        .agg(
            num_galleries=("positive_pct", "size"),
            mean_positive_pct=("positive_pct", "mean"),
            median_positive_pct=("positive_pct", "median"),
            max_positive_pct=("positive_pct", "max"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "combined_positive_ratio_overall.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5))
        for label, group in df.groupby("run_label"):
            plt.hist(group["positive_pct"], bins=30, alpha=0.45, label=label)
        plt.xlabel("Positive images in gallery (%)")
        plt.ylabel("Number of galleries")
        plt.title("Positive fraction in constructed galleries")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "combined_positive_pct_hist.png", dpi=200)
        plt.close()
    except ImportError:
        pass
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

