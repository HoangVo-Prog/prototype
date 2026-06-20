"""Cluster bootstrap confidence intervals over query clusters."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    metrics: Sequence[str],
    cluster_cols: Sequence[str],
    iters: int,
    seed: int,
) -> pd.DataFrame:
    columns = [
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "bootstrap_iters",
        "cluster_count",
        "trial_count",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    missing = set(cluster_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Cluster bootstrap missing cluster columns: {sorted(missing)}")

    cluster_codes = df.groupby(list(cluster_cols), dropna=False).ngroup().to_numpy()
    cluster_count = int(cluster_codes.max()) + 1 if len(cluster_codes) else 0
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        metric_values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(metric_values)
        values = metric_values[valid_mask]
        if values.size == 0:
            rows.append(
                {
                    "metric": metric,
                    "mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "bootstrap_iters": iters,
                    "cluster_count": cluster_count,
                    "trial_count": int(len(df)),
                }
            )
            continue

        valid_codes = cluster_codes[valid_mask]
        cluster_sums = np.bincount(valid_codes, weights=values, minlength=cluster_count).astype(float)
        cluster_counts = np.bincount(valid_codes, minlength=cluster_count).astype(float)
        sampled = rng.integers(0, cluster_count, size=(iters, cluster_count))
        sampled_sums = cluster_sums[sampled].sum(axis=1)
        sampled_counts = cluster_counts[sampled].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = sampled_sums / sampled_counts
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci_low": float(np.nanpercentile(boot, 2.5)),
                "ci_high": float(np.nanpercentile(boot, 97.5)),
                "bootstrap_iters": iters,
                "cluster_count": cluster_count,
                "trial_count": int(len(df)),
            }
        )
    return pd.DataFrame(rows, columns=columns)
