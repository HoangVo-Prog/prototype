"""Cluster bootstrap confidence intervals over query clusters."""

from __future__ import annotations

from typing import Iterable, Sequence

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

    clusters = df[list(cluster_cols)].drop_duplicates().reset_index(drop=True)
    cluster_keys = [tuple(row) for row in clusters.to_numpy()]
    grouped = {key: group for key, group in df.groupby(list(cluster_cols), dropna=False)}
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "metric": metric,
                    "mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "bootstrap_iters": iters,
                    "cluster_count": int(len(cluster_keys)),
                    "trial_count": int(len(df)),
                }
            )
            continue

        boot_means = []
        for _ in range(iters):
            sampled = rng.integers(0, len(cluster_keys), size=len(cluster_keys))
            sample_values = []
            for sampled_index in sampled:
                group = grouped[cluster_keys[int(sampled_index)]]
                sample_values.extend(pd.to_numeric(group[metric], errors="coerce").dropna().tolist())
            boot_means.append(float(np.mean(sample_values)) if sample_values else np.nan)
        boot = np.asarray(boot_means, dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "ci_low": float(np.nanpercentile(boot, 2.5)),
                "ci_high": float(np.nanpercentile(boot, 97.5)),
                "bootstrap_iters": iters,
                "cluster_count": int(len(cluster_keys)),
                "trial_count": int(len(df)),
            }
        )
    return pd.DataFrame(rows, columns=columns)

