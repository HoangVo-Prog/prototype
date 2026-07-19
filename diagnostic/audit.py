"""Audit helpers and warnings for diagnostic validity."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from diagnostic.bootstrap import bootstrap_count_summary, cluster_bootstrap_callback, cluster_bootstrap_ci
from diagnostic.constants import (
    HARDNESS_AUDIT_METRICS,
    HARDNESS_AUDIT_COLUMNS,
    RESIDUAL_HARDNESS_ADJUSTED_METRICS,
    RESIDUAL_HARDNESS_ADJUSTED_SUMMARY_COLUMNS,
    SUMMARY_CI_COLUMNS,
    TIGHT_HARDNESS_METRICS,
    TIGHT_HARDNESS_SUMMARY_COLUMNS,
    UNIQUE_QUERY_CLUSTER_COLS,
)


def empty_ci_rows(
    metrics: Sequence[str],
    *,
    bootstrap_iters: int,
    bootstrap_unit: str,
    cluster_count: int = 0,
    unique_query_count: int = 0,
    case_query_count: int = 0,
    trial_count: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "bootstrap_iters": int(bootstrap_iters),
                "bootstrap_unit": bootstrap_unit,
                "cluster_count": int(cluster_count),
                "unique_query_count": int(unique_query_count),
                "case_query_count": int(case_query_count),
                "trial_count": int(trial_count),
            }
            for metric in metrics
        ],
        columns=SUMMARY_CI_COLUMNS,
    )


def _infer_retriever_name(df: pd.DataFrame, retriever_name: str | None) -> str:
    if retriever_name is not None:
        return str(retriever_name)
    if "retriever_name" not in df.columns or df.empty:
        return ""
    values = [str(value) for value in df["retriever_name"].dropna().unique().tolist()]
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return "mixed"
    return ""


def _attach_retriever_column(ci: pd.DataFrame, retriever_name: str) -> pd.DataFrame:
    ci = ci.reindex(columns=SUMMARY_CI_COLUMNS)
    ci.insert(0, "retriever", retriever_name)
    return ci.reindex(columns=HARDNESS_AUDIT_COLUMNS)


def _prepare_hardness_audit_frame(paired_delta_df: pd.DataFrame) -> pd.DataFrame:
    df = paired_delta_df.copy()
    if (
        "mean_normalized_max_negative_gap" not in df.columns
        and "a_normalized_max_negative_gap" in df.columns
        and "b_normalized_max_negative_gap" in df.columns
    ):
        a_gap = pd.to_numeric(df["a_normalized_max_negative_gap"], errors="coerce")
        b_gap = pd.to_numeric(df["b_normalized_max_negative_gap"], errors="coerce")
        df["mean_normalized_max_negative_gap"] = 0.5 * (a_gap + b_gap)
    if "mean_normalized_max_negative_gap" in df.columns:
        df["mean_signed_normalized_max_negative_gap"] = pd.to_numeric(
            df["mean_normalized_max_negative_gap"],
            errors="coerce",
        )
    else:
        df["mean_signed_normalized_max_negative_gap"] = np.nan
    if "max_abs_normalized_max_negative_gap" in df.columns:
        df["mean_max_abs_normalized_max_negative_gap"] = pd.to_numeric(
            df["max_abs_normalized_max_negative_gap"],
            errors="coerce",
        )
    else:
        df["mean_max_abs_normalized_max_negative_gap"] = np.nan
    return df


def build_hardness_audit_with_ci(
    paired_delta_df: pd.DataFrame,
    *,
    bootstrap_iters: int,
    bootstrap_seed: int,
    retriever_name: str | None = None,
) -> pd.DataFrame:
    resolved_retriever = _infer_retriever_name(paired_delta_df, retriever_name)
    audit_df = _prepare_hardness_audit_frame(paired_delta_df)
    if audit_df.empty:
        return _attach_retriever_column(
            empty_ci_rows(
                HARDNESS_AUDIT_METRICS,
                bootstrap_iters=bootstrap_iters,
                bootstrap_unit="unique_query",
            ),
            resolved_retriever,
        )
    result = cluster_bootstrap_ci(
        audit_df,
        HARDNESS_AUDIT_METRICS,
        cluster_cols=UNIQUE_QUERY_CLUSTER_COLS,
        iters=bootstrap_iters,
        seed=bootstrap_seed,
        bootstrap_unit="unique_query",
    )
    if result.empty:
        counts = bootstrap_count_summary(audit_df, UNIQUE_QUERY_CLUSTER_COLS)
        result = empty_ci_rows(
            HARDNESS_AUDIT_METRICS,
            bootstrap_iters=bootstrap_iters,
            bootstrap_unit="unique_query",
            **counts,
        )
    present_metrics = set(result["metric"].tolist()) if "metric" in result.columns else set()
    missing_metrics = [metric for metric in HARDNESS_AUDIT_METRICS if metric not in present_metrics]
    if missing_metrics:
        counts = bootstrap_count_summary(audit_df, UNIQUE_QUERY_CLUSTER_COLS)
        result = pd.concat(
            [
                result,
                empty_ci_rows(
                    missing_metrics,
                    bootstrap_iters=bootstrap_iters,
                    bootstrap_unit="unique_query",
                    **counts,
                ),
            ],
            ignore_index=True,
        )
    return _attach_retriever_column(result, resolved_retriever)


def residual_hardness_valid_frame(
    paired_delta_df: pd.DataFrame,
    *,
    outcome: str = "delta_r1_flip",
    covariate: str = "mean_normalized_max_negative_gap",
) -> pd.DataFrame:
    df = _prepare_hardness_audit_frame(paired_delta_df)
    if covariate not in df.columns or outcome not in df.columns:
        return df.iloc[0:0].copy()
    y = pd.to_numeric(df[outcome], errors="coerce")
    g = pd.to_numeric(df[covariate], errors="coerce")
    valid_mask = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(g.to_numpy(dtype=float))
    valid_df = df.loc[valid_mask].copy()
    valid_df[outcome] = y.loc[valid_mask].astype(float)
    valid_df[covariate] = g.loc[valid_mask].astype(float)
    return valid_df


def fit_residual_hardness_adjustment(
    paired_delta_df: pd.DataFrame,
    *,
    outcome: str = "delta_r1_flip",
    covariate: str = "mean_normalized_max_negative_gap",
    constant_tolerance: float = 1e-12,
) -> dict[str, float]:
    valid_df = residual_hardness_valid_frame(
        paired_delta_df,
        outcome=outcome,
        covariate=covariate,
    )
    empty = {
        "raw_delta_r1_flip": np.nan,
        "adjusted_delta_r1_flip_at_zero_gap": np.nan,
        "hardness_slope_delta_per_z": np.nan,
        "adjustment_change": np.nan,
    }
    if valid_df.empty:
        return empty

    y = valid_df[outcome].to_numpy(dtype=float)
    g = valid_df[covariate].to_numpy(dtype=float)
    if y.size == 0 or g.size == 0:
        return empty

    raw_delta = float(np.mean(y))
    if (
        float(np.max(g) - np.min(g)) <= float(constant_tolerance)
        or float(np.max(np.abs(g))) <= float(constant_tolerance)
    ):
        return {
            "raw_delta_r1_flip": raw_delta,
            "adjusted_delta_r1_flip_at_zero_gap": raw_delta,
            "hardness_slope_delta_per_z": 0.0,
            "adjustment_change": 0.0,
        }

    design = np.column_stack([np.ones(len(g), dtype=float), g])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    adjusted = float(alpha)
    return {
        "raw_delta_r1_flip": raw_delta,
        "adjusted_delta_r1_flip_at_zero_gap": adjusted,
        "hardness_slope_delta_per_z": float(beta),
        "adjustment_change": float(adjusted - raw_delta),
    }


def build_residual_hardness_adjusted_summary_with_ci(
    paired_delta_df: pd.DataFrame,
    *,
    retriever_name: str,
    bootstrap_iters: int,
    bootstrap_seed: int,
    outcome: str = "delta_r1_flip",
    covariate: str = "mean_normalized_max_negative_gap",
) -> pd.DataFrame:
    model = "ols_paired_delta_on_normalized_hardness"
    valid_df = residual_hardness_valid_frame(
        paired_delta_df,
        outcome=outcome,
        covariate=covariate,
    )
    if valid_df.empty:
        counts = bootstrap_count_summary(valid_df, UNIQUE_QUERY_CLUSTER_COLS)
        point = fit_residual_hardness_adjustment(valid_df, outcome=outcome, covariate=covariate)
        bootstrap_stats: list[dict[str, float]] = []
    else:
        point, bootstrap_stats, counts = cluster_bootstrap_callback(
            valid_df,
            UNIQUE_QUERY_CLUSTER_COLS,
            bootstrap_iters,
            bootstrap_seed,
            lambda sample: fit_residual_hardness_adjustment(
                sample,
                outcome=outcome,
                covariate=covariate,
            ),
        )

    rows = []
    for metric in RESIDUAL_HARDNESS_ADJUSTED_METRICS:
        boot_values = np.asarray(
            [
                float(stat[metric])
                for stat in bootstrap_stats
                if metric in stat and np.isfinite(float(stat[metric]))
            ],
            dtype=float,
        )
        valid_bootstrap_iters = int(len(boot_values))
        if valid_bootstrap_iters >= 2:
            ci_low = float(np.nanpercentile(boot_values, 2.5))
            ci_high = float(np.nanpercentile(boot_values, 97.5))
        else:
            ci_low = np.nan
            ci_high = np.nan
        rows.append(
            {
                "retriever": str(retriever_name),
                "metric": metric,
                "mean": float(point.get(metric, np.nan)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_iters": int(bootstrap_iters),
                "valid_bootstrap_iters": valid_bootstrap_iters,
                "bootstrap_unit": "unique_query",
                "cluster_count": counts["cluster_count"],
                "unique_query_count": counts["unique_query_count"],
                "case_query_count": counts["case_query_count"],
                "trial_count": counts["trial_count"],
                "outcome": outcome,
                "covariate": covariate,
                "model": model,
            }
        )
    return pd.DataFrame(rows, columns=RESIDUAL_HARDNESS_ADJUSTED_SUMMARY_COLUMNS)


def build_tight_hardness_summary_with_ci(
    paired_delta_df: pd.DataFrame,
    *,
    retriever_name: str,
    tight_hardness_z_tolerance: float,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    all_valid_trial_count = int(len(paired_delta_df))
    if paired_delta_df.empty or "tight_hardness_match" not in paired_delta_df.columns:
        tight_df = paired_delta_df.iloc[0:0].copy()
    else:
        tight_mask = paired_delta_df["tight_hardness_match"].fillna(False).astype(bool)
        tight_df = paired_delta_df.loc[tight_mask].copy()

    tight_trial_count = int(len(tight_df))
    tight_trial_rate = float(tight_trial_count / all_valid_trial_count) if all_valid_trial_count else 0.0
    counts = bootstrap_count_summary(tight_df, UNIQUE_QUERY_CLUSTER_COLS)
    if tight_df.empty:
        ci = empty_ci_rows(
            TIGHT_HARDNESS_METRICS,
            bootstrap_iters=bootstrap_iters,
            bootstrap_unit="unique_query",
            **counts,
        )
    else:
        ci = cluster_bootstrap_ci(
            tight_df,
            TIGHT_HARDNESS_METRICS,
            cluster_cols=UNIQUE_QUERY_CLUSTER_COLS,
            iters=bootstrap_iters,
            seed=bootstrap_seed,
            bootstrap_unit="unique_query",
        )
        if ci.empty:
            ci = empty_ci_rows(
                TIGHT_HARDNESS_METRICS,
                bootstrap_iters=bootstrap_iters,
                bootstrap_unit="unique_query",
                **counts,
            )

    ci = ci.reindex(columns=SUMMARY_CI_COLUMNS)
    ci.insert(0, "tight_trial_rate", tight_trial_rate)
    ci.insert(0, "tight_case_query_count", counts["case_query_count"])
    ci.insert(0, "tight_unique_query_count", counts["unique_query_count"])
    ci.insert(0, "tight_trial_count", tight_trial_count)
    ci.insert(0, "tight_hardness_z_tolerance", float(tight_hardness_z_tolerance))
    ci.insert(0, "retriever", retriever_name)
    return ci.reindex(columns=TIGHT_HARDNESS_SUMMARY_COLUMNS)


def log_validity_warnings(
    summary_overall: pd.DataFrame,
    logger: logging.Logger,
    min_valid_pair_rate: float = 0.2,
    min_mean_cue_shift: float = 0.01,
) -> None:
    if summary_overall.empty:
        logger.warning("No summary rows were produced.")
        return
    row = summary_overall.iloc[0]
    if float(row.get("valid_pair_rate", 0.0)) < min_valid_pair_rate:
        logger.warning("Low valid-pair rate: %.4f", float(row.get("valid_pair_rate", 0.0)))
    cue_shift = row.get("mean_cue_shift")
    if pd.notna(cue_shift) and float(cue_shift) < min_mean_cue_shift:
        logger.warning("Weak mean Cue Shift: %.4f", float(cue_shift))

