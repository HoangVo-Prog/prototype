from __future__ import annotations

import argparse
import contextlib
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


ROOT = Path(__file__).resolve().parent
CSV_PATHS = [
    Path(r"C:\Users\Hi\OneDrive - VNU-HCMUS\Download\wandb_export_2026-06-02T21_26_54.842+07_00.csv"),
    Path(r"C:\Users\Hi\OneDrive - VNU-HCMUS\Download\wandb_export_2026-06-02T21_21_26.432+07_00.csv"),
]
REPORT_PATH = ROOT / "target_aware_hyperparam_report.md"

GROUPS = {
    "target-aware text enrichment": [
        "target_enrichment",
        "enrichment_start",
        "enrichment_space",
        "pool_k_mode",
        "pool_k",
        "pool_k_candidates",
        "use_shared_k",
        "pool_coverage_epochs",
        "top_m",
        "extractor_mode",
        "num_parts",
        "use_freeze_indices",
        "pnp_text_only",
        "robust_hard_k",
        "pool_clusters",
        "positive_ratio_max",
        "pool_dist_metric",
        "pool_dist_threshold",
        "enrich_gamma",
        "residual_gate",
        "residual_gate_hidden_dim",
        "recompute_level",
        "recompute_interval",
    ],
    "mlp-mixer module settings": [
        "context_module",
        "mixer_dim",
        "mixer_depth",
        "mixer_hidden_part",
        "mixer_hidden_rank",
        "mixer_hidden_channel",
        "mixer_hidden_readout",
        "context_pooling",
    ],
    "target-aware loss settings": [
        "lambda_ret",
        "lambda_rob",
        "lambda_gain",
        "gain_margin",
        "use_target_retrieval_loss",
        "use_target_robust_loss",
    ],
}
PARAMS = [param for params in GROUPS.values() for param in params]

ALIASES = {
    "freeze_indices": "use_freeze_indices",
    "hard_neg_k": "robust_hard_k",
    "eta": "positive_ratio_max",
    "epsilon": "pool_dist_threshold",
    "gate_mode": "residual_gate",
    "pool_interval": "recompute_interval",
    "mixer_context_pooling": "context_pooling",
}

CLI_META = {
    "target_enrichment": {"default": False, "kind": "flag", "choices": "", "aliases": "", "validation": ""},
    "enrichment_start": {"default": 1, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "enrichment_space": {"default": "global", "kind": "str", "choices": "global, grab", "aliases": "", "validation": ""},
    "pool_k_mode": {"default": "static", "kind": "str", "choices": "static, adaptive", "aliases": "", "validation": ""},
    "pool_k": {"default": 1024, "kind": "int", "choices": "", "aliases": "", "validation": ""},
    "pool_k_candidates": {"default": "512,1024,2048,4096,8192", "kind": "str", "choices": "", "aliases": "", "validation": ""},
    "use_shared_k": {"default": False, "kind": "flag", "choices": "", "aliases": "", "validation": "requires target_enrichment"},
    "pool_coverage_epochs": {"default": 15, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "top_m": {"default": 32, "kind": "int", "choices": "", "aliases": "", "validation": ""},
    "extractor_mode": {
        "default": "global,horizontal",
        "kind": "str",
        "choices": "comma-separated: global, horizontal, vertical, grid",
        "aliases": "global_horizontal, global_vertical, global_grid",
        "validation": "at least one supported mode; aliases expand and duplicates are removed",
    },
    "num_parts": {"default": 6, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "use_freeze_indices": {"default": False, "kind": "flag", "choices": "", "aliases": "--freeze_indices", "validation": "requires target_enrichment"},
    "pnp_text_only": {
        "default": False,
        "kind": "flag",
        "choices": "",
        "aliases": "",
        "validation": "requires freeze_host, no_use_host_loss, use_freeze_indices, enrichment_space=global",
    },
    "robust_hard_k": {"default": 32, "kind": "int", "choices": "", "aliases": "--hard_neg_k", "validation": ""},
    "pool_clusters": {"default": 16, "kind": "int", "choices": "", "aliases": "", "validation": ""},
    "positive_ratio_max": {"default": 0.5, "kind": "float", "choices": "", "aliases": "--eta", "validation": ""},
    "pool_dist_metric": {"default": "l1", "kind": "str", "choices": "l1, js", "aliases": "", "validation": ""},
    "pool_dist_threshold": {"default": 0.25, "kind": "float", "choices": "", "aliases": "--epsilon", "validation": ""},
    "enrich_gamma": {"default": None, "kind": "optional_float", "choices": "", "aliases": "", "validation": "required only when residual_gate=static; forbidden when residual_gate=residual"},
    "residual_gate": {"default": "residual", "kind": "str", "choices": "static, residual", "aliases": "--gate_mode", "validation": ""},
    "residual_gate_hidden_dim": {"default": 128, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "recompute_level": {"default": "epoch", "kind": "str", "choices": "epoch, step", "aliases": "", "validation": ""},
    "recompute_interval": {"default": 1, "kind": "int", "choices": "", "aliases": "--pool_interval", "validation": "-1 means compute once; otherwise refresh every N epochs/steps"},
    "context_module": {"default": "mixer", "kind": "str", "choices": "mixer", "aliases": "", "validation": ""},
    "mixer_dim": {"default": 256, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "mixer_depth": {"default": 2, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "mixer_hidden_part": {"default": 32, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "mixer_hidden_rank": {"default": 64, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "mixer_hidden_channel": {"default": 512, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "mixer_hidden_readout": {"default": 128, "kind": "int", "choices": "", "aliases": "", "validation": ">= 1"},
    "context_pooling": {"default": "mlp", "kind": "str", "choices": "mlp, late_attention, hybrid_attention", "aliases": "--mixer_context_pooling", "validation": ""},
    "lambda_ret": {"default": 1.0, "kind": "float", "choices": "", "aliases": "", "validation": ""},
    "lambda_rob": {"default": 0.1, "kind": "float", "choices": "", "aliases": "", "validation": ""},
    "lambda_gain": {"default": 1.0, "kind": "float", "choices": "", "aliases": "", "validation": ""},
    "gain_margin": {"default": 0.01, "kind": "float", "choices": "", "aliases": "", "validation": ""},
    "use_target_retrieval_loss": {"default": False, "kind": "flag", "choices": "", "aliases": "", "validation": ""},
    "use_target_robust_loss": {"default": False, "kind": "flag", "choices": "", "aliases": "", "validation": ""},
}

EXTRACTOR_ALIASES = {
    "global_horizontal": "global,horizontal",
    "global_vertical": "global,vertical",
    "global_grid": "global,grid",
}
EXTRACTOR_BASE = ("global", "horizontal", "vertical", "grid")


def load_parser_defaults() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    old_argv = sys.argv[:]
    try:
        sys.argv = ["target_aware_wandb_analysis"]
        from utils.options import get_args

        return dict(vars(get_args()))
    finally:
        sys.argv = old_argv
        with contextlib.suppress(ValueError):
            sys.path.remove(str(ROOT))


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_bool(value: Any, default: bool) -> bool:
    if is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return default


def normalize_extractor_mode(value: Any) -> str:
    if is_missing(value) or str(value).strip() == "":
        return "global,horizontal"
    raw_tokens = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    expanded: list[str] = []
    for token in raw_tokens:
        expanded.extend(EXTRACTOR_ALIASES.get(token, token).split(","))
    normalized: list[str] = []
    for token in expanded:
        if token in EXTRACTOR_BASE and token not in normalized:
            normalized.append(token)
    return ",".join(normalized) if normalized else "global,horizontal"


def coerce_param(param: str, value: Any, parser_defaults: dict[str, Any]) -> Any:
    default = parser_defaults.get(param, CLI_META[param]["default"])
    if is_missing(value) or str(value).strip() == "":
        value = default

    kind = CLI_META[param]["kind"]
    if param == "extractor_mode":
        return normalize_extractor_mode(value)
    if kind == "flag":
        return parse_bool(value, bool(default))
    if kind == "int":
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)
    if kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
    if kind == "optional_float":
        if is_missing(value) or str(value).strip().lower() in {"", "none", "nan"}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return str(value).strip()


def value_key(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return "None"
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def values_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def parse_runtime_seconds(value: Any) -> float | None:
    if is_missing(value) or str(value).strip() == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    try:
        return float(text)
    except ValueError:
        pass

    total = 0.0
    matched = False
    for number, unit in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*([dhms])", text):
        matched = True
        mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
        total += float(number) * mult
    if matched:
        return total
    hms = text.split(":")
    if all(part.replace(".", "", 1).isdigit() for part in hms):
        nums = [float(part) for part in hms]
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
    return None


def read_exports(paths: list[Path], parser_defaults: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for index, path in enumerate(paths):
        df = pd.read_csv(path)
        df["_source_path"] = str(path)
        df["_source_rank"] = index
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True, sort=False)

    # Resolve aliases if any export used legacy names.
    for alias, canonical in ALIASES.items():
        if alias in raw.columns and canonical not in raw.columns:
            raw[canonical] = raw[alias]
        elif alias in raw.columns and canonical in raw.columns:
            raw[canonical] = raw[canonical].where(~raw[canonical].isna(), raw[alias])

    if "Name" in raw.columns:
        raw["_identity"] = raw["Name"].astype(str)
    elif "ID" in raw.columns:
        raw["_identity"] = raw["ID"].fillna("").astype(str)
    else:
        raw["_identity"] = raw.index.astype(str)
    raw["_non_null_count"] = raw.notna().sum(axis=1)
    raw = raw.sort_values(["_identity", "_source_rank", "_non_null_count"], ascending=[True, False, False])
    raw = raw.drop_duplicates("_identity", keep="last").reset_index(drop=True)

    for param in PARAMS:
        if param not in raw.columns:
            raw[param] = pd.NA
        raw[f"cfg__{param}"] = [coerce_param(param, value, parser_defaults) for value in raw[param]]

    raw["best_R1"] = pd.to_numeric(raw.get("best_R1"), errors="coerce")
    raw["epoch"] = pd.to_numeric(raw.get("epoch"), errors="coerce")
    raw["num_epoch"] = pd.to_numeric(raw.get("num_epoch"), errors="coerce")
    raw["_runtime_seconds"] = [parse_runtime_seconds(v) for v in raw.get("Runtime", pd.Series([None] * len(raw)))]
    raw["_name"] = raw.get("Name", pd.Series([""] * len(raw))).astype(str)
    raw["_state"] = raw.get("State", pd.Series([""] * len(raw))).astype(str).str.lower()

    diff_lists = []
    diff_counts = []
    diff_groups = []
    for _, row in raw.iterrows():
        diffs = []
        groups = set()
        for group, params in GROUPS.items():
            for param in params:
                val = row[f"cfg__{param}"]
                default = parser_defaults.get(param, CLI_META[param]["default"])
                if not values_equal(val, default):
                    diffs.append(param)
                    groups.add(group)
        diff_lists.append(diffs)
        diff_counts.append(len(diffs))
        diff_groups.append("; ".join(sorted(groups)))
    raw["_diff_params"] = diff_lists
    raw["_diff_count"] = diff_counts
    raw["_diff_groups"] = diff_groups

    reasons = []
    for _, row in raw.iterrows():
        row_reasons = []
        if row["_state"] != "finished":
            row_reasons.append(row["_state"] or "state_missing")
        if pd.isna(row["best_R1"]):
            row_reasons.append("missing best_R1")
        if pd.isna(row["epoch"]) or pd.isna(row["num_epoch"]):
            row_reasons.append("missing epoch/num_epoch")
        elif row["epoch"] < row["num_epoch"]:
            row_reasons.append(f"unfinished epoch {int(row['epoch'])}/{int(row['num_epoch'])}")
        reasons.append("; ".join(row_reasons))
    raw["_exclude_reason"] = reasons
    raw["_valid"] = raw["_exclude_reason"].eq("")
    return raw


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", "<br>") for cell in row) + " |")
    return "\n".join(lines)


def run_label(row: pd.Series) -> str:
    return str(row.get("Name") or row.get("ID") or row.get("_identity") or "")


def config_short(row: pd.Series, params: list[str] | None = None) -> str:
    if params is None:
        params = row["_diff_params"][:]
    if not params:
        return "anchor"
    pieces = []
    for param in params:
        pieces.append(f"{param}={value_key(row[f'cfg__{param}'])}")
    return ", ".join(pieces)


def stats_for(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.count() >= 2 else math.nan,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def conclusion_from_delta(delta: float, noise: float, n: int) -> tuple[str, str]:
    if n == 0 or not math.isfinite(delta):
        return "insufficient evidence", "insufficient"
    if abs(delta) <= noise:
        return "neutral/within noise", "weak" if n < 2 else "moderate"
    if delta > noise:
        return ("beneficial", "moderate" if n >= 2 else "weak")
    return ("harmful", "moderate" if n >= 2 else "weak")


def marginal_table(valid: pd.DataFrame, param: str, limit: int | None = None) -> list[list[Any]]:
    grouped = valid.groupby(f"cfg__{param}", dropna=False)["best_R1"]
    rows = []
    for value, scores in grouped:
        stat = stats_for(scores)
        rows.append([
            value_key(value),
            stat["count"],
            fmt_num(stat["mean"]),
            fmt_num(stat["max"]),
            fmt_num(stat["min"]),
            fmt_num(stat["std"]) if stat["count"] >= 2 else "",
        ])
    rows.sort(key=lambda r: (float(r[3]) if r[3] else -999, float(r[2]) if r[2] else -999), reverse=True)
    return rows[:limit] if limit else rows


def combo_summary(valid: pd.DataFrame, params: list[str], baseline_mean: float, noise: float) -> dict[str, Any]:
    cols = [f"cfg__{p}" for p in params]
    if any(col not in valid.columns for col in cols):
        return {"support": "insufficient", "best": "", "bad": "", "n_combos": 0, "n_runs": 0}
    group = valid.groupby(cols, dropna=False)
    rows = []
    for key, df in group:
        key_tuple = key if isinstance(key, tuple) else (key,)
        stat = stats_for(df["best_R1"])
        best_row = df.loc[df["best_R1"].idxmax()]
        rows.append(
            {
                "combo": ", ".join(f"{p}={value_key(v)}" for p, v in zip(params, key_tuple)),
                "count": stat["count"],
                "mean": stat["mean"],
                "max": stat["max"],
                "min": stat["min"],
                "best_run": run_label(best_row),
            }
        )
    if not rows:
        return {"support": "insufficient", "best": "", "bad": "", "n_combos": 0, "n_runs": 0}
    rows_by_best = sorted(rows, key=lambda item: item["max"], reverse=True)
    bad_rows = [r for r in sorted(rows, key=lambda item: item["mean"]) if r["mean"] < baseline_mean - noise]
    n_combos = len(rows)
    n_runs = sum(r["count"] for r in rows)
    repeated_combos = sum(1 for r in rows if r["count"] >= 2)
    varied = all(valid[col].nunique(dropna=False) >= 2 for col in cols)
    if not varied or n_combos < 2:
        support = "insufficient"
    elif n_combos >= 4 and repeated_combos >= 2 and n_runs >= 8:
        support = "strong"
    elif n_combos >= 3 and n_runs >= 5:
        support = "moderate"
    else:
        support = "weak"
    best = rows_by_best[0]
    return {
        "support": support,
        "best": f"{best['combo']} -> max {fmt_num(best['max'])}, mean {fmt_num(best['mean'])}, n={best['count']}, run={best['best_run']}",
        "bad": "; ".join(f"{r['combo']} mean {fmt_num(r['mean'])} n={r['count']}" for r in bad_rows[:3]) or "none clearly below baseline-noise",
        "n_combos": n_combos,
        "n_runs": n_runs,
    }


def capacity_score(row: pd.Series) -> float:
    score = 0.0
    for param in ["mixer_dim", "mixer_depth", "mixer_hidden_channel", "mixer_hidden_readout", "mixer_hidden_part", "mixer_hidden_rank"]:
        val = row[f"cfg__{param}"]
        default = CLI_META[param]["default"]
        if isinstance(val, (int, float)) and default:
            score += math.log2(float(val) / float(default)) if val > 0 else -3
    return score


def capacity_bucket(score: float) -> str:
    if score <= -1:
        return "small"
    if score >= 1:
        return "large"
    return "medium/default-ish"


def loss_profile(row: pd.Series) -> str:
    ret = bool(row["cfg__use_target_retrieval_loss"])
    rob = bool(row["cfg__use_target_robust_loss"])
    if ret and rob:
        return "both"
    if ret:
        return "retrieval_only"
    if rob:
        return "robust_only"
    return "neither"


def pool_profile(row: pd.Series) -> str:
    return (
        f"{value_key(row['cfg__pool_k_mode'])}/k={value_key(row['cfg__pool_k'])}/"
        f"top_m={value_key(row['cfg__top_m'])}/shared={value_key(row['cfg__use_shared_k'])}"
    )


def is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def cli_flag_for_value(param: str, value: Any) -> list[str]:
    if value is None:
        return []
    if CLI_META[param]["kind"] == "flag" or is_bool_like(value):
        return [f"--{param}"] if parse_bool(value, False) else []
    return [f"--{param} {value_key(value)}"]


def run_analysis() -> str:
    parser_defaults = load_parser_defaults()
    for param in PARAMS:
        CLI_META[param]["default"] = parser_defaults.get(param, CLI_META[param]["default"])

    data = read_exports(CSV_PATHS, parser_defaults)
    valid = data[data["_valid"]].copy()
    excluded = data[~data["_valid"]].copy()

    data["_capacity_score"] = data.apply(capacity_score, axis=1)
    data["_capacity_bucket"] = data["_capacity_score"].map(capacity_bucket)
    data["_loss_profile"] = data.apply(loss_profile, axis=1)
    data["_pool_profile"] = data.apply(pool_profile, axis=1)
    valid["_capacity_score"] = valid.apply(capacity_score, axis=1)
    valid["_capacity_bucket"] = valid["_capacity_score"].map(capacity_bucket)
    valid["_loss_profile"] = valid.apply(loss_profile, axis=1)
    valid["_pool_profile"] = valid.apply(pool_profile, axis=1)

    parser_default_values = {param: parser_defaults.get(param, CLI_META[param]["default"]) for param in PARAMS}
    default_runs = valid[valid["_diff_count"].eq(0)].copy()
    default_stats = stats_for(default_runs["best_R1"])

    valid["_config_key"] = [
        tuple((param, value_key(row[f"cfg__{param}"])) for param in PARAMS)
        for _, row in valid.iterrows()
    ]
    config_summary_rows = []
    for key, group in valid.groupby("_config_key"):
        stat = stats_for(group["best_R1"])
        config_summary_rows.append(
            {
                "key": key,
                "count": stat["count"],
                "mean": stat["mean"],
                "diff_count": int(group["_diff_count"].iloc[0]),
            }
        )
    config_summary_rows.sort(key=lambda item: (-item["count"], item["diff_count"], -item["mean"]))
    anchor_key = config_summary_rows[0]["key"] if config_summary_rows else None
    anchor_source = valid[valid["_config_key"].map(lambda key: key == anchor_key)].iloc[0] if anchor_key is not None else None
    anchor_values: dict[str, Any] = {
        param: (anchor_source[f"cfg__{param}"] if anchor_source is not None else parser_default_values[param])
        for param in PARAMS
    }

    def attach_anchor_diffs(frame: pd.DataFrame) -> pd.DataFrame:
        anchor_diff_lists = []
        anchor_diff_counts = []
        for _, row in frame.iterrows():
            diffs = []
            for param in PARAMS:
                if not values_equal(row[f"cfg__{param}"], anchor_values[param]):
                    diffs.append(param)
            anchor_diff_lists.append(diffs)
            anchor_diff_counts.append(len(diffs))
        frame["_anchor_diff_params"] = anchor_diff_lists
        frame["_anchor_diff_count"] = anchor_diff_counts
        return frame

    data = attach_anchor_diffs(data)
    valid = attach_anchor_diffs(valid)
    excluded = data[~data["_valid"]].copy()

    anchor_runs = valid[valid["_anchor_diff_count"].eq(0)].copy()
    anchor_stats = stats_for(anchor_runs["best_R1"])
    if anchor_stats["count"] > 0:
        baseline_mean = anchor_stats["mean"]
        baseline_best = anchor_stats["max"]
        baseline_label = "operational sweep anchor"
    else:
        baseline_mean = default_stats["mean"]
        baseline_best = default_stats["max"]
        baseline_label = "official parser default"

    repeated_config_stds = []
    for _, group in valid.groupby("_config_key"):
        if len(group) >= 2:
            s = group["best_R1"].std(ddof=1)
            if pd.notna(s):
                repeated_config_stds.append(float(s))
    if default_stats["count"] >= 2 and math.isfinite(default_stats["std"]):
        noise = float(default_stats["std"])
        noise_note = "baseline noise estimated from repeated exact official parser-default runs"
    elif anchor_stats["count"] >= 2 and math.isfinite(anchor_stats["std"]):
        noise = max(0.25, float(anchor_stats["std"]))
        noise_note = "official parser default absent; noise estimated from repeated operational-anchor runs with a conservative 0.25 floor"
    elif repeated_config_stds:
        noise = max(0.25, float(pd.Series(repeated_config_stds).median()))
        noise_note = "no repeated official/anchor baseline; conservative floor plus repeated-config median std"
    else:
        noise = 0.25
        noise_note = "no repeated official/anchor baseline; conservative fixed threshold"

    suspicious = pd.DataFrame()
    completed_runtime = data[(data["_state"].eq("finished")) & data["epoch"].ge(data["num_epoch"]) & data["_runtime_seconds"].notna()].copy()
    if not completed_runtime.empty:
        thresholds = completed_runtime.groupby("num_epoch")["_runtime_seconds"].median().mul(0.25).to_dict()
        suspicious = data[
            data["_runtime_seconds"].notna()
            & data.apply(lambda row: row["_runtime_seconds"] < thresholds.get(row["num_epoch"], completed_runtime["_runtime_seconds"].median() * 0.25), axis=1)
        ].copy()

    varied_all = []
    fixed_all = []
    varied_valid = []
    fixed_valid = []
    for param in PARAMS:
        nunique_all = data[f"cfg__{param}"].map(value_key).nunique(dropna=False)
        nunique_valid = valid[f"cfg__{param}"].map(value_key).nunique(dropna=False) if not valid.empty else 0
        (varied_all if nunique_all > 1 else fixed_all).append(param)
        (varied_valid if nunique_valid > 1 else fixed_valid).append(param)

    lines = []
    lines.append("# Target-Aware Hyperparameter Impact Analysis")
    lines.append("")
    lines.append("Primary metric: `best_R1`. Secondary metrics were not used to override `best_R1` conclusions.")
    lines.append("")
    lines.append("## 1. Data Inventory")
    lines.append("")
    source_rows = [[Path(p).name, shape[0], shape[1]] for p, shape in [(str(path), pd.read_csv(path).shape) for path in CSV_PATHS]]
    lines.append(md_table(["CSV", "Rows", "Columns"], source_rows))
    lines.append("")
    lines.append(f"- Combined/de-duplicated runs: **{len(data)}**. De-dup key: W&B `Name` when present, else `ID`.")
    lines.append(f"- Valid runs: **{len(valid)}**.")
    lines.append(f"- Excluded runs: **{len(excluded)}**.")
    lines.append(f"- Crashed runs: **{int(data['_state'].eq('crashed').sum())}**.")
    lines.append(f"- Running runs: **{int(data['_state'].eq('running').sum())}**.")
    lines.append(f"- Finished but unfinished epoch: **{int(data['_exclude_reason'].str.contains('unfinished', regex=False).sum())}**.")
    lines.append(f"- Missing `best_R1`: **{int(data['best_R1'].isna().sum())}**.")
    lines.append(f"- Suspiciously short runtime: **{len(suspicious)}** using <25% of median runtime for the same `num_epoch`.")
    lines.append("")
    if not excluded.empty:
        rows = []
        for _, row in excluded.sort_values(["_state", "Name"]).iterrows():
            rows.append([run_label(row), row.get("ID", ""), row.get("State", ""), fmt_num(row.get("best_R1")), f"{fmt_num(row.get('epoch'), 0)}/{fmt_num(row.get('num_epoch'), 0)}", row["_exclude_reason"]])
        lines.append("### Excluded Runs")
        lines.append(md_table(["Name", "ID", "State", "best_R1", "epoch/num_epoch", "Reason"], rows))
        lines.append("")
    if len(suspicious):
        rows = []
        for _, row in suspicious.sort_values("_runtime_seconds").iterrows():
            rows.append([run_label(row), row.get("State", ""), fmt_num(row.get("best_R1")), fmt_num(row.get("_runtime_seconds"), 0), row["_exclude_reason"] or "valid"])
        lines.append("### Suspiciously Short Runtime")
        lines.append(md_table(["Name", "State", "best_R1", "runtime_sec", "status"], rows))
        lines.append("")

    lines.append("### Detected CLI Parameters From `utils/options.py`")
    meta_rows = []
    for group, params in GROUPS.items():
        for param in params:
            meta = CLI_META[param]
            meta_rows.append([group, param, value_key(meta["default"]), meta["choices"], meta["aliases"], meta["validation"]])
    lines.append(md_table(["Group", "Canonical parameter", "Default", "Choices", "Aliases", "Validation"], meta_rows))
    lines.append("")
    lines.append(f"- Varied in combined CSV: {', '.join(varied_all) if varied_all else 'none'}.")
    lines.append(f"- Fixed in combined CSV: {', '.join(fixed_all) if fixed_all else 'none'}.")
    lines.append(f"- Varied among valid runs: {', '.join(varied_valid) if varied_valid else 'none'}.")
    lines.append(f"- Fixed among valid runs: {', '.join(fixed_valid) if fixed_valid else 'none'}.")
    lines.append("")

    lines.append("## 2. Default Baseline And Noise")
    lines.append("")
    best_default_row = default_runs.loc[default_runs["best_R1"].idxmax()] if not default_runs.empty else None
    official_rows = [[
        default_stats["count"],
        fmt_num(default_stats["mean"]),
        fmt_num(default_stats["std"]) if default_stats["count"] >= 2 else "",
        fmt_num(default_stats["min"]),
        fmt_num(default_stats["max"]),
        run_label(best_default_row) if best_default_row is not None else "",
    ]]
    lines.append("### Official Parser Default")
    lines.append(md_table(["count", "mean best_R1", "std", "min", "max", "best default run"], official_rows))
    lines.append("")
    anchor_diff_from_parser = [param for param in PARAMS if not values_equal(anchor_values[param], parser_default_values[param])]
    best_anchor_row = anchor_runs.loc[anchor_runs["best_R1"].idxmax()] if not anchor_runs.empty else None
    anchor_rows = [[
        anchor_stats["count"],
        fmt_num(anchor_stats["mean"]),
        fmt_num(anchor_stats["std"]) if anchor_stats["count"] >= 2 else "",
        fmt_num(anchor_stats["min"]),
        fmt_num(anchor_stats["max"]),
        run_label(best_anchor_row) if best_anchor_row is not None else "",
        ", ".join(f"{p}={value_key(anchor_values[p])}" for p in anchor_diff_from_parser) or "same as parser default",
    ]]
    lines.append("### Operational Sweep Anchor")
    lines.append(md_table(["count", "mean best_R1", "std", "min", "max", "best anchor run", "Anchor differs from parser default by"], anchor_rows))
    lines.append("")
    lines.append(f"Official parser-default valid runs are {'absent' if default_stats['count'] == 0 else 'present'}. Downstream deltas use the **{baseline_label}** because it is the measured baseline available in the CSV.")
    lines.append(f"Baseline noise threshold used for conclusions: **{fmt_num(noise)} best_R1** ({noise_note}).")
    lines.append("Deltas smaller than this threshold are treated as weak or inconclusive.")
    lines.append("")

    lines.append("## 3. Overall Ranking By `best_R1`")
    lines.append("")
    def ranking_rows(df: pd.DataFrame, n: int, ascending: bool = False) -> list[list[Any]]:
        rows = []
        for _, row in df.sort_values("best_R1", ascending=ascending).head(n).iterrows():
            rows.append([
                run_label(row),
                fmt_num(row["best_R1"]),
                fmt_num(row["best_R1"] - baseline_mean) if math.isfinite(baseline_mean) else "",
                fmt_num(row["best_R1"] - baseline_best) if math.isfinite(baseline_best) else "",
                row["_anchor_diff_count"],
                config_short(row, row["_anchor_diff_params"]),
            ])
        return rows

    lines.append("### Top 10 Valid Configurations")
    lines.append(md_table(["Run", "best_R1", "delta vs anchor mean", "delta vs best anchor", "#anchor diffs", "Anchor-relative diffs"], ranking_rows(valid, 10)))
    lines.append("")
    lines.append("### Bottom 10 Valid Configurations")
    lines.append(md_table(["Run", "best_R1", "delta vs anchor mean", "delta vs best anchor", "#anchor diffs", "Anchor-relative diffs"], ranking_rows(valid, 10, ascending=True)))
    lines.append("")
    best_overall = valid.loc[valid["best_R1"].idxmax()] if not valid.empty else None
    non_default = valid[valid["_anchor_diff_count"] > 0]
    best_non_default = non_default.loc[non_default["best_R1"].idxmax()] if not non_default.empty else None
    best_rows = []
    if best_overall is not None:
        best_rows.append(["overall", run_label(best_overall), fmt_num(best_overall["best_R1"]), config_short(best_overall, best_overall["_anchor_diff_params"])])
    if best_non_default is not None:
        best_rows.append(["non-anchor", run_label(best_non_default), fmt_num(best_non_default["best_R1"]), config_short(best_non_default, best_non_default["_anchor_diff_params"])])
    for group, params in GROUPS.items():
        subset = valid[valid["_anchor_diff_params"].map(lambda items, params=params: any(p in items for p in params))]
        if not subset.empty:
            row = subset.loc[subset["best_R1"].idxmax()]
            best_rows.append([group, run_label(row), fmt_num(row["best_R1"]), config_short(row, [p for p in row["_anchor_diff_params"] if p in params])])
        else:
            best_rows.append([group, "", "", "no non-anchor valid runs"])
    lines.append(md_table(["Category", "Run", "best_R1", "Relevant diffs"], best_rows))
    lines.append("")

    lines.append("## 4. One-Factor-At-A-Time Analysis")
    lines.append("")
    ofat_rows_by_group = defaultdict(list)
    lines.append("Strict parser-default OFAT is unavailable because there are no valid official parser-default runs. This OFAT table is relative to the operational sweep anchor.")
    lines.append("")
    ofat = valid[valid["_anchor_diff_count"].eq(1)]
    for group, params in GROUPS.items():
        for param in params:
            sub = ofat[ofat["_anchor_diff_params"].map(lambda items, p=param: items == [p])]
            if sub.empty:
                ofat_rows_by_group[group].append([param, "not tested OFAT", "", "", "", "insufficient evidence"])
                continue
            for value, df in sub.groupby(f"cfg__{param}", dropna=False):
                stat = stats_for(df["best_R1"])
                delta_mean = stat["max"] - baseline_mean
                delta_best = stat["max"] - baseline_best
                conclusion, confidence = conclusion_from_delta(delta_mean, noise, stat["count"])
                if stat["count"] == 1 and abs(delta_mean) <= max(noise * 2, 0.5):
                    confidence = "weak"
                ofat_rows_by_group[group].append([
                    param,
                    value_key(value),
                    stat["count"],
                    fmt_num(stat["max"]),
                    fmt_num(delta_mean),
                    f"{conclusion} ({confidence}); delta vs best anchor {fmt_num(delta_best)}",
                ])
    for group in GROUPS:
        lines.append(f"### {group}")
        lines.append(md_table(["Parameter", "Tested value", "n", "best_R1", "delta vs anchor mean", "Conclusion"], ofat_rows_by_group[group]))
        lines.append("")

    lines.append("## 5. Marginal Trend Analysis")
    lines.append("")
    lines.append("Marginal trends use all valid runs, including combo runs, so they are confounded when parameters move together.")
    for group, params in GROUPS.items():
        lines.append(f"### {group}")
        for param in params:
            if param not in varied_valid:
                lines.append(f"- `{param}`: fixed at `{value_key(valid[f'cfg__{param}'].iloc[0]) if not valid.empty else value_key(CLI_META[param]['default'])}` among valid runs; not enough evidence.")
            else:
                lines.append(f"**`{param}`**")
                lines.append(md_table(["value", "n", "mean", "max", "min", "std"], marginal_table(valid, param)))
        lines.append("")

    lines.append("## 6. Interaction Analysis")
    lines.append("")
    interaction_specs = [
        ("enrichment_space x extractor_mode", ["enrichment_space", "extractor_mode"]),
        ("pool_k_mode x pool_k", ["pool_k_mode", "pool_k"]),
        ("use_shared_k x pool_k x top_m", ["use_shared_k", "pool_k", "top_m"]),
        ("top_m x robust_hard_k", ["top_m", "robust_hard_k"]),
        ("residual_gate x enrich_gamma", ["residual_gate", "enrich_gamma"]),
        ("residual_gate x residual_gate_hidden_dim", ["residual_gate", "residual_gate_hidden_dim"]),
        ("recompute_level x recompute_interval", ["recompute_level", "recompute_interval"]),
        ("positive_ratio_max x pool_dist_threshold", ["positive_ratio_max", "pool_dist_threshold"]),
        ("pool_clusters x pool_dist_metric", ["pool_clusters", "pool_dist_metric"]),
        ("mixer_dim x mixer_depth", ["mixer_dim", "mixer_depth"]),
        ("mixer_depth x mixer_hidden_channel", ["mixer_depth", "mixer_hidden_channel"]),
        ("mixer_depth x mixer_hidden_readout", ["mixer_depth", "mixer_hidden_readout"]),
        ("mixer_hidden_part x mixer_hidden_rank", ["mixer_hidden_part", "mixer_hidden_rank"]),
        ("context_pooling x mixer_depth", ["context_pooling", "mixer_depth"]),
        ("use_target_retrieval_loss x lambda_ret", ["use_target_retrieval_loss", "lambda_ret"]),
        ("use_target_robust_loss x lambda_rob", ["use_target_robust_loss", "lambda_rob"]),
        ("lambda_rob x lambda_gain", ["lambda_rob", "lambda_gain"]),
        ("lambda_gain x gain_margin", ["lambda_gain", "gain_margin"]),
        ("extractor_mode/num_parts x mixer_hidden_part", ["extractor_mode", "num_parts", "mixer_hidden_part"]),
        ("top_m/pool_k x mixer_hidden_rank", ["top_m", "pool_k", "mixer_hidden_rank"]),
        ("robust_hard_k x lambda_rob/gain_margin", ["robust_hard_k", "lambda_rob", "gain_margin"]),
    ]
    interaction_rows = []
    for label, params in interaction_specs:
        summary = combo_summary(valid, params, baseline_mean, noise)
        interaction_rows.append([label, summary["support"], summary["n_runs"], summary["n_combos"], summary["best"], summary["bad"]])

    derived_specs = [
        ("small-capacity vs large-capacity mixer profile", ["_capacity_bucket"]),
        ("retrieval loss only vs robust loss only vs both", ["_loss_profile"]),
        ("target enrichment pool settings x mixer capacity", ["_pool_profile", "_capacity_bucket"]),
        ("residual gate settings x target-aware loss profile", ["cfg__residual_gate", "_loss_profile"]),
    ]
    for label, cols in derived_specs:
        group = valid.groupby(cols, dropna=False)
        rows = []
        for key, df in group:
            key_tuple = key if isinstance(key, tuple) else (key,)
            stat = stats_for(df["best_R1"])
            rows.append({"combo": ", ".join(map(str, key_tuple)), "count": stat["count"], "mean": stat["mean"], "max": stat["max"]})
        if rows:
            rows_by_best = sorted(rows, key=lambda x: x["max"], reverse=True)
            bad = [r for r in sorted(rows, key=lambda x: x["mean"]) if r["mean"] < baseline_mean - noise]
            support = "moderate" if len(rows) >= 3 and sum(r["count"] for r in rows) >= 5 else ("weak" if len(rows) >= 2 else "insufficient")
            interaction_rows.append([
                label,
                support,
                sum(r["count"] for r in rows),
                len(rows),
                f"{rows_by_best[0]['combo']} -> max {fmt_num(rows_by_best[0]['max'])}, mean {fmt_num(rows_by_best[0]['mean'])}, n={rows_by_best[0]['count']}",
                "; ".join(f"{r['combo']} mean {fmt_num(r['mean'])} n={r['count']}" for r in bad[:3]) or "none clearly below baseline-noise",
            ])
    lines.append(md_table(["Interaction", "Support", "n runs", "n combos", "Best observed", "Clearly bad combos"], interaction_rows))
    lines.append("")

    lines.append("## 7. Capacity And Stability Analysis")
    lines.append("")
    cap_rows = []
    for param in ["mixer_dim", "mixer_depth", "mixer_hidden_channel", "mixer_hidden_readout", "mixer_hidden_part", "mixer_hidden_rank", "top_m", "pool_k", "num_parts", "robust_hard_k", "residual_gate_hidden_dim"]:
        rows = marginal_table(valid, param)
        if len(rows) <= 1:
            cap_rows.append([param, "fixed", rows[0][0] if rows else value_key(CLI_META[param]["default"]), "not enough evidence"])
            continue
        numeric_pairs = []
        for row in rows:
            try:
                numeric_pairs.append((float(row[0]), float(row[2]), float(row[3]), int(row[1])))
            except ValueError:
                continue
        trend = "mixed/confounded"
        if len(numeric_pairs) >= 2:
            by_val = sorted(numeric_pairs)
            low, high = by_val[0], by_val[-1]
            if high[1] > low[1] + noise:
                trend = "larger mean improves"
            elif high[1] < low[1] - noise:
                trend = "larger mean hurts"
            else:
                trend = "within noise"
        cap_rows.append([param, "; ".join(f"{r[0]} n={r[1]} mean={r[2]} max={r[3]}" for r in rows), "", trend])
    lines.append(md_table(["Capacity parameter", "Observed values", "Fixed value", "Direction"], cap_rows))
    lines.append("")
    crash_rows = []
    for param in ["mixer_dim", "mixer_depth", "mixer_hidden_channel", "mixer_hidden_readout", "mixer_hidden_part", "mixer_hidden_rank", "top_m", "pool_k", "num_parts", "robust_hard_k", "residual_gate_hidden_dim"]:
        if param in data.columns:
            for value, df in data.groupby(f"cfg__{param}", dropna=False):
                crashes = int(df["_state"].eq("crashed").sum())
                if crashes:
                    crash_rows.append([param, value_key(value), crashes, len(df)])
    lines.append("Stability/crash regions:")
    lines.append(md_table(["Parameter", "Value", "Crashed runs", "All runs"], crash_rows) if crash_rows else "_No crashes tied cleanly to these capacity values._")
    lines.append("")

    lines.append("## 8. Freeze-Region Recommendation")
    lines.append("")
    freeze_rows = []
    for group, params in GROUPS.items():
        for param in params:
            parser_default = parser_defaults.get(param, CLI_META[param]["default"])
            anchor_default = anchor_values[param]
            if param not in varied_valid:
                if values_equal(anchor_default, parser_default):
                    freeze_rows.append([group, param, "GREEN / Freeze", value_key(anchor_default), "insufficient", "Fixed at parser default in valid data; not enough evidence to tune it."])
                else:
                    freeze_rows.append([group, param, "YELLOW / Keep Testing", value_key(anchor_default), "insufficient", "Fixed at a non-parser sweep-base value; keep it for continuity, but add a parser-default control before calling it better."])
                continue
            val_stats = []
            for value, df in valid.groupby(f"cfg__{param}", dropna=False):
                stat = stats_for(df["best_R1"])
                val_stats.append((value, stat))
            non_default_stats = [(value, stat) for value, stat in val_stats if not values_equal(value, anchor_default)]
            best_value, best_stat = max(val_stats, key=lambda item: item[1]["max"])
            best_non_default_stat = max(non_default_stats, key=lambda item: item[1]["max"]) if non_default_stats else None
            if best_non_default_stat is None:
                status = "GREEN / Freeze"
                rec = value_key(anchor_default)
                confidence = "insufficient"
                reason = "No non-anchor valid value."
            else:
                nd_value, nd_stat = best_non_default_stat
                delta = nd_stat["max"] - baseline_mean
                if delta > noise and nd_stat["count"] >= 2:
                    status = "YELLOW / Keep Testing" if values_equal(best_value, nd_value) else "YELLOW / Keep Testing"
                    rec = value_key(nd_value)
                    confidence = "moderate"
                    reason = f"Best non-anchor max {fmt_num(nd_stat['max'])}, delta {fmt_num(delta)}, n={nd_stat['count']}; confirm before hard freeze."
                elif delta > noise * 2:
                    status = "YELLOW / Keep Testing"
                    rec = value_key(nd_value)
                    confidence = "weak"
                    reason = f"Large single-run/non-repeated gain: max {fmt_num(nd_stat['max'])}, delta {fmt_num(delta)}, n={nd_stat['count']}."
                else:
                    status = "GREEN / Freeze"
                    rec = value_key(anchor_default)
                    confidence = "weak" if len(non_default_stats) <= 1 else "moderate"
                    reason = f"No non-anchor value clearly beats the measured anchor beyond noise; best non-anchor {value_key(nd_value)} max {fmt_num(nd_stat['max'])}, delta {fmt_num(delta)}."
                harmful_values = []
                for value, stat in non_default_stats:
                    if stat["mean"] < baseline_mean - noise and stat["count"] >= 1:
                        harmful_values.append(f"{value_key(value)} mean {fmt_num(stat['mean'])}")
                if harmful_values and not values_equal(best_value, anchor_default):
                    reason += " Avoid broad use of: " + "; ".join(harmful_values[:3]) + "."
            freeze_rows.append([group, param, status, rec, confidence, reason])
    lines.append(md_table(["Group", "Parameter", "Region", "Freeze/test value", "Confidence", "Evidence summary"], freeze_rows))
    lines.append("")

    lines.append("## 9. Main Conclusions")
    lines.append("")
    positive = []
    negative = []
    needs_more = []
    for param in varied_valid:
        rows = marginal_table(valid, param)
        if not rows:
            continue
        best = rows[0]
        anchor_default = value_key(anchor_values[param])
        anchor_row = next((r for r in rows if r[0] == anchor_default), None)
        best_delta = float(best[3]) - baseline_mean if best[3] else math.nan
        if best[0] != anchor_default and math.isfinite(best_delta) and best_delta > noise:
            positive.append(f"`{param}={best[0]}` max {best[3]} (delta {fmt_num(best_delta)})")
        elif anchor_row and float(anchor_row[2]) - float(best[2]) > noise:
            negative.append(f"`{param}` non-anchor values trail anchor")
        else:
            needs_more.append(f"`{param}`")
    lines.append(f"- Strongest positive trends: {', '.join(positive[:8]) if positive else 'none clearly beyond noise with enough support'}.")
    lines.append(f"- Strongest negative/anchor-favoring trends: {', '.join(negative[:8]) if negative else 'mostly within noise or confounded'}.")
    lines.append(f"- Parameters needing more evidence: {', '.join(needs_more[:20]) if needs_more else 'none'}.")
    if best_overall is not None:
        lines.append(f"- Best current configuration: `{run_label(best_overall)}` with best_R1 **{fmt_num(best_overall['best_R1'])}**; anchor-relative diffs: {config_short(best_overall, best_overall['_anchor_diff_params'])}.")
    cost_candidates = valid.copy()
    # Prefer small/default-ish capacity within noise of best observed.
    if best_overall is not None:
        cost_candidates = cost_candidates[cost_candidates["best_R1"] >= best_overall["best_R1"] - noise]
    if not cost_candidates.empty:
        cost_row = cost_candidates.sort_values(["_capacity_score", "_runtime_seconds"], ascending=[True, True], na_position="last").iloc[0]
        lines.append(f"- Best cost-effective configuration: `{run_label(cost_row)}` best_R1 **{fmt_num(cost_row['best_R1'])}**, capacity `{cost_row['_capacity_bucket']}`; anchor-relative diffs: {config_short(cost_row, cost_row['_anchor_diff_params'])}.")
    lines.append(f"- Risky/unstable regions: {int(data['_state'].eq('crashed').sum())} crashed runs; exclude unfinished and crashed runs from conclusions.")
    lines.append("")

    lines.append("## 10. Recommended Next Experiments")
    lines.append("")
    lines.append("These minimize run count by repeating only the promising regions and adding small local probes. The repo has no local `add_run` definition, so commands are written as `add_run \"<train.py CLI flags>\"` queue entries.")
    top_cmd_rows = []
    command_flags = []

    anchor_base_params = [param for param in PARAMS if not values_equal(anchor_values[param], parser_default_values[param])]

    def flags_from_values(values: dict[str, Any], changed_params: list[str]) -> str:
        ordered_params = []
        for param in anchor_base_params + changed_params:
            if param not in ordered_params:
                ordered_params.append(param)
        parts: list[str] = []
        for param in ordered_params:
            parts.extend(cli_flag_for_value(param, values[param]))
        if values.get("pnp_text_only") is True or value_key(values.get("pnp_text_only")) == "True":
            if "--freeze_host" not in parts:
                parts.append("--freeze_host")
            if "--no_use_host_loss" not in parts:
                parts.append("--no_use_host_loss")
        return " ".join(parts)

    if best_non_default is not None:
        best_values = {param: best_non_default[f"cfg__{param}"] for param in PARAMS}
        base = flags_from_values(best_values, best_non_default["_anchor_diff_params"])
        command_flags.append(("confirm_best_nondefault_a", base))
        command_flags.append(("confirm_best_nondefault_b", base))
    def probe_flags(overrides: dict[str, Any]) -> str:
        values = dict(anchor_values)
        values.update(overrides)
        return flags_from_values(values, list(overrides.keys()))

    # Local probes around common varied parameters if present.
    probe_defs = [
        ("probe_topm16", probe_flags({"top_m": 16})),
        ("probe_topm64_repeat", probe_flags({"top_m": 64})),
        ("probe_static_gamma03", probe_flags({"residual_gate": "static", "enrich_gamma": 0.3})),
        ("probe_static_gamma05", probe_flags({"residual_gate": "static", "enrich_gamma": 0.5})),
        ("probe_late_attention_depth2", probe_flags({"context_pooling": "late_attention", "mixer_depth": 2})),
        ("probe_hybrid_attention_depth2", probe_flags({"context_pooling": "hybrid_attention", "mixer_depth": 2})),
        ("probe_robust_loss_low", probe_flags({"use_target_robust_loss": True, "lambda_rob": 0.05, "gain_margin": 0.01})),
        ("probe_ret_robust_balanced", probe_flags({"use_target_robust_loss": True, "lambda_rob": 0.1, "lambda_gain": 1.0, "gain_margin": 0.01})),
    ]
    existing_cmds = {flags for _, flags in command_flags}
    for name, flags in probe_defs:
        if flags not in existing_cmds:
            command_flags.append((name, flags))
    for name, flags in command_flags[:8]:
        top_cmd_rows.append([name, f'add_run "{flags} --wandb_run_name {name}"'])
    lines.append(md_table(["Purpose", "Command"], top_cmd_rows))
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="print report to stdout")
    args = parser.parse_args()
    report = run_analysis()
    if args.print:
        print(report)
    else:
        print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
