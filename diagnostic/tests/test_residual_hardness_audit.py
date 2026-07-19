import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.audit import (
    build_hardness_audit_with_ci,
    build_residual_hardness_adjusted_summary_with_ci,
    build_tight_hardness_summary_with_ci,
    fit_residual_hardness_adjustment,
)
from diagnostic.bootstrap import (
    cluster_bootstrap_callback,
    cluster_index_groups,
    concatenate_cluster_sample_indices,
)
from diagnostic.constants import (
    HARDNESS_AUDIT_METRICS,
    OUTPUT_FILES,
    PAIRED_DELTA_COLUMNS,
    RESIDUAL_HARDNESS_ADJUSTED_METRICS,
    RESIDUAL_HARDNESS_ADJUSTED_SUMMARY_COLUMNS,
    UNIQUE_QUERY_CLUSTER_COLS,
)
from diagnostic.metrics import paired_hardness_gaps
from diagnostic.outputs import write_run_outputs


def paired_row(
    *,
    retriever_name="itself",
    case_id="case_a",
    query_id=10,
    trial_id=0,
    gap=0.0,
    delta=0.0,
):
    return {
        "dataset": "RSTPReid",
        "retriever_name": retriever_name,
        "cue_scorer": "off_the_shelf_clip",
        "case_id": case_id,
        "query_id": query_id,
        "pid": query_id,
        "trial_id": trial_id,
        "cue_a": "red",
        "cue_b": "blue",
        "r1_flip": 0.0,
        "hm_r1_flip": 0.0,
        "delta_r1_flip": delta,
        "rank_shift": 0.0,
        "hm_rank_shift": 0.0,
        "delta_rank_shift": 0.0,
        "ap_delta": 0.0,
        "hm_ap_delta": 0.0,
        "delta_ap_delta": 0.0,
        "cue_shift": 0.0,
        "negative_score_scale": 1.0,
        "a_signed_max_negative_gap": gap,
        "b_signed_max_negative_gap": gap,
        "a_signed_margin_gap": -gap,
        "b_signed_margin_gap": -gap,
        "mean_signed_max_negative_gap": gap,
        "mean_signed_margin_gap": -gap,
        "mean_abs_max_negative_gap": abs(gap),
        "a_normalized_max_negative_gap": gap,
        "b_normalized_max_negative_gap": gap,
        "mean_normalized_max_negative_gap": gap,
        "max_abs_normalized_max_negative_gap": abs(gap),
        "tight_hardness_match": abs(gap) <= 0.10,
        "tight_hardness_z_tolerance": 0.10,
        "score_mode": "global",
        "lambda_global": 1.0,
        "seed": 1,
    }


def regression_frame(gaps, *, intercept=0.20, slope=0.50, retriever_name="itself"):
    rows = []
    for index, gap in enumerate(gaps):
        rows.append(
            paired_row(
                retriever_name=retriever_name,
                case_id=f"case_{index % 2}",
                query_id=100 + index,
                trial_id=0,
                gap=float(gap),
                delta=float(intercept + slope * gap),
            )
        )
    return pd.DataFrame(rows)


class ResidualHardnessAuditTest(unittest.TestCase):
    def test_trial_field_mean_normalized_gap(self):
        cue_metrics = {
            "a_dense": {"max_negative_score": 2.0, "positive_negative_margin": -1.0},
            "b_dense": {"max_negative_score": 1.0, "positive_negative_margin": 0.0},
        }
        hm_metrics = {
            "hm_a": {"max_negative_score": 0.0, "positive_negative_margin": 1.0},
            "hm_b": {"max_negative_score": 2.0, "positive_negative_margin": -1.0},
        }
        gaps = paired_hardness_gaps(
            cue_metrics,
            hm_metrics,
            safe_scale=10.0,
            tight_hardness_z_tolerance=1.0,
        )

        self.assertAlmostEqual(gaps["a_normalized_max_negative_gap"], 0.2)
        self.assertAlmostEqual(gaps["b_normalized_max_negative_gap"], -0.1)
        self.assertAlmostEqual(gaps["mean_normalized_max_negative_gap"], 0.05)

    def test_exact_regression(self):
        df = regression_frame([-2.0, -1.0, 0.0, 1.0, 2.0])
        stats = fit_residual_hardness_adjustment(df)

        self.assertAlmostEqual(stats["adjusted_delta_r1_flip_at_zero_gap"], 0.20)
        self.assertAlmostEqual(stats["hardness_slope_delta_per_z"], 0.50)

    def test_raw_versus_adjusted_nonzero_mean_gap(self):
        df = regression_frame([1.0, 2.0, 3.0])
        stats = fit_residual_hardness_adjustment(df)

        self.assertNotAlmostEqual(
            stats["raw_delta_r1_flip"],
            stats["adjusted_delta_r1_flip_at_zero_gap"],
        )
        self.assertAlmostEqual(
            stats["adjustment_change"],
            stats["adjusted_delta_r1_flip_at_zero_gap"] - stats["raw_delta_r1_flip"],
        )

    def test_zero_covariate_degenerates_to_raw_mean(self):
        df = pd.DataFrame(
            [
                paired_row(gap=0.0, delta=0.0, query_id=1),
                paired_row(gap=0.0, delta=1.0, query_id=2),
            ]
        )
        stats = fit_residual_hardness_adjustment(df)

        self.assertAlmostEqual(stats["raw_delta_r1_flip"], 0.5)
        self.assertAlmostEqual(stats["adjusted_delta_r1_flip_at_zero_gap"], 0.5)
        self.assertEqual(stats["hardness_slope_delta_per_z"], 0.0)
        self.assertEqual(stats["adjustment_change"], 0.0)

    def test_cluster_bootstrap_samples_queries_with_duplicate_cluster_copies(self):
        df = pd.DataFrame(
            [
                paired_row(query_id=10, trial_id=0, gap=0.0, delta=0.0),
                paired_row(query_id=10, trial_id=1, gap=0.0, delta=0.0),
                paired_row(query_id=20, trial_id=0, gap=1.0, delta=1.0),
                paired_row(query_id=20, trial_id=1, gap=1.0, delta=1.0),
                paired_row(query_id=20, trial_id=2, gap=1.0, delta=1.0),
            ]
        )
        groups = cluster_index_groups(df, UNIQUE_QUERY_CLUSTER_COLS)
        sample_indices = concatenate_cluster_sample_indices(groups, [0, 0, 1])
        sampled_queries = df.iloc[sample_indices]["query_id"].tolist()

        self.assertEqual(sampled_queries, [10, 10, 10, 10, 20, 20, 20])
        first = cluster_bootstrap_callback(
            df,
            UNIQUE_QUERY_CLUSTER_COLS,
            iters=5,
            seed=99,
            statistic_fn=lambda sample: {"row_count": int(len(sample))},
        )
        second = cluster_bootstrap_callback(
            df,
            UNIQUE_QUERY_CLUSTER_COLS,
            iters=5,
            seed=99,
            statistic_fn=lambda sample: {"row_count": int(len(sample))},
        )
        self.assertEqual(first[1], second[1])

    def test_empty_and_invalid_adjusted_summaries_have_expected_columns(self):
        empty_df = pd.DataFrame(columns=PAIRED_DELTA_COLUMNS)
        empty_summary = build_residual_hardness_adjusted_summary_with_ci(
            empty_df,
            retriever_name="itself",
            bootstrap_iters=10,
            bootstrap_seed=1,
        )
        self.assertEqual(empty_summary.columns.tolist(), RESIDUAL_HARDNESS_ADJUSTED_SUMMARY_COLUMNS)
        self.assertEqual(empty_summary["metric"].tolist(), list(RESIDUAL_HARDNESS_ADJUSTED_METRICS))
        self.assertTrue(empty_summary["mean"].map(math.isnan).all())
        self.assertEqual(empty_summary["valid_bootstrap_iters"].unique().tolist(), [0])

        invalid_df = pd.DataFrame(
            [
                paired_row(query_id=1, gap=float("nan"), delta=0.0),
                paired_row(query_id=2, gap=0.0, delta=float("nan")),
            ]
        )
        invalid_summary = build_residual_hardness_adjusted_summary_with_ci(
            invalid_df,
            retriever_name="itself",
            bootstrap_iters=10,
            bootstrap_seed=1,
        )
        self.assertTrue(invalid_summary["mean"].map(math.isnan).all())
        self.assertEqual(invalid_summary["valid_bootstrap_iters"].unique().tolist(), [0])

    def test_hardness_audit_has_normalized_metrics(self):
        df = regression_frame([-0.1, 0.0, 0.2])
        audit = build_hardness_audit_with_ci(
            df,
            retriever_name="itself",
            bootstrap_iters=10,
            bootstrap_seed=1,
        )

        self.assertIn("retriever", audit.columns)
        self.assertTrue(set(HARDNESS_AUDIT_METRICS).issubset(set(audit["metric"])))
        signed_row = audit.loc[audit["metric"] == "mean_signed_normalized_max_negative_gap"].iloc[0]
        self.assertAlmostEqual(signed_row["mean"], df["mean_normalized_max_negative_gap"].mean())

    def test_retriever_paths_share_adjusted_summary(self):
        for retriever_name in ("trained_clip", "itself"):
            df = regression_frame([-1.0, 0.0, 1.0], retriever_name=retriever_name)
            summary = build_residual_hardness_adjusted_summary_with_ci(
                df,
                retriever_name=retriever_name,
                bootstrap_iters=10,
                bootstrap_seed=1,
            )
            self.assertEqual(summary["retriever"].unique().tolist(), [retriever_name])
            self.assertEqual(summary["bootstrap_unit"].unique().tolist(), ["unique_query"])
            self.assertEqual(summary["model"].unique().tolist(), ["ols_paired_delta_on_normalized_hardness"])

    def test_output_writer_emits_additive_residual_summary_file(self):
        df = regression_frame([-1.0, 0.0, 1.0])
        hardness_audit = build_hardness_audit_with_ci(
            df,
            retriever_name="itself",
            bootstrap_iters=10,
            bootstrap_seed=1,
        )
        tight_summary = build_tight_hardness_summary_with_ci(
            df,
            retriever_name="itself",
            tight_hardness_z_tolerance=0.10,
            bootstrap_iters=10,
            bootstrap_seed=1,
        )
        adjusted_summary = build_residual_hardness_adjusted_summary_with_ci(
            df,
            retriever_name="itself",
            bootstrap_iters=10,
            bootstrap_seed=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_run_outputs(
                output_dir,
                selected_queries=[],
                constructibility_rows=[],
                validity_counts=[],
                per_gallery_rows=[],
                paired_cue_rows=[],
                paired_hm_rows=[],
                paired_delta_rows=df.to_dict("records"),
                summary_by_case=pd.DataFrame(),
                summary_overall=pd.DataFrame(),
                summary_ci=pd.DataFrame(),
                summary_ci_by_unit={},
                hardness_audit_ci=hardness_audit,
                tight_hardness_summary_ci=tight_summary,
                residual_hardness_adjusted_summary_ci=adjusted_summary,
                skipped_rows=[],
                gallery_rows=[],
                save_galleries=False,
            )

            self.assertTrue((output_dir / OUTPUT_FILES["paired_delta"]).exists())
            self.assertTrue((output_dir / OUTPUT_FILES["hardness_audit_ci"]).exists())
            self.assertTrue((output_dir / OUTPUT_FILES["tight_hardness_summary_ci"]).exists())
            residual_path = output_dir / OUTPUT_FILES["residual_hardness_adjusted_summary_ci"]
            self.assertTrue(residual_path.exists())
            written = pd.read_csv(residual_path)
            self.assertEqual(written["metric"].tolist(), list(RESIDUAL_HARDNESS_ADJUSTED_METRICS))


if __name__ == "__main__":
    unittest.main()
