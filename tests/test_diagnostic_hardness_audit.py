import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.audit import build_hardness_audit_with_ci, build_tight_hardness_summary_with_ci
from diagnostic.metrics import paired_hardness_gaps, query_negative_score_scale, retriever_hardness_stats


def make_stats(scores, is_positive):
    return retriever_hardness_stats(
        np.asarray(scores, dtype=float),
        np.asarray(is_positive, dtype=bool),
    )


class DiagnosticHardnessStatisticTest(unittest.TestCase):
    def test_gallery_stats_and_gap_signs(self):
        cue_metrics = {
            "a_dense": make_stats([0.60, 0.30, 0.55, 0.20], [True, True, False, False]),
            "b_dense": make_stats([0.60, 0.30, 0.40, 0.20], [True, True, False, False]),
        }
        hm_metrics = {
            "hm_a": make_stats([0.60, 0.30, 0.45, 0.10], [True, True, False, False]),
            "hm_b": make_stats([0.60, 0.30, 0.50, 0.10], [True, True, False, False]),
        }

        self.assertAlmostEqual(cue_metrics["a_dense"]["best_positive_score"], 0.60)
        self.assertAlmostEqual(cue_metrics["a_dense"]["max_negative_score"], 0.55)
        self.assertAlmostEqual(cue_metrics["a_dense"]["positive_negative_margin"], 0.05)

        gaps = paired_hardness_gaps(
            cue_metrics,
            hm_metrics,
            safe_scale=0.20,
            tight_hardness_z_tolerance=0.60,
        )
        self.assertAlmostEqual(gaps["a_signed_max_negative_gap"], 0.10)
        self.assertAlmostEqual(gaps["a_signed_margin_gap"], -0.10)
        self.assertAlmostEqual(gaps["b_signed_max_negative_gap"], -0.10)
        self.assertAlmostEqual(gaps["b_signed_margin_gap"], 0.10)
        self.assertAlmostEqual(gaps["mean_signed_max_negative_gap"], 0.0)
        self.assertAlmostEqual(gaps["mean_signed_margin_gap"], 0.0)
        self.assertAlmostEqual(gaps["mean_abs_max_negative_gap"], 0.10)
        self.assertAlmostEqual(gaps["a_normalized_max_negative_gap"], 0.50)
        self.assertAlmostEqual(gaps["b_normalized_max_negative_gap"], -0.50)
        self.assertAlmostEqual(gaps["max_abs_normalized_max_negative_gap"], 0.50)
        self.assertTrue(gaps["tight_hardness_match"])

        loose_gaps = paired_hardness_gaps(
            cue_metrics,
            hm_metrics,
            safe_scale=0.20,
            tight_hardness_z_tolerance=0.40,
        )
        self.assertFalse(loose_gaps["tight_hardness_match"])

    def test_query_negative_score_scale_excludes_positives(self):
        full_scores = np.asarray([0.60, 0.30, 0.55, 0.20, 0.45, 0.10], dtype=float)
        full_pids = np.asarray([1, 1, 2, 3, 4, 5], dtype=np.int64)
        scale, safe_scale = query_negative_score_scale(full_scores, full_pids, query_pid=1)

        self.assertAlmostEqual(scale, float(np.std([0.55, 0.20, 0.45, 0.10], ddof=0)))
        self.assertAlmostEqual(safe_scale, max(scale, 1e-12))

    def test_zero_score_scale_uses_safe_floor(self):
        full_scores = np.asarray([0.60, 0.60, 0.10, 0.10, 0.10], dtype=float)
        full_pids = np.asarray([1, 1, 2, 3, 4], dtype=np.int64)
        scale, safe_scale = query_negative_score_scale(full_scores, full_pids, query_pid=1)

        self.assertAlmostEqual(scale, 0.0)
        self.assertEqual(safe_scale, 1e-12)


def make_tight_rows():
    return pd.DataFrame(
        [
            {
                "dataset": "RSTPReid",
                "retriever_name": "itself",
                "case_id": "case_a",
                "query_id": 10,
                "trial_id": 0,
                "r1_flip": 1.0,
                "hm_r1_flip": 0.0,
                "delta_r1_flip": 1.0,
                "tight_hardness_match": True,
                "mean_signed_max_negative_gap": 0.10,
                "mean_signed_margin_gap": -0.10,
                "mean_abs_max_negative_gap": 0.10,
            },
            {
                "dataset": "RSTPReid",
                "retriever_name": "itself",
                "case_id": "case_a",
                "query_id": 10,
                "trial_id": 1,
                "r1_flip": 0.0,
                "hm_r1_flip": 0.0,
                "delta_r1_flip": 0.0,
                "tight_hardness_match": True,
                "mean_signed_max_negative_gap": 0.20,
                "mean_signed_margin_gap": -0.20,
                "mean_abs_max_negative_gap": 0.20,
            },
            {
                "dataset": "RSTPReid",
                "retriever_name": "itself",
                "case_id": "case_b",
                "query_id": 10,
                "trial_id": 0,
                "r1_flip": 1.0,
                "hm_r1_flip": 1.0,
                "delta_r1_flip": 0.0,
                "tight_hardness_match": True,
                "mean_signed_max_negative_gap": -0.10,
                "mean_signed_margin_gap": 0.10,
                "mean_abs_max_negative_gap": 0.10,
            },
            {
                "dataset": "RSTPReid",
                "retriever_name": "itself",
                "case_id": "case_c",
                "query_id": 20,
                "trial_id": 0,
                "r1_flip": 0.0,
                "hm_r1_flip": 1.0,
                "delta_r1_flip": -1.0,
                "tight_hardness_match": False,
                "mean_signed_max_negative_gap": 3.0,
                "mean_signed_margin_gap": -3.0,
                "mean_abs_max_negative_gap": 3.0,
            },
        ]
    )


class DiagnosticTightHardnessBootstrapTest(unittest.TestCase):
    def test_tight_summary_filters_rows_and_clusters_by_unique_query(self):
        df = make_tight_rows()
        summary = build_tight_hardness_summary_with_ci(
            df,
            retriever_name="itself",
            tight_hardness_z_tolerance=0.10,
            bootstrap_iters=25,
            bootstrap_seed=5,
        )

        r1_row = summary.loc[summary["metric"] == "r1_flip"].iloc[0]
        self.assertAlmostEqual(r1_row["mean"], 2.0 / 3.0)
        self.assertEqual(int(r1_row["cluster_count"]), 1)
        self.assertEqual(int(r1_row["trial_count"]), 3)
        self.assertEqual(int(r1_row["tight_trial_count"]), 3)
        self.assertEqual(int(r1_row["tight_unique_query_count"]), 1)
        self.assertEqual(int(r1_row["tight_case_query_count"]), 2)
        self.assertAlmostEqual(r1_row["tight_trial_rate"], 3.0 / 4.0)

    def test_tight_summary_empty_result_has_expected_rows(self):
        df = make_tight_rows()
        df["tight_hardness_match"] = False
        summary = build_tight_hardness_summary_with_ci(
            df,
            retriever_name="trained_clip",
            tight_hardness_z_tolerance=0.10,
            bootstrap_iters=25,
            bootstrap_seed=5,
        )

        self.assertEqual(summary["metric"].tolist(), ["r1_flip", "hm_r1_flip", "delta_r1_flip"])
        self.assertTrue(summary["mean"].map(math.isnan).all())
        self.assertEqual(summary["retriever"].unique().tolist(), ["trained_clip"])
        self.assertEqual(summary["tight_trial_count"].unique().tolist(), [0])
        self.assertEqual(summary["tight_unique_query_count"].unique().tolist(), [0])
        self.assertEqual(summary["tight_case_query_count"].unique().tolist(), [0])
        self.assertEqual(summary["tight_trial_rate"].unique().tolist(), [0.0])

    def test_hardness_audit_uses_all_valid_trials(self):
        df = make_tight_rows()
        audit = build_hardness_audit_with_ci(
            df,
            bootstrap_iters=25,
            bootstrap_seed=7,
        )

        signed_row = audit.loc[audit["metric"] == "mean_signed_max_negative_gap"].iloc[0]
        self.assertAlmostEqual(signed_row["mean"], df["mean_signed_max_negative_gap"].mean())
        self.assertEqual(int(signed_row["cluster_count"]), 2)
        self.assertEqual(int(signed_row["trial_count"]), 4)

    def test_summary_accepts_both_retriever_paths_without_aliases(self):
        df = make_tight_rows()
        for retriever_name in ("trained_clip", "itself"):
            summary = build_tight_hardness_summary_with_ci(
                df,
                retriever_name=retriever_name,
                tight_hardness_z_tolerance=0.10,
                bootstrap_iters=10,
                bootstrap_seed=3,
            )
            self.assertEqual(summary["retriever"].unique().tolist(), [retriever_name])
            self.assertIn("tight_hardness_z_tolerance", summary.columns)
            self.assertNotIn(f"{retriever_name}_tight_hardness_z_tolerance", summary.columns)


if __name__ == "__main__":
    unittest.main()
