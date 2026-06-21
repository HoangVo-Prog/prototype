import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.bootstrap import bootstrap_count_summary, cluster_bootstrap_ci, cluster_row_counts
from diagnostic.constants import CASE_QUERY_CLUSTER_COLS, SUMMARY_CI_METRICS, UNIQUE_QUERY_CLUSTER_COLS


def make_bootstrap_rows():
    rows = []
    specs = [
        ("case_a", 10, [0.0, 1.0]),
        ("case_b", 10, [1.0, 1.0]),
        ("case_c", 20, [0.0, 0.0]),
    ]
    for case_id, query_id, r1_values in specs:
        for trial_id, r1_flip in enumerate(r1_values):
            hm_r1_flip = 1.0 - r1_flip
            rank_shift = float(query_id + trial_id)
            hm_rank_shift = float(query_id - trial_id)
            rows.append(
                {
                    "dataset": "RSTPReid",
                    "retriever_name": "itself",
                    "case_id": case_id,
                    "query_id": query_id,
                    "trial_id": trial_id,
                    "r1_flip": r1_flip,
                    "rank_shift": rank_shift,
                    "hm_r1_flip": hm_r1_flip,
                    "hm_rank_shift": hm_rank_shift,
                    "delta_r1_flip": r1_flip - hm_r1_flip,
                    "delta_rank_shift": rank_shift - hm_rank_shift,
                    "cue_shift": 0.1 * (trial_id + 1),
                }
            )
    return pd.DataFrame(rows)


class DiagnosticBootstrapUnitTest(unittest.TestCase):
    def test_cluster_counts(self):
        df = make_bootstrap_rows()
        case_counts = bootstrap_count_summary(df, CASE_QUERY_CLUSTER_COLS)
        unique_counts = bootstrap_count_summary(df, UNIQUE_QUERY_CLUSTER_COLS)
        self.assertEqual(case_counts["cluster_count"], 3)
        self.assertEqual(unique_counts["cluster_count"], 2)
        self.assertEqual(case_counts["trial_count"], 6)
        self.assertEqual(unique_counts["trial_count"], 6)
        self.assertEqual(case_counts["unique_query_count"], 2)
        self.assertEqual(case_counts["case_query_count"], 3)

    def test_joint_unique_query_grouping_keeps_cases_together(self):
        df = make_bootstrap_rows()
        grouped = df.groupby(UNIQUE_QUERY_CLUSTER_COLS, dropna=False)
        query_10_group = grouped.get_group(("RSTPReid", "itself", 10))
        self.assertEqual(set(query_10_group["case_id"]), {"case_a", "case_b"})
        self.assertEqual(len(query_10_group), 4)

        counts = cluster_row_counts(df, UNIQUE_QUERY_CLUSTER_COLS)
        query_10_count = counts.loc[counts["query_id"] == 10, "row_count"].iloc[0]
        self.assertEqual(int(query_10_count), 4)

    def test_variable_sized_unique_query_clusters(self):
        rows = []
        for case_id, query_id, num_cases in [("a", 1, 1), ("b", 2, 3), ("c", 2, 3), ("d", 2, 3)]:
            for trial_id in range(2):
                rows.append(
                    {
                        "dataset": "RSTPReid",
                        "retriever_name": "itself",
                        "case_id": case_id,
                        "query_id": query_id,
                        "trial_id": trial_id,
                        "r1_flip": float(trial_id),
                        "rank_shift": float(num_cases),
                        "hm_r1_flip": 0.0,
                        "hm_rank_shift": 0.0,
                        "delta_r1_flip": float(trial_id),
                        "delta_rank_shift": float(num_cases),
                        "cue_shift": 0.2,
                    }
                )
        df = pd.DataFrame(rows)
        counts = cluster_row_counts(df, UNIQUE_QUERY_CLUSTER_COLS).sort_values("query_id")
        self.assertEqual(counts["row_count"].tolist(), [2, 6])
        result = cluster_bootstrap_ci(
            df,
            SUMMARY_CI_METRICS,
            UNIQUE_QUERY_CLUSTER_COLS,
            iters=25,
            seed=7,
            bootstrap_unit="unique_query",
        )
        self.assertEqual(int(result["cluster_count"].iloc[0]), 2)
        self.assertEqual(int(result["trial_count"].iloc[0]), 8)

    def test_point_estimates_identical_between_units(self):
        df = make_bootstrap_rows()
        case_result = cluster_bootstrap_ci(
            df,
            SUMMARY_CI_METRICS,
            CASE_QUERY_CLUSTER_COLS,
            iters=50,
            seed=11,
            bootstrap_unit="case_query",
        )
        unique_result = cluster_bootstrap_ci(
            df,
            SUMMARY_CI_METRICS,
            UNIQUE_QUERY_CLUSTER_COLS,
            iters=50,
            seed=11,
            bootstrap_unit="unique_query",
        )
        merged = case_result[["metric", "mean"]].merge(
            unique_result[["metric", "mean"]],
            on="metric",
            suffixes=("_case", "_unique"),
        )
        for _, row in merged.iterrows():
            self.assertAlmostEqual(row["mean_case"], row["mean_unique"])

    def test_reproducibility(self):
        df = make_bootstrap_rows()
        first = cluster_bootstrap_ci(
            df,
            SUMMARY_CI_METRICS,
            UNIQUE_QUERY_CLUSTER_COLS,
            iters=50,
            seed=123,
            bootstrap_unit="unique_query",
        )
        second = cluster_bootstrap_ci(
            df,
            SUMMARY_CI_METRICS,
            UNIQUE_QUERY_CLUSTER_COLS,
            iters=50,
            seed=123,
            bootstrap_unit="unique_query",
        )
        pd.testing.assert_frame_equal(first, second)

    def test_paired_delta_uses_delta_observations(self):
        df = make_bootstrap_rows()
        result = cluster_bootstrap_ci(
            df,
            ["delta_r1_flip", "delta_rank_shift"],
            UNIQUE_QUERY_CLUSTER_COLS,
            iters=50,
            seed=3,
            bootstrap_unit="unique_query",
        )
        delta_r1 = result[result["metric"] == "delta_r1_flip"].iloc[0]
        delta_rank = result[result["metric"] == "delta_rank_shift"].iloc[0]
        self.assertAlmostEqual(delta_r1["mean"], df["delta_r1_flip"].mean())
        self.assertAlmostEqual(delta_rank["mean"], df["delta_rank_shift"].mean())

    def test_case_query_unit_preserves_existing_cluster_key(self):
        df = make_bootstrap_rows()
        result = cluster_bootstrap_ci(
            df,
            ["r1_flip"],
            CASE_QUERY_CLUSTER_COLS,
            iters=25,
            seed=5,
            bootstrap_unit="case_query",
        )
        row = result.iloc[0]
        self.assertEqual(row["bootstrap_unit"], "case_query")
        self.assertEqual(int(row["cluster_count"]), 3)
        self.assertEqual(int(row["unique_query_count"]), 2)
        self.assertEqual(int(row["case_query_count"]), 3)
        self.assertEqual(int(row["trial_count"]), 6)


if __name__ == "__main__":
    unittest.main()

