import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.gallery_construction import construct_cue_swap_galleries


def build_fixture(
    *,
    gallery_pids,
    psi_a,
    psi_b,
    gallery_size=8,
    dense_ratio=0.5,
    seed=13,
    case_id="case",
    query_id=101,
    trial_id=0,
):
    return construct_cue_swap_galleries(
        pid=1,
        gallery_pids=np.asarray(gallery_pids, dtype=np.int64),
        psi_a=np.asarray(psi_a, dtype=float),
        psi_b=np.asarray(psi_b, dtype=float),
        gallery_size=gallery_size,
        dense_ratio=dense_ratio,
        lambda_contrast=0.0,
        seed=seed,
        case_id=case_id,
        query_id=query_id,
        trial_id=trial_id,
        neutral_strategy="random",
        neutral_pool_factor=5,
    )


class SharedNeutralGalleryConstructionTest(unittest.TestCase):
    def test_shared_neutral_invariants(self):
        gallery_pids = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        psi_a = [0, 0, 0.95, 0.90, 0.85, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
        psi_b = [0, 0, 0.10, 0.09, 0.08, 0.95, 0.90, 0.85, 0.07, 0.06, 0.05, 0.04]

        build, reason = build_fixture(gallery_pids=gallery_pids, psi_a=psi_a, psi_b=psi_b)

        self.assertIsNone(reason)
        self.assertIsNotNone(build)
        assert build is not None
        shared = build.shared_neutral
        dense_a = build.dense["a_dense"]
        dense_b = build.dense["b_dense"]
        positives = build.positive_indices

        np.testing.assert_array_equal(build.neutral["a_dense"], shared)
        np.testing.assert_array_equal(build.neutral["b_dense"], shared)
        np.testing.assert_array_equal(build.galleries["a_dense"][-len(shared) :], shared)
        np.testing.assert_array_equal(build.galleries["b_dense"][-len(shared) :], shared)

        self.assertEqual(set(positives.tolist()), {0, 1})
        self.assertTrue(set(shared.tolist()).isdisjoint(set(positives.tolist())))
        self.assertTrue(set(shared.tolist()).isdisjoint(set(dense_a.tolist()) | set(dense_b.tolist())))
        self.assertEqual(set(build.galleries["a_dense"][: len(positives)].tolist()), set(positives.tolist()))
        self.assertEqual(set(build.galleries["b_dense"][: len(positives)].tolist()), set(positives.tolist()))
        self.assertEqual(len(build.galleries["a_dense"]), 8)
        self.assertEqual(len(build.galleries["b_dense"]), 8)
        self.assertEqual(len(np.unique(build.galleries["a_dense"])), 8)
        self.assertEqual(len(np.unique(build.galleries["b_dense"])), 8)
        self.assertEqual(
            set(build.galleries["a_dense"].tolist()) - set(positives.tolist()) - set(shared.tolist()),
            set(dense_a.tolist()),
        )
        self.assertEqual(
            set(build.galleries["b_dense"].tolist()) - set(positives.tolist()) - set(shared.tolist()),
            set(dense_b.tolist()),
        )

    def test_deterministic_for_identical_inputs(self):
        gallery_pids = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        psi_a = [0, 0, 0.95, 0.90, 0.85, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
        psi_b = [0, 0, 0.10, 0.09, 0.08, 0.95, 0.90, 0.85, 0.07, 0.06, 0.05, 0.04]

        first, _ = build_fixture(gallery_pids=gallery_pids, psi_a=psi_a, psi_b=psi_b)
        second, _ = build_fixture(gallery_pids=gallery_pids, psi_a=psi_a, psi_b=psi_b)

        assert first is not None and second is not None
        for key in ("a_dense", "b_dense"):
            np.testing.assert_array_equal(first.dense[key], second.dense[key])
            np.testing.assert_array_equal(first.galleries[key], second.galleries[key])
        np.testing.assert_array_equal(first.shared_neutral, second.shared_neutral)

    def test_insufficient_shared_neutral_pool_has_stable_reason(self):
        gallery_pids = [1, 2, 3, 4, 5, 6]
        psi_a = [0, 0.95, 0.90, 0.10, 0.09, 0.08]
        psi_b = [0, 0.10, 0.09, 0.95, 0.90, 0.08]

        build, reason = build_fixture(
            gallery_pids=gallery_pids,
            psi_a=psi_a,
            psi_b=psi_b,
            gallery_size=6,
            dense_ratio=0.4,
        )

        self.assertIsNone(build)
        self.assertEqual(reason, "not_enough_remaining_for_shared_neutral_fill")

    def test_dense_overlap_does_not_break_construction(self):
        gallery_pids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        psi_a = [0, 0.99, 0.98, 0.97, 0.10, 0.09, 0.08, 0.07, 0.06]
        psi_b = [0, 0.99, 0.98, 0.10, 0.97, 0.09, 0.08, 0.07, 0.06]

        build, reason = build_fixture(
            gallery_pids=gallery_pids,
            psi_a=psi_a,
            psi_b=psi_b,
            gallery_size=7,
            dense_ratio=0.5,
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(build)
        assert build is not None
        self.assertTrue(set(build.dense["a_dense"].tolist()) & set(build.dense["b_dense"].tolist()))
        self.assertTrue(
            set(build.shared_neutral.tolist()).isdisjoint(
                set(build.dense["a_dense"].tolist()) | set(build.dense["b_dense"].tolist())
            )
        )
        self.assertEqual(len(np.unique(build.galleries["a_dense"])), len(build.galleries["a_dense"]))
        self.assertEqual(len(np.unique(build.galleries["b_dense"])), len(build.galleries["b_dense"]))


if __name__ == "__main__":
    unittest.main()
