import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cue_swap_counterfactual_eval import compute_swap_strength_variants


class SwapStrengthFormulaTest(unittest.TestCase):
    def assertVariantsAlmostEqual(self, values, raw, normalized, clamped):
        self.assertAlmostEqual(values["swap_strength_raw"], raw)
        self.assertAlmostEqual(values["swap_strength_normalized"], normalized)
        self.assertAlmostEqual(values["swap_strength_clamped"], clamped)

    def test_example_is_raw_sum_not_probability(self):
        values = compute_swap_strength_variants(
            d_soft_a_in_a_dense=0.8,
            d_soft_a_in_b_dense=0.2,
            d_soft_b_in_a_dense=0.2,
            d_soft_b_in_b_dense=0.8,
        )
        self.assertVariantsAlmostEqual(values, raw=1.2, normalized=0.6, clamped=0.6)

    def test_identical_galleries_is_zero(self):
        values = compute_swap_strength_variants(0.5, 0.5, 0.5, 0.5)
        self.assertVariantsAlmostEqual(values, raw=0.0, normalized=0.0, clamped=0.0)

    def test_perfect_swap_raw_is_two(self):
        values = compute_swap_strength_variants(1.0, 0.0, 0.0, 1.0)
        self.assertVariantsAlmostEqual(values, raw=2.0, normalized=1.0, clamped=1.0)

    def test_both_cues_dense_or_sparse_is_zero(self):
        dense = compute_swap_strength_variants(0.8, 0.8, 0.8, 0.8)
        sparse = compute_swap_strength_variants(0.2, 0.2, 0.2, 0.2)
        self.assertAlmostEqual(dense["swap_strength_raw"], 0.0)
        self.assertAlmostEqual(sparse["swap_strength_raw"], 0.0)

    def test_helper_does_not_force_raw_inputs_into_probability_range(self):
        values = compute_swap_strength_variants(1.2, -0.2, -0.2, 1.2)
        self.assertVariantsAlmostEqual(values, raw=2.8, normalized=1.4, clamped=1.0)


if __name__ == "__main__":
    unittest.main()
