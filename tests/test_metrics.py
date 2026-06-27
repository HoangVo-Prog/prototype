import sys
import types
import unittest
from types import SimpleNamespace

import torch


def _ensure_prettytable_importable():
    try:
        import prettytable  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("prettytable")

        class PrettyTable:
            def __init__(self, *args, **kwargs):
                self.rows = []
                self.custom_format = {}

            def add_row(self, row):
                self.rows.append(row)

            def __str__(self):
                return "\n".join(str(row) for row in self.rows)

        module.PrettyTable = PrettyTable
        sys.modules["prettytable"] = module


def _legacy_rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(
            similarity,
            k=max_rank,
            dim=1,
            largest=True,
            sorted=True,
        )
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    inp = [
        tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
        for i, match_row in enumerate(matches)
    ]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class RetrievalMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_prettytable_importable()
        import utils.metrics as train_metrics
        import utils.test_metrics as test_metrics

        cls.train_metrics = train_metrics
        cls.test_metrics = test_metrics

    def test_rank_matches_legacy_full_metric_formula(self):
        generator = torch.Generator().manual_seed(123)
        similarity = torch.randn(6, 12, generator=generator)
        qids = torch.tensor([0, 1, 2, 3, 4, 5])
        gids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])

        expected = _legacy_rank(similarity, qids, gids, max_rank=10, get_mAP=True)
        for module in (self.train_metrics, self.test_metrics):
            actual = module.rank(similarity, qids, gids, max_rank=10, get_mAP=True)
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])
            torch.testing.assert_close(actual[2], expected[2])
            self.assertTrue(torch.equal(actual[3], expected[3]))

    def test_rank_matches_legacy_topk_formula(self):
        generator = torch.Generator().manual_seed(456)
        similarity = torch.randn(5, 11, generator=generator)
        qids = torch.tensor([0, 1, 2, 3, 4])
        gids = torch.tensor([4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4])

        expected = _legacy_rank(similarity, qids, gids, max_rank=10, get_mAP=False)
        for module in (self.train_metrics, self.test_metrics):
            actual = module.rank(similarity, qids, gids, max_rank=10, get_mAP=False)
            torch.testing.assert_close(actual[0], expected[0])
            self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_eval_task_iterator_keeps_ablation_order_and_count(self):
        args = SimpleNamespace(only_global=False)
        evaluator = self.train_metrics.Evaluator(None, None, args)
        sims_global = torch.randn(2, 3)
        sims_grab = torch.randn(2, 3)
        sims_target = torch.randn(2, 3)

        base_tasks = evaluator._build_base_tasks(sims_global, sims_grab)
        names = [
            name
            for name, _ in evaluator._iter_eval_tasks(base_tasks, sims_target)
        ]

        self.assertEqual(13, len(base_tasks))
        self.assertEqual(13 * 12, len(names))
        self.assertEqual("global", names[0])
        self.assertEqual("grab", names[1])
        self.assertEqual("global+grab(0.1)", names[2])
        self.assertEqual("global+proto(0)", names[13])
        self.assertEqual("global+grab(0.32)+proto(1)", names[-1])


if __name__ == "__main__":
    unittest.main()
