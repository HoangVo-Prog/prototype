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


class _FakeAblationModel(torch.nn.Module):
    def __init__(self, dim=8, max_enrich_batch=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.dim = dim
        self.max_enrich_batch = max_enrich_batch
        self.reset_counts()

    def reset_counts(self):
        self.text_bundle_calls = 0
        self.image_bundle_calls = 0
        self.encode_text_calls = 0
        self.encode_image_calls = 0
        self.encode_text_grab_calls = 0
        self.encode_image_grab_calls = 0
        self.encode_target_image_cache_calls = 0
        self.enrich_text_calls = 0

    def _features(self, payload, offset, dim=None):
        dim = dim or self.dim
        values = payload.float().view(-1, 1) + offset
        basis = torch.arange(1, dim + 1, device=payload.device).float().view(1, -1)
        return torch.sin(values * basis * 0.37) + torch.cos(values * basis * 0.11)

    def encode_text(self, text):
        self.encode_text_calls += 1
        return self._features(text, 0.1)

    def encode_image(self, image):
        self.encode_image_calls += 1
        return self._features(image, 0.2)

    def encode_text_grab(self, text):
        self.encode_text_grab_calls += 1
        return self._features(text, 0.3, dim=self.dim + 2)

    def encode_image_grab(self, image):
        self.encode_image_grab_calls += 1
        return self._features(image, 0.4, dim=self.dim + 2)

    def encode_eval_text_bundle(self, text, include_grab=False):
        self.text_bundle_calls += 1
        bundle = {"host_features": self._features(text, 0.1)}
        if include_grab:
            bundle["grab_features"] = self._features(text, 0.3, dim=self.dim + 2)
        return bundle

    def encode_eval_image_bundle(
        self,
        image,
        include_grab=False,
        cache_target=False,
        cache_prototypes=True,
    ):
        self.image_bundle_calls += 1
        host_features = self._features(image, 0.2)
        bundle = {"host_features": host_features}
        if include_grab:
            bundle["grab_features"] = self._features(image, 0.4, dim=self.dim + 2)
        if cache_target:
            cache = {"host_image_features": host_features}
            if cache_prototypes:
                cache["retrieval_features"] = host_features + 0.05
                cache["prototypes"] = host_features.unsqueeze(1)
            bundle["target_cache"] = cache
        return bundle

    def encode_target_image_cache(self, image):
        self.encode_target_image_cache_calls += 1
        return self.encode_eval_image_bundle(
            image,
            include_grab=False,
            cache_target=True,
        )["target_cache"]

    def finalize_target_cache(self, cache):
        return cache

    def enrich_text_features(
        self,
        query_features,
        host_text_features,
        target_cache,
        grab_text_features=None,
    ):
        self.enrich_text_calls += 1
        if (
            self.max_enrich_batch is not None
            and host_text_features.shape[0] > self.max_enrich_batch
        ):
            raise torch.cuda.OutOfMemoryError("synthetic enrichment OOM")
        context = target_cache["host_image_features"].mean(dim=0, keepdim=True)
        return host_text_features + 0.05 * context


def _fake_loaders():
    txt_loader = [
        (torch.tensor([0, 1]), torch.tensor([0.0, 1.0])),
        (torch.tensor([2, 3]), torch.tensor([2.0, 3.0])),
    ]
    img_loader = [
        (torch.tensor([0, 1, 2, 3, 0]), torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])),
        (torch.tensor([1, 2, 3, 0, 1]), torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0])),
    ]
    return img_loader, txt_loader


def _legacy_full_table(module, img_loader, txt_loader, args, model):
    device = next(model.parameters()).device

    qids, gids, qfeats, gfeats = [], [], [], []
    for pid, caption in txt_loader:
        caption = caption.to(device)
        with torch.no_grad():
            qfeats.append(model.encode_text(caption).cpu())
        qids.append(pid.view(-1))
    for pid, image in img_loader:
        image = image.to(device)
        with torch.no_grad():
            gfeats.append(model.encode_image(image).cpu())
        gids.append(pid.view(-1))
    qids = torch.cat(qids, 0).cpu()
    gids = torch.cat(gids, 0).cpu()
    qfeats = torch.cat(qfeats, 0).cpu()
    gfeats = torch.cat(gfeats, 0).cpu()
    sims_global = torch.nn.functional.normalize(qfeats, p=2, dim=1) @ torch.nn.functional.normalize(gfeats, p=2, dim=1).t()

    vq_feats, vg_feats = [], []
    for _, caption in txt_loader:
        caption = caption.to(device)
        with torch.no_grad():
            vq_feats.append(model.encode_text_grab(caption).cpu())
    for _, image in img_loader:
        image = image.to(device)
        with torch.no_grad():
            vg_feats.append(model.encode_image_grab(image).cpu())
    vq_feats = torch.cat(vq_feats, 0).cpu()
    vg_feats = torch.cat(vg_feats, 0).cpu()
    sims_grab = torch.nn.functional.normalize(vq_feats, p=2, dim=1) @ torch.nn.functional.normalize(vg_feats, p=2, dim=1).t()

    base_tasks = [("global", sims_global), ("grab", sims_grab)]
    for lambda_value in module._global_grab_lambdas():
        alpha = module._format_lambda(lambda_value)
        base_tasks.append((
            "global+grab({})".format(alpha),
            module._scaled_fuse(sims_global, sims_grab, lambda_value),
        ))

    cache_chunks = []
    target_gids = []
    for pid, image in img_loader:
        image = image.to(device)
        with torch.no_grad():
            cache = model.encode_target_image_cache(image)
        target_gids.append(pid.view(-1))
        cache_chunks.append({key: value.detach().cpu() for key, value in cache.items()})
    target_gids = torch.cat(target_gids, 0).cpu()
    target_cache = {}
    for key in cache_chunks[0].keys():
        target_cache[key] = torch.cat([chunk[key] for chunk in cache_chunks], dim=0).to(device)
    target_cache["pids"] = target_gids.to(device)
    target_cache = model.finalize_target_cache(target_cache)

    target_qids, target_qfeats = [], []
    for pid, caption in txt_loader:
        caption = caption.to(device)
        with torch.no_grad():
            host_text_feat = model.encode_text(caption)
            text_feat = model.enrich_text_features(
                host_text_feat,
                host_text_feat,
                target_cache,
                grab_text_features=None,
            ).cpu()
        target_qids.append(pid.view(-1))
        target_qfeats.append(text_feat)
    target_qids = torch.cat(target_qids, 0).cpu()
    target_qfeats = torch.cat(target_qfeats, 0).cpu()
    target_gfeats = target_cache["retrieval_features"].detach().cpu()
    sims_target = torch.nn.functional.normalize(target_qfeats, p=2, dim=1) @ torch.nn.functional.normalize(target_gfeats, p=2, dim=1).t()

    rows = {}
    best_task = None
    best_row = None
    for key, sims in base_tasks:
        row = module.get_metrics(sims, qids, target_gids, "{}-t2i".format(key), False)
        rows[key] = row

    for proto_lambda in module._prototype_lambdas():
        proto_value = module._format_lambda(proto_lambda)
        for base_name, base_scores in base_tasks:
            key = "{}+proto({})".format(base_name, proto_value)
            scaled_base_scores = module._scale_scores_like(base_scores, sims_target)
            sims = (1.0 - proto_lambda) * scaled_base_scores + proto_lambda * sims_target
            row = module.get_metrics(sims, target_qids, target_gids, "{}-t2i".format(key), False)
            rows[key] = row
            if best_row is None or row[1] > best_row[1]:
                best_task = key
                best_row = row

    return rows, best_task, float(best_row[1])


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

    def test_optimized_task_iterator_matches_legacy_score_matrices(self):
        args = SimpleNamespace(only_global=False)
        generator = torch.Generator().manual_seed(789)
        sims_global = torch.randn(4, 11, generator=generator)
        sims_grab = torch.randn(4, 11, generator=generator)
        sims_target = torch.randn(4, 11, generator=generator)

        evaluator = self.train_metrics.Evaluator(None, None, args)
        actual_base_tasks = evaluator._build_base_tasks(sims_global, sims_grab)
        expected_base_tasks = [("global", sims_global), ("grab", sims_grab)]
        for lambda_value in self.train_metrics._global_grab_lambdas():
            expected_base_tasks.append((
                "global+grab({})".format(self.train_metrics._format_lambda(lambda_value)),
                self.train_metrics._scaled_fuse(sims_global, sims_grab, lambda_value),
            ))

        actual_tasks = list(evaluator._iter_eval_tasks(actual_base_tasks, sims_target))
        expected_tasks = list(expected_base_tasks)
        for proto_lambda in self.train_metrics._prototype_lambdas():
            proto_value = self.train_metrics._format_lambda(proto_lambda)
            for base_name, base_scores in expected_base_tasks:
                scaled_base = self.train_metrics._scale_scores_like(base_scores, sims_target)
                expected_tasks.append((
                    "{}+proto({})".format(base_name, proto_value),
                    (1.0 - proto_lambda) * scaled_base + proto_lambda * sims_target,
                ))

        self.assertEqual([name for name, _ in expected_tasks], [name for name, _ in actual_tasks])
        for (_, expected_scores), (_, actual_scores) in zip(expected_tasks, actual_tasks):
            torch.testing.assert_close(actual_scores, expected_scores)

    def test_optimized_full_table_matches_legacy_and_reuses_branch_work(self):
        args = SimpleNamespace(
            only_global=False,
            target_enrichment=True,
            enrichment_space="global",
            topm_rank_space="host_global",
            eval_log_interval=0.0,
        )

        for module in (self.train_metrics, self.test_metrics):
            with self.subTest(module=module.__name__):
                img_loader, txt_loader = _fake_loaders()
                legacy_model = _FakeAblationModel()
                expected_rows, expected_best_task, expected_top1 = _legacy_full_table(
                    module,
                    img_loader,
                    txt_loader,
                    args,
                    legacy_model,
                )

                optimized_model = _FakeAblationModel()
                evaluator = module.Evaluator(img_loader, txt_loader, args)
                actual_top1 = evaluator.eval(
                    optimized_model,
                    use_target_enrichment=True,
                )

                self.assertEqual(13 * 12, len(expected_rows))
                self.assertEqual(expected_best_task, evaluator.last_best_task)
                self.assertAlmostEqual(expected_top1, actual_top1, places=6)

                expected_metrics = {}
                for row in expected_rows.values():
                    expected_metrics.update(module._row_to_eval_metrics(row))
                for key, expected_value in expected_metrics.items():
                    self.assertIn(key, evaluator.last_metrics)
                    self.assertAlmostEqual(
                        expected_value,
                        evaluator.last_metrics[key],
                        places=5,
                    )

                self.assertEqual(len(txt_loader), optimized_model.text_bundle_calls)
                self.assertEqual(len(img_loader), optimized_model.image_bundle_calls)
                self.assertEqual(len(txt_loader), optimized_model.enrich_text_calls)
                self.assertEqual(0, optimized_model.encode_text_calls)
                self.assertEqual(0, optimized_model.encode_image_calls)
                self.assertEqual(0, optimized_model.encode_text_grab_calls)
                self.assertEqual(0, optimized_model.encode_image_grab_calls)
                self.assertEqual(0, optimized_model.encode_target_image_cache_calls)

    def test_enriched_eval_retries_smaller_chunks_after_cuda_oom(self):
        args = SimpleNamespace(
            only_global=False,
            target_enrichment=True,
            enrichment_space="global",
            topm_rank_space="host_global",
            eval_log_interval=0.0,
        )

        for module in (self.train_metrics, self.test_metrics):
            with self.subTest(module=module.__name__):
                img_loader, txt_loader = _fake_loaders()
                expected_model = _FakeAblationModel()
                expected_evaluator = module.Evaluator(img_loader, txt_loader, args)
                expected_top1 = expected_evaluator.eval(
                    expected_model,
                    use_target_enrichment=True,
                )

                oom_model = _FakeAblationModel(max_enrich_batch=1)
                oom_evaluator = module.Evaluator(img_loader, txt_loader, args)
                actual_top1 = oom_evaluator.eval(
                    oom_model,
                    use_target_enrichment=True,
                )

                self.assertAlmostEqual(expected_top1, actual_top1, places=6)
                self.assertEqual(expected_evaluator.last_best_task, oom_evaluator.last_best_task)
                for key, expected_value in expected_evaluator.last_metrics.items():
                    self.assertIn(key, oom_evaluator.last_metrics)
                    self.assertAlmostEqual(
                        expected_value,
                        oom_evaluator.last_metrics[key],
                        places=5,
                    )
                self.assertGreater(oom_model.enrich_text_calls, len(txt_loader))


if __name__ == "__main__":
    unittest.main()
