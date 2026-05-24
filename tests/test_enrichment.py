import importlib.util
import importlib
import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


modules = load_module("enrichment_modules", "model/enrichment/modules.py")
pool = load_module("enrichment_pool", "model/enrichment/pool.py")


def args(**overrides):
    defaults = dict(
        top_m=3,
        robust_hard_k=5,
        enrich_gamma=0.1,
        tau=0.015,
        lambda_att=0.1,
        lambda_ret=1.0,
        lambda_rob=0.1,
        lambda_gain=1.0,
        att_margin=0.1,
        gain_margin=0.01,
        use_target_retrieval_loss=True,
        use_target_attention_loss=True,
        use_target_robust_loss=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeImageEncoder(torch.nn.Module):
    def __init__(self, embed_dim=512, num_proto=7):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.embed_dim = embed_dim
        self.num_proto = num_proto

    def encode_target_image_cache(self, images, cache_prototypes=True):
        batch = images.shape[0]
        device = images.device
        base = torch.arange(batch, dtype=torch.float32, device=device).unsqueeze(1)
        host = base.repeat(1, self.embed_dim)
        cache = {"host_image_features": host}
        if cache_prototypes:
            cache["retrieval_features"] = host + 1
            cache["prototypes"] = torch.ones(batch, self.num_proto, self.embed_dim, device=device)
        return cache


def make_pool_manager(**overrides):
    manager = pool.TargetPoolManager.__new__(pool.TargetPoolManager)
    defaults = dict(
        pool_k=4,
        pool_k_mode="static",
        pool_k_candidates="2,4",
        test_batch_size=4,
        num_workers=0,
        positive_ratio_max=0.75,
        pool_dist_metric="l1",
        pool_dist_threshold=0.01,
        recompute_level="step",
        recompute_interval=2,
    )
    defaults.update(overrides)
    manager.args = SimpleNamespace(**defaults)
    manager.logger = None
    manager.records = [
        {"pid": 0, "image_id": 10, "img_path": "a.jpg"},
        {"pid": 1, "image_id": 11, "img_path": "b.jpg"},
        {"pid": 2, "image_id": 12, "img_path": "c.jpg"},
        {"pid": 3, "image_id": 13, "img_path": "d.jpg"},
    ]
    manager.record_index_by_image_id = {10: 0, 11: 1, 12: 2, 13: 3}
    manager.cluster_to_indices = {0: [0, 1], 1: [2, 3]}
    manager.cluster_distribution = torch.tensor([0.5, 0.5])
    manager.cluster_labels = torch.tensor([0, 0, 1, 1])
    manager.rng = __import__("random").Random(7)
    manager.interval_cache = None
    manager.active_interval_id = None
    manager.last_refresh_unit = None
    manager.refresh = lambda model, epoch, step: None
    manager._encode_records = lambda model, records, cache_prototypes=True: {
        "host_image_features": torch.zeros(len(records), 512),
        "retrieval_features": torch.zeros(len(records), 512),
        "prototypes": torch.zeros(len(records), 7, 512),
    }
    return manager


class EnrichmentShapeTests(unittest.TestCase):
    def test_part_prototypes_follow_document_shape(self):
        token_features = torch.randn(2, 193, 512)
        prototypes = modules.build_part_prototypes(token_features, num_parts=6, grid_size=(24, 8))
        self.assertEqual(prototypes.shape, (2, 7, 512))
        norms = prototypes.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_global_forward_and_losses_are_finite(self):
        enricher = modules.TargetPrototypeEnricher(512, 4096, args()).float()
        cache = {
            "host_image_features": torch.randn(8, 512),
            "retrieval_features": torch.randn(8, 512),
            "prototypes": torch.randn(8, 7, 512),
            "pids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 4]),
        }
        out = enricher(
            query_features=torch.randn(3, 512, requires_grad=True),
            host_text_features=torch.randn(3, 512, requires_grad=True),
            query_pids=torch.tensor([0, 1, 2]),
            pool_cache=cache,
            space="global",
        )
        self.assertEqual(out["enriched_features"].shape, (3, 512))
        self.assertEqual(out["top_indices"].shape, (3, 3))
        self.assertEqual(enricher.robust_hard_k, 5)
        self.assertTrue(torch.isfinite(out["total_loss"]))
        out["total_loss"].backward()

    def test_grab_forward_projects_prototypes_to_grab_space(self):
        enricher = modules.TargetPrototypeEnricher(512, 4096, args()).float()
        cache = {
            "host_image_features": torch.randn(8, 512),
            "retrieval_features": torch.randn(8, 4096),
            "prototypes": torch.randn(8, 7, 512),
            "pids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 4]),
        }
        out = enricher(
            query_features=torch.randn(3, 4096, requires_grad=True),
            host_text_features=torch.randn(3, 512),
            query_pids=torch.tensor([0, 1, 2]),
            pool_cache=cache,
            space="grab",
        )
        self.assertEqual(out["enriched_features"].shape, (3, 4096))
        self.assertTrue(torch.isfinite(out["total_loss"]))

    def test_grab_retrieval_features_must_match_grab_dimension(self):
        enricher = modules.TargetPrototypeEnricher(512, 4096, args()).float()
        cache = {
            "host_image_features": torch.randn(8, 512),
            "retrieval_features": torch.randn(8, 512),
            "prototypes": torch.randn(8, 7, 512),
            "pids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 4]),
        }
        with self.assertRaises(RuntimeError):
            enricher(
                query_features=torch.randn(3, 4096),
                host_text_features=torch.randn(3, 512),
                query_pids=torch.tensor([0, 1, 2]),
                pool_cache=cache,
                space="grab",
            )

    def test_method_losses_can_be_disabled_independently(self):
        enricher = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(
                use_target_retrieval_loss=False,
                use_target_attention_loss=False,
                use_target_robust_loss=False,
            ),
        ).float()
        cache = {
            "host_image_features": torch.randn(8, 512),
            "retrieval_features": torch.randn(8, 512),
            "prototypes": torch.randn(8, 7, 512),
            "pids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 4]),
        }
        out = enricher(
            query_features=torch.randn(3, 512, requires_grad=True),
            host_text_features=torch.randn(3, 512, requires_grad=True),
            query_pids=torch.tensor([0, 1, 2]),
            pool_cache=cache,
            space="global",
        )
        self.assertTrue(torch.isfinite(out["target_loss"]))
        self.assertTrue(torch.isfinite(out["att_loss"]))
        self.assertTrue(torch.isfinite(out["robust_loss"]))
        self.assertTrue(torch.allclose(out["total_loss"], torch.zeros_like(out["total_loss"])))

    def test_target_losses_default_to_disabled_in_enricher(self):
        defaults = args()
        delattr(defaults, "use_target_retrieval_loss")
        delattr(defaults, "use_target_attention_loss")
        delattr(defaults, "use_target_robust_loss")
        enricher = modules.TargetPrototypeEnricher(512, 4096, defaults).float()
        self.assertFalse(enricher.use_target_retrieval_loss)
        self.assertFalse(enricher.use_target_attention_loss)
        self.assertFalse(enricher.use_target_robust_loss)

    def test_lambda_ret_scales_target_retrieval_objective(self):
        cache = {
            "host_image_features": torch.randn(8, 512),
            "retrieval_features": torch.randn(8, 512),
            "prototypes": torch.randn(8, 7, 512),
            "pids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 4]),
        }
        query = torch.randn(3, 512, requires_grad=True)
        host = torch.randn(3, 512, requires_grad=True)
        common = dict(
            query_features=query,
            host_text_features=host,
            query_pids=torch.tensor([0, 1, 2]),
            pool_cache=cache,
            space="global",
        )
        base = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(use_target_attention_loss=False, use_target_robust_loss=False, lambda_ret=1.0),
        ).float()
        scaled = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(use_target_attention_loss=False, use_target_robust_loss=False, lambda_ret=2.0),
        ).float()
        scaled.load_state_dict(base.state_dict())
        base_out = base(**common)
        scaled_out = scaled(**common)
        self.assertTrue(torch.allclose(scaled_out["target_loss"], base_out["target_loss"]))
        self.assertTrue(torch.allclose(scaled_out["total_loss"], base_out["total_loss"] * 2, atol=1e-5))

    def test_top_m_selection_is_label_free(self):
        torch.manual_seed(11)
        enricher = modules.TargetPrototypeEnricher(512, 4096, args()).float()
        host_image_features = torch.randn(8, 512)
        retrieval_features = torch.randn(8, 512)
        prototypes = torch.randn(8, 7, 512)
        host_text_features = torch.randn(3, 512)
        query_features = torch.randn(3, 512, requires_grad=True)
        cache_a = {
            "host_image_features": host_image_features,
            "retrieval_features": retrieval_features,
            "prototypes": prototypes,
            "pids": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        }
        cache_b = {
            "host_image_features": host_image_features,
            "retrieval_features": retrieval_features,
            "prototypes": prototypes,
            "pids": torch.tensor([7, 6, 5, 4, 3, 2, 1, 0]),
        }
        out_a = enricher(
            query_features=query_features,
            host_text_features=host_text_features,
            query_pids=torch.tensor([0, 1, 2]),
            pool_cache=cache_a,
            space="global",
        )
        out_b = enricher(
            query_features=query_features,
            host_text_features=host_text_features,
            query_pids=torch.tensor([9, 8, 7]),
            pool_cache=cache_b,
            space="global",
        )
        self.assertTrue(torch.equal(out_a["top_indices"], out_b["top_indices"]))


class PoolManagerTests(unittest.TestCase):
    def test_unique_records_and_recompute_policy(self):
        manager = pool.TargetPoolManager.__new__(pool.TargetPoolManager)
        records = manager._build_unique_records([
            (0, 10, "a.jpg", "first caption"),
            (0, 10, "a.jpg", "second caption"),
            (1, 11, "b.jpg", "caption"),
        ])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["image_id"], 10)

        manager.args = SimpleNamespace(recompute_level="epoch", recompute_interval=-1)
        manager.cluster_to_indices = None
        manager.interval_cache = None
        self.assertTrue(manager._should_refresh(epoch=1, step=1))
        manager.cluster_to_indices = {0: [0]}
        manager.interval_cache = {"host_image_features": torch.zeros(1, 512)}
        manager.active_interval_id = 0
        self.assertFalse(manager._should_refresh(epoch=99, step=99))

        manager.args = SimpleNamespace(recompute_level="step", recompute_interval=3)
        manager.active_interval_id = 1
        self.assertFalse(manager._should_refresh(epoch=1, step=6))
        self.assertTrue(manager._should_refresh(epoch=1, step=7))

    def test_pool_construction_inserts_positives_before_quota_fill(self):
        manager = make_pool_manager(positive_ratio_max=0.75)
        batch = {
            "image_ids": torch.tensor([10, 10, 11]),
            "pids": torch.tensor([0, 0, 1]),
            "images": torch.randn(3, 3, 8, 8),
        }
        model = FakeImageEncoder()
        model.train()
        distractors, diagnostics = manager._sample_distractor_indices(
            manager._required_positives(batch),
            pool_k=4,
        )
        self.assertEqual(set(distractors), {2, 3})
        self.assertEqual(diagnostics["pool_cluster_distribution_distance"], 0.0)
        cache = manager._build_batch_pool_cache(model, batch)
        self.assertEqual(set(cache["image_ids"].tolist()), {10, 11, 12, 13})
        self.assertIn(10, cache["image_ids"].tolist())
        self.assertIn(11, cache["image_ids"].tolist())
        self.assertEqual(cache["diagnostics"]["pool_num_required_positives"], 2.0)
        self.assertEqual(cache["diagnostics"]["pool_num_inserted_positives"], 2.0)
        self.assertEqual(cache["diagnostics"]["pool_positive_ratio"], 0.5)
        self.assertEqual(cache["diagnostics"]["pool_cluster_distribution_distance"], 0.0)
        self.assertEqual(cache["diagnostics"]["pool_final_pool_size"], 4.0)
        self.assertEqual(cache["diagnostics"]["pool_missing_positive_count"], 0.0)
        self.assertIn("positive_ratio", cache["diagnostics"])
        self.assertIn("cluster_distribution_distance", cache["diagnostics"])
        self.assertTrue(model.training)

    def test_interval_pool_cache_reused_for_multiple_steps(self):
        manager = make_pool_manager(recompute_interval=2)
        model = FakeImageEncoder()
        batch1 = {
            "image_ids": torch.tensor([10, 11]),
            "pids": torch.tensor([0, 1]),
            "images": torch.randn(2, 3, 8, 8),
        }
        batch2 = {
            "image_ids": torch.tensor([12, 13]),
            "pids": torch.tensor([2, 3]),
            "images": torch.randn(2, 3, 8, 8),
        }
        cache1 = manager.get_train_cache(model, batch1, epoch=1, step=1)
        cache2 = manager.get_train_cache(model, batch2, epoch=1, step=2)
        self.assertIs(cache1, cache2)
        self.assertEqual(cache2["diagnostics"]["pool_interval_reused"], 1.0)
        self.assertEqual(cache2["diagnostics"]["pool_interval_id"], 0.0)

    def test_static_mode_uses_exact_pool_k(self):
        manager = make_pool_manager(pool_k=3, positive_ratio_max=1.0)
        batch = {
            "image_ids": torch.tensor([10]),
            "pids": torch.tensor([0]),
            "images": torch.randn(1, 3, 8, 8),
        }
        cache = manager._build_batch_pool_cache(FakeImageEncoder(), batch)
        self.assertEqual(cache["diagnostics"]["pool_selected_k"], 3.0)
        self.assertEqual(cache["diagnostics"]["pool_final_pool_size"], 3.0)
        self.assertEqual(cache["image_ids"].numel(), 3)

    def test_adaptive_mode_uses_document_max_formula(self):
        manager = make_pool_manager(
            pool_k=1,
            pool_k_mode="adaptive",
            pool_k_candidates="2",
            positive_ratio_max=0.5,
        )
        batch = {
            "image_ids": torch.tensor([10, 12]),
            "pids": torch.tensor([0, 2]),
            "images": torch.randn(2, 3, 8, 8),
        }
        cache = manager._build_batch_pool_cache(FakeImageEncoder(), batch)
        self.assertEqual(cache["diagnostics"]["pool_selected_k"], 4.0)
        self.assertEqual(cache["diagnostics"]["pool_k_valid_target"], 2.0)
        self.assertEqual(cache["diagnostics"]["pool_k_dilute_target"], 4.0)
        self.assertEqual(cache["diagnostics"]["pool_k_dist_target"], 2.0)
        self.assertEqual(cache["diagnostics"]["pool_positive_ratio"], 0.5)
        self.assertEqual(cache["diagnostics"]["pool_k_valid"], 1.0)
        self.assertEqual(cache["diagnostics"]["pool_k_dilute"], 1.0)

    def test_k_valid_failure_raises_clear_error(self):
        manager = pool.TargetPoolManager.__new__(pool.TargetPoolManager)
        manager.args = SimpleNamespace(pool_k=1)
        manager.records = [{"pid": 0, "image_id": 10, "img_path": "a.jpg"}]
        manager.record_index_by_image_id = {10: 0, 11: 1}
        manager.cluster_labels = torch.tensor([0, 0])
        batch = {
            "image_ids": torch.tensor([10, 11]),
            "pids": torch.tensor([0, 1]),
            "images": torch.randn(2, 3, 8, 8),
        }
        with self.assertRaisesRegex(ValueError, "K_valid failed"):
            manager._build_batch_pool_cache(FakeImageEncoder(), batch)


class SchedulerOptionTests(unittest.TestCase):
    def test_positive_boolean_cli_parser(self):
        options = importlib.import_module("utils.options")
        self.assertTrue(options.str2bool("true"))
        self.assertFalse(options.str2bool("false"))

    def test_only_host_loss_defaults_to_enabled(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test"]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertTrue(parsed.use_host_loss)
        self.assertFalse(parsed.use_target_retrieval_loss)
        self.assertFalse(parsed.use_target_attention_loss)
        self.assertFalse(parsed.use_target_robust_loss)

    def test_lr_total_epoch_overrides_training_epoch_count(self):
        build = importlib.import_module("solver.build")
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0)
        sched = build.build_lr_scheduler(SimpleNamespace(
            milestones=[],
            gamma=0.1,
            warmup_factor=1.0,
            warmup_epochs=0,
            warmup_method="linear",
            lr_total_epoch=200,
            num_epoch=60,
            lrscheduler="cosine",
            target_lr=0,
            power=0.9,
        ), optimizer)
        self.assertEqual(sched.total_epochs, 200)


if __name__ == "__main__":
    unittest.main()
