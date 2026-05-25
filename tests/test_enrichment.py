import contextlib
import importlib.util
import importlib
import io
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
        enrichment_space="global",
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
        enricher = modules.TargetPrototypeEnricher(512, 4096, args(enrichment_space="grab")).float()
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
        enricher = modules.TargetPrototypeEnricher(512, 4096, args(enrichment_space="grab")).float()
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

    def test_global_enrichment_prunes_grab_enrichment_modules(self):
        enricher = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(enrichment_space="global", only_global=False),
        ).float()
        self.assertTrue(enricher.enable_global)
        self.assertFalse(enricher.enable_grab)
        self.assertFalse(hasattr(enricher, "proto_to_grab"))
        self.assertFalse(hasattr(enricher, "grab_query_proj"))
        self.assertFalse(hasattr(enricher, "grab_proto_proj"))
        self.assertFalse(hasattr(enricher, "grab_fusion"))
        self.assertLess(sum(p.numel() for p in enricher.parameters()), 2_000_000)

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

        with self.assertRaisesRegex(ValueError, "GRAB enrichment is disabled"):
            enricher(
                query_features=torch.randn(3, 4096),
                host_text_features=torch.randn(3, 512),
                query_pids=torch.tensor([0, 1, 2]),
                pool_cache=cache,
                space="grab",
            )

    def test_grab_enrichment_prunes_global_enrichment_modules(self):
        enricher = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(enrichment_space="grab", only_global=False),
        ).float()
        self.assertFalse(enricher.enable_global)
        self.assertTrue(enricher.enable_grab)
        self.assertFalse(hasattr(enricher, "global_query_proj"))
        self.assertFalse(hasattr(enricher, "global_proto_proj"))
        self.assertFalse(hasattr(enricher, "global_fusion"))

    def test_grab_enrichment_requires_grab_features(self):
        with self.assertRaisesRegex(ValueError, "requires GRAB features"):
            modules.TargetPrototypeEnricher(
                512,
                4096,
                args(enrichment_space="grab", only_global=True),
            )

    def test_disabled_target_losses_return_zero_components(self):
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
        self.assertTrue(torch.allclose(out["target_loss"], torch.zeros_like(out["target_loss"])))
        self.assertTrue(torch.allclose(out["att_loss"], torch.zeros_like(out["att_loss"])))
        self.assertTrue(torch.allclose(out["robust_loss"], torch.zeros_like(out["robust_loss"])))
        self.assertTrue(torch.allclose(out["guard_loss"], torch.zeros_like(out["guard_loss"])))
        self.assertTrue(torch.allclose(out["gain_loss"], torch.zeros_like(out["gain_loss"])))
        self.assertTrue(torch.allclose(out["total_loss"], torch.zeros_like(out["total_loss"])))

    def test_only_target_retrieval_loss_skips_auxiliary_components(self):
        enricher = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(
                use_target_retrieval_loss=True,
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
        self.assertTrue(torch.allclose(out["att_loss"], torch.zeros_like(out["att_loss"])))
        self.assertTrue(torch.allclose(out["robust_loss"], torch.zeros_like(out["robust_loss"])))
        self.assertTrue(torch.allclose(out["guard_loss"], torch.zeros_like(out["guard_loss"])))
        self.assertTrue(torch.allclose(out["gain_loss"], torch.zeros_like(out["gain_loss"])))
        self.assertTrue(torch.allclose(out["total_loss"], out["target_loss"], atol=1e-5))

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

    def test_forward_uses_supplied_frozen_top_indices(self):
        torch.manual_seed(13)
        enricher = modules.TargetPrototypeEnricher(512, 4096, args()).float()
        supplied = torch.tensor([
            [5, 4, 3],
            [2, 1, 0],
        ])
        cache = {
            "host_image_features": torch.randn(6, 512),
            "retrieval_features": torch.randn(6, 512),
            "prototypes": torch.randn(6, 7, 512),
            "pids": torch.tensor([0, 1, 2, 3, 4, 5]),
            "top_indices": supplied,
        }
        out = enricher(
            query_features=torch.randn(2, 512, requires_grad=True),
            host_text_features=torch.randn(2, 512, requires_grad=True),
            query_pids=torch.tensor([0, 1]),
            pool_cache=cache,
            space="global",
        )
        self.assertTrue(torch.equal(out["top_indices"].cpu(), supplied))


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

    def test_frozen_indices_bypass_recompute_interval(self):
        manager = make_pool_manager(use_freeze_indices=True, top_m=2)
        manager.refresh = lambda model, epoch, step: self.fail("frozen indices should not refresh")
        manager.frozen_cache = {
            "host_image_features": torch.zeros(4, 512),
            "retrieval_features": torch.zeros(4, 512),
            "prototypes": torch.zeros(4, 7, 512),
            "pids": torch.tensor([0, 1, 2, 3]),
        }
        manager.frozen_rank_indices = torch.tensor([
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ])
        manager.frozen_index_depth = 4
        manager.frozen_cache_requests = 0

        cache1 = manager.get_train_cache(
            FakeImageEncoder(),
            {"index": torch.tensor([0, 2])},
            epoch=1,
            step=1,
        )
        cache2 = manager.get_train_cache(
            FakeImageEncoder(),
            {"index": torch.tensor([3, 1])},
            epoch=99,
            step=99,
        )

        self.assertTrue(torch.equal(cache1["top_indices"], torch.tensor([[0, 1], [2, 3]])))
        self.assertTrue(torch.equal(cache2["top_indices"], torch.tensor([[3, 2], [1, 0]])))
        self.assertEqual(cache1["diagnostics"]["pool_interval_reused"], 0.0)
        self.assertEqual(cache2["diagnostics"]["pool_interval_reused"], 1.0)
        self.assertEqual(cache2["diagnostics"]["pool_k_mode"], "frozen_indices")

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
    def test_host_clip_finetune_cli_parses_checkpoint_path(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--finetune_clip", "host_clip.pth"]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(parsed.finetune_clip, "host_clip.pth")

    def test_freeze_host_cli_requires_target_enrichment(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--freeze_host"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
            sys.argv = ["test", "--freeze_host", "--target_enrichment"]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertTrue(parsed.freeze_host)
        self.assertTrue(parsed.target_enrichment)

    def test_freeze_indices_cli_requires_target_enrichment(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--use_freeze_indices"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
            sys.argv = ["test", "--target_enrichment", "--use_freeze_indices"]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertTrue(parsed.target_enrichment)
        self.assertTrue(parsed.use_freeze_indices)

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
        self.assertEqual(parsed.enrichment_start, 1)
        self.assertFalse(parsed.use_target_retrieval_loss)
        self.assertFalse(parsed.use_target_attention_loss)
        self.assertFalse(parsed.use_target_robust_loss)

    def test_enrichment_start_cli_parses_start_epoch(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--enrichment_start", "5"]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(parsed.enrichment_start, 5)

    def test_enrichment_start_cli_requires_positive_epoch(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--enrichment_start", "0"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
        finally:
            sys.argv = old_argv

    def test_loss_cli_flags_are_store_true_and_host_can_be_disabled(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--no_use_host_loss",
                "--use_target_retrieval_loss",
                "--use_target_attention_loss",
                "--use_target_robust_loss",
            ]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertFalse(parsed.use_host_loss)
        self.assertTrue(parsed.use_target_retrieval_loss)
        self.assertTrue(parsed.use_target_attention_loss)
        self.assertTrue(parsed.use_target_robust_loss)

    def test_lr_total_epochs_overrides_training_epoch_count(self):
        build = importlib.import_module("solver.build")
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0)
        sched = build.build_lr_scheduler(SimpleNamespace(
            milestones=[],
            gamma=0.1,
            warmup_factor=1.0,
            warmup_epochs=0,
            warmup_method="linear",
            lr_total_epochs=200,
            num_epoch=60,
            lrscheduler="cosine",
            target_lr=0,
            power=0.9,
        ), optimizer)
        self.assertEqual(sched.total_epochs, 200)


class HostLossForwardTests(unittest.TestCase):
    def import_model_build(self):
        if "ftfy" not in sys.modules:
            sys.modules["ftfy"] = SimpleNamespace(fix_text=lambda text: text)
        return importlib.import_module("model.build")

    def test_no_use_host_loss_skips_host_objectives(self):
        build = self.import_model_build()

        class FakeBaseModel(torch.nn.Module):
            def forward(self, images, caption_ids):
                batch_size = images.shape[0]
                embed_dim = 4
                image_feats = torch.arange(
                    batch_size * 2 * embed_dim,
                    dtype=torch.float32,
                ).reshape(batch_size, 2, embed_dim)
                text_feats = torch.arange(
                    batch_size * caption_ids.shape[1] * embed_dim,
                    dtype=torch.float32,
                ).reshape(batch_size, caption_ids.shape[1], embed_dim)
                return image_feats, None, text_feats, None

        def fail_host_objective(*_args, **_kwargs):
            raise AssertionError("host loss objective should not be computed")

        patched_names = [
            "cosine_similarity_matrix",
            "sample_hard_negatives",
            "update_labels_for_negatives",
            "create_sample_pairs",
            "compute_cid",
            "compute_id",
            "compute_TAL",
        ]
        originals = {name: getattr(build.objectives, name) for name in patched_names}
        try:
            for name in patched_names:
                setattr(build.objectives, name, fail_host_objective)

            model = build.ITSELF.__new__(build.ITSELF)
            torch.nn.Module.__init__(model)
            model.args = SimpleNamespace(
                use_host_loss=False,
                return_all=False,
                only_global=True,
                target_enrichment=False,
                lambda_host=1.0,
                margin=0.1,
                tau=0.015,
            )
            model.current_task = ["tal", "cid"]
            model.logit_scale = torch.ones([])
            model.base_model = FakeBaseModel()

            ret = build.ITSELF.forward(model, {
                "images": torch.zeros(2, 3, 8, 8),
                "caption_ids": torch.tensor([[0, 2, 1], [1, 0, 2]]),
                "pids": torch.tensor([0, 1]),
            })
        finally:
            for name, original in originals.items():
                setattr(build.objectives, name, original)

        self.assertNotIn("cid_loss", ret)
        self.assertNotIn("tal_loss", ret)
        self.assertEqual(ret["host_loss"].item(), 0.0)
        self.assertEqual(ret["loss"].item(), 0.0)


class HostCheckpointTests(unittest.TestCase):
    def test_extract_host_state_dict_from_itself_checkpoint(self):
        checkpoint = importlib.import_module("utils.checkpoint")
        state_dict = checkpoint.unwrap_checkpoint_state_dict({
            "model": {
                "module.base_model.visual.conv1.weight": torch.ones(1),
                "module.classifier_global.weight": torch.zeros(1),
            }
        })
        host_state_dict = checkpoint.extract_host_model_state_dict(state_dict)
        self.assertEqual(list(host_state_dict.keys()), ["visual.conv1.weight"])
        self.assertTrue(torch.equal(host_state_dict["visual.conv1.weight"], torch.ones(1)))

    def test_raw_clip_state_dict_is_kept(self):
        checkpoint = importlib.import_module("utils.checkpoint")
        state_dict = checkpoint.unwrap_checkpoint_state_dict({
            "state_dict": {
                "visual.conv1.weight": torch.ones(1),
                "transformer.resblocks.0.attn.in_proj_weight": torch.zeros(1),
            }
        })
        host_state_dict = checkpoint.extract_host_model_state_dict(state_dict)
        self.assertEqual(set(host_state_dict.keys()), set(state_dict.keys()))


class HostFreezeTests(unittest.TestCase):
    def import_model_build(self):
        if "ftfy" not in sys.modules:
            sys.modules["ftfy"] = SimpleNamespace(fix_text=lambda text: text)
        return importlib.import_module("model.build")

    def test_freeze_host_keeps_only_target_enricher_trainable(self):
        build = self.import_model_build()
        model = torch.nn.Module()
        model.base_model = torch.nn.Linear(2, 2)
        model.classifier_global = torch.nn.Linear(2, 2)
        model.target_enricher = torch.nn.Linear(2, 2)

        frozen_params, trainable_params = build.freeze_host_parameters(model)

        self.assertGreater(frozen_params, 0)
        self.assertGreater(trainable_params, 0)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("target_enricher."))

    def test_freeze_host_requires_enrichment_parameters(self):
        build = self.import_model_build()
        model = torch.nn.Module()
        model.base_model = torch.nn.Linear(2, 2)

        with self.assertRaisesRegex(ValueError, "target enrichment"):
            build.freeze_host_parameters(model)


if __name__ == "__main__":
    unittest.main()
