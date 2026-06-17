import contextlib
import importlib.util
import importlib
import io
import pathlib
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np
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
        enrich_gamma=None,
        residual_gate="residual",
        residual_gate_hidden_dim=128,
        context_pooling="mlp",
        tau=0.015,
        lambda_ret=1.0,
        enrichment_space="global",
        topm_rank_space="host_global",
        topm_rank_lambda=0.5,
        extractor_mode="global,horizontal",
        num_parts=6,
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
        test_batch_size=4,
        num_workers=0,
        recompute_level="step",
        recompute_interval=2,
        use_freeze_indices=False,
        top_m=3,
        topm_rank_space="host_global",
        topm_rank_lambda=0.5,
        enrichment_space="global",
        seed=1,
    )
    defaults.update(overrides)
    manager.args = SimpleNamespace(**defaults)
    manager.seed = defaults["seed"]
    manager.logger = None
    manager.train_dataset = SimpleNamespace(tokenizer=None, text_length=77, truncate=True)
    manager.records = [
        {"pid": 0, "image_id": 10, "img_path": "a.jpg"},
        {"pid": 1, "image_id": 11, "img_path": "b.jpg"},
        {"pid": 2, "image_id": 12, "img_path": "c.jpg"},
        {"pid": 3, "image_id": 13, "img_path": "d.jpg"},
    ]
    manager.query_records = [
        {"pid": record["pid"], "query_index": index, "caption": f"caption {index}"}
        for index, record in enumerate(manager.records)
    ]
    manager.full_training_cache = None
    manager.full_training_interval_id = None
    manager.full_training_cache_requests = 0
    manager.frozen_cache = None
    manager.frozen_rank_indices = None
    manager.frozen_gallery_image_ids = None
    manager.frozen_index_depth = None
    manager.frozen_cache_requests = 0
    manager._encode_records = lambda model, records, cache_prototypes=True: {
        "host_image_features": torch.zeros(len(records), 512),
        "retrieval_features": torch.zeros(len(records), 512),
        "prototypes": torch.zeros(len(records), 7, 512),
    }
    return manager


class EnrichmentShapeTests(unittest.TestCase):
    def test_rank_part_mixer_adapter_shape_and_diagnostics(self):
        adapter = modules.RankPartQueryConditionedMixerAdapter(
            embed_dim=16,
            num_ranks=3,
            num_slots=5,
            mixer_dim=8,
            depth=2,
            hidden_part=6,
            hidden_rank=7,
            hidden_channel=12,
            hidden_readout=4,
        ).float()
        z_q = torch.randn(2, 16, requires_grad=True)
        B_q_M = torch.randn(2, 3, 5, 16, requires_grad=True)

        c_q = adapter(z_q, B_q_M)

        self.assertEqual(c_q.shape, (2, 16))
        for key in (
            "mixer/context_norm",
            "mixer/context_delta_cosine",
            "mixer/output_delta_norm",
            "mixer/rank_mixing_weight_norm",
            "mixer/part_mixing_weight_norm",
            "mixer/channel_mixing_weight_norm",
            "mixer/readout_weight_norm",
            "mixer/film_scale_mean",
            "mixer/film_scale_std",
            "mixer/film_shift_mean",
            "mixer/film_shift_std",
            "mixer/H_mean",
            "mixer/H_std",
            "mixer/H_flat_token_std",
            "mixer/readout_output_norm",
        ):
            if key in ("mixer/context_delta_cosine", "mixer/output_delta_norm"):
                continue
            self.assertIn(key, adapter.last_diagnostics)
        c_q.sum().backward()

    def test_rank_part_mixer_rejects_removed_attention_pooling_modes(self):
        for context_pooling in ("late_attention", "hybrid_attention"):
            with self.subTest(context_pooling=context_pooling):
                with self.assertRaisesRegex(ValueError, "context_pooling"):
                    modules.RankPartQueryConditionedMixerAdapter(
                        embed_dim=16,
                        num_ranks=3,
                        num_slots=5,
                        mixer_dim=8,
                        context_pooling=context_pooling,
                    )

    def test_rank_part_mixer_rejects_unknown_context_pooling(self):
        with self.assertRaisesRegex(ValueError, "context_pooling"):
            modules.RankPartQueryConditionedMixerAdapter(
                embed_dim=16,
                num_ranks=3,
                num_slots=5,
                mixer_dim=8,
                context_pooling="attention",
            )

    def test_part_prototypes_follow_document_shape(self):
        token_features = torch.randn(2, 193, 512)
        prototypes = modules.build_part_prototypes(token_features, num_parts=6, grid_size=(24, 8))
        self.assertEqual(prototypes.shape, (2, 7, 512))
        norms = prototypes.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_extractor_modes_produce_expected_slot_counts(self):
        token_features = torch.randn(2, 13, 16)
        expected_slots = {
            "global": 1,
            "horizontal": 2,
            "vertical": 2,
            "grid": 4,
            "global,horizontal": 3,
            "global,vertical": 3,
            "global,grid": 5,
            "horizontal,vertical": 4,
            "global,grid,vertical,horizontal": 9,
        }
        for mode, slot_count in expected_slots.items():
            with self.subTest(mode=mode):
                prototypes = modules.build_part_prototypes(
                    token_features,
                    num_parts=2,
                    grid_size=(3, 4),
                    mode=mode,
                )
                self.assertEqual(prototypes.shape, (2, slot_count, 16))
                self.assertEqual(modules.prototype_slot_count(mode, 2), slot_count)
                norms = prototypes.norm(dim=-1)
                self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_evidence_modes_produce_expected_slot_counts(self):
        token_features = torch.randn(2, 13, 16)
        retrieval_features = torch.randn(2, 16)
        expected_slots = {
            "retrieval_backbone": 1,
            "cluster": 1,
            "cluster_residual": 1,
            "cluster_density": 1,
            "cluster_rarity": 1,
            "global,retrieval_backbone,cluster": 3,
            "grid,cluster,cluster_residual,cluster_density,cluster_rarity": 8,
        }
        for mode, slot_count in expected_slots.items():
            with self.subTest(mode=mode):
                evidence = modules.build_evidence_bank(
                    token_features,
                    num_parts=2,
                    grid_size=(3, 4),
                    mode=mode,
                    retrieval_features=retrieval_features,
                )
                self.assertEqual(evidence.shape, (2, slot_count, 16))
                self.assertEqual(modules.prototype_slot_count(mode, 2), slot_count)

    def test_target_relative_finalizer_fills_full_pool_evidence(self):
        cfg = args(
            extractor_mode="cluster,cluster_residual,cluster_density,cluster_rarity",
            target_relative_space="host_global",
            target_relative_num_clusters=2,
            target_relative_cluster_method="kmeans",
            evidence_projection="auto",
        )
        cache = {
            "host_image_features": torch.tensor([
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]),
            "retrieval_features": torch.randn(4, 2),
            "evidence_bank": torch.zeros(4, 4, 2),
        }
        finalized = modules.finalize_target_evidence_cache(cache, cfg, evidence_dim=2)
        self.assertTrue(torch.equal(finalized["prototypes"], finalized["evidence_bank"]))
        self.assertEqual(finalized["evidence_bank"].shape, (4, 4, 2))
        self.assertIn("cluster_density_scalar", finalized)
        self.assertIn("cluster_rarity_scalar", finalized)
        self.assertIn("target_relative_cluster_ids", finalized)
        self.assertEqual(finalized["cluster_density_scalar"].shape, (4, 1))
        self.assertEqual(finalized["cluster_rarity_scalar"].shape, (4, 1))
        self.assertTrue(torch.isfinite(finalized["evidence_bank"]).all())
        self.assertTrue(torch.isfinite(finalized["cluster_density_scalar"]).all())
        self.assertTrue(torch.isfinite(finalized["cluster_rarity_scalar"]).all())

    def test_vertical_and_grid_extractors_require_grid_shape(self):
        token_features = torch.randn(2, 13, 16)
        with self.assertRaisesRegex(ValueError, "requires a valid patch grid_size"):
            modules.build_part_prototypes(token_features, num_parts=2, mode="vertical")
        with self.assertRaisesRegex(ValueError, "requires a valid patch grid_size"):
            modules.build_part_prototypes(token_features, num_parts=2, mode="grid")

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
        self.assertNotIn("attention_weights", out)
        self.assertIn("mixer/context_norm", out)
        self.assertIn("mixer/context_delta_cosine", out)
        self.assertIn("mixer/output_delta_norm", out)
        self.assertIn("target_residual_gate_mean", out)
        self.assertTrue(torch.isfinite(out["total_loss"]))
        self.assertTrue(torch.allclose(out["total_loss"], out["target_retrieval_loss"], atol=1e-5))
        self.assertNotIn("robust_loss", out)
        self.assertNotIn("guard_loss", out)
        self.assertNotIn("gain_loss", out)
        out["total_loss"].backward()

    def test_enricher_uses_extractor_mode_slot_count(self):
        enricher = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(extractor_mode="global,grid,horizontal", num_parts=2),
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
        self.assertEqual(out["enriched_features"].shape, (3, 512))
        self.assertTrue(torch.isfinite(out["total_loss"]))

    def test_residual_gate_mode_learns_query_adaptive_scale(self):
        enricher = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(
                residual_gate="residual",
                residual_gate_hidden_dim=32,
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
            host_text_features=torch.randn(3, 512),
            query_pids=torch.tensor([0, 1, 2]),
            pool_cache=cache,
            space="global",
        )
        self.assertTrue(hasattr(enricher, "global_residual_gate"))
        self.assertTrue(torch.allclose(
            out["target_residual_gate_mean"],
            torch.tensor(0.1),
            atol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            out["target_residual_gate_std"],
            torch.zeros_like(out["target_residual_gate_std"]),
            atol=1e-6,
        ))
        out["enriched_features"].sum().backward()
        self.assertIsNotNone(enricher.global_residual_gate.net[-1].bias.grad)

    def test_enricher_validates_static_gamma_contract(self):
        with self.assertRaisesRegex(ValueError, "static requires"):
            modules.TargetPrototypeEnricher(
                512,
                4096,
                args(residual_gate="static", enrich_gamma=None),
            )

        with self.assertRaisesRegex(ValueError, "only valid"):
            modules.TargetPrototypeEnricher(
                512,
                4096,
                args(residual_gate="residual", enrich_gamma=0.2),
            )

        enricher = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(residual_gate="static", enrich_gamma=0.2),
        )
        self.assertFalse(hasattr(enricher, "global_residual_gate"))

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
        self.assertLess(sum(p.numel() for p in enricher.parameters()), 3_000_000)

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

    def test_target_retrieval_loss_is_mandatory(self):
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
        self.assertTrue(torch.isfinite(out["target_retrieval_loss"]))
        self.assertTrue(torch.allclose(out["total_loss"], out["target_retrieval_loss"], atol=1e-5))

    def test_target_retrieval_loss_requires_positive_pool_matches(self):
        enricher = modules.TargetPrototypeEnricher(512, 4096, args()).float()
        cache = {
            "host_image_features": torch.randn(8, 512),
            "retrieval_features": torch.randn(8, 512),
            "prototypes": torch.randn(8, 7, 512),
            "pids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 4]),
        }
        with self.assertRaisesRegex(ValueError, "positive image in the target pool"):
            enricher(
                query_features=torch.randn(3, 512, requires_grad=True),
                host_text_features=torch.randn(3, 512, requires_grad=True),
                query_pids=torch.tensor([0, 1, 9]),
                pool_cache=cache,
                space="global",
            )

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
            args(lambda_ret=1.0),
        ).float()
        scaled = modules.TargetPrototypeEnricher(
            512,
            4096,
            args(lambda_ret=2.0),
        ).float()
        scaled.load_state_dict(base.state_dict())
        base_out = base(**common)
        scaled_out = scaled(**common)
        self.assertTrue(torch.allclose(
            scaled_out["target_retrieval_loss"],
            base_out["target_retrieval_loss"],
        ))
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
            query_pids=torch.tensor([7, 6, 5]),
            pool_cache=cache_b,
            space="global",
        )
        self.assertTrue(torch.equal(out_a["top_indices"], out_b["top_indices"]))

    def test_top_m_rank_space_retrieval_uses_retrieval_features(self):
        enricher = modules.TargetPrototypeEnricher(
            4,
            8,
            args(
                top_m=1,
                topm_rank_space="retrieval",
            ),
        ).float()
        cache = {
            "host_image_features": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]),
            "retrieval_features": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.1, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]),
            "prototypes": torch.randn(3, 7, 4),
            "pids": torch.tensor([0, 1, 2]),
        }
        out = enricher(
            query_features=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            host_text_features=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            query_pids=torch.tensor([0]),
            pool_cache=cache,
            space="global",
        )
        self.assertTrue(torch.equal(out["top_indices"].cpu(), torch.tensor([[2]])))

    def test_extractor_mode_does_not_override_topm_rank_space(self):
        enricher = modules.TargetPrototypeEnricher(
            4,
            8,
            args(
                extractor_mode="retrieval_backbone",
                top_m=1,
                topm_rank_space="host_global",
            ),
        ).float()
        cache = {
            "host_image_features": torch.tensor([
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]),
            "retrieval_features": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]),
            "evidence_bank": torch.randn(2, 1, 4),
            "prototypes": torch.randn(2, 1, 4),
            "pids": torch.tensor([0, 1]),
        }
        out = enricher(
            query_features=torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True),
            host_text_features=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            query_pids=torch.tensor([0]),
            pool_cache=cache,
            space="global",
        )
        self.assertTrue(torch.equal(out["top_indices"].cpu(), torch.tensor([[1]])))

    def test_scalar_target_relative_evidence_is_projected_after_gather(self):
        enricher = modules.TargetPrototypeEnricher(
            4,
            8,
            args(
                extractor_mode="cluster_density",
                top_m=2,
            ),
        ).float()
        cache = {
            "host_image_features": torch.randn(3, 4),
            "retrieval_features": torch.randn(3, 4),
            "evidence_bank": torch.zeros(3, 1, 4),
            "prototypes": torch.zeros(3, 1, 4),
            "cluster_density_scalar": torch.tensor([[0.0], [1.0], [-1.0]]),
            "target_relative_cluster_ids": torch.tensor([0, 0, 1]),
            "pids": torch.tensor([0, 1, 2]),
        }
        out = enricher(
            query_features=torch.randn(2, 4, requires_grad=True),
            host_text_features=torch.randn(2, 4),
            query_pids=torch.tensor([0, 1]),
            pool_cache=cache,
            space="global",
        )
        out["enriched_features"].sum().backward()
        grad = enricher.scalar_evidence_projectors["cluster_density"].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_top_m_rank_space_hybrid_fuses_global_and_grab_scores(self):
        enricher = modules.TargetPrototypeEnricher(
            4,
            8,
            args(
                top_m=1,
                topm_rank_space="hybrid_global_grab",
                topm_rank_lambda=0.25,
            ),
        ).float()
        cache = {
            "host_image_features": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]),
            "retrieval_features": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]),
            "grab_image_features": torch.tensor([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            "prototypes": torch.randn(2, 7, 4),
            "pids": torch.tensor([0, 1]),
        }
        out = enricher(
            query_features=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            host_text_features=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            grab_text_features=torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            query_pids=torch.tensor([0]),
            pool_cache=cache,
            space="global",
        )
        self.assertTrue(torch.equal(out["top_indices"].cpu(), torch.tensor([[1]])))

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
        self.assertEqual(manager._interval_id(epoch=1, step=1), 0)
        self.assertEqual(manager._interval_id(epoch=99, step=99), 0)

        manager.args = SimpleNamespace(recompute_level="step", recompute_interval=3)
        self.assertEqual(manager._interval_id(epoch=1, step=6), 1)
        self.assertEqual(manager._interval_id(epoch=1, step=7), 2)

        manager.args = SimpleNamespace(recompute_level="step", recompute_interval=0)
        with self.assertRaisesRegex(ValueError, "--recompute_interval"):
            manager._interval_id(epoch=1, step=1)

    def test_full_training_set_cache_reused_for_multiple_steps(self):
        manager = make_pool_manager(recompute_interval=2)
        encode_calls = []

        def encode_records(model, records, cache_prototypes=True):
            encode_calls.append((len(records), cache_prototypes))
            return {
                "host_image_features": torch.zeros(len(records), 512),
                "retrieval_features": torch.zeros(len(records), 512),
                "prototypes": torch.zeros(len(records), 7, 512),
            }

        manager._encode_records = encode_records
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

        self.assertEqual(encode_calls, [(4, True)])
        self.assertEqual(set(cache1["image_ids"].tolist()), {10, 11, 12, 13})
        self.assertEqual(cache1["pids"].numel(), 4)
        self.assertNotIn("top_indices", cache1)
        self.assertEqual(cache1["diagnostics"]["pool_cache_size"], 4.0)
        self.assertEqual(cache1["diagnostics"]["pool_interval_reused"], 0.0)
        self.assertEqual(cache2["diagnostics"]["pool_interval_reused"], 1.0)
        self.assertEqual(cache2["diagnostics"]["pool_interval_id"], 0.0)

    def test_full_training_set_cache_refreshes_on_new_interval(self):
        manager = make_pool_manager(recompute_interval=2)
        encode_calls = []

        def encode_records(model, records, cache_prototypes=True):
            encode_calls.append((len(records), cache_prototypes))
            value = float(len(encode_calls))
            return {
                "host_image_features": torch.full((len(records), 512), value),
                "retrieval_features": torch.full((len(records), 512), value),
                "prototypes": torch.full((len(records), 7, 512), value),
            }

        manager._encode_records = encode_records
        batch = {
            "image_ids": torch.tensor([10]),
            "pids": torch.tensor([0]),
            "images": torch.randn(1, 3, 8, 8),
        }

        cache1 = manager.get_train_cache(FakeImageEncoder(), batch, epoch=1, step=1)
        cache2 = manager.get_train_cache(FakeImageEncoder(), batch, epoch=1, step=3)

        self.assertEqual(encode_calls, [(4, True), (4, True)])
        self.assertEqual(cache1["diagnostics"]["pool_interval_id"], 0.0)
        self.assertEqual(cache2["diagnostics"]["pool_interval_id"], 1.0)
        self.assertEqual(cache2["diagnostics"]["pool_interval_reused"], 0.0)
        self.assertFalse(torch.equal(cache1["host_image_features"], cache2["host_image_features"]))

    def test_frozen_indices_bypass_recompute_interval(self):
        manager = make_pool_manager(use_freeze_indices=True, top_m=2)
        manager.frozen_cache = {
            "host_image_features": torch.zeros(4, 512),
            "retrieval_features": torch.zeros(4, 512),
            "prototypes": torch.zeros(4, 7, 512),
            "image_ids": torch.tensor([10, 11, 12, 13]),
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
        self.assertEqual(cache2["diagnostics"]["pool_cache_size"], 4.0)
        self.assertEqual(cache2["diagnostics"]["frozen_indices_used"], 1.0)

    def test_frozen_index_build_keeps_full_cache_and_uses_top_m_depth(self):
        manager = make_pool_manager(use_freeze_indices=True, top_m=2)
        cache_prototype_calls = []

        def encode_records(model, records, cache_prototypes=True):
            cache_prototype_calls.append(cache_prototypes)
            host = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.6, 0.8, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.2, 0.98, 0.0, 0.0],
            ])
            cache = {"host_image_features": host}
            if cache_prototypes:
                cache["retrieval_features"] = host
                cache["prototypes"] = torch.zeros(len(records), 7, 4)
            return cache

        manager._encode_records = encode_records
        manager._encode_text_records = lambda model, records: torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.0, 0.0],
        ])

        manager._build_frozen_index_cache(FakeImageEncoder(embed_dim=4))

        self.assertEqual(cache_prototype_calls, [True])
        self.assertIsNotNone(manager.frozen_cache)
        self.assertIn("prototypes", manager.frozen_cache)
        self.assertIn("retrieval_features", manager.frozen_cache)
        self.assertTrue(torch.equal(
            manager._frozen_ranked_image_ids(torch.tensor([0]), top_n=2),
            torch.tensor([[10, 11]]),
        ))
        self.assertFalse(hasattr(manager, "frozen_query_features"))

    def test_frozen_index_build_uses_hybrid_rank_space(self):
        manager = make_pool_manager(
            use_freeze_indices=True,
            top_m=1,
            topm_rank_space="hybrid_global_grab",
            topm_rank_lambda=0.25,
        )

        def encode_records(model, records, cache_prototypes=True):
            host = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            return {
                "host_image_features": host,
                "retrieval_features": host,
                "grab_image_features": torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
                "prototypes": torch.zeros(len(records), 7, 4),
            }

        manager._encode_records = encode_records
        manager._encode_text_records = lambda model, records: torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        manager._encode_grab_text_records = lambda model, records: torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        manager._build_frozen_index_cache(FakeImageEncoder(embed_dim=4))

        self.assertTrue(torch.equal(
            manager._frozen_ranked_image_ids(torch.tensor([0]), top_n=1),
            torch.tensor([[11]]),
        ))
        self.assertTrue(torch.equal(
            manager._frozen_ranked_image_ids(torch.tensor([1]), top_n=1),
            torch.tensor([[10]]),
        ))


class ReproducibilityTests(unittest.TestCase):
    def test_identity_sampler_is_seeded_per_epoch(self):
        sampler_module = load_module("identity_sampler", "datasets/sampler.py")
        data = [
            (0, 0, "a.jpg", "a"),
            (0, 0, "b.jpg", "b"),
            (1, 1, "c.jpg", "c"),
            (1, 1, "d.jpg", "d"),
            (2, 2, "e.jpg", "e"),
            (2, 2, "f.jpg", "f"),
        ]

        first = sampler_module.RandomIdentitySampler(data, batch_size=4, num_instances=2, seed=99)
        first.set_epoch(3)
        order_a = list(iter(first))

        random.seed(12345)
        np.random.seed(12345)
        random.random()
        np.random.rand()

        second = sampler_module.RandomIdentitySampler(data, batch_size=4, num_instances=2, seed=99)
        second.set_epoch(3)
        order_b = list(iter(second))
        self.assertEqual(order_a, order_b)

        third = sampler_module.RandomIdentitySampler(data, batch_size=4, num_instances=2, seed=99)
        third.set_epoch(4)
        self.assertNotEqual(order_a, list(iter(third)))


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

    def test_removed_pool_cli_args_are_unrecognized(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        removed_args = [
            ["--use_" + "shared" + "_k"],
            ["--pool" + "_k", "4"],
            ["--pool" + "_k_mode", "static"],
            ["--pool" + "_k_candidates", "2,4"],
            ["--pool" + "_coverage_epochs", "15"],
            ["--pool" + "_clusters", "4"],
            ["--positive" + "_ratio" + "_max", "0.5"],
            ["--eta", "0.5"],
            ["--pool" + "_dist_metric", "l1"],
            ["--pool" + "_dist_threshold", "0.25"],
            ["--epsilon", "0.25"],
        ]
        try:
            for argv in removed_args:
                with self.subTest(argv=argv):
                    stderr = io.StringIO()
                    sys.argv = ["test"] + argv
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                        options.get_args()
                    self.assertIn("unrecognized arguments", stderr.getvalue())
        finally:
            sys.argv = old_argv

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
        self.assertEqual(parsed.context_module, "mixer")
        self.assertEqual(parsed.extractor_mode, "global,horizontal")
        self.assertEqual(parsed.context_pooling, "mlp")
        self.assertEqual(parsed.mixer_dim, 256)
        self.assertFalse(hasattr(parsed, "use_" + "shared" + "_k"))
        self.assertFalse(hasattr(parsed, "pool" + "_coverage_epochs"))
        self.assertEqual(parsed.residual_gate, "residual")
        self.assertIsNone(parsed.enrich_gamma)
        self.assertEqual(parsed.residual_gate_hidden_dim, 128)
        self.assertEqual(parsed.seed, 1)
        self.assertTrue(parsed.deterministic)
        self.assertFalse(parsed.deterministic_warn_only)
        self.assertFalse(hasattr(parsed, "use_target_retrieval_loss"))
        self.assertFalse(hasattr(parsed, "use_target_robust_loss"))
        self.assertFalse(parsed.pnp_text_only)
        self.assertEqual(parsed.topm_rank_space, "host_global")
        self.assertEqual(parsed.topm_rank_lambda, 0.5)
        self.assertEqual(parsed.target_relative_space, "host_global")
        self.assertEqual(parsed.target_relative_num_clusters, 16)
        self.assertEqual(parsed.target_relative_cluster_method, "kmeans")
        self.assertEqual(parsed.evidence_token_budget, 0)
        self.assertEqual(parsed.evidence_projection, "auto")
        self.assertEqual(parsed.wandb_project, "enrichment")
        self.assertFalse(parsed.delete_checkpoints_after_run)

    def test_delete_checkpoints_after_run_cli_parses(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--delete_checkpoints_after_run"]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertTrue(parsed.delete_checkpoints_after_run)

    def test_wandb_project_cli_parses_and_validates_name(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--wandb_project", "custom-project"]
            parsed = options.get_args()
            self.assertEqual(parsed.wandb_project, "custom-project")

            sys.argv = ["test", "--wandb_project", "   "]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
        finally:
            sys.argv = old_argv

    def test_residual_gate_cli_parses_static_gamma_and_rejects_residual_gamma(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--residual_gate",
                "static",
                "--enrich_gamma",
                "0.2",
                "--residual_gate_hidden_dim",
                "32",
            ]
            parsed = options.get_args()
            self.assertEqual(parsed.residual_gate, "static")
            self.assertEqual(parsed.enrich_gamma, 0.2)
            self.assertEqual(parsed.residual_gate_hidden_dim, 32)

            sys.argv = ["test", "--enrich_gamma", "0.2"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()

            sys.argv = ["test", "--residual_gate", "static"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()

            sys.argv = ["test", "--residual_gate", "residual", "--enrich_gamma", "0.2"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
        finally:
            sys.argv = old_argv

    def test_context_pooling_cli_rejects_removed_attention_modes(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            for argv in (
                ["--context_pooling", "late_attention"],
                ["--mixer_context_pooling", "hybrid_attention"],
            ):
                with self.subTest(argv=argv):
                    sys.argv = ["test", *argv]
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        options.get_args()
        finally:
            sys.argv = old_argv

    def test_extractor_mode_cli_parses_mode_and_validates_num_parts(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--extractor_mode",
                "global,grid,vertical,horizontal,retrieval_backbone,cluster,cluster_residual,cluster_density,cluster_rarity",
                "--num_parts",
                "3",
                "--target_relative_space",
                "retrieval",
                "--target_relative_num_clusters",
                "8",
                "--evidence_token_budget",
                "64",
                "--evidence_projection",
                "linear",
            ]
            parsed = options.get_args()
            self.assertEqual(
                parsed.extractor_mode,
                "global,grid,vertical,horizontal,retrieval_backbone,cluster,cluster_residual,cluster_density,cluster_rarity",
            )
            self.assertEqual(parsed.num_parts, 3)
            self.assertEqual(parsed.target_relative_space, "retrieval")
            self.assertEqual(parsed.target_relative_num_clusters, 8)
            self.assertEqual(parsed.evidence_token_budget, 64)
            self.assertEqual(parsed.evidence_projection, "linear")

            sys.argv = ["test", "--extractor_mode", "global_grid"]
            parsed = options.get_args()
            self.assertEqual(parsed.extractor_mode, "global,grid")

            sys.argv = ["test", "--extractor_mode", "global,diagonal"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()

            sys.argv = ["test", "--num_parts", "0"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()

            sys.argv = ["test", "--extractor_mode", "grid", "--num_parts", "3", "--evidence_token_budget", "8"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
        finally:
            sys.argv = old_argv

    def test_topm_rank_space_cli_parses_and_validates_lambda(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--topm_rank_space",
                "hybrid_global_grab",
                "--topm_rank_lambda",
                "0.3",
            ]
            parsed = options.get_args()
            self.assertEqual(parsed.topm_rank_space, "hybrid_global_grab")
            self.assertEqual(parsed.topm_rank_lambda, 0.3)

            sys.argv = ["test", "--topm_rank_lambda", "1.1"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()

            sys.argv = [
                "test",
                "--target_enrichment",
                "--only_global",
                "--topm_rank_space",
                "hybrid_global_grab",
            ]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
        finally:
            sys.argv = old_argv

    def test_pnp_text_only_cli_requires_frozen_global_no_host_config(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--pnp_text_only",
                "--target_enrichment",
                "--freeze_host",
                "--only_global",
                "--use_freeze_indices",
            ]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
            sys.argv = [
                "test",
                "--pnp_text_only",
                "--target_enrichment",
                "--freeze_host",
                "--no_use_host_loss",
                "--only_global",
                "--use_freeze_indices",
            ]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertTrue(parsed.pnp_text_only)
        self.assertFalse(parsed.use_host_loss)
        self.assertEqual(parsed.enrichment_space, "global")

    def test_reproducibility_cli_parses_seed_and_determinism(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = [
                "test",
                "--seed",
                "123",
                "--non_deterministic",
                "--deterministic_warn_only",
            ]
            parsed = options.get_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(parsed.seed, 123)
        self.assertFalse(parsed.deterministic)
        self.assertTrue(parsed.deterministic_warn_only)

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

    def test_host_loss_can_be_disabled_and_removed_target_loss_flags_are_rejected(self):
        options = importlib.import_module("utils.options")
        old_argv = sys.argv
        try:
            sys.argv = ["test", "--no_use_host_loss"]
            parsed = options.get_args()

            removed_flags = [
                ["--use_target_retrieval_loss"],
                ["--use_target_robust_loss"],
                ["--hard_neg_k", "1"],
                ["--lambda_rob", "0.1"],
                ["--lambda_gain", "1.0"],
                ["--gain_margin", "0.01"],
            ]
            for flag_args in removed_flags:
                sys.argv = ["test", *flag_args]
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    options.get_args()

            sys.argv = ["test", "--lambda_ret", "0"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                options.get_args()
        finally:
            sys.argv = old_argv
        self.assertFalse(parsed.use_host_loss)

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
