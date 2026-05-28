import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixer import RankPartQueryConditionedMixerAdapter, _FusionMLP
from .prototypes import prototype_slot_count


def _masked_logsumexp(values, mask, dim):
    neg_inf = torch.finfo(values.dtype).min
    return torch.logsumexp(values.masked_fill(~mask, neg_inf), dim=dim)


class _ResidualGateMLP(nn.Module):
    def __init__(self, dim, hidden_dim, initial_value):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("--residual_gate_hidden_dim must be a positive integer")
        initial_value = min(max(float(initial_value), 1e-4), 1.0 - 1e-4)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(initial_value / (1.0 - initial_value)))

    def forward(self, query, context):
        gate_input = torch.cat([query, context, query * context], dim=-1)
        return torch.sigmoid(self.net(gate_input))


class TargetPrototypeEnricher(nn.Module):
    def __init__(self, embed_dim, grab_embed_dim, args):
        super().__init__()
        self.embed_dim = embed_dim
        self.grab_embed_dim = grab_embed_dim
        if args.top_m < 1:
            raise ValueError("--top_m must be a positive integer")
        self.top_m = args.top_m
        self.robust_hard_k = getattr(args, "robust_hard_k", self.top_m)
        if self.robust_hard_k < 1:
            raise ValueError("--robust_hard_k must be a positive integer")
        self.gamma = args.enrich_gamma
        self.residual_gate_mode = getattr(args, "residual_gate", "static")
        if self.residual_gate_mode not in ("static", "residual"):
            raise ValueError("--residual_gate must be either 'static' or 'residual'")
        if self.residual_gate_mode == "residual" and not (0 < float(self.gamma) < 1):
            raise ValueError("--enrich_gamma must be in (0, 1) when --residual_gate is residual")
        self.tau = args.tau
        self.lambda_ret = getattr(args, "lambda_ret", 1.0)
        self.lambda_rob = args.lambda_rob
        self.lambda_gain = args.lambda_gain
        self.gain_margin = args.gain_margin
        self.use_target_retrieval_loss = getattr(args, "use_target_retrieval_loss", False)
        self.context_module = getattr(args, "context_module", "mixer")
        if self.context_module != "mixer":
            raise ValueError("--context_module must be mixer; attention context construction has been removed")
        self.use_target_attention_loss = False
        self.use_target_robust_loss = getattr(args, "use_target_robust_loss", False)
        self.enrichment_space = getattr(args, "enrichment_space", "global")
        if self.enrichment_space not in ("global", "grab"):
            raise ValueError("--enrichment_space must be either 'global' or 'grab'")
        if self.enrichment_space == "grab" and getattr(args, "only_global", False):
            raise ValueError("--enrichment_space grab requires GRAB features; remove --only_global")
        self.enable_global = self.enrichment_space == "global"
        self.enable_grab = self.enrichment_space == "grab"

        self.extractor_mode = getattr(args, "extractor_mode", "global_horizontal")
        self.num_parts = getattr(args, "num_parts", 6)
        num_slots = prototype_slot_count(self.extractor_mode, self.num_parts)
        mixer_kwargs = dict(
            num_ranks=self.top_m,
            num_slots=num_slots,
            mixer_dim=getattr(args, "mixer_dim", 256),
            depth=getattr(args, "mixer_depth", 2),
            hidden_part=getattr(args, "mixer_hidden_part", 32),
            hidden_rank=getattr(args, "mixer_hidden_rank", 64),
            hidden_channel=getattr(args, "mixer_hidden_channel", 512),
            hidden_readout=getattr(args, "mixer_hidden_readout", 128),
        )
        if self.enable_global:
            self.global_context = RankPartQueryConditionedMixerAdapter(embed_dim, **mixer_kwargs)
            self.global_fusion = _FusionMLP(embed_dim)
            if self.residual_gate_mode == "residual":
                self.global_residual_gate = _ResidualGateMLP(
                    embed_dim,
                    getattr(args, "residual_gate_hidden_dim", 128),
                    self.gamma,
                )

        if self.enable_grab:
            self.proto_to_grab = nn.Linear(embed_dim, grab_embed_dim)
            self.grab_context = RankPartQueryConditionedMixerAdapter(grab_embed_dim, **mixer_kwargs)
            self.grab_fusion = _FusionMLP(grab_embed_dim)
            if self.residual_gate_mode == "residual":
                self.grab_residual_gate = _ResidualGateMLP(
                    grab_embed_dim,
                    getattr(args, "residual_gate_hidden_dim", 128),
                    self.gamma,
                )

    def _require_global(self):
        if not self.enable_global:
            raise ValueError("Global enrichment is disabled because --enrichment_space is set to grab")

    def _require_grab(self):
        if not self.enable_grab:
            raise ValueError(
                "GRAB enrichment is disabled because --enrichment_space is not grab "
                "or --only_global is enabled"
            )

    def _project_prototypes(self, prototypes, space):
        if space == "grab":
            self._require_grab()
            prototypes = self.proto_to_grab(prototypes.float())
        return F.normalize(prototypes.float(), p=2, dim=-1)

    def _context(self, query_features, selected_prototypes, space):
        if space == "grab":
            self._require_grab()
            return self.grab_context(query_features.float(), selected_prototypes.float())
        self._require_global()
        return self.global_context(query_features.float(), selected_prototypes.float())

    def _context_diagnostics(self, space):
        if space == "grab":
            self._require_grab()
            return dict(self.grab_context.last_diagnostics)
        self._require_global()
        return dict(self.global_context.last_diagnostics)

    def _fusion_delta(self, query_features, context, space):
        if space == "grab":
            self._require_grab()
            delta = self.grab_fusion(query_features.float(), context.float())
        else:
            self._require_global()
            delta = self.global_fusion(query_features.float(), context.float())
        return delta

    def _residual_gate(self, query_features, context, space):
        if self.residual_gate_mode == "static":
            return query_features.new_full((query_features.shape[0], 1), float(self.gamma))
        if space == "grab":
            self._require_grab()
            return self.grab_residual_gate(query_features.float(), context.float())
        self._require_global()
        return self.global_residual_gate(query_features.float(), context.float())

    def _fuse(self, query_features, context, space):
        delta = self._fusion_delta(query_features, context, space)
        residual_gate = self._residual_gate(query_features, context, space)
        return F.normalize(query_features.float() + residual_gate * delta, p=2, dim=-1)

    def _fuse_with_delta(self, query_features, context, space):
        delta = self._fusion_delta(query_features, context, space)
        residual_gate = self._residual_gate(query_features, context, space)
        enriched = F.normalize(query_features.float() + residual_gate * delta, p=2, dim=-1)
        return enriched, delta, residual_gate

    def _top_indices(self, host_text_features, host_image_features, pool_cache):
        supplied = pool_cache.get("top_indices")
        if supplied is not None:
            top_indices = supplied.long()
            if top_indices.dim() != 2:
                raise ValueError("pool_cache['top_indices'] must have shape [batch, top_m]")
            if top_indices.shape[0] != host_text_features.shape[0]:
                raise ValueError("pool_cache['top_indices'] batch size must match query batch size")
            top_m = min(self.top_m, host_image_features.shape[0], top_indices.shape[1])
            if top_m < 1:
                raise ValueError("pool_cache['top_indices'] must contain at least one column")
            top_indices = top_indices[:, :top_m].to(host_image_features.device)
            if int(top_indices.min().item()) < 0 or int(top_indices.max().item()) >= host_image_features.shape[0]:
                raise ValueError("pool_cache['top_indices'] contains indices outside the target pool")
            return top_indices

        with torch.no_grad():
            host_scores = host_text_features @ host_image_features.t()
            top_m = min(self.top_m, host_image_features.shape[0])
            return host_scores.topk(k=top_m, dim=1, largest=True, sorted=True).indices

    def forward(self, query_features, host_text_features, query_pids, pool_cache, space):
        host_image_features = F.normalize(pool_cache["host_image_features"].float(), p=2, dim=-1)
        retrieval_features = F.normalize(pool_cache["retrieval_features"].float(), p=2, dim=-1)
        prototypes = pool_cache["prototypes"].float()
        pool_pids = pool_cache["pids"].long()

        host_text_features = F.normalize(host_text_features.float(), p=2, dim=-1)
        top_indices = self._top_indices(host_text_features, host_image_features, pool_cache)

        gathered = prototypes[top_indices]
        selected_prototypes = self._project_prototypes(gathered, space)

        normalized_query = F.normalize(query_features.float(), p=2, dim=-1)
        context = self._context(normalized_query, selected_prototypes, space)
        enriched, delta, residual_gate = self._fuse_with_delta(normalized_query, context, space)

        losses = self.compute_losses(
            raw_query=normalized_query,
            enriched_query=enriched,
            retrieval_features=retrieval_features,
            top_indices=top_indices,
            query_pids=query_pids.long(),
            pool_pids=pool_pids,
        )
        diagnostics = self.compute_diagnostics(
            raw_query=normalized_query,
            enriched_query=enriched,
            context=context,
            host_text_features=host_text_features,
            host_image_features=host_image_features,
            retrieval_features=retrieval_features,
            delta=delta,
            residual_gate=residual_gate,
            top_indices=top_indices,
            query_pids=query_pids.long(),
            pool_pids=pool_pids,
            mixer_diagnostics=self._context_diagnostics(space),
        )

        return {
            "enriched_features": enriched,
            "top_indices": top_indices,
            **losses,
            **diagnostics,
        }

    def enrich_only(self, query_features, host_text_features, pool_cache, space):
        host_image_features = F.normalize(pool_cache["host_image_features"].float(), p=2, dim=-1)
        prototypes = pool_cache["prototypes"].float()
        host_text_features = F.normalize(host_text_features.float(), p=2, dim=-1)
        top_indices = self._top_indices(host_text_features, host_image_features, pool_cache)

        gathered = prototypes[top_indices]
        selected_prototypes = self._project_prototypes(gathered, space)
        normalized_query = F.normalize(query_features.float(), p=2, dim=-1)
        context = self._context(normalized_query, selected_prototypes, space)
        return self._fuse(normalized_query, context, space)

    def compute_losses(
        self,
        raw_query,
        enriched_query,
        retrieval_features,
        top_indices,
        query_pids,
        pool_pids,
    ):
        zero = enriched_query.sum() * 0.0
        target_retrieval_loss = zero
        att_loss = zero
        robust_loss = zero
        guard_loss = zero
        gain_loss = zero

        positive_mask = None
        if self.use_target_retrieval_loss or self.use_target_robust_loss:
            positive_mask = query_pids.view(-1, 1).eq(pool_pids.view(1, -1))

        if self.use_target_retrieval_loss:
            valid_positive = positive_mask.any(dim=1)
            if valid_positive.any():
                retrieval_scores = enriched_query @ retrieval_features.t() / max(self.tau, 1e-6)
                pos_lse = _masked_logsumexp(retrieval_scores, positive_mask, dim=1)
                all_lse = torch.logsumexp(retrieval_scores, dim=1)
                target_retrieval_loss = -(pos_lse[valid_positive] - all_lse[valid_positive]).mean()

        top_positive = None
        if self.use_target_robust_loss:
            top_pids = pool_pids[top_indices]
            top_positive = top_pids.eq(query_pids.view(-1, 1))

        if self.use_target_robust_loss:
            robust_loss, guard_loss, gain_loss = self._compute_robust_loss(
                raw_query=raw_query,
                enriched_query=enriched_query,
                retrieval_features=retrieval_features,
                positive_mask=positive_mask,
                reliable=top_positive.any(dim=1),
                zero=zero,
            )
        total = zero
        if self.use_target_retrieval_loss:
            total = total + self.lambda_ret * target_retrieval_loss
        if self.use_target_robust_loss:
            total = total + self.lambda_rob * robust_loss
        return {
            "target_retrieval_loss": target_retrieval_loss,
            "att_loss": att_loss,
            "robust_loss": robust_loss,
            "guard_loss": guard_loss,
            "gain_loss": gain_loss,
            "total_loss": total,
        }

    def compute_diagnostics(
        self,
        raw_query,
        enriched_query,
        context,
        host_text_features,
        host_image_features,
        retrieval_features,
        delta,
        residual_gate,
        top_indices,
        query_pids,
        pool_pids,
        mixer_diagnostics,
    ):
        with torch.no_grad():
            zero = raw_query.new_tensor(0.0)
            top_m = top_indices.shape[1]
            top_pids = pool_pids[top_indices]
            top_positive = top_pids.eq(query_pids.view(-1, 1))
            positive_mask = query_pids.view(-1, 1).eq(pool_pids.view(1, -1))
            negative_mask = ~positive_mask

            positive_in_pool = positive_mask.any(dim=1)
            positive_in_topm = top_positive.any(dim=1)
            top_positive_count = top_positive.sum(dim=1).float()
            pool_positive_count = positive_mask.sum(dim=1).float()
            host_topm_recall = top_positive_count / pool_positive_count.clamp_min(1.0)

            ranks = torch.arange(1, top_m + 1, device=top_indices.device).view(1, -1)
            absent_rank = raw_query.new_full((top_indices.shape[0], top_m), float(top_m + 1))
            first_rank = torch.where(top_positive, ranks.float(), absent_rank).min(dim=1).values
            if positive_in_topm.any():
                first_rank_when_present = first_rank[positive_in_topm].mean()
            else:
                first_rank_when_present = zero

            host_scores = host_text_features @ host_image_features.t()
            selected_scores = host_scores.gather(1, top_indices)
            host_topm_gap = selected_scores[:, 0] - selected_scores[:, -1]

            raw_enriched_cosine = (raw_query * enriched_query).sum(dim=1)
            raw_context_cosine = (raw_query * F.normalize(context.float(), p=2, dim=-1)).sum(dim=1)
            enrichment_shift = (enriched_query - raw_query).norm(dim=1)
            context_delta_cosine = F.cosine_similarity(context.float(), delta.float(), dim=-1)
            residual_gate = residual_gate.detach().float()

            diagnostics = {
                "target_positive_in_pool_rate": positive_in_pool.float().mean(),
                "target_positive_in_topm_rate": positive_in_topm.float().mean(),
                "target_num_positive_in_pool": pool_positive_count.mean(),
                "target_num_positive_in_topm": top_positive_count.mean(),
                "target_host_topm_recall": host_topm_recall.mean(),
                "target_first_positive_rank": first_rank_when_present,
                "target_first_positive_rank_with_absent": first_rank.mean(),
                "target_missing_topm_rate": (~positive_in_topm).float().mean(),
                "target_host_top1_score": selected_scores[:, 0].mean(),
                "target_host_topm_score": selected_scores.mean(),
                "target_host_top1_topm_gap": host_topm_gap.mean(),
                "target_raw_enriched_cosine": raw_enriched_cosine.mean(),
                "target_raw_context_cosine": raw_context_cosine.mean(),
                "target_context_norm": context.norm(dim=1).mean(),
                "target_enrichment_shift_norm": enrichment_shift.mean(),
                "target_residual_gate_mean": residual_gate.mean(),
                "target_residual_gate_std": residual_gate.std(unbiased=False),
                "target_residual_gate_min": residual_gate.min(),
                "target_residual_gate_max": residual_gate.max(),
                "mixer/context_delta_cosine": context_delta_cosine.mean(),
                "mixer/output_delta_norm": delta.norm(dim=1).mean(),
            }
            diagnostics.update(mixer_diagnostics)

            valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
            if valid.any():
                raw_scores = raw_query @ retrieval_features.t()
                enriched_scores = enriched_query @ retrieval_features.t()
                hard_k = min(self.robust_hard_k, retrieval_features.shape[0])
                hard_indices = raw_scores.masked_fill(positive_mask, torch.finfo(raw_scores.dtype).min).topk(
                    k=hard_k, dim=1, largest=True, sorted=True
                ).indices
                hard_mask = negative_mask.gather(1, hard_indices)

                raw_pos = _masked_logsumexp(raw_scores, positive_mask, dim=1)
                enr_pos = _masked_logsumexp(enriched_scores, positive_mask, dim=1)
                raw_hard = _masked_logsumexp(raw_scores.gather(1, hard_indices), hard_mask, dim=1)
                enr_hard = _masked_logsumexp(enriched_scores.gather(1, hard_indices), hard_mask, dim=1)

                raw_margin = raw_pos - raw_hard
                enriched_margin = enr_pos - enr_hard
                margin_gain = enriched_margin - raw_margin
                reliable_valid = positive_in_topm & valid

                diagnostics.update({
                    "target_valid_robust_rate": valid.float().mean(),
                    "target_raw_margin": raw_margin[valid].mean(),
                    "target_enriched_margin": enriched_margin[valid].mean(),
                    "target_margin_gain": margin_gain[valid].mean(),
                    "target_guard_violation_rate": (raw_margin[valid] > enriched_margin[valid]).float().mean(),
                    "target_reliable_robust_rate": reliable_valid.float().mean(),
                    "target_margin_gain_reliable": margin_gain[reliable_valid].mean()
                    if reliable_valid.any() else zero,
                    "target_gain_satisfied_rate": (margin_gain[reliable_valid] >= self.gain_margin).float().mean()
                    if reliable_valid.any() else zero,
                })
            else:
                diagnostics.update({
                    "target_valid_robust_rate": zero,
                    "target_raw_margin": zero,
                    "target_enriched_margin": zero,
                    "target_margin_gain": zero,
                    "target_guard_violation_rate": zero,
                    "target_reliable_robust_rate": zero,
                    "target_margin_gain_reliable": zero,
                    "target_gain_satisfied_rate": zero,
                })

            return diagnostics

    def _compute_robust_loss(self, raw_query, enriched_query, retrieval_features, positive_mask, reliable, zero):
        negative_mask = ~positive_mask
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return zero, zero, zero

        raw_scores = raw_query @ retrieval_features.t()
        enriched_scores = enriched_query @ retrieval_features.t()
        hard_k = min(self.robust_hard_k, retrieval_features.shape[0])
        hard_indices = raw_scores.masked_fill(positive_mask, torch.finfo(raw_scores.dtype).min).topk(
            k=hard_k, dim=1, largest=True, sorted=True
        ).indices
        hard_mask = negative_mask.gather(1, hard_indices)

        raw_pos = _masked_logsumexp(raw_scores, positive_mask, dim=1)
        enr_pos = _masked_logsumexp(enriched_scores, positive_mask, dim=1)
        raw_hard = _masked_logsumexp(raw_scores.gather(1, hard_indices), hard_mask, dim=1)
        enr_hard = _masked_logsumexp(enriched_scores.gather(1, hard_indices), hard_mask, dim=1)

        raw_margin = raw_pos - raw_hard
        enriched_margin = enr_pos - enr_hard
        guard_loss = F.relu(raw_margin[valid] - enriched_margin[valid]).mean()

        reliable_valid = reliable & valid
        if reliable_valid.any():
            gain_loss = F.relu(
                self.gain_margin - (enriched_margin[reliable_valid] - raw_margin[reliable_valid])
            ).mean()
        else:
            gain_loss = zero
        robust_loss = guard_loss + self.lambda_gain * gain_loss
        return robust_loss, guard_loss, gain_loss
