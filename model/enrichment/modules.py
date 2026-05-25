import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_part_prototypes(token_features, num_parts, grid_size=None):
    token_features = token_features.float()
    global_feature = F.normalize(token_features[:, 0, :], p=2, dim=-1)
    patch_features = token_features[:, 1:, :]
    batch_size, num_patches, dim = patch_features.shape

    parts = []
    if grid_size is not None:
        grid_h, grid_w = grid_size
    else:
        grid_h, grid_w = None, None

    if grid_h is not None and grid_w is not None and grid_h * grid_w == num_patches:
        patch_grid = patch_features.reshape(batch_size, grid_h, grid_w, dim)
        boundaries = torch.linspace(0, grid_h, steps=num_parts + 1, device=patch_features.device)
        boundaries = boundaries.round().long().tolist()
        for part_idx in range(num_parts):
            start = boundaries[part_idx]
            end = boundaries[part_idx + 1]
            if end <= start:
                end = min(start + 1, grid_h)
                start = max(0, end - 1)
            part = patch_grid[:, start:end, :, :].mean(dim=(1, 2))
            parts.append(F.normalize(part, p=2, dim=-1))
    else:
        boundaries = torch.linspace(0, num_patches, steps=num_parts + 1, device=patch_features.device)
        boundaries = boundaries.round().long().tolist()
        for part_idx in range(num_parts):
            start = boundaries[part_idx]
            end = boundaries[part_idx + 1]
            if end <= start:
                end = min(start + 1, num_patches)
                start = max(0, end - 1)
            part = patch_features[:, start:end, :].mean(dim=1)
            parts.append(F.normalize(part, p=2, dim=-1))

    return torch.stack([global_feature] + parts, dim=1)


def _masked_logsumexp(values, mask, dim):
    neg_inf = torch.finfo(values.dtype).min
    return torch.logsumexp(values.masked_fill(~mask, neg_inf), dim=dim)


class _FusionMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, query, context):
        return self.net(torch.cat([query, context, query * context], dim=-1))


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
        self.tau = args.tau
        self.lambda_att = args.lambda_att
        self.lambda_ret = getattr(args, "lambda_ret", 1.0)
        self.lambda_rob = args.lambda_rob
        self.lambda_gain = args.lambda_gain
        self.att_margin = args.att_margin
        self.gain_margin = args.gain_margin
        self.use_target_retrieval_loss = getattr(args, "use_target_retrieval_loss", False)
        self.use_target_attention_loss = getattr(args, "use_target_attention_loss", False)
        self.use_target_robust_loss = getattr(args, "use_target_robust_loss", False)
        self.enrichment_space = getattr(args, "enrichment_space", "global")
        if self.enrichment_space not in ("global", "grab"):
            raise ValueError("--enrichment_space must be either 'global' or 'grab'")
        if self.enrichment_space == "grab" and getattr(args, "only_global", False):
            raise ValueError("--enrichment_space grab requires GRAB features; remove --only_global")
        self.enable_global = self.enrichment_space == "global"
        self.enable_grab = self.enrichment_space == "grab"

        att_dim = embed_dim
        if self.enable_global:
            self.global_query_proj = nn.Linear(embed_dim, att_dim)
            self.global_proto_proj = nn.Linear(embed_dim, att_dim)
            self.global_fusion = _FusionMLP(embed_dim)

        if self.enable_grab:
            self.proto_to_grab = nn.Linear(embed_dim, grab_embed_dim)
            self.grab_query_proj = nn.Linear(grab_embed_dim, att_dim)
            self.grab_proto_proj = nn.Linear(grab_embed_dim, att_dim)
            self.grab_fusion = _FusionMLP(grab_embed_dim)

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

    def _attention(self, query_features, selected_prototypes, space):
        if space == "grab":
            self._require_grab()
            query_proj = self.grab_query_proj(query_features.float())
            proto_proj = self.grab_proto_proj(selected_prototypes.float())
        else:
            self._require_global()
            query_proj = self.global_query_proj(query_features.float())
            proto_proj = self.global_proto_proj(selected_prototypes.float())

        scale = math.sqrt(query_proj.shape[-1])
        scores = torch.bmm(proto_proj, query_proj.unsqueeze(-1)).squeeze(-1) / scale
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), selected_prototypes).squeeze(1)
        return context, weights

    def _fuse(self, query_features, context, space):
        if space == "grab":
            self._require_grab()
            delta = self.grab_fusion(query_features.float(), context.float())
        else:
            self._require_global()
            delta = self.global_fusion(query_features.float(), context.float())
        return F.normalize(query_features.float() + self.gamma * delta, p=2, dim=-1)

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
        batch_size, top_m, num_proto, _ = gathered.shape
        selected_prototypes = self._project_prototypes(gathered, space)
        flat_prototypes = selected_prototypes.reshape(batch_size, top_m * num_proto, -1)

        normalized_query = F.normalize(query_features.float(), p=2, dim=-1)
        context, weights = self._attention(normalized_query, flat_prototypes, space)
        enriched = self._fuse(normalized_query, context, space)
        attention_weights = weights.reshape(batch_size, top_m, num_proto)

        losses = self.compute_losses(
            raw_query=normalized_query,
            enriched_query=enriched,
            retrieval_features=retrieval_features,
            selected_prototypes=selected_prototypes,
            attention_weights=attention_weights,
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
            attention_weights=attention_weights,
            top_indices=top_indices,
            query_pids=query_pids.long(),
            pool_pids=pool_pids,
        )

        return {
            "enriched_features": enriched,
            "top_indices": top_indices,
            "attention_weights": weights,
            **losses,
            **diagnostics,
        }

    def enrich_only(self, query_features, host_text_features, pool_cache, space):
        host_image_features = F.normalize(pool_cache["host_image_features"].float(), p=2, dim=-1)
        prototypes = pool_cache["prototypes"].float()
        host_text_features = F.normalize(host_text_features.float(), p=2, dim=-1)
        top_indices = self._top_indices(host_text_features, host_image_features, pool_cache)

        gathered = prototypes[top_indices]
        batch_size, top_m, num_proto, _ = gathered.shape
        selected_prototypes = self._project_prototypes(gathered, space)
        flat_prototypes = selected_prototypes.reshape(batch_size, top_m * num_proto, -1)
        normalized_query = F.normalize(query_features.float(), p=2, dim=-1)
        context, _ = self._attention(normalized_query, flat_prototypes, space)
        return self._fuse(normalized_query, context, space)

    def compute_losses(
        self,
        raw_query,
        enriched_query,
        retrieval_features,
        selected_prototypes,
        attention_weights,
        top_indices,
        query_pids,
        pool_pids,
    ):
        zero = enriched_query.sum() * 0.0
        target_loss = zero
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
                target_loss = -(pos_lse[valid_positive] - all_lse[valid_positive]).mean()

        top_positive = None
        if self.use_target_attention_loss or self.use_target_robust_loss:
            top_pids = pool_pids[top_indices]
            top_positive = top_pids.eq(query_pids.view(-1, 1))

        if self.use_target_attention_loss:
            top_negative = ~top_positive
            valid_attention = top_positive.any(dim=1) & top_negative.any(dim=1)
            if valid_attention.any():
                proto_sims = (
                    raw_query.view(raw_query.shape[0], 1, 1, -1) * selected_prototypes
                ).sum(dim=-1)
                evidence = (attention_weights * proto_sims).sum(dim=-1)
                pos_evidence = _masked_logsumexp(evidence, top_positive, dim=1)
                neg_evidence = _masked_logsumexp(evidence, top_negative, dim=1)
                att_loss = F.softplus(
                    neg_evidence[valid_attention]
                    - pos_evidence[valid_attention]
                    + self.att_margin
                ).mean()

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
            total = total + self.lambda_ret * target_loss
        if self.use_target_attention_loss:
            total = total + self.lambda_att * att_loss
        if self.use_target_robust_loss:
            total = total + self.lambda_rob * robust_loss
        return {
            "target_loss": target_loss,
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
        attention_weights,
        top_indices,
        query_pids,
        pool_pids,
    ):
        with torch.no_grad():
            zero = raw_query.new_tensor(0.0)
            top_m = top_indices.shape[1]
            num_proto = attention_weights.shape[2]
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

            image_attention = attention_weights.sum(dim=2)
            pos_attention = (image_attention * top_positive.float()).sum(dim=1)
            neg_attention = (image_attention * (~top_positive).float()).sum(dim=1)
            if positive_in_topm.any():
                pos_attention_when_present = pos_attention[positive_in_topm].mean()
            else:
                pos_attention_when_present = zero

            flat_attention = attention_weights.reshape(attention_weights.shape[0], -1).clamp_min(1e-12)
            attention_entropy = -(flat_attention * flat_attention.log()).sum(dim=1)
            attention_entropy_norm = attention_entropy / max(math.log(flat_attention.shape[1]), 1e-12)

            host_scores = host_text_features @ host_image_features.t()
            selected_scores = host_scores.gather(1, top_indices)
            host_topm_gap = selected_scores[:, 0] - selected_scores[:, -1]

            raw_enriched_cosine = (raw_query * enriched_query).sum(dim=1)
            raw_context_cosine = (raw_query * F.normalize(context.float(), p=2, dim=-1)).sum(dim=1)
            enrichment_shift = (enriched_query - raw_query).norm(dim=1)

            diagnostics = {
                "target_positive_in_pool_rate": positive_in_pool.float().mean(),
                "target_positive_in_topm_rate": positive_in_topm.float().mean(),
                "target_num_positive_in_pool": pool_positive_count.mean(),
                "target_num_positive_in_topm": top_positive_count.mean(),
                "target_host_topm_recall": host_topm_recall.mean(),
                "target_first_positive_rank": first_rank_when_present,
                "target_first_positive_rank_with_absent": first_rank.mean(),
                "target_missing_topm_rate": (~positive_in_topm).float().mean(),
                "target_attention_mass_positive": pos_attention.mean(),
                "target_attention_mass_positive_when_present": pos_attention_when_present,
                "target_attention_mass_negative": neg_attention.mean(),
                "target_attention_pos_minus_neg": (pos_attention - neg_attention).mean(),
                "target_attention_entropy": attention_entropy.mean(),
                "target_attention_entropy_norm": attention_entropy_norm.mean(),
                "target_attention_top1_mass": flat_attention.max(dim=1).values.mean(),
                "target_attention_mass_image_proto": attention_weights[:, :, 0].sum(dim=1).mean(),
                "target_attention_mass_part_proto": attention_weights[:, :, 1:].sum(dim=(1, 2)).mean()
                if num_proto > 1 else zero,
                "target_host_top1_score": selected_scores[:, 0].mean(),
                "target_host_topm_score": selected_scores.mean(),
                "target_host_top1_topm_gap": host_topm_gap.mean(),
                "target_raw_enriched_cosine": raw_enriched_cosine.mean(),
                "target_raw_context_cosine": raw_context_cosine.mean(),
                "target_context_norm": context.norm(dim=1).mean(),
                "target_enrichment_shift_norm": enrichment_shift.mean(),
            }

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
