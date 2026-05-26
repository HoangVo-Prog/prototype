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


def _two_layer_mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class _RankPartMixerBlock(nn.Module):
    def __init__(self, num_ranks, num_slots, mixer_dim, hidden_part, hidden_rank, hidden_channel):
        super().__init__()
        self.part_norm = nn.LayerNorm(mixer_dim)
        self.part_mlp = _two_layer_mlp(num_slots, hidden_part, num_slots)
        self.rank_norm = nn.LayerNorm(mixer_dim)
        self.rank_mlp = _two_layer_mlp(num_ranks, hidden_rank, num_ranks)
        self.channel_norm = nn.LayerNorm(mixer_dim)
        self.channel_mlp = _two_layer_mlp(mixer_dim, hidden_channel, mixer_dim)

    def forward(self, x, rank_mask=None):
        part_delta = self.part_mlp(self.part_norm(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = x + part_delta
        if rank_mask is not None:
            x = x * rank_mask

        rank_delta = self.rank_mlp(self.rank_norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + rank_delta
        if rank_mask is not None:
            x = x * rank_mask

        x = x + self.channel_mlp(self.channel_norm(x))
        if rank_mask is not None:
            x = x * rank_mask
        return x


class RankPartQueryConditionedMixerAdapter(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_ranks,
        num_slots,
        mixer_dim=256,
        depth=2,
        hidden_part=32,
        hidden_rank=64,
        hidden_channel=512,
        hidden_readout=128,
    ):
        super().__init__()
        if num_ranks < 1:
            raise ValueError("num_ranks must be a positive integer")
        if num_slots < 1:
            raise ValueError("num_slots must be a positive integer")
        if mixer_dim < 1:
            raise ValueError("mixer_dim must be a positive integer")
        if depth < 1:
            raise ValueError("mixer_depth must be a positive integer")

        self.embed_dim = embed_dim
        self.num_ranks = num_ranks
        self.num_slots = num_slots
        self.mixer_dim = mixer_dim
        self.num_tokens = num_ranks * num_slots

        self.w_in = nn.Linear(embed_dim, mixer_dim)
        self.rank_emb = nn.Parameter(torch.zeros(1, num_ranks, 1, mixer_dim))
        self.part_emb = nn.Parameter(torch.zeros(1, 1, num_slots, mixer_dim))
        self.w_q = nn.Linear(embed_dim, mixer_dim)
        self.film_mlp = _two_layer_mlp(mixer_dim, mixer_dim, mixer_dim * 2)
        self.film_ln = nn.LayerNorm(mixer_dim)
        self.blocks = nn.ModuleList([
            _RankPartMixerBlock(
                num_ranks=num_ranks,
                num_slots=num_slots,
                mixer_dim=mixer_dim,
                hidden_part=hidden_part,
                hidden_rank=hidden_rank,
                hidden_channel=hidden_channel,
            )
            for _ in range(depth)
        ])
        self.final_ln = nn.LayerNorm(mixer_dim)
        self.readout_mlp = _two_layer_mlp(self.num_tokens, hidden_readout, 1)
        self.w_out = nn.Linear(mixer_dim, embed_dim)
        self.last_diagnostics = {}

        nn.init.trunc_normal_(self.rank_emb, std=0.02)
        nn.init.trunc_normal_(self.part_emb, std=0.02)

    def _pad_to_configured_ranks(self, prototype_bank):
        batch_size, num_ranks, num_slots, dim = prototype_bank.shape
        if dim != self.embed_dim:
            raise ValueError(
                "B_q_M last dimension must match adapter embed_dim: "
                f"got {dim}, expected {self.embed_dim}"
            )
        if num_slots != self.num_slots:
            raise ValueError(
                "B_q_M prototype-slot dimension must match adapter num_slots: "
                f"got {num_slots}, expected {self.num_slots}"
            )
        if num_ranks > self.num_ranks:
            raise ValueError(
                "B_q_M rank dimension exceeds adapter num_ranks: "
                f"got {num_ranks}, expected at most {self.num_ranks}"
            )
        if num_ranks == self.num_ranks:
            return prototype_bank, None

        pad = prototype_bank.new_zeros(
            batch_size,
            self.num_ranks - num_ranks,
            num_slots,
            dim,
        )
        padded = torch.cat([prototype_bank, pad], dim=1)
        rank_mask = prototype_bank.new_zeros(1, self.num_ranks, 1, 1)
        rank_mask[:, :num_ranks] = 1.0
        return padded, rank_mask

    def _weight_norm(self, modules):
        total = None
        for module in modules:
            for parameter in module.parameters():
                if parameter.dim() < 2:
                    continue
                value = parameter.detach().float().norm()
                total = value if total is None else total + value
        if total is None:
            return self.rank_emb.detach().float().sum() * 0.0
        return total

    def _weight_diagnostics(self):
        return {
            "mixer/rank_mixing_weight_norm": self._weight_norm([block.rank_mlp for block in self.blocks]),
            "mixer/part_mixing_weight_norm": self._weight_norm([block.part_mlp for block in self.blocks]),
            "mixer/channel_mixing_weight_norm": self._weight_norm([block.channel_mlp for block in self.blocks]),
            "mixer/readout_weight_norm": self._weight_norm([self.readout_mlp]),
        }

    def forward(self, z_q, B_q_M):
        if z_q.dim() != 2:
            raise ValueError(f"z_q must have shape [B, d], got {tuple(z_q.shape)}")
        if B_q_M.dim() != 4:
            raise ValueError(f"B_q_M must have shape [B, M, P+1, d], got {tuple(B_q_M.shape)}")
        if z_q.shape[0] != B_q_M.shape[0]:
            raise ValueError("z_q and B_q_M batch dimensions must match")
        if z_q.shape[1] != self.embed_dim:
            raise ValueError(
                "z_q last dimension must match adapter embed_dim: "
                f"got {z_q.shape[1]}, expected {self.embed_dim}"
            )

        B_q_M, rank_mask = self._pad_to_configured_ranks(B_q_M.float())
        rank_mask = None if rank_mask is None else rank_mask.to(device=B_q_M.device, dtype=B_q_M.dtype)

        x = self.w_in(B_q_M)
        x = x + self.rank_emb.to(dtype=x.dtype) + self.part_emb.to(dtype=x.dtype)
        if rank_mask is not None:
            x = x * rank_mask

        q = self.w_q(z_q.float())
        scale, shift = self.film_mlp(q).chunk(2, dim=-1)
        scale = torch.tanh(scale)
        x = self.film_ln(x) * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        if rank_mask is not None:
            x = x * rank_mask

        for block in self.blocks:
            x = block(x, rank_mask=rank_mask)

        h_structured = self.final_ln(x)
        if rank_mask is not None:
            h_structured = h_structured * rank_mask
        h_flat = h_structured.flatten(1, 2)
        h_t = h_flat.transpose(1, 2)
        h = self.readout_mlp(h_t).squeeze(-1)
        c_q = self.w_out(h)

        with torch.no_grad():
            diagnostics = {
                "mixer/context_norm": c_q.detach().float().norm(dim=1).mean(),
                "mixer/film_scale_mean": scale.detach().float().mean(),
                "mixer/film_scale_std": scale.detach().float().std(unbiased=False),
                "mixer/film_shift_mean": shift.detach().float().mean(),
                "mixer/film_shift_std": shift.detach().float().std(unbiased=False),
                "mixer/H_mean": h_structured.detach().float().mean(),
                "mixer/H_std": h_structured.detach().float().std(unbiased=False),
                "mixer/H_flat_token_std": h_flat.detach().float().std(dim=1, unbiased=False).mean(),
                "mixer/readout_output_norm": h.detach().float().norm(dim=1).mean(),
            }
            diagnostics.update(self._weight_diagnostics())
            self.last_diagnostics = diagnostics

        return c_q


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

        num_slots = getattr(args, "num_parts", 6) + 1
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

        if self.enable_grab:
            self.proto_to_grab = nn.Linear(embed_dim, grab_embed_dim)
            self.grab_context = RankPartQueryConditionedMixerAdapter(grab_embed_dim, **mixer_kwargs)
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

    def _fuse(self, query_features, context, space):
        delta = self._fusion_delta(query_features, context, space)
        return F.normalize(query_features.float() + self.gamma * delta, p=2, dim=-1)

    def _fuse_with_delta(self, query_features, context, space):
        delta = self._fusion_delta(query_features, context, space)
        enriched = F.normalize(query_features.float() + self.gamma * delta, p=2, dim=-1)
        return enriched, delta

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
        enriched, delta = self._fuse_with_delta(normalized_query, context, space)

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
