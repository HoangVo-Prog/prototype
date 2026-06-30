import torch
import torch.nn as nn

from utils.qcrs_mixer_variants import qcrs_mixer_variant_config


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
    def __init__(
        self,
        num_ranks,
        num_slots,
        mixer_dim,
        hidden_part,
        hidden_rank,
        hidden_channel,
        use_slotmix=True,
        use_rankmix=True,
        use_chanmix=True,
    ):
        super().__init__()
        self.use_slotmix = use_slotmix
        self.use_rankmix = use_rankmix
        self.use_chanmix = use_chanmix
        if self.use_slotmix:
            self.part_norm = nn.LayerNorm(mixer_dim)
            self.part_mlp = _two_layer_mlp(num_slots, hidden_part, num_slots)
        if self.use_rankmix:
            self.rank_norm = nn.LayerNorm(mixer_dim)
            self.rank_mlp = _two_layer_mlp(num_ranks, hidden_rank, num_ranks)
        if self.use_chanmix:
            self.channel_norm = nn.LayerNorm(mixer_dim)
            self.channel_mlp = _two_layer_mlp(mixer_dim, hidden_channel, mixer_dim)

    def forward(self, x, rank_mask=None):
        if self.use_slotmix:
            part_delta = self.part_mlp(self.part_norm(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            x = x + part_delta
            if rank_mask is not None:
                x = x * rank_mask

        if self.use_rankmix:
            rank_delta = self.rank_mlp(self.rank_norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + rank_delta
            if rank_mask is not None:
                x = x * rank_mask

        if self.use_chanmix:
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
        context_pooling="mlp",
        qcrs_mixer_variant="qcrs_full",
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
        if context_pooling != "mlp":
            raise ValueError("context_pooling must be mlp")
        variant_config = qcrs_mixer_variant_config(qcrs_mixer_variant)

        self.embed_dim = embed_dim
        self.num_ranks = num_ranks
        self.num_slots = num_slots
        self.mixer_dim = mixer_dim
        self.num_tokens = num_ranks * num_slots
        self.context_pooling = context_pooling
        self.qcrs_mixer_variant = qcrs_mixer_variant
        self.mixer_mode = variant_config["mixer_mode"]
        self.use_film = variant_config["use_film"]
        self.use_rank_emb = variant_config["use_rank_emb"]
        self.use_slot_emb = variant_config["use_slot_emb"]
        self.use_slotmix = variant_config["use_slotmix"]
        self.use_rankmix = variant_config["use_rankmix"]
        self.use_chanmix = variant_config["use_chanmix"]

        self.w_in = nn.Linear(embed_dim, mixer_dim)
        if self.use_rank_emb:
            self.rank_emb = nn.Parameter(torch.zeros(1, num_ranks, 1, mixer_dim))
        else:
            self.register_parameter("rank_emb", None)
        if self.use_slot_emb:
            self.part_emb = nn.Parameter(torch.zeros(1, 1, num_slots, mixer_dim))
        else:
            self.register_parameter("part_emb", None)
        if self.use_film:
            self.w_q = nn.Linear(embed_dim, mixer_dim)
            self.film_mlp = _two_layer_mlp(mixer_dim, mixer_dim, mixer_dim * 2)
            self.film_ln = nn.LayerNorm(mixer_dim)
        if self.mixer_mode == "qcrs":
            self.blocks = nn.ModuleList([
                _RankPartMixerBlock(
                    num_ranks=num_ranks,
                    num_slots=num_slots,
                    mixer_dim=mixer_dim,
                    hidden_part=hidden_part,
                    hidden_rank=hidden_rank,
                    hidden_channel=hidden_channel,
                    use_slotmix=self.use_slotmix,
                    use_rankmix=self.use_rankmix,
                    use_chanmix=self.use_chanmix,
                )
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList()
        self.final_ln = nn.LayerNorm(mixer_dim)
        if self.mixer_mode == "qcrs":
            self.readout_mlp = _two_layer_mlp(self.num_tokens, hidden_readout, 1)
        elif self.mixer_mode == "flat_mlp":
            self.flat_mlp = nn.Sequential(
                nn.LayerNorm(self.num_tokens * mixer_dim),
                nn.Linear(self.num_tokens * mixer_dim, hidden_readout),
                nn.GELU(),
                nn.Linear(hidden_readout, mixer_dim),
            )
        elif self.mixer_mode != "mean_pool":
            raise ValueError("Unsupported QCRS mixer mode: {}".format(self.mixer_mode))
        self.w_out = nn.Linear(mixer_dim, embed_dim)
        self.last_diagnostics = {}

        if self.rank_emb is not None:
            nn.init.trunc_normal_(self.rank_emb, std=0.02)
        if self.part_emb is not None:
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
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.dim() < 2:
                    continue
                value = parameter.detach().float().norm()
                total = value if total is None else total + value
        if total is None:
            return self._zero_diagnostic()
        return total

    def _zero_diagnostic(self):
        parameter = next(self.parameters(), None)
        if parameter is None:
            return torch.tensor(0.0)
        return parameter.detach().float().sum() * 0.0

    def _weight_diagnostics(self):
        readout_modules = []
        if hasattr(self, "readout_mlp"):
            readout_modules.append(self.readout_mlp)
        if hasattr(self, "flat_mlp"):
            readout_modules.append(self.flat_mlp)
        return {
            "mixer/rank_mixing_weight_norm": self._weight_norm([
                getattr(block, "rank_mlp", None) for block in self.blocks
            ]),
            "mixer/part_mixing_weight_norm": self._weight_norm([
                getattr(block, "part_mlp", None) for block in self.blocks
            ]),
            "mixer/channel_mixing_weight_norm": self._weight_norm([
                getattr(block, "channel_mlp", None) for block in self.blocks
            ]),
            "mixer/readout_weight_norm": self._weight_norm(readout_modules),
        }

    def _mlp_pool(self, h_flat):
        h_t = h_flat.transpose(1, 2)
        return self.readout_mlp(h_t).squeeze(-1)

    def _masked_mean_pool(self, x, rank_mask):
        if rank_mask is None:
            return x.mean(dim=(1, 2))
        token_mask = rank_mask.expand(x.shape[0], -1, x.shape[2], -1)
        denominator = token_mask.sum(dim=(1, 2, 3)).clamp_min(1.0).to(dtype=x.dtype).view(-1, 1)
        return (x * rank_mask).sum(dim=(1, 2)) / denominator

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
        if self.rank_emb is not None:
            x = x + self.rank_emb.to(dtype=x.dtype)
        if self.part_emb is not None:
            x = x + self.part_emb.to(dtype=x.dtype)
        if rank_mask is not None:
            x = x * rank_mask

        scale = None
        shift = None
        if self.use_film:
            q = self.w_q(z_q.float())
            scale, shift = self.film_mlp(q).chunk(2, dim=-1)
            scale = torch.tanh(scale)
            x = self.film_ln(x) * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
            if rank_mask is not None:
                x = x * rank_mask

        if self.mixer_mode == "qcrs":
            for block in self.blocks:
                x = block(x, rank_mask=rank_mask)

            h_structured = self.final_ln(x)
            if rank_mask is not None:
                h_structured = h_structured * rank_mask
            h_flat = h_structured.flatten(1, 2)
            h = self._mlp_pool(h_flat)
        elif self.mixer_mode == "flat_mlp":
            h_structured = self.final_ln(x)
            if rank_mask is not None:
                h_structured = h_structured * rank_mask
            h_flat = h_structured.flatten(1, 2)
            h = self.flat_mlp(h_structured.flatten(1))
        else:
            h_structured = x
            h_flat = h_structured.flatten(1, 2)
            h = self._masked_mean_pool(h_structured, rank_mask)
        c_q = self.w_out(h)

        with torch.no_grad():
            diagnostics = {
                "mixer/context_norm": c_q.detach().float().norm(dim=1).mean(),
                "mixer/H_mean": h_structured.detach().float().mean(),
                "mixer/H_std": h_structured.detach().float().std(unbiased=False),
                "mixer/H_flat_token_std": h_flat.detach().float().std(dim=1, unbiased=False).mean(),
                "mixer/readout_output_norm": h.detach().float().norm(dim=1).mean(),
            }
            if scale is not None and shift is not None:
                diagnostics.update({
                    "mixer/film_scale_mean": scale.detach().float().mean(),
                    "mixer/film_scale_std": scale.detach().float().std(unbiased=False),
                    "mixer/film_shift_mean": shift.detach().float().mean(),
                    "mixer/film_shift_std": shift.detach().float().std(unbiased=False),
                })
            diagnostics.update(self._weight_diagnostics())
            self.last_diagnostics = diagnostics

        return c_q
