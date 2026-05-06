import torch
import torch.nn as nn


class PrototypeFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, base_feats: torch.Tensor, prototype_query: torch.Tensor) -> torch.Tensor:
        original_dtype = base_feats.dtype
        fusion_dtype = prototype_query.dtype
        if any(param.dtype != fusion_dtype for param in self.fusion.parameters()):
            self.fusion = self.fusion.to(dtype=fusion_dtype)

        fused = torch.cat([base_feats.to(dtype=fusion_dtype), prototype_query], dim=-1)
        enriched = self.fusion(fused)
        if enriched.dtype != original_dtype:
            enriched = enriched.to(dtype=original_dtype)
        return enriched


class VisualPrototypeEnrichment(nn.Module):
    """
    Branch-faithful visual enrichment: route from image features and fuse back into image features.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fusion = PrototypeFusion(embed_dim)

    def forward(self, image_feats: torch.Tensor, prototype_query: torch.Tensor) -> torch.Tensor:
        return self.fusion(image_feats, prototype_query)
