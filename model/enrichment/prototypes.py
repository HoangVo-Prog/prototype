import torch
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
