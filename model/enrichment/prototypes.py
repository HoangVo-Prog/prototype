import torch
import torch.nn.functional as F


EXTRACTOR_MODES = (
    "global",
    "horizontal",
    "vertical",
    "grid",
    "global_horizontal",
    "global_vertical",
    "global_grid",
)


def prototype_slot_count(mode, num_parts):
    if mode not in EXTRACTOR_MODES:
        raise ValueError(f"Unknown extractor mode: {mode}")
    if num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")
    if mode == "global":
        return 1
    if mode in ("grid", "global_grid"):
        slots = num_parts * num_parts
    else:
        slots = num_parts
    return slots + int(mode.startswith("global_"))


def _balanced_bounds(size, num_parts, device):
    boundaries = torch.linspace(0, size, steps=num_parts + 1, device=device)
    boundaries = boundaries.round().long().tolist()
    bounds = []
    for part_idx in range(num_parts):
        start = boundaries[part_idx]
        end = boundaries[part_idx + 1]
        if end <= start:
            end = min(start + 1, size)
            start = max(0, end - 1)
        bounds.append((start, end))
    return bounds


def _resolve_patch_grid(patch_features, grid_size, mode):
    batch_size, num_patches, dim = patch_features.shape
    if grid_size is not None:
        grid_h, grid_w = grid_size
        if grid_h * grid_w == num_patches:
            return patch_features.reshape(batch_size, grid_h, grid_w, dim)
    if mode in ("vertical", "grid", "global_vertical", "global_grid"):
        raise ValueError(f"--extractor_mode {mode} requires a valid patch grid_size")
    return None


def _horizontal_prototypes(patch_features, patch_grid, num_parts):
    parts = []
    device = patch_features.device
    if patch_grid is not None:
        grid_h = patch_grid.shape[1]
        for start, end in _balanced_bounds(grid_h, num_parts, device):
            part = patch_grid[:, start:end, :, :].mean(dim=(1, 2))
            parts.append(F.normalize(part, p=2, dim=-1))
        return parts

    num_patches = patch_features.shape[1]
    for start, end in _balanced_bounds(num_patches, num_parts, device):
        part = patch_features[:, start:end, :].mean(dim=1)
        parts.append(F.normalize(part, p=2, dim=-1))
    return parts


def _vertical_prototypes(patch_grid, num_parts):
    parts = []
    grid_w = patch_grid.shape[2]
    for start, end in _balanced_bounds(grid_w, num_parts, patch_grid.device):
        part = patch_grid[:, :, start:end, :].mean(dim=(1, 2))
        parts.append(F.normalize(part, p=2, dim=-1))
    return parts


def _grid_prototypes(patch_grid, num_parts):
    parts = []
    row_bounds = _balanced_bounds(patch_grid.shape[1], num_parts, patch_grid.device)
    col_bounds = _balanced_bounds(patch_grid.shape[2], num_parts, patch_grid.device)
    for row_start, row_end in row_bounds:
        for col_start, col_end in col_bounds:
            part = patch_grid[:, row_start:row_end, col_start:col_end, :].mean(dim=(1, 2))
            parts.append(F.normalize(part, p=2, dim=-1))
    return parts


def build_part_prototypes(token_features, num_parts, grid_size=None, mode="global_horizontal"):
    if mode not in EXTRACTOR_MODES:
        raise ValueError(f"--extractor_mode must be one of {EXTRACTOR_MODES}, got {mode}")
    if num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")

    token_features = token_features.float()
    global_feature = F.normalize(token_features[:, 0, :], p=2, dim=-1)
    if mode == "global":
        return global_feature.unsqueeze(1)

    patch_features = token_features[:, 1:, :]
    patch_grid = _resolve_patch_grid(patch_features, grid_size, mode)

    if mode in ("horizontal", "global_horizontal"):
        parts = _horizontal_prototypes(patch_features, patch_grid, num_parts)
    elif mode in ("vertical", "global_vertical"):
        parts = _vertical_prototypes(patch_grid, num_parts)
    else:
        parts = _grid_prototypes(patch_grid, num_parts)

    if mode.startswith("global_"):
        parts = [global_feature] + parts
    return torch.stack(parts, dim=1)
