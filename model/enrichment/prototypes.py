import torch
import torch.nn.functional as F


EXTRACTOR_MODES = (
    "global",
    "horizontal",
    "vertical",
    "grid",
)

_LEGACY_EXTRACTOR_MODE_ALIASES = {
    "global_horizontal": "global,horizontal",
    "global_vertical": "global,vertical",
    "global_grid": "global,grid",
}


def _normalize_extractor_modes(mode):
    if isinstance(mode, str):
        raw_tokens = [token.strip().lower() for token in mode.split(",") if token.strip()]
    elif isinstance(mode, (list, tuple)):
        raw_tokens = [str(token).strip().lower() for token in mode if str(token).strip()]
    else:
        raise ValueError("--extractor_mode must be a comma-separated string or sequence of mode names")

    expanded_tokens = []
    for token in raw_tokens:
        alias = _LEGACY_EXTRACTOR_MODE_ALIASES.get(token)
        if alias is not None:
            expanded_tokens.extend(alias.split(","))
        else:
            expanded_tokens.append(token)

    modes = []
    seen = set()
    for token in expanded_tokens:
        if token not in EXTRACTOR_MODES:
            raise ValueError(
                f"--extractor_mode supports comma-separated values from {EXTRACTOR_MODES}, got {mode}"
            )
        if token not in seen:
            seen.add(token)
            modes.append(token)

    if not modes:
        raise ValueError("--extractor_mode must contain at least one mode")
    return tuple(modes)


def canonicalize_extractor_mode(mode):
    return ",".join(_normalize_extractor_modes(mode))


def prototype_slot_count(mode, num_parts):
    modes = _normalize_extractor_modes(mode)
    if num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")

    slots = 0
    for extractor in modes:
        if extractor == "global":
            slots += 1
        elif extractor == "grid":
            slots += num_parts * num_parts
        else:
            slots += num_parts
    return slots


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


def _resolve_patch_grid(patch_features, grid_size, modes):
    batch_size, num_patches, dim = patch_features.shape
    if grid_size is not None:
        grid_h, grid_w = grid_size
        if grid_h * grid_w == num_patches:
            return patch_features.reshape(batch_size, grid_h, grid_w, dim)

    requires_grid = any(extractor in ("vertical", "grid") for extractor in modes)
    if requires_grid:
        raise ValueError(
            f"--extractor_mode {','.join(modes)} requires a valid patch grid_size "
            "for vertical/grid extractors"
        )
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


def build_part_prototypes(token_features, num_parts, grid_size=None, mode="global,horizontal"):
    modes = _normalize_extractor_modes(mode)
    if num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")

    token_features = token_features.float()
    global_feature = F.normalize(token_features[:, 0, :], p=2, dim=-1)
    if modes == ("global",):
        return global_feature.unsqueeze(1)

    patch_features = token_features[:, 1:, :]
    patch_grid = _resolve_patch_grid(patch_features, grid_size, modes)

    parts = []
    for extractor in modes:
        if extractor == "global":
            parts.append(global_feature)
        elif extractor == "horizontal":
            parts.extend(_horizontal_prototypes(patch_features, patch_grid, num_parts))
        elif extractor == "vertical":
            parts.extend(_vertical_prototypes(patch_grid, num_parts))
        else:
            parts.extend(_grid_prototypes(patch_grid, num_parts))

    return torch.stack(parts, dim=1)
