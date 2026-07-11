"""Evaluate gallery-conditioned query specificity for prototype CLIP/ITSELF.

Examples:
    python scripts/eval_gallery_conditioned_query.py \
        --host-model clip \
        --config configs/clip_base.yaml \
        --base-checkpoint path/to/base_host.pth \
        --gate-checkpoint path/to/gate.pth \
        --output-dir runs/gallery_conditioned_clip

    python scripts/eval_gallery_conditioned_query.py \
        --host-model itself \
        --config configs/itself_base.yaml \
        --base-checkpoint path/to/base_host.pth \
        --gate-checkpoint path/to/gate.pth \
        --output-dir runs/gallery_conditioned_itself
"""

from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = WORKSPACE_ROOT / ".agents"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from gallery_conditioned_query_common import RepoSpec, run_gallery_conditioned_query


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_gallery_conditioned_query(
        RepoSpec(
            repository_name="prototype",
            repo_kind="prototype",
            code_root=repo_root,
            default_host_model="itself",
            metrics_module="utils.test_metrics",
            logger_name="prototype.gallery_conditioned",
            supports_host_model=True,
        )
    )


if __name__ == "__main__":
    main()

