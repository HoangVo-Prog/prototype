"""Evaluate gallery-conditioned query specificity for prototype CLIP/ITSELF.

Examples:
    python scripts/eval_gallery_conditioned_query.py \
        --host-model clip \
        --base-checkpoint path/to/base_host.pth \
        --gate-checkpoint path/to/gate.pth \
        --config path/to/clip_gate_run/configs.yaml \
        --data CUHK-PEDES \
        --root-dir path/to/data_root \
        --output-dir runs/gallery_conditioned_clip

    python scripts/eval_gallery_conditioned_query.py \
        --host-model itself \
        --base-checkpoint path/to/base_host.pth \
        --gate-checkpoint path/to/gate.pth \
        --config path/to/itself_gate_run/configs.yaml \
        --data RSTPReid \
        --root-dir path/to/data_root \
        --output-dir runs/gallery_conditioned_itself
"""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
COMMON_MODULE_PATH = REPO_ROOT / "gallery_conditioned_query_common.py"


def _common_import_error(reason: str) -> str:
    return (
        "Unable to import repository-local gallery_conditioned_query_common.\n"
        "Reason: {}\n"
        "Resolved script directory: {}\n"
        "Resolved repository root: {}\n"
        "Expected common-module location: {}\n"
        "Current sys.path: {}"
    ).format(reason, SCRIPT_DIR, REPO_ROOT, COMMON_MODULE_PATH, sys.path)


if not COMMON_MODULE_PATH.is_file():
    raise ImportError(_common_import_error("expected common module file is missing"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from gallery_conditioned_query_common import RepoSpec, run_gallery_conditioned_query
except ImportError as error:
    raise ImportError(_common_import_error(str(error))) from error


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
