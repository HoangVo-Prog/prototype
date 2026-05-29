import sys
import types
import unittest
from pathlib import Path

from utils.wandb_utils import upload_best_checkpoint_artifact


ROOT = Path(__file__).resolve().parents[1]


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class FakeRun:
    def __init__(self, name="run+name"):
        self.name = name
        self.summary = {}
        self.logged_artifacts = []

    def log_artifact(self, artifact, aliases=None):
        self.logged_artifacts.append((artifact, aliases))


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)


class WandbCheckpointArtifactTests(unittest.TestCase):
    def setUp(self):
        self.original_wandb = sys.modules.get("wandb")
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.Artifact = FakeArtifact
        sys.modules["wandb"] = fake_wandb

    def tearDown(self):
        if self.original_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = self.original_wandb

    def test_upload_best_checkpoint_artifact_logs_best_file(self):
        run = FakeRun(name="20260529_010203_ITSELF_tal+cid")

        artifact = upload_best_checkpoint_artifact(
            run,
            ROOT,
            checkpoint_name="requirements.txt",
        )

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.name, "20260529_010203_ITSELF_tal-cid-best")
        self.assertEqual(artifact.type, "model")
        self.assertEqual(artifact.files[0][1], "requirements.txt")
        self.assertEqual(run.logged_artifacts[0][1], ["best", "latest"])
        self.assertEqual(run.summary["best_checkpoint_artifact"], artifact.name)

    def test_upload_best_checkpoint_artifact_skips_missing_file(self):
        run = FakeRun()
        logger = FakeLogger()

        artifact = upload_best_checkpoint_artifact(
            run,
            ROOT,
            logger=logger,
            checkpoint_name="missing-best.pth",
        )

        self.assertIsNone(artifact)
        self.assertEqual(run.logged_artifacts, [])
        self.assertIn("not found", logger.warnings[0])


if __name__ == "__main__":
    unittest.main()
