import sys
import shutil
import types
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from utils.checkpoint import delete_output_checkpoints
from utils.wandb_utils import (
    build_wandb_config,
    get_wandb_project,
    upload_best_checkpoint_artifact,
    upload_checkpoint_artifacts,
)


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


class WandbConfigTests(unittest.TestCase):
    def test_build_wandb_config_uses_cli_project_name(self):
        args = SimpleNamespace(wandb_project="custom-project", use_wandb=True)

        config = build_wandb_config(args, run_name="run-1", output_dir="logs/run-1")

        self.assertEqual(config["wandb_project"], "custom-project")
        self.assertEqual(config["wandb_run_name"], "run-1")
        self.assertEqual(config["output_dir"], "logs/run-1")

    def test_wandb_project_falls_back_for_legacy_args(self):
        self.assertEqual(get_wandb_project(SimpleNamespace()), "enrichment")


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

    def _make_tmp_output_dir(self):
        tmp_path = ROOT / "tests_tmp" / f"wandb_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, tmp_path, True)
        return tmp_path

    def test_upload_best_checkpoint_artifact_logs_best_file(self):
        run = FakeRun(name="20260529_010203_ITSELF_tal+cid")
        tmp_path = self._make_tmp_output_dir()
        checkpoint_name = "best.pth"
        (tmp_path / checkpoint_name).write_text("checkpoint", encoding="utf-8")

        artifact = upload_best_checkpoint_artifact(
            run,
            tmp_path,
            checkpoint_name=checkpoint_name,
        )

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.name, "20260529_010203_ITSELF_tal-cid-best")
        self.assertEqual(artifact.type, "model")
        self.assertEqual(artifact.files[0][1], checkpoint_name)
        self.assertEqual(run.logged_artifacts[0][1], ["best", "latest"])
        self.assertEqual(run.summary["best_checkpoint_artifact"], artifact.name)

    def test_upload_best_checkpoint_artifact_uploads_config_yaml(self):
        run = FakeRun(name="demo-run")
        tmp_path = self._make_tmp_output_dir()
        (tmp_path / "best.pth").write_text("checkpoint", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("foo: bar\n", encoding="utf-8")

        artifact = upload_best_checkpoint_artifact(run, tmp_path)

        self.assertIsNotNone(artifact)
        uploaded_names = [name for _, name in artifact.files]
        self.assertIn("best.pth", uploaded_names)
        self.assertIn("config.yaml", uploaded_names)
        self.assertEqual(artifact.metadata["config"], "config.yaml")
        self.assertIn("best_checkpoint_config_path", run.summary)

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

    def test_upload_checkpoint_artifacts_logs_all_run_checkpoints(self):
        run = FakeRun(name="demo-run")
        tmp_path = self._make_tmp_output_dir()
        (tmp_path / "best.pth").write_text("best", encoding="utf-8")
        (tmp_path / "epoch_2.pth").write_text("epoch", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("foo: bar\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("not a checkpoint", encoding="utf-8")

        artifacts = upload_checkpoint_artifacts(run, tmp_path)

        self.assertIsNotNone(artifacts)
        self.assertEqual(len(artifacts), 2)
        uploaded_checkpoints = [artifact.metadata["checkpoint"] for artifact in artifacts]
        self.assertEqual(uploaded_checkpoints, ["best.pth", "epoch_2.pth"])
        self.assertEqual(run.logged_artifacts[0][1], ["best", "latest"])
        self.assertEqual(run.logged_artifacts[1][1], ["epoch_2"])
        self.assertEqual(run.summary["checkpoint_artifact_count"], 2)
        self.assertEqual(
            run.summary["checkpoint_artifacts"],
            ["demo-run-best", "demo-run-epoch_2"],
        )


class CheckpointCleanupTests(unittest.TestCase):
    def _make_tmp_output_dir(self):
        tmp_path = ROOT / "tests_tmp" / f"cleanup_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, tmp_path, True)
        return tmp_path

    def test_delete_output_checkpoints_removes_only_direct_pth_files(self):
        tmp_path = self._make_tmp_output_dir()
        nested_path = tmp_path / "nested"
        nested_path.mkdir()
        (tmp_path / "best.pth").write_text("best", encoding="utf-8")
        (tmp_path / "epoch_2.pth").write_text("epoch", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("foo: bar\n", encoding="utf-8")
        (nested_path / "nested.pth").write_text("nested", encoding="utf-8")

        deleted = delete_output_checkpoints(tmp_path)

        self.assertEqual([path.name for path in deleted], ["best.pth", "epoch_2.pth"])
        self.assertFalse((tmp_path / "best.pth").exists())
        self.assertFalse((tmp_path / "epoch_2.pth").exists())
        self.assertTrue((tmp_path / "config.yaml").exists())
        self.assertTrue((nested_path / "nested.pth").exists())


if __name__ == "__main__":
    unittest.main()
