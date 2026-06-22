import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_inference_module():
    spec = importlib.util.spec_from_file_location("inference_test_py", ROOT / "test.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InferenceConfigLoadingTests(unittest.TestCase):
    def test_config_overlay_keeps_new_train_defaults(self):
        inference = load_inference_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text(
                "\n".join(
                    [
                        "dataset_name: RSTPReid",
                        "only_global: true",
                        "pretrain_choice: ViT-B/16",
                    ]
                ),
                encoding="utf-8",
            )

            args, config_keys = inference._load_inference_config(str(config_file))
            inference._ensure_inference_defaults(args, str(config_file), config_keys)

            self.assertEqual(args.dataset_name, "RSTPReid")
            self.assertTrue(args.only_global)
            self.assertTrue(hasattr(args, "topm_rank_space"))
            self.assertTrue(hasattr(args, "mixer_hidden_readout"))
            self.assertEqual(args.eval_log_interval, 30.0)

    def test_checkpoint_resolves_from_config_output_dir(self):
        inference = load_inference_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "demo"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            checkpoint = weights_dir / "clip_or_itself.pth"
            checkpoint.write_bytes(b"placeholder")
            config_file = run_dir / "config.yaml"
            config_file.write_text("output_dir: .\ncheckpoint: weights/clip_or_itself.pth\n", encoding="utf-8")

            cli_args = SimpleNamespace(
                checkpoint=None,
                output_eval_dir=None,
                results_file=None,
            )
            args = SimpleNamespace(
                output_dir=".",
                checkpoint="weights/clip_or_itself.pth",
            )

            resolved_checkpoint, eval_dir, results_file = inference._resolve_paths(
                cli_args,
                args,
                str(config_file),
                None,
                {"output_dir", "checkpoint"},
            )

            self.assertEqual(Path(resolved_checkpoint), checkpoint)
            self.assertEqual(Path(eval_dir), run_dir / "eval")
            self.assertEqual(Path(results_file), run_dir / "eval" / "eval_results.json")


class CheckpointerLoadingTests(unittest.TestCase):
    def test_checkpointer_loads_state_dict_wrapper(self):
        import torch
        from utils.checkpoint import Checkpointer

        model = torch.nn.Linear(2, 2)
        replacement = torch.nn.Linear(2, 2)
        with torch.no_grad():
            replacement.weight.fill_(3.0)
            replacement.bias.fill_(1.0)

        Checkpointer(model)._load_model({"state_dict": replacement.state_dict()})

        self.assertTrue(torch.equal(model.weight, replacement.weight))
        self.assertTrue(torch.equal(model.bias, replacement.bias))


if __name__ == "__main__":
    unittest.main()
