import shutil
import unittest
import uuid
from pathlib import Path

import log_reader


ROOT = Path(__file__).resolve().parents[1]


SAMPLE_LOG = """2026-05-31 12:21:56,993 ITSELF INFO: Namespace(tau=0.015
 output_dir='ITSELF/RSTPReid/run'
 name='PPL'
 img_size=(384
 128)
 milestones=(45
 50)
 dataset_name='RSTPReid'
 extractor_mode='global
horizontal'
 target_enrichment=True
 top_m=64
 mixer_hidden_rank=128
 lambda_ret=0.5
 use_target_retrieval_loss=True
 num_epoch=30
 use_host_loss=False
 distributed=False)
2026-05-31 13:53:17,833 ITSELF.train INFO: best R1: 62.30000305175781 at epoch 17
wandb: View run demo at: https://wandb.ai/hoang_1/enrichment/runs/d119p8xi
"""


class LogReaderTests(unittest.TestCase):
    def test_parse_multiline_namespace_values(self):
        config = log_reader.parse_log_config(SAMPLE_LOG)

        self.assertEqual(config["img_size"], (384, 128))
        self.assertEqual(config["milestones"], (45, 50))
        self.assertEqual(config["extractor_mode"], "global,horizontal")
        self.assertFalse(config["distributed"])

    def test_extract_run_url_and_best_r1(self):
        self.assertEqual(
            log_reader.extract_run_url(SAMPLE_LOG),
            "https://wandb.ai/hoang_1/enrichment/runs/d119p8xi",
        )
        self.assertEqual(log_reader.extract_best_r1(SAMPLE_LOG), (62.3, 17))
        self.assertEqual(log_reader.extract_best_r1("Traceback: failed before final metric"), ("", ""))

    def test_parse_log_file_skips_incomplete_runs(self):
        tmp_dir = ROOT / "tests_tmp" / f"log_reader_{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        log_path = tmp_dir / "crashed.log"
        log_path.write_text(
            "2026-05-31 12:21:56,993 ITSELF INFO: Namespace(target_enrichment=True)\n"
            "Traceback: failed before final metric\n",
            encoding="utf-8",
        )

        self.assertIsNone(log_reader.parse_log_file(log_path, tmp_dir, {}, []))

    def test_section_config_keys_are_read_from_options(self):
        keys = log_reader.extract_section_config_keys(
            ROOT / "utils" / "options.py",
            log_reader.EXTRA_CONFIG_SECTIONS,
        )

        self.assertIn("target_enrichment", keys)
        self.assertIn("mixer_hidden_rank", keys)
        self.assertIn("lambda_ret", keys)
        self.assertNotIn("batch_size", keys)

    def test_diff_configs_filters_keys_and_formats_one_cell(self):
        log_config = {"target_enrichment": True, "top_m": 64, "batch_size": 64}
        default_config = {"target_enrichment": False, "top_m": 32, "batch_size": 256}

        diffs = log_reader.diff_configs(
            log_config,
            default_config,
            ["target_enrichment", "top_m"],
        )

        self.assertEqual(diffs, {"target_enrichment": True, "top_m": 64})
        self.assertEqual(log_reader.format_extra_configs(diffs), "- target_enrichment\n- top_m: 64")

    def test_append_csv_roundtrip_and_skip_existing(self):
        tmp_dir = ROOT / "tests_tmp" / f"log_reader_{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        output = tmp_dir / "logs.csv"
        rows = [
            {
                "log_file": "a.log",
                "run": "https://wandb.ai/x/y/runs/abc",
                "r1": 62.3,
                "best_epoch": 17,
                "extra_config": "- target_enrichment\n- top_m: 64",
            },
            {
                "log_file": "crashed.log",
                "run": "https://wandb.ai/x/y/runs/crashed",
                "r1": "",
                "best_epoch": "",
                "extra_config": "- target_enrichment",
            }
        ]

        added = log_reader.append_rows(output, rows, skip_existing=False)
        self.assertEqual(added, 1)

        added = log_reader.append_rows(output, rows[:1], skip_existing=True)
        self.assertEqual(added, 0)

        headers, existing_rows = log_reader.read_csv(output)
        self.assertEqual(headers, log_reader.BASE_COLUMNS)
        self.assertEqual(len(existing_rows), 1)
        self.assertEqual(existing_rows[0]["log_file"], "a.log")
        self.assertEqual(existing_rows[0]["r1"], "62.3")
        self.assertEqual(existing_rows[0]["extra_config"], "- target_enrichment\n- top_m: 64")

    def test_duplicate_extra_config_keeps_highest_r1(self):
        rows = [
            {
                "log_file": "low.log",
                "run": "https://wandb.ai/x/y/runs/low",
                "r1": 61.0,
                "best_epoch": 10,
                "extra_config": "- target_enrichment\n- top_m: 64",
            },
            {
                "log_file": "high.log",
                "run": "https://wandb.ai/x/y/runs/high",
                "r1": 62.5,
                "best_epoch": 12,
                "extra_config": " - target_enrichment\r\n- top_m: 64 ",
            },
            {
                "log_file": "other.log",
                "run": "https://wandb.ai/x/y/runs/other",
                "r1": 60.0,
                "best_epoch": 8,
                "extra_config": "- target_enrichment\n- top_m: 32",
            },
        ]

        deduped = log_reader.dedupe_rows_by_extra_config(rows)

        self.assertEqual([row["log_file"] for row in deduped], ["high.log", "other.log"])


if __name__ == "__main__":
    unittest.main()
