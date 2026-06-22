import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.build_atomic_cue_cases import build_arg_parser, parse_args, run_builder
from diagnostic.cue_cases import load_cases, select_queries_for_cases
from diagnostic.cue_vocabulary.datasets import load_all_training_captions
from diagnostic.cue_vocabulary.match import match_training_caption
from diagnostic.cue_vocabulary.models import CueDefinition, DatasetLoadResult, TrainingCaption
from diagnostic.cue_vocabulary.pairs import build_cue_pairs
from diagnostic.cue_vocabulary.support import collect_support
from diagnostic.data_loading import QueryRecord


def cue(
    cue_id,
    family,
    slot,
    value,
    display_name,
    aliases,
    matcher,
    *,
    broader_than=(),
    narrower_than=(),
    incompatible_with=(),
):
    return CueDefinition(
        cue_id=cue_id,
        family=family,
        slot=slot,
        value=value,
        display_name=display_name,
        runtime_expression=display_name,
        aliases=tuple(aliases),
        matcher=matcher,
        broader_than=tuple(broader_than),
        narrower_than=tuple(narrower_than),
        incompatible_with=tuple(incompatible_with),
    )


BACKPACK_CUE = cue(
    "carried_item__backpack",
    "carried_item",
    "carried_item",
    "backpack",
    "backpack",
    ("backpack", "rucksack", "back pack"),
    {"type": "phrase"},
)
BAG_CUE = cue(
    "carried_item__bag",
    "carried_item",
    "carried_item",
    "bag",
    "bag",
    ("bag",),
    {"type": "phrase"},
    narrower_than=("carried_item__backpack", "carried_item__handbag"),
)
HANDBAG_CUE = cue(
    "carried_item__handbag",
    "carried_item",
    "carried_item",
    "handbag",
    "handbag",
    ("handbag", "hand bag"),
    {"type": "phrase"},
    broader_than=("carried_item__bag",),
)
CAP_CUE = cue("worn_accessory__cap", "worn_accessory", "worn_accessory", "cap", "cap", ("cap", "baseball cap"), {"type": "phrase"})
HAT_CUE = cue("worn_accessory__hat", "worn_accessory", "worn_accessory", "hat", "hat", ("hat",), {"type": "phrase"})
STRIPED_CUE = cue("clothing_pattern__striped", "clothing_pattern", "clothing_pattern", "striped", "striped clothing", ("striped", "stripes"), {"type": "phrase"}, broader_than=("clothing_pattern__patterned",))
PATTERNED_CUE = cue("clothing_pattern__patterned", "clothing_pattern", "clothing_pattern", "patterned", "patterned clothing", ("patterned",), {"type": "phrase"}, narrower_than=("clothing_pattern__striped",))
TSHIRT_CUE = cue("upper_garment_type__t_shirt", "garment_footwear_type", "upper_garment_type", "t_shirt", "t-shirt", ("t-shirt", "t shirt", "tee"), {"type": "phrase"})
SHIRT_CUE = cue("upper_garment_type__shirt", "garment_footwear_type", "upper_garment_type", "shirt", "shirt", ("shirt",), {"type": "phrase"})
GRAY_UPPER_CUE = cue("upper_body_color__gray", "clothing_footwear_color", "upper_body_color", "gray", "gray upper-body clothing", ("gray", "grey"), {"type": "color_context", "context_group": "upper_body", "max_token_distance": 3})
BLACK_UPPER_CUE = cue("upper_body_color__black", "clothing_footwear_color", "upper_body_color", "black", "black upper-body clothing", ("black",), {"type": "color_context", "context_group": "upper_body", "max_token_distance": 3})
BLACK_LOWER_CUE = cue("lower_body_color__black", "clothing_footwear_color", "lower_body_color", "black", "black lower-body clothing", ("black",), {"type": "color_context", "context_group": "lower_body", "max_token_distance": 3})
BLACK_FOOTWEAR_CUE = cue("footwear_color__black", "clothing_footwear_color", "footwear_color", "black", "black footwear", ("black",), {"type": "color_context", "context_group": "footwear", "max_token_distance": 3})
RED_UPPER_CUE = cue("upper_body_color__red", "clothing_footwear_color", "upper_body_color", "red", "red upper-body clothing", ("red",), {"type": "color_context", "context_group": "upper_body", "max_token_distance": 3})
JACKET_CUE = cue("upper_garment_type__jacket", "garment_footwear_type", "upper_garment_type", "jacket", "jacket", ("jacket",), {"type": "phrase"})
LONG_SLEEVES_CUE = cue("sleeve_length__long", "sleeve_length", "sleeve_length", "long", "long sleeves", ("long sleeves", "long sleeved", "long sleeve"), {"type": "phrase"})


def make_dataset_result(dataset_name, cue_mentions_by_identity):
    records = []
    for identity_id, captions in cue_mentions_by_identity.items():
        for index, caption in enumerate(captions):
            records.append(
                TrainingCaption(
                    dataset=dataset_name,
                    identity_id=str(identity_id),
                    caption=caption,
                    image_ref=f"{dataset_name}/{identity_id}/{index}.jpg",
                )
            )
    return DatasetLoadResult(
        dataset_name=dataset_name,
        annotation_file=Path(f"{dataset_name}.json"),
        records=tuple(records),
        num_training_identities=len(cue_mentions_by_identity),
        num_training_captions=len(records),
    )


def write_catalog(path, cues):
    payload = {"version": 1, "cues": []}
    for cue_def in cues:
        payload["cues"].append(
            {
                "id": cue_def.cue_id,
                "family": cue_def.family,
                "slot": cue_def.slot,
                "value": cue_def.value,
                "display_name": cue_def.display_name,
                "runtime_expression": cue_def.runtime_expression,
                "aliases": list(cue_def.aliases),
                "matcher": dict(cue_def.matcher),
                "broader_than": list(cue_def.broader_than),
                "narrower_than": list(cue_def.narrower_than),
                "incompatible_with": list(cue_def.incompatible_with),
            }
        )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class AtomicCueBuilderTest(unittest.TestCase):
    def test_training_split_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            specs = {
                "CUHK-PEDES": ("reid_raw.json", "file_path"),
                "ICFG-PEDES": ("ICFG-PEDES.json", "file_path"),
                "RSTPReid": ("data_captions.json", "img_path"),
            }
            for dataset_name, (anno_name, image_key) in specs.items():
                dataset_root = root / dataset_name
                (dataset_root / "imgs").mkdir(parents=True)
                payload = [
                    {"split": "train", "id": 1, image_key: "a.jpg", "captions": ["plain person"]},
                    {"split": "test", "id": 2, image_key: "b.jpg", "captions": ["person with backpack"]},
                    {"split": "val", "id": 3, image_key: "c.jpg", "captions": ["person with backpack"]},
                ]
                (dataset_root / anno_name).write_text(json.dumps(payload), encoding="utf-8")
            datasets = load_all_training_captions(
                cuhk_root=root / "CUHK-PEDES",
                icfg_root=root / "ICFG-PEDES",
                rstp_root=root / "RSTPReid",
            )
            decisions, _ = collect_support(
                dataset_results=datasets,
                cues=[BACKPACK_CUE],
                min_identities=1,
                min_datasets=1,
            )
            self.assertEqual(decisions["carried_item__backpack"].counts_by_dataset["CUHK-PEDES"], 0)
            self.assertEqual(decisions["carried_item__backpack"].counts_by_dataset["ICFG-PEDES"], 0)
            self.assertEqual(decisions["carried_item__backpack"].counts_by_dataset["RSTPReid"], 0)

    def test_identity_level_deduplication(self):
        datasets = [
            make_dataset_result("CUHK-PEDES", {1: ["red jacket", "red jacket with backpack"], 2: ["red jacket"]}),
            make_dataset_result("ICFG-PEDES", {1: ["red jacket"], 2: ["red jacket"]}),
            make_dataset_result("RSTPReid", {}),
        ]
        decisions, _ = collect_support(
            dataset_results=datasets,
            cues=[JACKET_CUE],
            min_identities=2,
            min_datasets=2,
        )
        counts = decisions["upper_garment_type__jacket"].counts_by_dataset
        self.assertEqual(counts["CUHK-PEDES"], 2)
        self.assertEqual(counts["ICFG-PEDES"], 2)
        self.assertEqual(counts["RSTPReid"], 0)

    def test_threshold_boundaries(self):
        cuhk_24 = {idx: ["backpack"] for idx in range(24)}
        icfg_25 = {idx: ["backpack"] for idx in range(25)}
        rstp_25 = {idx: ["backpack"] for idx in range(25)}
        datasets = [
            make_dataset_result("CUHK-PEDES", cuhk_24),
            make_dataset_result("ICFG-PEDES", icfg_25),
            make_dataset_result("RSTPReid", rstp_25),
        ]
        decisions, _ = collect_support(
            dataset_results=datasets,
            cues=[BACKPACK_CUE],
            min_identities=25,
            min_datasets=2,
        )
        decision = decisions["carried_item__backpack"]
        self.assertEqual(decision.counts_by_dataset["CUHK-PEDES"], 24)
        self.assertEqual(decision.counts_by_dataset["ICFG-PEDES"], 25)
        self.assertEqual(decision.counts_by_dataset["RSTPReid"], 25)
        self.assertTrue(decision.retained)

        decisions_one_pass, _ = collect_support(
            dataset_results=[
                make_dataset_result("CUHK-PEDES", cuhk_24),
                make_dataset_result("ICFG-PEDES", icfg_25),
                make_dataset_result("RSTPReid", {}),
            ],
            cues=[BACKPACK_CUE],
            min_identities=25,
            min_datasets=2,
        )
        self.assertFalse(decisions_one_pass["carried_item__backpack"].retained)

    def test_synonyms(self):
        record = TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="grey shirt with rucksack")
        matched = {evidence.cue_id for evidence in match_training_caption(record, [GRAY_UPPER_CUE, BACKPACK_CUE])}
        self.assertEqual(matched, {"upper_body_color__gray", "carried_item__backpack"})

    def test_distinct_related_concepts(self):
        backpack_record = TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person with backpack")
        handbag_record = TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person with handbag")
        cap_record = TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person in cap")
        striped_record = TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person in striped shirt")
        self.assertEqual({e.cue_id for e in match_training_caption(backpack_record, [BACKPACK_CUE, BAG_CUE])}, {"carried_item__backpack"})
        self.assertEqual({e.cue_id for e in match_training_caption(handbag_record, [HANDBAG_CUE, BAG_CUE])}, {"carried_item__handbag"})
        self.assertEqual({e.cue_id for e in match_training_caption(cap_record, [CAP_CUE, HAT_CUE])}, {"worn_accessory__cap"})
        self.assertEqual({e.cue_id for e in match_training_caption(striped_record, [STRIPED_CUE, PATTERNED_CUE])}, {"clothing_pattern__striped"})

    def test_longest_specific_matching(self):
        record = TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person in t-shirt")
        matched = {evidence.cue_id for evidence in match_training_caption(record, [TSHIRT_CUE, SHIRT_CUE])}
        self.assertEqual(matched, {"upper_garment_type__t_shirt"})

    def test_color_context(self):
        self.assertEqual(
            {e.cue_id for e in match_training_caption(TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person in black shirt"), [BLACK_UPPER_CUE])},
            {"upper_body_color__black"},
        )
        self.assertEqual(
            {e.cue_id for e in match_training_caption(TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person in black pants"), [BLACK_LOWER_CUE])},
            {"lower_body_color__black"},
        )
        self.assertEqual(
            {e.cue_id for e in match_training_caption(TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person in black shoes"), [BLACK_FOOTWEAR_CUE])},
            {"footwear_color__black"},
        )
        self.assertEqual(
            {e.cue_id for e in match_training_caption(TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="person with black backpack"), [BLACK_UPPER_CUE, BLACK_LOWER_CUE, BLACK_FOOTWEAR_CUE])},
            set(),
        )

    def test_atomic_decomposition(self):
        record = TrainingCaption(dataset="CUHK-PEDES", identity_id="1", caption="red jacket with backpack")
        matched = {evidence.cue_id for evidence in match_training_caption(record, [RED_UPPER_CUE, JACKET_CUE, BACKPACK_CUE])}
        self.assertEqual(matched, {"upper_body_color__red", "upper_garment_type__jacket", "carried_item__backpack"})

    def test_exhaustive_pairing(self):
        retained = [BACKPACK_CUE, BAG_CUE, JACKET_CUE, LONG_SLEEVES_CUE]
        pairs, exclusions = build_cue_pairs(retained)
        n = len(retained)
        self.assertEqual(len(pairs) + len(exclusions), n * (n - 1) // 2)
        unordered = {tuple(sorted((pair.cue_a_id, pair.cue_b_id))) for pair in pairs}
        self.assertEqual(len(unordered), len(pairs))
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(exclusions[0].reason, "generic_specific_relation")

    def test_outcome_independence_interface(self):
        option_strings = {option for action in build_arg_parser()._actions for option in action.option_strings}
        prohibited = {
            "--retriever_checkpoint",
            "--retriever_config",
            "--cue_shift",
            "--bootstrap_iters",
            "--gallery_size",
            "--min_pair_cue_shift",
        }
        self.assertTrue(prohibited.isdisjoint(option_strings))

    def test_existing_loader_compatibility_and_regex_only_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "catalog.yaml"
            write_catalog(catalog_path, [BACKPACK_CUE, BLACK_UPPER_CUE, JACKET_CUE])
            self._write_fixture_datasets(root)
            output_dir = root / "out"
            args = parse_args(
                [
                    "--cuhk_root",
                    str(root / "CUHK-PEDES"),
                    "--icfg_root",
                    str(root / "ICFG-PEDES"),
                    "--rstp_root",
                    str(root / "RSTPReid"),
                    "--output_dir",
                    str(output_dir),
                    "--candidate_catalog",
                    str(catalog_path),
                    "--min_identities",
                    "2",
                    "--min_datasets",
                    "2",
                ]
            )
            run_builder(args)
            cases = load_cases(output_dir / "cue_cases.yaml")
            self.assertTrue(cases)
            query_records = [
                QueryRecord(query_id=1, text="person in black shirt with rucksack", pid=1),
                QueryRecord(query_id=2, text="plain person", pid=2),
            ]
            selected, skipped = select_queries_for_cases(
                "CUHK-PEDES",
                [case for case in cases if case["cue_a"] == "backpack" or case["cue_b"] == "backpack"],
                query_records,
                gallery_pids=self._gallery_pids([1, 2]),
                max_queries_per_case=None,
            )
            self.assertTrue(selected)
            self.assertFalse(any(row["reason"] == "case_below_min_queries_after_validation" for row in skipped))

    def test_determinism(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "catalog.yaml"
            write_catalog(catalog_path, [BACKPACK_CUE, BLACK_UPPER_CUE, JACKET_CUE, LONG_SLEEVES_CUE])
            self._write_fixture_datasets(root)
            output_a = root / "run_a"
            output_b = root / "run_b"
            args_a = parse_args(
                [
                    "--cuhk_root",
                    str(root / "CUHK-PEDES"),
                    "--icfg_root",
                    str(root / "ICFG-PEDES"),
                    "--rstp_root",
                    str(root / "RSTPReid"),
                    "--output_dir",
                    str(output_a),
                    "--candidate_catalog",
                    str(catalog_path),
                    "--min_identities",
                    "2",
                    "--min_datasets",
                    "2",
                    "--seed",
                    "7",
                ]
            )
            args_b = parse_args(
                [
                    "--cuhk_root",
                    str(root / "CUHK-PEDES"),
                    "--icfg_root",
                    str(root / "ICFG-PEDES"),
                    "--rstp_root",
                    str(root / "RSTPReid"),
                    "--output_dir",
                    str(output_b),
                    "--candidate_catalog",
                    str(catalog_path),
                    "--min_identities",
                    "2",
                    "--min_datasets",
                    "2",
                    "--seed",
                    "7",
                ]
            )
            run_builder(args_a)
            run_builder(args_b)
            for filename in ["cue_vocabulary.yaml", "cue_support.csv", "cue_cases.yaml", "matcher_audit.csv", "construction_summary.md"]:
                self.assertEqual((output_a / filename).read_text(encoding="utf-8"), (output_b / filename).read_text(encoding="utf-8"))
            summary_a = json.loads((output_a / "construction_summary.json").read_text(encoding="utf-8"))
            summary_b = json.loads((output_b / "construction_summary.json").read_text(encoding="utf-8"))
            summary_a["run_timestamp_utc"] = "<normalized>"
            summary_b["run_timestamp_utc"] = "<normalized>"
            self.assertEqual(summary_a, summary_b)

    def _gallery_pids(self, values):
        import numpy as np

        return np.asarray(values, dtype="int64")

    def _write_fixture_datasets(self, root):
        specs = {
            "CUHK-PEDES": ("reid_raw.json", "file_path", {1: ["black shirt with backpack", "long sleeves jacket"], 2: ["black jacket with rucksack"], 3: ["plain person"]}),
            "ICFG-PEDES": ("ICFG-PEDES.json", "file_path", {1: ["black shirt with backpack"], 2: ["black jacket with backpack", "long sleeves"], 3: ["plain person"]}),
            "RSTPReid": ("data_captions.json", "img_path", {1: ["backpack"], 2: ["plain person"]}),
        }
        for dataset_name, (anno_name, image_key, train_rows) in specs.items():
            dataset_root = root / dataset_name
            (dataset_root / "imgs").mkdir(parents=True)
            payload = []
            for identity_id, captions in train_rows.items():
                payload.append({"split": "train", "id": identity_id, image_key: f"{identity_id}.jpg", "captions": captions})
            payload.append({"split": "test", "id": 999, image_key: "test.jpg", "captions": ["black shirt with backpack"]})
            (dataset_root / anno_name).write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
