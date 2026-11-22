"""Lightweight tests for dataset validation; no ML dependencies are needed."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_model import inspect_input_tensor, load_dataset, load_readiness_report, model_metadata, resample_sequence, split_by_signer, split_for_test, split_sample_level, split_training_only, training_history_document, validate_training_options
from validate_dataset import FEATURES, FEATURE_VERSION, dataset_fingerprint, find_signer_split, inspect_dataset, normalize_label

TEST_LABELS = ["WORD_A", "WORD_B", "WORD_C", "WORD_D", "WORD_E"]


def frame():
    values = [0.0] * FEATURES
    values[0] = 0.5
    return values


def complete_payload():
    samples = []
    for signer_index in range(5):
        signer = f"signer-{signer_index+1:02d}"
        for label_index, label in enumerate(TEST_LABELS):
            samples.append({
                "id": f"{signer}-{label_index:02d}",
                "signerId": signer,
                "label": label,
                "verified": True,
                "consentRecorded": True,
                "verificationNotes": "Reference reviewed with FSL teacher",
                "features": [frame(), frame()],
            })
    return {
        "schemaVersion": 1,
        "featureVersion": FEATURE_VERSION,
        "featureDimension": FEATURES,
        "labels": TEST_LABELS,
        "samples": samples,
    }


class ValidateDatasetTests(unittest.TestCase):
    def test_input_tensor_diagnostics_accept_normalized_landmarks_and_missing_values(self):
        x = np.zeros((1, 30, FEATURES), dtype=np.float32)
        x[:, :, 0] = 0.5
        x[:, :, 132:258] = -3.0

        stats = inspect_input_tensor(x, "test input")

        self.assertEqual(stats["nan"], 0)
        self.assertEqual(stats["inf"], 0)
        self.assertEqual(stats["zero_frames"], 0)
        self.assertEqual(stats["zero_sequences"], 0)
        self.assertGreater(stats["missing_values"], 0)

    def test_input_tensor_rejects_nonfinite_raw_scale_and_empty_sequences(self):
        x = np.zeros((1, 30, FEATURES), dtype=np.float32)
        x[:, :, 0] = 0.5
        x[0, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            inspect_input_tensor(x)

        x[0, 0, 1] = 0.0
        x[0, 0, 2] = 10.0
        with self.assertRaisesRegex(ValueError, "normalized feature range"):
            inspect_input_tensor(x)

        with self.assertRaisesRegex(ValueError, "all-zero sequence"):
            inspect_input_tensor(np.zeros((1, 30, FEATURES), dtype=np.float32))

    def test_input_tensor_rejects_individual_zero_frames(self):
        x = np.zeros((1, 30, FEATURES), dtype=np.float32)
        x[:, :, 0] = 0.5
        x[0, 7] = 0.0

        with self.assertRaisesRegex(ValueError, "all-zero frame"):
            inspect_input_tensor(x)

    def test_training_artifacts_preserve_ordered_labels_and_epoch_curves(self):
        labels = ["WORD_B", "WORD_A"]
        self.assertEqual(model_metadata(labels), {
            "labels": labels,
            "sequence_length": 30,
            "feature_count": FEATURES,
        })

        document = training_history_document(type("History", (), {
            "history": {
                "loss": [0.8, 0.4],
                "accuracy": [0.5, 0.9],
                "val_loss": [0.9, 0.6],
                "val_accuracy": [0.4, 0.8],
            }
        })())
        self.assertEqual(document["epochs"], [1, 2])
        self.assertEqual(document["metrics"]["val_accuracy"], [0.4, 0.8])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            training_history_document(type("History", (), {"history": {"loss": [float("nan")]}})())

    def test_all_zero_sequences_are_not_loaded_as_valid_classes(self):
        payload = complete_payload()
        payload["samples"][0]["features"] = [[0.0] * FEATURES, [0.0] * FEATURES]
        report = inspect_dataset(payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, _ = load_dataset(path)

        self.assertEqual(len(samples), len(payload["samples"]) - 1)
        self.assertEqual(report["excludedReasons"]["all_zero_sequence"], 1)

    def test_zero_feature_frames_and_huge_feature_numbers_are_excluded(self):
        payload = complete_payload()
        payload["samples"][0]["features"][0] = [0.0] * FEATURES
        payload["samples"][1]["features"][0][0] = 10 ** 400

        report = inspect_dataset(payload)

        self.assertEqual(report["excludedReasons"]["zero_feature_frame"], 1)
        self.assertEqual(report["excludedReasons"]["invalid_feature_values"], 1)
        self.assertEqual(report["excludedSamples"], 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, _ = load_dataset(path)
        self.assertEqual(len(samples), len(payload["samples"]) - 2)

    def test_loads_precomputed_readiness_report(self):
        report = {"readyForCandidateTraining": True, "samplesInExport": 25, "datasetFingerprint": "matching"}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset-report.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            self.assertEqual(load_readiness_report(path, "matching"), report)
            with self.assertRaisesRegex(ValueError, "does not match the dataset"):
                load_readiness_report(path, "different")

            path.write_text('{"readyForCandidateTraining":"yes"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
                load_readiness_report(path)

    def test_readiness_report_must_match_recomputed_eligibility(self):
        payload = complete_payload()
        payload["samples"][0]["verified"] = False
        expected = inspect_dataset(payload)
        forged = dict(expected)
        forged["readyForCandidateTraining"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset-report.json"
            path.write_text(json.dumps(forged), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match the current validator"):
                load_readiness_report(
                    path,
                    dataset_fingerprint(payload),
                    expected_report=expected,
                )

    def test_training_options_enforce_the_evaluation_floor(self):
        self.assertIsNone(validate_training_options(1, 1, 0.5))
        self.assertIsNone(validate_training_options(200, 32, 1.0))
        for options in ((0, 32, 0.5), (1, 0, 0.5), (1, 32, 0.49), (1, 32, 1.01), (1, 32, float("nan"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_training_options(*options)

    def test_complete_signer_independent_coverage(self):
        report = inspect_dataset(complete_payload())
        self.assertTrue(report["readyForSignerIndependentTraining"])
        self.assertEqual(report["validSamples"], 25)
        self.assertEqual(report["distinctSigners"], 5)
        self.assertEqual(report["featureDimension"], 456)
        self.assertEqual(report["datasetFingerprint"], dataset_fingerprint(complete_payload()))

    def test_sparse_labels_report_that_fit_is_possible_but_evaluation_is_not(self):
        payload = complete_payload()
        payload["samples"] = payload["samples"][:len(TEST_LABELS)]
        report = inspect_dataset(payload)
        self.assertFalse(report["readyForCandidateTraining"])
        self.assertIn("at least 3 valid samples per label", report["nextStep"])
        self.assertIn("fit an unscored, non-deployable candidate", report["nextStep"])

    def test_one_frame_video_remains_valid_and_is_resampled_as_a_static_snapshot(self):
        payload = complete_payload()
        payload["samples"][0]["features"] = [frame()]

        report = inspect_dataset(payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, _ = load_dataset(path)

        self.assertEqual(report["validSamples"], 25)
        self.assertEqual(report["excludedSamples"], 0)
        self.assertEqual(len(samples[0]["x"]), 30)
        self.assertTrue(all(sample_frame.tolist() == frame() for sample_frame in samples[0]["x"]))
        with self.assertRaises(ValueError):
            resample_sequence([])

        payload["samples"][1]["features"] = []
        empty_report = inspect_dataset(payload)
        self.assertEqual(empty_report["excludedSamples"], 1)
        self.assertEqual(empty_report["excludedReasons"]["too_few_frames"], 1)

    def test_unconsented_sample_is_excluded_once(self):
        payload = complete_payload()
        payload["samples"][0]["consentRecorded"] = False
        report = inspect_dataset(payload)
        self.assertEqual(report["excludedSamples"], 1)
        self.assertEqual(report["excludedReasons"]["consent_not_confirmed"], 1)
        self.assertFalse(report["readyForSignerIndependentTraining"])

    def test_invalid_feature_width_is_excluded(self):
        payload = complete_payload()
        payload["samples"][0]["features"][0] = [0.0] * 455
        report = inspect_dataset(payload)
        self.assertEqual(report["excludedSamples"], 1)
        self.assertEqual(report["excludedReasons"]["invalid_feature_values"], 1)
        self.assertFalse(report["readyForSignerIndependentTraining"])

    def test_truthy_variants_and_trimmed_labels_are_accepted(self):
        payload = complete_payload()
        for sample in payload["samples"]:
            sample["verified"] = "true"
            sample["consentRecorded"] = 1
            sample["label"] = f" {sample['label']} "
        report = inspect_dataset(payload)
        self.assertTrue(report["readyForSignerIndependentTraining"])
        self.assertEqual(report["validSamples"], 25)

    def test_feature_schema_mismatch_is_error(self):
        payload = complete_payload()
        payload["featureVersion"] = "old-version"
        report = inspect_dataset(payload)
        self.assertTrue(any("featureVersion" in e for e in report["errors"]))
        self.assertFalse(report["readyForSignerIndependentTraining"])

    def test_pre_resampled_sequence_metadata_matches_the_model_input(self):
        payload = complete_payload()
        payload.update(featuresResampled=True, sequenceLength=29)

        report = inspect_dataset(payload)

        self.assertIn("Pre-resampled datasets must declare sequenceLength 30", report["errors"])
        self.assertFalse(report["readyForCandidateTraining"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain 30 frames"):
                load_dataset(path)

    def test_pre_resampled_samples_must_contain_exactly_30_frames(self):
        payload = complete_payload()
        payload.update(featuresResampled=True, sequenceLength=30)

        report = inspect_dataset(payload)

        self.assertEqual(
            report["excludedReasons"]["invalid_resampled_sequence_length"],
            len(payload["samples"]),
        )
        self.assertFalse(report["readyForCandidateTraining"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No samples with valid 456-feature"):
                load_dataset(path)

    def test_resampled_flag_must_be_boolean(self):
        payload = complete_payload()
        payload.update(featuresResampled="false", sequenceLength=30)

        report = inspect_dataset(payload)

        self.assertIn("featuresResampled must be a boolean", report["errors"])
        self.assertFalse(report["readyForCandidateTraining"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "featuresResampled field must be a boolean"):
                load_dataset(path)

    def test_readiness_rejects_out_of_range_coordinates_and_invalid_sentinels(self):
        for value in (10.0, -2.7, -3.1):
            with self.subTest(value=value):
                payload = complete_payload()
                payload["samples"][0]["features"][0][0] = value

                report = inspect_dataset(payload)

                self.assertEqual(report["excludedReasons"]["invalid_feature_values"], 1)
                self.assertFalse(report["readyForCandidateTraining"])

    def test_split_requires_every_label_in_train_validation_and_test(self):
        payload = complete_payload()
        support = {
            "WORD_A": {"signer-01", "signer-02", "signer-03"},
            "WORD_B": {"signer-03", "signer-04", "signer-05"},
        }
        payload["samples"] = [
            sample for sample in payload["samples"]
            if sample["label"] not in support or sample["signerId"] in support[sample["label"]]
        ]
        report = inspect_dataset(payload)
        self.assertFalse(report["readyForSignerIndependentTraining"])
        self.assertTrue(any("Could not find a signer-independent split" in warning for warning in report["warnings"]))

    def test_shared_split_contains_each_label_in_every_partition(self):
        signer_labels = {f"signer-{i:02d}": set(TEST_LABELS) for i in range(1, 6)}
        split = find_signer_split(signer_labels, TEST_LABELS, seed=7)
        self.assertIsNotNone(split)
        for signer_ids in split.values():
            present = set().union(*(signer_labels[signer] for signer in signer_ids))
            self.assertEqual(present, set(TEST_LABELS))
        self.assertIsNone(find_signer_split(signer_labels, TEST_LABELS, seed=7, tries=0))

    def test_non_object_dataset_root_is_reported(self):
        report = inspect_dataset([])
        self.assertTrue(any("root must be a JSON object" in error for error in report["errors"]))
        self.assertFalse(report["readyForSignerIndependentTraining"])

    def test_non_string_labels_and_signer_aliases_are_not_coerced_into_valid_values(self):
        self.assertEqual(normalize_label(None), "")
        payload = complete_payload()
        payload["samples"][0]["label"] = None
        payload["samples"][1]["signerId"] = None
        report = inspect_dataset(payload)
        self.assertEqual(report["excludedSamples"], 2)
        self.assertEqual(report["excludedReasons"]["unknown_label"], 1)
        self.assertEqual(report["excludedReasons"]["missing_signer"], 1)

    def test_trainer_loads_structurally_valid_samples_without_verification_metadata(self):
        payload = complete_payload()
        payload["samples"][0]["verified"] = False
        payload["samples"][0]["consentRecorded"] = False
        payload["labels"] = ["WORD_A"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, labels = load_dataset(path)
        self.assertEqual(len(samples), 25)
        self.assertEqual(len(labels), 5)
        self.assertFalse(samples[0]["verified"] and samples[0]["consentRecorded"])

    def test_trainer_loads_one_label_without_signer_aliases(self):
        payload = {
            "labels": ["HELLO"],
            "samples": [{
                "label": "HELLO",
                "features": [frame(), frame()],
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, labels = load_dataset(path)
        parts, split = split_for_test(samples, seed=7)
        self.assertEqual(labels, ["HELLO"])
        self.assertEqual(len(samples), 1)
        self.assertEqual([len(part) for part in parts], [1, 0, 0])
        self.assertEqual(split, {"train": [], "validation": [], "test": []})

    def test_test_mode_accepts_unverified_unconsented_single_signer_single_class_data(self):
        payload = {
            "labels": [" sample "],
            "samples": [{
                "label": "sample",
                "features": [frame(), frame()],
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, labels = load_dataset(path)
        parts, split = split_for_test(samples, seed=7)
        self.assertEqual(labels, ["SAMPLE"])
        self.assertEqual([len(part) for part in parts], [1, 0, 0])
        self.assertEqual(split, {"train": [], "validation": [], "test": []})

    def test_test_mode_split_is_stratified_and_never_reuses_samples(self):
        samples = [
            {"id": f"{label}-{index}", "label": label}
            for label in ("WORD_A", "WORD_B")
            for index in range(3)
        ]

        partitions, _ = split_for_test(samples, seed=7)

        self.assertEqual([len(part) for part in partitions], [2, 2, 2])
        sample_ids = [{sample["id"] for sample in partition} for partition in partitions]
        self.assertFalse(sample_ids[0] & sample_ids[1])
        self.assertFalse(sample_ids[0] & sample_ids[2])
        self.assertFalse(sample_ids[1] & sample_ids[2])
        for partition in partitions:
            self.assertEqual({sample["label"] for sample in partition}, {"WORD_A", "WORD_B"})

    def test_test_mode_infers_labels_and_still_rejects_bad_feature_shape(self):
        payload = {
            "samples": [
                {"label": "one", "features": [frame(), frame()]},
                {"label": "two", "features": [frame(), frame()]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples, labels = load_dataset(path)
            self.assertEqual(labels, ["ONE", "TWO"])
            self.assertEqual(len(samples), 2)

            payload["samples"][0]["features"][0] = [0.0] * 455
            payload["samples"][1]["features"][0] = [0.0] * 455
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No samples with valid 456-feature"):
                load_dataset(path)

    def test_trainer_loads_ready_dataset_and_covers_labels_in_every_partition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            path.write_text(json.dumps(complete_payload()), encoding="utf-8")
            samples, labels = load_dataset(path)
        partitions, _ = split_by_signer(samples, labels, seed=7)
        self.assertEqual(len(samples), 25)
        for partition in partitions:
            self.assertEqual({sample["label"] for sample in partition}, set(TEST_LABELS))

    def test_missing_signer_alias_cannot_be_reported_as_signer_independent(self):
        samples = [
            {**sample, "signer": sample["signerId"]}
            for sample in complete_payload()["samples"]
        ]
        samples[0]["signer"] = ""

        with self.assertRaisesRegex(ValueError, "signer alias for every sample"):
            split_by_signer(samples, TEST_LABELS, seed=7)

    def test_sparse_labels_fall_back_to_training_without_claiming_evaluation(self):
        samples = [
            {"label": "WORD_A", "signer": "signer-01"},
            {"label": "WORD_A", "signer": "signer-02"},
            {"label": "WORD_A", "signer": "signer-03"},
            {"label": "WORD_B", "signer": "signer-01"},
        ]
        with self.assertRaisesRegex(ValueError, "at least 3 separate samples"):
            split_sample_level(samples, ["WORD_A", "WORD_B"], seed=7)

        parts, split = split_training_only(samples, ["WORD_A", "WORD_B"], seed=7)
        self.assertEqual(len(parts[0]), len(samples))
        self.assertEqual(parts[1:], ([], []))
        self.assertEqual(set(split["train"]), {"signer-01", "signer-02", "signer-03"})
        self.assertEqual(split["validation"], [])
        self.assertEqual(split["test"], [])

    def test_training_only_split_rejects_labels_without_valid_samples(self):
        samples = [{"label": "WORD_A", "signer": "signer-01"}]
        with self.assertRaisesRegex(ValueError, "at least one valid sample.*WORD_B"):
            split_training_only(samples, ["WORD_A", "WORD_B"], seed=7)

    def test_trainer_refuses_split_missing_a_label_from_one_partition(self):
        signer_labels = {
            "signer-01": {"WORD_A", "WORD_C", "WORD_D", "WORD_E"},
            "signer-02": {"WORD_A", "WORD_C", "WORD_D", "WORD_E"},
            "signer-03": set(TEST_LABELS),
            "signer-04": {"WORD_B", "WORD_C", "WORD_D", "WORD_E"},
            "signer-05": {"WORD_B", "WORD_C", "WORD_D", "WORD_E"},
        }
        samples = [
            {"signer": signer, "label": label}
            for signer, labels in signer_labels.items()
            for label in labels
        ]
        with self.assertRaisesRegex(ValueError, "Could not find a signer-independent split"):
            split_by_signer(samples, TEST_LABELS, seed=7)


if __name__ == "__main__":
    unittest.main()
