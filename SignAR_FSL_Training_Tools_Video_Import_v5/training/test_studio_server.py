import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import studio_server


class CandidateDeploymentTests(unittest.TestCase):
    def setUp(self):
        studio_server._set_training_progress(
            active=False,
            percent=0,
            stage="Trainer idle",
            epoch=0,
            totalEpochs=0,
            batch=0,
            batchesPerEpoch=0,
            completed=False,
            failed=False,
            jobId=None,
            result=None,
            error=None,
            output="",
        )
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.output_directory = self.root / "trained-model"
        self.model_directory = self.root / "SignLanguageAR" / "model"
        self.output_directory.mkdir()
        self.model_directory.mkdir(parents=True)

        evaluation = {
            "testMode": False,
            "deploymentEligible": True,
            "evaluationScope": "sample-level",
            "testAccuracy": 0.8,
            "minimumAccuracy": 0.5,
            "minimumClassRecall": 0.5,
            "classRecall": {"A": 1.0, "B": 1.0, "NO SIGN": 1.0},
            "classificationReport": {
                "A": {"recall": 1.0}, "B": {"recall": 1.0}, "NO SIGN": {"recall": 1.0}
            },
            "labels": ["A", "B", "NO SIGN"],
        }
        manifest = {
            "trained": True,
            "testMode": False,
            "modelPath": "./model/model.json",
            "featureVersion": "signar-fsl-456-v2",
            "sequenceLength": 30,
            "featureDimension": 456,
            "outputClasses": 3,
            "labels": ["A", "B", "NO SIGN"],
            "minimumSignersPerLabel": 1,
            "distinctSignersPerLabel": {"A": 2, "B": 2, "NO SIGN": 2},
            "evaluation": evaluation,
        }
        model = {
            "modelTopology": {
                "model_config": {
                    "config": {
                        "layers": [
                            {
                                "class_name": "InputLayer",
                                "name": "landmark_sequence",
                                "config": {"batchInputShape": [None, 30, 456]},
                            },
                            {
                                "class_name": "Dense",
                                "name": "fsl_class_probabilities",
                                "config": {"units": 3},
                            },
                        ],
                        "input_layers": [["landmark_sequence", 0, 0]],
                        "output_layers": [["fsl_class_probabilities", 0, 0]],
                    }
                }
            },
            "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": []}],
        }
        self._write_json(self.output_directory / "vocabulary.json", manifest)
        self._write_json(self.output_directory / "evaluation.json", evaluation)
        self._write_json(self.output_directory / "metadata.json", {
            "labels": ["A", "B", "NO SIGN"],
            "sequence_length": 30,
            "feature_count": 456,
        })
        self._write_json(self.output_directory / "training-history.json", {
            "epochs": [1],
            "metrics": {"loss": [0.4], "accuracy": [0.8]},
        })
        self._write_json(self.output_directory / "model.json", model)
        (self.output_directory / "group1-shard1of1.bin").write_bytes(b"test model weights")

        self._write_json(self.model_directory / "vocabulary.json", {"modelPath": "./model/model.json", "labels": ["OLD"]})
        self._write_json(self.model_directory / "evaluation.json", {"testAccuracy": 0.6})
        self._write_json(self.model_directory / "metadata.json", {"labels": ["OLD"]})
        (self.model_directory / "model.json").write_text("old active model", encoding="utf-8")
        (self.model_directory / "group1-shard1of1.bin").write_bytes(b"old active weights")
        (self.model_directory / "landmark-dataset.json").write_text("keep this file", encoding="utf-8")

        self.patches = [
            patch.object(studio_server, "TOOLKIT_DIR", self.root / "toolkit"),
            patch.object(studio_server, "DATASET_PATH", self.root / "toolkit" / "fsl-dataset.json"),
            patch.object(studio_server, "OUTPUT_DIR", self.output_directory),
            patch.object(studio_server, "MODEL_DIR", self.model_directory),
            patch.object(studio_server, "EXPORT_PATH", self.model_directory / "landmark-dataset.json"),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_json(path: Path, value: dict):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_deploys_only_validated_candidate_and_preserves_existing_files(self):
        result = studio_server._deploy_candidate()

        active_manifest = json.loads((self.model_directory / "vocabulary.json").read_text(encoding="utf-8"))
        active_metadata = json.loads((self.model_directory / "metadata.json").read_text(encoding="utf-8"))
        active_model_path = self.model_directory / Path(active_manifest["modelPath"]).name
        active_model = json.loads(active_model_path.read_text(encoding="utf-8"))
        active_weight_name = active_model["weightsManifest"][0]["paths"][0]

        self.assertTrue(result["deployed"])
        self.assertEqual(active_manifest["labels"], ["A", "B", "NO SIGN"])
        self.assertEqual(active_manifest["metadataPath"], "./model/metadata.json")
        self.assertEqual(active_metadata["labels"], active_manifest["labels"])
        self.assertEqual(active_metadata["sequence_length"], 30)
        self.assertEqual(active_metadata["feature_count"], 456)
        self.assertTrue(active_weight_name.startswith("weights-"))
        self.assertEqual((self.model_directory / active_weight_name).read_bytes(), b"test model weights")
        self.assertEqual((self.model_directory / "model.json").read_text(encoding="utf-8"), "old active model")
        self.assertEqual((self.model_directory / "landmark-dataset.json").read_text(encoding="utf-8"), "keep this file")
        backup_directories = list((self.model_directory / "deployment-backups").iterdir())
        self.assertEqual(len(backup_directories), 1)
        self.assertTrue((backup_directories[0] / "metadata.json").is_file())

    def test_rejects_candidate_with_metadata_label_order_mismatch(self):
        metadata_path = self.output_directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["labels"] = ["B", "A"]
        self._write_json(metadata_path, metadata)

        eligible, message = studio_server._candidate_status()

        self.assertFalse(eligible)
        self.assertIn("metadata labels or input dimensions", message)

    def test_rejects_candidate_without_no_sign_background_class(self):
        manifest_path = self.output_directory / "vocabulary.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["labels"] = ["A", "B"]
        self._write_json(manifest_path, manifest)

        eligible, message = studio_server._candidate_status()

        self.assertFalse(eligible)
        self.assertIn("NO SIGN background class", message)

    def test_rejects_candidate_with_zero_recall_for_a_label(self):
        evaluation_path = self.output_directory / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["classRecall"]["B"] = 0.0
        evaluation["classificationReport"]["B"]["recall"] = 0.0
        self._write_json(evaluation_path, evaluation)

        manifest_path = self.output_directory / "vocabulary.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluation"]["classRecall"]["B"] = 0.0
        self._write_json(manifest_path, manifest)

        eligible, message = studio_server._candidate_status()

        self.assertFalse(eligible)
        self.assertIn("B", message)

    def test_rejects_candidate_that_lowers_the_minimum_accuracy_floor(self):
        evaluation_path = self.output_directory / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["minimumAccuracy"] = 0.49
        self._write_json(evaluation_path, evaluation)

        manifest_path = self.output_directory / "vocabulary.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluation"]["minimumAccuracy"] = 0.49
        self._write_json(manifest_path, manifest)

        eligible, message = studio_server._candidate_status()

        self.assertFalse(eligible)
        self.assertIn("at least 50% held-out accuracy", message)

    def test_rejects_ineligible_candidate_without_replacing_active_manifest(self):
        manifest_path = self.output_directory / "vocabulary.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["trained"] = False
        self._write_json(manifest_path, manifest)
        before = (self.model_directory / "vocabulary.json").read_text(encoding="utf-8")

        eligible, _ = studio_server._candidate_status()
        with self.assertRaises(studio_server.CandidateValidationError):
            studio_server._deploy_candidate()

        self.assertFalse(eligible)
        self.assertEqual((self.model_directory / "vocabulary.json").read_text(encoding="utf-8"), before)

    def test_rejects_weight_path_traversal(self):
        model_path = self.output_directory / "model.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["weightsManifest"][0]["paths"] = ["../outside.bin"]
        self._write_json(model_path, model)

        with self.assertRaisesRegex(studio_server.CandidateValidationError, "unsafe weight-file path"):
            studio_server._deploy_candidate()

        self.assertEqual(
            json.loads((self.model_directory / "vocabulary.json").read_text(encoding="utf-8"))["labels"],
            ["OLD"],
        )

    def test_batch_progress_is_fractional_monotonic_and_keeps_epoch_counts(self):
        studio_server._set_training_progress(
            active=True,
            percent=0,
            stage="Preparing the local trainer",
            epoch=0,
            totalEpochs=0,
            batch=0,
            batchesPerEpoch=0,
        )

        studio_server._record_training_output("SIGNAR_STAGE training")
        studio_server._record_training_output("SIGNAR_BATCH 1/4 1/4")
        first_batch = studio_server._training_progress_snapshot()
        studio_server._record_training_output("SIGNAR_BATCH 1/4 2/4")
        second_batch = studio_server._training_progress_snapshot()
        studio_server._record_training_output("SIGNAR_BATCH 1/4 5/4")
        invalid_batch = studio_server._training_progress_snapshot()
        studio_server._record_training_output("SIGNAR_PROGRESS 1/4")
        completed_epoch = studio_server._training_progress_snapshot()
        studio_server._record_training_output("SIGNAR_STAGE evaluating")
        evaluation = studio_server._training_progress_snapshot()
        studio_server._record_training_output("SIGNAR_STAGE dataset")
        after_earlier_stage = studio_server._training_progress_snapshot()

        self.assertEqual(first_batch["percent"], 14.38)
        self.assertEqual(first_batch["epoch"], 1)
        self.assertEqual(first_batch["totalEpochs"], 4)
        self.assertEqual(first_batch["batch"], 1)
        self.assertEqual(first_batch["batchesPerEpoch"], 4)
        self.assertEqual(second_batch["percent"], 18.75)
        self.assertEqual(invalid_batch, second_batch)
        self.assertEqual(completed_epoch["percent"], 27.5)
        self.assertEqual(evaluation["percent"], completed_epoch["percent"])
        self.assertEqual(after_earlier_stage["percent"], completed_epoch["percent"])
        self.assertEqual(after_earlier_stage["stage"], evaluation["stage"])

    def test_training_percent_calculation_bounds_work_units(self):
        self.assertIsNone(studio_server._training_percent(0, 1, 4, 8))
        self.assertIsNone(studio_server._training_percent(1, 5, 4, 8))
        self.assertEqual(studio_server._training_percent(1, 1, 4, 8), 12.19)
        self.assertEqual(studio_server._training_percent(8, 4, 4, 8), 80)

    def test_early_stopping_advances_to_evaluation_without_claiming_unrun_epochs(self):
        studio_server._set_training_progress(active=True, percent=0, stage="Starting training epochs")
        studio_server._record_training_output("SIGNAR_BATCH 4/8 2/4")
        before_evaluation = studio_server._training_progress_snapshot()
        studio_server._record_training_output("SIGNAR_STAGE evaluating")
        evaluating = studio_server._training_progress_snapshot()
        studio_server._record_training_output("SIGNAR_STAGE exporting")
        exporting = studio_server._training_progress_snapshot()

        self.assertEqual(before_evaluation["epoch"], 4)
        self.assertEqual(before_evaluation["totalEpochs"], 8)
        self.assertEqual(before_evaluation["percent"], 40.62)
        self.assertEqual(evaluating["percent"], before_evaluation["percent"])
        self.assertEqual(evaluating["epoch"], 4)
        self.assertEqual(exporting["percent"], before_evaluation["percent"])
        self.assertFalse(exporting["completed"])

    def test_training_route_streams_epoch_progress_and_finishes_at_100_percent(self):
        class TrainerProcess:
            @property
            def stdout(self):
                return iter([
                    "SIGNAR_STAGE dataset\n",
                    "SIGNAR_STAGE dependencies\n",
                    "SIGNAR_STAGE training\n",
                    "SIGNAR_BATCH 1/4 1/4\n",
                    "SIGNAR_PROGRESS 1/4\n",
                    "SIGNAR_STAGE evaluating\n",
                    "SIGNAR_STAGE exporting\n",
                ])

            @staticmethod
            def wait():
                return 0

        self._write_json(self.output_directory / "dataset-report.json", {"samplesInExport": 1})
        server = ThreadingHTTPServer(("127.0.0.1", 0), studio_server.StudioHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        request = Request(
            base_url + "/api/training/run",
            data=json.dumps({"labels": ["A"], "samples": [{"label": "A"}]}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Training-Job-ID": "test-job-001"},
            method="POST",
        )
        try:
            with patch.object(studio_server.subprocess, "Popen", return_value=TrainerProcess()):
                with urlopen(request) as response:
                    started = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(response.status, 202)
                    self.assertTrue(started["started"])
                for _ in range(100):
                    progress = studio_server._training_progress_snapshot()
                    if progress["completed"] or progress["failed"]:
                        break
                    time.sleep(0.01)
            progress = studio_server._training_progress_snapshot()
            result = progress["result"]

            self.assertIsNotNone(result)
            self.assertTrue(result["completed"])
            self.assertEqual(progress["percent"], 100)
            self.assertEqual(progress["stage"], "Training finished")
            self.assertFalse(progress["active"])
            self.assertTrue(progress["completed"])
            self.assertFalse(progress["failed"])
            self.assertEqual(progress["jobId"], "test-job-001")
            self.assertIsNone(progress["error"])
            self.assertTrue((self.model_directory / "landmark-dataset.json").is_file())
            uploaded = json.loads(studio_server.DATASET_PATH.read_text(encoding="utf-8"))
            exported = json.loads((self.model_directory / "landmark-dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(uploaded["samples"], [{"label": "A"}])
            self.assertEqual(exported, uploaded)
            self.assertFalse(studio_server.DATASET_PATH.with_name(studio_server.DATASET_PATH.name + ".uploading").exists())

            self._write_json(self.output_directory / "dataset-report.json", [])
            invalid_report_request = Request(
                base_url + "/api/training/run",
                data=json.dumps({"labels": ["A"], "samples": [{"label": "A"}]}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Training-Job-ID": "test-job-002"},
                method="POST",
            )
            with patch.object(studio_server.subprocess, "Popen", return_value=TrainerProcess()):
                with urlopen(invalid_report_request) as response:
                    self.assertEqual(response.status, 202)
                for _ in range(100):
                    failed_progress = studio_server._training_progress_snapshot()
                    if failed_progress["completed"] or failed_progress["failed"]:
                        break
                    time.sleep(0.01)
            failed_progress = studio_server._training_progress_snapshot()
            self.assertFalse(failed_progress["active"])
            self.assertFalse(failed_progress["completed"])
            self.assertTrue(failed_progress["failed"])
            self.assertIn("valid dataset readiness report", failed_progress["error"])
            self.assertLess(failed_progress["percent"], 100)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_trainer_failure_stops_progress_without_marking_completion(self):
        class FailedTrainerProcess:
            stdout = iter(["SIGNAR_STAGE training\n", "SIGNAR_BATCH 3/8 2/4\n"])

            @staticmethod
            def wait():
                return 1

        server = ThreadingHTTPServer(("127.0.0.1", 0), studio_server.StudioHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/training/run",
            data=json.dumps({"labels": ["A"], "samples": [{"label": "A"}]}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Training-Job-ID": "test-job-failure"},
            method="POST",
        )
        try:
            with patch.object(studio_server.subprocess, "Popen", return_value=FailedTrainerProcess()):
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                for _ in range(100):
                    progress = studio_server._training_progress_snapshot()
                    if progress["completed"] or progress["failed"]:
                        break
                    time.sleep(0.01)
            progress = studio_server._training_progress_snapshot()
            self.assertFalse(progress["active"])
            self.assertFalse(progress["completed"])
            self.assertTrue(progress["failed"])
            self.assertEqual(progress["stage"], "Training failed")
            self.assertIn("exited with code 1", progress["error"])
            self.assertIn("SIGNAR_BATCH 3/8 2/4", progress["output"])
            self.assertLess(progress["percent"], 100)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_training_start_returns_while_trainer_is_still_running(self):
        trainer_started = threading.Event()
        allow_trainer_exit = threading.Event()

        class PausedTrainerProcess:
            stdout = ()

            @staticmethod
            def wait():
                allow_trainer_exit.wait(timeout=5)
                return 1

        def start_paused_trainer(*args, **kwargs):
            trainer_started.set()
            return PausedTrainerProcess()

        server = ThreadingHTTPServer(("127.0.0.1", 0), studio_server.StudioHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/training/run",
            data=json.dumps({"labels": ["A"], "samples": [{"label": "A"}]}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Training-Job-ID": "test-job-paused"},
            method="POST",
        )
        try:
            with patch.object(studio_server.subprocess, "Popen", side_effect=start_paused_trainer):
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                    self.assertTrue(json.loads(response.read().decode("utf-8"))["started"])
                self.assertTrue(trainer_started.wait(timeout=2))
                self.assertTrue(studio_server._training_progress_snapshot()["active"])
                allow_trainer_exit.set()
                for _ in range(100):
                    progress = studio_server._training_progress_snapshot()
                    if progress["failed"]:
                        break
                    time.sleep(0.01)
            self.assertTrue(studio_server._training_progress_snapshot()["failed"])
        finally:
            allow_trainer_exit.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_local_status_and_deploy_routes(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), studio_server.StudioHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            studio_server._set_training_progress(
                active=True,
                percent=42,
                stage="Training epoch 3 of 8",
                epoch=3,
                totalEpochs=8,
            )
            with urlopen(base_url + "/api/training/progress") as response:
                progress = json.loads(response.read().decode("utf-8"))
            self.assertEqual(progress["percent"], 42)
            self.assertEqual(progress["epoch"], 3)

            with urlopen(base_url + "/api/training/status") as response:
                status = json.loads(response.read().decode("utf-8"))
            self.assertTrue(status["candidateDeployable"])

            request = Request(base_url + "/api/model/deploy", data=b"", method="POST")
            with urlopen(request) as response:
                result = json.loads(response.read().decode("utf-8"))
            self.assertTrue(result["deployed"])
            self.assertEqual(
                json.loads((self.model_directory / "vocabulary.json").read_text(encoding="utf-8"))["modelPath"],
                result["modelPath"],
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
