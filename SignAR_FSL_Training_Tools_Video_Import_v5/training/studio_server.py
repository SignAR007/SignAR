"""Local web server that lets the collection page start the Windows trainer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlsplit
from uuid import uuid4

from evaluation import (
    MAXIMUM_NO_SIGN_SHARE,
    MINIMUM_NO_SIGN_FALSE_ACCEPT_RATE,
    MINIMUM_NO_SIGN_SHARE,
    MINIMUM_STRESS_ACCEPT_OR_REJECT,
    MINIMUM_CLASS_RECALL,
    MINIMUM_TEST_ACCURACY,
    NO_SIGN_LABEL,
    class_recall_failures,
    has_no_sign_label,
)


TOOLKIT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOLKIT_DIR.parent
MODEL_DIR = WORKSPACE_ROOT / "SignLanguageAR" / "model"
DATASET_PATH = TOOLKIT_DIR / "fsl-dataset.json"
OUTPUT_DIR = TOOLKIT_DIR / "trained-model"
MAX_REQUEST_BYTES = 128 * 1024 * 1024
TRAINING_LOCK = Lock()
TRAINING_PROGRESS_LOCK = Lock()
TRAINING_PROGRESS = {
    "active": False,
    "percent": 0,
    "stage": "Trainer idle",
    "epoch": 0,
    "totalEpochs": 0,
    "batch": 0,
    "batchesPerEpoch": 0,
    "completed": False,
    "failed": False,
    "jobId": None,
    "result": None,
    "error": None,
}
FEATURE_VERSION = "signar-fsl-holistic-v4"
FEATURE_SEQUENCE_LENGTH = 30
FEATURE_DIMENSION = 422


def _set_training_progress(**updates) -> None:
    with TRAINING_PROGRESS_LOCK:
        TRAINING_PROGRESS.update(updates)


def _training_progress_snapshot() -> dict:
    with TRAINING_PROGRESS_LOCK:
        return dict(TRAINING_PROGRESS)


def _training_percent(epoch: int, batch: int, batches_per_epoch: int, total_epochs: int) -> float | None:
    if (
        total_epochs < 1
        or batches_per_epoch < 1
        or epoch < 1
        or epoch > total_epochs
        or batch < 1
        or batch > batches_per_epoch
    ):
        return None
    completed_epochs = epoch - 1 + batch / batches_per_epoch
    return round(10 + 70 * completed_epochs / total_epochs, 2)


def _fail_training_progress(stage: str, error: str) -> None:
    _set_training_progress(active=False, completed=False, failed=True, stage=stage, error=error)


def _record_training_output(line: str) -> None:
    stage_match = re.search(r"SIGNAR_STAGE ([a-z_]+)", line)
    epoch_match = re.search(r"SIGNAR_PROGRESS (\d+)/(\d+)", line)
    batch_match = re.search(r"SIGNAR_BATCH (\d+)/(\d+) (\d+)/(\d+)", line)
    stage_progress = {
        "environment": (1, "Preparing the Python environment"),
        "dataset": (2, "Validating the dataset"),
        "dependencies": (5, "Preparing Python dependencies"),
        "model": (8, "Preparing the model"),
        "training": (10, "Starting training epochs"),
        "evaluating": (None, "Evaluating model results"),
        "exporting": (None, "Exporting the candidate model"),
        "unchanged": (None, "No new samples; keeping the previous candidate"),
    }
    with TRAINING_PROGRESS_LOCK:
        if stage_match and stage_match.group(1) in stage_progress:
            percent, stage = stage_progress[stage_match.group(1)]
            if percent is None:
                TRAINING_PROGRESS.update(stage=stage)
            elif percent >= float(TRAINING_PROGRESS["percent"]):
                TRAINING_PROGRESS.update(percent=percent, stage=stage)
        if batch_match:
            epoch, total, batch, batches_per_epoch = (int(value) for value in batch_match.groups())
            percent = _training_percent(epoch, batch, batches_per_epoch, total)
            if percent is not None and percent >= float(TRAINING_PROGRESS["percent"]):
                TRAINING_PROGRESS.update(
                    percent=percent,
                    stage=f"Training epoch {epoch} of {total}",
                    epoch=epoch,
                    totalEpochs=total,
                    batch=batch,
                    batchesPerEpoch=batches_per_epoch,
                )
        elif epoch_match:
            epoch, total = (int(value) for value in epoch_match.groups())
            percent = _training_percent(epoch, 1, 1, total)
            if percent is not None and percent >= float(TRAINING_PROGRESS["percent"]):
                TRAINING_PROGRESS.update(
                    percent=percent,
                    stage=f"Training epoch {epoch} of {total}",
                    epoch=epoch,
                    totalEpochs=total,
                    batch=0,
                    batchesPerEpoch=0,
                )


def _run_training() -> None:
    output = ""
    try:
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(TOOLKIT_DIR / "train.ps1"),
            "-Dataset",
            str(DATASET_PATH),
            "-OutputDir",
            str(OUTPUT_DIR),
            "-Incremental",
        ]
        process = subprocess.Popen(
            command,
            cwd=TOOLKIT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise OSError("The trainer process output stream could not be opened.")
        for line in process.stdout:
            output = (output + line)[-12000:]
            _record_training_output(line)
        return_code = process.wait()
        output = output.strip()
        if return_code != 0:
            _fail_training_progress(
                "Training failed",
                "The PowerShell trainer exited with code " + str(return_code) + ".",
            )
            _set_training_progress(output=output[-12000:])
            return

        manifest_path = OUTPUT_DIR / "vocabulary.json"
        evaluation_path = OUTPUT_DIR / "evaluation.json"
        required_outputs = (manifest_path, evaluation_path)
        if not all(path.is_file() for path in required_outputs):
            raise ValueError("The trainer exited successfully but did not produce its evaluation reports.")
        readiness_report = json.loads((OUTPUT_DIR / "dataset-report.json").read_text(encoding="utf-8"))
        if not isinstance(readiness_report, dict):
            raise ValueError("The trainer did not produce a valid dataset readiness report.")
        sample_count = readiness_report.get("samplesInExport")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("The trainer did not produce a valid dataset sample count.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if manifest.get("trained") is True:
            required_model_outputs = (
                OUTPUT_DIR / "model.json",
                OUTPUT_DIR / "metadata.json",
                OUTPUT_DIR / "training-history.json",
            )
            if not all(path.is_file() for path in required_model_outputs):
                raise ValueError("The trainer marked the model deployable but did not produce all model files.")
            try:
                _validated_candidate()
                candidate_deployable = True
                candidate_message = "The candidate passed validation and is ready to transfer."
            except CandidateValidationError as error:
                candidate_deployable = False
                candidate_message = str(error)
        else:
            candidate_deployable = False
            blockers = manifest.get("readinessBlockers") or evaluation.get("readinessBlockers")
            candidate_message = "; ".join(blockers) if isinstance(blockers, list) and blockers else "Evaluation or data-readiness gates did not pass; no model was exported."

        result = {
            "completed": True,
            "samples": sample_count,
            "datasetExport": str(DATASET_PATH),
            "candidateDirectory": str(OUTPUT_DIR),
            "trained": bool(manifest.get("trained")),
            "deploymentEligible": candidate_deployable,
            "deploymentMessage": candidate_message,
            "evaluation": evaluation,
            "output": output[-12000:],
        }
        _set_training_progress(
            active=False,
            percent=100,
            stage="Training finished",
            completed=True,
            failed=False,
            result=result,
            error=None,
        )
    except FileNotFoundError as error:
        _fail_training_progress(
            "Trainer executable not found",
            f"Could not find the local trainer executable or script: {error.filename}",
        )
        _set_training_progress(output=output[-12000:])
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        _fail_training_progress("Training failed", f"Local training failed: {error}")
        _set_training_progress(output=output[-12000:])
    finally:
        if _training_progress_snapshot()["active"]:
            _fail_training_progress("Training failed", "The local trainer stopped unexpectedly.")
        TRAINING_LOCK.release()


class CandidateValidationError(ValueError):
    """Raised when the model candidate is not safe to activate."""


def _read_candidate_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise CandidateValidationError(f"Candidate is missing {description}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateValidationError(f"Candidate {description} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise CandidateValidationError(f"Candidate {description} must contain a JSON object.")
    return value


def _validated_candidate() -> tuple[dict, dict, dict, dict, dict[str, Path]]:
    manifest = _read_candidate_json(OUTPUT_DIR / "vocabulary.json", "vocabulary.json")
    evaluation = _read_candidate_json(OUTPUT_DIR / "evaluation.json", "evaluation.json")
    model_document = _read_candidate_json(OUTPUT_DIR / "model.json", "model.json")
    metadata = _read_candidate_json(OUTPUT_DIR / "metadata.json", "metadata.json")

    manifest_labels = manifest.get("labels")
    if not has_no_sign_label(manifest_labels):
        raise CandidateValidationError(
            f"Candidate must include the required {NO_SIGN_LABEL} background class."
        )
    if manifest.get("trained") is not True or manifest.get("testMode") is not False:
        raise CandidateValidationError("Candidate is not marked as a completed, deployable training run.")
    if manifest.get("modelPath") != "./model/model.json":
        raise CandidateValidationError("Candidate model path does not match the expected training output.")
    if manifest.get("featureVersion") != FEATURE_VERSION:
        raise CandidateValidationError("Candidate feature schema is incompatible with the interpreter.")
    if manifest.get("rejectLabel") != NO_SIGN_LABEL or manifest_labels[0] != NO_SIGN_LABEL:
        raise CandidateValidationError("Candidate must put NO SIGN first as the reject class.")
    if manifest.get("sequenceLength") != FEATURE_SEQUENCE_LENGTH or manifest.get("featureDimension") != FEATURE_DIMENSION:
        raise CandidateValidationError("Candidate input shape is incompatible with the interpreter.")
    if manifest.get("acceptanceProfile") != "signar-standard-v1":
        raise CandidateValidationError("Candidate does not declare the signar-standard-v1 validation profile.")

    labels = manifest.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len({label.strip().upper() for label in labels}) != len(labels)
    ):
        raise CandidateValidationError("Candidate must contain at least two non-empty sign labels.")
    if manifest.get("outputClasses") != len(labels):
        raise CandidateValidationError("Candidate class count does not match its sign labels.")
    label_metadata = manifest.get("labelMetadata")
    if not isinstance(label_metadata, dict) or set(label_metadata) != set(labels):
        raise CandidateValidationError("Candidate must declare unit, motion, and display metadata for every sign label.")
    for label in labels:
        entry = label_metadata[label]
        if not isinstance(entry, dict):
            raise CandidateValidationError(f"Candidate metadata for {label!r} must be an object.")
        unit_type = entry.get("unitType")
        motion_type = entry.get("motionType")
        display_text = entry.get("displayText")
        if (
            not isinstance(unit_type, str)
            or unit_type not in {"background", "fingerspelling", "word", "phrase"}
            or not isinstance(motion_type, str)
            or motion_type not in {"static", "dynamic"}
            or not isinstance(display_text, str)
            or ((unit_type == "background") != (label == NO_SIGN_LABEL))
            or (unit_type == "background" and motion_type != "static")
        ):
            raise CandidateValidationError(f"Candidate metadata for {label!r} has an invalid unit, motion, or display value.")
    if (
        metadata.get("labels") != labels
        or metadata.get("sequence_length") != manifest.get("sequenceLength")
        or metadata.get("feature_count") != manifest.get("featureDimension")
        or metadata.get("feature_version") != FEATURE_VERSION
        or metadata.get("reject_label") != NO_SIGN_LABEL
        or metadata.get("label_metadata") != label_metadata
    ):
        raise CandidateValidationError("Candidate metadata labels, sign metadata, or input dimensions do not match its vocabulary.")
    if metadata.get("sequence_length") != FEATURE_SEQUENCE_LENGTH or metadata.get("feature_count") != FEATURE_DIMENSION:
        raise CandidateValidationError("Candidate metadata input dimensions are incompatible with the interpreter.")

    minimum_signers = manifest.get("minimumSignersPerLabel")
    signer_counts = manifest.get("distinctSignersPerLabel")
    if not isinstance(minimum_signers, int) or isinstance(minimum_signers, bool) or minimum_signers < 1:
        raise CandidateValidationError("Candidate does not declare valid signer coverage requirements.")
    if not isinstance(signer_counts, dict) or any(
        not isinstance(signer_counts.get(label), int)
        or isinstance(signer_counts.get(label), bool)
        or signer_counts[label] < minimum_signers
        for label in labels
    ):
        raise CandidateValidationError("Candidate does not meet its signer coverage requirements.")

    manifest_evaluation = manifest.get("evaluation")
    if not isinstance(manifest_evaluation, dict):
        raise CandidateValidationError("Candidate manifest has no evaluation results.")
    if evaluation.get("labels") != labels:
        raise CandidateValidationError("Candidate evaluation labels do not match its vocabulary.")
    if evaluation.get("testMode") is not False:
        raise CandidateValidationError("Test-mode model candidates cannot be transferred.")
    for result, description in ((manifest_evaluation, "manifest"), (evaluation, "evaluation.json")):
        if result.get("acceptanceProfile") != "signar-standard-v1":
            raise CandidateValidationError(f"Candidate {description} does not declare the SignAR validation profile.")
        if result.get("deploymentEligible") is not True:
            raise CandidateValidationError(f"Candidate {description} does not mark the model eligible.")
        if result.get("evaluationScope") != "signer-independent":
            raise CandidateValidationError(f"Candidate {description} does not use leave-one-signer-out evaluation.")
        accuracy = result.get("testAccuracy")
        minimum_accuracy = result.get("minimumAccuracy")
        if (
            isinstance(minimum_accuracy, bool)
            or not isinstance(minimum_accuracy, (int, float))
            or not math.isfinite(minimum_accuracy)
            or minimum_accuracy < MINIMUM_TEST_ACCURACY
            or minimum_accuracy > 1
        ):
            raise CandidateValidationError(
                f"Candidate {description} must require at least {MINIMUM_TEST_ACCURACY:.0%} held-out accuracy."
            )
        if (
            isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(accuracy)
            or accuracy < minimum_accuracy
        ):
            raise CandidateValidationError(f"Candidate {description} did not meet its held-out accuracy requirement.")
        minimum_class_recall = result.get("minimumClassRecall")
        if (
            isinstance(minimum_class_recall, bool)
            or not isinstance(minimum_class_recall, (int, float))
            or not math.isfinite(minimum_class_recall)
            or minimum_class_recall < MINIMUM_CLASS_RECALL
        ):
            raise CandidateValidationError(
                f"Candidate {description} does not meet the {MINIMUM_CLASS_RECALL:.0%} per-label recall requirement."
            )
        recall_report = (
            {label: {"recall": recall} for label, recall in result.get("classRecall", {}).items()}
            if description == "manifest" and isinstance(result.get("classRecall"), dict)
            else evaluation.get("classificationReport")
        )
        failures = class_recall_failures(recall_report, labels)
        if failures:
            raise CandidateValidationError(
                f"Candidate {description} has held-out recall below {MINIMUM_CLASS_RECALL:.0%} for: "
                + ", ".join(failures)
                + "."
            )
        loso = result.get("leaveOneSignerOut")
        loso_accuracy = loso.get("meanAccuracy") if isinstance(loso, dict) else None
        if not isinstance(loso, dict) or not isinstance(loso.get("folds"), list) or len(loso["folds"]) < 5 or not isinstance(loso_accuracy, (int, float)) or not math.isfinite(loso_accuracy) or loso_accuracy < MINIMUM_TEST_ACCURACY:
            raise CandidateValidationError(f"Candidate {description} misses the {MINIMUM_TEST_ACCURACY:.0%} five-fold leave-one-signer-out target.")
        false_accept = result.get("noSignFalseAcceptRate")
        if not isinstance(false_accept, (int, float)) or not math.isfinite(false_accept) or false_accept > MINIMUM_NO_SIGN_FALSE_ACCEPT_RATE:
            raise CandidateValidationError(f"Candidate {description} has NO SIGN false accepts above {MINIMUM_NO_SIGN_FALSE_ACCEPT_RATE:.0%}.")
        if not isinstance(result.get("noSignSampleCount"), int) or result["noSignSampleCount"] < 10:
            raise CandidateValidationError(f"Candidate {description} needs at least 10 NO SIGN background samples.")
        no_sign_share = result.get("noSignShare")
        if not isinstance(no_sign_share, (int, float)) or not MINIMUM_NO_SIGN_SHARE <= no_sign_share <= MAXIMUM_NO_SIGN_SHARE:
            raise CandidateValidationError(f"Candidate {description} needs NO SIGN to make up {MINIMUM_NO_SIGN_SHARE:.0%}–{MAXIMUM_NO_SIGN_SHARE:.0%} of samples.")
        stress = result.get("stressTestsCorrectOrRejected")
        if not isinstance(stress, dict) or not stress or any(not isinstance(value, (int, float)) or value < MINIMUM_STRESS_ACCEPT_OR_REJECT for value in stress.values()):
            raise CandidateValidationError(f"Candidate {description} does not pass all {MINIMUM_STRESS_ACCEPT_OR_REJECT:.0%} stress-test checks.")
    if not math.isclose(
        manifest_evaluation["testAccuracy"],
        evaluation["testAccuracy"],
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise CandidateValidationError("Candidate evaluation results do not match the manifest.")
    if manifest_evaluation["evaluationScope"] != evaluation["evaluationScope"] or not math.isclose(
        manifest_evaluation["minimumAccuracy"],
        evaluation["minimumAccuracy"],
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise CandidateValidationError("Candidate evaluation requirements do not match the manifest.")
    if not math.isclose(
        manifest_evaluation["minimumClassRecall"],
        evaluation["minimumClassRecall"],
        rel_tol=0,
        abs_tol=1e-9,
    ) or manifest_evaluation["classRecall"] != evaluation["classRecall"]:
        raise CandidateValidationError("Candidate per-label recall results do not match the manifest.")

    try:
        model_config = model_document["modelTopology"]["model_config"]["config"]
        layers = model_config["layers"]
        input_names = model_config["input_layers"]
        output_names = model_config["output_layers"]
    except (KeyError, TypeError) as error:
        raise CandidateValidationError("Candidate model has no valid TensorFlow.js layer configuration.") from error
    if not isinstance(layers, list) or not isinstance(input_names, list) or not input_names or not isinstance(output_names, list) or not output_names:
        raise CandidateValidationError("Candidate model is missing its input or output layer.")
    def endpoint_layer_name(entry):
        if isinstance(entry, str):
            return entry
        if isinstance(entry, list):
            return next((name for item in entry if (name := endpoint_layer_name(item))), None)
        return None

    input_name = endpoint_layer_name(input_names[0])
    output_name = endpoint_layer_name(output_names[0])
    input_layer = next((layer for layer in layers if isinstance(layer, dict) and layer.get("name") == input_name), None)
    output_layer = next((layer for layer in layers if isinstance(layer, dict) and layer.get("name") == output_name), None)
    input_layer_config = input_layer.get("config") if input_layer else None
    output_layer_config = output_layer.get("config") if output_layer else None
    if not isinstance(input_layer_config, dict) or not isinstance(output_layer_config, dict):
        raise CandidateValidationError("Candidate model input or output layer configuration is invalid.")
    input_shape = (
        input_layer_config.get("batchInputShape")
        or input_layer_config.get("batch_shape")
        or input_layer_config.get("batch_input_shape")
    )
    output_units = output_layer_config.get("units")
    output_class = output_layer.get("class_name") if output_layer else None
    if output_units is None:
        tensor_shapes = []

        def collect_tensor_shapes(value):
            if isinstance(value, dict):
                if value.get("class_name") == "__keras_tensor__":
                    shape = value.get("config", {}).get("shape")
                    if isinstance(shape, list) and shape:
                        tensor_shapes.append(shape)
                for child in value.values():
                    collect_tensor_shapes(child)
            elif isinstance(value, list):
                for child in value:
                    collect_tensor_shapes(child)

        collect_tensor_shapes(output_layer.get("inbound_nodes", []))
        if tensor_shapes and isinstance(tensor_shapes[-1][-1], int):
            output_units = tensor_shapes[-1][-1]
    if output_units != len(labels) and output_class in {"Softmax", "Activation"}:
        output_config = output_layer_config
        is_softmax = output_class == "Softmax" or output_config.get("activation") == "softmax"
        output_nodes = output_layer.get("inbound_nodes", [])
        source_names = []
        for node in output_nodes:
            if not isinstance(node, list):
                continue
            for connection in node:
                if isinstance(connection, list) and connection and isinstance(connection[0], str):
                    source_names.append(connection[0])
        source_layer = next((layer for layer in layers if isinstance(layer, dict) and layer.get("name") in source_names), None)
        source_names = []
        if source_layer and source_layer.get("class_name") == "Concatenate":
            for node in source_layer.get("inbound_nodes", []):
                if not isinstance(node, list):
                    continue
                for connection in node:
                    if isinstance(connection, list) and connection and isinstance(connection[0], str):
                        source_names.append(connection[0])
        source_units = sum(
            int(layer.get("config", {}).get("units", 0))
            for layer in layers
            if isinstance(layer, dict) and layer.get("name") in source_names
        )
        if is_softmax and source_units == len(labels):
            output_units = len(labels)
    if input_shape != [None, metadata["sequence_length"], metadata["feature_count"]] or output_units != len(labels):
        raise CandidateValidationError("Candidate model input or output dimensions do not match its manifest.")

    weights_manifest = model_document.get("weightsManifest")
    if not isinstance(weights_manifest, list) or not weights_manifest:
        raise CandidateValidationError("Candidate model does not reference any weight files.")
    weight_files: dict[str, Path] = {}
    output_directory = OUTPUT_DIR.resolve()
    for group in weights_manifest:
        paths = group.get("paths") if isinstance(group, dict) else None
        if not isinstance(paths, list) or not paths:
            raise CandidateValidationError("Candidate model has an invalid weight-file manifest.")
        for relative_path in paths:
            if not isinstance(relative_path, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", relative_path) or relative_path in {".", ".."}:
                raise CandidateValidationError("Candidate model contains an unsafe weight-file path.")
            source = OUTPUT_DIR / relative_path
            if not source.is_file() or source.resolve().parent != output_directory:
                raise CandidateValidationError(f"Candidate is missing a safe weight file: {relative_path}")
            weight_files[relative_path] = source

    return manifest, evaluation, model_document, metadata, weight_files


def _candidate_status() -> tuple[bool, str]:
    if TRAINING_LOCK.locked():
        return False, "A training run is active. Wait for it to finish before transferring a model."
    try:
        _validated_candidate()
    except (CandidateValidationError, OSError, TypeError, OverflowError) as error:
        return False, str(error)
    return True, "Candidate passed automated checks. Review its sign labels and evaluation before transferring it to the interpreter."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deploy_candidate() -> dict:
    manifest, evaluation, model_document, metadata, weight_files = _validated_candidate()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def strip_training_regularizers(value: object) -> None:
        if isinstance(value, dict):
            for key in list(value):
                if key.endswith("_regularizer"):
                    del value[key]
                else:
                    strip_training_regularizers(value[key])
        elif isinstance(value, list):
            for item in value:
                strip_training_regularizers(item)

    # Regularizers affect training only and are not needed for inference.
    strip_training_regularizers(model_document["modelTopology"])

    # Keras 3 changed InputLayer and graph connection serialization. Restore
    # the Keras 2 topology format expected by TensorFlow.js 4.22.
    topology_config = model_document["modelTopology"]["model_config"]["config"]
    for layer in topology_config["layers"]:
        if layer.get("class_name") != "InputLayer":
            input_config = layer.get("config", {})
        else:
            input_config = layer["config"]
            input_shape = (
                input_config.get("batchInputShape")
                or input_config.get("batch_shape")
                or input_config.get("batch_input_shape")
            )
            if input_shape is None:
                raise CandidateValidationError(
                    "Candidate InputLayer has no batchInputShape or inputShape."
                )
            input_config["batchInputShape"] = input_shape
            input_config.pop("batch_shape", None)
            input_config.pop("batch_input_shape", None)

        dtype = input_config.get("dtype")
        if isinstance(dtype, dict) and dtype.get("class_name") == "DTypePolicy":
            input_config["dtype"] = dtype.get("config", {}).get("name", "float32")

        inbound_nodes = layer.get("inbound_nodes", [])
        if any(isinstance(node, dict) for node in inbound_nodes):
            normalized_nodes = []
            for node in inbound_nodes:
                if not isinstance(node, dict):
                    normalized_nodes.append(node)
                    continue
                connections = []

                def collect_keras_history(value: object) -> None:
                    if isinstance(value, dict):
                        if value.get("class_name") == "__keras_tensor__":
                            history = value.get("config", {}).get("keras_history")
                            if isinstance(history, list) and len(history) == 3:
                                connections.append([*history, {}])
                            return
                        for child in value.values():
                            collect_keras_history(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect_keras_history(child)

                collect_keras_history(node.get("args", []))
                normalized_nodes.append(connections)
            layer["inbound_nodes"] = normalized_nodes

    for endpoint_name in ("input_layers", "output_layers"):
        endpoints = topology_config.get(endpoint_name)
        if (
            isinstance(endpoints, list)
            and len(endpoints) == 3
            and isinstance(endpoints[0], str)
        ):
            topology_config[endpoint_name] = [endpoints]

    deployed_names: dict[str, str] = {}
    for relative_path, source in weight_files.items():
        digest = _sha256_file(source)
        suffix = source.suffix or ".bin"
        deployed_names[relative_path] = f"weights-{digest}{suffix}"
    for group in model_document["weightsManifest"]:
        group["paths"] = [deployed_names[path] for path in group["paths"]]

    model_bytes = json.dumps(model_document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    model_name = "model-" + hashlib.sha256(model_bytes).hexdigest()[:20] + ".json"
    manifest["modelPath"] = "./model/" + model_name
    manifest["metadataPath"] = "./model/metadata.json"
    vocabulary_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    evaluation_bytes = (json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    with tempfile.TemporaryDirectory(prefix=".deploy-", dir=MODEL_DIR) as temporary_directory:
        staging = Path(temporary_directory)
        for relative_path, source in weight_files.items():
            shutil.copyfile(source, staging / deployed_names[relative_path])
        (staging / model_name).write_bytes(model_bytes)
        (staging / "vocabulary.json").write_bytes(vocabulary_bytes)
        (staging / "evaluation.json").write_bytes(evaluation_bytes)
        (staging / "metadata.json").write_bytes(metadata_bytes)

        existing_files = [
            name for name in ("vocabulary.json", "evaluation.json", "metadata.json")
            if (MODEL_DIR / name).is_file()
        ]
        backup_directory = None
        if existing_files:
            backup_root = MODEL_DIR / "deployment-backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_directory = backup_root / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
            )
            backup_directory.mkdir()
            for name in existing_files:
                shutil.copy2(MODEL_DIR / name, backup_directory / name)

        for relative_path, deployed_name in deployed_names.items():
            staged_weight = staging / deployed_name
            target_weight = MODEL_DIR / deployed_name
            if target_weight.is_file():
                expected_hash = deployed_name[len("weights-"):].rsplit(".", 1)[0]
                if _sha256_file(target_weight) != expected_hash:
                    raise OSError(f"Existing model asset failed integrity verification: {deployed_name}")
            else:
                os.replace(staged_weight, target_weight)

        staged_model = staging / model_name
        target_model = MODEL_DIR / model_name
        if target_model.is_file():
            if target_model.read_bytes() != model_bytes:
                raise OSError(f"Existing model file failed integrity verification: {model_name}")
        else:
            os.replace(staged_model, target_model)
        os.replace(staging / "evaluation.json", MODEL_DIR / "evaluation.json")
        os.replace(staging / "metadata.json", MODEL_DIR / "metadata.json")
        os.replace(staging / "vocabulary.json", MODEL_DIR / "vocabulary.json")

    return {
        "deployed": True,
        "modelPath": manifest["modelPath"],
        "labelCount": len(manifest["labels"]),
        "destination": str(MODEL_DIR),
        "backupDirectory": str(backup_directory) if backup_directory else None,
    }


class StudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE_ROOT), **kwargs)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _is_local_request(self) -> bool:
        host = (urlsplit("//" + (self.headers.get("Host") or "")).hostname or "").lower()
        if host not in {"localhost", "127.0.0.1", "::1"}:
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        origin_host = (urlsplit(origin).hostname or "").lower()
        return origin_host in {"localhost", "127.0.0.1", "::1"}

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if route == "/api/training/progress":
            if not self._is_local_request():
                self._send_json(403, {"error": "The local training service only accepts this device."})
                return
            self._send_json(200, _training_progress_snapshot())
            return
        if route == "/api/training/status":
            if not self._is_local_request():
                self._send_json(403, {"error": "The local training service only accepts this device."})
                return
            candidate_deployable, candidate_message = _candidate_status()
            self._send_json(
                200,
                {
                    "available": True,
                    "trainer": "train.ps1",
                    "candidateDeployable": candidate_deployable,
                    "candidateMessage": candidate_message,
                    "hasTrainingHistory": (OUTPUT_DIR / "training-state.json").is_file(),
                },
            )
            return
        super().do_GET()

    def _deploy_model(self) -> None:
        if not self._is_local_request():
            self._send_json(403, {"error": "The local training service only accepts this device."})
            return
        if not TRAINING_LOCK.acquire(blocking=False):
            self._send_json(409, {"error": "Training or model transfer is already running in this local studio."})
            return
        try:
            result = _deploy_candidate()
            self._send_json(200, result)
        except CandidateValidationError as error:
            self._send_json(409, {"error": str(error)})
        except OSError as error:
            self._send_json(500, {"error": f"Could not transfer the model files: {error}"})
        finally:
            TRAINING_LOCK.release()

    def do_POST(self) -> None:
        route = urlsplit(self.path).path
        if route == "/api/model/deploy":
            self._deploy_model()
            return
        if route != "/api/training/run":
            self._send_json(404, {"error": "Unknown local training endpoint."})
            return
        if not self._is_local_request():
            self._send_json(403, {"error": "The local training service only accepts this device."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "The request size is invalid."})
            return
        if content_length < 1 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "The dataset is empty or exceeds the local upload limit."})
            return
        job_id = self.headers.get("X-Training-Job-ID", "")
        if not re.fullmatch(r"[a-zA-Z0-9-]{8,64}", job_id):
            self._send_json(400, {"error": "A valid training job ID is required."})
            return

        if not TRAINING_LOCK.acquire(blocking=False):
            self._send_json(409, {"error": "Training is already running in this local studio."})
            return

        _set_training_progress(
            active=True,
            percent=0,
            stage="Preparing the local trainer",
            epoch=0,
            totalEpochs=0,
            batch=0,
            batchesPerEpoch=0,
            completed=False,
            failed=False,
            jobId=job_id,
            result=None,
            error=None,
            output="",
        )
        try:
            TOOLKIT_DIR.mkdir(parents=True, exist_ok=True)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            temporary_dataset = DATASET_PATH.with_name(DATASET_PATH.name + ".uploading")
            try:
                with temporary_dataset.open("wb") as dataset_file:
                    remaining = content_length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise OSError("The dataset upload ended before all bytes arrived.")
                        dataset_file.write(chunk)
                        remaining -= len(chunk)
                temporary_dataset.replace(DATASET_PATH)
            except OSError:
                if temporary_dataset.exists():
                    temporary_dataset.unlink()
                raise
            Thread(target=_run_training, daemon=True).start()
        except (OSError, ValueError, RuntimeError) as error:
            _fail_training_progress("Training failed", f"Could not start the local trainer: {error}")
            TRAINING_LOCK.release()
            self._send_json(500, {"error": f"Could not start the local trainer: {error}"})
            return
        self._send_json(202, {"started": True})


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8001), StudioHandler)
    print("SignAR training studio: http://localhost:8001/SignAR_FSL_Training_Tools_Video_Import_v5/training/collect.html")
    print("This server is local to this computer. Keep this window open while using the studio.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the SignAR training studio.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
