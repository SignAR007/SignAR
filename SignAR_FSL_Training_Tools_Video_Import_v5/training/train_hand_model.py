#!/usr/bin/env python3
"""Train and evaluate the SignAR holistic-v4 decoder with signer-held-out checks."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import types
from collections import Counter
from pathlib import Path

# tensorflowjs 4.22 expects the Keras 2 model config format. The training
# environment includes tf-keras for this compatibility path.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from evaluation import (
    MAXIMUM_NO_SIGN_SHARE,
    MINIMUM_CLASS_RECALL,
    MINIMUM_NO_SIGN_FALSE_ACCEPT_RATE,
    MINIMUM_NO_SIGN_SHARE,
    MINIMUM_STRESS_ACCEPT_OR_REJECT,
    MINIMUM_TEST_ACCURACY,
)

FEATURE_VERSION = "signar-fsl-holistic-v4"
HAND_V3_FEATURE_VERSION = "signar-fsl-hand-v3"
LEGACY_FEATURE_VERSION = "signar-fsl-456-v2"
FEATURES = 422
SEQ_LEN = 30
HAND_V3_FEATURES = 68
HAND_V3_SEQ_LEN = 10
LEGACY_FEATURES = 456
MISSING_VALUE = -3.0
LEFT_HAND_OFFSET = 0
RIGHT_HAND_OFFSET = 71
POSE_OFFSET = 142
FACE_OFFSET = 274
PRESENCE_OFFSET = 418
REJECT = "NO SIGN"
MIN_NO_SIGN_SHARE = MINIMUM_NO_SIGN_SHARE
MAX_NO_SIGN_SHARE = MAXIMUM_NO_SIGN_SHARE
MIN_LOSO_ACCURACY = MINIMUM_TEST_ACCURACY
MIN_NO_SIGN_FAR = MINIMUM_NO_SIGN_FALSE_ACCEPT_RATE
MIN_STRESS_ACCEPT_OR_REJECT = MINIMUM_STRESS_ACCEPT_OR_REJECT
CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.18
LABEL_SMOOTHING = 0.05


def json_default(value: object):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Training artifacts cannot contain NaN or infinite JSON values.")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ordered_labels(values: object) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("Dataset labels must be a list.")
    labels = [str(label).strip().upper() for label in values if str(label).strip()]
    if not labels or REJECT not in labels:
        raise ValueError("Dataset must contain the NO SIGN background label.")
    if len(set(labels)) != len(labels):
        raise ValueError("Dataset labels must contain each observed label exactly once.")

    alphabet = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    ordered = [REJECT]
    ordered.extend(label for label in alphabet if label in labels)
    ordered.extend(sorted((label for label in labels if label not in ordered), key=str.casefold))
    return ordered


def resample(frames: object) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURES or not len(values):
        raise ValueError(f"Expected a non-empty [time, {FEATURES}] holistic-v4 window; got {values.shape}.")
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 3.0001:
        raise ValueError("Features must be finite and within the holistic-v4 [-3, 3] range.")
    if len(values) == SEQ_LEN:
        return values
    positions = np.linspace(0, len(values) - 1, SEQ_LEN, dtype=np.float32)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (positions - lower)[:, None]
    first, second = values[lower], values[upper]
    output = first * (1 - weight) + second * weight
    missing = (first == MISSING_VALUE) | (second == MISSING_VALUE)
    both_missing = (first == MISSING_VALUE) & (second == MISSING_VALUE)
    nearest = np.where(weight < 0.5, first, second)
    return np.where(both_missing, MISSING_VALUE, np.where(missing, nearest, output))


def canonicalize_hand_v3_frame(frame: object) -> list[float]:
    values = np.asarray(frame, dtype=np.float32)
    if values.shape != (HAND_V3_FEATURES,) or not np.isfinite(values).all():
        raise ValueError("Hand-v3 frames must contain 68 finite features.")
    output = np.full(FEATURES, MISSING_VALUE, dtype=np.float32)
    output[PRESENCE_OFFSET:] = 0
    if np.all(values == 0):
        return output.tolist()
    output[RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 63] = values[:63]
    output[RIGHT_HAND_OFFSET + 63:RIGHT_HAND_OFFSET + 68] = values[63:]
    output[PRESENCE_OFFSET + 1] = 1
    return output.tolist()


def canonicalize_legacy_frame(frame: object) -> list[float]:
    values = np.asarray(frame, dtype=np.float32)
    if values.shape != (456,) or not np.isfinite(values).all():
        raise ValueError("Legacy frames must contain 456 finite features.")
    result = np.full(FEATURES, MISSING_VALUE, dtype=np.float32)
    result[PRESENCE_OFFSET:] = 0
    pose = values[:132].reshape(33, 4)
    for index, point in enumerate(pose):
        if np.any(point[:3] > -2.75):
            result[POSE_OFFSET + index * 4:POSE_OFFSET + index * 4 + 3] = np.clip(point[:3], -3, 3)
            result[POSE_OFFSET + index * 4 + 3] = np.clip(point[3], 0, 1)
            result[PRESENCE_OFFSET + 2] = 1
    chains = ((1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
    for slot in (0, 1):
        base = 132 + slot * 126
        output_base = LEFT_HAND_OFFSET if slot == 0 else RIGHT_HAND_OFFSET
        points = values[base:base + 126].reshape(21, 6)[:, 3:6]
        valid = np.isfinite(points).all(axis=1) & (points[:, 0] > -2.75)
        if not valid.any():
            continue
        for index, point in enumerate(points):
            if not valid[index]:
                continue
            target = output_base + index * 3
            result[target:target + 3] = np.clip(point, -3, 3)
            if slot == 0:
                result[target] *= -1
        for angle_index, (first, joint, last) in enumerate(chains):
            if not (valid[first] and valid[joint] and valid[last]):
                continue
            a = result[output_base + first * 3:output_base + first * 3 + 3] - result[output_base + joint * 3:output_base + joint * 3 + 3]
            b = result[output_base + last * 3:output_base + last * 3 + 3] - result[output_base + joint * 3:output_base + joint * 3 + 3]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator:
                result[output_base + 63 + angle_index] = math.acos(float(np.clip(np.dot(a, b) / denominator, -1, 1))) / math.pi
        result[output_base + 68:output_base + 71] = 0
        result[PRESENCE_OFFSET + slot] = 1
    old_face = values[384:456].reshape(24, 3)
    for index, point in enumerate(old_face):
        if np.all(point > -2.75):
            result[FACE_OFFSET + index * 3:FACE_OFFSET + index * 3 + 3] = np.clip(point, -3, 3)
            result[PRESENCE_OFFSET + 3] = 1
    return result.tolist()


def load_dataset(path: Path) -> tuple[list[str], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_version = payload.get("featureVersion")
    legacy = source_version == LEGACY_FEATURE_VERSION and payload.get("featureDimension") == LEGACY_FEATURES
    hand_v3 = source_version == HAND_V3_FEATURE_VERSION and payload.get("featureDimension") == HAND_V3_FEATURES
    holistic = source_version == FEATURE_VERSION and payload.get("featureDimension") == FEATURES
    if not (legacy or hand_v3 or holistic):
        raise ValueError(f"Dataset must use {FEATURE_VERSION}/{FEATURES}, {HAND_V3_FEATURE_VERSION}/{HAND_V3_FEATURES}, or the supported legacy 456 schema.")
    if payload.get("featuresResampled") is not True:
        raise ValueError("Dataset feature windows must be marked resampled.")
    if legacy and payload.get("sequenceLength") != 30:
        raise ValueError("Legacy 456-feature windows must declare sequenceLength 30.")
    if hand_v3 and payload.get("sequenceLength") != HAND_V3_SEQ_LEN:
        raise ValueError(f"Hand-v3 windows must declare sequenceLength {HAND_V3_SEQ_LEN}.")
    if holistic and payload.get("sequenceLength") != SEQ_LEN:
        raise ValueError(f"Holistic-v4 windows must be resampled to {SEQ_LEN} frames.")
    labels = ordered_labels(payload.get("labels"))
    label_set = set(labels)
    samples = []
    ids = set()
    for item in payload.get("samples", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().upper()
        signer = str(item.get("signerId", "")).strip()
        sample_id = str(item.get("id", ""))
        if label not in label_set or not signer or not sample_id or sample_id in ids:
            continue
        if item.get("verified") is not True or item.get("consentRecorded") is not True:
            continue
        if len(str(item.get("verificationNotes", "")).strip()) < 5:
            continue
        try:
            frames = item.get("features")
            if legacy:
                frames = [canonicalize_legacy_frame(frame) for frame in frames]
            elif hand_v3:
                frames = [canonicalize_hand_v3_frame(frame) for frame in frames]
            sequence = resample(frames)
        except (ValueError, TypeError):
            continue
        if label != REJECT and not np.any(sequence[:, PRESENCE_OFFSET:PRESENCE_OFFSET + 4] >= 0.5):
            continue
        ids.add(sample_id)
        samples.append({
            "id": sample_id,
            "label": label,
            "signer": signer,
            "sourceType": str(item.get("sourceType", "")),
            "unitType": str(item.get("unitType", "background" if label == REJECT else "fingerspelling" if len(label) == 1 and label.isalpha() else "word")),
            "motionType": str(item.get("motionType", "dynamic" if label in {"J", "Z"} else "static")),
            "displayText": str(item.get("displayText", "")).strip(),
            "x": sequence,
        })
    if not samples:
        raise ValueError("No verified, consented holistic sign samples were found.")
    return labels, samples


def readiness(labels: list[str], samples: list[dict]) -> dict:
    signers = sorted({sample["signer"] for sample in samples})
    signer_class_counts = {
        signer: Counter(sample["label"] for sample in samples if sample["signer"] == signer)
        for signer in signers
    }
    live_counts = {
        signer: sum(sample["sourceType"] == "camera" for sample in samples if sample["signer"] == signer)
        for signer in signers
    }
    source_counts = Counter(sample["sourceType"] for sample in samples)
    live_label_counts = Counter(
        sample["label"] for sample in samples if sample["sourceType"] == "camera"
    )
    class_signers = {
        label: sum(counts[label] > 0 for counts in signer_class_counts.values())
        for label in labels
    }
    reject_share = sum(sample["label"] == REJECT for sample in samples) / len(samples)
    no_sign_count = sum(sample["label"] == REJECT for sample in samples)
    return {
        "distinctSigners": len(signers),
        "distinctSignersPerLabel": class_signers,
        "liveCameraSamplesPerSigner": live_counts,
        "sampleSourceCounts": dict(sorted(source_counts.items())),
        "liveCameraSamplesPerLabel": {
            label: live_label_counts[label] for label in labels
        },
        "noSignShare": reject_share,
        "noSignSampleCount": no_sign_count,
        "allLabelsAtLeastFiveSigners": all(class_signers[label] >= 5 for label in labels),
        "ready": (
            len(signers) >= 5
            and all(class_signers[label] >= 5 for label in labels)
            and no_sign_count >= 10
            and MIN_NO_SIGN_SHARE <= reject_share <= MAX_NO_SIGN_SHARE
        ),
    }


def augment_training_data(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    augmented = []
    for sequence in x:
        sample = sequence.copy()
        gamma = rng.uniform(0.8, 1.25)
        positions = np.linspace(0, SEQ_LEN - 1, SEQ_LEN, dtype=np.float32) ** gamma
        positions *= (SEQ_LEN - 1) / max(1, positions[-1])
        lo = np.floor(positions).astype(int)
        hi = np.minimum(lo + 1, SEQ_LEN - 1)
        weight = (positions - lo)[:, None]
        first, second = sample[lo], sample[hi]
        blended = first * (1 - weight) + second * weight
        missing = (first == MISSING_VALUE) | (second == MISSING_VALUE)
        both_missing = (first == MISSING_VALUE) & (second == MISSING_VALUE)
        nearest = np.where(weight < 0.5, first, second)
        sample = np.where(both_missing, MISSING_VALUE, np.where(missing, nearest, blended))

        if rng.random() < 0.5:
            left = sample[:, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 71].copy()
            right = sample[:, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 71].copy()
            sample[:, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 71] = right
            sample[:, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 71] = left
            left_present = sample[:, PRESENCE_OFFSET].copy()
            sample[:, PRESENCE_OFFSET] = sample[:, PRESENCE_OFFSET + 1]
            sample[:, PRESENCE_OFFSET + 1] = left_present
            for base in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
                xyz = sample[:, base:base + 63].reshape(SEQ_LEN, 21, 3)
                x_values = xyz[:, :, 0]
                x_values[:] = np.where(x_values == MISSING_VALUE, MISSING_VALUE, -x_values)
                wrist_x = sample[:, base + 68]
                wrist_x[:] = np.where(wrist_x == MISSING_VALUE, MISSING_VALUE, -wrist_x)
            pose_x = sample[:, POSE_OFFSET:FACE_OFFSET].reshape(SEQ_LEN, 33, 4)[:, :, 0]
            pose_x[:] = np.where(pose_x == MISSING_VALUE, MISSING_VALUE, -pose_x)
            face_x = sample[:, FACE_OFFSET:PRESENCE_OFFSET].reshape(SEQ_LEN, 48, 3)[:, :, 0]
            face_x[:] = np.where(face_x == MISSING_VALUE, MISSING_VALUE, -face_x)

        scale = rng.uniform(0.85, 1.15)
        angle = math.radians(rng.uniform(-15, 15))
        cosine, sine = math.cos(angle), math.sin(angle)
        coordinate_groups = []
        for base in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
            coordinate_groups.append(sample[:, base:base + 63].reshape(SEQ_LEN, 21, 3))
            coordinate_groups.append(sample[:, base + 68:base + 71].reshape(SEQ_LEN, 1, 3))
        coordinate_groups.append(sample[:, POSE_OFFSET:FACE_OFFSET].reshape(SEQ_LEN, 33, 4)[:, :, :3])
        coordinate_groups.append(sample[:, FACE_OFFSET:PRESENCE_OFFSET].reshape(SEQ_LEN, 48, 3))
        for xyz in coordinate_groups:
            point_present = ~np.all(xyz == MISSING_VALUE, axis=-1)
            x_values, y_values = xyz[:, :, 0].copy(), xyz[:, :, 1].copy()
            transformed_x = scale * (cosine * x_values - sine * y_values)
            transformed_y = scale * (sine * x_values + cosine * y_values)
            noise = rng.normal(0, 0.015, size=xyz.shape).astype(np.float32)
            for axis, transformed in enumerate((transformed_x, transformed_y, xyz[:, :, 2] * scale)):
                updated = np.clip(transformed + noise[:, :, axis], -3, 3)
                xyz[:, :, axis] = np.where(point_present, updated, MISSING_VALUE)
        for base in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
            angles = sample[:, base + 63:base + 68]
            known = angles != MISSING_VALUE
            angles[:] = np.where(known, np.clip(angles + rng.normal(0, 0.01, size=angles.shape), 0, 1), MISSING_VALUE)

        if rng.random() < 0.65:
            dropped = rng.choice(SEQ_LEN, size=int(rng.integers(1, 3)), replace=False)
            sample[dropped, :PRESENCE_OFFSET] = MISSING_VALUE
            sample[dropped, PRESENCE_OFFSET:] = 0
        augmented.append(sample)
    return np.concatenate([x, np.asarray(augmented, dtype=np.float32)]), np.concatenate([y, y])


def build_model(labels: list[str], seed: int = 42):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(SEQ_LEN, FEATURES), name="holistic_sequence")
    regularizer = tf.keras.regularizers.l2(1e-4)
    handshape = tf.keras.layers.Dense(
        48,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=regularizer,
        name="spatial_cue_encoder",
    )(inputs)
    handshape = tf.keras.layers.GlobalAveragePooling1D(name="static_shape_pool")(handshape)
    movement = tf.keras.layers.GRU(
        24,
        dropout=0.1,
        reset_after=False,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 2),
        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed + 3),
        seed=seed + 4,
        name="temporal_encoder",
    )(inputs)
    features = tf.keras.layers.Concatenate(name="shape_and_motion")([handshape, movement])
    logits = tf.keras.layers.Dense(
        len(labels),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
        name="fsl_logits",
    )(features)
    outputs = tf.keras.layers.Softmax(name="fsl_class_probabilities")(logits)
    model = tf.keras.Model(inputs, outputs, name="signar_holistic_v4_decoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def load_tensorflowjs_converter():
    """Load the Keras exporter without requiring the unused TFDF graph converter on Windows."""
    if (
        os.name == "nt"
        and "tensorflow_decision_forests" not in sys.modules
        and importlib.util.find_spec("tensorflow_decision_forests") is None
    ):
        # tensorflowjs imports every converter at package load time. This training
        # path exports a Keras Layers model and never converts TFDF SavedModels.
        sys.modules.setdefault("tensorflow_decision_forests", types.ModuleType("tensorflow_decision_forests"))
    import tensorflowjs

    return tensorflowjs


def fit_model(labels: list[str], samples: list[dict], seed: int, epochs: int, batch_size: int, validation: tuple[np.ndarray, np.ndarray] | None = None):
    epochs = int(epochs)
    batch_size = int(batch_size)
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    x = np.asarray([sample["x"] for sample in samples], dtype=np.float32)
    indexes = {label: index for index, label in enumerate(labels)}
    y = np.asarray([indexes[sample["label"]] for sample in samples], dtype=np.int32)
    x, y = augment_training_data(x, y, seed)
    smooth = np.full((len(y), len(labels)), LABEL_SMOOTHING / len(labels), dtype=np.float32)
    smooth[np.arange(len(y)), y] += 1 - LABEL_SMOOTHING
    model = build_model(labels, seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss" if validation else "loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss" if validation else "loss", factor=0.5, patience=3, min_lr=1e-5),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    fit_kwargs = {"epochs": epochs, "batch_size": min(batch_size, len(y)), "shuffle": True, "verbose": 0, "callbacks": callbacks}
    if validation:
        vx, vy = validation
        vy_onehot = tf.keras.utils.to_categorical(vy, len(labels))
        fit_kwargs["validation_data"] = (vx, vy_onehot)
    history = model.fit(x, smooth, **fit_kwargs)
    return model, history.history


def softmax_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-8, 1))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, y: np.ndarray) -> float:
    best_temperature, best_loss = 1.0, float("inf")
    for temperature in np.linspace(0.5, 4.0, 351):
        scaled = softmax_temperature(probabilities, float(temperature))
        loss = -np.log(np.clip(scaled[np.arange(len(y)), y], 1e-8, 1)).mean()
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), float(loss)
    return best_temperature


def decode(probabilities: np.ndarray, labels: list[str]) -> tuple[str, bool]:
    order = np.argsort(probabilities)[::-1]
    best = int(order[0])
    margin = float(probabilities[best] - probabilities[int(order[1])])
    if labels[best] == REJECT or probabilities[best] < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
        return REJECT, True
    return labels[best], False


def expected_calibration_error(probabilities: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    error = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def transform_stress(x: np.ndarray, mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = x.copy()
    if mode == "slot_swap":
        left = values[:, :, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 71].copy()
        values[:, :, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 71] = values[:, :, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 71]
        values[:, :, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 71] = left
        left_present = values[:, :, PRESENCE_OFFSET].copy()
        values[:, :, PRESENCE_OFFSET] = values[:, :, PRESENCE_OFFSET + 1]
        values[:, :, PRESENCE_OFFSET + 1] = left_present
    elif mode == "missing_face":
        values[:, :, FACE_OFFSET:PRESENCE_OFFSET] = MISSING_VALUE
        values[:, :, PRESENCE_OFFSET + 3] = 0
    elif mode == "missing_pose":
        values[:, :, POSE_OFFSET:FACE_OFFSET] = MISSING_VALUE
        values[:, :, PRESENCE_OFFSET + 2] = 0
    elif mode == "static_hold":
        values[:] = values[:, SEQ_LEN // 2:SEQ_LEN // 2 + 1]
    elif mode == "dropped_frames":
        mask = rng.random(values.shape[:2]) < 0.25
        values[mask, :PRESENCE_OFFSET] = MISSING_VALUE
        values[mask, PRESENCE_OFFSET:] = 0
    elif mode in {"jitter", "scale_rotation"}:
        groups = []
        for base in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
            groups.extend((
                values[:, :, base:base + 63].reshape(len(values), SEQ_LEN, 21, 3),
                values[:, :, base + 68:base + 71].reshape(len(values), SEQ_LEN, 1, 3),
            ))
        groups.extend((
            values[:, :, POSE_OFFSET:FACE_OFFSET].reshape(len(values), SEQ_LEN, 33, 4)[:, :, :, :3],
            values[:, :, FACE_OFFSET:PRESENCE_OFFSET].reshape(len(values), SEQ_LEN, 48, 3),
        ))
        for xyz in groups:
            present = ~np.all(xyz == MISSING_VALUE, axis=-1)
            if mode == "jitter":
                noise = rng.normal(0, 0.025, size=xyz.shape).astype(np.float32)
                xyz[:] = np.where(present[:, :, :, None], np.clip(xyz + noise, -3, 3), MISSING_VALUE)
            else:
                scale = rng.uniform(0.85, 1.15, size=(len(values), 1, 1)).astype(np.float32)
                theta = rng.uniform(-math.pi / 12, math.pi / 12, size=(len(values), 1, 1)).astype(np.float32)
                old_x, old_y = xyz[:, :, :, 0].copy(), xyz[:, :, :, 1].copy()
                xyz[:, :, :, 0] = np.where(present, scale * (np.cos(theta) * old_x - np.sin(theta) * old_y), MISSING_VALUE)
                xyz[:, :, :, 1] = np.where(present, scale * (np.sin(theta) * old_x + np.cos(theta) * old_y), MISSING_VALUE)
                xyz[:, :, :, 2] = np.where(present, scale * xyz[:, :, :, 2], MISSING_VALUE)
                np.clip(xyz, -3, 3, out=xyz)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("fsl-dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("trained-model"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--incremental", action="store_true", help="Accepted for CLI compatibility; holistic-v4 performs a reproducible full fit.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        labels, samples = load_dataset(args.dataset)
        status = readiness(labels, samples)

        signers = sorted({sample["signer"] for sample in samples})
        label_indexes = {label: index for index, label in enumerate(labels)}
        all_true, all_pred, all_prob = [], [], []
        fold_reports, fold_temperatures, stress_results = [], [], {}
        print("SIGNAR_STAGE evaluating", flush=True)
        for fold_index, held_out in enumerate(signers):
            remaining = [signer for signer in signers if signer != held_out]
            if len(remaining) < 4:
                raise ValueError("Leave-one-signer-out requires at least five signers.")
            calibration_signer = remaining[(fold_index + args.seed) % len(remaining)]
            training_signers = [signer for signer in remaining if signer != calibration_signer]
            fit_samples = [sample for sample in samples if sample["signer"] in training_signers]
            calibration = [sample for sample in samples if sample["signer"] == calibration_signer]
            test = [sample for sample in samples if sample["signer"] == held_out]
            train_x = np.asarray([sample["x"] for sample in fit_samples], dtype=np.float32)
            train_y = np.asarray([label_indexes[sample["label"]] for sample in fit_samples], dtype=np.int32)
            cal_x = np.asarray([sample["x"] for sample in calibration], dtype=np.float32)
            cal_y = np.asarray([label_indexes[sample["label"]] for sample in calibration], dtype=np.int32)
            test_x = np.asarray([sample["x"] for sample in test], dtype=np.float32)
            test_y = np.asarray([label_indexes[sample["label"]] for sample in test], dtype=np.int32)
            if set(train_y) != set(range(len(labels))) or set(cal_y) != set(range(len(labels))) or set(test_y) != set(range(len(labels))):
                raise ValueError(f"Signer fold {held_out} does not contain every class in train, calibration, and test partitions.")
            model, _history = fit_model(labels, fit_samples, args.seed + fold_index, args.epochs, args.batch_size, (cal_x, cal_y))
            calibration_prob = model.predict(cal_x, verbose=0)
            temperature = fit_temperature(calibration_prob, cal_y)
            test_prob = softmax_temperature(model.predict(test_x, verbose=0), temperature)
            test_pred = test_prob.argmax(axis=1)
            fold_accuracy = float((test_pred == test_y).mean())
            fold_temperatures.append(temperature)
            fold_reports.append({
                "heldOutSigner": held_out,
                "calibrationSigner": calibration_signer,
                "accuracy": fold_accuracy,
                "temperature": temperature,
                "trainSampleCount": len(fit_samples),
                "validationSampleCount": len(calibration),
                "testSampleCount": len(test),
                "sampleCount": len(test),
            })
            all_true.extend(test_y.tolist())
            all_pred.extend(test_pred.tolist())
            all_prob.extend(test_prob.tolist())
            for mode in ("slot_swap", "missing_face", "missing_pose", "jitter", "scale_rotation", "static_hold", "dropped_frames"):
                stressed = transform_stress(test_x, mode, args.seed + fold_index)
                stressed_prob = softmax_temperature(model.predict(stressed, verbose=0), temperature)
                correct_or_reject = sum(
                    decode(scores, labels)[0] == labels[int(actual)]
                    or decode(scores, labels)[1]
                    for scores, actual in zip(stressed_prob, test_y)
                ) / len(test_y)
                stress_results.setdefault(mode, []).append(float(correct_or_reject))
            model.save(args.output_dir / f"loso-{fold_index}.keras") if args.test_mode else None
            model._signar_keepalive = None
            del model
            print(f"SIGNAR_PROGRESS {fold_index + 1}/{len(signers)}", flush=True)
            print(f"LOSO {held_out}: accuracy={fold_accuracy:.4f}, calibration signer={calibration_signer}, T={temperature:.2f}", flush=True)

        y_true = np.asarray(all_true, dtype=np.int32)
        y_pred = np.asarray(all_pred, dtype=np.int32)
        probabilities = np.asarray(all_prob, dtype=np.float32)
        report = classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
        recalls = {label: float(report[label]["recall"]) for label in labels}
        no_sign_indexes = y_true == label_indexes[REJECT]
        false_accepts = [decode(scores, labels)[0] != REJECT for scores in probabilities[no_sign_indexes]]
        false_accept_rate = float(np.mean(false_accepts)) if false_accepts else 1.0
        mean_accuracy = float(np.mean([fold["accuracy"] for fold in fold_reports]))
        stress_summary = {name: float(np.mean(values)) for name, values in stress_results.items()}
        calibration_error = expected_calibration_error(probabilities, y_true)
        readiness_blockers = []
        if status["distinctSigners"] < 5:
            readiness_blockers.append("At least five distinct signers are required.")
        if not status["allLabelsAtLeastFiveSigners"]:
            readiness_blockers.append("Every label must have samples from at least five signers.")
        if status["noSignSampleCount"] < 10:
            readiness_blockers.append("The dataset needs at least 10 NO SIGN background samples.")
        if not MIN_NO_SIGN_SHARE <= status["noSignShare"] <= MAX_NO_SIGN_SHARE:
            readiness_blockers.append(f"NO SIGN must be {MIN_NO_SIGN_SHARE:.0%} to {MAX_NO_SIGN_SHARE:.0%} of the dataset.")
        if mean_accuracy < MIN_LOSO_ACCURACY:
            readiness_blockers.append(f"Mean leave-one-signer-out accuracy is below {MIN_LOSO_ACCURACY:.0%}.")
        if false_accept_rate > MIN_NO_SIGN_FAR:
            readiness_blockers.append(f"NO SIGN false-accept rate is above {MIN_NO_SIGN_FAR:.0%}.")
        if any(value < MIN_STRESS_ACCEPT_OR_REJECT for value in stress_summary.values()):
            readiness_blockers.append("At least one stress test is below 90% correct-or-reject.")
        sign_recalls = {label: value for label, value in recalls.items() if label != REJECT}
        if any(value < MINIMUM_CLASS_RECALL for value in sign_recalls.values()):
            readiness_blockers.append(f"At least one sign label recall is below {MINIMUM_CLASS_RECALL:.0%}.")
        trained = not args.test_mode and not readiness_blockers
        evaluation = {
            "testMode": args.test_mode,
            "acceptanceProfile": "signar-standard-v1",
            "acceptanceThresholds": {
                "meanLeaveOneSignerOutAccuracy": MINIMUM_TEST_ACCURACY,
            "minimumSignRecall": MINIMUM_CLASS_RECALL,
                "maximumNoSignFalseAcceptRate": MINIMUM_NO_SIGN_FALSE_ACCEPT_RATE,
                "minimumNoSignSamples": 10,
                "noSignShareRange": [MINIMUM_NO_SIGN_SHARE, MAXIMUM_NO_SIGN_SHARE],
                "minimumStressCorrectOrReject": MINIMUM_STRESS_ACCEPT_OR_REJECT,
                "minimumSignersPerLabel": 5,
            },
            "evaluationScope": "signer-independent",
            "deploymentEligible": trained,
            "testAccuracy": mean_accuracy,
            "minimumAccuracy": MIN_LOSO_ACCURACY,
            "minimumClassRecall": MINIMUM_CLASS_RECALL,
            "readinessBlockers": readiness_blockers,
            "classRecall": recalls,
            "classificationReport": report,
            "confusionMatrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
            "labels": labels,
            "leaveOneSignerOut": {"meanAccuracy": mean_accuracy, "folds": fold_reports},
            "noSignFalseAcceptRate": false_accept_rate,
            "noSignShare": status["noSignShare"],
            "noSignSampleCount": status["noSignSampleCount"],
            "liveCameraSamplesPerSigner": status["liveCameraSamplesPerSigner"],
            "sampleSourceCounts": status["sampleSourceCounts"],
            "liveCameraSamplesPerLabel": status["liveCameraSamplesPerLabel"],
            "expectedCalibrationError": calibration_error,
            "temperatureScaling": {"method": "held-out signer grid search", "foldTemperatures": fold_temperatures, "finalTemperature": float(np.mean(fold_temperatures))},
            "stressTestsCorrectOrRejected": stress_summary,
            "signerCount": status["distinctSigners"],
            "sampleCounts": {
                "dataset": len(samples),
                "train": int(round(np.mean([fold["trainSampleCount"] for fold in fold_reports]))),
                "validation": int(round(np.mean([fold["validationSampleCount"] for fold in fold_reports]))),
                "test": len(y_true),
            },
        }
        label_metadata = {}
        for label in labels:
            sample = next((entry for entry in samples if entry["label"] == label), {})
            unit_type = "background" if label == REJECT else sample.get("unitType", "fingerspelling" if len(label) == 1 and label.isalpha() else "word")
            motion_type = "static" if label == REJECT else sample.get("motionType", "dynamic" if label in {"J", "Z"} else "static")
            label_metadata[label] = {
                "unitType": unit_type,
                "motionType": "static" if unit_type == "background" else motion_type,
                "displayText": sample.get("displayText", ""),
            }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
        if not args.test_mode and trained:
            print("SIGNAR_STAGE training final model", flush=True)
            final_model, history = fit_model(labels, samples, args.seed, args.epochs, args.batch_size)
            tfjs = load_tensorflowjs_converter()
            tfjs.converters.save_keras_model(final_model, str(args.output_dir))
            metadata = {"labels": labels, "sequence_length": SEQ_LEN, "feature_count": FEATURES, "feature_version": FEATURE_VERSION, "reject_label": REJECT, "label_metadata": label_metadata}
            (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
            (args.output_dir / "training-history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {
            "project": "SignAR FSL Decoder", "schemaVersion": 4, "featureVersion": FEATURE_VERSION,
            "trained": trained, "testMode": args.test_mode, "modelPath": "./model/model.json" if trained and not args.test_mode else None,
            "rejectLabel": REJECT, "sequenceLength": SEQ_LEN, "featureDimension": FEATURES,
            "samplingFps": 15, "sampleWindowSeconds": 3.0, "outputClasses": len(labels), "labels": labels,
            "labelMetadata": label_metadata,
            "minimumSignersPerLabel": 5, "distinctSignersPerLabel": status["distinctSignersPerLabel"],
            "readinessBlockers": readiness_blockers,
            "evaluation": evaluation,
            "featureSchema": {
                "hands": {"slots": ["left", "right"], "pointCount": 21, "coordinateValues": ["wristRelativePalmX", "wristRelativePalmY", "wristRelativePalmZ"], "fingerAngleCount": 5, "wristPlacementValues": 3},
                "pose": {"pointCount": 33, "valuesPerPoint": ["bodyRelativeX", "bodyRelativeY", "bodyRelativeZ", "visibility"]},
                "face": {"pointCount": 48, "valuesPerPoint": ["noseRelativeX", "noseRelativeY", "noseRelativeZ"], "regions": ["eyes", "eyebrows", "nose", "mouth"]},
                "presenceFlags": 4, "dimension": FEATURES, "missingValue": MISSING_VALUE,
            },
            "acceptanceProfile": "signar-standard-v1",
            "note": "Signer-held-out holistic-v4 evaluation covers two-hand, face, body, and temporal cues in the recorded dataset. Results can differ across signers, cameras, and environments.",
        }
        (args.output_dir / "vocabulary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"deploymentEligible": trained, "evaluation": evaluation}, indent=2, default=json_default, allow_nan=False), flush=True)
        return 0
    except Exception as error:
        import traceback

        print(f"Holistic-v4 training stopped: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
