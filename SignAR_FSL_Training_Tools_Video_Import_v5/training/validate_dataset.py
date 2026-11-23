#!/usr/bin/env python3
"""Validate a SignAR FSL dataset and report label/signer coverage before training.

Uses only Python's standard library so it can run before TensorFlow is installed.
It never prints landmark arrays or any raw video data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

FEATURE_VERSION = "signar-fsl-holistic-v4"
HAND_V3_FEATURE_VERSION = "signar-fsl-hand-v3"
LEGACY_FEATURE_VERSION = "signar-fsl-456-v2"
SEQ_LEN = 30
FEATURES = 422
HAND_V3_SEQ_LEN = 10
HAND_V3_FEATURES = 68
LEGACY_SEQ_LEN = 30
LEGACY_FEATURES = 456
MISSING_VALUE = -3.0
MISSING_THRESHOLD = -3.1
COORDINATE_LIMIT = 3.0
MIN_SIGNERS_PER_LABEL = 5
MIN_SIGNERS_PER_LABEL_FOR_INDEPENDENT_EVALUATION = 5
MIN_TOTAL_SIGNERS_FOR_INDEPENDENT_EVALUATION = 5
MIN_NO_SIGN_SAMPLES = 10
MIN_NO_SIGN_SHARE = 0.01
MAX_NO_SIGN_SHARE = 0.25
UNIT_TYPES = {"background", "fingerspelling", "word", "phrase"}
MOTION_TYPES = {"static", "dynamic"}


def dataset_fingerprint(payload: object) -> str:
    """Return a stable digest for binding a readiness report to its dataset."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_label(value: object) -> str:
    """Normalize dataset labels to a consistent uppercase, trimmed form."""
    text = value.strip() if isinstance(value, str) else ""
    return text.upper() if text else ""


def inferred_label_metadata(label: str, declared: object = None, sample: object = None) -> dict:
    normalized = label.strip().upper()
    declared_values = declared if isinstance(declared, dict) else {}
    sample_values = sample if isinstance(sample, dict) else {}
    inferred_unit = "background" if normalized == "NO SIGN" else ("fingerspelling" if len(normalized) == 1 and normalized.isalpha() else "word")
    inferred_motion = "dynamic" if normalized in {"J", "Z"} else "static"
    unit_type = str(sample_values.get("unitType") or declared_values.get("unitType") or inferred_unit).strip().lower()
    motion_type = str(sample_values.get("motionType") or declared_values.get("motionType") or inferred_motion).strip().lower()
    display_text = str(sample_values.get("displayText") or declared_values.get("displayText") or "").strip()
    valid = (
        unit_type in UNIT_TYPES
        and motion_type in MOTION_TYPES
        and ((unit_type == "background") == (normalized == "NO SIGN"))
        and (unit_type != "background" or motion_type == "static")
    )
    return {"unitType": unit_type, "motionType": motion_type, "displayText": display_text, "valid": valid}


def _truthy(value: object) -> bool:
    """Accept common JSON/CSV truthy encodings without excluding valid samples."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return value == 1
    return False


def _finite_feature(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _valid_feature_value(value: object) -> bool:
    if not _finite_feature(value):
        return False
    numeric = float(value)
    if numeric <= MISSING_THRESHOLD:
        return math.isclose(numeric, MISSING_VALUE, rel_tol=1e-5, abs_tol=1e-8)
    return -COORDINATE_LIMIT - 1e-5 <= numeric <= COORDINATE_LIMIT + 1e-5


def canonicalize_hand_v3_frame(frame: object) -> list[float]:
    """Lift an older hand-only frame into the holistic-v4 layout."""
    if not isinstance(frame, list) or len(frame) != HAND_V3_FEATURES or not all(_finite_feature(value) for value in frame):
        raise ValueError("Hand-v3 frames must contain 68 finite values.")
    result = [MISSING_VALUE] * FEATURES
    result[418:422] = [0.0, 0.0, 0.0, 0.0]
    if all(value == 0 for value in frame):
        return result
    result[71:134] = [max(-COORDINATE_LIMIT, min(COORDINATE_LIMIT, float(value))) for value in frame[:63]]
    result[134:139] = [max(-COORDINATE_LIMIT, min(COORDINATE_LIMIT, float(value))) for value in frame[63:68]]
    result[419] = 1.0
    return result


def canonicalize_legacy_frame(frame: object) -> list[float]:
    """Convert one legacy 456-value frame into holistic-v4 features."""
    if not isinstance(frame, list) or len(frame) != LEGACY_FEATURES or not all(_finite_feature(value) for value in frame):
        raise ValueError("Legacy frames must contain 456 finite values.")
    result = [MISSING_VALUE] * FEATURES
    result[418:422] = [0.0, 0.0, 0.0, 0.0]
    for point_index in range(33):
        values = frame[point_index * 4:point_index * 4 + 4]
        if any(value > -2.75 for value in values[:3]):
            target = 142 + point_index * 4
            result[target:target + 4] = [max(-3.0, min(3.0, float(value))) for value in values[:3]] + [max(0.0, min(1.0, float(values[3])))]
            result[420] = 1.0
    chains = ((1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
    for slot in (0, 1):
        source_base = 132 + slot * 126
        target_base = slot * 71
        points = [frame[source_base + point * 6 + 3:source_base + point * 6 + 6] for point in range(21)]
        valid = [len(point) == 3 and point[0] > -2.75 for point in points]
        if not any(valid):
            continue
        for index, point in enumerate(points):
            if not valid[index]:
                continue
            result[target_base + index * 3:target_base + index * 3 + 3] = [
                max(-3.0, min(3.0, float(point[0]) * (-1 if slot == 0 else 1))),
                max(-3.0, min(3.0, float(point[1]))),
                max(-3.0, min(3.0, float(point[2]))),
            ]
        for angle_index, (first, joint, last) in enumerate(chains):
            if not (valid[first] and valid[joint] and valid[last]):
                continue
            a = [result[target_base + first * 3 + axis] - result[target_base + joint * 3 + axis] for axis in range(3)]
            b = [result[target_base + last * 3 + axis] - result[target_base + joint * 3 + axis] for axis in range(3)]
            a_length = math.sqrt(sum(value * value for value in a))
            b_length = math.sqrt(sum(value * value for value in b))
            if a_length and b_length:
                cosine = sum(a[axis] * b[axis] for axis in range(3)) / (a_length * b_length)
                result[target_base + 63 + angle_index] = math.acos(max(-1.0, min(1.0, cosine))) / math.pi
        result[target_base + 68:target_base + 71] = [0.0, 0.0, 0.0]
        result[418 + slot] = 1.0
    for index in range(24):
        point = frame[384 + index * 3:384 + index * 3 + 3]
        if all(value > -2.75 for value in point):
            result[274 + index * 3:274 + index * 3 + 3] = [max(-3.0, min(3.0, float(value))) for value in point]
            result[421] = 1.0
    return result


def find_signer_split(signer_labels: dict[str, set[str]], labels: list[str], seed: int = 42, tries: int = 2000) -> dict | None:
    """Try bounded deterministic signer splits; every label must occur in every partition."""
    signers = sorted(signer_labels)
    if len(signers) < MIN_TOTAL_SIGNERS_FOR_INDEPENDENT_EVALUATION or not labels or tries < 1:
        return None
    if any(
        sum(label in signer_labels[signer] for signer in signers)
        < MIN_SIGNERS_PER_LABEL_FOR_INDEPENDENT_EVALUATION
        for label in labels
    ):
        return None
    n_train = max(3, int(len(signers) * .60))
    n_val = max(1, int(len(signers) * .20))
    if n_train + n_val >= len(signers):
        n_train, n_val = len(signers) - 2, 1

    rng = random.Random(seed)
    for _ in range(tries):
        order = signers[:]
        rng.shuffle(order)
        groups = {
            "train": order[:n_train],
            "validation": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:],
        }
        if all(all(any(label in signer_labels[signer] for signer in group) for label in labels)
               for group in groups.values()):
            return {name: sorted(group) for name, group in groups.items()}
    return None


def signer_split_possible(samples: list[dict], labels: list[str], seed: int = 42, tries: int = 2000) -> bool:
    signer_labels: dict[str, set[str]] = {}
    for sample in samples:
        signer_labels.setdefault(sample["signerId"], set()).add(sample["label"])
    return find_signer_split(signer_labels, labels, seed, tries) is not None


def inspect_dataset(payload: dict) -> dict:
    errors = []
    warnings = []
    if not isinstance(payload, dict):
        payload = {}
        errors.append("dataset root must be a JSON object")
    source_version = payload.get("featureVersion")
    legacy = source_version == LEGACY_FEATURE_VERSION and payload.get("featureDimension") == LEGACY_FEATURES
    hand_v3 = source_version == HAND_V3_FEATURE_VERSION and payload.get("featureDimension") == HAND_V3_FEATURES
    holistic = source_version == FEATURE_VERSION and payload.get("featureDimension") == FEATURES
    if not (legacy or hand_v3 or holistic):
        errors.append(f"featureVersion must be {FEATURE_VERSION!r}")
    if not (legacy or hand_v3 or holistic):
        errors.append(f"featureDimension must be {FEATURES}")
    features_resampled = payload.get("featuresResampled")
    if "featuresResampled" in payload and not isinstance(features_resampled, bool):
        errors.append("featuresResampled must be a boolean")
    elif features_resampled is True:
        expected_length = LEGACY_SEQ_LEN if legacy else HAND_V3_SEQ_LEN if hand_v3 else SEQ_LEN
        if payload.get("sequenceLength") != expected_length:
            errors.append(f"Pre-resampled datasets must declare sequenceLength {expected_length}")
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(l, str) for l in labels):
        errors.append("dataset must declare a non-empty labels array of strings")
        labels = []
    else:
        normalized_labels = []
        seen = set()
        for label in labels:
            clean = normalize_label(label)
            if not clean:
                errors.append("labels must be non-empty strings without surrounding whitespace")
                continue
            if clean in seen:
                errors.append("labels must not contain duplicates after normalization")
                continue
            seen.add(clean)
            normalized_labels.append(clean)
        labels = normalized_labels
        if labels and any(normalize_label(label) != label for label in payload.get("labels", [])):
            warnings.append("Labels were normalized to uppercase without surrounding whitespace.")
        if len(labels) != len(set(labels)):
            errors.append("labels must not contain duplicates")
    label_set = set(labels)
    declared_label_metadata = payload.get("labelMetadata", {})
    if not isinstance(declared_label_metadata, dict):
        errors.append("labelMetadata must be an object keyed by sign label")
        declared_label_metadata = {}
    metadata_signatures: dict[str, set[tuple[str, str, str]]] = {}
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raw_samples = []
        errors.append("samples must be a JSON array")

    ids = set()
    valid = []
    invalid_counts = Counter()
    excluded_sample_count = 0
    for index, sample in enumerate(raw_samples):
        reasons = []
        if not isinstance(sample, dict):
            invalid_counts["sample_not_object"] += 1
            excluded_sample_count += 1
            continue
        sample_id = str(sample.get("id", ""))
        if not sample_id:
            reasons.append("missing_id")
        elif sample_id in ids:
            reasons.append("duplicate_id")
        ids.add(sample_id)
        label = normalize_label(sample.get("label", ""))
        sign_metadata = inferred_label_metadata(label, declared_label_metadata.get(label), sample)
        if not sign_metadata["valid"]:
            reasons.append("invalid_sign_metadata")
        elif label in label_set:
            metadata_signatures.setdefault(label, set()).add((sign_metadata["unitType"], sign_metadata["motionType"], sign_metadata["displayText"]))
        signer_value = sample.get("signerId", "")
        signer = signer_value.strip() if isinstance(signer_value, str) else ""
        if label not in label_set:
            reasons.append("unknown_label")
        if not signer:
            reasons.append("missing_signer")
        if not _truthy(sample.get("verified")):
            reasons.append("not_verified")
        if not _truthy(sample.get("consentRecorded")):
            reasons.append("consent_not_confirmed")
        if len(str(sample.get("verificationNotes", "")).strip()) < 5:
            reasons.append("missing_verification_notes")
        features = sample.get("features")
        if not isinstance(features, list) or len(features) < 1:
            reasons.append("too_few_frames")
        else:
            expected_length = LEGACY_SEQ_LEN if legacy else HAND_V3_SEQ_LEN if hand_v3 else SEQ_LEN
            if features_resampled is True and len(features) != expected_length:
                reasons.append("invalid_resampled_sequence_length")
            bad_frame = False
            zero_frame_count = 0
            for frame in features:
                try:
                    normalized = canonicalize_legacy_frame(frame) if legacy else canonicalize_hand_v3_frame(frame) if hand_v3 else frame
                except (TypeError, ValueError):
                    bad_frame = True
                    break
                if not isinstance(normalized, list) or len(normalized) != FEATURES:
                    bad_frame = True
                    break
                if not all(_valid_feature_value(value) for value in normalized):
                    bad_frame = True
                    break
                if not any(value >= 0.5 for value in normalized[418:422]):
                    zero_frame_count += 1
            if not bad_frame and zero_frame_count == len(features) and label != "NO SIGN":
                reasons.append("no_tracked_cues")
            if bad_frame:
                reasons.append("invalid_feature_values")
        if reasons:
            excluded_sample_count += 1
            for reason in reasons:
                invalid_counts[reason] += 1
            continue
        valid.append({"id":sample_id,"label":label,"signerId":signer,"frameCount":len(features),"sourceType":str(sample.get("sourceType", "")),"unitType":sign_metadata["unitType"],"motionType":sign_metadata["motionType"]})

    inconsistent_metadata = [label for label, signatures in metadata_signatures.items() if len(signatures) > 1]
    if inconsistent_metadata:
        errors.append("Samples for each label must agree on sign unit, motion type, and display text: " + ", ".join(inconsistent_metadata))

    label_counts = Counter(s["label"] for s in valid)
    signer_counts = Counter(s["signerId"] for s in valid)
    live_camera_counts = Counter(s["signerId"] for s in valid if s["sourceType"] == "camera")
    no_sign_share = label_counts["NO SIGN"] / max(1, len(valid))
    label_signers = {label: sorted({s["signerId"] for s in valid if s["label"] == label}) for label in labels}
    missing_labels = [label for label in labels if label_counts[label] == 0]
    missing_signer_coverage = [
        label for label,signers in label_signers.items()
        if len(signers) < MIN_SIGNERS_PER_LABEL
    ]
    limited_signer_coverage = {
        label:len(signers) for label,signers in label_signers.items()
        if len(signers) < MIN_SIGNERS_PER_LABEL_FOR_INDEPENDENT_EVALUATION
    }
    single_class = len(labels) == 1
    sample_level_ready = (
        not single_class
        and not missing_labels
        and all(label_counts[label] >= 3 for label in labels)
    )
    no_sign_sample_count = label_counts["NO SIGN"]
    no_sign_share_ready = no_sign_sample_count >= MIN_NO_SIGN_SAMPLES and MIN_NO_SIGN_SHARE <= no_sign_share <= MAX_NO_SIGN_SHARE
    candidate_ready = (
        sample_level_ready and not missing_signer_coverage and not errors and not invalid_counts
        and len(signer_counts) >= MIN_TOTAL_SIGNERS_FOR_INDEPENDENT_EVALUATION
        and not limited_signer_coverage and no_sign_share_ready
    )
    split_ready = not single_class and not missing_labels and not limited_signer_coverage and signer_split_possible(valid, labels)
    if single_class:
        warnings.append("Single-label data can be trained, but cannot teach the model to distinguish signs.")
    if missing_labels:
        warnings.append("Collect samples for missing labels: " + ", ".join(missing_labels))
    if missing_signer_coverage:
        warnings.append("Every label needs at least 1 valid signer alias: " + ", ".join(missing_signer_coverage))
    if len(signer_counts) < 5:
        warnings.append(f"Only {len(signer_counts)} signer aliases are represented; the trainer will use sample-level evaluation and will not measure performance on unseen signers.")
    if no_sign_sample_count < MIN_NO_SIGN_SAMPLES:
        warnings.append(f"Collect at least {MIN_NO_SIGN_SAMPLES} NO SIGN background samples; current count: {no_sign_sample_count}.")
    if not no_sign_share_ready:
        warnings.append(f"NO SIGN should make up {MIN_NO_SIGN_SHARE:.0%} to {MAX_NO_SIGN_SHARE:.0%} of valid samples; current share is {no_sign_share:.1%}.")
    low_sample_labels = [label for label in labels if 0 < label_counts[label] < 3]
    if low_sample_labels:
        warnings.append("Sample-level train/validation/test evaluation needs at least 3 valid samples per label: " + ", ".join(low_sample_labels))
    if limited_signer_coverage and not missing_signer_coverage:
        warnings.append(
            "Signer-independent evaluation needs at least 3 signers per label; current coverage: "
            + ", ".join(f"{k} ({v})" for k,v in limited_signer_coverage.items())
        )
    if not split_ready and not single_class and not missing_labels and not limited_signer_coverage and len(signer_counts) >= MIN_TOTAL_SIGNERS_FOR_INDEPENDENT_EVALUATION:
        warnings.append("Could not find a signer-independent split with all labels in every partition. Collect more words from more signers.")
    if invalid_counts:
        warnings.append("Some samples were not counted as evaluation-ready because of metadata or feature checks; the trainer can use structurally valid feature sequences.")

    next_step = (
        f"SignAR validation profile: five signers cover every label, and the dataset has at least {MIN_NO_SIGN_SAMPLES} NO SIGN samples ({MIN_NO_SIGN_SHARE:.0%}–{MAX_NO_SIGN_SHARE:.0%} of the dataset). Live-camera samples are optional and reported as coverage context. "
        "Evaluation metrics describe this dataset and may differ across signers, cameras, and environments."
    )

    return {
        "schemaVersion":1,
        "datasetFingerprint":dataset_fingerprint(payload),
        "featureVersion":payload.get("featureVersion"),
        "featureDimension":payload.get("featureDimension"),
        "samplesInExport":len(raw_samples),
        "validSamples":len(valid),
        "excludedSamples":excluded_sample_count,
        "distinctSigners":len(signer_counts),
        "liveCameraSamplesPerSigner":dict(sorted(live_camera_counts.items())),
        "noSignShare":no_sign_share,
        "minimumSignersPerLabel":MIN_SIGNERS_PER_LABEL,
        "readyForCandidateTraining":bool(candidate_ready),
        "readyForSignerIndependentTraining":bool(split_ready and not errors and not invalid_counts),
        "labelCounts":{label:label_counts[label] for label in labels},
        "distinctSignersPerLabel":{label:len(label_signers[label]) for label in labels},
        "labelsMissingSignerCoverage":missing_signer_coverage,
        "samplesPerSigner":dict(sorted(signer_counts.items())),
        "excludedReasons":dict(invalid_counts),
        "errors":errors,
        "warnings":warnings,
        "nextStep":next_step
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("fsl-dataset.json"))
    parser.add_argument("--report", type=Path, default=Path("dataset-report.json"), help="Where to save a metadata-only report")
    parser.add_argument("--strict", action="store_true", help="Return nonzero unless the dataset supports signer-independent evaluation")
    args = parser.parse_args()
    try:
        payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read dataset: {exc}", file=sys.stderr)
        return 2
    report = inspect_dataset(payload)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict:
        if report["errors"]:
            return 2
        if not report["readyForSignerIndependentTraining"]:
            return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
