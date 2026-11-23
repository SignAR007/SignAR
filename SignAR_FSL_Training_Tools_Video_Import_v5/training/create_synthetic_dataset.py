#!/usr/bin/env python3
"""Create a synthetic dataset to smoke-test the training pipeline."""
import argparse
import json
import random
from pathlib import Path

TEST_LABELS = ["WORD_A", "WORD_B", "WORD_C", "WORD_D", "WORD_E"]
FEATURE_VERSION = "signar-fsl-456-v2"
FEATURES = 456
SEQ_LEN = 30

def make_sample(signer_id, label, frame_count, rng, verified=True, consent=True):
    class_index = TEST_LABELS.index(label)
    features = []
    for frame_index in range(frame_count):
        frame = [0.0] * FEATURES
        frame[class_index] = 0.8 + rng.random() * 0.1
        frame[32] = 0.2 + 0.4 * frame_index / max(1, frame_count - 1)
        features.append(frame)
    return {
        "id": f"{signer_id}_{label}_{frame_count}",
        "signerId": signer_id,
        "label": label,
        "verificationNotes": f"verified for {label}",
        "verified": verified,
        "consentRecorded": consent,
        "sourceType": "camera",
        "frameCount": frame_count,
        "durationMs": frame_count * 67,
        "createdAt": 1000 + rng.randint(0, 100000),
        "features": features,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("test_synthetic_dataset.json"),
        help="Where to write the synthetic dataset",
    )
    args = parser.parse_args()

    rng = random.Random(42)
    samples = []
    for i in range(5):
        signer = f"signer-{i+1:02d}"
        for label in TEST_LABELS:
            samples.append(make_sample(signer, label, SEQ_LEN, rng))

    payload = {
        "schemaVersion": 1,
        "featureVersion": FEATURE_VERSION,
        "featureDimension": FEATURES,
        "sequenceLength": SEQ_LEN,
        "featuresResampled": True,
        "labels": TEST_LABELS,
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Created synthetic dataset: {args.output}")
    print(f"Signers: 5, Labels: {len(TEST_LABELS)}, Samples: {len(samples)}")
    print(f"Each sample: {SEQ_LEN} frames x {FEATURES} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())