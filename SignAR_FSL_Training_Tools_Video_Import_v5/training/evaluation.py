"""Shared held-out evaluation checks for model candidates."""
import math


MINIMUM_TEST_ACCURACY = 0.90
MINIMUM_CLASS_RECALL = 0.80
MINIMUM_NO_SIGN_FALSE_ACCEPT_RATE = 0.10
MINIMUM_NO_SIGN_SHARE = 0.01
MAXIMUM_NO_SIGN_SHARE = 0.25
MINIMUM_STRESS_ACCEPT_OR_REJECT = 0.90
NO_SIGN_LABEL = "NO SIGN"


def has_no_sign_label(labels: object) -> bool:
    return isinstance(labels, list) and any(
        isinstance(label, str) and label.strip().upper() == NO_SIGN_LABEL
        for label in labels
    )


def class_recall_failures(report: object, labels: list[str]) -> list[str]:
    if not isinstance(report, dict):
        return list(labels)

    failures = []
    for label in labels:
        if isinstance(label, str) and label.strip().upper() == NO_SIGN_LABEL:
            continue
        metrics = report.get(label)
        recall = metrics.get("recall") if isinstance(metrics, dict) else None
        if (
            isinstance(recall, bool)
            or not isinstance(recall, (int, float))
            or not math.isfinite(recall)
            or not 0 <= recall <= 1
            or recall < MINIMUM_CLASS_RECALL
        ):
            failures.append(label)
    return failures
