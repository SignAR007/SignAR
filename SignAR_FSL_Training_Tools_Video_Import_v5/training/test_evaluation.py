import unittest

from evaluation import MINIMUM_CLASS_RECALL, class_recall_failures, has_no_sign_label


class ClassRecallTests(unittest.TestCase):
    def test_rejects_labels_below_the_minimum_and_missing_metrics(self):
        report = {
            "A": {"recall": 0.5},
            "B": {"recall": 0.0},
        }

        self.assertEqual(MINIMUM_CLASS_RECALL, 0.5)
        self.assertEqual(class_recall_failures(report, ["A", "B", "C"]), ["B", "C"])

    def test_accepts_all_labels_at_or_above_the_minimum(self):
        report = {
            "A": {"recall": 0.5},
            "B": {"recall": 1.0},
        }

        self.assertEqual(class_recall_failures(report, ["A", "B"]), [])

    def test_rejects_invalid_recall_values(self):
        report = {
            "A": {"recall": True},
            "B": {"recall": 1.5},
        }

        self.assertEqual(class_recall_failures(report, ["A", "B"]), ["A", "B"])


class NoSignEligibilityTests(unittest.TestCase):
    def test_recognizes_the_reserved_background_label_case_insensitively(self):
        self.assertTrue(has_no_sign_label(["A", " no sign "]))

    def test_rejects_missing_or_malformed_label_lists(self):
        self.assertFalse(has_no_sign_label(["A", "B"]))
        self.assertFalse(has_no_sign_label(None))

