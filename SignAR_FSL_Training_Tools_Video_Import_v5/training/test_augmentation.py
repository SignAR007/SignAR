"""Unit tests for training-only temporal augmentation (NumPy only)."""
import unittest
import numpy as np
from augmentation import augment_training_data


class AugmentationTests(unittest.TestCase):
    def test_augmented_dataset_doubles_and_preserves_shape(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(4, 30, 456)).astype(np.float32)
        y = np.array([0, 1, 1, 2], dtype=np.int32)
        original_x = x.copy()
        original_y = y.copy()
        result_x, result_y = augment_training_data(x, y, seed=42)
        self.assertEqual(result_x.shape, (8, 30, 456))
        self.assertEqual(result_y.shape, (8,))
        self.assertTrue(np.isfinite(result_x).all())
        self.assertTrue(np.array_equal(x, original_x))
        self.assertTrue(np.array_equal(y, original_y))
        self.assertTrue(np.array_equal(result_x[:4], original_x))
        self.assertTrue(np.array_equal(result_y, np.concatenate([original_y, original_y])))
        self.assertEqual(sorted(result_y.tolist()), sorted([0, 1, 1, 2, 0, 1, 1, 2]))

    def test_large_batches_keep_rows_and_labels_aligned_across_chunks(self):
        x = np.zeros((65, 30, 456), dtype=np.float32)
        y = np.arange(65, dtype=np.int32) % 4
        x[:, :, 132:258] = -3.0
        x[:, :, 384:456] = -3.0

        result_x, result_y = augment_training_data(x, y, seed=10)

        self.assertEqual(result_x.shape, (130, 30, 456))
        self.assertTrue(np.array_equal(result_x[:65], x))
        self.assertTrue(np.array_equal(result_y, np.concatenate([y, y])))
        self.assertTrue(np.isin(result_x[65:, :, 132:258], [-3.0]).all())
        self.assertTrue(np.isin(result_x[65:, :, 384:456], [-3.0]).all())

    def test_same_seed_is_reproducible(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(2, 30, 456)).astype(np.float32)
        y = np.array([2, 7], dtype=np.int32)
        a_x, a_y = augment_training_data(x, y, seed=77)
        b_x, b_y = augment_training_data(x, y, seed=77)
        self.assertTrue(np.array_equal(a_x, b_x))
        self.assertTrue(np.array_equal(a_y, b_y))

    def test_missing_landmark_sentinel_is_preserved(self):
        x = np.zeros((2, 30, 456), dtype=np.float32)
        y = np.array([0, 1], dtype=np.int32)
        x[:, :, 132:258] = -3.0  # Left hand absent.
        x[:, :, 384:456] = -3.0  # Selected face points absent.
        result_x, _ = augment_training_data(x, y, seed=9)
        self.assertTrue(np.isin(result_x[:, :, 132:258], [-3.0]).all())
        self.assertTrue(np.isin(result_x[:, :, 384:456], [-3.0]).all())

    def test_timing_warp_never_blends_missing_sentinels_into_coordinates(self):
        x = np.zeros((1, 30, 456), dtype=np.float32)
        y = np.array([0], dtype=np.int32)
        x[0, 12, 132] = -3.0

        result_x, _ = augment_training_data(x, y, seed=13)

        self.assertFalse(np.any((result_x[:, :, 132] > -3.0) & (result_x[:, :, 132] <= -2.75)))

    def test_augmented_coordinates_stay_in_collector_range_and_keep_sentinels(self):
        x = np.zeros((1, 30, 456), dtype=np.float32)
        y = np.array([0], dtype=np.int32)
        x[:, :, 0:2] = 2.5
        x[:, :, 4:6] = -2.5
        x[:, :, 132:258] = -3.0
        x[:, :, 384:456] = -3.0

        augmented, _ = augment_training_data(x, y, seed=17)
        generated = augmented[1:]
        valid = generated > -2.75

        self.assertLessEqual(float(generated[valid].max()), 2.5)
        self.assertGreaterEqual(float(generated[valid].min()), -2.5)
        self.assertTrue(np.isin(generated[:, :, 132:258], [-3.0]).all())
        self.assertTrue(np.isin(generated[:, :, 384:456], [-3.0]).all())


if __name__ == "__main__":
    unittest.main()
