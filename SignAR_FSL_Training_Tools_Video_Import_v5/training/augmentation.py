"""Holistic-v4 training augmentation, retained for toolkit imports."""
from __future__ import annotations

try:
    from .train_hand_model import augment_training_data
except ImportError:
    from train_hand_model import augment_training_data

SEQ_LEN = 30
FEATURES = 422

__all__ = ["FEATURES", "SEQ_LEN", "augment_training_data"]
