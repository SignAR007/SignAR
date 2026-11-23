#!/usr/bin/env python3
"""Backward-compatible entry point for the SignAR holistic-v4 trainer."""
from __future__ import annotations

try:
    from .train_hand_model import main
except ImportError:
    from train_hand_model import main


if __name__ == "__main__":
    raise SystemExit(main())
