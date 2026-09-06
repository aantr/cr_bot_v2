from __future__ import annotations

import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from train_classification import train_classification as classification_training


DEFAULT_DATASET = V2_DIR / "dataset_cards"
DEFAULT_SPLIT_DIR = V2_DIR / "dataset_cards_yolo"
DEFAULT_PROJECT = V2_DIR / "runs_cards"
DEFAULT_RUN_NAME = "yolo26s_cls_cards_224"


def main() -> None:
    """Run the standard classification pipeline with card-specific paths."""
    classification_training.DEFAULT_DATASET = DEFAULT_DATASET
    classification_training.DEFAULT_SPLIT_DIR = DEFAULT_SPLIT_DIR
    classification_training.DEFAULT_PROJECT = DEFAULT_PROJECT
    classification_training.DEFAULT_RUN_NAME = DEFAULT_RUN_NAME
    classification_training.main()


if __name__ == "__main__":
    main()
