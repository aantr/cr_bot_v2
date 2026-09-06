from __future__ import annotations

import argparse
import random
from pathlib import Path


def split_annotations(
    annotations_file: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    train_name: str = "train.txt",
    val_name: str = "val.txt",
) -> tuple[Path, Path, int, int]:
    """Split an image manifest into reproducible train and validation manifests."""
    annotations_file = annotations_file.expanduser().resolve()
    if not annotations_file.is_file():
        raise FileNotFoundError(f"Annotation manifest not found: {annotations_file}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be greater than 0 and less than 1")

    entries = [
        line.strip()
        for line in annotations_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(entries) < 2:
        raise ValueError("At least two image paths are required for train/val split")
    if len(entries) != len(set(entries)):
        raise ValueError("The annotation manifest contains duplicate image paths")

    random.Random(seed).shuffle(entries)
    val_count = max(1, min(len(entries) - 1, round(len(entries) * val_ratio)))
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]

    output_dir = annotations_file.parent
    train_path = output_dir / train_name
    val_path = output_dir / val_name
    train_path.write_text("\n".join(train_entries) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_entries) + "\n", encoding="utf-8")

    return train_path, val_path, len(train_entries), len(val_entries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split yolo_annotations.txt into train.txt and val.txt"
    )
    parser.add_argument(
        "annotations_file",
        type=Path,
        help="Path to yolo_annotations.txt",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction, from 0 to 1 (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible shuffling (default: 42)",
    )
    parser.add_argument("--train-name", default="train.txt")
    parser.add_argument("--val-name", default="val.txt")
    args = parser.parse_args()

    try:
        train_path, val_path, train_count, val_count = split_annotations(
            annotations_file=args.annotations_file,
            val_ratio=args.val_ratio,
            seed=args.seed,
            train_name=args.train_name,
            val_name=args.val_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Train: {train_count} images -> {train_path}")
    print(f"Val:   {val_count} images -> {val_path}")


if __name__ == "__main__":
    main()
