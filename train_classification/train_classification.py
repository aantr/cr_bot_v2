from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


V2_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = V2_DIR / "dataset_centered"
DEFAULT_SPLIT_DIR = V2_DIR / "dataset_centered_yolo"
DEFAULT_PROJECT = V2_DIR / "runs_game"
DEFAULT_MODEL = V2_DIR / "yolo26s-cls.pt"
DEFAULT_RUN_NAME = "yolo26s_cls_224"

IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
TRACK_RE = re.compile(
    r"^(?P<video>.+)_track_(?P<track>\d+)_frame_(?P<frame>\d+)$",
    re.IGNORECASE,
)
SPLIT_FORMAT_VERSION = 2


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_classes(dataset_dir: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        images = image_files(class_dir)
        if images:
            classes[class_dir.name] = images
    return classes


def dataset_fingerprint(dataset_dir: Path, classes: dict[str, list[Path]]) -> str:
    digest = hashlib.sha256()
    for class_name, paths in sorted(classes.items()):
        digest.update(class_name.encode("utf-8"))
        for path in paths:
            stat = path.stat()
            digest.update(path.relative_to(dataset_dir).as_posix().encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def track_group(path: Path) -> tuple[str, str]:
    """Return a stable group key so frames from one track stay in one split."""
    match = TRACK_RE.match(path.stem)
    if match:
        return match.group("video"), match.group("track")
    # For arbitrary classification images, each image is an independent group.
    return "image", path.as_posix()


def choose_validation_groups(
    groups: list[list[Path]],
    val_ratio: float,
) -> set[int]:
    """Choose whole tracks whose image count is closest to the requested ratio."""
    target_images = sum(len(group) for group in groups) * val_ratio
    target_groups = len(groups) * val_ratio
    # total image count -> one deterministic subset of group indices
    subsets: dict[int, tuple[int, ...]] = {0: ()}
    for index, group in enumerate(groups):
        additions: dict[int, tuple[int, ...]] = {}
        for image_count, subset in subsets.items():
            new_count = image_count + len(group)
            additions.setdefault(new_count, (*subset, index))
        for image_count, subset in additions.items():
            subsets.setdefault(image_count, subset)

    candidates = [
        (image_count, subset)
        for image_count, subset in subsets.items()
        if subset and len(subset) < len(groups)
    ]
    _, best_subset = min(
        candidates,
        key=lambda item: (
            abs(item[0] - target_images),
            abs(len(item[1]) - target_groups),
            item[1],
        ),
    )
    return set(best_subset)


def split_class_images(
    paths: list[Path],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], bool]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        groups[track_group(path)].append(path)

    grouped_paths = [sorted(group) for _, group in sorted(groups.items())]
    rng.shuffle(grouped_paths)

    if len(grouped_paths) >= 2:
        val_indices = choose_validation_groups(grouped_paths, val_ratio)
        val_groups = [
            group for index, group in enumerate(grouped_paths) if index in val_indices
        ]
        train_groups = [
            group for index, group in enumerate(grouped_paths) if index not in val_indices
        ]
        train = sorted(path for group in train_groups for path in group)
        val = sorted(path for group in val_groups for path in group)
        return train, val, False

    # A class represented by one track cannot be split without leakage. Keep both
    # splits usable, but report the fallback prominently to the user.
    shuffled = paths.copy()
    rng.shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    val_count = max(1, min(len(shuffled) - 1, val_count))
    return sorted(shuffled[val_count:]), sorted(shuffled[:val_count]), True


def copy_split_image(
    source: Path,
    source_class_dir: Path,
    destination_class_dir: Path,
) -> None:
    relative = source.relative_to(source_class_dir)
    destination = destination_class_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Duplicate destination image: {destination}")
    shutil.copy2(source, destination)


def validate_dataset(dataset_dir: Path) -> dict[str, dict[str, int]]:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Classification dataset must contain train/ and val/: {dataset_dir}"
        )

    train = discover_classes(train_dir)
    val = discover_classes(val_dir)
    if len(train) < 2:
        raise ValueError(f"At least two non-empty classes are required in {train_dir}")
    if set(train) != set(val):
        only_train = sorted(set(train) - set(val))
        only_val = sorted(set(val) - set(train))
        raise ValueError(
            "Classes in train and val differ: "
            f"only train={only_train}, only val={only_val}"
        )

    return {
        name: {"train": len(train[name]), "val": len(val[name])}
        for name in sorted(train)
    }


def prepare_dataset(
    source_dir: Path,
    split_dir: Path,
    val_ratio: float,
    seed: int,
    rebuild: bool,
) -> tuple[Path, dict[str, dict[str, int]]]:
    source_dir = source_dir.expanduser().resolve()
    split_dir = split_dir.expanduser().resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source_dir}")

    # A dataset already laid out for Ultralytics needs no generated copy.
    if (source_dir / "train").is_dir() and (source_dir / "val").is_dir():
        return source_dir, validate_dataset(source_dir)

    if split_dir == source_dir or source_dir in split_dir.parents:
        raise ValueError("--split-dir must be outside the source dataset directory")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be greater than 0 and less than 1")

    classes = discover_classes(source_dir)
    if len(classes) < 2:
        raise ValueError(
            f"At least two class directories with images are required: {source_dir}"
        )
    too_small = {name: len(paths) for name, paths in classes.items() if len(paths) < 2}
    if too_small:
        raise ValueError(f"Each class needs at least two images: {too_small}")

    fingerprint = dataset_fingerprint(source_dir, classes)
    manifest_path = split_dir / ".split.json"
    requested_manifest = {
        "format_version": SPLIT_FORMAT_VERSION,
        "source": str(source_dir),
        "source_fingerprint": fingerprint,
        "seed": seed,
        "val_ratio": val_ratio,
    }

    if not rebuild and manifest_path.is_file():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}
        if all(existing_manifest.get(key) == value for key, value in requested_manifest.items()):
            try:
                return split_dir, validate_dataset(split_dir)
            except (FileNotFoundError, ValueError):
                pass

    temporary_dir = split_dir.with_name(f".{split_dir.name}.tmp")
    if temporary_dir.exists():
        if temporary_dir.is_dir():
            shutil.rmtree(temporary_dir)
        else:
            temporary_dir.unlink()
    temporary_dir.mkdir(parents=True)

    counts: dict[str, dict[str, int]] = {}
    fallback_classes: list[str] = []
    rng = random.Random(seed)
    try:
        for class_name, paths in sorted(classes.items()):
            train_paths, val_paths, used_fallback = split_class_images(
                paths, val_ratio, rng
            )
            if used_fallback:
                fallback_classes.append(class_name)

            source_class_dir = source_dir / class_name
            for split_name, split_paths in (
                ("train", train_paths),
                ("val", val_paths),
            ):
                destination_class_dir = temporary_dir / split_name / class_name
                destination_class_dir.mkdir(parents=True, exist_ok=True)
                for path in split_paths:
                    copy_split_image(path, source_class_dir, destination_class_dir)

            counts[class_name] = {
                "train": len(train_paths),
                "val": len(val_paths),
            }

        manifest = {
            **requested_manifest,
            "classes": counts,
            "image_level_fallback_classes": fallback_classes,
        }
        (temporary_dir / ".split.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if split_dir.exists():
            if split_dir.is_dir():
                shutil.rmtree(split_dir)
            else:
                split_dir.unlink()
        temporary_dir.replace(split_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    if fallback_classes:
        print(
            "WARNING: these classes contain only one track, so their images were "
            "split at image level: " + ", ".join(fallback_classes),
            file=sys.stderr,
        )
    return split_dir, validate_dataset(split_dir)


def print_dataset_summary(
    dataset_dir: Path,
    counts: dict[str, dict[str, int]],
) -> None:
    total_train = sum(item["train"] for item in counts.values())
    total_val = sum(item["val"] for item in counts.values())
    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {len(counts)}; train: {total_train}; val: {total_val}")
    for class_name, class_counts in counts.items():
        print(
            f"  {class_name:<24} "
            f"train={class_counts['train']:>4}  val={class_counts['val']:>4}"
        )


def cache_value(value: str) -> str | bool:
    return False if value == "none" else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a YOLO classification model on class directories produced by "
            "build_tracking_dataset.py."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help="Source class directory or an existing train/val dataset.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Generated YOLO train/val dataset directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Recreate train/val even when the cached split is current.",
    )

    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0", help="For example: 0, 0,1, cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument(
        "--cache",
        choices=("disk", "ram", "none"),
        default="disk",
    )
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--hsv-h", type=float, default=0.010)
    parser.add_argument("--hsv-s", type=float, default=0.30)
    parser.add_argument("--hsv-v", type=float, default=0.20)
    parser.add_argument("--erasing", type=float, default=0.10)
    parser.add_argument(
        "--auto-augment",
        choices=("none", "randaugment", "autoaugment", "augmix"),
        default="none",
    )
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate the dataset without starting training.",
    )
    return parser.parse_args()


def validate_train_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32")
    if args.batch == 0 or args.batch < -1:
        raise ValueError("--batch must be -1 or a positive integer")
    for option in ("dropout", "erasing", "fliplr", "flipud"):
        value = getattr(args, option.replace("-", "_"))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{option.replace('_', '-')} must be between 0 and 1")
    if not 0.0 <= args.scale < 1.0:
        raise ValueError("--scale must be between 0 (inclusive) and 1 (exclusive)")


def main() -> None:
    args = parse_args()
    validate_train_args(args)

    dataset_dir, counts = prepare_dataset(
        source_dir=args.data,
        split_dir=args.split_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        rebuild=args.rebuild_split,
    )
    print_dataset_summary(dataset_dir, counts)

    train_args = {
        "data": str(dataset_dir),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "cache": cache_value(args.cache),
        "amp": args.amp,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "optimizer": args.optimizer,
        "dropout": args.dropout,
        "scale": args.scale,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "erasing": args.erasing,
        "auto_augment": None if args.auto_augment == "none" else args.auto_augment,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "project": str(args.project.expanduser().resolve()),
        "name": args.name,
        "save": True,
        "save_period": args.save_period,
        "plots": True,
        "exist_ok": args.exist_ok,
    }

    if args.dry_run:
        print("Dry run: training was not started.")
        print(json.dumps(train_args, ensure_ascii=False, indent=2))
        return

    print(f"Model: {args.model}")
    model = YOLO(args.model)
    model.train(**train_args)


if __name__ == "__main__":
    main()
