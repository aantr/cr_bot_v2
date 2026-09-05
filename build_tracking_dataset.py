from __future__ import annotations

import argparse
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from model_paths import (
    BATTLEFIELDS,
    CLASSIFICATION_MODEL_PATH,
    DETECTION_ENGINE_PATH,
)

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = DETECTION_ENGINE_PATH
DEFAULT_VIDEO = SCRIPT_DIR / "screenshots/input_omydays_cutted.mp4"
DEFAULT_OUTPUT = SCRIPT_DIR / "tracking_dataset"
DEFAULT_CLASSES_DIR = SCRIPT_DIR / "dataset_centered"
DEFAULT_CLASSIFIER = CLASSIFICATION_MODEL_PATH

IMGSZ = 1280
CONF = 0.20
IOU = 0.50
MAX_DET = 500

QUANTIZE = 16
TRACKER = "bytetrack.yaml"

LEVEL_CLASS_ID = 1
BAR_CLASS_ID = 0
LEN_POSES = 5
BLUE_RECT_SIZE = 128
SIZE_RESTRICTIONS = ((10, 10), (50, 60))
WINDOW_NAME = "Tracking object samples"

CLASS_NAME_ALIASES = {
    "battle-ram-ev1": "battle-ram-evolution",
    "goblins-hero": "goblin-hero",
    "minions": "minion",
}


@dataclass(frozen=True)
class CropSample:
    frame_index: int
    jpeg: bytes

    def decode(self) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(self.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode crop from frame {self.frame_index}")
        return image


@dataclass(frozen=True)
class ClassPrediction:
    class_name: str
    confidence: float
    model_class_name: str


def validate_crop(
    width: int,
    height: int,
    crop: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = crop
    if not (0 <= x1 < x2 <= width):
        raise ValueError(f"Invalid horizontal crop {(x1, x2)} for width {width}")
    if not (0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid vertical crop {(y1, y2)} for height {height}")
    return crop


def select_battlefield(
    width: int,
    height: int,
    crop_override: list[int] | None,
) -> tuple[int, int, int, int]:
    if crop_override is not None:
        return validate_crop(width, height, tuple(crop_override))
    video_size = (width, height)
    if video_size not in BATTLEFIELDS:
        raise ValueError(
            f"Unsupported video resolution: {width}x{height}. "
            "Pass --crop X1 Y1 X2 Y2 or add this resolution to "
            "BATTLEFIELDS in model_paths.py."
        )
    return validate_crop(width, height, BATTLEFIELDS[video_size])


def extract_blue_rect(
    battlefield: np.ndarray,
    level_box: tuple[float, float, float, float],
    bar_width: float,
) -> np.ndarray | None:
    x1, _, x2, y2 = level_box
    center_x = (x1 + x2 + bar_width) / 2
    left = round(center_x - BLUE_RECT_SIZE / 2)
    top = round(y2)
    right = left + BLUE_RECT_SIZE
    bottom = top + BLUE_RECT_SIZE

    frame_height, frame_width = battlefield.shape[:2]
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(frame_width, right)
    source_bottom = min(frame_height, bottom)
    if source_right <= source_left or source_bottom <= source_top:
        return None

    # Fixed 128x128 output. Areas outside the battlefield use YOLO's padding color.
    crop = np.full((BLUE_RECT_SIZE, BLUE_RECT_SIZE, 3), 114, dtype=np.uint8)
    target_left = source_left - left
    target_top = source_top - top
    crop[
        target_top : target_top + (source_bottom - source_top),
        target_left : target_left + (source_right - source_left),
    ] = battlefield[source_top:source_bottom, source_left:source_right]
    return crop


def is_valid_level_detection(
    box: tuple[float, float, float, float],
    class_id: int,
) -> bool:
    x1, y1, x2, y2 = box
    return (
        class_id == LEVEL_CLASS_ID
        and SIZE_RESTRICTIONS[0][0] <= x2 - x1 <= SIZE_RESTRICTIONS[1][0]
        and SIZE_RESTRICTIONS[0][1] <= y2 - y1 <= SIZE_RESTRICTIONS[1][1]
    )


def add_reservoir_sample(
    samples: dict[int, list[CropSample]],
    sampled_counts: dict[int, int],
    track_id: int,
    sample: CropSample,
    max_samples: int,
    rng: random.Random,
) -> None:
    """Keep a uniform sample across the full lifetime of a track."""
    sampled_counts[track_id] += 1
    seen = sampled_counts[track_id]
    track_samples = samples[track_id]
    if len(track_samples) < max_samples:
        track_samples.append(sample)
        return
    replacement_index = rng.randrange(seen)
    if replacement_index < max_samples:
        track_samples[replacement_index] = sample


def collect_tracking_crops(
    model_path: Path,
    video_path: Path,
    crop_override: list[int] | None,
    sample_stride: int,
    max_samples_per_track: int,
    jpeg_quality: int,
    seed: int,
    max_frames: int,
) -> dict[int, list[CropSample]]:
    model_path = model_path.expanduser().resolve()
    video_path = video_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Модель: {model_path}")
    print(f"Видео:  {video_path}")
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop_x1, crop_y1, crop_x2, crop_y2 = select_battlefield(
        width, height, crop_override
    )
    print(
        f"Кадр: {width}x{height}; battlefield: "
        f"({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})"
    )

    bars_place: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    bar_for_level: dict[int, float] = defaultdict(float)
    eligible_counts: dict[int, int] = defaultdict(int)
    sampled_counts: dict[int, int] = defaultdict(int)
    samples: dict[int, list[CropSample]] = defaultdict(list)
    rng = random.Random(seed)
    frame_index = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            battlefield = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            result = model.track(
                source=battlefield,
                persist=True,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                max_det=MAX_DET,
                device=DEVICE,
                quantize=QUANTIZE,
                tracker=TRACKER,
                verbose=False,
            )[0]

            detections: list[tuple[tuple[float, float, float, float], int, int]] = []
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                detections = [
                    (tuple(map(float, box)), track_id, class_id)
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids)
                ]

            # First update the recent positions of all level markers.
            for box, track_id, class_id in detections:
                if is_valid_level_detection(box, class_id):
                    bars_place[track_id].append(box)
                    del bars_place[track_id][:-LEN_POSES]

            # Match full bars to tracked level markers, as in predict_video.py.
            for box, _, class_id in detections:
                if class_id != BAR_CLASS_ID:
                    continue
                x1, y1, x2, y2 = box
                x_left = x1
                y_left = (y1 + y2) / 2
                for level_track_id, level_positions in bars_place.items():
                    matching_widths = []
                    for level_x1, level_y1, level_x2, level_y2 in level_positions:
                        level_center_x = (level_x1 + level_x2) / 2
                        if (
                            level_y1 <= y_left <= level_y2
                            and level_center_x
                            <= x_left
                            <= level_center_x + (level_x2 - level_x1)
                        ):
                            matching_widths.append(x2 - x1)
                    if (
                        len(level_positions) >= LEN_POSES
                        and len(matching_widths) > LEN_POSES / 2
                    ):
                        bar_for_level[level_track_id] = sum(matching_widths) / len(
                            matching_widths
                        )

            # Extract clean blue_rect crops before any drawing is applied.
            for box, track_id, class_id in detections:
                if not is_valid_level_detection(box, class_id):
                    continue
                eligible_counts[track_id] += 1
                if (eligible_counts[track_id] - 1) % sample_stride:
                    continue
                crop = extract_blue_rect(
                    battlefield,
                    box,
                    bar_for_level[track_id],
                )
                if crop is None:
                    continue
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not encoded_ok:
                    continue
                add_reservoir_sample(
                    samples=samples,
                    sampled_counts=sampled_counts,
                    track_id=track_id,
                    sample=CropSample(frame_index, encoded.tobytes()),
                    max_samples=max_samples_per_track,
                    rng=rng,
                )

            if frame_index % 100 == 0 or frame_index == total_frames:
                progress = frame_index / total_frames * 100 if total_frames else 0
                crop_count = sum(len(items) for items in samples.values())
                print(
                    f"\r{progress:6.2f}% | {frame_index}/{total_frames} | "
                    f"tracks={len(samples)} | crops={crop_count}",
                    end="",
                    flush=True,
                )
            if max_frames and frame_index >= max_frames:
                break
    finally:
        cap.release()

    print()
    print(
        f"Tracking завершён: кадров={frame_index}, "
        f"tracks={len(samples)}, crops={sum(map(len, samples.values()))}"
    )
    return dict(samples)


def canonical_model_class(
    model_class_name: str,
    known_classes: list[str],
) -> str | None:
    names_by_casefold = {name.casefold(): name for name in known_classes}
    requested_name = CLASS_NAME_ALIASES.get(
        model_class_name.casefold(),
        model_class_name,
    )
    return names_by_casefold.get(requested_name.casefold())


def predict_track_class(
    classifier: YOLO,
    samples: list[CropSample],
    known_classes: list[str],
    sample_count: int,
    imgsz: int,
    device: str,
    quantize: int,
) -> tuple[ClassPrediction | None, list[ClassPrediction]]:
    """Classify evenly spaced samples and average probabilities over a track."""
    count = min(sample_count, len(samples))
    indexes = np.linspace(0, len(samples) - 1, count, dtype=int)
    images = [samples[index].decode() for index in indexes]
    results = classifier.predict(
        source=images,
        imgsz=imgsz,
        device=device,
        quantize=quantize,
        verbose=False,
    )

    probabilities = []
    names: dict[int, str] | None = None
    for result in results:
        if result.probs is None:
            raise RuntimeError("Classifier did not return class probabilities")
        probabilities.append(result.probs.data.detach().cpu().numpy())
        names = result.names
    if not probabilities or names is None:
        raise RuntimeError("Classifier returned no results")

    mean_probabilities = np.mean(np.stack(probabilities), axis=0)
    top_count = min(3, len(mean_probabilities))
    top_ids = np.argsort(mean_probabilities)[::-1][:top_count]
    predictions: list[ClassPrediction] = []
    suggestion: ClassPrediction | None = None
    for rank, class_id in enumerate(top_ids):
        model_class_name = names[int(class_id)]
        class_name = canonical_model_class(model_class_name, known_classes)
        prediction = ClassPrediction(
            class_name=class_name or model_class_name,
            confidence=float(mean_probabilities[class_id]),
            model_class_name=model_class_name,
        )
        predictions.append(prediction)
        if rank == 0 and class_name is not None:
            suggestion = prediction
    return suggestion, predictions


def build_preview(
    track_id: int,
    samples: list[CropSample],
    preview_count: int,
    tile_size: int = 224,
    columns: int = 3,
) -> np.ndarray:
    count = min(preview_count, len(samples))
    indexes = np.linspace(0, len(samples) - 1, count, dtype=int)
    selected = [samples[index] for index in indexes]
    rows = math.ceil(len(selected) / columns)
    canvas = np.full((rows * tile_size, columns * tile_size, 3), 32, dtype=np.uint8)

    for index, sample in enumerate(selected):
        image = sample.decode()
        image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        cv2.putText(
            image,
            f"track {track_id} | frame {sample.frame_index}",
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(index, columns)
        canvas[
            row * tile_size : (row + 1) * tile_size,
            column * tile_size : (column + 1) * tile_size,
        ] = image
    return canvas


def resolve_classes_dir(classes_dir: Path) -> Path:
    classes_dir = classes_dir.expanduser().resolve()
    if not classes_dir.is_dir():
        raise FileNotFoundError(f"Classes directory not found: {classes_dir}")

    # Some archives contain an extra `segment/segment` wrapper.
    nested_segment = classes_dir / "segment"
    if (
        not (classes_dir / "segment_allowed.txt").is_file()
        and nested_segment.is_dir()
        and (nested_segment / "segment_allowed.txt").is_file()
    ):
        classes_dir = nested_segment
    return classes_dir


def load_known_classes(classes_dir: Path) -> tuple[Path, list[str]]:
    classes_dir = resolve_classes_dir(classes_dir)
    allowed_file = classes_dir / "segment_allowed.txt"
    if allowed_file.is_file():
        names = [
            line.strip()
            for line in allowed_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        names = sorted(
            path.name
            for path in classes_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError(f"No classes found in: {classes_dir}")
    return classes_dir, names


def safe_class_name(raw_name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw_name.strip()).strip(" .")
    if not name or name in {".", ".."}:
        raise ValueError("Class name is empty")
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", name):
        name = f"_{name}"
    return name


def resolve_class_input(raw_value: str, known_classes: list[str]) -> str:
    if raw_value.isdigit():
        index = int(raw_value) - 1
        if not 0 <= index < len(known_classes):
            raise ValueError(f"Class number must be from 1 to {len(known_classes)}")
        return known_classes[index]

    requested_name = safe_class_name(raw_value)
    names_by_casefold = {name.casefold(): name for name in known_classes}
    if requested_name.casefold() not in names_by_casefold:
        raise ValueError(
            f"Unknown class: {requested_name!r}. Use ?text to search the class list."
        )
    return names_by_casefold[requested_name.casefold()]


def show_class_search(query: str, known_classes: list[str]) -> None:
    matches = [
        (index, name)
        for index, name in enumerate(known_classes, start=1)
        if query.lower() in name.lower()
    ]
    if not matches:
        print("Совпадений нет")
        return
    for index, name in matches:
        print(f"  {index:3d}: {name}")


def label_and_save_tracks(
    samples_by_track: dict[int, list[CropSample]],
    output_dir: Path,
    video_stem: str,
    known_classes: list[str],
    classifier: YOLO,
    classification_samples: int,
    classification_imgsz: int,
    classification_device: str,
    classification_quantize: int,
    preview_count: int,
    min_track_samples: int,
) -> tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_tracks = 0
    skipped_tracks = 0
    saved_images = 0
    stop = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    ordered_tracks = sorted(
        samples_by_track.items(),
        key=lambda item: min(sample.frame_index for sample in item[1]),
    )
    for position, (track_id, samples) in enumerate(ordered_tracks, start=1):
        samples.sort(key=lambda sample: sample.frame_index)
        if len(samples) < min_track_samples:
            skipped_tracks += 1
            continue

        first_frame = samples[0].frame_index
        last_frame = samples[-1].frame_index
        suggestion, model_predictions = predict_track_class(
            classifier=classifier,
            samples=samples,
            known_classes=known_classes,
            sample_count=classification_samples,
            imgsz=classification_imgsz,
            device=classification_device,
            quantize=classification_quantize,
        )
        preview = build_preview(track_id, samples, preview_count)
        cv2.imshow(WINDOW_NAME, preview)
        cv2.waitKey(1)
        print()
        print(
            f"Track {track_id} ({position}/{len(ordered_tracks)}): "
            f"samples={len(samples)}, frames={first_frame}..{last_frame}"
        )
        print(
            f"Модель по {min(classification_samples, len(samples))} "
            "кадрам track: "
            + ", ".join(
                f"{prediction.class_name}={prediction.confidence:.3f}"
                for prediction in model_predictions
            )
        )
        if suggestion is None:
            print(
                "Лучший класс модели отсутствует в dataset_centered; "
                "выберите класс вручную."
            )
        elif suggestion.model_class_name != suggestion.class_name:
            print(
                f"Сопоставление класса: {suggestion.model_class_name} -> "
                f"{suggestion.class_name}"
            )

        while True:
            raw_value = input(
                "Класс (имя/номер; = — принять модель; ?текст — поиск; "
                "s — пропустить; q — закончить): "
            ).strip()
            command = raw_value.lower()
            if command == "=":
                if suggestion is None:
                    print("Предложенный класс нельзя использовать")
                    continue
                class_name = suggestion.class_name
                print(
                    f"Принят класс модели: {class_name} "
                    f"({suggestion.confidence:.3f})"
                )
                break
            if command == "s":
                skipped_tracks += 1
                class_name = None
                break
            if command == "q":
                class_name = None
                stop = True
                break
            if raw_value.startswith("?"):
                show_class_search(raw_value[1:].strip(), known_classes)
                continue
            try:
                class_name = resolve_class_input(raw_value, known_classes)
            except ValueError as exc:
                print(f"Ошибка: {exc}")
                continue
            if class_name != raw_value and not raw_value.isdigit():
                print(f"Имя каталога нормализовано: {class_name}")
            break

        if stop:
            break
        if class_name is None:
            continue
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        safe_video_stem = safe_class_name(video_stem)
        for sample in samples:
            filename = (
                f"{safe_video_stem}_track_{track_id:05d}_"
                f"frame_{sample.frame_index:07d}.jpg"
            )
            (class_dir / filename).write_bytes(sample.jpeg)
            saved_images += 1
        saved_tracks += 1
        (output_dir / "classes.txt").write_text(
            "\n".join(known_classes) + "\n",
            encoding="utf-8",
        )
        print(f"Сохранено в {class_dir}: {len(samples)}")

    cv2.destroyAllWindows()
    return saved_tracks, skipped_tracks, saved_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track level markers, preview several blue_rect crops per object, "
            "then save manually labeled classification data"
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--classifier",
        type=Path,
        default=DEFAULT_CLASSIFIER,
        help="Trained YOLO classification model used to suggest a track class.",
    )
    parser.add_argument(
        "--classification-samples",
        type=int,
        default=6,
        help="Number of evenly spaced track crops used for the suggestion.",
    )
    parser.add_argument("--classification-imgsz", type=int, default=224)
    parser.add_argument("--classification-device", default="0")
    parser.add_argument(
        "--classification-quantize",
        type=int,
        choices=(16, 32),
        default=16,
    )
    parser.add_argument(
        "--classes-dir",
        type=Path,
        default=DEFAULT_CLASSES_DIR,
        help=(
            "Directory containing class subdirectories or segment_allowed.txt "
            "(default: dataset_centered)"
        ),
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Override battlefield crop for an unsupported video resolution",
    )
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--max-samples-per-track", type=int, default=80)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--min-track-samples", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only this many frames; 0 means the full video",
    )
    args = parser.parse_args()

    if args.sample_stride < 1:
        parser.error("--sample-stride must be at least 1")
    if args.max_samples_per_track < 1:
        parser.error("--max-samples-per-track must be at least 1")
    if args.preview_count < 1:
        parser.error("--preview-count must be at least 1")
    if args.classification_samples < 1:
        parser.error("--classification-samples must be at least 1")
    if args.classification_imgsz < 32:
        parser.error("--classification-imgsz must be at least 32")
    if args.min_track_samples < 1:
        parser.error("--min-track-samples must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be from 1 to 100")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    try:
        classes_dir, known_classes = load_known_classes(args.classes_dir)
        print(f"Классы: {classes_dir} ({len(known_classes)})")
        classifier_path = args.classifier.expanduser().resolve()
        if not classifier_path.is_file():
            raise FileNotFoundError(
                f"Classification model not found: {classifier_path}"
            )
        samples_by_track = collect_tracking_crops(
            model_path=args.model,
            video_path=args.video,
            crop_override=args.crop,
            sample_stride=args.sample_stride,
            max_samples_per_track=args.max_samples_per_track,
            jpeg_quality=args.jpeg_quality,
            seed=args.seed,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not samples_by_track:
        print("Подходящие tracking-объекты не найдены")
        return

    print(f"Классификатор: {classifier_path}")
    classifier = YOLO(str(classifier_path))
    saved_tracks, skipped_tracks, saved_images = label_and_save_tracks(
        samples_by_track=samples_by_track,
        output_dir=args.output.expanduser().resolve(),
        video_stem=args.video.stem,
        known_classes=known_classes,
        classifier=classifier,
        classification_samples=args.classification_samples,
        classification_imgsz=args.classification_imgsz,
        classification_device=args.classification_device,
        classification_quantize=args.classification_quantize,
        preview_count=args.preview_count,
        min_track_samples=args.min_track_samples,
    )
    print()
    print(
        f"Готово: tracks={saved_tracks}, skipped={skipped_tracks}, "
        f"images={saved_images}, output={args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
