from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from model_paths import (
    BATTLEFIELDS,
    CLASSIFICATION_MODEL_PATH,
    DETECTION_ENGINE_PATH,
    ELIXIR_DETECTION_ENGINE_PATH,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "screenshots/input_omydays.mp4"
DEFAULT_OUTPUT = SCRIPT_DIR / "screenshots/output_tracked_fast.mp4"

MODEL_PATH = DETECTION_ENGINE_PATH
ELIXIR_MODEL_PATH = ELIXIR_DETECTION_ENGINE_PATH

IMGSZ = 1280
CLASSIFICATION_IMGSZ = 224
CONF = 0.20
IOU = 0.50
MAX_DET = 500
DEVICE = "0"
QUANTIZE = 16

BAR_CLASS_ID = 0
LEVEL_CLASS_ID = 1
POSITION_HISTORY = 5
CLASSIFICATION_CROP_SIZE = 128
LEVEL_SIZE_RESTRICTIONS = ((10, 10), (50, 60))


@dataclass(frozen=True)
class TrackedDetection:
    box: tuple[float, float, float, float]
    track_id: int
    class_id: int
    confidence: float


@dataclass(frozen=True)
class CachedClassification:
    text: str
    frame_number: int


def validate_model(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def select_battlefield(
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    video_size = (width, height)
    if video_size not in BATTLEFIELDS:
        raise ValueError(
            f"Unsupported video resolution: {width}x{height}. "
            "Add its crop to BATTLEFIELDS in model_paths.py."
        )
    x1, y1, x2, y2 = BATTLEFIELDS[video_size]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Invalid BATTLEFIELDS crop {(x1, y1, x2, y2)} "
            f"for video size {width}x{height}"
        )
    return x1, y1, x2, y2


def is_level_detection(detection: TrackedDetection) -> bool:
    x1, y1, x2, y2 = detection.box
    return (
        detection.class_id == LEVEL_CLASS_ID
        and LEVEL_SIZE_RESTRICTIONS[0][0]
        <= x2 - x1
        <= LEVEL_SIZE_RESTRICTIONS[1][0]
        and LEVEL_SIZE_RESTRICTIONS[0][1]
        <= y2 - y1
        <= LEVEL_SIZE_RESTRICTIONS[1][1]
    )


def tracked_detections(result) -> list[TrackedDetection]:
    if result.boxes is None or result.boxes.id is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    track_ids = result.boxes.id.int().detach().cpu().tolist()
    class_ids = result.boxes.cls.int().detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    return [
        TrackedDetection(
            box=tuple(map(float, box)),
            track_id=int(track_id),
            class_id=int(class_id),
            confidence=float(confidence),
        )
        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            class_ids,
            confidences,
        )
    ]


def update_bar_widths(
    detections: list[TrackedDetection],
    level_positions: dict[int, deque[tuple[float, float, float, float]]],
    bar_widths: dict[int, float],
) -> None:
    for detection in detections:
        if is_level_detection(detection):
            level_positions[detection.track_id].append(detection.box)

    for detection in detections:
        if detection.class_id != BAR_CLASS_ID:
            continue
        bar_x1, bar_y1, bar_x2, bar_y2 = detection.box
        bar_center_y = (bar_y1 + bar_y2) / 2
        for track_id, positions in level_positions.items():
            matching_widths: list[float] = []
            for level_x1, level_y1, level_x2, level_y2 in positions:
                level_center_x = (level_x1 + level_x2) / 2
                if (
                    level_y1 <= bar_center_y <= level_y2
                    and level_center_x
                    <= bar_x1
                    <= level_center_x + (level_x2 - level_x1)
                ):
                    matching_widths.append(bar_x2 - bar_x1)
            if (
                len(positions) >= POSITION_HISTORY
                and len(matching_widths) > POSITION_HISTORY / 2
            ):
                bar_widths[track_id] = sum(matching_widths) / len(matching_widths)


def extract_classification_crop(
    image: np.ndarray,
    level_box: tuple[float, float, float, float],
    bar_width: float,
) -> np.ndarray | None:
    x1, _, x2, y2 = level_box
    center_x = (x1 + x2 + bar_width) / 2
    left = round(center_x - CLASSIFICATION_CROP_SIZE / 2)
    top = round(y2)
    right = left + CLASSIFICATION_CROP_SIZE
    bottom = top + CLASSIFICATION_CROP_SIZE

    image_height, image_width = image.shape[:2]
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image_width, right)
    source_bottom = min(image_height, bottom)
    if source_right <= source_left or source_bottom <= source_top:
        return None

    crop = np.full(
        (CLASSIFICATION_CROP_SIZE, CLASSIFICATION_CROP_SIZE, 3),
        114,
        dtype=np.uint8,
    )
    target_left = source_left - left
    target_top = source_top - top
    crop[
        target_top : target_top + source_bottom - source_top,
        target_left : target_left + source_right - source_left,
    ] = image[source_top:source_bottom, source_left:source_right]
    return crop


def update_classification_cache(
    classification_model: YOLO,
    battlefield: np.ndarray,
    detections: list[TrackedDetection],
    bar_widths: dict[int, float],
    cache: dict[int, CachedClassification],
    frame_number: int,
    refresh_frames: int,
) -> None:
    pending: list[tuple[int, np.ndarray]] = []
    pending_ids: set[int] = set()

    for detection in detections:
        if not is_level_detection(detection):
            continue
        track_id = detection.track_id
        bar_width = bar_widths.get(track_id, 0.0)
        if bar_width <= 0 or track_id in pending_ids:
            continue
        cached = cache.get(track_id)
        if cached is not None and (
            refresh_frames == 0
            or frame_number - cached.frame_number < refresh_frames
        ):
            continue
        crop = extract_classification_crop(battlefield, detection.box, bar_width)
        if crop is not None:
            pending.append((track_id, crop))
            pending_ids.add(track_id)

    if not pending:
        return

    results = classification_model.predict(
        source=[crop for _, crop in pending],
        imgsz=CLASSIFICATION_IMGSZ,
        device=DEVICE,
        quantize=QUANTIZE,
        verbose=False,
    )
    for (track_id, _), result in zip(pending, results):
        if result.probs is None:
            continue
        class_id = int(result.probs.top1)
        cache[track_id] = CachedClassification(
            text=result.names[class_id],
            frame_number=frame_number,
        )


def draw_track_details(
    image: np.ndarray,
    detections: list[TrackedDetection],
    bar_widths: dict[int, float],
    classifications: dict[int, CachedClassification],
) -> None:
    for detection in detections:
        if not is_level_detection(detection):
            continue
        x1, y1, x2, y2 = detection.box
        bar_width = bar_widths.get(detection.track_id, 0.0)
        right = round(x2 + bar_width)
        cv2.rectangle(
            image,
            (round(x1), round(y1)),
            (right, round(y2)),
            (0, 0, 255),
            2,
        )

        crop_left = round((x1 + right) / 2 - CLASSIFICATION_CROP_SIZE / 2)
        cv2.rectangle(
            image,
            (crop_left, round(y2)),
            (
                crop_left + CLASSIFICATION_CROP_SIZE,
                round(y2) + CLASSIFICATION_CROP_SIZE,
            ),
            (255, 0, 0),
            1,
        )
        classification = classifications.get(detection.track_id)
        if classification is not None:
            cv2.putText(
                image,
                classification.text,
                (round(x1), max(20, round(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


def process_video(
    input_path: Path,
    output_path: Path,
    *,
    display: bool,
    max_frames: int,
    classification_refresh: int,
    bar_conf: float,
    elixir_conf: float,
) -> None:
    bars_path = validate_model(MODEL_PATH, "Bars TensorRT engine")
    elixir_path = validate_model(ELIXIR_MODEL_PATH, "Elixir TensorRT engine")
    classifier_path = validate_model(
        CLASSIFICATION_MODEL_PATH,
        "Classification model",
    )
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    bars_model = YOLO(str(bars_path))
    elixir_model = YOLO(str(elixir_path))
    classification_model = YOLO(str(classifier_path))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop_x1, crop_y1, crop_x2, crop_y2 = select_battlefield(width, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    level_positions: dict[
        int,
        deque[tuple[float, float, float, float]],
    ] = defaultdict(lambda: deque(maxlen=POSITION_HISTORY))
    bar_widths: dict[int, float] = {}
    classifications: dict[int, CachedClassification] = {}
    frame_number = 0
    processing_started = time.perf_counter()

    print(f"Bars model:       {bars_path}")
    print(f"Elixir model:     {elixir_path}")
    print(f"Classifier:       {classifier_path}")
    print(f"Input:            {input_path}")
    print(f"Output:           {output_path}")
    print(f"Classify refresh: {classification_refresh} frames (0 = once per track)")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            clean_battlefield = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            inference_started = time.perf_counter()
            bars_result = bars_model.track(
                source=clean_battlefield,
                persist=True,
                imgsz=IMGSZ,
                conf=bar_conf,
                iou=IOU,
                max_det=MAX_DET,
                device=DEVICE,
                quantize=QUANTIZE,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
            elixir_result = elixir_model.predict(
                source=clean_battlefield,
                imgsz=IMGSZ,
                conf=elixir_conf,
                iou=IOU,
                max_det=MAX_DET,
                device=DEVICE,
                quantize=QUANTIZE,
                verbose=False,
            )[0]

            detections = tracked_detections(bars_result)
            update_bar_widths(detections, level_positions, bar_widths)
            update_classification_cache(
                classification_model=classification_model,
                battlefield=clean_battlefield,
                detections=detections,
                bar_widths=bar_widths,
                cache=classifications,
                frame_number=frame_number,
                refresh_frames=classification_refresh,
            )
            inference_time = time.perf_counter() - inference_started

            # Elixir results are drawn directly and are never classified.
            annotated = bars_result.plot(img=clean_battlefield)
            annotated = elixir_result.plot(img=annotated)
            draw_track_details(
                annotated,
                detections,
                bar_widths,
                classifications,
            )

            bar_count = len(detections)
            elixir_count = 0 if elixir_result.boxes is None else len(elixir_result.boxes)
            cv2.putText(
                annotated,
                f"Bars: {bar_count} | Elixir: {elixir_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Inference: {1 / inference_time:.1f} FPS" if inference_time else "",
                (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            output_frame = frame.copy()
            output_frame[crop_y1:crop_y2, crop_x1:crop_x2] = annotated
            writer.write(output_frame)

            if display:
                scale = min(1.0, 1000 / height, 1000 / width)
                preview = cv2.resize(
                    output_frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow("Fast tracking", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_number % 30 == 0 or frame_number == total_frames:
                elapsed = time.perf_counter() - processing_started
                processing_fps = frame_number / elapsed if elapsed else 0.0
                print(
                    f"\r{frame_number}/{total_frames or '?'} | "
                    f"bars={bar_count} | elixir={elixir_count} | "
                    f"processing={processing_fps:.1f} FPS",
                    end="",
                    flush=True,
                )
            if max_frames and frame_number >= max_frames:
                break
    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - processing_started
    print()
    print(
        f"Done: frames={frame_number}, time={elapsed:.2f}s, "
        f"average={frame_number / elapsed if elapsed else 0.0:.2f} FPS"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast dual-detector video processing with cached unit "
            "classification and no elixir classification."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bar-conf", type=float, default=CONF)
    parser.add_argument("--elixir-conf", type=float, default=CONF)
    parser.add_argument(
        "--classification-refresh",
        type=int,
        default=30,
        help="Reclassify a tracked unit after N frames; 0 means only once.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process at most N frames; 0 means the complete video.",
    )
    args = parser.parse_args()

    for name in ("bar_conf", "elixir_conf"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.classification_refresh < 0:
        parser.error("--classification-refresh cannot be negative")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        display=args.display,
        max_frames=args.max_frames,
        classification_refresh=args.classification_refresh,
        bar_conf=args.bar_conf,
        elixir_conf=args.elixir_conf,
    )


if __name__ == "__main__":
    main()
