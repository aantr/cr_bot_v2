from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from model_paths import BATTLEFIELDS, ELIXIR_DETECTION_ENGINE_PATH


DEFAULT_MODEL = ELIXIR_DETECTION_ENGINE_PATH
WINDOW_NAME = "Build elixir dataset"

SAVE_KEYS = {10, 13, ord("a"), ord("A"), ord("y"), ord("Y")}
SKIP_KEYS = {32, ord("n"), ord("N"), ord("s"), ord("S")}
QUIT_KEYS = {27, ord("q"), ord("Q")}


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


def select_frame_area(
    width: int,
    height: int,
    crop_override: list[int] | None,
    full_frame: bool,
) -> tuple[int, int, int, int]:
    if full_frame:
        return 0, 0, width, height
    if crop_override is not None:
        return validate_crop(width, height, tuple(crop_override))

    video_size = (width, height)
    if video_size not in BATTLEFIELDS:
        raise ValueError(
            f"Unsupported video resolution: {width}x{height}. "
            "Pass --crop X1 Y1 X2 Y2 or use --full-frame."
        )
    return validate_crop(width, height, BATTLEFIELDS[video_size])


def safe_video_stem(value: str) -> str:
    value = re.sub(r'[^0-9A-Za-zА-Яа-яЁё._-]+', "_", value).strip(" ._")
    return value or "video"


def extract_yolo_rows(result) -> np.ndarray:
    """Return rows shaped as x_center, y_center, width, height, class_id."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)

    xywhn = result.boxes.xywhn.detach().cpu().numpy().astype(np.float32)
    class_ids = (
        result.boxes.cls.detach().cpu().numpy().astype(np.int32).reshape(-1, 1)
    )
    if len(xywhn) != len(class_ids):
        raise RuntimeError("Model returned different numbers of boxes and classes")
    return np.concatenate((xywhn, class_ids.astype(np.float32)), axis=1)


def format_yolo_labels(rows: np.ndarray) -> str:
    lines: list[str] = []
    for row in np.asarray(rows, dtype=np.float32).reshape(-1, 5):
        coords = row[:4]
        if not np.isfinite(coords).all():
            raise ValueError(f"Non-finite YOLO coordinates: {coords}")
        if row[2] <= 0 or row[3] <= 0:
            raise ValueError(f"YOLO box must have positive width and height: {coords}")
        coords = np.clip(coords, 0.0, 1.0)
        lines.append(
            f"{int(row[4])} "
            + " ".join(f"{float(value):.6f}" for value in coords)
        )
    return "\n".join(lines) + ("\n" if lines else "")


def load_manifest(manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        return []
    entries: list[str] = []
    seen: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and entry not in seen:
            entries.append(entry)
            seen.add(entry)
    return entries


def write_manifest(manifest_path: Path, entries: list[str]) -> None:
    temporary_path = manifest_path.with_suffix(".txt.tmp")
    temporary_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)


def next_sample_paths(
    output_dir: Path,
    video_stem: str,
    frame_number: int,
) -> tuple[Path, Path]:
    base_name = f"{safe_video_stem(video_stem)}_frame_{frame_number:08d}"
    suffix = 1
    while True:
        name = base_name if suffix == 1 else f"{base_name}_{suffix}"
        image_path = output_dir / f"{name}.jpg"
        label_path = output_dir / f"{name}.txt"
        if not image_path.exists() and not label_path.exists():
            return image_path, label_path
        suffix += 1


def save_sample(
    output_dir: Path,
    video_stem: str,
    frame_number: int,
    frame: np.ndarray,
    yolo_rows: np.ndarray,
    manifest_entries: list[str],
    jpeg_quality: int,
) -> tuple[Path, Path]:
    image_path, label_path = next_sample_paths(
        output_dir,
        video_stem,
        frame_number,
    )
    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not encoded_ok:
        raise RuntimeError(f"Cannot encode frame {frame_number} as JPEG")

    image_path.write_bytes(encoded.tobytes())
    try:
        label_path.write_text(format_yolo_labels(yolo_rows), encoding="utf-8")
        entry = f"./{image_path.name}"
        if entry not in manifest_entries:
            manifest_entries.append(entry)
        write_manifest(output_dir / "yolo_annotations.txt", manifest_entries)
    except BaseException:
        image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
        raise
    return image_path, label_path


def build_preview(
    result,
    frame_number: int,
    total_frames: int,
    saved_count: int,
) -> np.ndarray:
    preview = result.plot(labels=True, conf=True, boxes=True)
    detection_count = 0 if result.boxes is None else len(result.boxes)
    total_text = str(total_frames) if total_frames > 0 else "?"
    lines = (
        f"Frame {frame_number}/{total_text} | detections: {detection_count} | saved: {saved_count}",
        "Enter/A: add to dataset | Space/S: skip | Q/Esc: quit",
    )
    overlay_height = 62
    cv2.rectangle(preview, (0, 0), (preview.shape[1], overlay_height), (20, 20, 20), -1)
    for index, text in enumerate(lines):
        cv2.putText(
            preview,
            text,
            (10, 24 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def wait_for_action() -> str:
    while True:
        key = cv2.waitKeyEx(0)
        if key < 0:
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    return "quit"
            except cv2.error:
                return "quit"
            continue
        key &= 0xFF
        if key in SAVE_KEYS:
            return "save"
        if key in SKIP_KEYS:
            return "skip"
        if key in QUIT_KEYS:
            return "quit"


def collect_frames(args: argparse.Namespace) -> tuple[int, int, int, int]:
    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop_x1, crop_y1, crop_x2, crop_y2 = select_frame_area(
        width,
        height,
        args.crop,
        args.full_frame,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "yolo_annotations.txt"
    manifest_entries = load_manifest(manifest_path)
    model = YOLO(str(model_path))

    print(f"Model:   {model_path}")
    print(f"Video:   {video_path}")
    print(f"Output:  {output_dir}")
    print(
        f"Frames:  {total_frames or '?'}; source={width}x{height}; "
        f"saved area=({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})"
    )
    print("Controls: Enter/A=add, Space/S=skip, Q/Esc=quit")

    reviewed = 0
    saved = 0
    skipped = 0
    empty_skipped = 0
    frame_number = args.start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    preview_width = crop_x2 - crop_x1
    preview_height = crop_y2 - crop_y1
    scale = min(1.0, 1400 / preview_width, 900 / preview_height)
    cv2.resizeWindow(
        WINDOW_NAME,
        max(320, round(preview_width * scale)),
        max(240, round(preview_height * scale)),
    )
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    try:
        while cap.isOpened():
            if args.end_frame and frame_number > args.end_frame:
                break
            ok, full_frame = cap.read()
            if not ok:
                break

            should_review = (frame_number - args.start_frame) % args.stride == 0
            if not should_review:
                frame_number += 1
                continue

            frame = full_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )[0]
            yolo_rows = extract_yolo_rows(result)

            if not len(yolo_rows) and not args.show_empty:
                empty_skipped += 1
                frame_number += 1
                continue

            reviewed += 1
            preview = build_preview(result, frame_number, total_frames, saved)
            cv2.imshow(WINDOW_NAME, preview)
            action = wait_for_action()

            if action == "quit":
                break
            if action == "save":
                image_path, label_path = save_sample(
                    output_dir=output_dir,
                    video_stem=video_path.stem,
                    frame_number=frame_number,
                    frame=frame,
                    yolo_rows=yolo_rows,
                    manifest_entries=manifest_entries,
                    jpeg_quality=args.jpeg_quality,
                )
                saved += 1
                print(
                    f"Saved frame {frame_number}: {image_path.name}, "
                    f"{label_path.name} ({len(yolo_rows)} objects)"
                )
            else:
                skipped += 1

            if args.max_reviewed and reviewed >= args.max_reviewed:
                break
            frame_number += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return reviewed, saved, skipped, empty_skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review detections from a video and save accepted battlefield frames "
            "with YOLO labels."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Override the battlefield crop for an unsupported video resolution.",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Run detection on and save the complete video frame.",
    )
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=0,
        help="Last frame to process (inclusive); 0 means the end of the video.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Review every Nth frame.",
    )
    parser.add_argument(
        "--max-reviewed",
        type=int,
        default=0,
        help="Stop after this many reviewed frames; 0 means unlimited.",
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Also review frames where the model found no objects.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")
    if args.max_det < 1:
        parser.error("--max-det must be at least 1")
    if args.start_frame < 1:
        parser.error("--start-frame must be at least 1")
    if args.end_frame < 0:
        parser.error("--end-frame cannot be negative")
    if args.end_frame and args.end_frame < args.start_frame:
        parser.error("--end-frame must be greater than or equal to --start-frame")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.max_reviewed < 0:
        parser.error("--max-reviewed cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be from 1 to 100")
    if args.crop is not None and args.full_frame:
        parser.error("--crop and --full-frame cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    try:
        reviewed, saved, skipped, empty_skipped = collect_frames(args)
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print()
    print(
        f"Done: reviewed={reviewed}, saved={saved}, skipped={skipped}, "
        f"empty_auto_skipped={empty_skipped}, output={args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
