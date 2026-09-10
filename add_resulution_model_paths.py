from __future__ import annotations

import argparse
import ast
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATHS = SCRIPT_DIR / "model_paths.py"
FRAME_WINDOW = "Select video frame"
ROI_WINDOW = "Select model_paths region"
REVIEW_WINDOW = "Review selected regions"

ACCEPT_KEYS = {10, 13, 32}
QUIT_KEYS = {27, ord("q"), ord("Q")}
RESTART_KEYS = {ord("r"), ord("R")}


@dataclass(frozen=True)
class RegionSpec:
    field_name: str
    display_name: str
    color: tuple[int, int, int]
    store_as_list: bool = False


REGIONS = (
    RegionSpec("BATTLEFIELDS", "1/7 Battlefield", (0, 255, 0)),
    RegionSpec("CARDS", "2/7 Cards", (255, 255, 0)),
    RegionSpec("ELIXIR_BAR", "3/7 Elixir bar", (255, 0, 255)),
    RegionSpec("TOWER_HP_1", "4/7 Ally tower HP 1", (255, 128, 0), True),
    RegionSpec("TOWER_HP_2", "5/7 Ally tower HP 2", (255, 128, 0), True),
    RegionSpec(
        "TOWER_HP_ENEMY_1",
        "6/7 Enemy tower HP 1",
        (0, 128, 255),
        True,
    ),
    RegionSpec(
        "TOWER_HP_ENEMY_2",
        "7/7 Enemy tower HP 2",
        (0, 128, 255),
        True,
    ),
)


class UserCancelled(Exception):
    pass


def resize_for_preview(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale >= 1.0:
        return image.copy(), 1.0
    preview = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return preview, scale


def put_help(image: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 24
    panel_height = 12 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, 22 + index * line_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def read_video_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Не удалось прочитать кадр {frame_index}")
    return frame


def select_video_frame(
    path: Path,
    frame_index: int | None,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть как изображение или видео: {path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count < 1:
            raise RuntimeError(f"Видео не содержит кадров: {path}")

        if frame_index is not None:
            if not 0 <= frame_index < frame_count:
                raise ValueError(
                    f"--frame-index должен быть от 0 до {frame_count - 1}"
                )
            return read_video_frame(capture, frame_index), frame_index

        selected = 0
        displayed = -1
        selected_frame: np.ndarray | None = None
        trackbar_max = max(1, frame_count - 1)
        cv2.namedWindow(FRAME_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Frame", FRAME_WINDOW, selected, trackbar_max, lambda _: None)

        while True:
            requested = min(
                frame_count - 1,
                cv2.getTrackbarPos("Frame", FRAME_WINDOW),
            )
            if requested != displayed:
                selected_frame = read_video_frame(capture, requested)
                displayed = requested
                selected = requested
                preview, _ = resize_for_preview(
                    selected_frame,
                    max_width,
                    max_height,
                )
                seconds = selected / fps if math.isfinite(fps) and fps > 0 else 0.0
                put_help(
                    preview,
                    [
                        f"Frame {selected}/{frame_count - 1}  time={seconds:.2f}s",
                        "Trackbar or A/D: 1 frame, J/L: 1 second",
                        "ENTER/SPACE: use frame, Q/ESC: cancel",
                    ],
                )
                cv2.imshow(FRAME_WINDOW, preview)

            key = cv2.waitKeyEx(30)
            low_key = key & 0xFF if key >= 0 else -1
            if low_key in ACCEPT_KEYS and selected_frame is not None:
                cv2.destroyWindow(FRAME_WINDOW)
                return selected_frame, selected
            if low_key in QUIT_KEYS:
                raise UserCancelled

            step_second = max(1, round(fps)) if math.isfinite(fps) and fps > 0 else 30
            new_position = None
            if low_key in {ord("a"), ord("A")}:
                new_position = selected - 1
            elif low_key in {ord("d"), ord("D")}:
                new_position = selected + 1
            elif low_key in {ord("j"), ord("J")}:
                new_position = selected - step_second
            elif low_key in {ord("l"), ord("L")}:
                new_position = selected + step_second
            if new_position is not None:
                cv2.setTrackbarPos(
                    "Frame",
                    FRAME_WINDOW,
                    min(frame_count - 1, max(0, new_position)),
                )
    finally:
        capture.release()
        try:
            cv2.destroyWindow(FRAME_WINDOW)
        except cv2.error:
            pass


def load_source_frame(
    path: Path,
    frame_index: int | None,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, int | None]:
    image = read_image(path)
    if image is not None:
        if frame_index is not None:
            raise ValueError("--frame-index применяется только к видео")
        return image, None
    return select_video_frame(path, frame_index, max_width, max_height)


def draw_regions(
    frame: np.ndarray,
    regions: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    result = frame.copy()
    specs = {spec.field_name: spec for spec in REGIONS}
    for field_name, (x1, y1, x2, y2) in regions.items():
        spec = specs[field_name]
        cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), spec.color, 3)
        label_y = y1 - 8 if y1 >= 25 else min(result.shape[0] - 8, y2 + 22)
        cv2.putText(
            result,
            field_name,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            field_name,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            spec.color,
            2,
            cv2.LINE_AA,
        )
    return result


def preview_roi_to_source(
    roi: tuple[int, int, int, int],
    scale: float,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    x, y, roi_width, roi_height = roi
    if roi_width <= 0 or roi_height <= 0:
        raise UserCancelled
    x1 = max(0, min(source_width - 1, math.floor(x / scale)))
    y1 = max(0, min(source_height - 1, math.floor(y / scale)))
    x2 = max(x1 + 1, min(source_width, math.ceil((x + roi_width) / scale)))
    y2 = max(y1 + 1, min(source_height, math.ceil((y + roi_height) / scale)))
    return x1, y1, x2, y2


def select_regions(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> dict[str, tuple[int, int, int, int]]:
    source_height, source_width = frame.shape[:2]
    selected: dict[str, tuple[int, int, int, int]] = {}

    for spec in REGIONS:
        annotated = draw_regions(frame, selected)
        preview, scale = resize_for_preview(annotated, max_width, max_height)
        put_help(
            preview,
            [
                f"Select {spec.display_name}",
                "Drag rectangle, then ENTER/SPACE. C cancels selection.",
            ],
        )
        print(f"Выберите область {spec.field_name} и нажмите Enter/Space")
        roi = cv2.selectROI(
            ROI_WINDOW,
            preview,
            showCrosshair=True,
            fromCenter=False,
        )
        selected[spec.field_name] = preview_roi_to_source(
            tuple(map(int, roi)),
            scale,
            source_width,
            source_height,
        )
        print(f"  {spec.field_name}: {selected[spec.field_name]}")

    cv2.destroyWindow(ROI_WINDOW)
    return selected


def review_regions(
    frame: np.ndarray,
    regions: dict[str, tuple[int, int, int, int]],
    max_width: int,
    max_height: int,
) -> str:
    preview, _ = resize_for_preview(
        draw_regions(frame, regions),
        max_width,
        max_height,
    )
    put_help(
        preview,
        [
            "ENTER/SPACE: save all regions",
            "R: redraw all regions, Q/ESC: cancel",
        ],
    )
    cv2.imshow(REVIEW_WINDOW, preview)
    while True:
        key = cv2.waitKeyEx(0)
        low_key = key & 0xFF if key >= 0 else -1
        if low_key in ACCEPT_KEYS:
            cv2.destroyWindow(REVIEW_WINDOW)
            return "save"
        if low_key in RESTART_KEYS:
            cv2.destroyWindow(REVIEW_WINDOW)
            return "restart"
        if low_key in QUIT_KEYS:
            raise UserCancelled


def source_offset(
    line_offsets: list[int],
    lineno: int,
    column: int,
) -> int:
    return line_offsets[lineno - 1] + column


def resolution_from_ast(node: ast.expr) -> tuple[int, int] | None:
    if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
        return None
    values = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, int):
            return None
        values.append(element.value)
    return values[0], values[1]


def find_model_path_dicts(source: str) -> dict[str, ast.Dict]:
    tree = ast.parse(source)
    required = {spec.field_name for spec in REGIONS}
    dictionaries: dict[str, ast.Dict] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id in required
            and isinstance(node.value, ast.Dict)
        ):
            dictionaries[target.id] = node.value

    missing = sorted(required - dictionaries.keys())
    if missing:
        raise ValueError(
            "В model_paths.py отсутствуют словари: " + ", ".join(missing)
        )
    return dictionaries


def existing_resolution_fields(
    model_paths: Path,
    resolution: tuple[int, int],
) -> list[str]:
    source = model_paths.read_text(encoding="utf-8")
    dictionaries = find_model_path_dicts(source)
    return [
        spec.field_name
        for spec in REGIONS
        if any(
            resolution_from_ast(key) == resolution
            for key in dictionaries[spec.field_name].keys
            if key is not None
        )
    ]


def format_region_value(
    region: tuple[int, int, int, int],
    store_as_list: bool,
) -> str:
    formatted = f"({region[0]}, {region[1]}, {region[2]}, {region[3]})"
    return f"[{formatted}]" if store_as_list else formatted


def update_model_paths_source(
    source: str,
    resolution: tuple[int, int],
    regions: dict[str, tuple[int, int, int, int]],
) -> str:
    dictionaries = find_model_path_dicts(source)
    lines = source.splitlines(keepends=True)
    line_offsets: list[int] = []
    position = 0
    for line in lines:
        line_offsets.append(position)
        position += len(line)
    newline = "\r\n" if "\r\n" in source else "\n"
    edits: list[tuple[int, int, str]] = []

    for spec in REGIONS:
        dictionary = dictionaries[spec.field_name]
        replacement = format_region_value(
            regions[spec.field_name],
            spec.store_as_list,
        )
        existing_value = None
        for key, value in zip(dictionary.keys, dictionary.values):
            if key is not None and resolution_from_ast(key) == resolution:
                existing_value = value
                break

        if existing_value is not None:
            start = source_offset(
                line_offsets,
                existing_value.lineno,
                existing_value.col_offset,
            )
            end = source_offset(
                line_offsets,
                existing_value.end_lineno,
                existing_value.end_col_offset,
            )
            edits.append((start, end, replacement))
            continue

        closing_brace = source_offset(
            line_offsets,
            dictionary.end_lineno,
            dictionary.end_col_offset,
        ) - 1
        before_closing = source[:closing_brace]
        prefix = ""
        if dictionary.keys and not before_closing.rstrip().endswith(","):
            prefix += ","
        if before_closing and not before_closing.endswith(("\n", "\r")):
            prefix += newline
        entry = (
            f"{prefix}    ({resolution[0]}, {resolution[1]}): "
            f"{replacement},{newline}"
        )
        edits.append((closing_brace, closing_brace, entry))

    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    ast.parse(source)
    return source


def write_model_paths(
    model_paths: Path,
    source: str,
    make_backup: bool,
) -> Path | None:
    backup_path = None
    if make_backup:
        backup_path = model_paths.with_suffix(model_paths.suffix + ".bak")
        shutil.copy2(model_paths, backup_path)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=model_paths.name + ".",
            suffix=".tmp",
            dir=model_paths.parent,
            delete=False,
        ) as temporary:
            temporary.write(source.encode("utf-8"))
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, model_paths)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Интерактивно добавить области нового разрешения в model_paths.py"
        )
    )
    parser.add_argument("source", type=Path, help="Путь к изображению или видео")
    parser.add_argument(
        "--frame-index",
        type=int,
        help="Использовать указанный кадр видео без окна выбора кадра",
    )
    parser.add_argument(
        "--model-paths",
        type=Path,
        default=DEFAULT_MODEL_PATHS,
        help=f"Изменяемый файл (по умолчанию {DEFAULT_MODEL_PATHS})",
    )
    parser.add_argument("--preview-width", type=int, default=900)
    parser.add_argument("--preview-height", type=int, default=900)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Не спрашивать подтверждение при замене существующего разрешения",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Не создавать model_paths.py.bak перед записью",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source.expanduser().resolve()
    model_paths = args.model_paths.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {source_path}")
    if not model_paths.is_file():
        raise FileNotFoundError(f"model_paths.py не найден: {model_paths}")
    if args.preview_width < 100 or args.preview_height < 100:
        raise ValueError("Размер предпросмотра должен быть не меньше 100 пикселей")

    frame, selected_frame_index = load_source_frame(
        source_path,
        args.frame_index,
        args.preview_width,
        args.preview_height,
    )
    height, width = frame.shape[:2]
    resolution = (width, height)
    frame_description = (
        "изображение"
        if selected_frame_index is None
        else f"кадр {selected_frame_index}"
    )
    print(f"Источник: {frame_description}, разрешение {width}x{height}")

    while True:
        regions = select_regions(frame, args.preview_width, args.preview_height)
        if review_regions(
            frame,
            regions,
            args.preview_width,
            args.preview_height,
        ) == "save":
            break

    existing = existing_resolution_fields(model_paths, resolution)
    if existing and not args.force:
        print(
            "Разрешение уже присутствует в: " + ", ".join(existing)
        )
        answer = input("Заменить эти значения? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "д", "да"}:
            print("Изменения не сохранены")
            return 0

    original = model_paths.read_text(encoding="utf-8")
    updated = update_model_paths_source(original, resolution, regions)
    backup_path = write_model_paths(
        model_paths,
        updated,
        make_backup=not args.no_backup,
    )
    print(f"Обновлён файл: {model_paths}")
    if backup_path is not None:
        print(f"Резервная копия: {backup_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserCancelled:
        cv2.destroyAllWindows()
        print("Отменено пользователем")
        raise SystemExit(0)
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        cv2.destroyAllWindows()
        print(f"Ошибка: {error}")
        raise SystemExit(1)
