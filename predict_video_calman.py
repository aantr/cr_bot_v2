import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from model_paths import (
    BATTLEFIELDS,
    CARDS,
    CLASSIFICATION_CARDS_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    DETECTION_ENGINE_PATH,
    ELIXIR_DETECTION_ENGINE_PATH,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# from efficient_net_predict import load_trained_model, predict_single_image
# from image2cards import get_image_cards_format
from predict_classification import classify_crop, get_top_predictions
from field import FIELD

# YOLO inference settings are kept in sync with process_video.py.
MODEL_PATH = DETECTION_ENGINE_PATH
ELIXIR_MODEL_PATH = ELIXIR_DETECTION_ENGINE_PATH
CARDS_MODEL_PATH = CLASSIFICATION_CARDS_MODEL_PATH
INPUT_VIDEO = SCRIPT_DIR / "screenshots/input_omydays.mp4"
OUTPUT_VIDEO = SCRIPT_DIR / "screenshots/output_tracked_calman.mp4"

IMGSZ = 1280
CONF = 0.20
IOU = 0.50
MAX_DET = 500
DEVICE = 0
QUANTIZE = 16  # FP16; use None for FP32


if not Path(MODEL_PATH).is_file():
    raise FileNotFoundError(f"Bars TensorRT engine not found: {MODEL_PATH}")
if not Path(ELIXIR_MODEL_PATH).is_file():
    raise FileNotFoundError(
        f"Elixir TensorRT engine not found: {ELIXIR_MODEL_PATH}. "
        "Run train_elixir\\export_tensor_rt.py after training finishes."
    )
if not Path(CARDS_MODEL_PATH).is_file():
    raise FileNotFoundError(f"Cards classification model not found: {CARDS_MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
elixir_model = YOLO(str(ELIXIR_MODEL_PATH))
classification_cards_model = YOLO(str(CARDS_MODEL_PATH))
cap = cv2.VideoCapture(str(INPUT_VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_size = (width, height)
if video_size not in BATTLEFIELDS:
    raise ValueError(
        f"Unsupported video resolution: {width}x{height}. "
        f"Add ({width}, {height}): (x1, y1, x2, y2) to "
        "BATTLEFIELDS in model_paths.py."
    )
if video_size not in CARDS:
    raise ValueError(
        f"Cards crop is not configured for video resolution {width}x{height}. "
        "Add this resolution to CARDS in model_paths.py."
    )

crop_x1, crop_y1, crop_x2, crop_y2 = BATTLEFIELDS[video_size]
if not (0 <= crop_x1 < crop_x2 <= width):
    raise ValueError(f"Invalid horizontal crop: {(crop_x1, crop_x2)} for width {width}")
if not (0 <= crop_y1 < crop_y2 <= height):
    raise ValueError(f"Invalid vertical crop: {(crop_y1, crop_y2)} for height {height}")

cards_x1, cards_y1, cards_x2, cards_y2 = CARDS[video_size]
if not (0 <= cards_x1 < cards_x2 <= width):
    raise ValueError(
        f"Invalid cards horizontal crop: {(cards_x1, cards_x2)} for width {width}"
    )
if not (0 <= cards_y1 < cards_y2 <= height):
    raise ValueError(
        f"Invalid cards vertical crop: {(cards_y1, cards_y2)} for height {height}"
    )
if cards_x2 - cards_x1 < 4:
    raise ValueError("Cards crop must be at least 4 pixels wide")

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError(f"Cannot create output video: {OUTPUT_VIDEO}")

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
# classification_model, classification_classes = load_trained_model(
#     CLASSIFICATION_MODEL_PATH, device
# )
# classification_cards, classification_classes_cards = load_trained_model(
#     CLASSIFICATION_CARDS_PATH, device
# )
classification_model = YOLO(
    CLASSIFICATION_MODEL_PATH
)

# Предсказание
# predicted_class = predict_single_image(classification_model, image_path, classification_classes, device)
# predictions = classify_crop(
#     classification_model,
#     blue_rect,
#     imgsz=224,
#     device="0",
#     top_k=3,
#     quantize=16,
# )

# class_name, confidence = predictions[0]
# cls_text = f"{class_name} {confidence:.2f}"

# end preprocess class. model

# Для сбора статистики по кадрам
frame_stats = []
object_history = defaultdict(list)  # история позиций объектов

bars_place = defaultdict(list)
bar_for_level = defaultdict(int)
LEN_POSES = 5
SIZE_OF_RECT = 128
SIZE_RESCTRICTIONS = (10, 10), (50, 60)
KALMAN_MAX_MISSED_FRAMES = 60
CARD_COUNT = 4
CARD_IMGSZ = 224
FIELD_ROWS = len(FIELD)
FIELD_COLUMNS = len(FIELD[0]) if FIELD else 0
FIELD_CELL_SIZE = 25
FIELD_WINDOW_NAME = "Arena 32x18"

if FIELD_ROWS != 32 or FIELD_COLUMNS != 18:
    raise ValueError(
        f"field.py must contain a 32x18 field, got {FIELD_ROWS}x{FIELD_COLUMNS}"
    )
if any(len(row) != FIELD_COLUMNS for row in FIELD):
    raise ValueError("All rows in field.py must have the same length")


def build_field_background() -> np.ndarray:
    """Render the 32x18 field with square cells."""
    canvas = np.empty(
        (FIELD_ROWS * FIELD_CELL_SIZE, FIELD_COLUMNS * FIELD_CELL_SIZE, 3),
        dtype=np.uint8,
    )
    for row_index, row in enumerate(FIELD):
        for column_index, cell in enumerate(row):
            x1 = column_index * FIELD_CELL_SIZE
            y1 = row_index * FIELD_CELL_SIZE
            x2 = x1 + FIELD_CELL_SIZE
            y2 = y1 + FIELD_CELL_SIZE
            color = (70, 70, 70) if cell == "#" else (45, 105, 45)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (25, 25, 25), 1)
    return canvas


def track_color(object_type: str, track_id: int | None) -> tuple[int, int, int]:
    """Return a stable, visually distinct BGR color for a tracked object."""
    if track_id is None:
        return 190, 190, 190
    namespace = 0 if object_type == "blue_rect" else 1
    hue = (int(track_id) * 47 + namespace * 83) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_objects_on_field(
    background: np.ndarray,
    objects: list[tuple[str, int | None, float, float]],
    battlefield_shape: tuple[int, ...],
) -> np.ndarray:
    """Highlight free field cells containing tracked object centers."""
    canvas = background.copy()
    battlefield_height, battlefield_width = battlefield_shape[:2]
    if battlefield_width <= 0 or battlefield_height <= 0:
        return canvas

    for object_type, track_id, center_x, center_y in objects:
        if not (
            0 <= center_x < battlefield_width
            and 0 <= center_y < battlefield_height
        ):
            continue

        column_index = int(center_x / battlefield_width * FIELD_COLUMNS)
        row_index = int(center_y / battlefield_height * FIELD_ROWS)
        if not (
            0 <= column_index < FIELD_COLUMNS
            and 0 <= row_index < FIELD_ROWS
        ):
            continue
        if FIELD[row_index][column_index] == "#":
            continue

        color = track_color(object_type, track_id)
        cell_x1 = column_index * FIELD_CELL_SIZE
        cell_y1 = row_index * FIELD_CELL_SIZE
        cell_x2 = cell_x1 + FIELD_CELL_SIZE - 1
        cell_y2 = cell_y1 + FIELD_CELL_SIZE - 1
        cv2.rectangle(canvas, (cell_x1, cell_y1), (cell_x2, cell_y2), color, -1)
        cv2.rectangle(
            canvas,
            (cell_x1, cell_y1),
            (cell_x2, cell_y2),
            (25, 25, 25),
            1,
        )
    return canvas


def split_cards_from_frame(
    frame: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split the configured cards crop into four left-to-right slots."""
    x1, y1, x2, y2 = crop_bounds
    cards_crop = frame[y1:y2, x1:x2]
    if cards_crop.size == 0:
        raise ValueError(f"Cards crop is empty: {crop_bounds}")

    crop_width = cards_crop.shape[1]
    edges = [index * crop_width // CARD_COUNT for index in range(CARD_COUNT + 1)]
    card_images: list[np.ndarray] = []
    card_slots: list[tuple[int, int]] = []
    for index in range(CARD_COUNT):
        local_x1 = edges[index]
        local_x2 = edges[index + 1]
        card_images.append(cards_crop[:, local_x1:local_x2])
        card_slots.append((x1 + local_x1, x1 + local_x2))
    return card_images, card_slots


def classify_cards(
    cards_model: YOLO,
    card_images: list[np.ndarray],
) -> list[tuple[str, float]]:
    """Classify all four card slots in one model call."""
    results = cards_model.predict(
        source=card_images,
        imgsz=CARD_IMGSZ,
        device=DEVICE,
        quantize=QUANTIZE,
        verbose=False,
    )
    if len(results) != CARD_COUNT:
        raise RuntimeError(
            f"Cards model returned {len(results)} results instead of {CARD_COUNT}"
        )
    return [get_top_predictions(result, top_k=1)[0] for result in results]


def draw_card_predictions(
    frame: np.ndarray,
    predictions: list[tuple[str, float]],
    card_slots: list[tuple[int, int]],
    cards_top: int,
) -> None:
    """Draw card numbers and classification results above their slots."""
    label_y = max(25, cards_top - 10)
    for card_number, ((class_name, confidence), (slot_x1, slot_x2)) in enumerate(
        zip(predictions, card_slots),
        start=1,
    ):
        label = f"{card_number}: {class_name} {confidence:.2f}"
        available_width = max(1, slot_x2 - slot_x1 - 8)
        font_scale = 0.65
        while font_scale > 0.30:
            (text_width, _), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                2,
            )
            if text_width <= available_width:
                break
            font_scale -= 0.05

        (text_width, _), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            2,
        )
        text_x = slot_x1 + max(4, (slot_x2 - slot_x1 - text_width) // 2)
        cv2.putText(
            frame,
            label,
            (text_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (text_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


class BoundingBoxKalmanFilter:
    """Constant-velocity Kalman filter for cx, cy, width and height."""

    def __init__(self, box: np.ndarray) -> None:
        self.filter = cv2.KalmanFilter(8, 4)
        self.filter.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.filter.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.filter.measurementMatrix[:4, :4] = np.eye(4, dtype=np.float32)
        self.filter.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.10
        self.filter.errorCovPost = np.eye(8, dtype=np.float32)

        cx, cy, width, height = self._xyxy_to_measurement(box).reshape(4)
        state = np.zeros((8, 1), dtype=np.float32)
        state[:4, 0] = (cx, cy, width, height)
        self.filter.statePost = state
        self.filter.statePre = state.copy()

    @staticmethod
    def _xyxy_to_measurement(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(float, box)
        return np.array(
            [
                [(x1 + x2) / 2],
                [(y1 + y2) / 2],
                [max(1.0, x2 - x1)],
                [max(1.0, y2 - y1)],
            ],
            dtype=np.float32,
        )

    def update(
        self,
        box: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        self.filter.predict()
        corrected = self.filter.correct(self._xyxy_to_measurement(box)).reshape(8)
        cx, cy, width, height = map(float, corrected[:4])
        width = max(1.0, width)
        height = max(1.0, height)
        return np.array(
            [
                np.clip(cx - width / 2, 0, image_width),
                np.clip(cy - height / 2, 0, image_height),
                np.clip(cx + width / 2, 0, image_width),
                np.clip(cy + height / 2, 0, image_height),
            ],
            dtype=np.float32,
        )


def smooth_result_boxes(
    result,
    filters: dict[int, BoundingBoxKalmanFilter],
    last_seen: dict[int, int],
    frame_index: int,
    image_shape: tuple[int, ...],
) -> None:
    """Replace tracked result boxes with their Kalman-smoothed coordinates."""
    if (
        result.boxes is not None
        and len(result.boxes) > 0
        and result.boxes.id is not None
    ):
        raw_boxes = result.boxes.xyxy.detach().cpu().numpy()
        track_ids = result.boxes.id.int().detach().cpu().tolist()
        image_height, image_width = image_shape[:2]
        smoothed_boxes: list[np.ndarray] = []
        for box, track_id in zip(raw_boxes, track_ids):
            track_id = int(track_id)
            if track_id not in filters:
                filters[track_id] = BoundingBoxKalmanFilter(box)
            smoothed_boxes.append(
                filters[track_id].update(box, image_width, image_height)
            )
            last_seen[track_id] = frame_index

        smoothed_coordinates = torch.as_tensor(
            np.stack(smoothed_boxes),
            dtype=result.boxes.data.dtype,
            device=result.boxes.data.device,
        )
        boxes_data = result.boxes.data.detach().clone()
        boxes_data[:, :4] = smoothed_coordinates
        result.boxes = type(result.boxes)(boxes_data, result.boxes.orig_shape)

    stale_ids = [
        track_id
        for track_id, seen_frame in last_seen.items()
        if frame_index - seen_frame > KALMAN_MAX_MISSED_FRAMES
    ]
    for track_id in stale_ids:
        filters.pop(track_id, None)
        last_seen.pop(track_id, None)


def extract_blue_rect(
    image: np.ndarray,
    level_box: tuple[float, float, float, float],
    bar_width: float,
) -> np.ndarray | None:
    """Extract a fixed-size clean crop, padding areas outside the frame."""
    x1, _, x2, y2 = level_box
    center_x = (x1 + x2 + bar_width) / 2
    left = round(center_x - SIZE_OF_RECT / 2)
    top = round(y2)
    right = left + SIZE_OF_RECT
    bottom = top + SIZE_OF_RECT

    image_height, image_width = image.shape[:2]
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image_width, right)
    source_bottom = min(image_height, bottom)
    if source_right <= source_left or source_bottom <= source_top:
        return None

    crop = np.full(
        (SIZE_OF_RECT, SIZE_OF_RECT, 3),
        114,
        dtype=np.uint8,
    )
    target_left = source_left - left
    target_top = source_top - top
    crop[
        target_top : target_top + (source_bottom - source_top),
        target_left : target_left + (source_right - source_left),
    ] = image[source_top:source_bottom, source_left:source_right]
    return crop


bar_kalman_filters: dict[int, BoundingBoxKalmanFilter] = {}
bar_kalman_last_seen: dict[int, int] = {}
elixir_kalman_filters: dict[int, BoundingBoxKalmanFilter] = {}
elixir_kalman_last_seen: dict[int, int] = {}
field_background = build_field_background()
cv2.namedWindow(FIELD_WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(
    FIELD_WINDOW_NAME,
    field_background.shape[1],
    field_background.shape[0],
)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame_cards = get_image_cards_format(frame)
    battlefield = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    # Inference settings match process_video.py; track() is retained so IDs persist.
    results = model.track(
        source=battlefield,
        persist=True,  # maintain track IDs across frames
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        device=DEVICE,
        quantize=QUANTIZE,
        tracker="bytetrack.yaml",  # tracking configuration
        project="detection_results",  # Папка для сохранения
        name="video_tracking",
        verbose=False,
    )

    # Elixir uses tracking only to obtain stable IDs for Kalman smoothing. Its
    # detections never enter the unit classification branch.
    elixir_result = elixir_model.track(
        source=battlefield,
        persist=True,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        device=DEVICE,
        quantize=QUANTIZE,
        tracker="bytetrack.yaml",
        verbose=False,
    )[0]

    smooth_result_boxes(
        results[0],
        bar_kalman_filters,
        bar_kalman_last_seen,
        frame_count,
        battlefield.shape,
    )
    smooth_result_boxes(
        elixir_result,
        elixir_kalman_filters,
        elixir_kalman_last_seen,
        frame_count,
        battlefield.shape,
    )

    # Собираем данные о кадре
    frame_data = {"frame_number": frame_count, "num_objects": 0, "objects": []}
    field_objects: list[tuple[str, int | None, float, float]] = []
    bar_detection_count = 0
    elixir_detection_count = (
        0 if elixir_result.boxes is None else len(elixir_result.boxes)
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        bar_detection_count = len(track_ids)
        frame_data["num_objects"] = bar_detection_count
        cls_text = ""

        for i, (box, track_id, conf, class_id) in enumerate(
            zip(boxes, track_ids, confs, class_ids)
        ):

            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Ищем совпадающие бары и левелы
            if (
                class_id == 1
                and SIZE_RESCTRICTIONS[0][0] <= x2 - x1 <= SIZE_RESCTRICTIONS[1][0]
                and SIZE_RESCTRICTIONS[0][1] <= y2 - y1 <= SIZE_RESCTRICTIONS[1][1]
            ):

                bars_place[track_id].append((x1, y1, x2, y2))
                while len(bars_place[track_id]) > LEN_POSES:
                    bars_place[track_id].pop(0)

                if bar_for_level[track_id]:
                    pass
                rect = (int(x1), int(y1)), (
                    int(x2 + bar_for_level[track_id]),
                    int(y2),
                )

                # Extract from the clean frame before drawing debug rectangles.
                blue_rect = extract_blue_rect(
                    battlefield,
                    (x1, y1, x2, y2),
                    bar_for_level[track_id],
                )

                cv2.rectangle(battlefield, *rect, (0, 0, 255), 3)

                blue_left = int(
                    (rect[0][0] + rect[1][0]) / 2 - SIZE_OF_RECT / 2
                )
                field_objects.append(
                    (
                        "blue_rect",
                        int(track_id),
                        blue_left + SIZE_OF_RECT / 2,
                        rect[1][1] + SIZE_OF_RECT / 2,
                    )
                )
                cv2.rectangle(
                    battlefield,
                    (blue_left, rect[1][1]),
                    (
                        blue_left + SIZE_OF_RECT,
                        rect[1][1] + SIZE_OF_RECT,
                    ),
                    (255, 0, 0),
                    2,
                )

                # predicted_class = predict_single_image(
                #     classification_model,
                #     blue_rect,
                #     classification_classes,
                #     device,
                #     verbose=False,
                # )
                if blue_rect is not None:
                    predictions = classify_crop(
                        classification_model,
                        blue_rect,
                        imgsz=224,
                        device="0",
                        top_k=3,
                        quantize=16,
                    )
                    class_name, confidence = predictions[0]
                    cls_text = f"{class_name} {confidence:.2f}"
                else:
                    cls_text = "None"
            else:
                cls_text = "None"
            if class_id == 0:
                x_left = x1
                y_left = center_y
                for key, level_bar in bars_place.items():
                    count_good_pos = 0
                    sum_len = 0
                    for pos in level_bar:
                        x_level1, y_level1, x_level2, y_level2 = pos

                        if (
                            y_level1 <= y_left <= y_level2
                            and (x_level1 + x_level2) / 2
                            <= x_left
                            <= (x_level1 + x_level2) / 2 + x_level2 - x_level1
                        ):

                            count_good_pos += 1
                            sum_len += x2 - x1

                    if len(level_bar) >= LEN_POSES and count_good_pos > LEN_POSES / 2:
                        bar_for_level[key] = sum_len / count_good_pos

            # Данные объекта
            obj_data = {
                "track_id": track_id,
                "class": model.names[class_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
            }
            frame_data["objects"].append(obj_data)

            # Сохраняем в историю для анализа траекторий
            object_history[track_id].append(
                {
                    "frame": frame_count,
                    "center": (center_x, center_y),
                    "bbox": (x1, y1, x2, y2),
                }
            )

            # Визуализация с дополнительной информацией
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{track_id} {cls_text}"
            cv2.putText(
                battlefield,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )

    # Elixir detections never pass through classify_crop().
    if elixir_result.boxes is not None:
        for box in elixir_result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            track_id = int(box.id[0]) if box.id is not None else None
            field_objects.append(
                (
                    "elixir",
                    track_id,
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                )
            )
            frame_data["objects"].append(
                {
                    "track_id": track_id,
                    "class": elixir_result.names[class_id],
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                }
            )
        battlefield = elixir_result.plot(img=battlefield)

    frame_data["num_objects"] += elixir_detection_count

    # Put the tracked and annotated battlefield back into the original frame.
    output_frame = frame.copy()
    output_frame[crop_y1:crop_y2, crop_x1:crop_x2] = battlefield

    card_images, card_slots = split_cards_from_frame(
        frame,
        (cards_x1, cards_y1, cards_x2, cards_y2),
    )
    card_predictions = classify_cards(classification_cards_model, card_images)
    draw_card_predictions(
        output_frame,
        card_predictions,
        card_slots,
        cards_y1,
    )

    # Показываем номер кадра
    cv2.putText(
        output_frame,
        f"Frame: {frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output_frame,
        f"Bars: {bar_detection_count} | Elixir: {elixir_detection_count}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    frame_height, frame_width = output_frame.shape[:2]
    new_width = frame_width // 2
    new_height = frame_height // 2
    resized_frame = cv2.resize(output_frame, (new_width, new_height))
    field_frame = draw_objects_on_field(
        field_background,
        field_objects,
        battlefield.shape,
    )
    cv2.imshow("Tracking", resized_frame)
    cv2.imshow(FIELD_WINDOW_NAME, field_frame)
    # cv2.imshow("Cards", frame_cards)

    out.write(output_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
out.release()
