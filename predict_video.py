import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from model_paths import (
    BATTLEFIELDS,
    CLASSIFICATION_MODEL_PATH,
    DETECTION_ENGINE_PATH,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# from efficient_net_predict import load_trained_model, predict_single_image
# from image2cards import get_image_cards_format
from predict_classification import classify_crop

# YOLO inference settings are kept in sync with process_video.py.
MODEL_PATH = DETECTION_ENGINE_PATH
INPUT_VIDEO = SCRIPT_DIR / "screenshots/input_omydays.mp4"
OUTPUT_VIDEO = SCRIPT_DIR / "screenshots/output_tracked.mp4"

IMGSZ = 1280
CONF = 0.20
IOU = 0.50
MAX_DET = 500
DEVICE = 0
QUANTIZE = 16  # FP16; use None for FP32


model = YOLO(str(MODEL_PATH))
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

crop_x1, crop_y1, crop_x2, crop_y2 = BATTLEFIELDS[video_size]
if not (0 <= crop_x1 < crop_x2 <= width):
    raise ValueError(f"Invalid horizontal crop: {(crop_x1, crop_x2)} for width {width}")
if not (0 <= crop_y1 < crop_y2 <= height):
    raise ValueError(f"Invalid vertical crop: {(crop_y1, crop_y2)} for height {height}")

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

    # Собираем данные о кадре
    frame_data = {"frame_number": frame_count, "num_objects": 0, "objects": []}

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        frame_data["num_objects"] = len(track_ids)
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

    # Put the tracked and annotated battlefield back into the original frame.
    output_frame = frame.copy()
    output_frame[crop_y1:crop_y2, crop_x1:crop_x2] = battlefield

    # predict cards in hand

    # for idx_card in range(4):
    #     height_, width_ = frame_cards.shape[:2]
    #     card = frame_cards[
    #         height_ // 10 : height_ // 10 * 9,
    #         width_ // 4 * idx_card : width_ // 4 * (idx_card + 1),
    #     ]
    #     predicted_class_card = predict_single_image(
    #         classification_cards,
    #         card,
    #         classification_classes_cards,
    #         device,
    #         verbose=False,
    #     )
    #     print(f"card {idx_card} {predicted_class_card}", end= ' ')
    # print()

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
    frame_height, frame_width = output_frame.shape[:2]
    new_width = frame_width // 2
    new_height = frame_height // 2
    resized_frame = cv2.resize(output_frame, (new_width, new_height))
    cv2.imshow("Tracking", resized_frame)
    # cv2.imshow("Cards", frame_cards)

    out.write(output_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
out.release()
