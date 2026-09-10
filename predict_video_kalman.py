import json
import logging
import math
import os
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_cublas_dir = Path(sys.prefix) / "Lib/site-packages/nvidia/cublas/bin"
if sys.platform == "win32" and _cublas_dir.is_dir():
    _cublas_dll_handle = os.add_dll_directory(str(_cublas_dir))
    os.environ["PATH"] = str(_cublas_dir) + os.pathsep + os.environ.get("PATH", "")

import torch
from ultralytics import YOLO

from model_paths import (
    BATTLEFIELDS,
    CARDS,
    CLASSIFICATION_CARDS_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    DETECTION_ENGINE_PATH,
    ELIXIR_BAR,
    ELIXIR_DETECTION_ENGINE_PATH,
    TOWER_HP_1,
    TOWER_HP_2,
    TOWER_HP_ENEMY_1,
    TOWER_HP_ENEMY_2,
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
INPUT_VIDEO = SCRIPT_DIR / "screenshots/last_20_percent.mp4"
OUTPUT_VIDEO = SCRIPT_DIR / "screenshots/output_tracked_kalman.mp4"

IMGSZ = 1280
CONF = 0.30
IOU = 0.50
MAX_DET = 500
DEVICE = 0
QUANTIZE = 16  # FP16; use None for FP32
FPS_PROCESS = 30  # Frames per video second to process; None uses source FPS.

# Current elixir is read from a narrow horizontal strip through ELIXIR_BAR.
# Brightness is the V channel in HSV, in the range 0..255. Values between the
# thresholds are treated as a soft transition and compared with their midpoint.
ELIXIR_DARK_BRIGHTNESS_MAX = 135.0
ELIXIR_LIGHT_BRIGHTNESS_MIN = 150.0
ELIXIR_SEARCH_STRIP_HEIGHT = 9
ELIXIR_SEARCH_SAMPLE_WIDTH = 11
ELIXIR_MAX_VALUE = 10.0

# Tower HP OCR (video-time rate, capped by FPS_PROCESS).
TOWER_HP_ENABLED = True
TOWER_HP_FPS = 2.0
# Re-evaluate every configured crop and select the most confident one. Between
# these scans OCR runs only on the last selected crop for each tower.
TOWER_HP_CROP_SELECTION_INTERVAL_MS = 2000
TOWER_HP_MIN_CONFIDENCE = 0.70  # PaddleOCR recognition score, 0..1.
TOWER_HP_CONFIRM_READINGS = 2
# Safety bounds, not actual tower starting
#  HP. Set for your battle/levels.
TOWER_HP_MAX_VALUES = {"ally_1": 10000, "ally_2": 10000,
                       "enemy_1": 10000, "enemy_2": 10000}
TOWER_HP_SCALE = 1.0  # Keep original BGR crops; PaddleOCR resizes internally.
TOWER_HP_MODEL_NAME = "en_PP-OCRv4_mobile_rec"
TOWER_HP_MODEL_DIR = None  # Optional local PaddleOCR inference model directory.
TOWER_HP_DEVICE = "cpu"  # Paddle 3.0.0rc1/cu123 gives invalid output on RTX 5080.
TOWER_HP_CPU_THREADS = 4
TOWER_HP_ENABLE_MKLDNN = False  # Paddle 3.0.0rc1 fails on this model with MKL-DNN.
TOWER_HP_STARTUP_TIMEOUT = 180.0  # Includes the first model download.
TOWER_HP_REQUEST_TIMEOUT = 15.0  # Kill a stuck worker; never reuse a late result.
TOWER_HP_LOG_PATH = OUTPUT_VIDEO.with_suffix(".tower_hp.jsonl")  # Appended per run.
show_tower_hp = False

if TOWER_HP_ENABLED:
    if not math.isfinite(TOWER_HP_FPS) or TOWER_HP_FPS <= 0:
        raise ValueError("TOWER_HP_FPS must be finite and positive")
    if (
        not math.isfinite(TOWER_HP_CROP_SELECTION_INTERVAL_MS)
        or TOWER_HP_CROP_SELECTION_INTERVAL_MS <= 0
    ):
        raise ValueError(
            "TOWER_HP_CROP_SELECTION_INTERVAL_MS must be finite and positive"
        )
    if not 0 <= TOWER_HP_MIN_CONFIDENCE <= 1:
        raise ValueError("TOWER_HP_MIN_CONFIDENCE must be between 0 and 1")
    if not isinstance(TOWER_HP_CONFIRM_READINGS, int) or TOWER_HP_CONFIRM_READINGS < 2:
        raise ValueError("TOWER_HP_CONFIRM_READINGS must be an integer >= 2")
    if not math.isfinite(TOWER_HP_SCALE) or TOWER_HP_SCALE <= 0:
        raise ValueError("TOWER_HP_SCALE must be finite and positive")
    if not isinstance(TOWER_HP_CPU_THREADS, int) or TOWER_HP_CPU_THREADS < 1:
        raise ValueError("TOWER_HP_CPU_THREADS must be a positive integer")

# Field cell color: compare R * RED_WEIGHT with B * BLUE_WEIGHT.
# Increase a channel's weight to select its color more often.
FIELD_COLOR_RED_WEIGHT = 1.0
FIELD_COLOR_BLUE_WEIGHT = 1.5

if any(
    not math.isfinite(weight) or weight <= 0
    for weight in (FIELD_COLOR_RED_WEIGHT, FIELD_COLOR_BLUE_WEIGHT)
):
    raise ValueError("Field color weights must be finite positive numbers")

if not (
    0 <= ELIXIR_DARK_BRIGHTNESS_MAX
    < ELIXIR_LIGHT_BRIGHTNESS_MIN
    <= 255
):
    raise ValueError(
        "Elixir brightness thresholds must satisfy "
        "0 <= dark < light <= 255"
    )
if (
    not isinstance(ELIXIR_SEARCH_STRIP_HEIGHT, int)
    or ELIXIR_SEARCH_STRIP_HEIGHT < 1
    or ELIXIR_SEARCH_STRIP_HEIGHT % 2 == 0
):
    raise ValueError("ELIXIR_SEARCH_STRIP_HEIGHT must be a positive odd integer")
if (
    not isinstance(ELIXIR_SEARCH_SAMPLE_WIDTH, int)
    or ELIXIR_SEARCH_SAMPLE_WIDTH < 1
    or ELIXIR_SEARCH_SAMPLE_WIDTH % 2 == 0
):
    raise ValueError("ELIXIR_SEARCH_SAMPLE_WIDTH must be a positive odd integer")

# Debug window with the cropped and annotated battlefield.
show_battlefield = False
BATTLEFIELD_DEBUG_WINDOW = "Battlefield debug"
BATTLEFIELD_DEBUG_WIDTH = 500

# Debug window with the four card slots and classification results.
show_cards = False
CARDS_DEBUG_WINDOW = "Cards debug"

# The Tracking preview used to be 50% of the source frame. A 0.25 scale makes
# the window 50% smaller in both dimensions than that previous preview.
TRACKING_WINDOW_SCALE = 0.375


LEN_POSES = 5
SIZE_OF_RECT = 128
SIZE_RESCTRICTIONS = (10, 10), (50, 60)
KALMAN_MAX_MISSED_FRAMES = 60
# Keep a unit visible while its detector track is briefly absent. The value is
# measured in processed frames, so it behaves the same for FPS_PROCESS=20/30/60.
TRACK_HOLD_PROCESSED_FRAMES = 5
TRACK_CONFIDENCE_DECAY = 0.85
TRACK_MIN_PREDICTED_CONFIDENCE = 0.15
# A new unit is classified for a few frames, then its class is cached by track_id.
# For strict one-shot mode, set CONFIRM_SAMPLES=1 and LOCK_MIN_CONFIDENCE=0.
UNIT_CLASSIFICATION_INTERVAL_MS = 200
UNIT_CLASSIFICATION_CONFIRM_SAMPLES = 5
UNIT_CLASSIFICATION_MAX_SAMPLES = 10
UNIT_CLASSIFICATION_LOCK_MIN_CONFIDENCE = 0.70
CARD_COUNT = 4
CARD_IMGSZ = 224
EMPTY_CARD_CLASS = "empty"
CARD_HISTORY_MS = 2000
CARD_EMPTY_LEAD_MS = 900  # было 150
CARD_PREVIOUS_LOOKBACK_MS = 500  # было 800
CARD_PREVIOUS_SAMPLE_MS = 100  # было 300
CARD_EMPTY_CONFIRM_MS = 200
CARD_MIN_PREVIOUS_FRAMES = 2  # было 3
ELIXIR_EVENT_DWELL_MS = 300
ELIXIR_EVENT_MAX_GAP_MS = 200
ELIXIR_EVENT_COOLDOWN_MS = 800
ELIXIR_EVENT_REGION_RADIUS_CELLS = 1.5
ELIXIR_EVENT_MIN_DETECTIONS = 3
EVENT_LOG_PATH = SCRIPT_DIR / "battle_events.log"
CARD_EVENT_MARKER_MS = 1000
FIELD_ROWS = len(FIELD)
FIELD_COLUMNS = len(FIELD[0]) if FIELD else 0
FIELD_CELL_SIZE = 25
FIELD_WINDOW_NAME = "Arena 32x18"

if (
    not isinstance(TRACK_HOLD_PROCESSED_FRAMES, int)
    or TRACK_HOLD_PROCESSED_FRAMES < 1
):
    raise ValueError("TRACK_HOLD_PROCESSED_FRAMES must be a positive integer")
if not 0 < TRACK_CONFIDENCE_DECAY <= 1:
    raise ValueError("TRACK_CONFIDENCE_DECAY must be in the range (0, 1]")
if not 0 <= TRACK_MIN_PREDICTED_CONFIDENCE <= 1:
    raise ValueError("TRACK_MIN_PREDICTED_CONFIDENCE must be in the range [0, 1]")
if (
    not isinstance(UNIT_CLASSIFICATION_INTERVAL_MS, (int, float))
    or not math.isfinite(UNIT_CLASSIFICATION_INTERVAL_MS)
    or UNIT_CLASSIFICATION_INTERVAL_MS < 0
):
    raise ValueError("UNIT_CLASSIFICATION_INTERVAL_MS must be finite and non-negative")
if (
    not isinstance(UNIT_CLASSIFICATION_CONFIRM_SAMPLES, int)
    or UNIT_CLASSIFICATION_CONFIRM_SAMPLES < 1
):
    raise ValueError("UNIT_CLASSIFICATION_CONFIRM_SAMPLES must be a positive integer")
if (
    not isinstance(UNIT_CLASSIFICATION_MAX_SAMPLES, int)
    or UNIT_CLASSIFICATION_MAX_SAMPLES < UNIT_CLASSIFICATION_CONFIRM_SAMPLES
):
    raise ValueError(
        "UNIT_CLASSIFICATION_MAX_SAMPLES must be >= "
        "UNIT_CLASSIFICATION_CONFIRM_SAMPLES"
    )
if not 0 <= UNIT_CLASSIFICATION_LOCK_MIN_CONFIDENCE <= 1:
    raise ValueError("UNIT_CLASSIFICATION_LOCK_MIN_CONFIDENCE must be in [0, 1]")
if FIELD_ROWS != 32 or FIELD_COLUMNS != 18:
    raise ValueError(
        f"field.py must contain a 32x18 field, got {FIELD_ROWS}x{FIELD_COLUMNS}"
    )
if any(len(row) != FIELD_COLUMNS for row in FIELD):
    raise ValueError("All rows in field.py must have the same length")


def create_tower_hp_recognizer():
    """Keep Paddle/cuDNN in a persistent process isolated from PyTorch/YOLO."""
    from tower_hp_process import TowerHPProcess
    return TowerHPProcess(
        model_name=TOWER_HP_MODEL_NAME, model_dir=TOWER_HP_MODEL_DIR,
        device=TOWER_HP_DEVICE, cpu_threads=TOWER_HP_CPU_THREADS,
        enable_mkldnn=TOWER_HP_ENABLE_MKLDNN,
        startup_timeout=TOWER_HP_STARTUP_TIMEOUT, request_timeout=TOWER_HP_REQUEST_TIMEOUT,
    )


def prepare_tower_hp_crop(crop: np.ndarray) -> np.ndarray:
    """Keep clean BGR pixels; thresholding/inversion loses game-font detail."""
    if TOWER_HP_SCALE == 1:
        return crop.copy()
    return cv2.resize(crop, None, fx=TOWER_HP_SCALE, fy=TOWER_HP_SCALE,
                      interpolation=cv2.INTER_CUBIC)


def parse_tower_hp_data(data: dict, max_hp: int) -> dict:
    """Validate PaddleOCR text without stripping arbitrary non-digit characters."""
    
    # Извлеките текст и уверенность из нового формата PaddleOCR 3.x
    if isinstance(data, dict) and "rec_text" in data and "rec_score" in data:
        raw_text = str(data["rec_text"]).strip()
        confidence = float(data["rec_score"])
    elif isinstance(data, dict):
        # Новый формат: rec_texts и rec_scores - это списки
        rec_texts = data.get('rec_texts', [])
        rec_scores = data.get('rec_scores', [])
        
        if len(rec_texts) > 1 or len(rec_scores) > 1:
            raise ValueError("Expected a single HP number, not multiple text regions")
        if rec_texts and rec_scores:
            # Берем первый элемент из списков
            raw_text = str(rec_texts[0]).strip()
            confidence = float(rec_scores[0])
        else:
            # Если списки пустые
            raw_text = ""
            confidence = 0.0
    else:
        raw_text = ""
        confidence = 0.0
    
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid PaddleOCR rec_score")
    
    reading = {"raw_text": raw_text, "confidence": confidence,
               "value": None, "error": None}
    
    if not raw_text:
        reading["error"] = "no_text"
    elif not raw_text.isascii() or not raw_text.isdecimal():
        reading["error"] = "not_single_number"
    elif len(raw_text) > len(str(max_hp)) or int(raw_text) > max_hp:
        reading["error"] = "out_of_range"
    elif confidence < TOWER_HP_MIN_CONFIDENCE:
        reading["error"] = "low_confidence"
    else:
        reading["value"] = int(raw_text)
    
    return reading


@dataclass
class TowerHPState:
    hp: int | None = None
    confidence: float = 0.0
    observed_at_ms: float | None = None  # First reading of the confirmed change.
    confirmed_at_ms: float | None = None
    last_seen_at_ms: float | None = None
    stale: bool = True
    pending_hp: int | None = None
    pending_count: int = 0
    pending_since_ms: float | None = None
    pending_confidence: float = 0.0

    def update(self, reading: dict, timestamp_ms: float) -> bool:
        """Only consecutive valid readings confirm a change; missing text != zero."""
        value = reading["value"]
        if value is None:
            self.pending_hp = None
            self.pending_count = 0
            self.stale = True
            return False
        if value == self.hp:
            self.confidence = reading["confidence"]
            self.last_seen_at_ms = timestamp_ms
            self.stale = False
            self.pending_hp = None
            self.pending_count = 0
            return False
        self.stale = True  # Old HP retained while a different value is unconfirmed.
        if value != self.pending_hp:
            self.pending_hp = value
            self.pending_count = 1
            self.pending_since_ms = timestamp_ms
            self.pending_confidence = reading["confidence"]
        else:
            self.pending_count += 1
            self.pending_confidence = min(self.pending_confidence, reading["confidence"])
        if self.pending_count < TOWER_HP_CONFIRM_READINGS:
            return False
        self.hp = value
        self.confidence = self.pending_confidence
        self.observed_at_ms = self.pending_since_ms
        self.confirmed_at_ms = timestamp_ms
        self.last_seen_at_ms = timestamp_ms
        self.stale = False
        self.pending_hp = None
        self.pending_count = 0
        return True

    def snapshot(self) -> dict:
        return {"hp": self.hp, "confidence": self.confidence,
                "observed_at_ms": self.observed_at_ms,
                "confirmed_at_ms": self.confirmed_at_ms,
                "last_seen_at_ms": self.last_seen_at_ms, "stale": self.stale}


@dataclass
class TowerHPSampler:
    interval_ms: float
    next_due_ms: float = 0.0

    def due(self, timestamp_ms: float) -> bool:
        if timestamp_ms + 1e-6 < self.next_due_ms:
            return False
        # Absolute video-time deadlines: no accumulated drift or catch-up bursts.
        self.next_due_ms = (math.floor((timestamp_ms + 1e-6) / self.interval_ms) + 1) * self.interval_ms
        return True


TOWER_HP_CROP_KEY_SEPARATOR = "::crop::"


def tower_hp_candidate_key(tower_id: str, crop_index: int) -> str:
    return f"{tower_id}{TOWER_HP_CROP_KEY_SEPARATOR}{crop_index}"


def parse_tower_hp_candidate_key(ocr_key: str) -> tuple[str, int | None]:
    tower_id, separator, crop_index = ocr_key.partition(
        TOWER_HP_CROP_KEY_SEPARATOR
    )
    if not separator:
        return ocr_key, None
    return tower_id, int(crop_index)


def build_tower_hp_candidate_batch(
    crop_candidates: dict[str, list[tuple[int, int, int, int]]],
) -> dict[str, tuple[int, int, int, int]]:
    return {
        tower_hp_candidate_key(tower_id, crop_index): crop
        for tower_id, crops in crop_candidates.items()
        for crop_index, crop in enumerate(crops)
    }


def select_best_tower_hp_crops(
    readings: dict,
    previews: dict,
    crop_candidates: dict[str, list[tuple[int, int, int, int]]],
    active_indices: dict[str, int],
) -> tuple[dict, dict, dict[str, int]]:
    """Collapse a candidate scan to one highest-confidence crop per tower."""
    selected_readings = {}
    selected_previews = {}
    selected_indices = dict(active_indices)

    for tower_id, crops in crop_candidates.items():
        available = []
        for crop_index in range(len(crops)):
            key = tower_hp_candidate_key(tower_id, crop_index)
            if key in readings:
                available.append((crop_index, key, readings[key]))
        if not available:
            continue

        # A valid number always wins over a confident non-numeric result. If all
        # candidates are invalid, expose the most confident error but keep the
        # previously selected coordinates.
        crop_index, key, reading = max(
            available,
            key=lambda item: (
                item[2].get("value") is not None,
                float(item[2].get("confidence", 0.0)),
            ),
        )
        if reading.get("value") is not None:
            selected_indices[tower_id] = crop_index
        selected_readings[tower_id] = reading
        selected_previews[tower_id] = previews[key]

    return selected_readings, selected_previews, selected_indices


def recognize_tower_hp_previews(previews: dict, recognizer) -> tuple[dict, dict]:
    """Recognize copied crops; this function may run in a background thread."""
    if not previews:
        return {}, {}
    started = time.perf_counter()
    try:
        results = list(recognizer.predict(input=list(previews.values()), batch_size=len(previews)))
        if len(results) != len(previews):
            raise ValueError(f"Expected {len(previews)} HP results, got {len(results)}")
    except (RuntimeError, OSError, ValueError) as exc:
        batch_ms = (time.perf_counter() - started) * 1000
        return {tower_id: {"raw_text": "", "confidence": 0.0, "value": None,
                           "error": f"ocr_error: {exc}", "batch_ms": round(batch_ms, 2),
                           "ocr_ms": round(batch_ms / len(previews), 2)}
                for tower_id in previews}, previews
    batch_ms = (time.perf_counter() - started) * 1000
    readings = {}
    for ocr_key, result in zip(previews, results):
        # Keep this helper independently testable when only selected functions
        # are loaded from the module.
        tower_id = ocr_key.partition("::crop::")[0]
        try:
            reading = parse_tower_hp_data(result, TOWER_HP_MAX_VALUES[tower_id])
        except (KeyError, TypeError, ValueError) as exc:
            reading = {"raw_text": "", "confidence": 0.0, "value": None,
                       "error": f"ocr_error: {exc}"}
        reading["batch_ms"] = round(batch_ms, 2)
        reading["ocr_ms"] = round(batch_ms / len(previews), 2)  # Amortized, not per-crop timing.
        readings[ocr_key] = reading
    return readings, previews


def read_tower_hp(frame: np.ndarray, crops: dict, recognizer) -> tuple[dict, dict]:
    previews = {tower_id: prepare_tower_hp_crop(frame[y1:y2, x1:x2])
                for tower_id, (x1, y1, x2, y2) in crops.items()}
    return recognize_tower_hp_previews(previews, recognizer)


class TowerHPAsyncReader:
    """Allow video inference to continue while a single OCR batch is running."""

    def __init__(self, recognizer) -> None:
        self.recognizer = recognizer
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tower-hp")
        self.future: Future | None = None
        self.timestamp_ms: float | None = None

    @property
    def is_busy(self) -> bool:
        return self.future is not None

    def submit(self, frame: np.ndarray, crops: dict, timestamp_ms: float) -> bool:
        if self.future is not None or not self.recognizer.is_running:
            return False
        # Copy only the small crops; the main loop can immediately reuse `frame`.
        previews = {tower_id: prepare_tower_hp_crop(frame[y1:y2, x1:x2])
                    for tower_id, (x1, y1, x2, y2) in crops.items()}
        self.timestamp_ms = timestamp_ms
        self.future = self.executor.submit(
            recognize_tower_hp_previews, previews, self.recognizer
        )
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, timestamp_ms = self.future, self.timestamp_ms
        self.future = None
        self.timestamp_ms = None
        readings, previews = future.result()
        return timestamp_ms, readings, previews

    def close(self) -> None:
        self.recognizer.close()
        self.executor.shutdown(wait=True, cancel_futures=True)


def draw_tower_hp(frame: np.ndarray, crops: dict, states: dict) -> None:
    for tower_id, (x1, y1, x2, y2) in crops.items():
        state = states[tower_id]
        hp_text = "?" if state.hp is None else str(state.hp)
        label = f"{tower_id}: {hp_text}" + (" stale" if state.stale else "")
        color = (0, 180, 255) if state.stale else (0, 255, 0)
        y = max(20, y1 - 10)
        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 1, cv2.LINE_AA)


def show_tower_hp_debug(frame: np.ndarray, crops: dict, previews: dict,
                        readings: dict) -> None:
    canvas = np.zeros((len(crops) * 140, 600, 3), dtype=np.uint8)
    for index, (tower_id, (x1, y1, x2, y2)) in enumerate(crops.items()):
        y = index * 140
        raw = cv2.resize(frame[y1:y2, x1:x2], (280, 90))
        processed = cv2.resize(previews[tower_id], (280, 90))
        canvas[y + 45:y + 135, 10:290] = raw
        canvas[y + 45:y + 135, 310:590] = processed
        reading = readings[tower_id]
        label = f"{tower_id}: {reading['raw_text']!r} {reading['confidence']:.2f}"
        cv2.putText(canvas, label, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(canvas, reading["error"] or "valid", (10, y + 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    cv2.imshow("Tower HP debug: original / OCR input", canvas)


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


def draw_card_event_markers(
    canvas: np.ndarray,
    markers: list[tuple[int, int, str, int]],
    frame_index: int,
) -> None:
    """Draw unexpired card events over units, using 1-based field cells."""
    markers[:] = [marker for marker in markers if frame_index < marker[3]]
    for column, row, card_name, _ in markers:
        left = (column - 1) * FIELD_CELL_SIZE
        top = (row - 1) * FIELD_CELL_SIZE
        right = left + FIELD_CELL_SIZE - 1
        bottom = top + FIELD_CELL_SIZE - 1
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 255), 3)
        cv2.drawMarker(
            canvas,
            (left + FIELD_CELL_SIZE // 2, top + FIELD_CELL_SIZE // 2),
            (255, 255, 255),
            cv2.MARKER_TILTED_CROSS,
            max(5, FIELD_CELL_SIZE // 2),
            2,
        )
        label = f"Played: {card_name}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        text_x = max(2, min(left, canvas.shape[1] - text_width - 3))
        text_y = top - 6
        if text_y < text_height + 3:
            text_y = bottom + text_height + 6
        cv2.rectangle(
            canvas,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )


def track_color(
    track_id: int | None,
    mean_color_bgr: tuple[float, float, float] | None,
) -> tuple[int, int, int]:
    """Compare weighted mean channels; vary the chosen shade by track ID."""
    if mean_color_bgr is None:
        return 190, 190, 190
    blue, _, red = mean_color_bgr
    red_score = red * FIELD_COLOR_RED_WEIGHT
    blue_score = blue * FIELD_COLOR_BLUE_WEIGHT
    intensity = 180 + (int(track_id) * 47 % 76) if track_id is not None else 255
    if red_score > blue_score:
        return 40, 40, intensity
    if blue_score > red_score:
        return intensity, 40, 40
    return 190, 190, 190


def draw_objects_on_field(
    background: np.ndarray,
    objects: list[tuple[str, int | None, float, float]],
    battlefield_shape: tuple[int, ...],
    mean_colors: dict[int, tuple[float, float, float] | None],
) -> np.ndarray:
    """Highlight field cells containing tracked unit centers."""
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
        color = track_color(track_id, mean_colors.get(track_id))
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


def recognize_elixir_bar(
    frame: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
) -> dict[str, float | int]:
    """Find the bright/dark elixir boundary with a horizontal binary search."""
    x1, y1, x2, y2 = crop_bounds
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Elixir bar crop is empty: {crop_bounds}")

    value_channel = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    crop_height, crop_width = value_channel.shape
    strip_height = min(ELIXIR_SEARCH_STRIP_HEIGHT, crop_height)
    strip_top = max(0, (crop_height - strip_height) // 2)
    strip_bottom = strip_top + strip_height
    sample_half_width = ELIXIR_SEARCH_SAMPLE_WIDTH // 2

    def brightness_at(column: int) -> float:
        sample_left = max(0, column - sample_half_width)
        sample_right = min(crop_width, column + sample_half_width + 1)
        sample = value_channel[
            strip_top:strip_bottom,
            sample_left:sample_right,
        ]
        return float(np.median(sample))

    transition_brightness = (
        ELIXIR_DARK_BRIGHTNESS_MAX + ELIXIR_LIGHT_BRIGHTNESS_MIN
    ) / 2

    def is_bright(brightness: float) -> bool:
        if brightness >= ELIXIR_LIGHT_BRIGHTNESS_MIN:
            return True
        if brightness <= ELIXIR_DARK_BRIGHTNESS_MAX:
            return False
        return brightness >= transition_brightness

    left_brightness = brightness_at(0)
    right_brightness = brightness_at(crop_width - 1)

    if not is_bright(left_brightness):
        boundary = 0.0
        boundary_brightness = left_brightness
    elif is_bright(right_brightness):
        boundary = float(crop_width)
        boundary_brightness = right_brightness
    else:
        # The left part of the bar is bright and the right part is dark. Keep
        # that invariant while narrowing the interval to the transition pixel.
        bright_column = 0
        dark_column = crop_width - 1
        boundary_brightness = brightness_at((bright_column + dark_column) // 2)
        while dark_column - bright_column > 1:
            middle = (bright_column + dark_column) // 2
            boundary_brightness = brightness_at(middle)
            if is_bright(boundary_brightness):
                bright_column = middle
            else:
                dark_column = middle
        boundary = (bright_column + dark_column) / 2

    fill_ratio = min(1.0, max(0.0, boundary / crop_width))
    return {
        "value": fill_ratio * ELIXIR_MAX_VALUE,
        "fill_ratio": fill_ratio,
        "boundary_x": int(round(x1 + boundary)),
        "boundary_brightness": boundary_brightness,
        "left_brightness": left_brightness,
        "right_brightness": right_brightness,
    }


def draw_elixir_bar_reading(
    frame: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    reading: dict[str, float | int],
) -> None:
    """Draw the measured boundary and current value below the source bar."""
    x1, y1, x2, y2 = crop_bounds
    boundary_x = int(reading["boundary_x"])
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
    cv2.line(
        frame,
        (boundary_x, y1),
        (boundary_x, y2 - 1),
        (0, 255, 0),
        3,
    )

    label = (
        f"Elixir: {float(reading['value']):.1f}/{ELIXIR_MAX_VALUE:g}  "
        f"fill={float(reading['fill_ratio']) * 100:.0f}%  "
        f"V={float(reading['boundary_brightness']):.0f}"
    )
    label_y = min(frame.shape[0] - 8, y2 + 30)
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


@dataclass(frozen=True)
class CardObservation:
    frame_index: int
    class_name: str
    confidence: float


@dataclass
class ElixirEventCandidate:
    first_frame: int
    last_frame: int
    positions: list[tuple[float, float]]
    emitted: bool = False

    @property
    def mean_position(self) -> tuple[float, float]:
        count = len(self.positions)
        return (
            sum(position[0] for position in self.positions) / count,
            sum(position[1] for position in self.positions) / count,
        )


class ElixirEventTracker:
    """Detect a persistent elixir drop while tolerating short missed detections."""

    def __init__(
        self,
        video_fps: float,
        battlefield_width: int,
        battlefield_height: int,
    ) -> None:
        self.fps = video_fps
        self.battlefield_width = battlefield_width
        self.battlefield_height = battlefield_height
        self.dwell_frames = max(
            1,
            math.ceil(video_fps * ELIXIR_EVENT_DWELL_MS / 1000),
        )
        self.max_gap_frames = max(
            1,
            math.ceil(video_fps * ELIXIR_EVENT_MAX_GAP_MS / 1000),
        )
        self.cooldown_frames = max(
            1,
            math.ceil(video_fps * ELIXIR_EVENT_COOLDOWN_MS / 1000),
        )
        self.candidates: list[ElixirEventCandidate] = []
        self.recent_events: list[tuple[int, tuple[float, float]]] = []

    def _grid_distance(
        self,
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        dx = (first[0] - second[0]) * FIELD_COLUMNS / self.battlefield_width
        dy = (first[1] - second[1]) * FIELD_ROWS / self.battlefield_height
        return math.hypot(dx, dy)

    def _inside_region(
        self,
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> bool:
        return self._grid_distance(first, second) <= ELIXIR_EVENT_REGION_RADIUS_CELLS

    def update(
        self,
        centers: list[tuple[float, float]],
        frame_index: int,
    ) -> list[ElixirEventCandidate]:
        self.candidates = [
            candidate
            for candidate in self.candidates
            if frame_index - candidate.last_frame <= self.max_gap_frames
        ]
        self.recent_events = [
            event
            for event in self.recent_events
            if frame_index - event[0] <= self.cooldown_frames
        ]

        matched_candidates: set[int] = set()
        for center in centers:
            nearest_index = None
            nearest_distance = float("inf")
            for candidate_index, candidate in enumerate(self.candidates):
                if candidate_index in matched_candidates:
                    continue
                distance = self._grid_distance(center, candidate.mean_position)
                if (
                    distance <= ELIXIR_EVENT_REGION_RADIUS_CELLS
                    and distance < nearest_distance
                ):
                    nearest_index = candidate_index
                    nearest_distance = distance

            if nearest_index is not None:
                candidate = self.candidates[nearest_index]
                candidate.positions.append(center)
                candidate.last_frame = frame_index
                matched_candidates.add(nearest_index)
                continue

            if any(
                self._inside_region(center, event_position)
                for _, event_position in self.recent_events
            ):
                continue

            self.candidates.append(
                ElixirEventCandidate(
                    first_frame=frame_index,
                    last_frame=frame_index,
                    positions=[center],
                )
            )
            matched_candidates.add(len(self.candidates) - 1)

        confirmed: list[ElixirEventCandidate] = []
        for candidate in self.candidates:
            if (
                not candidate.emitted
                and candidate.last_frame == frame_index
                and frame_index - candidate.first_frame >= self.dwell_frames
                and len(candidate.positions) >= ELIXIR_EVENT_MIN_DETECTIONS
            ):
                candidate.emitted = True
                confirmed.append(candidate)
                self.recent_events.append((frame_index, candidate.mean_position))
        return confirmed


def remember_card_predictions(
    histories: list[deque[CardObservation]],
    predictions: list[tuple[str, float]],
    frame_index: int,
) -> None:
    for history, (class_name, confidence) in zip(histories, predictions):
        history.append(
            CardObservation(
                frame_index=frame_index,
                class_name=class_name,
                confidence=confidence,
            )
        )


def resolve_played_card(
    histories: list[deque[CardObservation]],
    event_first_frame: int,
    event_last_frame: int,
    video_fps: float,
    sampling_fps: float | None = None,
    *, return_details: bool = False,
) -> tuple[str, int | None, float] | dict:
    """Find the slot that became empty and vote on its preceding card class."""
    sampling_fps = video_fps if sampling_fps is None else sampling_fps
    empty_lead_frames = max(1, math.ceil(video_fps * CARD_EMPTY_LEAD_MS / 1000))
    lookback_frames = max(
        math.ceil(CARD_MIN_PREVIOUS_FRAMES * video_fps / sampling_fps),
        math.ceil(video_fps * CARD_PREVIOUS_LOOKBACK_MS / 1000),
    )
    previous_sample_frames = max(
        CARD_MIN_PREVIOUS_FRAMES,
        math.ceil(sampling_fps * CARD_PREVIOUS_SAMPLE_MS / 1000),
    )
    empty_confirm_frames = max(
        1,
        math.ceil(video_fps * CARD_EMPTY_CONFIRM_MS / 1000),
    )
    best_match = None

    for slot_index, history in enumerate(histories):
        history_samples = list(history)
        possible_empty_transitions = [
            observation
            for sample_index, observation in enumerate(history_samples)
            if event_first_frame - empty_lead_frames
            <= observation.frame_index
            <= event_last_frame
            and observation.class_name.lower() == EMPTY_CARD_CLASS
            and (sample_index == 0
                 or history_samples[sample_index - 1].class_name.lower() != EMPTY_CARD_CLASS)
        ]
        for empty_observation in possible_empty_transitions:
            empty_frame = empty_observation.frame_index
            empty_support = [
                observation
                for observation in history
                if empty_frame
                <= observation.frame_index
                <= min(event_last_frame, empty_frame + empty_confirm_frames)
                and observation.class_name.lower() == EMPTY_CARD_CLASS
            ]
            if len(empty_support) < 2:
                continue

            previous_sequence = [
                observation
                for observation in history
                if empty_frame - lookback_frames
                <= observation.frame_index
                < empty_frame
            ][-previous_sample_frames:]
            previous_observations = [
                observation
                for observation in previous_sequence
                if observation.class_name.lower() != EMPTY_CARD_CLASS
            ]
            if (
                len(previous_observations) < CARD_MIN_PREVIOUS_FRAMES
                or len(previous_observations) / len(previous_sequence) < 0.60
            ):
                continue

            class_scores: dict[str, float] = defaultdict(float)
            class_counts: dict[str, int] = defaultdict(int)
            for observation in previous_observations:
                class_scores[observation.class_name] += observation.confidence
                class_counts[observation.class_name] += 1

            class_name = max(
                class_scores,
                key=lambda name: (class_scores[name], class_counts[name], name),
            )
            class_count = class_counts[class_name]
            average_confidence = class_scores[class_name] / class_count
            score = (
                -float(abs(empty_frame - event_first_frame)),
                float(len(empty_support)),
                float(class_count),
                average_confidence,
            )
            match = (score, class_name, slot_index + 1, average_confidence, empty_frame)
            if best_match is None or match[0] > best_match[0]:
                best_match = match

    if best_match is None:
        class_name, slot_number, confidence, empty_frame = "unknown", None, 0.0, None
    else:
        _, class_name, slot_number, confidence, empty_frame = best_match
    if return_details:
        return {"card": class_name, "slot": slot_number,
                "confidence": confidence, "empty_frame": empty_frame}
    return class_name, slot_number, confidence


def battlefield_position_to_cell(
    position: tuple[float, float],
    battlefield_shape: tuple[int, ...],
) -> tuple[int, int] | None:
    battlefield_height, battlefield_width = battlefield_shape[:2]
    center_x, center_y = position
    if not (
        0 <= center_x < battlefield_width
        and 0 <= center_y < battlefield_height
    ):
        return None
    column = int(center_x / battlefield_width * FIELD_COLUMNS) + 1
    row = int(center_y / battlefield_height * FIELD_ROWS) + 1
    return column, row


def battlefield_position_to_field_cell(
    position: tuple[float, float],
    battlefield_shape: tuple[int, ...],
) -> tuple[int, int] | None:
    """Return a 1-based cell inside the 32x18 field, including # tiles."""
    return battlefield_position_to_cell(position, battlefield_shape)


def create_event_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("battle_events")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


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
        return self._state_to_xyxy(corrected, image_width, image_height)

    @staticmethod
    def _state_to_xyxy(
        state: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        cx, cy, width, height = map(float, state[:4])
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

    def predict(self, image_width: int, image_height: int) -> np.ndarray:
        """Advance one missing frame and return the predicted box."""
        predicted = self.filter.predict().reshape(8)
        return self._state_to_xyxy(predicted, image_width, image_height)


@dataclass
class UnitClassificationCacheEntry:
    weighted_scores: dict[str, float] = field(default_factory=dict)
    confidence_sums: dict[str, float] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    sample_count: int = 0
    class_name: str = "unknown"
    confidence: float = 0.0
    locked: bool = False
    last_classified_at_ms: float | None = None

    def classification_due(self, timestamp_ms: float) -> bool:
        if self.locked:
            return False
        if self.last_classified_at_ms is None:
            return True
        return (
            timestamp_ms - self.last_classified_at_ms + 1e-6
            >= UNIT_CLASSIFICATION_INTERVAL_MS
        )

    def add(
        self,
        class_name: str,
        confidence: float,
        timestamp_ms: float,
    ) -> None:
        """Add one prediction and update the confidence-weighted winner."""
        confidence = float(confidence)
        self.last_classified_at_ms = float(timestamp_ms)
        self.sample_count += 1
        self.weighted_scores[class_name] = (
            self.weighted_scores.get(class_name, 0.0) + confidence
        )
        self.confidence_sums[class_name] = (
            self.confidence_sums.get(class_name, 0.0) + confidence
        )
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

        self.class_name = max(
            self.weighted_scores,
            key=lambda name: (self.weighted_scores[name], self.class_counts[name]),
        )
        winner_count = self.class_counts[self.class_name]
        self.confidence = self.confidence_sums[self.class_name] / winner_count
        enough_consistent_samples = (
            winner_count >= UNIT_CLASSIFICATION_CONFIRM_SAMPLES
            and self.confidence >= UNIT_CLASSIFICATION_LOCK_MIN_CONFIDENCE
        )
        self.locked = (
            enough_consistent_samples
            or self.sample_count >= UNIT_CLASSIFICATION_MAX_SAMPLES
        )

    @property
    def display_text(self) -> str:
        if self.sample_count == 0:
            return "None"
        return f"{self.class_name} {self.confidence:.2f}"


@dataclass
class UnitTrackMemory:
    confidence: float
    mean_color_bgr: tuple[float, float, float] | None
    classification_text: str


@dataclass(frozen=True)
class PredictedUnitTrack:
    track_id: int
    bbox: np.ndarray
    confidence: float
    missed_processed_frames: int
    memory: UnitTrackMemory


def predict_missing_unit_tracks(
    filters: dict[int, BoundingBoxKalmanFilter],
    last_seen: dict[int, int],
    memories: dict[int, UnitTrackMemory],
    detected_track_ids: set[int],
    frame_index: int,
    image_shape: tuple[int, ...],
    source_fps: float,
    sampling_fps: float,
) -> list[PredictedUnitTrack]:
    """Coast unit tracks through short detector gaps using Kalman predictions."""
    image_height, image_width = image_shape[:2]
    hold_source_frames = max(
        1,
        math.ceil(TRACK_HOLD_PROCESSED_FRAMES * source_fps / sampling_fps),
    )
    predictions: list[PredictedUnitTrack] = []

    for track_id, memory in list(memories.items()):
        if track_id in detected_track_ids:
            continue
        seen_frame = last_seen.get(track_id)
        kalman_filter = filters.get(track_id)
        if seen_frame is None or kalman_filter is None:
            memories.pop(track_id, None)
            continue

        missed_source_frames = frame_index - seen_frame
        if missed_source_frames <= 0:
            continue
        if missed_source_frames > hold_source_frames:
            memories.pop(track_id, None)
            continue

        # frame_index is a source-video index, while this setting is intentionally
        # expressed in processed frames. round() avoids 59.94 -> 30 rounding up.
        missed_processed_frames = max(
            1,
            round(missed_source_frames * sampling_fps / source_fps),
        )
        confidence = memory.confidence * (
            TRACK_CONFIDENCE_DECAY ** missed_processed_frames
        )
        if confidence < TRACK_MIN_PREDICTED_CONFIDENCE:
            continue

        predictions.append(
            PredictedUnitTrack(
                track_id=track_id,
                bbox=kalman_filter.predict(image_width, image_height),
                confidence=confidence,
                missed_processed_frames=missed_processed_frames,
                memory=memory,
            )
        )

    return predictions


def smooth_result_boxes(
    result,
    filters: dict[int, BoundingBoxKalmanFilter],
    last_seen: dict[int, int],
    frame_index: int,
    image_shape: tuple[int, ...],
) -> set[int]:
    """Replace tracked result boxes with their Kalman-smoothed coordinates."""
    detected_track_ids: set[int] = set()
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
            detected_track_ids.add(track_id)
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

    return detected_track_ids


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


def iter_processing_frames(capture, source_fps: float, target_fps: float):
    """Sample video uniformly, yielding original indices for all event timers."""
    source_index = -1
    sample_index = 0
    while capture.isOpened() and capture.grab():
        source_index += 1
        # Absolute deadlines also handle non-integer ratios, e.g. 59.94 -> 20.
        next_index = math.ceil(sample_index * source_fps / target_fps - 1e-9)
        if source_index < next_index:
            continue
        ret, frame = capture.retrieve()
        if not ret:
            break
        yield source_index, frame
        sample_index += 1


def run_video_prediction(
    input_video=INPUT_VIDEO, output_video=OUTPUT_VIDEO, *,
    process_fps_limit=FPS_PROCESS, display=True, write_video=True,
    observation_callback=None, hp_enabled=TOWER_HP_ENABLED, synchronous_hp=False,
    write_logs=True, annotation_callback=None,
):
    """Run shared perception; the callback receives causal per-frame observations.

    Offline callers use display=False, write_video=False, write_logs=False and
    synchronous_hp=True. Importing this module does not open videos or models.
    annotation_callback(output_frame, frame_data) may draw recommendations in
    place after perception, before display/video writing; it never sees raw crops.
    """
    INPUT_VIDEO = Path(input_video)
    OUTPUT_VIDEO = Path(output_video)
    FPS_PROCESS = process_fps_limit
    TOWER_HP_ENABLED = hp_enabled
    cap = None
    out = None
    tower_hp_recognizer = None
    tower_hp_async_reader = None
    try:
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
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps}")
        if FPS_PROCESS is not None and (
            not math.isfinite(FPS_PROCESS) or FPS_PROCESS <= 0
        ):
            raise ValueError("FPS_PROCESS must be a positive number or None")
        process_fps = fps if FPS_PROCESS is None else min(float(FPS_PROCESS), fps)
        print(f"Video FPS: {fps:g}; processing/output FPS: {process_fps:g}")

        video_size = (width, height)
        tower_hp_crop_candidates = {}
        if TOWER_HP_ENABLED:
            for tower_id, regions in (
                ("ally_1", TOWER_HP_1), ("ally_2", TOWER_HP_2),
                ("enemy_1", TOWER_HP_ENEMY_1), ("enemy_2", TOWER_HP_ENEMY_2),
            ):
                if video_size not in regions:
                    raise ValueError(
                        f"Missing HP crop for {tower_id} at {video_size} in model_paths.py. "
                        "Add its coordinates or set TOWER_HP_ENABLED=False."
                    )
                configured_crops = regions[video_size]
                if not isinstance(configured_crops, list) or not configured_crops:
                    raise ValueError(
                        f"HP crops for {tower_id} must be a non-empty list, "
                        f"got: {configured_crops!r}"
                    )
                validated_crops = []
                for crop_index, crop in enumerate(configured_crops):
                    if not isinstance(crop, (tuple, list)) or len(crop) != 4:
                        raise ValueError(
                            f"Invalid HP crop #{crop_index} for {tower_id}: {crop!r}"
                        )
                    x1, y1, x2, y2 = crop
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise ValueError(
                            f"Invalid HP crop #{crop_index} for {tower_id}: {crop!r}"
                        )
                    validated_crops.append((x1, y1, x2, y2))
                if (tower_id not in TOWER_HP_MAX_VALUES
                        or not isinstance(TOWER_HP_MAX_VALUES[tower_id], int)
                        or TOWER_HP_MAX_VALUES[tower_id] <= 0):
                    raise ValueError(f"Set a positive integer max HP for {tower_id}")
                tower_hp_crop_candidates[tower_id] = validated_crops

        tower_hp_active_crop_indices = {
            tower_id: 0 for tower_id in tower_hp_crop_candidates
        }
        tower_hp_crops = {
            tower_id: crops[0] for tower_id, crops in tower_hp_crop_candidates.items()
        }

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
        if video_size not in ELIXIR_BAR:
            raise ValueError(
                f"Elixir bar crop is not configured for video resolution "
                f"{width}x{height}. Add this resolution to ELIXIR_BAR in "
                "model_paths.py."
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

        elixir_bar_x1, elixir_bar_y1, elixir_bar_x2, elixir_bar_y2 = (
            ELIXIR_BAR[video_size]
        )
        if not (0 <= elixir_bar_x1 < elixir_bar_x2 <= width):
            raise ValueError(
                f"Invalid elixir bar horizontal crop: "
                f"{(elixir_bar_x1, elixir_bar_x2)} for width {width}"
            )
        if not (0 <= elixir_bar_y1 < elixir_bar_y2 <= height):
            raise ValueError(
                f"Invalid elixir bar vertical crop: "
                f"{(elixir_bar_y1, elixir_bar_y2)} for height {height}"
            )

        if write_video:
            OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, process_fps, (width, height))
            if not out.isOpened():
                raise RuntimeError(f"Cannot create output video: {OUTPUT_VIDEO}")

        #
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загрузка модели
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
        bar_kalman_filters: dict[int, BoundingBoxKalmanFilter] = {}
        bar_kalman_last_seen: dict[int, int] = {}
        unit_track_memories: dict[int, UnitTrackMemory] = {}
        unit_classification_cache: dict[int, UnitClassificationCacheEntry] = {}
        elixir_kalman_filters: dict[int, BoundingBoxKalmanFilter] = {}
        elixir_kalman_last_seen: dict[int, int] = {}
        elixir_event_tracker = ElixirEventTracker(
            video_fps=fps,
            battlefield_width=crop_x2 - crop_x1,
            battlefield_height=crop_y2 - crop_y1,
        )
        card_history_length = max(1, math.ceil(process_fps * CARD_HISTORY_MS / 1000) + 1)
        card_histories: list[deque[CardObservation]] = [
            deque(maxlen=card_history_length) for _ in range(CARD_COUNT)
        ]
        event_logger = (create_event_logger(EVENT_LOG_PATH) if write_logs
                        else logging.getLogger("offline_rl.recognition"))
        tower_hp_recognizer = create_tower_hp_recognizer() if TOWER_HP_ENABLED else None
        tower_hp_async_reader = (
            TowerHPAsyncReader(tower_hp_recognizer) if tower_hp_recognizer is not None and not synchronous_hp else None
        )
        tower_hp_states = {tower_id: TowerHPState() for tower_id in tower_hp_crops}
        tower_hp = {tower_id: state.snapshot() for tower_id, state in tower_hp_states.items()}
        tower_hp_sampler = TowerHPSampler(1000 / min(TOWER_HP_FPS, process_fps)) if TOWER_HP_ENABLED else None
        tower_hp_crop_selection_sampler = (
            TowerHPSampler(TOWER_HP_CROP_SELECTION_INTERVAL_MS)
            if TOWER_HP_ENABLED
            else None
        )
        tower_hp_candidate_batch = build_tower_hp_candidate_batch(
            tower_hp_crop_candidates
        )
        if TOWER_HP_ENABLED and write_logs:
            TOWER_HP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TOWER_HP_LOG_PATH.open("a", encoding="utf-8") as hp_log:
                hp_log.write(json.dumps({"type": "run", "video": str(INPUT_VIDEO),
                                         "source_fps": fps, "ocr_fps": min(TOWER_HP_FPS, process_fps),
                                         "crop_selection_interval_ms": TOWER_HP_CROP_SELECTION_INTERVAL_MS,
                                         "crop_candidates": tower_hp_crop_candidates,
                                         "active_crop_indices": tower_hp_active_crop_indices,
                                         "crops": tower_hp_crops, "ocr_backend": "paddleocr",
                                         "model": TOWER_HP_MODEL_NAME, "device": TOWER_HP_DEVICE}) + "\n")
        card_event_markers: list[tuple[int, int, str, int]] = []
        card_event_marker_frames = max(1, math.ceil(fps * CARD_EVENT_MARKER_MS / 1000))
        field_background = build_field_background()
        if display:
            cv2.namedWindow(FIELD_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                FIELD_WINDOW_NAME,
                field_background.shape[1],
                field_background.shape[0],
            )

        processed_frames = 0
        last_frame_index = -1
        stopped_by_user = False

        for frame_count, frame in iter_processing_frames(cap, fps, process_fps):
            # frame_count remains a source-video index: all ms thresholds use source fps.
            timestamp_ms = frame_count * 1000 / fps
            processed_frames += 1
            last_frame_index = frame_count
            completed_hp = tower_hp_async_reader.poll() if tower_hp_async_reader is not None else None
            if synchronous_hp and tower_hp_recognizer is not None:
                # Offline extraction blocks on scheduled OCR, so observations do not
                # depend on GPU speed or when the asynchronous worker happens to finish.
                scan_due = tower_hp_crop_selection_sampler.due(timestamp_ms)
                hp_due = tower_hp_sampler.due(timestamp_ms)
                if scan_due or hp_due:
                    readings, previews = read_tower_hp(
                        frame, tower_hp_candidate_batch if scan_due else tower_hp_crops,
                        tower_hp_recognizer,
                    )
                    completed_hp = (timestamp_ms, readings, previews)
                    if not tower_hp_recognizer.is_running:
                        raise RuntimeError("Tower HP OCR worker stopped during trajectory extraction")
            if completed_hp is not None:
                hp_timestamp_ms, hp_readings, hp_previews = completed_hp
                crop_selection_scan = any(
                    parse_tower_hp_candidate_key(ocr_key)[1] is not None
                    for ocr_key in hp_readings
                )
                if crop_selection_scan:
                    (
                        hp_readings,
                        hp_previews,
                        selected_crop_indices,
                    ) = select_best_tower_hp_crops(
                        hp_readings,
                        hp_previews,
                        tower_hp_crop_candidates,
                        tower_hp_active_crop_indices,
                    )
                    tower_hp_active_crop_indices.update(selected_crop_indices)
                    tower_hp_crops.update(
                        {
                            tower_id: tower_hp_crop_candidates[tower_id][crop_index]
                            for tower_id, crop_index in selected_crop_indices.items()
                        }
                    )
                hp_frame_index = round(hp_timestamp_ms * fps / 1000)
                for tower_id, reading in hp_readings.items():
                    state = tower_hp_states[tower_id]
                    if state.update(reading, hp_timestamp_ms):
                        event_logger.info(
                            "Tower HP: %s=%d; confidence=%.2f; observed_at_ms=%.1f; confirmed_at_ms=%.1f",
                            tower_id, state.hp, state.confidence, state.observed_at_ms, hp_timestamp_ms,
                        )
                    tower_hp[tower_id] = state.snapshot()
                batch_errors = sorted({reading["error"] for reading in hp_readings.values()
                                       if reading["error"] and reading["error"].startswith("ocr_error:")})
                for error in batch_errors:
                    event_logger.warning("Tower HP batch: %s", error)
                if write_logs:
                    with TOWER_HP_LOG_PATH.open("a", encoding="utf-8") as hp_log:
                        hp_log.write(json.dumps({"type": "observation", "frame_index": hp_frame_index,
                                                 "timestamp_ms": hp_timestamp_ms, "readings": hp_readings,
                                                 "crop_selection_scan": crop_selection_scan,
                                                 "active_crop_indices": tower_hp_active_crop_indices,
                                                 "active_crops": tower_hp_crops,
                                                 "tower_hp": tower_hp}, ensure_ascii=False) + "\n")
                if display and show_tower_hp:
                    show_tower_hp_debug(frame, tower_hp_crops, hp_previews, hp_readings)
                if not tower_hp_recognizer.is_running:
                    event_logger.error("Tower HP OCR worker stopped; OCR is disabled for this run")
                    tower_hp_sampler = None

            if (
                tower_hp_sampler is not None
                and tower_hp_async_reader is not None
                and not tower_hp_async_reader.is_busy
            ):
                # A full candidate scan has priority. Between scans, OCR only receives
                # the last highest-confidence crop selected for each tower.
                if tower_hp_crop_selection_sampler.due(timestamp_ms):
                    tower_hp_async_reader.submit(
                        frame,
                        tower_hp_candidate_batch,
                        timestamp_ms,
                    )
                elif tower_hp_sampler.due(timestamp_ms):
                    tower_hp_async_reader.submit(frame, tower_hp_crops, timestamp_ms)

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

            detected_bar_track_ids = smooth_result_boxes(
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
            # A tracker can eventually reuse a numeric ID. Drop its classification only
            # after the associated Kalman track has expired, not during a short gap.
            for stale_track_id in list(unit_classification_cache):
                if stale_track_id not in bar_kalman_filters:
                    unit_classification_cache.pop(stale_track_id, None)

            # Собираем данные о кадре
            frame_data = {
                "frame_number": frame_count, "timestamp_ms": timestamp_ms,
                "source_fps": fps, "processing_fps": process_fps,
                "num_objects": 0, "objects": [], "events": [],
            }
            frame_data["tower_hp"] = {tower_id: dict(state) for tower_id, state in tower_hp.items()}
            # Current-frame track_id -> mean (B, G, R), or None for an empty crop.
            bar_mean_colors: dict[int, tuple[float, float, float] | None] = {}
            field_objects: list[tuple[str, int | None, float, float]] = []
            elixir_centers: list[tuple[float, float]] = []
            bar_detection_count = 0
            predicted_unit_count = 0
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

                    # Use original pixels: battlefield may already contain annotations.
                    roi_left = max(0, min(battlefield.shape[1], math.floor(x1)))
                    roi_top = max(0, min(battlefield.shape[0], math.floor(y1)))
                    roi_right = max(0, min(battlefield.shape[1], math.ceil(x2)))
                    roi_bottom = max(0, min(battlefield.shape[0], math.ceil(y2)))
                    mean_color_bgr = None
                    if roi_right > roi_left and roi_bottom > roi_top:
                        color_roi = frame[
                            crop_y1 + roi_top : crop_y1 + roi_bottom,
                            crop_x1 + roi_left : crop_x1 + roi_right,
                        ]
                        mean_color_bgr = cv2.mean(color_roi)[:3]
                    bar_mean_colors[track_id] = mean_color_bgr

                    # Ищем совпадающие бары и левелы
                    is_unit_detection = (
                        class_id == 1
                        and SIZE_RESCTRICTIONS[0][0] <= x2 - x1 <= SIZE_RESCTRICTIONS[1][0]
                        and SIZE_RESCTRICTIONS[0][1] <= y2 - y1 <= SIZE_RESCTRICTIONS[1][1]
                    )
                    if is_unit_detection:

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
                            frame[crop_y1:crop_y2, crop_x1:crop_x2],
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
                        classification_state = unit_classification_cache.get(
                            int(track_id)
                        )
                        if classification_state is None and blue_rect is not None:
                            classification_state = UnitClassificationCacheEntry()
                            unit_classification_cache[int(track_id)] = classification_state

                        if (
                            blue_rect is not None
                            and classification_state is not None
                            and classification_state.classification_due(timestamp_ms)
                        ):
                            predictions = classify_crop(
                                classification_model,
                                blue_rect,
                                imgsz=224,
                                device="0",
                                top_k=1,
                                quantize=16,
                            )
                            class_name, confidence = predictions[0]
                            classification_state.add(
                                class_name,
                                confidence,
                                timestamp_ms,
                            )

                        cls_text = (
                            classification_state.display_text
                            if classification_state is not None
                            else "None"
                        )
                        unit_track_memories[int(track_id)] = UnitTrackMemory(
                            confidence=float(conf),
                            mean_color_bgr=mean_color_bgr,
                            classification_text=cls_text,
                        )
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
                        "predicted": False,
                        "mean_color_bgr": mean_color_bgr,
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

            predicted_unit_tracks = predict_missing_unit_tracks(
                bar_kalman_filters,
                bar_kalman_last_seen,
                unit_track_memories,
                detected_bar_track_ids,
                frame_count,
                battlefield.shape,
                fps,
                process_fps,
            )
            predicted_unit_count = len(predicted_unit_tracks)
            for predicted_track in predicted_unit_tracks:
                track_id = predicted_track.track_id
                x1, y1, x2, y2 = map(float, predicted_track.bbox)
                bar_width = float(bar_for_level[track_id])
                extended_right = min(float(battlefield.shape[1]), x2 + bar_width)
                unit_center_x = (x1 + extended_right) / 2
                unit_center_y = y2 + SIZE_OF_RECT / 2

                bar_mean_colors[track_id] = predicted_track.memory.mean_color_bgr
                field_objects.append(
                    ("blue_rect", track_id, unit_center_x, unit_center_y)
                )
                cv2.rectangle(
                    battlefield,
                    (round(x1), round(y1)),
                    (round(extended_right), round(y2)),
                    (0, 165, 255),
                    2,
                )
                predicted_left = round(unit_center_x - SIZE_OF_RECT / 2)
                cv2.rectangle(
                    battlefield,
                    (predicted_left, round(y2)),
                    (
                        predicted_left + SIZE_OF_RECT,
                        round(y2) + SIZE_OF_RECT,
                    ),
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    battlefield,
                    (
                        f"{track_id} {predicted_track.memory.classification_text} "
                        f"predicted {predicted_track.confidence:.2f}"
                    ),
                    (round(x1), max(20, round(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

                predicted_center_x = (x1 + x2) / 2
                predicted_center_y = (y1 + y2) / 2
                frame_data["objects"].append(
                    {
                        "track_id": track_id,
                        "class": model.names[1],
                        "confidence": predicted_track.confidence,
                        "predicted": True,
                        "missed_processed_frames": (
                            predicted_track.missed_processed_frames
                        ),
                        "mean_color_bgr": predicted_track.memory.mean_color_bgr,
                        "bbox": [x1, y1, x2, y2],
                        "center": [predicted_center_x, predicted_center_y],
                    }
                )
                object_history[track_id].append(
                    {
                        "frame": frame_count,
                        "center": (predicted_center_x, predicted_center_y),
                        "bbox": (x1, y1, x2, y2),
                        "predicted": True,
                    }
                )

            frame_data["num_objects"] += predicted_unit_count

            # Elixir detections never pass through classify_crop().
            if elixir_result.boxes is not None:
                for box in elixir_result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    track_id = int(box.id[0]) if box.id is not None else None
                    elixir_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    if battlefield_position_to_field_cell(
                        elixir_center,
                        battlefield.shape,
                    ) is not None:
                        elixir_centers.append(elixir_center)
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
            draw_tower_hp(output_frame, tower_hp_crops, tower_hp_states)

            elixir_bar_reading = recognize_elixir_bar(
                frame,
                (elixir_bar_x1, elixir_bar_y1, elixir_bar_x2, elixir_bar_y2),
            )
            frame_data["elixir_bar"] = elixir_bar_reading
            draw_elixir_bar_reading(
                output_frame,
                (elixir_bar_x1, elixir_bar_y1, elixir_bar_x2, elixir_bar_y2),
                elixir_bar_reading,
            )

            card_images, card_slots = split_cards_from_frame(
                frame,
                (cards_x1, cards_y1, cards_x2, cards_y2),
            )
            card_predictions = classify_cards(classification_cards_model, card_images)
            remember_card_predictions(card_histories, card_predictions, frame_count)
            draw_card_predictions(
                output_frame,
                card_predictions,
                card_slots,
                cards_y1,
            )

            frame_data["hand"] = [
                {"slot": slot, "card": name, "confidence": float(confidence)}
                for slot, (name, confidence) in enumerate(card_predictions, start=1)
            ]
            frame_data["elixir_centers"] = elixir_centers
            frame_data["units"] = []
            unit_boxes = {int(obj["track_id"]): obj for obj in frame_data["objects"]
                          if obj.get("track_id") is not None and obj["class"] == model.names[1]}
            for _, track_id, center_x, center_y in field_objects:
                cell = battlefield_position_to_field_cell((center_x, center_y), battlefield.shape)
                if cell is None:
                    continue
                cache = unit_classification_cache.get(track_id)
                color = bar_mean_colors.get(track_id)
                side = "unknown"
                if color is not None:
                    blue_score = color[0] * FIELD_COLOR_BLUE_WEIGHT
                    red_score = color[2] * FIELD_COLOR_RED_WEIGHT
                    if blue_score != red_score:
                        side = "ally" if blue_score > red_score else "enemy"
                obj = unit_boxes.get(track_id, {})
                frame_data["units"].append({
                    "track_id": int(track_id), "unit": cache.class_name if cache else "unknown",
                    "unit_confidence": float(cache.confidence) if cache else 0.0,
                    "side": side, "column": int(cell[0]), "row": int(cell[1]),
                    "center": [float(center_x), float(center_y)],
                    "bbox": [float(value) for value in obj.get("bbox", [])],
                    "detector_confidence": float(obj.get("confidence", 0.0)),
                    "predicted_by_kalman": bool(obj.get("predicted", False)),
                })

            confirmed_events = elixir_event_tracker.update(elixir_centers, frame_count)
            for event in confirmed_events:
                mean_position = event.mean_position
                cell = battlefield_position_to_field_cell(
                    mean_position,
                    battlefield.shape,
                )
                if cell is None:
                    continue
                card_match = resolve_played_card(
                    card_histories,
                    event.first_frame,
                    event.last_frame,
                    fps,
                    sampling_fps=process_fps,
                    return_details=True,
                )
                card_name, card_slot, card_confidence = (
                    card_match["card"], card_match["slot"], card_match["confidence"]
                )
                column, row = cell
                frame_data["events"].append({
                    **card_match,
                    "first_frame": event.first_frame, "confirmed_frame": frame_count,
                    "timestamp_ms": event.first_frame * 1000 / fps,
                    "confirmed_at_ms": timestamp_ms,
                    "action_timestamp_ms": (
                        card_match["empty_frame"] * 1000 / fps
                        if card_match["empty_frame"] is not None
                        else event.first_frame * 1000 / fps
                    ),
                    "column": column, "row": row,
                    "mean_position": list(mean_position),
                    "detections": len(event.positions),
                })
                card_event_markers.append(
                    (column, row, card_name, frame_count + card_event_marker_frames)
                )
                duration_ms = round(
                    (event.last_frame - event.first_frame) * 1000 / fps
                )
                battle_time_ms = round(event.first_frame * 1000 / fps)
                event_logger.info(
                    "Сыграна карта: %s; слот=%s; уверенность=%.2f; "
                    "клетка=(столбец=%d, строка=%d); "
                    "средняя позиция=(%.1f, %.1f); время_боя=%d мс; "
                    "длительность=%d мс; детекций=%d",
                    card_name,
                    card_slot if card_slot is not None else "?",
                    card_confidence,
                    column,
                    row,
                    mean_position[0],
                    mean_position[1],
                    battle_time_ms,
                    duration_ms,
                    len(event.positions),
                )

            if observation_callback is not None:
                observation_callback(frame_data)
            if annotation_callback is not None:
                annotation_callback(output_frame, frame_data)
            if display:
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
                    (
                        f"Bars: {bar_detection_count} | Predicted: {predicted_unit_count} "
                        f"| Elixir: {elixir_detection_count}"
                    ),
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                frame_height, frame_width = output_frame.shape[:2]
                new_width = max(1, round(frame_width * TRACKING_WINDOW_SCALE))
                new_height = max(1, round(frame_height * TRACKING_WINDOW_SCALE))
                resized_frame = cv2.resize(output_frame, (new_width, new_height))
                field_frame = draw_objects_on_field(
                    field_background,
                    field_objects,
                    battlefield.shape,
                    bar_mean_colors,
                )
                cv2.imshow("Tracking", resized_frame)
                draw_card_event_markers(field_frame, card_event_markers, frame_count)
                cv2.imshow(FIELD_WINDOW_NAME, field_frame)
                if show_battlefield:
                    battlefield_height, battlefield_width = battlefield.shape[:2]
                    debug_width = min(BATTLEFIELD_DEBUG_WIDTH, battlefield_width)
                    debug_height = max(
                        1,
                        round(battlefield_height * debug_width / battlefield_width),
                    )
                    battlefield_preview = cv2.resize(
                        battlefield,
                        (debug_width, debug_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow(BATTLEFIELD_DEBUG_WINDOW, battlefield_preview)
                if show_cards:
                    cards_preview = frame[cards_y1:cards_y2, cards_x1:cards_x2].copy()
                    local_card_slots = [
                        (slot_x1 - cards_x1, slot_x2 - cards_x1)
                        for slot_x1, slot_x2 in card_slots
                    ]
                    for slot_x1, slot_x2 in local_card_slots:
                        cv2.rectangle(
                            cards_preview,
                            (slot_x1, 0),
                            (slot_x2 - 1, cards_preview.shape[0] - 1),
                            (255, 255, 0),
                            2,
                        )
                    draw_card_predictions(
                        cards_preview,
                        card_predictions,
                        local_card_slots,
                        35,
                    )
                    cv2.imshow(CARDS_DEBUG_WINDOW, cards_preview)

            if out is not None:
                out.write(output_frame)
            if display and cv2.waitKey(1) & 0xFF == ord("q"):
                stopped_by_user = True
                break

        return {
            "source_fps": fps, "processing_fps": process_fps,
            "source_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "decoded_frame_count": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "processed_frames": processed_frames, "last_frame_index": last_frame_index,
            "stopped_by_user": stopped_by_user, "resolution": [width, height],
            "crops": {"battlefield": BATTLEFIELDS[video_size], "cards": CARDS[video_size],
                      "elixir_bar": ELIXIR_BAR[video_size],
                      "tower_hp": tower_hp_crop_candidates},
            "models": {"bars": str(MODEL_PATH), "elixir": str(ELIXIR_MODEL_PATH),
                       "units": str(CLASSIFICATION_MODEL_PATH), "cards": str(CARDS_MODEL_PATH)},
            "recognition_config": {
                name: value for name, value in globals().items()
                if name.startswith(("ELIXIR_", "CARD_", "UNIT_CLASSIFICATION_", "FIELD_COLOR_", "TOWER_HP_"))
                and isinstance(value, (int, float, bool, str, type(None)))
            },
            "vocabulary": {"units": classification_model.names, "cards": classification_cards_model.names},
        }

    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if tower_hp_async_reader is not None:
            tower_hp_async_reader.close()
        elif tower_hp_recognizer is not None:
            tower_hp_recognizer.close()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video_prediction()
