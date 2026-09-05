from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from model_paths import CLASSIFICATION_MODEL_PATH

DEFAULT_MODEL = CLASSIFICATION_MODEL_PATH


def get_top_predictions(result, top_k: int) -> list[tuple[str, float]]:
    """Convert an Ultralytics classification result to (class, confidence)."""
    if result.probs is None:
        raise RuntimeError("The loaded model did not return classification probabilities")

    probabilities = result.probs.data.detach().cpu()
    count = min(top_k, probabilities.numel())
    confidences, class_ids = probabilities.topk(count)
    return [
        (result.names[int(class_id)], float(confidence))
        for class_id, confidence in zip(class_ids, confidences)
    ]


def classify_crop(
    model: YOLO,
    blue_rect: np.ndarray,
    *,
    imgsz: int = 224,
    device: str = "0",
    top_k: int = 3,
    quantize: int | None = 16,
) -> list[tuple[str, float]]:
    """Classify one OpenCV BGR crop, for example blue_rect from predict_video.py."""
    if blue_rect.size == 0:
        raise ValueError("blue_rect is empty")

    result = model.predict(
        source=blue_rect,
        imgsz=imgsz,
        device=device,
        quantize=quantize,
        verbose=False,
    )[0]
    return get_top_predictions(result, top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained Ultralytics YOLO classification model."
    )
    parser.add_argument("source", type=Path, help="Image or directory with images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0", help="For example: 0 or cpu.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--quantize",
        type=int,
        choices=(16, 32),
        default=16,
        help="Inference precision: 16 (FP16) or 32 (FP32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        device=args.device,
        quantize=args.quantize,
        stream=True,
        verbose=False,
    )

    for result in results:
        predictions = get_top_predictions(result, args.top_k)
        print(result.path)
        for position, (class_name, confidence) in enumerate(predictions, start=1):
            print(f"  {position}. {class_name}: {confidence:.4f}")


if __name__ == "__main__":
    main()
