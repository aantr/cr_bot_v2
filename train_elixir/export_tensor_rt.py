import sys
from pathlib import Path

from ultralytics import YOLO

V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from model_paths import ELIXIR_DETECTION_WEIGHTS_PATH


def main() -> None:
    model = YOLO(str(ELIXIR_DETECTION_WEIGHTS_PATH))

    model.export(
        format="engine",
        imgsz=1280,
        quantize=16,
        device=0,
    )


if __name__ == "__main__":
    main()
