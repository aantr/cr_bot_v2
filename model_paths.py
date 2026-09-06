from pathlib import Path


V2_DIR = Path(__file__).resolve().parent

DETECTION_WEIGHTS_PATH = (
    V2_DIR / "runs/detect/runs_game/yolo26s_p2_1280-2/weights/best.pt"
)

DETECTION_ENGINE_PATH = (
    V2_DIR / "runs/detect/runs_game/yolo26s_p2_1280-2/weights/best.engine"
)

ELIXIR_DETECTION_WEIGHTS_PATH = (
    V2_DIR / "runs/detect/runs_elixir/yolo26s_p2_elixir_1280/weights/best.pt"
)

ELIXIR_DETECTION_ENGINE_PATH = (
    V2_DIR / "runs/detect/runs_elixir/yolo26s_p2_elixir_1280/weights/best.engine"
)

CLASSIFICATION_MODEL_PATH = (
    V2_DIR / "runs_game/yolo26s_cls_224-3/weights/best.pt"
)

BATTLEFIELDS = {
    (832, 1811): (0, 100, 832, 1450),
    (1206, 2622): (50, 360, 1180, 1990),
    (882, 1920): (0, 200, 882, 1920 - 200),
    (576, 1280): (0, 100, 576, 1280 - 100),
}

CARDS = {
    (1206, 2622): (260, 2154, 1168, 2466),
}

ELIXIR_BAR = {
    # todo
    (1206, 2622): (260, 2154, 1168, 2466),
}

