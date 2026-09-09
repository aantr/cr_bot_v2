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

CLASSIFICATION_CARDS_MODEL_PATH = (
    V2_DIR / "runs_cards/yolo26s_cls_cards_224-2/weights/best.pt"
)

BATTLEFIELDS = {
    (832, 1811): (0, 100, 832, 1450),
    (1206, 2622): (50, 360, 1180, 1990),
    (882, 1920): (0, 200, 882, 1920 - 200),
    (576, 1280): (0, 100, 576, 1280 - 100),
}

CARDS = {
    (1206, 2622): (260, 2154, 1168, 2466),
    (882, 1920): (190, 1578, 854, 1805),
}

ELIXIR_BAR = {
    # todo
    (1206, 2622): (260, 2154, 1168, 2466),
}

TOWER_HP_1 = {
    (1206, 2622): (196, 1618, 322, 1685),
}

TOWER_HP_2 = {
    (1206, 2622): (900, 1618, 1045, 1700),
}

TOWER_HP_ENEMY_1 = {
    (1206, 2622): (196, 500, 322, 560),
}

TOWER_HP_ENEMY_2 = {
    (1206, 2622): (900, 500, 1045, 560),
}

