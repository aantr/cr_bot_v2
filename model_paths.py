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
    (1206, 2622): (50, 360, 1180, 1990),
    (882, 1920): (32, 258, 860, 1479),
    (886, 1920): (27, 275, 867, 1483),
}

CARDS = {
    (1206, 2622): (260, 2154, 1168, 2466),
    (882, 1920): (200, 1574, 858, 1816),
    (886, 1920): (202, 1572, 850, 1805),
}

ELIXIR_BAR = {
    (1206, 2622): (309, 2488, 1162, 2540),
    (882, 1920): (204, 1821, 850, 1861),
    (886, 1920): (211, 1819, 852, 1856),
}

TOWER_HP_1 = {
    (1206, 2622): [
        (196, 1659, 300 + 50, 1703),
        (196, 1659 - 30, 300 + 50, 1703 - 30),
    ],
    (882, 1920): [
        (147, 1216, 248, 1248)
    ],
    (886, 1920): [
        (151, 1218, 246, 1251)
    ],
}

TOWER_HP_2 = {
    (1206, 2622): [
        (903, 1659, 1007 + 50, 1703),
        (903, 1659 - 30, 1007 + 50, 1703 - 30),
    ],
    (882, 1920): [
        (661, 1218, 762, 1248)
    ],
    (886, 1920): [
        (661, 1218, 758, 1251)
    ],
}

TOWER_HP_ENEMY_1 = {
    (1206, 2622): [
        (204, 469, 308 + 50, 513),
        (204, 469 + 30, 308 + 50, 513 + 30),
    ],
    (882, 1920): [
        (149, 364, 250, 399)
    ],
    (886, 1920): [
        (149, 330, 248, 361)
    ],
}

TOWER_HP_ENEMY_2 = {
    (1206, 2622): [
        (904, 469, 1008 + 50, 513),
        (904, 469 + 30, 1008 + 50, 513 + 30),
    ],
    (882, 1920): [
        (659, 362, 766, 402)
    ],
    (886, 1920): [
        (661, 326, 768, 361)
    ],
}
