from pathlib import Path

from ultralytics import YOLO


V2_DIR = Path(__file__).resolve().parents[1]
DATA_CONFIG = Path(__file__).resolve().with_name("bar-dataset.yaml")
PRETRAINED_WEIGHTS = V2_DIR / "yolo26s.pt"
PROJECT_DIR = V2_DIR / "runs" / "detect" / "runs_game"


def main():
    # P2 architecture:
    # P2/4
    # P3/8
    # P4/16
    # P5/32
    #
    # "s" выбирается автоматически из имени yolo26s-p2.yaml.
    model = YOLO("yolo26s-p2.yaml")

    # Переносим совместимые pretrained-веса
    # из стандартного YOLO26s.
    #
    # Новые P2-слои останутся случайно инициализированными,
    # остальные совместимые веса будут перенесены.
    model.load(str(PRETRAINED_WEIGHTS))

    model.train(
        data=str(DATA_CONFIG),

        # Для мелких объектов я бы начал с 1280.
        imgsz=1280,

        epochs=150,

        # начни с 8, потом попробуй 12/16,
        # если хватает VRAM.
        batch=8,

        device=0,

        workers=8,

        # Не останавливать обучение слишком рано
        patience=35,

        # кешировать датасет, если позволяет RAM
        cache="disk",

        # mixed precision
        amp=True,

        # augmentation
        hsv_h=0.010,
        hsv_s=0.40,
        hsv_v=0.30,

        translate=0.08,
        scale=0.40,

        fliplr=0.0,
        flipud=0.0,

        # Для игры геометрические деформации
        # обычно лучше держать небольшими.
        degrees=0.0,
        shear=0.0,
        perspective=0.0,

        # Mosaic особенно полезен для small objects,
        # но ближе к концу отключаем.
        mosaic=1.0,
        close_mosaic=15,

        project=str(PROJECT_DIR),
        name="yolo26s_p2_1280",

        save=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
