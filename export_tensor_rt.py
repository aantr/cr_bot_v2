from ultralytics import YOLO

from model_paths import DETECTION_WEIGHTS_PATH


model = YOLO(str(DETECTION_WEIGHTS_PATH))

model.export(
    format="engine",
    imgsz=1280,    
    quantize=16,
    device=0
)
