"""
test_gpu_ocr_v2.py - Тест OCR с GPU (PaddleOCR 2.7.3)
"""
import numpy as np
import cv2
import time
import sys
import os
from pathlib import Path

# Добавьте cuDNN в PATH
if sys.platform == 'win32':
    cudnn_dir = str(Path(sys.prefix) / 'Lib/site-packages/nvidia/cudnn/bin')
    if os.path.exists(cudnn_dir):
        os.add_dll_directory(cudnn_dir)

from paddleocr import PaddleOCR

# Создайте тестовое изображение
img = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(img, "42", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
cv2.imwrite("test_gpu_v2.png", img)

# Инициализируйте OCR с GPU
print("Creating OCR...")
ocr = PaddleOCR(
    use_gpu=True,
    lang='en',
    use_angle_cls=False,
    show_log=False,
)

# Распознайте (используйте ocr(), а не predict())
print("Recognizing...")
start = time.perf_counter()
result = ocr.ocr(img, cls=False)
elapsed = (time.perf_counter() - start) * 1000

print(f"\nTime: {elapsed:.2f}ms")
print(f"Result: {result}")

# Обработка результата
if result and result[0]:
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]
        print(f"\nText: {text}")
        print(f"Confidence: {confidence}")
else:
    print("\nNo text detected")
