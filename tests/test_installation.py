import paddle
import paddleocr
import cv2
import numpy as np

print(f"PaddlePaddle: {paddle.__version__}")
print(f"PaddleOCR: {paddleocr.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"GPU compiled: {paddle.is_compiled_with_cuda()}")

if paddle.is_compiled_with_cuda():
    print(f"GPU count: {paddle.device.cuda.device_count()}")
    if paddle.device.cuda.device_count() > 0:
        paddle.set_device('gpu:0')
        x = paddle.randn([3, 3])
        print(f"✅ GPU working! Device: {x.place}")
        print(f"GPU Name: {paddle.device.cuda.get_device_name(0)}")


