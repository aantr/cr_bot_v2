import numpy as np
from paddleocr import PaddleOCR

# Инициализируйте OCR
ocr = PaddleOCR(device='gpu')  # или device='cpu'

# Создайте тестовое изображение с текстом
test_image = np.full((50, 200, 3), 255, dtype=np.uint8)  # Белое изображение

# Попробуйте распознать
try:
    result = ocr.predict(input=[test_image])
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")
    
    for i, res in enumerate(result):
        print(f"\nResult {i}:")
        print(f"  Type: {type(res)}")
        print(f"  Value: {res}")
        
        if isinstance(res, dict):
            print(f"  Keys: {list(res.keys())}")
            for key, value in res.items():
                print(f"    {key}: {type(value)} = {value}")
        
        elif hasattr(res, '__dict__'):
            print(f"  Attributes: {vars(res)}")
        
        elif isinstance(res, list):
            print(f"  List with {len(res)} elements")
            for j, item in enumerate(res):
                print(f"    [{j}]: {type(item)} = {item}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()