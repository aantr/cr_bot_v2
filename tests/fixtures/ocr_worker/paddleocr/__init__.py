"""Subprocess protocol test double, only enabled by the tests' PYTHONPATH."""
import os
import sys
import time


class TextRecognition:
    def __init__(self, **config):
        assert "torch" not in sys.modules
        if config["model_name"] == "fail":
            raise RuntimeError("test initialization failure")
        print("library stdout must not corrupt JSON")
        os.write(1, b"native stdout must not corrupt JSON\n")

    def predict(self, *, input, batch_size):
        value = int(input[0][0, 0, 0])
        if value == 249:
            raise RuntimeError("test batch failure")
        if value == 250:
            time.sleep(10)
        if value == 251:
            os._exit(7)
        return [{"rec_text": str(int(crop[0, 0, 0])), "rec_score": .99} for crop in input]
