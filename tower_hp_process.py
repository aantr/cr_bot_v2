"""Persistent PaddleOCR subprocess. Never import Paddle into the YOLO process."""
import atexit
import base64
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading


class TowerHPProcess:
    def __init__(self, *, model_name, model_dir=None, device="cpu", cpu_threads=4,
                 enable_mkldnn=False, startup_timeout=180.0, request_timeout=15.0):
        if any(not math.isfinite(t) or t <= 0 for t in (startup_timeout, request_timeout)):
            raise ValueError("OCR timeouts must be finite positive numbers")
        self.request_timeout = request_timeout
        self._messages = queue.Queue()
        self._closed = False
        # Execute a separate entry point, not multiprocessing's re-import of the
        # video script (which would load torch and all detectors in the child).
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._reader = threading.Thread(target=self._read_messages, daemon=True)
        self._reader.start()
        atexit.register(self.close)
        try:
            self._send({"model_name": model_name,
                        "model_dir": str(model_dir) if model_dir is not None else None,
                        "device": device, "cpu_threads": cpu_threads,
                        "enable_mkldnn": enable_mkldnn})
            ready = self._receive(startup_timeout)
            if not ready.get("ready"):
                raise RuntimeError("OCR worker did not finish initialization")
            self.pid = ready["pid"]
        except BaseException:
            self.close()
            raise

    @property
    def is_running(self):
        return not self._closed and self._process.poll() is None

    def _read_messages(self):
        try:
            for line in self._process.stdout:
                self._messages.put(line)
        finally:
            self._messages.put(None)

    def _send(self, message):
        if not self.is_running:
            raise RuntimeError("PaddleOCR worker is not running")
        try:
            self._process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self._process.stdin.flush()
        except (OSError, ValueError) as exc:
            self.close()
            raise RuntimeError("Cannot send a batch to PaddleOCR worker") from exc

    def _receive(self, timeout):
        try:
            line = self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            # A late reply must never be mistaken for the next frame's result.
            self.close()
            raise RuntimeError(f"PaddleOCR worker timed out after {timeout:g}s") from exc
        if line is None:
            self.close()
            raise RuntimeError("PaddleOCR worker exited; see its stderr above")
        try:
            message = json.loads(line)
        except ValueError as exc:
            self.close()
            raise RuntimeError("Invalid PaddleOCR worker response") from exc
        if "error" in message:
            raise RuntimeError(message["error"])
        return message

    def predict(self, *, input, batch_size):
        import numpy as np

        if not input:
            return []
        images = []
        for crop in input:
            if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or not crop.size:
                raise ValueError("OCR expects nonempty uint8 BGR crops")
            images.append({"shape": crop.shape,
                           "pixels": base64.b64encode(crop.tobytes()).decode("ascii")})
        self._send({"images": images, "batch_size": batch_size})
        results = self._receive(self.request_timeout)["results"]
        if len(results) != len(images):
            raise RuntimeError("PaddleOCR returned the wrong number of results")
        return results

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        # EOF lets an idle worker exit normally; terminate if inference is stuck.
        try:
            self._process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._reader.join(timeout=2)
        self._process.stdout.close()


def worker_main():
    # Keep protocol output separate even from native-library stdout messages.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def reply(message):
        protocol.write(json.dumps(message, allow_nan=False) + "\n")
        protocol.flush()

    try:
        config = json.loads(sys.stdin.readline())
        import numpy as np
        from paddleocr import TextRecognition

        model = TextRecognition(**config)
        reply({"ready": True, "pid": os.getpid()})
    except Exception as exc:
        reply({"error": f"PaddleOCR initialization failed: {type(exc).__name__}: {exc}"})
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            images = [np.frombuffer(base64.b64decode(item["pixels"], validate=True),
                                    dtype=np.uint8).reshape(item["shape"]).copy()
                      for item in request["images"]]
            results = list(model.predict(input=images, batch_size=request["batch_size"]))
            # TextRecognition returns singular fields, unlike the full OCR pipeline.
            normalized = [{"rec_text": str(result["rec_text"]),
                           "rec_score": float(result["rec_score"])} for result in results]
            if len(normalized) != len(images):
                raise ValueError("OCR batch result count mismatch")
            reply({"results": normalized})
        except Exception as exc:
            reply({"error": f"PaddleOCR batch failed: {type(exc).__name__}: {exc}"})


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    worker_main()
