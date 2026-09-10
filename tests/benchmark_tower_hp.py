"""Read-only production-code OCR diagnostic; writes previews into a new temp folder."""
import ast
import json
import math
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch  # Match production import order to catch Paddle/PyTorch DLL conflicts.

from test_tower_hp import load_helpers


def main():
    v2 = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(v2))
    import model_paths

    ns = load_helpers()
    tree = ast.parse((v2 / "predict_video_kalman.py").read_text(encoding="utf-8-sig"))
    # Reuse actual production OCR settings, including any user changes.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("TOWER_HP_"):
                    try:
                        ns[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    started = time.perf_counter()
    recognizer = ns["create_tower_hp_recognizer"]()
    print("PaddleOCR startup seconds:", round(time.perf_counter() - started, 2), flush=True)
    video = v2 / "screenshots/IMG_1357.mp4"
    output = Path(tempfile.mkdtemp(prefix="hp_diagnostic_", dir=v2 / "screenshots"))
    print("Preview directory:", output, flush=True)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Video:", size, fps, flush=True)
    crops = {name: getattr(model_paths, constant)[size] for name, constant in (
        ("ally_1", "TOWER_HP_1"), ("ally_2", "TOWER_HP_2"),
        ("enemy_1", "TOWER_HP_ENEMY_1"), ("enemy_2", "TOWER_HP_ENEMY_2"))}
    states = {name: ns["TowerHPState"]() for name in crops}
    timings = []
    for seconds in (0, 0.1, 0.2, 1, 3, 8):
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
        ok, frame = capture.read()
        if not ok:
            continue
        readings, previews = ns["read_tower_hp"](frame, crops, recognizer)
        print(seconds, json.dumps(readings), flush=True)
        timings.append(next(iter(readings.values()))["batch_ms"])
        for name, reading in readings.items():
            states[name].update(reading, seconds * 1000)
        print("Confirmed HP:", {name: state.hp for name, state in states.items()}, flush=True)
        panels = []
        for name, (x1, y1, x2, y2) in crops.items():
            raw = cv2.resize(frame[y1:y2, x1:x2], (435, 246))
            processed = cv2.resize(previews[name], (435, 246))
            panel = np.vstack([np.full((30, 870, 3), 255, np.uint8), np.hstack([raw, processed])])
            cv2.putText(panel, name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
            panels.append(panel)
        cv2.imwrite(str(output / f"crops_{seconds}s.jpg"), np.vstack(panels))
        if seconds == 1:
            cv2.imwrite(str(output / "frame_1s.jpg"), frame)
    capture.release()
    recognizer.close()
    print("Mean batch ms (excluding first warm-up batch):", mean(timings[1:]) if len(timings) > 1 else None)
    log = v2 / "screenshots/output_tracked_calman.tower_hp.jsonl"
    rounds = []
    if log.is_file():
        with log.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("type") == "run":
                    rounds = []
                elif record.get("type") == "observation":
                    rounds.append(record)
    if rounds:
        readings = [r for record in rounds for r in record["readings"].values()]
        print("Latest log run:", len(rounds), "rounds; errors:", Counter(r["error"] for r in readings))
        print("Mean OCR ms/crop:", mean(r["ocr_ms"] for r in readings))
        print("Mean OCR ms/round:", mean(sum(r["ocr_ms"] for r in record["readings"].values()) for record in rounds))


if __name__ == "__main__":
    main()
