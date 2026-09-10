"""Test OCR helpers without loading video detectors or PaddleOCR weights."""
import ast
import math
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np


def load_helpers():
    path = Path(__file__).resolve().parents[1] / "predict_video_kalman.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {"prepare_tower_hp_crop", "parse_tower_hp_data", "TowerHPState",
             "TowerHPSampler", "read_tower_hp", "recognize_tower_hp_previews",
             "TowerHPAsyncReader", "draw_tower_hp", "create_tower_hp_recognizer"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    ns = dict(cv2=cv2, np=np, math=math, time=time, dataclass=dataclass,
              Future=Future, ThreadPoolExecutor=ThreadPoolExecutor,
              TOWER_HP_MIN_CONFIDENCE=0.7, TOWER_HP_CONFIRM_READINGS=2,
              TOWER_HP_SCALE=1.0,
              TOWER_HP_MAX_VALUES={"ally_1": 10000, "enemy_1": 10000})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


class TowerHPTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_helpers()

    def parse(self, text, confidence=95):
        return self.ns["parse_tower_hp_data"](
            {"rec_text": text, "rec_score": confidence / 100}, 10000)

    def test_parse(self):
        self.assertEqual(self.parse("2534")["value"], 2534)
        self.assertEqual(self.parse("0")["value"], 0)
        for text, conf, error in [("", 95, "no_text"), ("25O4", 95, "not_single_number"),
                                  ("-1", 95, "not_single_number"),
                                  ("99999", 95, "out_of_range"),
                                  ("123", 40, "low_confidence")]:
            result = self.parse(text, conf)
            self.assertIsNone(result["value"])
            self.assertEqual(result["error"], error)
        result = self.ns["parse_tower_hp_data"]({"rec_text": "12 345", "rec_score": .99}, 10000)
        self.assertEqual(result["error"], "not_single_number")
        with self.assertRaises(ValueError):
            self.parse("123", float("nan"))

    def test_confirmation_and_first_observation_time(self):
        state = self.ns["TowerHPState"]()
        self.assertFalse(state.update(self.parse("2534"), 100))
        self.assertIsNone(state.hp)
        self.assertTrue(state.update(self.parse("2534", 80), 200))
        self.assertEqual(state.hp, 2534)
        self.assertEqual(state.observed_at_ms, 100)
        self.assertEqual(state.confirmed_at_ms, 200)
        self.assertEqual(state.confidence, .8)
        self.assertFalse(state.update(self.parse("2534"), 300))
        self.assertEqual(state.observed_at_ms, 100)
        self.assertEqual(state.last_seen_at_ms, 300)

    def test_missing_reading_breaks_confirmation_and_keeps_hp(self):
        state = self.ns["TowerHPState"]()
        for timestamp in (0, 100):
            state.update(self.parse("2534"), timestamp)
        old = state.snapshot()
        state.update(self.parse("2400"), 200)
        state.update(self.parse(""), 300)
        self.assertEqual(state.hp, 2534)
        self.assertTrue(state.stale)
        self.assertFalse(state.update(self.parse("2400"), 400))
        self.assertTrue(state.update(self.parse("2400"), 500))
        self.assertEqual(state.observed_at_ms, 400)
        self.assertEqual(old["hp"], 2534)
        self.assertFalse(state.stale)
        state.update(self.parse(""), 600)
        self.assertEqual(state.hp, 2400)  # No text must never mean zero HP.
        self.assertFalse(state.update(self.parse("0"), 700))
        self.assertTrue(state.update(self.parse("0"), 800))
        self.assertEqual(state.hp, 0)

    def test_flicker_and_towers_are_independent(self):
        state = self.ns["TowerHPState"]()
        other = self.ns["TowerHPState"]()
        for timestamp, value in enumerate(("1000", "100", "1000", "100")):
            self.assertFalse(state.update(self.parse(value), timestamp * 100))
        self.assertIsNone(state.hp)
        self.assertIsNone(other.pending_hp)

    def test_video_time_sampling(self):
        for source_fps, process_fps in [(60, 30), (59.76354679802956, 30), (60, 20), (60, 5)]:
            sampler = self.ns["TowerHPSampler"](1000 / min(10, process_fps))
            calls = []
            for i in range(int(process_fps)):
                timestamp = math.ceil(i * source_fps / process_fps - 1e-9) * 1000 / source_fps
                if sampler.due(timestamp):
                    calls.append(timestamp)
                    self.assertFalse(sampler.due(timestamp))
            self.assertEqual(len(calls), min(10, process_fps))

    def test_crop_and_ocr_failure_isolation(self):
        frame = np.zeros((30, 60, 3), np.uint8)
        cv2.putText(frame, "12", (1, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        original = frame.copy()
        ocr = Mock(return_value=[{"rec_text": "1234", "rec_score": .95},
                                 {"rec_text": "oops", "rec_score": .95}])
        recognizer = SimpleNamespace(predict=ocr)
        crops = {"ally_1": (0, 0, 30, 30), "enemy_1": (30, 0, 60, 30)}
        readings, previews = self.ns["read_tower_hp"](frame, crops, recognizer)
        self.assertEqual(readings["ally_1"]["value"], 1234)
        self.assertIsNone(readings["enemy_1"]["value"])
        self.assertEqual(readings["enemy_1"]["error"], "not_single_number")
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(ocr.call_args.kwargs["batch_size"], 2)
        self.assertEqual(previews["ally_1"].shape, (30, 30, 3))
        self.assertTrue(np.array_equal(previews["ally_1"], frame[:, :30]))
        self.assertTrue(np.array_equal(frame, original))

    def test_batch_error_and_count_mismatch_do_not_shift_towers(self):
        frame = np.zeros((20, 40, 3), np.uint8)
        crops = {"ally_1": (0, 0, 20, 20), "enemy_1": (20, 0, 40, 20)}
        for predict in (Mock(side_effect=RuntimeError("failed")),
                        Mock(return_value=[{"rec_text": "1234", "rec_score": .95}])):
            readings, _ = self.ns["read_tower_hp"](frame, crops, SimpleNamespace(predict=predict))
            self.assertEqual(set(readings), set(crops))
            self.assertTrue(all(r["value"] is None for r in readings.values()))
            self.assertTrue(all(r["error"].startswith("ocr_error:") for r in readings.values()))

    def test_empty_batch_does_not_call_model(self):
        recognizer = Mock()
        self.assertEqual(self.ns["read_tower_hp"](np.zeros((10, 10, 3), np.uint8), {}, recognizer), ({}, {}))
        recognizer.predict.assert_not_called()

    def test_async_reader_skips_while_busy_and_keeps_sample_timestamp(self):
        started = threading.Event()
        release = threading.Event()

        class Recognizer:
            is_running = True

            def predict(self, *, input, batch_size):
                started.set()
                release.wait(2)
                return [{"rec_text": "1234", "rec_score": .99} for _ in input]

            def close(self):
                self.is_running = False
                release.set()

        reader = self.ns["TowerHPAsyncReader"](Recognizer())
        frame = np.zeros((20, 20, 3), np.uint8)
        crops = {"ally_1": (0, 0, 20, 20)}
        try:
            self.assertTrue(reader.submit(frame, crops, 125.0))
            self.assertTrue(started.wait(1))
            self.assertFalse(reader.submit(frame, crops, 225.0))
            self.assertIsNone(reader.poll())
            release.set()
            result = None
            for _ in range(100):
                result = reader.poll()
                if result is not None:
                    break
                time.sleep(.005)
            self.assertIsNotNone(result)
            timestamp, readings, _ = result
            self.assertEqual(timestamp, 125.0)
            self.assertEqual(readings["ally_1"]["value"], 1234)
        finally:
            reader.close()


if __name__ == "__main__":
    unittest.main()
