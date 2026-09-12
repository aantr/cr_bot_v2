"""Fake iPhone only: no connection, no actual phone taps or GUI."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from run_bot_iphone import IPhoneFrameSource, TapExecutor, TapPair, make_parser, tap_points, run


class Phone:
    def __init__(self):
        self.image = np.zeros((320, 180, 3), dtype=np.uint8)
        self.frame_age_ms = 0.
        self.taps = []
        self.on_tap = None
        self.on_frame = None

    def get_screen(self, **kwargs):
        if self.on_frame:
            self.on_frame()
        return self.image.copy() if kwargs.get("copy") else self.image

    def send_tap(self, x, y):
        self.taps.append((x, y))
        if self.on_tap:
            self.on_tap()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class IPhoneBotTests(unittest.TestCase):
    def setUp(self):
        self.phone = Phone()
        self.action = {"type": "play", "card": "knight", "slot": 1, "row": 16, "column": 9,
                       "battle_id": "test", "observation_timestamp_ms": 0.}
        self.pair = TapPair((25., 290.), (85., 155.), (180, 320), 10.)
        self.executor = TapExecutor(self.phone, execute=True, clock=lambda: 10., tap_gap_ms=0.)
        self.addCleanup(self.executor.close)

    def submit(self, action=None, pair=None):
        with redirect_stdout(io.StringIO()):
            return self.executor.submit(action or self.action, pair or self.pair)

    def test_coordinates_original_pixels_slots_and_field_corners(self):
        card, cell = tap_points(self.action, (180, 320), (20, 280, 180, 320), (0, 0, 180, 256))
        self.assertEqual(card, (40., 300.))
        self.assertEqual(cell, (85., 124.))
        action = {**self.action, "slot": 4, "row": 32, "column": 18}
        self.assertEqual(tap_points(action, (180, 320), (20, 280, 180, 320), (0, 0, 180, 256)),
                         ((160., 300.), (175., 252.)))
        for update in ({"slot": 0}, {"row": 33}, {"column": True}):
            with self.assertRaises(ValueError):
                tap_points({**action, **update}, (180, 320), (20, 280, 180, 320), (0, 0, 180, 256))
        with self.assertRaises(ValueError):
            tap_points(action, (180, 320), (0, 280, 181, 320), (0, 0, 180, 256))

    def test_pair_order_wait_and_duplicate(self):
        self.assertFalse(self.submit({**self.action, "type": "noop"}))
        self.assertTrue(self.submit())
        self.assertTrue(self.executor.future.result(timeout=2))
        with redirect_stdout(io.StringIO()):
            self.executor.poll()
        self.assertEqual(self.phone.taps, [self.pair.card, self.pair.cell])
        self.assertFalse(self.submit())
        self.assertTrue(self.submit({**self.action, "observation_timestamp_ms": 200.}))
        self.executor.future.result(timeout=2)
        self.assertEqual(self.phone.taps, [self.pair.card, self.pair.cell] * 2)

    def test_dry_run_never_sends(self):
        self.executor.execute = False
        self.assertTrue(self.submit())
        self.assertFalse(self.executor.busy)
        self.assertEqual(self.phone.taps, [])
        args = make_parser().parse_args(["--checkpoint", "unused.pt"])
        self.assertFalse(args.execute)

    def test_no_backlog_while_busy_and_stop_cancels_second_tap(self):
        entered, release = threading.Event(), threading.Event()
        def block():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
        self.phone.on_tap = block
        try:
            self.assertTrue(self.submit())
            self.assertTrue(entered.wait(2))
            self.assertFalse(self.submit({**self.action, "observation_timestamp_ms": 200.}))
            self.executor.stop.set()
        finally:
            release.set()
        self.assertFalse(self.executor.future.result(timeout=2))
        self.assertEqual(self.phone.taps, [self.pair.card])

    def test_failure_never_retries_or_sends_placement(self):
        def fail():
            raise TimeoutError("reply lost")
        self.phone.on_tap = fail
        self.submit()
        with self.assertRaises(TimeoutError):
            self.executor.future.result(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "without retry"):
            self.executor.poll()
        self.assertTrue(self.executor.stop.is_set())
        self.assertEqual(self.phone.taps, [self.pair.card])

    def test_stale_observation_and_stale_stream(self):
        stale = TapPair(self.pair.card, self.pair.cell, self.pair.size, 0.)
        self.assertFalse(self.submit(pair=stale))
        self.assertEqual(self.phone.taps, [])
        self.phone.frame_age_ms = 2000.
        self.submit({**self.action, "observation_timestamp_ms": 200.})
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.executor.future.result(timeout=2)
        self.assertEqual(self.phone.taps, [])

    def test_resolution_change_between_taps_stops(self):
        self.phone.on_tap = lambda: setattr(self.phone, "image", np.zeros((180, 320, 3), dtype=np.uint8))
        self.submit()
        with self.assertRaisesRegex(RuntimeError, "resolution"):
            self.executor.future.result(timeout=2)
        self.assertEqual(self.phone.taps, [self.pair.card])

    def test_pause_does_not_send(self):
        self.executor.paused.set()
        self.assertFalse(self.submit())
        self.assertEqual(self.phone.taps, [])

    def test_live_indices_use_elapsed_time_not_number_processed(self):
        now = [10.]
        source = IPhoneFrameSource(self.phone, clock=lambda: now[0])
        self.addCleanup(source.release)
        frames = source.iter_frames(30.)
        first, _ = next(frames)
        now[0] += 2.  # Expensive recognition: 2 seconds, not one frame.
        second, _ = next(frames)
        self.assertEqual((first, second), (0, 60))
        self.assertEqual(source.last_received_at, 12.)

    def test_live_timeout_and_orientation_fail_closed(self):
        source = IPhoneFrameSource(self.phone)
        self.addCleanup(source.release)
        self.phone.image = np.zeros((180, 320, 3), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "orientation"):
            next(source.iter_frames(30.))
        def fail():
            raise TimeoutError("missing frame")
        self.phone.on_frame = fail
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            next(source.iter_frames(30.))

    def test_help_does_not_connect(self):
        result = subprocess.run([sys.executable, str(V2_DIR / "run_bot_iphone.py"), "--help"],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--execute", result.stdout)

    def test_full_runner_with_fake_perception_and_phone(self):
        from test_build_trajectories import observation
        from test_predict_video_actions import FakePredictor
        predictor = FakePredictor()
        predictor.history_mode = "observations_only"
        completed = threading.Event()
        self.phone.on_tap = lambda: completed.set() if len(self.phone.taps) == 2 else None

        def perception(**kwargs):
            source = kwargs["frame_source"]
            index, frame = next(source.iter_frames(30.))
            self.assertEqual(index, 0)
            self.assertFalse(kwargs["write_video"])
            self.assertFalse(kwargs["write_logs"])
            kwargs["observation_callback"](observation(0))
            self.assertTrue(completed.wait(2))
            kwargs["annotation_callback"](frame, observation(0))
            kwargs["key_callback"](ord(" "))  # Pause: measured history still updates.
            kwargs["observation_callback"](observation(200))
            source.release()

        fake_perception = SimpleNamespace(run_video_prediction=perception, show_cards=False, show_battlefield=False)
        args = make_parser().parse_args(["--checkpoint", "not-loaded.pt", "--execute", "--tap-gap-ms", ".1"])
        with (patch("run_bot_iphone.create_remote", return_value=self.phone) as connect,
              patch("offline_rl.object_gru.predict_action.ObjectGRUPredictor", return_value=predictor),
              patch("model_paths.CARDS", {(180, 320): (20, 280, 180, 320)}),
              patch("model_paths.BATTLEFIELDS", {(180, 320): (0, 0, 180, 256)}),
              patch.dict(sys.modules, {"predict_video_kalman": fake_perception}),
              redirect_stdout(io.StringIO())):
            run(args)
        connect.assert_called_once()
        self.assertEqual(self.phone.taps, [(40., 300.), (85., 124.)])
        self.assertEqual(predictor.history[-1][0]["timestamp_ms"], 200)


if __name__ == "__main__":
    unittest.main()
