"""iPhone screen -> shared Object GRU perception/policy -> card tap, then field tap.

From v2 (preview without taps):
  python run_bot_iphone.py --mac-ip 10.10.10.1 --checkpoint runs/offline_rl/object_gru_weight1/best_play.pt --device 0
Add --execute to enable real taps. Q/Esc stops; Space pauses/resumes decisions.
Start on the battlefield at battle start, stop before menus/the next battle.
No automatic battle/menu detection, navigation, or purchase actions are provided.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

import cv2

V2_DIR = Path(__file__).resolve().parent


class StopBot(Exception):
    """Normal keyboard/time-limit stop, distinct from a control failure."""


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def validate_crop(crop, size, name):
    if not isinstance(crop, (tuple, list)) or len(crop) != 4:
        raise ValueError(f"Invalid {name} crop: {crop}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in crop):
        raise ValueError(f"Invalid {name} crop: {crop}")
    x1, y1, x2, y2 = crop
    width, height = size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{name} crop outside frame {size}: {crop}")
    return tuple(crop)


def tap_points(action, size, cards, battlefield):
    """Pixel coordinates in ORIGINAL iPhone frames, not scaled preview pixels.

    Public policy coordinates are 1-based. The remote client maps these pixels
    to logical iOS coordinates. '#' cells are not implicitly forbidden.
    """
    if action["type"] != "play":
        raise ValueError("Only PLAY has tap points")
    for name, maximum in (("slot", 4), ("row", 32), ("column", 18)):
        value = action.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"Invalid 1-based {name}: {value}")
    cx1, cy1, cx2, cy2 = validate_crop(cards, size, "CARDS")
    bx1, by1, bx2, by2 = validate_crop(battlefield, size, "BATTLEFIELDS")
    card = (cx1 + (action["slot"] - .5) * (cx2 - cx1) / 4, (cy1 + cy2) / 2)
    cell = (bx1 + (action["column"] - .5) * (bx2 - bx1) / 18,
            by1 + (action["row"] - .5) * (by2 - by1) / 32)
    return card, cell


def create_remote(**kwargs):
    # Import lazily: --help and fake-phone tests must not connect or need PyAV.
    from iphone_screen.windows_iphone_client_v2 import IPhoneRemote

    class BotRemote(IPhoneRemote):
        def rpc(self, method, params=None):
            if method == "device.io.tap":
                # Original client retries RPC on errors. A tap is non-idempotent:
                # timeout can mean it WAS executed but its reply was lost.
                # Send once, then stop the bot on uncertainty; never retry taps.
                return self._rpc_ws(method, params)
            return super().rpc(method, params)

    return BotRemote(**kwargs)


class IPhoneFrameSource:
    """Newest decoded frame only, with wall-clock indices for perception timers.

    Arrival age is available from the bridge client; it cannot measure hidden
    network/encoder buffering before a frame arrives at this computer.
    """
    def __init__(self, remote, *, stop=None, fps=30., frame_timeout=2., max_frame_age_ms=1000.,
                 max_seconds=None, clock=time.perf_counter):
        self.remote, self.stop = remote, stop if stop is not None else threading.Event()
        self.fps, self.frame_timeout = fps, frame_timeout
        self.max_frame_age_ms, self.max_seconds = max_frame_age_ms, max_seconds
        self.clock = clock
        frame = remote.get_screen(wait_new=False, timeout=10., copy=True)
        self.height, self.width = frame.shape[:2]
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR iPhone frame")
        self.size = (self.width, self.height)
        self.last_received_at = None
        self.decoded = 0

    def isOpened(self):
        return not self.stop.is_set()

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.width,
                cv2.CAP_PROP_FRAME_HEIGHT: self.height, cv2.CAP_PROP_FRAME_COUNT: 0,
                cv2.CAP_PROP_POS_FRAMES: self.decoded}.get(prop, 0)

    def release(self):
        self.stop.set()

    def iter_frames(self, target_fps):
        origin, next_due, previous_index = None, self.clock(), -1
        while not self.stop.is_set():
            if self.stop.wait(max(0., next_due - self.clock())):
                return
            try:
                frame = self.remote.get_screen(wait_new=True, timeout=self.frame_timeout, copy=True)
            except TimeoutError as error:
                raise RuntimeError("iPhone video timed out; stopping, no automatic reconnection of bot actions") from error
            now = self.clock()
            if frame.shape[:2] != (self.height, self.width):
                raise RuntimeError("iPhone resolution/orientation changed; restart with matching crops")
            age = self.remote.frame_age_ms
            if age is None or not math.isfinite(age) or age > self.max_frame_age_ms:
                raise RuntimeError("iPhone video is stale; stopping")
            self.last_received_at = now - max(0., age) / 1000
            if origin is None:
                origin = self.last_received_at
            elapsed = self.last_received_at - origin
            if self.max_seconds is not None and elapsed >= self.max_seconds:
                return
            # Source indices follow elapsed time even when YOLO/OCR is slower
            # than 30 FPS. Never replay/queue every missed camera frame.
            index = max(previous_index + 1, int(round(elapsed * self.fps)))
            previous_index = index
            self.decoded += 1
            next_due = now + 1 / target_fps
            yield index, frame


@dataclass(frozen=True)
class TapPair:
    card: tuple[float, float]
    cell: tuple[float, float]
    size: tuple[int, int]
    received_at: float


class TapExecutor:
    """At most ONE pair in flight, no backlog, no retries or implicit cooldown.

    Completion acknowledges RPCs only, not that the game accepted the play.
    New recommendations are suspended while busy, not queued for later use.
    """
    def __init__(self, remote, *, execute=False, stop=None, tap_gap_ms=100.,
                 max_frame_age_ms=1000., max_action_age_ms=1500., clock=time.perf_counter):
        self.remote, self.execute = remote, execute
        self.stop = stop if stop is not None else threading.Event()
        self.paused = threading.Event()
        self.tap_gap_ms, self.max_frame_age_ms = tap_gap_ms, max_frame_age_ms
        self.max_action_age_ms, self.clock = max_action_age_ms, clock
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bot-tap-pair")
        self.future = None
        self.last_key = None
        self.completed = 0

    @property
    def busy(self):
        return self.future is not None and not self.future.done()

    def poll(self):
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            try:
                completed = future.result()
            except Exception as error:
                self.stop.set()
                raise RuntimeError("Tap failed or acknowledgement is uncertain; stopped without retry") from error
            if completed:
                self.completed += 1
                print(f"[tap] pair completed ({self.completed}); game acceptance not confirmed", flush=True)

    def _check_screen(self, pair):
        if self.stop.is_set() or self.paused.is_set():
            return False
        age = self.remote.frame_age_ms
        if age is None or not math.isfinite(age) or age > self.max_frame_age_ms:
            raise RuntimeError("Refusing tap on stale video")
        frame = self.remote.get_screen(wait_new=False, timeout=.1, copy=False)
        if (frame.shape[1], frame.shape[0]) != pair.size:
            raise RuntimeError("Refusing tap after resolution/orientation change")
        return True

    def _run_pair(self, pair):
        if (self.clock() - pair.received_at) * 1000 > self.max_action_age_ms:
            print("[tap] skipped stale recommendation", flush=True)
            return False
        if not self._check_screen(pair):
            return False
        self.remote.send_tap(*pair.card)
        if self.stop.wait(self.tap_gap_ms / 1000) or not self._check_screen(pair):
            print("[tap] second tap cancelled; card may remain selected", flush=True)
            return False
        self.remote.send_tap(*pair.cell)
        return True

    def submit(self, action, pair):
        self.poll()
        if self.busy or self.stop.is_set() or self.paused.is_set() or action["type"] != "play":
            return False
        key = (action["battle_id"], action["observation_timestamp_ms"])
        if key == self.last_key:
            return False
        self.last_key = key
        if (self.clock() - pair.received_at) * 1000 > self.max_action_age_ms:
            print("[tap] skipped stale recommendation (recognition took too long)", flush=True)
            return False
        print(f"[{'tap' if self.execute else 'DRY RUN'}] {action['card']}: "
              f"card ({pair.card[0]:.1f}, {pair.card[1]:.1f}) -> "
              f"cell ({pair.cell[0]:.1f}, {pair.cell[1]:.1f})", flush=True)
        if self.execute:
            self.future = self.pool.submit(self._run_pair, pair)
        return True

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mac-ip", default="10.10.10.1")
    parser.add_argument("--control-port", type=int, default=22004)
    parser.add_argument("--video-port", type=int, default=22005)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true", help="Enable actual iPhone taps; default is dry-run")
    parser.add_argument("--headless", action="store_true", help="No windows; stop with Ctrl+C")
    parser.add_argument("--show-cards", action="store_true")
    parser.add_argument("--show-battlefield", action="store_true")
    parser.add_argument("--state-fps", type=positive, default=5.)
    parser.add_argument("--detection-fps", type=positive, default=30.)
    parser.add_argument("--source-fps", type=positive, default=30., help="Live time/index scale (normally bridge 30 FPS)")
    parser.add_argument("--play-threshold", type=float, default=.5)
    parser.add_argument("--min-card-confidence", type=float, default=.7)
    parser.add_argument("--card-costs", type=Path)
    parser.add_argument("--allowed-cells", type=Path)
    parser.add_argument("--no-tower-hp", action="store_true")
    parser.add_argument("--rpc-timeout", type=positive, default=3.)
    parser.add_argument("--frame-timeout", type=positive, default=2.)
    parser.add_argument("--max-frame-age-ms", type=positive, default=1000.)
    parser.add_argument("--max-action-age-ms", type=positive, default=1500.)
    parser.add_argument("--tap-gap-ms", type=positive, default=100.)
    parser.add_argument("--max-seconds", type=positive, help="Stop after this much live battle time")
    parser.add_argument("--predictions", type=Path, help="Write predictions to a NEW JSONL file")
    return parser


def run(args):
    from model_paths import CARDS, BATTLEFIELDS
    from offline_rl.object_gru.predict_action import ObjectGRUPredictor
    from offline_rl.build_trajectories import BuildConfig
    from predict_video_actions import VideoActionAdvisor

    for port in (args.control_port, args.video_port):
        if not 1 <= port <= 65535:
            raise ValueError("Ports must be in 1..65535")
    if args.state_fps > min(args.detection_fps, args.source_fps):
        raise ValueError("state-fps must not exceed detection-fps/source-fps")
    if args.predictions is not None and args.predictions.exists():
        raise FileExistsError(f"Refusing to overwrite {args.predictions}")
    costs = json.loads(args.card_costs.read_text(encoding="utf-8-sig")) if args.card_costs else None
    cells = json.loads(args.allowed_cells.read_text(encoding="utf-8-sig")) if args.allowed_cells else None
    if cells is not None:
        import numpy as np
        mask = np.asarray(cells)
        if mask.dtype != bool or mask.shape not in {(32, 18), (4, 32, 18)}:
            raise ValueError("allowed-cells must be boolean [32,18] or [4,32,18]")
    predictor = ObjectGRUPredictor(args.checkpoint, device=args.device, card_costs=costs,
                                   min_card_confidence=args.min_card_confidence, play_threshold=args.play_threshold)
    config = BuildConfig(state_fps=args.state_fps, detection_fps=args.detection_fps,
                         min_card_confidence=args.min_card_confidence)
    config.validate()
    # Lazy import: --help never loads detectors or connects to the bridge.
    import predict_video_kalman as perception
    perception.show_cards, perception.show_battlefield = args.show_cards, args.show_battlefield
    stop = threading.Event()
    with ExitStack() as stack:
        remote = stack.enter_context(create_remote(mac_ip=args.mac_ip, control_port=args.control_port,
                        video_port=args.video_port, expected_fps=args.source_fps, rpc_timeout=args.rpc_timeout))
        source = IPhoneFrameSource(remote, stop=stop, fps=args.source_fps, frame_timeout=args.frame_timeout,
                        max_frame_age_ms=args.max_frame_age_ms, max_seconds=args.max_seconds)
        stack.callback(source.release)
        for name, regions in (("CARDS", CARDS), ("BATTLEFIELDS", BATTLEFIELDS)):
            if source.size not in regions:
                raise ValueError(f"Missing {name} for iPhone {source.size} in model_paths.py; configure the actual screen, no resizing")
            validate_crop(regions[source.size], source.size, name)
        executor = TapExecutor(remote, execute=args.execute, stop=stop, tap_gap_ms=args.tap_gap_ms,
                    max_frame_age_ms=args.max_frame_age_ms, max_action_age_ms=args.max_action_age_ms)
        stack.callback(executor.close)
        log = stack.enter_context(args.predictions.open("x", encoding="utf-8")) if args.predictions else None
        advisor = VideoActionAdvisor(predictor, config, battle_id="iphone-live", hp_enabled=not args.no_tower_hp,
                                      output=log, allowed_cells=cells)

        def observe(frame_data):
            executor.poll()
            before = advisor.prediction_count
            advisor(frame_data, predict=not executor.busy and not executor.paused.is_set() and not stop.is_set())
            if advisor.prediction_count == before or advisor.latest["type"] != "play":
                return
            card, cell = tap_points(advisor.latest, source.size, CARDS[source.size], BATTLEFIELDS[source.size])
            executor.submit(advisor.latest, TapPair(card, cell, source.size, source.last_received_at))

        def annotate(frame, frame_data):
            advisor.annotate(frame, frame_data)
            status = "PAUSED" if executor.paused.is_set() else "TAP PAIR IN PROGRESS" if executor.busy else "READY"
            mode = "REAL TAPS" if args.execute else "DRY RUN - NO TAPS"
            top = max(0, frame.shape[0] - 85)
            cv2.rectangle(frame, (0, top), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 0), -1)
            cv2.putText(frame, f"{mode} | {status}", (10, top + 35), cv2.FONT_HERSHEY_SIMPLEX,
                        max(.4, frame.shape[1] / 1500), (0, 200, 255), 2)
            cv2.putText(frame, "Space: pause/resume | Q/Esc: stop", (10, top + 65), cv2.FONT_HERSHEY_SIMPLEX,
                        max(.4, frame.shape[1] / 1600), (0, 200, 255), 2)

        def keypress(key):
            if key in (27, ord("q")):
                stop.set()
                raise StopBot()
            if key == ord(" "):
                if executor.paused.is_set():
                    executor.paused.clear()
                else:
                    executor.paused.set()
                print("[bot] PAUSED" if executor.paused.is_set() else "[bot] RESUMED", flush=True)

        print(f"iPhone {source.size}; {'REAL TAPS ENABLED' if args.execute else 'DRY RUN, no taps'}", flush=True)
        print("Open a battle, not menus. Stop before battle end/next battle. Q/Esc stops; Space pauses.", flush=True)
        if costs is None:
            print("WARNING: no --card-costs; elixir affordability is NOT checked.", flush=True)
        if cells is None:
            print("WARNING: no --allowed-cells; game placement rules are NOT checked.", flush=True)
        try:
            perception.run_video_prediction(input_video="iphone-live", display=not args.headless,
                frame_source=source, key_callback=keypress, process_fps_limit=args.detection_fps,
                write_video=False, write_logs=False, hp_enabled=not args.no_tower_hp,
                synchronous_hp=True, observation_callback=observe, annotation_callback=annotate)
        finally:
            stop.set()
        executor.poll()
        print(f"Finished: {advisor.prediction_count} recommendations; {executor.completed} completed tap pairs.", flush=True)


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (StopBot, KeyboardInterrupt):
        print("Stopped. In-flight RPC cannot be undone; no new tap pair will start.", flush=True)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
