"""Video example: shared Kalman perception + trained action recommendations.

From v2: python predict_video_actions.py battle.mp4 --checkpoint runs/offline_rl/first/best.pt
Displays Tracking/Arena windows; Q exits. No clicks or game control are performed.
Saving annotated video or JSONL recommendations is opt-in.
Both imitation and IQL checkpoints are supported; only the actor runs here.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

V2_DIR = Path(__file__).resolve().parent
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from offline_rl.build_trajectories import BuildConfig, TrajectoryCollector, assemble_trajectory
from offline_rl.predict_action import ActionPredictor


class VideoActionAdvisor:
    """Adapt causal per-frame recognition to bounded policy history.

    Confirmed events may refer to an older interval. Rebuild the small context
    from currently available evidence, never move that play to the latest step.
    One extra state preserves the preceding transition at the left window edge.
    Recent unconfirmed noops are masked for feedback_delay_ms (a heuristic, not
    proof of inactivity). The latest state itself is not delayed.
    """

    def __init__(self, predictor, config, *, battle_id="video", feedback_delay_ms=1000,
                 hp_enabled=True, output=None, allowed_cells=None):
        if not math.isfinite(feedback_delay_ms) or feedback_delay_ms < 0:
            raise ValueError("feedback_delay_ms must be finite and nonnegative")
        self.predictor, self.config = predictor, config
        self.collector = TrajectoryCollector(config)
        self.battle_id, self.feedback_delay_ms = str(battle_id), feedback_delay_ms
        self.hp_enabled, self.output, self.allowed_cells = hp_enabled, output, allowed_cells
        self.latest = None
        self.last_prediction_ms = None
        self.last_observation_ms = None
        self.prediction_count = 0
        self.predictor.reset(self.battle_id)

    def __call__(self, frame, *, predict=True):
        self.collector(frame)
        collector = self.collector
        if not collector.observations:
            return
        timestamp = collector.observations[-1]["timestamp_ms"]
        if timestamp == self.last_observation_ms:
            return  # Evidence is retained until the next policy deadline.
        # Keep the same feature context size as training, plus its predecessor.
        capacity = self.predictor.encoder.sequence_length + 1
        collector.observations[:] = collector.observations[-capacity:]
        cutoff = collector.observations[0]["timestamp_ms"]
        collector.events[:] = [event for event in collector.events if event["action_timestamp_ms"] > cutoff]
        collector.empty_transitions[:] = [event for event in collector.empty_transitions
                                          if event["timestamp_ms"] + self.config.uncertainty_ms >= cutoff]
        collector.elixir_drops[:] = [(before, after) for before, after in collector.elixir_drops
                                    if after + self.config.uncertainty_ms >= cutoff]
        collector.hp_rewards.events[:] = [event for event in collector.hp_rewards.events
                                         if event["timestamp_ms"] > cutoff]
        if getattr(self.predictor, "history_mode", "executed_feedback") == "observations_only":
            # IQL consumes only causally observed states. Keep the true episode
            # step counter; no delayed event backfilling/rebuilding is necessary.
            self.predictor.observe(collector.observations[-1])
        else:
            self._rebuild_feedback_history(timestamp)
        self.last_observation_ms = timestamp
        if not predict:
            return  # Live action RPC can be busy; still preserve sampled history.
        self.latest = self.predictor.predict(allowed_cells=self.allowed_cells)
        self.latest["frame_number"] = frame["frame_number"]
        self.last_prediction_ms = timestamp
        self.prediction_count += 1
        recommendation = self.latest
        detail = (f"PLAY {recommendation['card']} slot={recommendation['slot']} "
                  f"row={recommendation['row']} col={recommendation['column']}"
                  if recommendation["type"] == "play" else "WAIT")
        print(f"[policy {timestamp / 1000:.2f}s] {detail} "
              f"P(play)={recommendation['play_probability']:.3f} "
              f"threshold={recommendation['play_threshold']:.3f} "
              f"reason={recommendation['reason']}", flush=True)
        if self.output is not None:
            self.output.write(json.dumps(recommendation, ensure_ascii=False, allow_nan=False) + "\n")
            self.output.flush()

    def _rebuild_feedback_history(self, timestamp):
        """Original imitation feedback contract; IQL does not use this path."""
        collector = self.collector
        self.predictor.reset(self.battle_id)
        self.predictor.observe(collector.observations[0])
        if len(collector.observations) > 1:
            # Reuse offline matching, ambiguity and HP-reward rules. Neutral
            # result only suppresses outcome reward in this scratch structure;
            # no result is inferred, and done/RTG fields never enter the policy.
            partial = assemble_trajectory(
                collector.observations, deepcopy(collector.events), collector.empty_transitions,
                collector.elixir_drops, collector.hp_rewards, "draw", self.config,
            )
            for index, transition in enumerate(partial["transitions"]):
                observed = collector.observations[index + 1]
                valid = transition["action_valid"]
                if (transition["action"]["type"] == "noop"
                        and timestamp - observed["timestamp_ms"] < self.feedback_delay_ms):
                    valid = False
                self.predictor.observe(
                    observed, previous_action=transition["action"], previous_action_valid=valid,
                    previous_reward=transition["reward"] if self.hp_enabled else None,
                )

    def annotate(self, frame, frame_data):
        """Draw only on the output image, never on perception's input crops."""
        import cv2
        from model_paths import BATTLEFIELDS

        recommendation = self.latest
        if recommendation is None:
            return
        height, width = frame.shape[:2]
        play = recommendation["type"] == "play"
        color = (0, 255, 255) if play else (80, 230, 80)
        if play:
            crop = BATTLEFIELDS.get((width, height))
            if crop is not None:
                x1, y1, x2, y2 = crop
                cell_width, cell_height = (x2 - x1) / 18, (y2 - y1) / 32
                left = round(x1 + (recommendation["column"] - 1) * cell_width)
                top = round(y1 + (recommendation["row"] - 1) * cell_height)
                right, bottom = round(left + cell_width), round(top + cell_height)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
                cv2.drawMarker(frame, ((left + right) // 2, (top + bottom) // 2), color,
                               cv2.MARKER_CROSS, 24, 3)
        first = (f"AI: PLAY {recommendation['card']} | slot {recommendation['slot']} | "
                 f"row {recommendation['row']}, col {recommendation['column']}" if play else "AI: WAIT (noop)")
        age_ms = frame_data["timestamp_ms"] - recommendation["observation_timestamp_ms"]
        second = (f"P(play) {recommendation['play_probability']:.3f} | "
                  f"threshold {recommendation['play_threshold']:.3f} | "
                  f"history {recommendation['history_length']} | age {age_ms:.0f} ms | recommendation only")
        checks = recommendation["constraints"]
        third = (f"Elixir check: {'ON' if checks['elixir_checked'] else 'OFF (provide costs)'} | "
                 f"Placement mask: {'ON' if checks['placement_mask_supplied'] else 'OFF'}")
        if recommendation["reason"] == "no_allowed_play":
            third += " | PLAY blocked: no allowed slot/cell"
        scale = max(.45, width / 1100)
        line_height = max(23, round(34 * scale))
        top = 84  # Leave the original frame/tracker counters visible.
        cv2.rectangle(frame, (0, top), (width - 1, min(height - 1, top + line_height * 3 + 12)), (20, 20, 20), -1)
        for index, text in enumerate((first, second, third)):
            # fit long classifier labels into the image width
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
            fit = min(scale, scale * (width - 20) / max(1, text_width))
            cv2.putText(frame, text, (10, top + (index + 1) * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, fit, color, 2, cv2.LINE_AA)


class _PreviewFinished(Exception):
    pass


def main(argv=None, *, predictor_class=ActionPredictor, default_checkpoint=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, default=V2_DIR / "screenshots/input_omydays_cutted.mp4")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint or V2_DIR / "runs/offline_rl/first/best.pt")
    parser.add_argument("--device", default="auto", help="Policy device only; detector device remains in predict_video_kalman.py")
    parser.add_argument("--state-fps", type=float, default=5.0)
    parser.add_argument("--detection-fps", type=float, default=30.0)
    parser.add_argument("--feedback-delay-ms", type=float, default=1000, help="Imitation feedback only; unused by IQL")
    parser.add_argument("--min-card-confidence", type=float, default=.7)
    parser.add_argument("--play-threshold", type=float, default=0.5,
                        help="Play if P(play) exceeds this threshold in [0,1]; default 0.5; lower means more plays")
    parser.add_argument("--card-costs", type=Path, help="JSON card-name -> elixir cost mapping")
    parser.add_argument("--allowed-cells", type=Path, help="JSON boolean [32,18] or [4,32,18] deployment mask")
    parser.add_argument("--damage-scale", type=float, default=1000)
    parser.add_argument("--tower-reward", type=float, default=1)
    parser.add_argument("--max-hp-drop", type=int, default=2500)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-tower-hp", action="store_true", help="Disable OCR; prior rewards will be unknown")
    parser.add_argument("--show-cards", action="store_true")
    parser.add_argument("--show-battlefield", action="store_true")
    parser.add_argument("--output-video", type=Path, help="Save annotated video to a NEW file")
    parser.add_argument("--predictions", type=Path, help="Save recommendations to a NEW JSONL file")
    parser.add_argument("--max-seconds", type=float, help="Stop after this many source-video seconds")
    args = parser.parse_args(argv)
    try:
        if not args.video.is_file():
            raise FileNotFoundError(f"Video not found: {args.video}")
        if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
            raise ValueError("max-seconds must be finite and positive")
        for path in (args.output_video, args.predictions):
            if path is not None and path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")
        if args.output_video and args.predictions and args.output_video.resolve() == args.predictions.resolve():
            raise ValueError("Video and JSONL output paths must differ")
        config = BuildConfig(state_fps=args.state_fps, detection_fps=args.detection_fps,
                             min_card_confidence=args.min_card_confidence, damage_scale=args.damage_scale,
                             tower_reward=args.tower_reward, max_hp_drop=args.max_hp_drop)
        config.validate()
        costs = json.loads(args.card_costs.read_text(encoding="utf-8-sig")) if args.card_costs else None
        cells = json.loads(args.allowed_cells.read_text(encoding="utf-8-sig")) if args.allowed_cells else None
        predictor = predictor_class(args.checkpoint, device=args.device, card_costs=costs,
                                    min_card_confidence=args.min_card_confidence,
                                    play_threshold=args.play_threshold)
        print(f"Policy: {predictor.policy_kind}; history: {predictor.history_mode}; "
              f"play threshold: {predictor.play_threshold:g}", flush=True)
        # Lazy import: --help and adapter unit tests don't load YOLO/OCR.
        import predict_video_kalman as perception

        perception.show_cards = args.show_cards
        perception.show_battlefield = args.show_battlefield
        with ExitStack() as stack:
            log = stack.enter_context(args.predictions.open("x", encoding="utf-8")) if args.predictions else None
            advisor = VideoActionAdvisor(predictor, config, battle_id=args.video.stem,
                                         feedback_delay_ms=args.feedback_delay_ms,
                                         hp_enabled=not args.no_tower_hp, output=log, allowed_cells=cells)
            def observe(frame):
                if args.max_seconds is not None and frame["timestamp_ms"] > args.max_seconds * 1000:
                    raise _PreviewFinished()
                advisor(frame)
            print("Recommendations only: no game input. Q closes the preview.", flush=True)
            if costs is None:
                print("WARNING: no card costs; elixir affordability is not checked.", flush=True)
            try:
                perception.run_video_prediction(
                    input_video=args.video, output_video=args.output_video or perception.OUTPUT_VIDEO,
                    process_fps_limit=args.detection_fps, display=not args.headless,
                    write_video=args.output_video is not None, write_logs=False,
                    hp_enabled=not args.no_tower_hp, synchronous_hp=True,
                    observation_callback=observe, annotation_callback=advisor.annotate,
                )
            except _PreviewFinished:
                print("Reached --max-seconds.", flush=True)
            print(f"Finished: {advisor.prediction_count} recommendations.", flush=True)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
