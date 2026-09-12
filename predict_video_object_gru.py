"""Video detections -> Object Transformer + GRU -> annotated recommendations.

From v2:
  python predict_video_object_gru.py battle.mp4 --checkpoint runs/offline_rl/object_gru/best_play.pt --device 0 --show-cards
All recognition, drawing, state cadence and masks reuse predict_video_actions.py.
"""
from pathlib import Path

from offline_rl.object_gru.predict_action import ObjectGRUPredictor
from predict_video_actions import main as shared_main


def main(argv=None):
    return shared_main(argv, predictor_class=ObjectGRUPredictor,
                        default_checkpoint=Path(__file__).resolve().parent / "runs/offline_rl/object_gru/best_play.pt")


if __name__ == "__main__":
    raise SystemExit(main())
