"""Object Transformer + GRU inference: recorded states or live JSONL, not game controls.

Uses the shared bounded history and legality checks; only model loading and the
joint 576-cell head differ from StARformer. Public slot/row/column are ONE-based.
"""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from offline_rl.predict_action import ActionPredictor, main as shared_main
from offline_rl.object_gru.model import ARCHITECTURE, ObjectGRUConfig, ObjectGRUPolicy


class ObjectGRUPredictor(ActionPredictor):
    @staticmethod
    def load_model(saved):
        if (saved.get("architecture") != ARCHITECTURE or saved.get("policy_kind") != "imitation"
                or saved.get("history_mode") != "observations_only"):
            raise ValueError("Expected an object_transformer_gru_v1 imitation checkpoint with observations_only history")
        return ObjectGRUPolicy(ObjectGRUConfig(**saved["model_config"]))

    def position_logits(self, outputs, timestep, slot):
        return outputs["position"][0, timestep].reshape(32, 18)


def main(argv=None):
    return shared_main(argv, predictor_class=ObjectGRUPredictor)


if __name__ == "__main__":
    raise SystemExit(main())
