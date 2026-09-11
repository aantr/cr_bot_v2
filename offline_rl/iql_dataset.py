"""One-step IQL transitions (causal history, action, reward, next history).

Unknown actions and partial placements are excluded, NOT replaced with wait.
Histories retain observations across excluded transitions, without jumping over
them in Bellman targets. The final observation is available as next_history.
"""
from collections import Counter

import torch
from torch.utils.data import Dataset

from .dataset import EMPTY_CARD_ID, PREV_PLAY, TrajectoryDataset, label
from .history_features import policy_inputs

HISTORY_MODE = "observations_only"


class IQLDataset(Dataset):
    def __init__(self, base: TrajectoryDataset):
        if base.stride != 1:
            raise ValueError("IQL needs stride=1; do not subsample Bellman transitions")
        if len(set(base.battle_ids)) != len(base.battle_ids):
            raise ValueError("Duplicate battle identities: use only one JSON version of each battle")
        self.base = base
        self.indices = []
        counts = Counter(total=0, noop=0, play=0, invalid=0, partial_position=0,
                         unknown_card=0, hand_mismatch=0, terminal_kept=0,
                         terminal_excluded=0, nonzero_rewards_excluded=0,
                         nonzero_rewards_kept=0)
        for bi, data in enumerate(base.trajectories):
            for ti, transition in enumerate(data["transitions"]):
                counts["total"] += 1
                action = transition["action"]
                reason = None
                if not transition["action_valid"]:
                    reason = "invalid"
                elif action["type"] == "play":
                    if not action.get("position_valid", True):
                        reason = "partial_position"
                    elif base.card_to_id.get(label(action["card"]), 1) <= EMPTY_CARD_ID:
                        reason = "unknown_card"
                    else:
                        hand = data["observations"][ti]["hand"]
                        before = next(c for c in hand if c["slot"] == action["slot"])
                        if label(before["card"]) != label(action["card"]):
                            reason = "hand_mismatch"
                if reason:
                    counts[reason] += 1
                    counts["terminal_excluded"] += int(transition["terminated"])
                    counts["nonzero_rewards_excluded"] += int(transition["reward"] != 0)
                    continue
                self.indices.append((bi, ti))
                counts[action["type"]] += 1
                counts["terminal_kept"] += int(transition["terminated"])
                counts["nonzero_rewards_kept"] += int(transition["reward"] != 0)
        self.stats = dict(counts, kept=len(self.indices))

    def __len__(self):
        return len(self.indices)

    def history(self, battle, end):
        data = self.base.trajectories[battle]
        obs = data["observations"]
        start = max(0, end + 1 - self.base.sequence_length)
        steps = list(range(start, end + 1))
        # Omit delayed feedback at source as well as enforcing the shared mask.
        previous = [None if i == 0 else {"action_valid": False, "reward": None} for i in steps]
        sample = self.base.encoder.encode(
            obs[start:end + 1], previous,
            [obs[max(0, i - 1)]["timestamp_ms"] for i in steps], steps,
            data["metadata"]["field_layout"], self.base.battle_ids[battle],
        )
        return policy_inputs(sample, HISTORY_MODE)

    def __getitem__(self, index):
        battle, step = self.indices[index]
        transition = self.base.trajectories[battle]["transitions"][step]
        encoded, valid = self.base.encoder._action(transition)
        assert valid
        # [type 0=noop/1=play, card ID, slot 1..4, row 1..32, column 1..18].
        # All fields except type are zero for noop (not arbitrary coordinates).
        action = [int(encoded[0] == PREV_PLAY), *encoded[1:]]
        return {"history": self.history(battle, step),
                "next_history": self.history(battle, step + 1),
                "action": torch.tensor(action, dtype=torch.long),
                "reward": torch.tensor(transition["reward"] / self.base.normalization.reward),
                "discount": torch.tensor(transition["discount"]),
                "terminated": torch.tensor(transition["terminated"]),
                "truncated": torch.tensor(transition["truncated"]),
                "dt_ms": torch.tensor(transition["dt_ms"]),
                "transition_index": torch.tensor(step)}
