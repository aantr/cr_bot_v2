"""Shared history contract for offline training and live inference."""
import torch

from .dataset import PREV_BOS, PREV_UNKNOWN

INPUT_KEYS = (
    "states", "terrain_mask", "previous_actions", "previous_action_valid",
    "previous_rewards", "previous_reward_mask", "attention_mask", "padding_mask",
    "causal_mask", "step_indices", "sequence_length", "battle_id",
)


def policy_inputs(sample, mode="executed_feedback"):
    """Accept an unbatched sample or batch; never mutate its tensors.

    IQL uses observations_only: no offline-only, retrospectively confirmed
    actions/rewards in history. Rewards still enter Bellman targets separately.
    Imitation checkpoints retain their original executed_feedback contract.
    """
    if mode not in {"executed_feedback", "observations_only"}:
        raise ValueError(f"Unsupported history mode: {mode}")
    result = {key: sample[key] for key in INPUT_KEYS}
    if mode == "observations_only":
        action = torch.zeros_like(sample["previous_actions"])
        action[..., 0] = torch.where(sample["attention_mask"], PREV_UNKNOWN, 0)
        action[..., 0].masked_fill_(sample["attention_mask"] & (sample["step_indices"] == 0), PREV_BOS)
        result.update(previous_actions=action,
                      previous_action_valid=torch.zeros_like(sample["previous_action_valid"]),
                      previous_rewards=torch.zeros_like(sample["previous_rewards"]),
                      previous_reward_mask=torch.zeros_like(sample["previous_reward_mask"]))
    return result
