"""Select only measured features from the shared offline/live observation encoder.

No separate on-disk dataset: offline_rl.dataset.TrajectoryDataset is reused.
Coordinates retain the original 32 rows x 18 columns. Track IDs, missing unit
HP/velocity/age, next card, king towers and opponent elixir are not inputs.
Rewards, action history, targets and future returns are never policy inputs.
"""
import torch


def object_features(batch):
    """Return [B,T,N,6]: team, x, y, class/detector confidence, Kalman flag.

    Team: ally +1, enemy -1, unknown 0. x/y are normalized cell centres,
    not raw video pixels. Padding is neutralized before embeddings/projections.
    """
    states = batch["states"]
    mask = states["unit_mask"] & batch["attention_mask"].unsqueeze(-1)
    side = states["unit_sides"]
    team = (side == 2).float() - (side == 3).float()
    cells = states["unit_cells"].float()
    xy = torch.stack(((cells[..., 1] + .5) / 18, (cells[..., 0] + .5) / 32), dim=-1)
    features = torch.cat((team.unsqueeze(-1), xy, states["unit_features"]), dim=-1)
    return (states["unit_ids"].masked_fill(~mask, 0),
            features.masked_fill(~mask.unsqueeze(-1), 0), mask)


def auxiliary_features(batch):
    """Four measured towers, four ordered hand slots and four globals.

    Missing HP/elixir is represented by its availability mask, not assumed dead
    towers or empty elixir. Globals use elapsed video time and preceding dt;
    elapsed video time is NOT claimed to be remaining battle time.
    """
    s = batch["states"]
    real = batch["attention_mask"].unsqueeze(-1)
    hp_known = s["tower_hp_known_mask"]
    towers = torch.stack((s["tower_hp"].masked_fill(~hp_known, 0), hp_known.float(),
                          (s["tower_hp_fresh_mask"] & hp_known).float(),
                          s["tower_hp_confidence"].masked_fill(~hp_known, 0),
                          s["tower_hp_age"].masked_fill(~hp_known, 0)), dim=-1).flatten(-2)
    hand_extra = torch.stack((s["hand_confidence"], s["hand_known_mask"].float(),
                              s["hand_nonempty_mask"].float()), dim=-1)
    globals_ = torch.cat((s["elixir"].masked_fill(~s["elixir_mask"], 0),
                          s["elixir_mask"].float(), s["time_features"][..., :2]), dim=-1)
    return (towers.masked_fill(~real, 0), s["hand_ids"].masked_fill(~real, 0),
            hand_extra.masked_fill(~real.unsqueeze(-1), 0), globals_.masked_fill(~real, 0))
