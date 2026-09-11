"""Exact 50/50 play/noop batches for imitation only; no trajectory rewriting."""
import math

import torch
from torch.utils.data import Sampler

from .dataset import PREV_PLAY


class BalancedActionBatchSampler(Sampler):
    """Each batch has batch_size/2 plays and batch_size/2 noops.

    Only windows with valid final labels enter either pool. Partial-position
    plays still supervise type/slot; the dataset masks their coordinate loss.
    All past observations remain in each original window, including unknown
    transitions. The larger pool is covered once per epoch, with at most half
    a batch of padding repeats. The smaller pool is reshuffled/cycled as needed.
    The caller saves generator state at epoch boundaries for exact resume.
    """
    def __init__(self, dataset, batch_size, *, generator):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 2 or batch_size % 2:
            raise ValueError("Balanced sampling requires an even batch_size >=2")
        if dataset.supervise != "last":
            raise ValueError("Balanced sampling requires supervise=last")
        self.batch_size, self.generator = batch_size, generator
        self.play_indices, self.noop_indices = [], []
        for index, battle, end in dataset.iter_window_endpoints():
            action, valid = dataset.encoder._action(dataset.trajectories[battle]["transitions"][end])
            if valid:
                (self.play_indices if action[0] == PREV_PLAY else self.noop_indices).append(index)
        if not self.play_indices or not self.noop_indices:
            raise ValueError("Balanced sampling needs both valid play and noop windows; check labels and stride")
        self.batches = math.ceil(max(len(self.play_indices), len(self.noop_indices)) / (batch_size // 2))
        self.stats = {"unique_plays": len(self.play_indices), "unique_noops": len(self.noop_indices),
                      "excluded_windows": len(dataset) - len(self.play_indices) - len(self.noop_indices),
                      "batches": self.batches, "samples_per_epoch": self.batches * batch_size}

    def __len__(self):
        return self.batches

    def _draw(self, pool, count):
        result = []
        while len(result) < count:
            order = torch.randperm(len(pool), generator=self.generator).tolist()
            result.extend(pool[i] for i in order[:count - len(result)])
        return result

    def __iter__(self):
        half = self.batch_size // 2
        plays = self._draw(self.play_indices, self.batches * half)
        noops = self._draw(self.noop_indices, self.batches * half)
        for start in range(0, len(plays), half):
            batch = plays[start:start + half] + noops[start:start + half]
            order = torch.randperm(self.batch_size, generator=self.generator).tolist()
            yield [batch[i] for i in order]
