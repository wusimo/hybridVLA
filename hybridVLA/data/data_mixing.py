"""
Multi-source data mixer for HybridVLA v2 training.

During Phase 2 training, each batch is mixed from multiple data sources
with configurable ratios:

Default mixing ratios (from DESIGN_V2.md):
  70% robot demonstration data    (flow matching loss → action expert)
  20% language co-training data   (LM loss → VLM backbone only)
  10% CoP/grounding data          (point loss → CoP module + VLM)

The mixer supports:
- Weighted sampling across datasets with different sizes
- Epoch-based iteration with automatic re-balancing
- Dynamic ratio adjustment during training
"""

import logging
import math
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class MixedDataset(Dataset):
    """
    Combines multiple datasets with weighted sampling.

    Each dataset is assigned a weight that controls how often its
    samples appear in the mixed stream. Datasets of different sizes
    are handled via oversampling the smaller ones.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        weights: dict[str, float] | None = None,
    ):
        self.datasets = datasets
        self.names = list(datasets.keys())

        # Default: equal weights
        if weights is None:
            weights = {name: 1.0 / len(datasets) for name in self.names}

        # Normalize weights
        total = sum(weights[n] for n in self.names)
        self.weights = {n: weights[n] / total for n in self.names}

        # Build a flat index: (dataset_name, local_idx)
        self._indices = []
        self._build_indices()

        logger.info(
            f"MixedDataset: {len(self._indices)} total samples from "
            + ", ".join(f"{n}({len(datasets[n])}×{self.weights[n]:.1%})" for n in self.names)
        )

    def _build_indices(self):
        """Build flat index list with weighted oversampling."""
        self._indices = []

        # Determine target count per dataset
        max_size = max(len(self.datasets[n]) for n in self.names)
        total_target = int(max_size / max(self.weights.values()))

        for name in self.names:
            ds = self.datasets[name]
            target = int(total_target * self.weights[name])
            # Repeat dataset to reach target
            repeats = math.ceil(target / max(len(ds), 1))
            for r in range(repeats):
                for i in range(len(ds)):
                    self._indices.append((name, i))
                    if len(self._indices) >= total_target:
                        break
                if len(self._indices) >= total_target:
                    break

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        name, local_idx = self._indices[idx]
        sample = self.datasets[name][local_idx]
        # Tag with source for loss routing
        sample["_source"] = name
        return sample


class MixedBatchSampler(Sampler):
    """
    Ensures each batch contains samples from a single source.

    This is important because different sources use different loss
    functions (flow matching vs LM loss vs point loss), and mixing
    them in a single batch would require complex loss routing.

    Instead, we alternate between sources at the batch level:
    - 7 batches from robotics data
    - 2 batches from language data
    - 1 batch from grounding data
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        weights: dict[str, float],
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        self.datasets = datasets
        self.names = list(datasets.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Normalize weights to batch counts per cycle
        total_w = sum(weights.values())
        self.batches_per_cycle = {
            name: max(1, round(10 * weights[name] / total_w))
            for name in self.names
        }

        # Build per-dataset index lists
        self._per_dataset_indices = {}
        for name, ds in datasets.items():
            indices = list(range(len(ds)))
            self._per_dataset_indices[name] = indices

        # Calculate total batches
        total_samples = sum(len(ds) for ds in datasets.values())
        self._num_batches = total_samples // batch_size

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle per-dataset indices
        per_ds = {}
        for name, indices in self._per_dataset_indices.items():
            perm = torch.randperm(len(indices)).tolist() if self.shuffle else list(range(len(indices)))
            per_ds[name] = iter(perm)

        # Build global offset for each dataset
        offset = {}
        cumulative = 0
        for name in self.names:
            offset[name] = cumulative
            cumulative += len(self.datasets[name])

        # Yield batches in round-robin with weights
        yielded = 0
        while yielded < self._num_batches:
            for name in self.names:
                for _ in range(self.batches_per_cycle.get(name, 1)):
                    batch = []
                    for _ in range(self.batch_size):
                        try:
                            local_idx = next(per_ds[name])
                        except StopIteration:
                            # Re-shuffle and restart
                            perm = torch.randperm(len(self._per_dataset_indices[name])).tolist()
                            per_ds[name] = iter(perm)
                            local_idx = next(per_ds[name])
                        batch.append(offset[name] + local_idx)
                    yield batch
                    yielded += 1
                    if yielded >= self._num_batches:
                        return

    def __len__(self) -> int:
        return self._num_batches
