"""Streaming Packed Token Sequence Dataset for Foundation Model Training."""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset


class SyntheticPackedDataset(IterableDataset):
    """High-throughput synthetic pre-training dataset providing packed token streams."""

    def __init__(
        self,
        vocab_size: int = 32000,
        seq_len: int = 2048,
        num_samples: int = 10000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        total_workers = worker_info.num_workers if worker_info is not None else 1

        samples_per_worker = self.num_samples // total_workers

        for i in range(samples_per_worker):
            # Seed uniquely per worker and sample
            torch.manual_seed(10000 * worker_id + i)
            # Generate random sequence of token IDs
            input_ids = torch.randint(
                low=10,
                high=self.vocab_size - 1,
                size=(self.seq_len,),
                dtype=torch.long,
            )
            labels = input_ids.clone()
            yield {
                "input_ids": input_ids,
                "labels": labels,
            }
