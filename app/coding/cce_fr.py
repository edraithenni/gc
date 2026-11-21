"""Kadhe, Koyluoglu, Ramchandran (2020): communication-efficient FR-GC."""

from itertools import combinations
from math import ceil

import torch

from .base import StaticScheme


class CCEGCFR(StaticScheme):
    def __init__(self, *, num_workers, num_partitions, stragglers, communication_factor, seed):
        self.n = num_workers
        self.k = num_partitions
        self.s = stragglers
        self.K = communication_factor
        self.N = self.K + self.s

        if self.N > self.n or self.n % self.N or self.k * self.N % self.n or self.K < 1:
            raise ValueError("CommFR requires m+s to divide n and n to divide k*(m+s)")

        self.groups = self.n // self.N
        self.partitions_per_group = self.k // self.groups

        generator = torch.Generator().manual_seed(seed)
        self.G = torch.randn(self.K, self.N, dtype=torch.float64, generator=generator)

        # Precompute decoding matrices for every valid subset.
        eye = torch.eye(self.K, dtype=torch.float64)
        self.decoders = {
            subset: torch.linalg.solve(self.G[:, list(subset)], eye).float()
            for subset in combinations(range(self.N), self.K)
        }

    def assigned_partitions(self, worker_id):
        start = (worker_id // self.N) * self.partitions_per_group
        return list(range(start, start + self.partitions_per_group))

    def encode(self, worker_id, gradients):
        # Accumulate gradients without creating intermediate sums.
        values = iter(gradients.values())
        vector = next(values).clone()
        for gradient in values:
            vector.add_(gradient)

        rows = ceil(vector.numel() / self.K)
        padded = torch.nn.functional.pad(vector, (0, rows * self.K - vector.numel()))
        column = self.G[:, worker_id % self.N].to(device=vector.device, dtype=vector.dtype)

        # Equivalent to padded.view(K, rows).T @ column, without transposing.
        return column @ padded.view(self.K, rows)

    def can_decode(self, workers):
        counts = [0] * self.groups
        for worker_id in set(workers):
            counts[worker_id // self.N] += 1
        return all(count >= self.K for count in counts)

    def decode(self, responses, gradient_size):
        first = next(iter(responses.values()))
        result = torch.zeros(gradient_size, dtype=first.dtype, device=first.device)
        rows = ceil(gradient_size / self.K)

        for group in range(self.groups):
            selected = sorted(w for w in responses if w // self.N == group)[:self.K]
            local = tuple(w % self.N for w in selected)
            decoder = self.decoders[local]

            if self.K == 2:
                # Direct accumulation without stack, matmul or tensor copies.
                for block in range(2):
                    start, end = block * rows, min((block + 1) * rows, gradient_size)
                    target = result[start:end]
                    for i, worker_id in enumerate(selected):
                        target.add_(responses[worker_id][:end-start], alpha=decoder[i, block].item())
            else:
                # General decoding with the correct output layout.
                Y = torch.stack([responses[w] for w in selected], dim=0)
                X = decoder.T.to(device=Y.device, dtype=Y.dtype) @ Y
                result.add_(X.reshape(-1)[:gradient_size])

        return result.div_(self.k)
