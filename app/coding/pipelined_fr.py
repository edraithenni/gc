
"""Su & Li (2026), Section IV.A: Pipelined Gradient Coding with Fractional Repetition."""


class PipelinedFR:
    def __init__(self, *, num_workers, num_partitions, stragglers):
        self.n = num_workers
        self.s = stragglers
        self.c = stragglers + 1
        if self.n <= 0 or num_partitions != self.n or not 0 <= self.s < self.n:
            raise ValueError("PGC-FR requires 0 <= s < n = k")
        if self.n % self.c:
            raise ValueError("PGC-FR requires n divisible by c=s+1")
        self.groups = self.n // self.c
        self.required = self.n - self.s

    def assigned_partitions(self, worker_id):
        start = (worker_id // self.c) * self.c
        return list(range(start, start + self.c))

    def partitions_to_compute(self, worker_id, step, cache, last_step):
        assigned = self.assigned_partitions(worker_id)
        if last_step is None or step != last_step + 1 or any(p not in cache for p in assigned):
            return assigned
        return [assigned[(step - 1) % self.c]]

    def encode(self, worker_id, gradients):
        return sum(gradients[p] for p in self.assigned_partitions(worker_id))

    def can_decode(self, workers):
        received = set(workers)
        return (len(received) >= self.required and
                len({worker_id // self.c for worker_id in received}) == self.groups)

    def decode(self, responses, gradient_size):
        selected = [next(w for w in responses if w // self.c == group)
                    for group in range(self.groups)]
        return sum(responses[w] for w in selected) / self.n
