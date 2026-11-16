
import os
import time

import ray

class ResponseCollector:
    def __init__(self, workers):
        self.workers = workers
        self.pending = {}
        self.failed = {}
        self.timeout_s = float(os.getenv("GC_STEP_TIMEOUT_S", "180"))

    def submit(self, worker_ids, epoch, step, weights):
        for worker_id in worker_ids:
            meta_ref, grad_ref = self.workers[worker_id].compute_step.remote(epoch, step, weights)
            self.pending[meta_ref] = (grad_ref, worker_id, epoch, step, time.perf_counter())

    def collect(self, epoch, step, weights, scheme):
        accepted = {}
        stale = 0
        started = time.perf_counter()
        next_report = 15.0

        while not scheme.can_decode(accepted):
            elapsed = time.perf_counter() - started
            if elapsed > self.timeout_s or not self.pending:
                raise TimeoutError(f"epoch={epoch+1} step={step+1} pending={list(self.pending.values())} failed={self.failed}")
            ready, _ = ray.wait(list(self.pending), num_returns=1, timeout=2, fetch_local=False)
            if not ready:
                if elapsed >= next_report:
                    print(f"WAIT epoch={epoch+1} step={step+1} {elapsed:.0f}s received={sorted(accepted)}", flush=True)
                    next_report += 15.0
                continue

            meta_ref = ready[0]
            grad_ref, worker_id, sent_epoch, sent_step, sent_at = self.pending.pop(meta_ref)
            try:
                meta = ray.get(meta_ref)
            except ray.exceptions.RayError as error:
                self.failed[worker_id] = str(error)
                print(f"GCWorker {worker_id} failed: {error}", flush=True)
                continue

            if (sent_epoch, sent_step) != (epoch, step):
                stale += 1
                self.submit([worker_id], epoch, step, weights)
                continue

            accepted[worker_id] = (grad_ref, meta, sent_at)

        metadata_wait_s = time.perf_counter() - started
        ids = sorted(accepted)
        grad_refs = [accepted[worker_id][0] for worker_id in ids]
        remaining = max(0.0, self.timeout_s - metadata_wait_s)
        ready, _ = ray.wait(grad_refs, num_returns=len(grad_refs), timeout=remaining, fetch_local=True)
        if len(ready) != len(grad_refs):
            raise TimeoutError(f"Gradient transfer timeout: epoch={epoch+1} step={step+1}")

        gradients = ray.get(grad_refs)
        finished = time.perf_counter()
        responses = dict(zip(ids, gradients))
        timings = [{"worker_id": wid, **accepted[wid][1]["profile"],
                    "response_time_s": finished - accepted[wid][2],
                    "gradient_bytes": responses[wid].numel() * responses[wid].element_size()}
                   for wid in ids]

        return responses, sum(accepted[wid][1]["loss"] for wid in ids) / len(ids), {
            "wait_time_s": finished - started,
            "metadata_wait_time_s": metadata_wait_s,
            "gradient_fetch_time_s": finished - started - metadata_wait_s,
            "stale_results_seen": stale,
            "worker_timings": timings,
        }
