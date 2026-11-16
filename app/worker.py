import os
import time

import ray
import torch
from torch.nn import functional as F

from coding import make_scheme
from data import TrainingBatches, evaluation_loader
from model import build_model, flatten_gradients, load_flat_parameters


@ray.remote(num_cpus=1, resources={"gc_worker": 1}, max_restarts=0)
class GCWorker:
    def __init__(self, worker_id, config):
        torch.set_num_threads(1)
        self.worker_id = worker_id
        self.config = config
        self.device = torch.device(os.getenv("GC_DEVICE", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside GCWorker")
        self.model = build_model().to(self.device)
        self.batches = TrainingBatches(config, root=os.getenv("DATA_ROOT", "/data/cifar10"))
        self.scheme = make_scheme(config)
        self.gradient_cache = {}
        self.loss_cache = {}
        self.last_step = None
        self.eval_loaders = {}
        self.profile = os.getenv("GC_PROFILE_STEPS", "1") == "1"
        print(f"GCWorker {worker_id}: device={self.device}, pid={os.getpid()}", flush=True)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @ray.method(num_returns=2)
    def compute_step(self, epoch, step, weights):
        started = time.perf_counter()
        detailed = self.profile and epoch == 0 and step in (0, 1)

        if self.worker_id == self.config.slow_worker:
            time.sleep(self.config.straggler_delay)

        load_flat_parameters(self.model, weights)
        if detailed:
            self.sync()
        weight_s = time.perf_counter() - started

        data_s = 0.0
        backward_s = 0.0
        global_step = epoch * self.batches.steps_per_epoch + step
        partitions = self.scheme.partitions_to_compute(
            self.worker_id, global_step, self.gradient_cache, self.last_step
        )

        for partition in partitions:
            begin = time.perf_counter()
            inputs, labels = self.batches.get(partition, epoch, step)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            if detailed:
                self.sync()
                data_s += time.perf_counter() - begin

            begin = time.perf_counter()
            self.model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(inputs), labels)
            loss.backward()
            self.gradient_cache[partition] = flatten_gradients(self.model)
            self.loss_cache[partition] = loss.detach()
            if detailed:
                self.sync()
                backward_s += time.perf_counter() - begin

        self.last_step = global_step
        begin = time.perf_counter()
        payload = self.scheme.encode(self.worker_id, self.gradient_cache).detach().cpu()
        loss_value = torch.stack(list(self.loss_cache.values())).mean().item()
        encode_s = time.perf_counter() - begin

        timing = {
            "compute_time_s": time.perf_counter() - started,
            "weight_time_s": weight_s if detailed else None,
            "data_time_s": data_s if detailed else None,
            "backward_time_s": backward_s if detailed else None,
            "encode_transfer_time_s": encode_s if detailed else None,
        }

        if detailed:
            print(
                f"GCWorker {self.worker_id} step={step+1}: "
                + ", ".join(f"{k}={v:.3f}" for k, v in timing.items() if v is not None),
                flush=True,
            )

        return {"loss": loss_value, "profile": timing}, payload

    @torch.inference_mode()
    def evaluate_weights(self, weights, train=False):
        load_flat_parameters(self.model, weights)
        self.model.eval()
        if train not in self.eval_loaders:
            self.eval_loaders[train] = evaluation_loader(self.config, root=os.getenv("DATA_ROOT", "/data/cifar10"), train=train, batch_size=int(os.getenv("GC_EVAL_BATCH", "64")))
        loader = self.eval_loaders[train]
        total_loss = total_correct = total_samples = 0
        label = "train" if train else "test"
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            predictions = self.model(images)
            total_loss += F.cross_entropy(predictions, targets, reduction="sum").item()
            total_correct += (predictions.argmax(dim=1) == targets).sum().item()
            total_samples += len(targets)
            if (i + 1) % 50 == 0:
                print(f"GCWorker {self.worker_id}: {label} eval {total_samples} samples", flush=True)
        self.sync()
        self.model.train()
        return total_loss / total_samples, total_correct / total_samples
