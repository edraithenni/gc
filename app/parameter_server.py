
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from uuid import uuid4

import mlflow
import ray
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from coding import make_scheme
from config import Config
from coordinator import ResponseCollector
from model import assign_flat_gradient, build_model, flatten_parameters
from worker import GCWorker


def write_jsonl(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


@ray.remote(num_cpus=1, resources={"gc_parameter_server": 1})
class ParameterServer:
    def run_training(self):
        config = Config.from_env()
        torch.set_num_threads(int(os.getenv("AGGREGATOR_THREADS", "1")))
        torch.manual_seed(config.seed)
        scheme = make_scheme(config)
        model = build_model()
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.momentum > 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        parameter_count = sum(p.numel() for p in model.parameters())
        steps_per_epoch = config.train_limit // config.num_partitions // config.batch_size
        gpu_fraction = float(os.getenv("GC_GPU_FRACTION", "1"))
        interval = int(os.getenv("GC_TRAIN_EVAL_INTERVAL", "5"))
        run_name = f"{config.scheme}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        results = Path(os.getenv("RESULTS_DIR", "/data/results")) / run_name
        results.mkdir(parents=True, exist_ok=True)
        for name in ("step_metrics.jsonl", "epoch_metrics.jsonl"):
            (results / name).write_text("", encoding="utf-8")
        settings = asdict(config)
        settings.update({"parameter_count": parameter_count, "steps_per_epoch": steps_per_epoch,
                         "gpu_fraction": gpu_fraction, "eval_device": "worker_cuda",
                         "train_eval_interval": interval})
        (results / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("gradient-coding-cifar10")
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(settings)
        mlflow.set_tags({"hardware": "RTX 2070 shared CUDA", "platform": "k3s-kuberay",
                         "scheme": config.scheme, "dataset": "CIFAR-10"})
        status = "FAILED"
        try:
            workers = [GCWorker.options(num_cpus=1, num_gpus=gpu_fraction).remote(i, config)
                       for i in range(config.num_workers)]
            collector = ResponseCollector(workers)
            weights = ray.put(flatten_parameters(model))
            collector.submit(range(config.num_workers), 0, 0, weights)
            print(f"RUN {run_name}: {config.scheme}, workers={config.num_workers}, "
                  f"steps/epoch={steps_per_epoch}, epochs={config.epochs}, GPU/actor={gpu_fraction}", flush=True)
            for epoch in range(config.epochs):
                epoch_started = time.perf_counter()
                training_loss = 0.0
                waits, computes, decode_times, update_times = [], [], [], []
                stale_total = 0
                for step in range(steps_per_epoch):
                    step_started = time.perf_counter()
                    responses, loss, timing = collector.collect(epoch, step, weights, scheme)
                    started = time.perf_counter()
                    gradient = scheme.decode(responses, parameter_count)
                    decode_s = time.perf_counter() - started
                    started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    assign_flat_gradient(model, gradient)
                    optimizer.step()
                    ema_model.update_parameters(model)
                    update_s = time.perf_counter() - started
                    training_loss += loss
                    waits.append(timing["wait_time_s"])
                    computes.extend(t["compute_time_s"] for t in timing["worker_timings"])
                    decode_times.append(decode_s)
                    update_times.append(update_s)
                    stale_total += timing["stale_results_seen"]
                    if step + 1 < steps_per_epoch:
                        weights = ray.put(flatten_parameters(model))
                        collector.submit(responses, epoch, step + 1, weights)
                    write_jsonl(results / "step_metrics.jsonl", {
                        "epoch": epoch + 1, "step": step + 1, "responders": sorted(responses),
                        "decode_time_s": decode_s, "update_time_s": update_s,
                        "step_time_s": time.perf_counter() - step_started, **timing,
                    })
                    if (step + 1) % 10 == 0 or step + 1 == steps_per_epoch:
                        global_step = epoch * steps_per_epoch + step + 1
                        try:
                            mlflow.log_metrics({
                                "progress/completed_steps": global_step,
                                "steps/mean_wait_s": mean(waits[-10:]),
                                "steps/mean_decode_s": mean(decode_times[-10:]),
                                "steps/mean_update_s": mean(update_times[-10:]),
                            }, step=global_step)
                        except Exception as error:
                            print(f"WARNING: Step metrics upload failed: {error}", flush=True)
                        print(f"EPOCH {epoch+1:03d}/{config.epochs} STEP {step+1:03d}/{steps_per_epoch} "
                              f"loss={training_loss/(step+1):.4f} wait={mean(waits):.3f}s "
                              f"worker={mean(computes):.3f}s decode={mean(decode_times):.3f}s "
                              f"update={mean(update_times):.3f}s stale={stale_total} "
                              f"responders={sorted(responses)}", flush=True)
                train_time_s = time.perf_counter() - epoch_started
                eval_started = time.perf_counter()
                eval_worker = workers[next(iter(responses))]
                print(f"EPOCH {epoch+1:03d}: GPU evaluation started (worker actor)", flush=True)
                train_eval_loss = None
                if interval and (epoch + 1) % interval == 0:
                    ref = eval_worker.evaluate_weights.remote(ray.put(flatten_parameters(model)), train=True)
                    ready, _ = ray.wait([ref], timeout=float(os.getenv("GC_EVAL_TIMEOUT_S", "600")))
                    if not ready:
                        raise TimeoutError("Train evaluation exceeded GC_EVAL_TIMEOUT_S")
                    train_eval_loss, _ = ray.get(ref)
                ref = eval_worker.evaluate_weights.remote(ray.put(flatten_parameters(ema_model.module)), train=False)
                ready, _ = ray.wait([ref], timeout=float(os.getenv("GC_EVAL_TIMEOUT_S", "600")))
                if not ready:
                    raise TimeoutError("Test evaluation exceeded GC_EVAL_TIMEOUT_S")
                test_loss, test_accuracy = ray.get(ref)
                eval_time_s = time.perf_counter() - eval_started
                scheduler.step()
                epoch_time_s = time.perf_counter() - epoch_started
                record = {"epoch": epoch+1, "train_loss": training_loss / steps_per_epoch,
                          "train_eval_loss": train_eval_loss, "test_loss": test_loss,
                          "test_accuracy": test_accuracy, "train_time_s": train_time_s,
                          "eval_time_s": eval_time_s, "epoch_time_s": epoch_time_s,
                          "mean_step_wait_s": mean(waits), "mean_worker_compute_s": mean(computes),
                          "mean_decode_s": mean(decode_times), "mean_update_s": mean(update_times),
                          "stale_results": stale_total}
                write_jsonl(results / "epoch_metrics.jsonl", record)
                mlflow.log_metrics({k: v for k, v in record.items() if k != "epoch" and v is not None}, step=epoch + 1)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    try:
                        for name in ("config.json", "step_metrics.jsonl", "epoch_metrics.jsonl"):
                            mlflow.log_artifact(str(results / name), artifact_path="results")
                        print(f"MLflow: artifacts checkpointed at epoch {epoch + 1}", flush=True)
                    except Exception as error:
                        print(f"WARNING: MLflow checkpoint failed: {error}", flush=True)
                print(f"EPOCH {epoch+1:03d}/{config.epochs} DONE: train_loss={record['train_loss']:.4f} "
                      f"test_loss={test_loss:.4f} test_acc={test_accuracy*100:.2f}% "
                      f"train={train_time_s:.1f}s eval={eval_time_s:.1f}s total={epoch_time_s:.1f}s", flush=True)
                if epoch + 1 < config.epochs:
                    weights = ray.put(flatten_parameters(model))
                    collector.submit(responses, epoch + 1, 0, weights)
            status = "FINISHED"
        finally:
            try:
                for name in ("config.json", "step_metrics.jsonl", "epoch_metrics.jsonl"):
                    mlflow.log_artifact(str(results / name), artifact_path="results")
            except Exception as error:
                print(f"WARNING: Final metrics upload failed: {error}", flush=True)
            mlflow.end_run(status=status)
