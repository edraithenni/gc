import argparse
import json
from pathlib import Path
from statistics import mean

import mlflow


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def log_experiment(results, experiment, tracking_uri, name):
    config = json.loads((results / "config.json").read_text(encoding="utf-8"))
    epochs = read_jsonl(results / "epoch_metrics.jsonl")
    steps = read_jsonl(results / "step_metrics.jsonl")
    scheme = config.get("scheme", config.get("gc_scheme", "unknown"))
    n = int(config.get("num_workers", config.get("workers", 3)))
    s = int(config.get("stragglers", config.get("stragglers_tolerated", 1)))
    steps_per_epoch = int(config["steps_per_epoch"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name or f"{scheme}-n{n}-s{s}-{results.name}"):
        mlflow.log_params({
            key: value
            for key, value in config.items()
            if isinstance(value, (str, int, float, bool))
        })
        mlflow.set_tags({"coding_scheme": scheme, "dataset": "CIFAR-10", "model": "ResNet-18"})

        for record in epochs:
            number = int(record["epoch"])
            mlflow.log_metrics({
                "quality/test_accuracy": record["test_accuracy"],
                "quality/test_loss": record["test_loss"],
                "quality/train_loss_reported": record["train_loss"],
                "time/epoch_s": record["epoch_time_s"],
            }, step=number)

        for record in steps:
            position = (int(record["epoch"]) - 1) * steps_per_epoch + int(record["step"])
            timings = record.get("worker_timings", [])
            accepted = {int(w["worker_id"]): w for w in timings}
            metrics = {
                "time/step_wait_s": record["wait_time_s"],
                "resilience/stale_results": record.get("stale_results_seen", 0),
                "resilience/selected_workers": len(record["responders"]),
                "communication/accepted_payload_mb": sum(
                    w["gradient_bytes"] for w in timings
                ) / (1024 * 1024),
            }
            if timings:
                metrics["time/mean_selected_compute_s"] = mean(w["compute_time_s"] for w in timings)
                metrics["time/mean_selected_response_s"] = mean(w["response_time_s"] for w in timings)
                metrics["time/max_selected_response_s"] = max(w["response_time_s"] for w in timings)
            computed = [w["computed_partitions"] for w in timings if "computed_partitions" in w]
            ages = [w["max_cache_age_steps"] for w in timings if "max_cache_age_steps" in w]
            if computed:
                metrics["compute/mean_partitions_per_selected_worker"] = mean(computed)
            if ages:
                metrics["pipeline/max_cache_age_steps"] = max(ages)
            for worker_id in range(n):
                metrics[f"workers/worker_{worker_id}_selected"] = int(worker_id in record["responders"])
                if worker_id in accepted:
                    metrics[f"workers/worker_{worker_id}_response_s"] = accepted[worker_id]["response_time_s"]
            mlflow.log_metrics(metrics, step=position)

        if epochs:
            mlflow.log_metrics({
                "summary/best_test_accuracy": max(row["test_accuracy"] for row in epochs),
                "summary/total_epoch_time_s": sum(row["epoch_time_s"] for row in epochs),
                "summary/completed_epochs": len(epochs),
            })
        for filename in ("config.json", "epoch_metrics.jsonl", "step_metrics.jsonl"):
            path = results / filename
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="raw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="Gradient Coding - CIFAR10")
    parser.add_argument("--name")
    args = parser.parse_args()
    log_experiment(args.results, args.experiment, args.tracking_uri, args.name)


if __name__ == "__main__":
    main()
