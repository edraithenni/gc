#!/usr/bin/env bash
set -euo pipefail

kubectl delete rayjob gc-lab-v1-run --ignore-not-found
kubectl apply -f k8s/rayjob.yaml
kubectl get rayjob gc-lab-v1-run -w
