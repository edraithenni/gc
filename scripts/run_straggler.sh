#!/usr/bin/env bash
set -euo pipefail

kubectl delete rayjob gc-lab-v1-straggler --ignore-not-found
kubectl apply -f k8s/rayjob-straggler.yaml
kubectl get rayjob gc-lab-v1-straggler -w
