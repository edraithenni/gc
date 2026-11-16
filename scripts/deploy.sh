#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/raycluster.yaml
kubectl get raycluster gc-lab-v1
kubectl get pods -o wide
