#!/usr/bin/env bash
export PATH="/opt/homebrew/bin:$PATH"
set -euo pipefail

command -v helm >/dev/null || { echo "helm is required"; exit 1; }
command -v kubectl >/dev/null || { echo "kubectl is required"; exit 1; }

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true
helm repo update

kubectl create namespace ray-system --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install kuberay-operator \
  kuberay/kuberay-operator \
  --namespace ray-system \
  --version 1.7.0 \
  --wait

kubectl get pods -n ray-system
