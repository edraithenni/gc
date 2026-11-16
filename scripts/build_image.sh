#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker build \
  --target mlflow \
  -t gc-mlflow:3.6.0 .

docker build \
  --target ray \
  -t gc-lab-v1:cuda .

TMP_IMAGE=$(mktemp --suffix=.tar)
trap 'rm -f "$TMP_IMAGE"' EXIT

docker save \
  gc-mlflow:3.6.0 \
  gc-lab-v1:cuda \
  -o "$TMP_IMAGE"

sudo /usr/local/bin/k3s ctr \
  -n k8s.io images import "$TMP_IMAGE"

echo "Images imported into K3s."
