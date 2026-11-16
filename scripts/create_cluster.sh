#!/usr/bin/env bash
export PATH="/opt/homebrew/bin:$PATH"
set -euo pipefail

CLUSTER_NAME="gc-lab-v1"
NODE_IMAGE="kindest/node:v1.34.3"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

command -v kind >/dev/null || { echo "kind is required"; exit 1; }
command -v kubectl >/dev/null || { echo "kubectl is required"; exit 1; }
command -v docker >/dev/null || { echo "docker is required"; exit 1; }

mkdir -p "$ROOT/data/cifar10" "$ROOT/data/results"

if kind get clusters | grep -qx "$CLUSTER_NAME"; then
  echo "kind cluster $CLUSTER_NAME already exists"
  exit 0
fi

CONFIG="$(mktemp)"
trap 'rm -f "$CONFIG"' EXIT

cat > "$CONFIG" <<YAML
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    image: ${NODE_IMAGE}
    extraMounts:
      - hostPath: ${ROOT}/data
        containerPath: /gc-lab-v1/data

  - role: worker
    image: ${NODE_IMAGE}
    extraMounts:
      - hostPath: ${ROOT}/data
        containerPath: /gc-lab-v1/data

  - role: worker
    image: ${NODE_IMAGE}
    extraMounts:
      - hostPath: ${ROOT}/data
        containerPath: /gc-lab-v1/data

  - role: worker
    image: ${NODE_IMAGE}
    extraMounts:
      - hostPath: ${ROOT}/data
        containerPath: /gc-lab-v1/data

  - role: worker
    image: ${NODE_IMAGE}
    extraMounts:
      - hostPath: ${ROOT}/data
        containerPath: /gc-lab-v1/data
YAML

kind create cluster --name "$CLUSTER_NAME" --config "$CONFIG" --wait 5m

kubectl label node ${CLUSTER_NAME}-worker gc-role=aggregator --overwrite
kubectl label node ${CLUSTER_NAME}-worker2 gc-role=training --overwrite
kubectl label node ${CLUSTER_NAME}-worker3 gc-role=training --overwrite
kubectl label node ${CLUSTER_NAME}-worker4 gc-role=training --overwrite
kubectl label node ${CLUSTER_NAME}-control-plane gc-role=control-plane --overwrite

kubectl get nodes --show-labels
