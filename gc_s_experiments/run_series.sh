#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Arguments: series epochs scheme:s:m ...
SERIES=$1; EPOCHS=$2; shift 2
TEMPLATE=k8s/rayjob.yaml
NS=default
CLUSTER=gc-lab-v1
[[ -f "$TEMPLATE" ]] || { echo 'Run from the repository root (k8s/rayjob.yaml required).'; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EPOCHS'; exit 1; }

# Protect against simultaneous runs of these scripts (not older benchmark.sh).
exec 9>"$HOME/.gc_experiment.lock"
flock -n 9 || { echo 'Another new experiment series is running.'; exit 1; }
# Do not start if an existing RayJob on this cluster is still active.
kubectl -n "$NS" get rayjobs -o json | python3 -c '
import json, sys
active=[]
for j in json.load(sys.stdin)["items"]:
    if j.get("spec", {}).get("clusterSelector", {}).get("ray.io/cluster") != "gc-lab-v1":
        continue
    s=j.get("status", {})
    if s.get("jobStatus") not in ("SUCCEEDED", "FAILED", "STOPPED") and s.get("jobDeploymentStatus") not in ("Complete", "Failed", "Suspended"):
        active.append(j["metadata"]["name"])
if active:
    sys.exit("Active RayJobs: " + ", ".join(active))
'

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$HOME/Desktop/gc_${SERIES}_${STAMP}_$$"
mkdir -p "$OUT/manifests" "$OUT/diagnostics"
printf 'label\tscheme\ts\tm\tepochs\tjob\tstart_ms\tend_ms\n' > "$OUT/runs.tsv"
HEAD=$(kubectl -n "$NS" get pods -l "ray.io/cluster=$CLUSTER,ray.io/node-type=head" -o jsonpath='{.items[0].metadata.name}')
[[ -n "$HEAD" ]] || { echo 'Ray Head not found'; exit 1; }
kubectl -n "$NS" get raycluster "$CLUSTER" -o yaml > "$OUT/raycluster.yaml"
echo "Results and manifests: $OUT"
for SPEC in "$@"; do
    IFS=: read -r SCHEME S M <<< "$SPEC"
    [[ "$S" =~ ^[0-5]$ && "$M" =~ ^[1-6]$ ]] || { echo "Invalid spec: $SPEC"; exit 1; }
    case "$SCHEME" in
        classic|pgc_cr) ;;
        cce_gc_cr) (( S+M <= 6 )) || exit 1 ;;
        cce_gc_fr) (( S+M <= 6 && 6 % (S+M) == 0 )) || exit 1 ;;
        pgc_fr) (( 6 % (S+1) == 0 )) || exit 1 ;;
        *) echo "Unknown scheme: $SCHEME"; exit 1 ;;
    esac
    LABEL="${SCHEME}_s${S}_m${M}"
    JOB="gc-${STAMP}-$$-${SCHEME//_/-}-s${S}-m${M}"
    echo "========== $LABEL ($EPOCHS epochs) =========="
    # Use a generic metadata.name replacement, regardless of its current value.
    # Pin all experiment parameters to the settings of the previous series.
    python3 - "$TEMPLATE" "$JOB" "$SCHEME" "$S" "$M" "$EPOCHS" <<'PY' > "$OUT/manifests/$LABEL.yaml"
import re, sys
path, job, scheme, s, m, epochs = sys.argv[1:]
text = open(path).read()
text, count = re.subn(r'^  name:[^\n]*$', '  name: ' + job, text, count=1, flags=re.M)
assert count == 1, 'metadata.name not found (check manifest indentation)'
settings = dict(GC_SCHEME=scheme, NUM_WORKERS='6', NUM_PARTITIONS='6',
    GC_STRAGGLERS=s, CCE_M=m, GC_GPU_FRACTION='1', GC_PROFILE_STEPS='1',
    GC_STEP_TIMEOUT_S='180', GC_EVAL_TIMEOUT_S='600', GC_EVAL_BATCH='64',
    GC_TRAIN_EVAL_INTERVAL='5', GC_SEED='1234', SEED='1234',
    TRAIN_LIMIT='24000', TEST_LIMIT='10000', BATCH_SIZE='40', EPOCHS=epochs,
    LR='0.05', MOMENTUM='0.9', WEIGHT_DECAY='0.0005', EMA_DECAY='0.99',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', AGGREGATOR_THREADS='1',
    SLOW_WORKER='-1', STRAGGLER_DELAY='0',
    MLFLOW_TRACKING_URI='http://mlflow.default.svc.cluster.local:5000')
for key, value in settings.items():
    text, count = re.subn(r'^([ \t]+)' + key + r':[^\n]*$',
        lambda match: match[1] + key + ': "' + value + '"', text, flags=re.M)
    assert count == 1, f'{key} missing or duplicated in {path}'
print(text, end='')
PY
    kubectl -n "$NS" exec "$HEAD" -c ray-head -- ray memory --stats-only > "$OUT/diagnostics/${LABEL}_before.txt" 2>&1 || true
    START=$(date +%s%3N)
    kubectl -n "$NS" create -f "$OUT/manifests/$LABEL.yaml"
    BEGIN=$SECONDS
    while true; do
        STATUS=$(kubectl -n "$NS" get rayjob "$JOB" -o jsonpath='{.status.jobStatus}/{.status.jobDeploymentStatus}')
        case "$STATUS" in
            FAILED/*|STOPPED/*|*/Failed|*/Suspended)
                echo "$JOB failed: $STATUS" >&2; exit 1 ;;
            SUCCEEDED/*) break ;;
        esac
        if (( SECONDS-BEGIN > ${JOB_TIMEOUT_S:-43200} )); then
            echo "Timeout waiting for $JOB; job was not deleted." >&2; exit 1
        fi
        sleep 10
    done
    END=$(date +%s%3N)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$SCHEME" "$S" "$M" "$EPOCHS" "$JOB" "$START" "$END" >> "$OUT/runs.tsv"
    kubectl -n "$NS" exec "$HEAD" -c ray-head -- ray memory --stats-only > "$OUT/diagnostics/${LABEL}_after.txt" 2>&1 || true
    echo "$LABEL completed"
done
echo "Completed: $OUT"
echo "Export: bash gc_s_experiments/export_series.sh '$OUT'"
