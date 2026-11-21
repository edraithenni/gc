
#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-$HOME/Desktop/gc_store_2g_results}"
mkdir -p "$OUT/diagnostics"
date +%s%3N > "$OUT/start_ms.txt"

STAMP=$(date +%Y%m%d-%H%M%S)

HEAD=$(kubectl get pod -l ray.io/node-type=head \
    -o jsonpath='{.items[0].metadata.name}')

for MODE in dgd classic cce_gc_cr cce_gc_fr pgc_cr pgc_fr
do
    SCHEME="$MODE"
    S=1

    if [ "$MODE" = "dgd" ]; then
        SCHEME=classic
        S=0
    fi

    JOB="gc-${STAMP}-${MODE//_/-}"

    echo "========== $MODE =========="

    sed \
        -e "s/^  name:.*/  name: $JOB/" \
        -e "s/GC_SCHEME:.*/GC_SCHEME: \"$SCHEME\"/" \
        -e "s/GC_STRAGGLERS:.*/GC_STRAGGLERS: \"$S\"/" \
        -e 's/CCE_M:.*/CCE_M: "2"/' \
        -e 's/EPOCHS:.*/EPOCHS: "5"/' \
        -e 's/SLOW_WORKER:.*/SLOW_WORKER: "-1"/' \
        -e 's/STRAGGLER_DELAY:.*/STRAGGLER_DELAY: "0"/' \
        k8s/rayjob.yaml | kubectl create -f -

    while true; do
        STATUS=$(kubectl get rayjob "$JOB" \
            -o jsonpath='{.status.jobStatus}/{.status.jobDeploymentStatus}')

        case "$STATUS" in
            SUCCEEDED/*)
                echo "$MODE completed"
                break
                ;;
            FAILED/*|STOPPED/*|*/Failed)
                echo "$MODE failed"
                exit 1
                ;;
        esac

        sleep 10
    done

    kubectl exec "$HEAD" -c ray-head -- \
        ray memory --stats-only \
        | tee "$OUT/diagnostics/${MODE}_memory.txt"

done

echo "All six experiments completed"
