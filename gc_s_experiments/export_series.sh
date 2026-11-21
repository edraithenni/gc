#!/usr/bin/env bash
set -euo pipefail
OUT=$(realpath "${1:?Usage: export_series.sh /path/to/series}")
[[ -s "$OUT/runs.tsv" ]]
# Use individual recorded time windows and configuration to identify runs.
# Avoid relying on the six latest experiments or exporting older 100-epoch runs.
RUNS=$(base64 -w0 "$OUT/runs.tsv")
TMP=$(kubectl -n default exec deploy/mlflow -- mktemp -d /tmp/gc-export-XXXXXXXX)
trap 'kubectl -n default exec deploy/mlflow -- rm -rf "$TMP" >/dev/null 2>&1 || true' EXIT
kubectl -n default exec -i deploy/mlflow -- python - "$TMP" "$RUNS" <<'PY'
import base64, csv, io, json, sys
from pathlib import Path
from mlflow import MlflowClient
root = Path(sys.argv[1])
rows = list(csv.DictReader(io.StringIO(base64.b64decode(sys.argv[2]).decode()), delimiter='\t'))
assert rows, 'No completed jobs in runs.tsv'
client = MlflowClient('http://localhost:5000')
exp = client.get_experiment_by_name('gradient-coding-cifar10')
assert exp is not None, 'Experiment not found'
for r in rows:
    query = (f"attributes.start_time >= {r['start_ms']} AND "
             f"attributes.start_time <= {r['end_ms']} AND "
             "attributes.status = 'FINISHED' AND "
             f"params.scheme = '{r['scheme']}' AND "
             f"params.stragglers = '{r['s']}' AND "
             f"params.communication_factor = '{r['m']}' AND "
             f"params.epochs = '{r['epochs']}'")
    runs = client.search_runs([exp.experiment_id], filter_string=query, max_results=100)
    assert len(runs) == 1, f"{r['label']}: expected 1 matching run, found {len(runs)}"
    run = runs[0]
    path = root / r['label']; path.mkdir()
    client.download_artifacts(run.info.run_id, '', str(path))
    (path / 'mlflow_run.json').write_text(json.dumps(dict(run_id=run.info.run_id, params=run.data.params, tags=run.data.tags, latest_metrics=run.data.metrics), indent=2))
    print(r['label'], run.info.run_id)
PY
kubectl -n default exec deploy/mlflow -- tar -czf - -C "$TMP" . | tar -xzf - -C "$OUT"
echo "Exported: $OUT"
