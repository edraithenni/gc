#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd -- "$(dirname -- "$0")" && pwd)
exec bash "$DIR/run_series.sh" convergence_fr 100 pgc_fr:1:2 pgc_fr:2:2
