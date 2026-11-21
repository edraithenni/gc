#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd -- "$(dirname -- "$0")" && pwd)
exec bash "$DIR/run_series.sh" convergence_cr 100 pgc_cr:1:2 pgc_cr:2:2 pgc_cr:3:2
