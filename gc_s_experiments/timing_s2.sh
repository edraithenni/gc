#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd -- "$(dirname -- "$0")" && pwd)
exec bash "$DIR/run_series.sh" timing_s2 "${EPOCHS:-5}" classic:2:2 cce_gc_cr:2:2 pgc_cr:2:2 pgc_fr:2:2
