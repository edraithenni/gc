#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd -- "$(dirname -- "$0")" && pwd)
exec bash "$DIR/run_series.sh" timing_s3 "${EPOCHS:-5}" classic:3:2 cce_gc_cr:3:2 pgc_cr:3:2
