#!/usr/bin/env bash
# Run the full pipeline end-to-end.
set -euo pipefail
PY=/opt/anaconda3/envs/statarb/bin/python
cd "$(dirname "$0")/.."
$PY scripts/01_pull_data.py "$@"
$PY scripts/02_build_features.py
$PY scripts/03_run_v1_pairs.py
$PY scripts/04_run_v2_graph.py
$PY scripts/05_run_v3_altdata.py
$PY scripts/06_compare_results.py
