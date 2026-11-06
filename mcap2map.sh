#!/bin/bash
# mcap2map wrapper: runs the pipeline with the venv that has its dependencies.
#   ./mcap2map.sh run BAG.mcap [--out DIR] [--config CFG.json] [--from STAGE] ...
#   ./mcap2map.sh inspect BAG.mcap | status DIR | stages
# Override the interpreter with MCAP2MAP_PYTHON=/path/to/python.
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${MCAP2MAP_PYTHON:-}
if [ -z "$PY" ]; then
  for c in "$HERE/.venv/bin/python" /home/matyko/Downloads/baross_map/.venv/bin/python; do
    [ -x "$c" ] && PY=$c && break
  done
fi
[ -z "$PY" ] && { echo "no Python with the requirements found; set MCAP2MAP_PYTHON (see README)"; exit 1; }
cd "$HERE" && exec "$PY" -m mcap2map "$@"
