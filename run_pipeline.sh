#!/usr/bin/env bash
# Run the whole pipeline (Colab / Linux / macOS). All options are passed to run_pipeline.py:
#   bash run_pipeline.sh                                   # env + smoke test + full benchmark
#   bash run_pipeline.sh --quick                           # ~2 min check
#   bash run_pipeline.sh --out /content/drive/MyDrive/CRB  # results on a mounted Google Drive
#   bash run_pipeline.sh --with-studies --with-scania      # everything
#   bash run_pipeline.sh --help                            # all options
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
exec "$PY" run_pipeline.py "$@"
