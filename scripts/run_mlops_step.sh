#!/usr/bin/env bash
set -euo pipefail

STEP="${1:-}"
if [[ -z "$STEP" ]]; then
  echo "Usage: run_mlops_step.sh <step>"
  exit 1
fi

VENV_DIR="/opt/airflow/.venvs/project"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  python -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --no-cache-dir -r /opt/airflow/requirements.txt
fi

export PYTHONPATH=/opt/airflow/src
exec "$PYTHON_BIN" -m mlops "$STEP"
