#!/usr/bin/env bash
set -e

echo "Pipeline container ready."
echo "Use Airflow DAG to run steps, or run manually:"
echo "  python -m mlops build-series"
echo "  python -m mlops build-features"
echo "  python -m mlops train"
echo "  python -m mlops score"
echo "  python -m mlops persist"
sleep infinity
