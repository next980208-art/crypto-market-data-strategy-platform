#!/usr/bin/env bash
set -euo pipefail

# Run the existing Airflow DAG sequentially for faster one-time historical catch-up.
# The DAG still performs planning, collection, PyFlink processing, and quality checks.
RUN_COUNT="${1:-1}"
DAG_ID="${DAG_ID:-btcusdt_usdm_historical_backfill}"
CHUNKS_PER_TRIGGER="${CHUNKS_PER_TRIGGER:-8}"

if ! [[ "$RUN_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUN_COUNT must be a positive integer." >&2
    exit 2
fi

if ! [[ "$CHUNKS_PER_TRIGGER" =~ ^[1-8]$ ]]; then
    echo "CHUNKS_PER_TRIGGER must be between 1 and 8." >&2
    exit 2
fi

for ((index = 1; index <= RUN_COUNT; index++)); do
    execution_date="$(date -u -d "+${index} minutes" '+%Y-%m-%dT%H:%M:%S+00:00')"
    echo "[$index/$RUN_COUNT] Starting $DAG_ID at $execution_date with $CHUNKS_PER_TRIGGER chunks."
    airflow dags test "$DAG_ID" "$execution_date" --conf "{\"chunks_per_trigger\":$CHUNKS_PER_TRIGGER}"
done
