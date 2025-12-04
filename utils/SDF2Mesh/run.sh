#!/bin/bash
BASE_DIR="../../levelset"
OUTPUT_DIR="../../plys"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TOTAL=600
NUM_THREADS=30

PER_THREAD=$((TOTAL / NUM_THREADS))

for i in $(seq 0 $((NUM_THREADS - 1))); do
    START=$((i * PER_THREAD))
    END=$(((i + 1) * PER_THREAD))
    echo "Starting thread $i: processing $START to $END"
    uv run "$SCRIPT_DIR/all2mesh.py" "$BASE_DIR" "$OUTPUT_DIR" $START $END &
done

wait

echo "All threads completed!"
