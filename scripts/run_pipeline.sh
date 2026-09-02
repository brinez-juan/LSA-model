#!/usr/bin/env bash
# run_pipeline.sh — run the full data pipeline end-to-end.
#
# Requires data/raw/genius_raw.csv to exist. Stops at the first failing step.

set -euo pipefail

echo "── Phase 1: preprocess ─────────────────────────"
make preprocess

echo "── Phase 2: pseudo-label ──────────────────────"
make label

echo "── Phase 2b: train/val/test split ─────────────"
make split

echo "── Pipeline complete ──────────────────────────"
