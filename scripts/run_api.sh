#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for local dev
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-5000}

uvicorn app.api.main:app --host "$HOST" --port "$PORT" --reload
