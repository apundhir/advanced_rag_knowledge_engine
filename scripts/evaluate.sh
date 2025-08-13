#!/usr/bin/env bash
set -euo pipefail

conda run -n advanced-rag-knowledge-engine python -m app.evaluation.evaluate --dataset data/eval/qa.jsonl
