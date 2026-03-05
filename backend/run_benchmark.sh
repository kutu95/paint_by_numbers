#!/usr/bin/env bash
# Run recipe benchmark/quality harness using a venv (avoids Homebrew PEP 668).
# Usage: from repo root, ./backend/run_benchmark.sh [capture|benchmark|regression|profile] [--quick] [--library NAME]

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv"

if [[ ! -d "$VENV" ]]; then
  echo "Creating venv at $VENV ..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r "$REPO_ROOT/backend/requirements.txt"
fi

exec "$VENV/bin/python" -m backend.tests.benchmark_recipe_quality "$@"
