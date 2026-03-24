#!/usr/bin/env bash
set -euo pipefail

# Repo root = parent of project/ (where pyproject.toml and .venv live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR=".venv"
HACKATHON_JSON="$REPO_ROOT/hackathon.json"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-0}"

echo "== configure =="

# Goal of this step:
# - Install dependencies
# - Download / materialize model artifacts from hackathon.json (optional)

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtualenv at $VENV_DIR"
  USE_SYSTEM_SITE_PACKAGES_NORM="$(printf '%s' "$USE_SYSTEM_SITE_PACKAGES" | tr '[:upper:]' '[:lower:]')"
  case "$USE_SYSTEM_SITE_PACKAGES_NORM" in
    1|true|yes)
      python3 -m venv "$VENV_DIR" --system-site-packages
      ;;
    0|false|no|"")
      python3 -m venv "$VENV_DIR"
      ;;
    *)
      echo "ERROR: USE_SYSTEM_SITE_PACKAGES must be one of: 1,true,yes,0,false,no (got: $USE_SYSTEM_SITE_PACKAGES)" >&2
      exit 1
      ;;
  esac
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install -U pip >/dev/null

if [[ -f requirements.txt ]]; then
  echo "Installing Python dependencies from requirements.txt"
  "$PY" -m pip install -r requirements.txt
elif [[ -f pyproject.toml ]]; then
  echo "Installing project and dependencies from pyproject.toml"
  "$PY" -m pip install -e .
else
  echo "No requirements.txt or pyproject.toml found. Skipping dependency install."
fi

echo "Ensuring ipykernel is available for notebooks"
"$PY" -m pip install ipykernel

echo "OK"
