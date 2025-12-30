#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PORT:-}" ]]; then
  exec python3 "${ROOT_DIR}/web/server.py" --port "${PORT}"
else
  exec python3 "${ROOT_DIR}/web/server.py"
fi
