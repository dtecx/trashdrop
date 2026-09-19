#!/usr/bin/env bash
# Thin wrapper so `./scripts/bootstrap_model.sh` keeps working on macOS/Linux.
# The real implementation is Python, because most of the team is on Windows.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/bootstrap_model.py" "$@"
