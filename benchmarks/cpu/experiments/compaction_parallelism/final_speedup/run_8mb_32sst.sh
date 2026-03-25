#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_final_speedup_case.sh" --sst-size-mb 8 --input-ssts 32 "$@"
