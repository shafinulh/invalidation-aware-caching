#!/usr/bin/env bash

# GPU compaction sweep defaults — sources shared compaction_defaults.sh.

_GPU_DEFAULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SHARED_DEFAULTS="${_GPU_DEFAULTS_DIR}/../../common/compaction_defaults.sh"
if [[ -f "${_SHARED_DEFAULTS}" ]]; then
  # shellcheck disable=SC1090
  source "${_SHARED_DEFAULTS}"
fi
