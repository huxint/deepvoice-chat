#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && "${VOICEAGENT_FORCE_UV:-0}" != "1" ]]; then
  voiceagent doctor
  python -m pytest
else
  uv run voiceagent doctor
  uv run pytest
fi
