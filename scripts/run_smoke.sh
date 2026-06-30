#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate the conda environment first: conda activate voiceagent" >&2
  exit 1
fi

voiceagent doctor
python -m pytest
