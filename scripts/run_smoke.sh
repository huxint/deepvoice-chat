#!/usr/bin/env bash
set -euo pipefail

uv run voiceagent doctor
uv run pytest
