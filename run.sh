#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Launch the TTS lab server.
#   PORT       listen port            (default 8000)
#   CLONE_REF  reference voice WAV    (default refs/voice.wav)
# ROCm/gfx1100 perf env lives in env.rocm.sh (harmless on Mac).
source ./env.rocm.sh
export CLONE_REF="${CLONE_REF:-refs/voice.wav}"
PORT="${PORT:-8000}"
exec uvicorn server:app --host 0.0.0.0 --port "$PORT"
