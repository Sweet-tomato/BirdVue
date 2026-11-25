#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export BIRD_CONFIG="${BIRD_CONFIG:-config/pipeline_config.json}"
export BIRD_AUDIO_CONCURRENCY="${BIRD_AUDIO_CONCURRENCY:-2}"
export BIRD_MAX_AUDIO_BYTES="${BIRD_MAX_AUDIO_BYTES:-20971520}"
export PYTHONPATH="$PWD/pipeline:${PYTHONPATH:-}"

for env_file in config/service.env config/doubao.env; do
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

exec /root/venv/bin/uvicorn api.bird_audio_service:app \
  --host "${BIRD_AUDIO_HOST:-0.0.0.0}" \
  --port "${BIRD_AUDIO_PORT:-8001}" \
  --workers 1 \
  --timeout-keep-alive 75
