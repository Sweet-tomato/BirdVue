#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export BIRD_CONFIG="${BIRD_CONFIG:-config/pipeline_config.json}"
export BIRD_MAX_CONCURRENCY="${BIRD_MAX_CONCURRENCY:-8}"
export BIRD_QUEUE_WORKERS="${BIRD_QUEUE_WORKERS:-8}"
export BIRD_MAX_PENDING="${BIRD_MAX_PENDING:-64}"
export BIRD_LOCAL_CONCURRENCY="${BIRD_LOCAL_CONCURRENCY:-2}"
export DOUBAO_REASONING_EFFORT="${DOUBAO_REASONING_EFFORT:-minimal}"
export DOUBAO_MAX_OUTPUT_TOKENS="${DOUBAO_MAX_OUTPUT_TOKENS:-2048}"
export PYTHONPATH="$PWD/pipeline:${PYTHONPATH:-}"

# Runtime config (systemd EnvironmentFile 使用同一份文件，保证手工启动与守护进程一致)。
# config/service.env 权限 600，含密钥，不入库；示例见 config/service.env.example。
for env_file in config/service.env config/doubao.env; do
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

# Safe defaults for the current endpoint/model; server can override.
export OPENAI_BASE_URL1="${OPENAI_BASE_URL1:-https://ark.cn-beijing.volces.com/api/v3}"
export OPENAI_VLM_MODEL1="${OPENAI_VLM_MODEL1:-doubao-seed-2-0-lite-260428}"

exec uvicorn api.bird_frames_service:app \
  --host "${BIRD_HOST:-0.0.0.0}" \
  --port "${BIRD_PORT:-8000}" \
  --workers "${BIRD_WORKERS:-1}" \
  --timeout-keep-alive 75
