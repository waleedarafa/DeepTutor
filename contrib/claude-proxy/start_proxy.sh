#!/usr/bin/env bash
# Launch the claude -p -> OpenAI proxy that backs DeepTutor's LLM.
# Usage: ./start_proxy.sh            (defaults to the haiku model)
#        CLAUDE_PROXY_MODEL=sonnet ./start_proxy.sh
#        CLAUDE_PROXY_MODEL=opus   ./start_proxy.sh   (best quality, priciest)
set -euo pipefail
cd "$(dirname "$0")"

export CLAUDE_PROXY_MODEL="${CLAUDE_PROXY_MODEL:-haiku}"
export CLAUDE_PROXY_PORT="${CLAUDE_PROXY_PORT:-8088}"

# Stop any previous instance, then run in the background, logging to proxy.log.
pkill -f claude_proxy.py 2>/dev/null || true
sleep 1
nohup python3 claude_proxy.py > proxy.log 2>&1 &
sleep 2

if curl -fsS "http://127.0.0.1:${CLAUDE_PROXY_PORT}/v1/models" >/dev/null; then
  echo "claude-proxy up on http://127.0.0.1:${CLAUDE_PROXY_PORT} (model=${CLAUDE_PROXY_MODEL}, pid=$(pgrep -f claude_proxy.py))"
else
  echo "claude-proxy failed to start; see proxy.log" >&2
  exit 1
fi
