#!/usr/bin/env bash
# run.sh — Launch the best available server for a given model.
#
# Usage:
#   ./run.sh [MODEL]
#
# MODEL aliases (case-insensitive, partial match supported):
#   qwen3-coder-next   → servers/qwen3-coder-next-fp8/run-v23.sh  (~46-47 tok/s)
#   qwen3-next         → servers/qwen3-next-nvfp4/run.sh           (~65-70 tok/s)
#   qwen35             → servers/qwen35-122b-a10b-nvfp4/run.sh
#   qwen3-coder-nvfp4  → servers/qwen3-coder-next-nvfp4/run.sh
#   minimax            → servers/minimax-m2.5-reap-139b-nvfp4/run-v23.sh
#
# If no MODEL is given, defaults to qwen3-coder-next (recommended).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Known model-serving container names — stopped before launching a new one
MODEL_CONTAINERS=(
  qwen3-fp8-server
  qwen3-next-server
  qwen35-122b-server
  qwen3-coder-nvfp4-server
  minimax-m2.5-server
)

stop_model_containers() {
  for cname in "${MODEL_CONTAINERS[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^${cname}$"; then
      echo "==> Stopping running container '$cname'..."
      docker stop "$cname" 2>/dev/null || true
      docker rm "$cname" 2>/dev/null || true
    fi
  done
}

usage() {
  cat <<EOF
Usage: $0 [MODEL]

Available models:
  qwen3-coder-next    Qwen3-Coder-Next-FP8        via v23 container    (~46-47 tok/s)  [DEFAULT]
  qwen3-next          Qwen3-Next-NVFP4             via NVFP4 container  (~65-70 tok/s)
  qwen35              Qwen3.5-122B-A10B-NVFP4      via vllm nightly
  qwen3-coder-nvfp4   Qwen3-Coder-Next-NVFP4      via NVFP4 container
  minimax             MiniMax-M2.5-REAP-139B        via v23 container

Examples:
  $0                   # defaults to qwen3-coder-next
  $0 qwen3-coder-next
  $0 qwen3-next
EOF
}

MODEL="${1:-qwen3-coder-next}"
MODEL="${MODEL,,}"  # lowercase

stop_model_containers

case "$MODEL" in
  qwen3-coder-next*|coder-next*|qwen3-coder-fp8*)
    exec bash "$SCRIPT_DIR/servers/qwen3-coder-next-fp8/run-v23.sh"
    ;;
  qwen3-next*|qwen3next*)
    exec bash "$SCRIPT_DIR/servers/qwen3-next-nvfp4/run.sh"
    ;;
  qwen35*|qwen3.5*)
    exec bash "$SCRIPT_DIR/servers/qwen35-122b-a10b-nvfp4/run.sh"
    ;;
  qwen3-coder-nvfp4*|coder-nvfp4*)
    exec bash "$SCRIPT_DIR/servers/qwen3-coder-next-nvfp4/run.sh"
    ;;
  minimax*|m2.5*)
    exec bash "$SCRIPT_DIR/servers/minimax-m2.5-reap-139b-nvfp4/run-v23.sh"
    ;;
  *)
    echo "Error: unknown model '$1'"
    echo
    usage
    exit 1
    ;;
esac
