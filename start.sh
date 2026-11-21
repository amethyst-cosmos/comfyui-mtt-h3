#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
comfy_dir="${COMFYUI_DIR:-/root/ComfyUI}"
python_bin="${COMFYUI_PYTHON:-python3}"
listen="${COMFYUI_LISTEN:-127.0.0.1}"
port="${COMFYUI_PORT:-6006}"

if [[ ! -f "$comfy_dir/main.py" ]]; then
  printf '找不到 ComfyUI。请设置 COMFYUI_DIR。\n' >&2
  exit 1
fi
plugin="$comfy_dir/custom_nodes/mtt-s4000-h3"
if [[ ! -e "$plugin" && ! -L "$plugin" ]]; then
  mkdir -p "$comfy_dir/custom_nodes"
  ln -s "$package_dir" "$plugin"
fi
if [[ ! "$plugin" -ef "$package_dir" ]]; then
  printf '已有同名节点，需在镜像内核对版本：%s\n' "$plugin" >&2
  exit 1
fi
cd -- "$comfy_dir"
exec "$python_bin" main.py --listen "$listen" --port "$port" --cpu \
  --disable-auto-launch --offline \
  --disable-all-custom-nodes --whitelist-custom-nodes mtt-s4000-h3 \
  --output-directory "${COMFYUI_OUTPUT:-/root/autodl-tmp/mtt-h3-output}"
