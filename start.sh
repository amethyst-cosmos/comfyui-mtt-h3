#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
comfy_dir="${COMFYUI_DIR:-/root/ComfyUI}"
if [[ -d "$comfy_dir" ]]; then
  comfy_dir="$(cd -- "$comfy_dir" && pwd)"
fi
python_bin="${COMFYUI_PYTHON:-python3}"
listen="${COMFYUI_LISTEN:-127.0.0.1}"
port="${COMFYUI_PORT:-6006}"
backend="${H3_WORKBENCH_BACKEND:-auto}"
if [[ "$backend" == auto ]]; then
  if command -v npu-smi >/dev/null && npu-smi info 2>/dev/null | grep Ascend910 >/dev/null; then
    backend=910c
  else
    backend=s4000
  fi
fi
case "$backend" in
  s4000) output="${COMFYUI_OUTPUT:-/root/autodl-tmp/mtt-h3-output}"; device_args=(--cpu) ;;
  910c) output="${COMFYUI_OUTPUT:-$comfy_dir/output}"; device_args=(--use-pytorch-cross-attention) ;;
  *) printf 'H3_WORKBENCH_BACKEND 应为 auto、s4000 或 910c。\n' >&2; exit 2 ;;
esac
if [[ "$output" != /* && -d "$comfy_dir" ]]; then
  output="$(cd -- "$comfy_dir" && pwd)/$output"
fi
export COMFYUI_OUTPUT="$output" H3_WORKBENCH_BACKEND="$backend"

if [[ "${1:-}" != --comfy-only && "${H3_BOOT_PORT:-6008}" != 0 ]]; then
  exec "$python_bin" "$package_dir/boot_status.py" \
    --listen "$listen" --port "${H3_BOOT_PORT:-6008}" --comfy-port "$port" \
    --log "${H3_BOOT_LOG:-$(dirname -- "$output")/h3-workbench-comfy.log}" -- bash "$package_dir/start.sh" --comfy-only
fi

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
mkdir -p -- "$output"
model_args=()
if [[ -n "${COMFYUI_MODEL_PATHS:-}" ]]; then
  if [[ ! -r "$COMFYUI_MODEL_PATHS" ]]; then
    printf '无法读取 COMFYUI_MODEL_PATHS · %s\n' "$COMFYUI_MODEL_PATHS" >&2
    exit 1
  fi
  model_config="$(cd -- "$(dirname -- "$COMFYUI_MODEL_PATHS")" && pwd)/$(basename -- "$COMFYUI_MODEL_PATHS")"
  model_args=(--extra-model-paths-config "$model_config")
fi
cd -- "$comfy_dir"
exec "$python_bin" main.py --listen "$listen" --port "$port" "${device_args[@]}" "${model_args[@]}" \
  --disable-auto-launch --offline \
  --disable-all-custom-nodes --whitelist-custom-nodes mtt-s4000-h3 \
  --output-directory "$output"
