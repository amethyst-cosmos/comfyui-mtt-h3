"""Pinned MiniMax H3 native inference for a single MTT S4000."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


PROFILES = {
    "448×800 基础档": (448, 800, 390, False),
    "576×1024 首尾帧试验档": (576, 1024, 660, True),
}
SIZE_ALIGNMENT = 32
MAX_SIDE = max(max(profile[:2]) for profile in PROFILES.values())
FRAMES = 124
FPS = 24
LORA = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16"
LORA_SHA256 = "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc"
WEIGHTS = {
    "diffusion-model": (
        "diffusion_models/minimax_h3_fl2va_pruned-Q8_0.gguf", 21437786208,
        "c2055d8f94040e27e9ac4dddc159ee009132d49bf17ec554756b0e8ca4f84a03",
    ),
    "llm": (
        "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf", 18218065024,
        "e4c5f2cfead99e8f516e4ddd882fc0da2116d6e28bd610e04a1faae8cd1efd87",
    ),
    "vae": (
        "vae/minimax_h3_sol_vae_fp16.safetensors", 3596335728,
        "d640ddade51eda6a4eec7b59bf3280dced2da8a71c3e8976297d1e81a08a5b1c",
    ),
    "audio-vae": (
        "vae/minimax_h3_audio_vae_fp32.safetensors", 605254808,
        "2b97c2b4379a0acb625c5eb09b87be84a9f42b1c99a3c0dd69be819d6a09a5c2",
    ),
}
SIGMAS = "1.0,0.947368421,0.857142857,0.666666667,0.0"
TRACE_STEP = re.compile(rb"\b([1-4])/4\s+-")


def check_s4000():
    try:
        detected = subprocess.check_output(["mthreads-gmi"], text=True, stderr=subprocess.STDOUT, timeout=12)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("检测不到 MUSA 运行环境。请在 MTT S4000 实例运行。") from error
    devices = re.findall(r"^\s*\d+\s+([^|\n]+)\|[0-9a-fA-F:.]+", detected, re.M)
    if [name.strip() for name in devices] != ["MTT S4000"]:
        raise RuntimeError("该配置需要单张可见的 MTT S4000。")


def model_roots():
    root = Path(os.environ.get("MTT_H3_RUNTIME", "/root/mtt-h3-runtime")).resolve()
    return root, Path(os.environ.get("MTT_H3_MODELS", str(root / "models"))).resolve()


def sha256_file(path):
    checksum = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def resolve_size(profile, width=0, height=0):
    default_width, default_height, _, _ = PROFILES[profile]
    width, height = width or default_width, height or default_height
    for side in (width, height):
        if not isinstance(side, int) or not SIZE_ALIGNMENT <= side <= MAX_SIDE or side % SIZE_ALIGNMENT:
            raise RuntimeError(f"宽和高须为 {SIZE_ALIGNMENT} 的倍数，单边范围 {SIZE_ALIGNMENT}–{MAX_SIDE} 像素。")
    max_pixels = default_width * default_height
    if width * height > max_pixels:
        raise RuntimeError(f"当前档位最多允许 {max_pixels:,} 像素，请减少宽或高。")
    return width, height


def runtime_paths(profile, device=True):
    experimental = PROFILES[profile][3]
    root, models = model_roots()
    engine = root / ("sd-cli-chunk4096" if experimental else "sd-cli")
    native = root / "native_flash_bridge.so"
    trace = root / "trace_bridge.so"
    weights = {role: models / definition[0] for role, definition in WEIGHTS.items()}
    lora = models / "loras" / (LORA + ".safetensors")
    required = [engine, native, lora, *weights.values()]
    if experimental:
        required.append(trace)
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError("缺少 MUSA 程序或模型文件：\n" + "\n".join(missing))
    if not os.access(engine, os.X_OK):
        raise RuntimeError(f"推理程序没有执行权限：{engine}")
    if any(not shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise RuntimeError("请安装 FFmpeg 与 FFprobe。")
    if device:
        check_s4000()
    return engine, native, trace, weights, lora


def verify_weights(weights, lora):
    for role, path in weights.items():
        _, expected_size, expected_edges = WEIGHTS[role]
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"模型大小与已验证版本不符：{path.name}")
        with path.open("rb") as file:
            first = file.read(65536)
            file.seek(-65536, 2)
            actual_edges = hashlib.sha256(first + file.read(65536)).hexdigest()
        if actual_edges != expected_edges:
            raise RuntimeError(f"模型首尾指纹不符：{path.name}")
    if sha256_file(lora) != LORA_SHA256:
        raise RuntimeError(f"LoRA 指纹不符：{lora.name}")


def build_invocation(profile, paths, first, tail, prompt, seed, output, size=None):
    _, _, _, experimental = PROFILES[profile]
    width, height = resolve_size(profile) if size is None else size
    engine, native, trace, weights, lora = paths
    command = [str(engine), "--mode", "vid_gen"]
    for key, path in weights.items():
        command.extend(["--" + key, str(path)])
    command.extend([
        "--backend", "gpu,te=cpu", "--max-vram", "42", "--auto-fit", "on",
        "--conv-direct", "off", "--flash-attn", "off",
        "--model-args", "minimax_h3_attn_chunk_budget=0",
        "--lora-model-dir", str(lora.parent), "--lora-apply-mode", "at_runtime",
        "--width", str(width), "--height", str(height), "--fps", str(FPS),
        "--video-frames", str(FRAMES), "--steps", "4", "--cfg-scale", "1",
        "--flow-shift", "6", "--sampling-method", "euler", "--scheduler", "discrete",
        "--rng", "cpu", "--seed", str(seed), "--prompt", prompt + f"<lora:{LORA}:1>",
        "--output", str(output), "--compression-quality", "100", "--preview", "none",
        "--init-img", str(first), "--conditioning-cache-size", "4", "--sigmas", SIGMAS,
        "--threads", "15", "-v",
    ])
    environment = os.environ | {
        "SD_MUSA_EXIT_RESET": "1", "GGML_CUDA_DISABLE_GRAPHS": "1",
        "MUSA_CONTINUOUS_COUNT": "1", "GGML_MUSA_QUANT_F16": "1",
        "GGML_MUSA_F16_OUT_F32": "1", "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f16",
        "GGML_MUSA_H3_TILELANG_SO": str(trace if experimental else native),
        "OMP_NUM_THREADS": "15",
    }
    if experimental:
        command.extend(["--end-img", str(tail)])
        environment["H3_ORIGINAL_FLASH_BRIDGE"] = str(native)
    return command, environment


def run_engine(command, environment, log, timeout, report_progress, check_interrupted,
               step_pattern=TRACE_STEP):
    started = time.monotonic()
    with log.open("wb") as writer, log.open("rb") as reader:
        process = subprocess.Popen(command, env=environment, stdout=writer,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            step = 0
            remainder = b""
            while process.poll() is None:
                check_interrupted()
                if time.monotonic() - started >= timeout:
                    raise RuntimeError(f"超过推理运行时限 {timeout} 秒。")
                chunk = reader.read()
                if chunk:
                    data = remainder + chunk
                    next_step = max((int(match[1]) for match in step_pattern.finditer(data)), default=step)
                    if next_step > step:
                        step = next_step
                        report_progress(step)
                    remainder = data[-64:]
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
    return process.returncode, round(time.monotonic() - started, 3)


def verify_media(path, width, height):
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)
    ], text=True, timeout=40))
    streams = metadata["streams"]
    video = next((stream for stream in streams if stream["codec_type"] == "video"), {})
    if (video.get("width"), video.get("height"), video.get("nb_read_frames")) != (width, height, str(FRAMES)):
        raise RuntimeError("出片尺寸或帧数与预设不符。")
    if not any(stream["codec_type"] == "audio" for stream in streams):
        raise RuntimeError("输出缺少原声音轨。")
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-xerror", "-i", str(path), "-vf",
        "blackdetect=d=0.2:pix_th=0.1", "-af", "astats", "-f", "null", "-"
    ], check=True, capture_output=True, text=True, timeout=75)
    black = sum(float(value) for value in re.findall(r"\bblack_duration:([\d.]+)", result.stderr))
    dc = re.findall(r"\bDC offset:\s*(-?[\d.]+)", result.stderr)
    if black >= FRAMES / FPS * 0.95:
        raise RuntimeError("成片几乎全黑。请查看运行日志。")
    if any(abs(float(value)) >= 0.995 for value in dc):
        raise RuntimeError("音轨出现饱和异常。请查看运行日志。")
    return black


def render(profile, paths, first, tail, prompt, seed, folder, report_progress, check_interrupted, size=None):
    default_width, default_height, timeout, _ = PROFILES[profile]
    width, height = (default_width, default_height) if size is None else size
    original = folder / "native.webm"
    log = folder / "render.log"
    command, env = build_invocation(profile, paths, first, tail, prompt, seed, original, (width, height))
    try:
        code, seconds = run_engine(command, env, log, timeout, report_progress, check_interrupted)
        # This runtime has previously aborted during cleanup after successfully saving a complete clip.
        if code and (code != -signal.SIGABRT or b"save result video" not in log.read_bytes()):
            raise RuntimeError(f"推理退出码 {code}。")
        black = verify_media(original, width, height)
        video = folder / "preview.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(original),
            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-crf", "16",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(video)
        ], check=True, timeout=75)
        (folder / "receipt.json").write_text(json.dumps({
            "profile": profile, "width": width, "height": height, "frames": FRAMES,
            "fps": FPS, "seed": seed, "engine_seconds": seconds, "engine_exit": code,
            "custom_size": (width, height) != (default_width, default_height),
            "black_seconds": black, "audio_present": True,
        }, indent=2, ensure_ascii=False) + "\n")
        return video
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"{error}\n日志：{log}") from error


def main():
    parser = argparse.ArgumentParser(description="检查 MiniMax H3 单卡运行依赖。")
    parser.add_argument("--detail", action="store_true", help="检查 576×1024 实验档")
    parser.add_argument("--files-only", action="store_true", help="无卡模式复核文件")
    options = parser.parse_args()
    profile = list(PROFILES)[int(options.detail)]
    try:
        _, _, _, weights, lora = runtime_paths(profile, device=not options.files_only)
        verify_weights(weights, lora)
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"预检通过 · {profile}")


if __name__ == "__main__":
    main()
