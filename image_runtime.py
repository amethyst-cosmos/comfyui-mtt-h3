"""Shared-weight staging and native H3 Image commands for the S4000."""

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import subprocess

if __package__:
    from . import runtime
else:
    import runtime


DIFFUSION_WEIGHTS = {
    "q8": ("diffusion_models/minimax_h3_image_pruned-Q8_0.gguf",
           22026760928, "ff5ed88d968e9fc3c3352101592ba83f963879ae3118ce9ab9eb7946aad70255"),
    "convrot-int8": ("diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
                     20979314802, "83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622"),
}
COMMON_WEIGHTS = {
    "llm": ("text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            18218065024, "11e6efe70a57ce7f4838c47bdbd1a1c4b8ce10e2b7747f1b065990b70f4b05fc"),
    "vae": ("vae/minimax_h3_image_vae_fp16.safetensors",
            1168505312, "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af"),
}
SIZE = (576, 1024)
SIGMAS = "1.0,0.9675236940383911,0.8875485062599182,0.75,0.5327739119529724,0.23200887441635132,0.0"
TRACE_STEP = re.compile(rb"\b([1-6])/6\s+-")
TIMEOUT = 900


def weight_specs():
    image_format = os.environ.get("MTT_H3_IMAGE_FORMAT", "q8")
    if image_format not in DIFFUSION_WEIGHTS:
        raise RuntimeError("MTT_H3_IMAGE_FORMAT 必须是 q8 或 convrot-int8。")
    return {"diffusion-model": DIFFUSION_WEIGHTS[image_format], **COMMON_WEIGHTS}


def check_file(path, definition):
    try:
        if not path.is_file() or path.stat().st_size != definition[1]:
            raise RuntimeError(f"模型文件缺失或大小不符 · {path}")
        with path.open("rb") as file:
            file.read(8)
    except OSError as error:
        raise RuntimeError(f"无法读取模型文件 · {path}") from error


def stage_weights(shared_root, models, definitions: Mapping | None = None):
    """Validate every path and checksum before creating any link."""
    if definitions is None:
        definitions = weight_specs()
    pending = []
    for role, spec in definitions.items():
        dest = models / spec[0]
        if dest.exists():
            source = dest.resolve(strict=True)
        elif dest.is_symlink():
            raise RuntimeError(f"模型软链接失效 · {dest}")
        else:
            source = (shared_root / spec[0]).resolve(strict=False)
            pending.append((source, dest))
        check_file(source, spec)
        if runtime.sha256_file(source) != spec[2]:
            raise RuntimeError(f"模型 SHA-256 不符 · {source}")
    for source, dest in pending:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(source)
        except FileExistsError:
            if not dest.is_symlink() or dest.resolve(strict=True) != source:
                raise
    return {role: str((models / definition[0]).resolve(strict=True))
            for role, definition in definitions.items()}


def image_paths(check_device=True):
    root, models = runtime.model_roots()
    engine = Path(os.environ.get("MTT_H3_IMAGE_ENGINE", str(root / "sd-cli")))
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise RuntimeError(f"缺少可执行的 H3 Image MUSA 推理程序 · {engine}")
    if check_device:
        runtime.check_s4000()
    definitions = weight_specs()
    weights = {role: models / spec[0] for role, spec in definitions.items()}
    for role, spec in definitions.items():
        check_file(weights[role], spec)
    return engine, weights


def check_image_cli(engine):
    try:
        help_text = subprocess.run([str(engine), "--help"], capture_output=True,
                                   text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"无法探测 H3 Image 推理程序 · {engine}") from error
    text = help_text.stdout + help_text.stderr
    missing = [name for name in ("img_gen", "--ref-image", "--vae-tiling", "--diffusion-fa")
               if name not in text]
    if missing:
        raise RuntimeError("生图运行时缺少入口 · " + "、".join(missing))


def build_invocation(paths, prompt, seed, width, height, output, reference=None):
    if min(width, height) < 32 or max(width, height) > 1024 or width % 32 or height % 32:
        raise ValueError("首发试验尺寸要求宽高为 32 的倍数，单边 32–1024。")
    engine, weights = paths
    command = [str(engine), "--mode", "img_gen"]
    for role, path in weights.items():
        command.extend(["--" + role, str(path)])
    command.extend([
        "--backend", "gpu,te=cpu", "--max-vram", "42", "--auto-fit", "on", "--conv-direct", "off",
        "--diffusion-fa",
        "--vae-tiling", "--width", str(width), "--height", str(height), "--steps", "6",
        "--cfg-scale", "1", "--sampling-method", "euler", "--sigmas", SIGMAS,
        "--rng", "cpu", "--seed", str(seed), "--output", str(output), "--threads", "15", "-v",
    ])
    if reference is not None:
        command.extend(["--ref-image", str(reference)])
    command.extend(["--prompt", prompt])
    environment = os.environ | {
        "SD_MUSA_EXIT_RESET": "1", "GGML_CUDA_DISABLE_GRAPHS": "1",
        "MUSA_CONTINUOUS_COUNT": "1", "GGML_MUSA_QUANT_F16": "1",
        "GGML_MUSA_F16_OUT_F32": "1", "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f16",
        "OMP_NUM_THREADS": "15",
    }
    return command, environment


def main():
    parser = argparse.ArgumentParser(description="校验共享权重并为 S4000 建立本地软链接。")
    parser.add_argument("--shared", required=True, type=Path)
    args = parser.parse_args()
    _, models = runtime.model_roots()
    try:
        print(json.dumps(stage_weights(args.shared, models), ensure_ascii=False, indent=2))
    except (OSError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
