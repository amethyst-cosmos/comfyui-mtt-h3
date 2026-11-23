"""Prepare a single editable H3 shot for the selected ComfyUI runtime."""

import asyncio
from copy import deepcopy
import os
from pathlib import Path
import shutil

from aiohttp import web
import folder_paths
import nodes
import torch
from comfy import model_management

from . import cann_runtime, image_runtime, runtime, shot_canvas


MUSA_FAST, MUSA_HD = runtime.PROFILES
PROFILES = {
    "musa-fast": {
        "backend": "s4000", "title": "快速试镜 · 448×800", "model": "H3 Q8 · 4 步",
        "native": MUSA_FAST, "seconds": 243.5, "evidence": "S4000 首帧原生实测",
        "sources": ["generate", "existing"], "requires_last": False, "required_frames": "只需首帧",
    },
    "musa-hd": {
        "backend": "s4000", "title": "高清 · 576×1024", "model": "H3 Q8 · 4 步",
        "native": MUSA_HD, "seconds": 482.0, "evidence": "S4000 首尾帧试验实测",
        "sources": ["generate", "existing"], "requires_last": True, "required_frames": "需要首帧和尾帧",
    },
    "cann-int8": {
        "backend": "910c", "title": "高清 · 576×1024 · INT8", "model": "H3 ConvRot INT8 · 8 步",
        "seconds": 135.0, "evidence": "910C 单镜带追踪实测",
        "sources": ["generate", "existing"], "requires_last": True, "required_frames": "需要首帧和尾帧",
    },
    "cann-bf16": {
        "backend": "910c", "title": "高清 · 576×1024 · BF16", "model": "H3 剪枝 BF16 · 8 步",
        "seconds": 151.7, "evidence": "910C 六镜纯视频推理均值",
        "sources": ["generate", "existing"], "requires_last": True, "required_frames": "需要首帧和尾帧",
    },
}
PROFILE_ORDER = ("musa-hd", "musa-fast", "cann-bf16", "cann-int8")


def prepare(profile_key, source, motion, dialogue="", first_prompt="", last_prompt=""):
    profile = PROFILES.get(profile_key)
    if profile is None or source not in profile["sources"]:
        raise ValueError("请为当前设备选择可用的画质和关键帧来源。")
    motion, dialogue, first_prompt, last_prompt = (
        value.strip() for value in (motion, dialogue, first_prompt, last_prompt)
    )
    if not motion:
        raise ValueError("请填写镜头动作和机位。")
    if source == "generate" and (not first_prompt or (profile["requires_last"] and not last_prompt)):
        raise ValueError("请填写当前档位所需的关键帧提示词。")
    if profile["backend"] == "s4000":
        return shot_canvas.musa_canvas(profile["native"], source, motion, dialogue, first_prompt, last_prompt)
    return shot_canvas.ascend_canvas(profile_key, source, motion, dialogue, first_prompt, last_prompt)


def check(label, ok, detail):
    return {"label": label, "ok": ok, "detail": detail}


def cann_checks():
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return [check("昇腾设备", False, "当前实例的 TorchNPU 未提供可用设备。")]
    device = model_management.get_torch_device()
    count = npu.device_count()
    result = [check("昇腾设备", device.type == "npu" and device.index in (0, None) and count >= 2,
                    f"ComfyUI 当前设备 {device}，检测到 {count} 个逻辑 NPU。此档需要 npu:0 和 npu:1。")]
    for index in range(min(2, count)):
        props = npu.get_device_properties(index)
        result.append(check(f"npu:{index} 可见显存", True,
                            f"{props.total_memory / 2**30:.1f} GiB 总量"))
    result.append(check("视频注意力", cann_runtime.video_attention_is_ready(),
                        "用 start.sh 启动 910C 后端，视频使用 PyTorch 注意力。"))
    return result


def cann_files(profile_key):
    result = {}
    for group, models in shot_canvas.ascend_weights(profile_key).items():
        missing = []
        for directory, filename in models:
            path = folder_paths.get_full_path(directory, filename)
            if path is None or Path(path).stat().st_size <= 8:
                missing.append(filename)
        result[group] = check("视频权重" if group == "video" else "生图权重", not missing,
                              "已找到文件" if not missing else "缺少或为空 · " + "、".join(missing))
    return result


def musa_files(profile):
    if not shutil.which("mthreads-gmi"):
        return check("MUSA 运行环境", False, "当前实例未检测到 mthreads-gmi。")
    try:
        _, _, _, weights, _ = runtime.runtime_paths(profile["native"])
        bad_size = [path.name for role, path in weights.items()
                    if path.stat().st_size != runtime.WEIGHTS[role][1]]
        if bad_size:
            raise RuntimeError("权重大小不符 · " + "、".join(bad_size))
    except (RuntimeError, OSError) as error:
        return check("S4000 运行环境", False, str(error))
    return check("S4000 运行环境", True, "已确认单张 S4000、推理程序、FFmpeg 和权重文件大小。")


def get_catalog():
    catalog = {key: deepcopy(PROFILES[key]) for key in PROFILE_ORDER}
    try:
        ascend = cann_checks()
    except (OSError, RuntimeError, AttributeError) as error:
        ascend = [check("昇腾设备", False, f"读取 NPU 信息失败 · {error}")]
    output = Path(folder_paths.get_output_directory())
    writable = output.is_dir() and os.access(output, os.W_OK | os.X_OK)
    storage = check("输出目录", writable, f"{output}" if writable else f"目录不可写 · {output}")
    try:
        engine, _ = image_runtime.image_paths(check_device=False)
        image_runtime.check_image_cli(engine)
        image_check = check("S4000 生图", True, "模型文件可读，原生程序提供生图与参考编辑入口。")
    except (RuntimeError, OSError) as error:
        image_check = check("S4000 生图", False, str(error))
    for key, profile in catalog.items():
        if profile["backend"] == "s4000":
            checks = [musa_files(profile), storage]
            profile["checks"] = {"existing": checks, "generate": [*checks, image_check]}
        else:
            files = cann_files(key)
            checks = [*ascend, files["video"], storage]
            image_nodes = "MiniMaxH3ImageConditioning" in nodes.NODE_CLASS_MAPPINGS
            profile["checks"] = {"existing": checks, "generate": [*checks, files["image"],
                                 check("生图节点", image_nodes, "ComfyUI 需支持 H3 Image 六步生图。") ]}
        profile["ready"] = any(all(item["ok"] for item in checks)
                               for checks in profile["checks"].values())
    suggested = next((key for key, profile in catalog.items() if profile["ready"]), None)
    if suggested is None:
        suggested = "cann-bf16" if os.getenv("H3_WORKBENCH_BACKEND") == "910c" or ascend[0]["ok"] else "musa-hd"
    return {"suggested": suggested, "profiles": catalog}


def register_routes(routes):
    frontend_seen = False

    @routes.get("/h3-workbench/frontend-ready")
    async def frontend_ready(_request):
        return web.json_response({"opened": frontend_seen})

    @routes.post("/h3-workbench/frontend-ready")
    async def mark_frontend_ready(_request):
        nonlocal frontend_seen
        frontend_seen = True
        return web.json_response({"opened": frontend_seen})

    @routes.get("/h3-workbench/profiles")
    async def profiles(_request):
        return web.json_response(await asyncio.to_thread(get_catalog))

    @routes.post("/h3-workbench/workflow")
    async def workflow(request):
        try:
            config = await request.json()
            fields = ("profile_key", "source", "motion", "dialogue", "first_prompt", "last_prompt")
            if not isinstance(config, dict) or any(
                not isinstance(config.get(key, ""), str) or len(config.get(key, "")) > 16000
                for key in fields
            ):
                raise ValueError("请检查镜头配置字段。")
            result = prepare(*(config.get(key, "") for key in fields))
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response(result)
