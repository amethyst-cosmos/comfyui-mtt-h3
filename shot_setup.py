"""Prepare a single editable H3 shot for the selected ComfyUI runtime."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from aiohttp import web
import folder_paths
import nodes
import torch
from comfy import model_management

from . import cann_runtime, image_runtime, runtime
from .tools import prison_trial


EXAMPLES = Path(__file__).parent / "example_workflows"
MUSA_FAST, MUSA_HD = runtime.PROFILES
CANN_INT8 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
PROFILES = {
    "musa-fast": {
        "backend": "s4000", "title": "快速试镜 · 448×800", "model": "H3 Q8 · 4 步",
        "template": "01", "native": MUSA_FAST, "seconds": 243.5, "evidence": "S4000 首帧原生实测",
        "sources": ["generate", "existing"], "requires_last": False, "required_frames": "只需首帧",
    },
    "musa-hd": {
        "backend": "s4000", "title": "高清 · 576×1024", "model": "H3 Q8 · 4 步",
        "template": "02", "native": MUSA_HD, "seconds": 482.0, "evidence": "S4000 首尾帧试验实测",
        "sources": ["generate", "existing"], "requires_last": True, "required_frames": "需要首帧和尾帧",
    },
    "cann-int8": {
        "backend": "910c", "title": "高清 · 576×1024 · INT8", "model": "H3 ConvRot INT8 · 8 步",
        "template": "06", "seconds": 135.0, "evidence": "910C 单镜带追踪实测",
        "sources": ["generate", "existing"], "requires_last": True, "required_frames": "需要首帧和尾帧",
    },
    "cann-bf16": {
        "backend": "910c", "title": "高清 · 576×1024 · BF16", "model": "H3 剪枝 BF16 · 8 步",
        "template": "06", "seconds": 151.7, "evidence": "910C 六镜纯视频推理均值",
        "sources": ["generate", "existing"], "requires_last": True, "required_frames": "需要首帧和尾帧",
    },
}
PROFILE_ORDER = ("musa-hd", "musa-fast", "cann-bf16", "cann-int8")


def load_template(number):
    path, = sorted(EXAMPLES.glob(f"{number} · *.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def use_existing_frames(workflow, by_id):
    """Replace both generated anchors, then keep only ancestors of video saving."""
    loaders = [n for n in load_template("02")["nodes"] if n["type"] == "LoadImage"]
    for loader, socket, y in zip(loaders, by_id[17]["inputs"][2:4], (100, 560), strict=True):
        if socket["name"] not in ("first_frame", "last_frame"):
            raise ValueError("高清模板缺少首尾帧输入。")
        link = next(link for link in workflow["links"] if link[0] == socket["link"])
        workflow["last_node_id"] += 1
        loader["id"] = workflow["last_node_id"]
        loader["pos"] = [1120, y]
        loader["size"] = [340, 370]
        loader["outputs"][0]["links"] = [link[0]]
        link[1:3] = [loader["id"], 0]
        by_id[loader["id"]] = loader
        workflow["nodes"].append(loader)

    by_link = {link[0]: link for link in workflow["links"]}
    pending = [27]
    used_nodes = set()
    used_links = set()
    while pending:
        node_id = pending.pop()
        if node_id in used_nodes:
            continue
        used_nodes.add(node_id)
        for socket in by_id[node_id]["inputs"]:
            link_id = socket["link"]
            if link_id is not None:
                used_links.add(link_id)
                pending.append(by_link[link_id][1])
    workflow["nodes"] = [n for n in workflow["nodes"] if n["id"] in used_nodes]
    workflow["links"] = [link for link in workflow["links"] if link[0] in used_links]
    for node in workflow["nodes"]:
        for socket in node["outputs"]:
            socket["links"] = [link for link in socket["links"] or [] if link in used_links]
    workflow["groups"] = [group for group in workflow["groups"] if group["id"] in (1, 4)]
    return loaders


def use_cann_image_worker(workflow, loaders, first_prompt, last_prompt):
    first, last = loaders
    last_link, = last["outputs"][0]["links"]
    first.update(type="MttAscendH3ImagePair", title="H3 Image · 独立生成首尾帧",
                 size=[395, 320], inputs=[{"name": "reference", "type": "IMAGE", "link": None}],
                 outputs=[first["outputs"][0], {"name": "尾帧", "type": "IMAGE", "links": [last_link]}],
                 properties={"Node name for S&R": "MttAscendH3ImagePair"},
                 widgets_values=[first_prompt.strip(), "Edit <Picture 1>. " + last_prompt.strip(),
                                 2026112301, "fixed", *cann_runtime.SIZE])
    first["outputs"][0]["name"] = "首帧"
    next(link for link in workflow["links"] if link[0] == last_link)[1:3] = [first["id"], 1]
    workflow["nodes"].remove(last)
    for slot, point in enumerate(("first", "last")):
        workflow["last_node_id"] += 1
        output = {"id": workflow["last_node_id"], "type": "SaveImage", "pos": [1130, 470 + 300 * slot],
                  "size": [290, 235], "flags": {}, "mode": 0, "order": len(workflow["nodes"]),
                  "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                  "outputs": [], "widgets_values": [f"h3-cann/{point}"]}
        workflow["nodes"].append(output)
        connect(workflow, first, output, "IMAGE", source_slot=slot)


def prepare(profile_key, source, motion, dialogue="", first_prompt="", last_prompt=""):
    profile = PROFILES.get(profile_key)
    if profile is None or source not in profile["sources"]:
        raise ValueError("请为当前设备选择可用的画质和关键帧来源。")
    if not motion.strip():
        raise ValueError("请填写镜头动作和机位。")
    if source == "generate" and (not first_prompt.strip() or profile["requires_last"] and not last_prompt.strip()):
        raise ValueError("请填写当前档位所需的关键帧提示词。")
    workflow = load_template(profile["template"])
    by_id = {node["id"]: node for node in workflow["nodes"]}
    if profile["backend"] == "s4000":
        video = by_id[4]
        video["widgets_values"][1:3] = [motion.strip(), dialogue.strip()]
        if source == "generate":
            add_musa_images(workflow, first_prompt, last_prompt, profile["requires_last"])
        save_video = next(node for node in load_template("06")["nodes"] if node["type"] == "SaveVideo")
        workflow["last_node_id"] += 1
        save_video.update(id=workflow["last_node_id"], order=len(workflow["nodes"]),
                          pos=[video["pos"][0] + video["size"][0] + 70, video["pos"][1]])
        save_video["widgets_values"][0] = "h3-musa/shot"
        workflow["nodes"].append(save_video)
        connect(workflow, video, save_video, "VIDEO")
        return workflow

    if profile_key == "cann-int8":
        model = by_id[14]
        model["title"] = "H3 FL2VA · 剪枝 INT8 ConvRot"
        model["widgets_values"][0] = CANN_INT8
        model_file = model["properties"]["models"][0]
        model_file["url"] = model_file["url"].rsplit("/", 1)[0] + "/" + CANN_INT8
        model_file["name"] = CANN_INT8
    loaders = use_existing_frames(workflow, by_id)
    clip = by_id[1]
    clip.update(type="MttAscendH3Clip", title="H3 文本编码 · npu:1",
                properties={**clip["properties"], "Node name for S&R": "MttAscendH3Clip"},
                widgets_values=clip["widgets_values"][:1])
    if source == "generate":
        use_cann_image_worker(workflow, loaders, first_prompt, last_prompt)

    shot_prompt = motion.strip()
    if dialogue.strip():
        shot_prompt += f"\n角色用普通话清楚说一次：「{dialogue.strip()}」。说完闭嘴。"
    by_id[17]["widgets_values"][0] = shot_prompt
    return workflow


def add_musa_images(workflow, first_prompt, last_prompt, needs_tail):
    """Replace the PNG loaders with native generation and retain preview PNGs."""
    first = None
    for node in [n for n in workflow["nodes"] if n["type"] == "LoadImage"]:
        is_first = first is None
        label = "首帧" if is_first else "尾帧"
        x = 35 if is_first else 515
        prompt = first_prompt.strip() if is_first else "Edit <Picture 1>. " + last_prompt.strip()
        kind = "MttS4000H3Image"
        node.update(type=kind, title=f"H3 Image · {label}输入", pos=[x, 70], size=[380, 246],
                    inputs=[{"name": "reference", "type": "IMAGE", "link": None}],
                    outputs=[node["outputs"][0]],
                    properties={"Node name for S&R": kind},
                    widgets_values=[prompt, 2026112301 if is_first else 2026112311,
                                    "fixed", *image_runtime.SIZE])
        if is_first:
            first = node
        else:
            connect(workflow, first, node, "IMAGE")

        workflow["last_node_id"] += 1
        output = {
            "id": workflow["last_node_id"], "type": "SaveImage", "title": f"保存{label} PNG",
            "pos": [x, 385], "size": [310, 235],
            "flags": {}, "mode": 0, "order": len(workflow["nodes"]),
            "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
            "outputs": [], "widgets_values": [f"h3-musa/{'first' if is_first else 'last'}"],
        }
        workflow["nodes"].append(output)
        connect(workflow, node, output, "IMAGE")
    video = next(n for n in workflow["nodes"] if n["type"] == "MttS4000H3Shot")
    video["pos"] = [1025 if needs_tail else 535, 70]


def connect(workflow, source, target, data_type, source_slot=0):
    workflow["last_link_id"] += 1
    link = workflow["last_link_id"]
    target["inputs"][0]["link"] = link
    output = source["outputs"][source_slot]
    output["links"] = [*(output["links"] or []), link]
    workflow["links"].append([link, source["id"], source_slot, target["id"], 0, data_type])


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


def cann_files(profile_key, profile):
    workflow = load_template(profile["template"])
    groups = {"video": [], "image": []}
    for node in workflow["nodes"]:
        for model in node["properties"].get("models", []):
            filename = model["name"]
            if profile_key == "cann-int8" and node["id"] == 14:
                filename = CANN_INT8
            name = folder_paths.get_full_path(model["directory"], filename)
            if name is None or Path(name).stat().st_size <= 8:
                groups["image" if node["id"] in (2, 3) else "video"].append(filename)
    return {group: check("视频权重" if group == "video" else "生图权重", not missing,
                         "已找到文件" if not missing else "缺少或为空 · " + "、".join(missing))
            for group, missing in groups.items()}


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
            files = cann_files(key, profile)
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

    @routes.get("/h3-workbench/example/{profile_key}")
    async def example(request):
        backend = {"musa-hd": "s4000", "cann-bf16": "910c-bf16", "cann-int8": "910c-int8"}.get(
            request.match_info["profile_key"])
        if backend is None:
            return web.json_response({"error": "请选择 576×1024 画质档。"}, status=400)
        project = json.loads(prison_trial.DEFAULT_PROJECT.read_text(encoding="utf-8"))
        return web.json_response(prison_trial.build(project, backend, automatic=True, save_film=True))

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
