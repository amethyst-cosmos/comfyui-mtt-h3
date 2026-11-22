"""ComfyUI node for the MiniMax H3 MUSA runtime."""

import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
import torch
import folder_paths
from comfy import model_management
from comfy.utils import ProgressBar
from comfy_api.latest import InputImpl
from server import PromptServer

from . import cann_runtime, image_runtime, runtime
from .shot_setup import register_routes


def save_frame(images, destination, fit_to=None):
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[-1] != 3:
        raise RuntimeError("每个关键帧口只接受一张 RGB 图片。")
    pixels = images[0].detach().cpu().clamp(0, 1).mul(255).round().byte().numpy()
    frame = Image.fromarray(pixels)
    if fit_to is not None and frame.size != fit_to:
        frame = ImageOps.fit(frame, fit_to, method=Image.Resampling.LANCZOS)
    frame.save(destination)


class MttS4000H3Shot:
    CATEGORY = "video/MTT S4000"
    DESCRIPTION = "MiniMax H3 单卡有声试片。基础档用首帧，实验档用首尾帧。"
    FUNCTION = "generate"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("有声视频",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first": ("IMAGE", {"tooltip": "镜头第一帧。"}),
                "profile": (list(runtime.PROFILES), {"tooltip": "试验档必须连接尾帧。"}),
                "prompt": ("STRING", {"multiline": True, "default": "人物缓慢抬头，镜头平稳推进。"}),
                "dialogue": ("STRING", {"default": "", "tooltip": "可选。逐字填写台词，生成后用原声审片。"}),
                "seed": ("INT", {"default": 20261121211, "min": 0, "max": 2**53 - 1}),
            },
            "optional": {
                "tail": ("IMAGE", {"tooltip": "试验档的最后一帧。"}),
                "width": ("INT", {"default": 0, "min": 0, "max": runtime.MAX_SIDE, "step": runtime.SIZE_ALIGNMENT,
                                  "tooltip": "输出宽度。0 采用档位默认值。自选尺寸属于试验配置，输入图按目标画幅居中裁切。"}),
                "height": ("INT", {"default": 0, "min": 0, "max": runtime.MAX_SIDE, "step": runtime.SIZE_ALIGNMENT,
                                   "tooltip": "输出高度。0 采用档位默认值。自选尺寸属于试验配置，输入图按目标画幅居中裁切。"}),
            },
        }

    def generate(self, first, profile, prompt, dialogue, seed, tail=None, width=0, height=0):
        default_width, default_height, _, needs_tail = runtime.PROFILES[profile]
        size = runtime.resolve_size(profile, width, height)
        fit_to = size if size != (default_width, default_height) else None
        if needs_tail != (tail is not None):
            raise RuntimeError("请为试验档连接尾帧。基础档只接受首帧。")
        prompt, dialogue = prompt.strip(), dialogue.strip()
        if not prompt:
            raise RuntimeError("请填写镜头动作。")
        if dialogue:
            prompt += f"\n角色用普通话清楚说一次：「{dialogue}」。说完闭嘴。"
        if "<lora:" in prompt.lower():
            raise RuntimeError("该节点固定使用已验证的四步 LoRA。")

        paths = runtime.runtime_paths(profile)
        runtime.verify_weights(paths[3], paths[4])
        output = Path(folder_paths.get_output_directory())
        root = Path(os.environ.get("MTT_H3_OUTPUT", str(output / "mtt-h3"))).resolve()
        try:
            relative = root.relative_to(output.resolve())
        except ValueError as error:
            raise RuntimeError("MTT_H3_OUTPUT 应位于 ComfyUI 输出目录内。") from error
        job_id = uuid4().hex
        folder = root / job_id
        folder.mkdir(parents=True, exist_ok=False)
        first_file = folder / "first.png"
        tail_file = folder / "tail.png" if needs_tail else None
        save_frame(first, first_file, fit_to)
        if needs_tail:
            save_frame(tail, tail_file, fit_to)
        bar = ProgressBar(4)
        path = runtime.render(profile, paths, first_file, tail_file, prompt, seed, folder,
                              bar.update_absolute, model_management.throw_exception_if_processing_interrupted, size=size)
        item = {"filename": path.name, "subfolder": str(relative / job_id), "type": "output"}
        return {"ui": {"images": [item], "animated": (True,)}, "result": (InputImpl.VideoFromFile(str(path)),)}


class MttS4000H3Image:
    CATEGORY = "image/MTT S4000"
    DESCRIPTION = "通过独立 MUSA 进程运行 H3 Image。参考图可选，编辑提示词用 <Picture 1>。"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        def dimension(default):
            return ("INT", {"default": default, "min": 32, "max": 1024, "step": 32})

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 2026112301, "min": 0, "max": 2**53 - 1}),
                "width": dimension(image_runtime.SIZE[0]),
                "height": dimension(image_runtime.SIZE[1]),
            },
            "optional": {"reference": ("IMAGE",)},
        }

    def generate(self, prompt, seed, width, height, reference=None):
        if not prompt.strip():
            raise ValueError("请填写生图或编辑提示词。")
        paths = image_runtime.image_paths()
        folder = Path(folder_paths.get_output_directory()) / "mtt-h3-image" / uuid4().hex
        folder.mkdir(parents=True)
        log = folder / "render.log"
        target = folder / "native.png"
        reference_path = None
        try:
            if reference is not None:
                reference_path = folder / "reference.png"
                save_frame(reference, reference_path)
            command, env = image_runtime.build_invocation(
                paths, prompt.strip(), seed, width, height, target, reference_path,
            )
            bar = ProgressBar(6)
            code, seconds = runtime.run_engine(
                command, env, log, image_runtime.TIMEOUT, bar.update_absolute,
                model_management.throw_exception_if_processing_interrupted,
                step_pattern=image_runtime.TRACE_STEP,
            )
            if code and (code != -signal.SIGABRT or b"1/1 images saved" not in log.read_bytes()):
                raise RuntimeError(f"生图进程退出码 {code}。")
            with Image.open(target) as image:
                if image.size != (width, height):
                    raise RuntimeError(f"生图尺寸为 {image.size}，预期 {(width, height)}。")
                image = image.convert("RGB")
                if all(lo == hi for lo, hi in image.getextrema()):
                    raise RuntimeError("生成图像只有一种颜色。")
                rgb = np.asarray(image, dtype=np.float32) / 255
            (folder / "receipt.json").write_text(json.dumps({
                "mode": "reference_edit" if reference_path else "text_to_image",
                "model": paths[1]["diffusion-model"].name,
                "width": width, "height": height, "seed": seed, "steps": 6,
                "engine_seconds": seconds, "engine_exit": code,
            }, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(f"{error}\n日志：{log}") from error
        return (torch.from_numpy(rgb).unsqueeze(0),)


class MttAscendH3ImagePair:
    CATEGORY = "image/Ascend 910C"
    DESCRIPTION = "独立启动分割注意力服务。先出首帧，再引用首帧编辑尾帧，关闭生图服务后将两张图交给视频。"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("首帧", "尾帧")

    @classmethod
    def INPUT_TYPES(cls):
        def dimension(default):
            return ("INT", {"default": default, "min": 32, "max": 1024, "step": 32})

        return {"required": {
            "first_prompt": ("STRING", {"multiline": True}),
            "last_prompt": ("STRING", {"multiline": True}),
            "seed": ("INT", {"default": 2026112301, "min": 0, "max": 2**53 - 11}),
            "width": dimension(cann_runtime.SIZE[0]),
            "height": dimension(cann_runtime.SIZE[1]),
        }, "optional": {"reference": ("IMAGE",)}}

    def generate(self, first_prompt, last_prompt, seed, width, height, reference=None):
        if not first_prompt.strip() or not last_prompt.strip() or width % 32 or height % 32:
            raise ValueError("请填写首尾帧提示词。宽高必须是 32 的倍数。")
        if not model_management.is_ascend_npu() or torch.npu.device_count() < 2:
            raise RuntimeError("H3 多阶段生图需要 910C 上的两颗逻辑 NPU。")
        if not cann_runtime.video_attention_is_ready():
            raise RuntimeError("请用 910C 工作台启动视频服务，让工作台自动管理独立生图进程。")
        folder = Path(folder_paths.get_output_directory()) / "h3-cann-image" / uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        log = folder / "worker.log"
        try:
            with TemporaryDirectory(prefix="worker-", dir=folder) as temp:
                directory = Path(temp)
                reference_name = None
                if reference is not None:
                    (directory / "input").mkdir()
                    reference_name = "reference.png"
                    save_frame(reference, directory / "input" / reference_name)
                frames = cann_runtime.run_images(directory, log, first_prompt, last_prompt, seed,
                                                 (width, height), reference_name)
                images = []
                for frame in frames:
                    with Image.open(frame) as image:
                        if image.size != (width, height):
                            raise ValueError(f"生图尺寸为 {image.size}，预期 {(width, height)}。")
                        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                    images.append(torch.from_numpy(rgb).unsqueeze(0))
                return tuple(images)
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(f"{error}\n日志：{log}") from error


NODE_CLASS_MAPPINGS = {
    "MttS4000H3Shot": MttS4000H3Shot,
    "MttS4000H3Image": MttS4000H3Image,
    "MttAscendH3Clip": cann_runtime.MttAscendH3Clip,
    "MttAscendH3ImagePair": MttAscendH3ImagePair,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MttS4000H3Shot": "MiniMax H3｜摩尔线程 S4000 有声试片",
    "MttS4000H3Image": "MiniMax H3 Image｜摩尔线程 S4000 生图与编辑",
    "MttAscendH3Clip": "MiniMax H3｜昇腾文本编码 npu:1",
    "MttAscendH3ImagePair": "MiniMax H3 Image｜昇腾首尾帧",
}


WEB_DIRECTORY = "./web"

cann_runtime.install_cast_compatibility()

if getattr(PromptServer, "instance", None) is not None:
    register_routes(PromptServer.instance.routes)
