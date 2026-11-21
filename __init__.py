"""ComfyUI node for the MiniMax H3 MUSA runtime."""

import os
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps
import folder_paths
from comfy import model_management
from comfy.utils import ProgressBar
from comfy_api.latest import InputImpl

from . import runtime


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


NODE_CLASS_MAPPINGS = {"MttS4000H3Shot": MttS4000H3Shot}
NODE_DISPLAY_NAME_MAPPINGS = {"MttS4000H3Shot": "MiniMax H3｜摩尔线程 S4000 有声试片"}
