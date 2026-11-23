"""Run the Ascend H3 image stages in a short-lived, loopback-only ComfyUI worker."""

from contextlib import contextmanager
from functools import wraps
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import uuid4

import folder_paths
import torch
import yaml
from comfy import model_management, ops, sd
from comfy.cli_args import args

from .cann_graph import CLIP, IMAGE, IMAGE_VAE, SIZE, Graph, cann_image
from .tools.queue_project import api


def install_cast_compatibility():
    if not model_management.is_ascend_npu():
        return
    original = ops.cast_bias_weight
    if getattr(original, "h3_npu_device_argument", False):
        return

    @wraps(original)
    def cast(s, input=None, dtype=None, device=None, *positional, **options):
        # Some TorchNPU versions pass the target device as the first cast argument.
        if isinstance(input, torch.device):
            device = input if device is None else device
            input = None
        return original(s, input, dtype, device, *positional, **options)

    cast.h3_npu_device_argument = True
    ops.cast_bias_weight = cast


class MttAscendH3Clip:
    CATEGORY = "model/loaders"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip_name": (folder_paths.get_filename_list("text_encoders"), {"default": CLIP})}}

    def load(self, clip_name):
        if not model_management.is_ascend_npu() or torch.npu.device_count() < 2:
            raise RuntimeError("H3 昇腾文本编码需要第二颗逻辑 NPU。")
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return (sd.load_clip(
            ckpt_paths=[path], embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=sd.CLIPType.MINIMAX, model_options={"load_device": torch.device("npu:1")},
        ),)


def plugin_location():
    package = Path(__file__).resolve().parent
    for root in folder_paths.get_folder_paths("custom_nodes"):
        for path in Path(root).glob("*"):
            if path.resolve() == package:
                return path.parent, path.name
    raise RuntimeError("ComfyUI 的 custom_nodes 中找不到本工作台。请用 start.sh 启动。")


def video_attention_is_ready():
    return args.use_pytorch_cross_attention and not os.getenv("H3_CANN_IMAGE_WORKER")


@contextmanager
def image_worker(directory, log_path):
    plugin_root, plugin_name = plugin_location()
    for name in ("input", "output", "user", "temp"):
        (directory / name).mkdir(exist_ok=True)
    config = directory / "paths.yaml"
    paths = {name: "\n".join(folder_paths.get_folder_paths(name))
             for name in ("diffusion_models", "text_encoders", "vae", "embeddings")}
    config.write_text(yaml.safe_dump({"workbench": dict(paths, custom_nodes=str(plugin_root), is_default=True)}),
                      encoding="utf-8")
    with socket.socket() as port_finder:
        port_finder.bind(("127.0.0.1", 0))
        port = port_finder.getsockname()[1]
    comfy = Path(folder_paths.__file__).resolve().parent
    command = [sys.executable, "-u", str(comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
               "--extra-model-paths-config", str(config), "--input-directory", str(directory / "input"),
               "--output-directory", str(directory / "output"), "--user-directory", str(directory / "user"),
               "--temp-directory", str(directory / "temp"), "--preview-method", "none", "--offline",
               "--disable-auto-launch", "--disable-all-custom-nodes", "--whitelist-custom-nodes", plugin_name,
               "--use-split-cross-attention"]
    base_url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, H3_CANN_IMAGE_WORKER="1")
    with log_path.open("wb") as log:
        child = subprocess.Popen(command, cwd=comfy, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(160):
                model_management.throw_exception_if_processing_interrupted()
                if child.poll() is not None:
                    raise RuntimeError(f"生图服务启动失败（退出码 {child.returncode}）。")
                try:
                    with urlopen(base_url + "/object_info/MttAscendH3Clip", timeout=1) as response:
                        if "MttAscendH3Clip" in json.load(response):
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("等待独立生图服务超时。")
            yield base_url, child
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def image_graph(first_prompt, last_prompt, seed, size, reference_name=None):
    graph = Graph()
    clip = graph.add("MttAscendH3Clip", clip_name=CLIP)
    unet = graph.add("UNETLoader", unet_name=IMAGE, weight_dtype="default")
    vae = graph.add("VAELoader", vae_name=IMAGE_VAE)
    reference = graph.add("LoadImage", image=reference_name) if reference_name else None
    first = cann_image(graph, unet, clip, vae, first_prompt, seed, reference, size)
    last = cann_image(graph, unet, clip, vae, last_prompt, seed + 10, first, size)
    graph.add("SaveImage", images=first, filename_prefix="first")
    graph.add("SaveImage", images=last, filename_prefix="last")
    return graph.nodes


def run_images(directory, log, first_prompt, last_prompt, seed, size, reference_name):
    graph = image_graph(first_prompt, last_prompt, seed, size, reference_name)
    model_management.unload_all_models()
    gc.collect()
    for index in range(2):
        with torch.npu.device(index):
            model_management.soft_empty_cache()
    with image_worker(directory, log) as (server, child):
        try:
            job = api(server, "/prompt", {"prompt": graph})["prompt_id"]
        except HTTPError as error:
            details = error.read().decode(errors="replace")
            raise RuntimeError(f"生图任务校验失败 · {details[:2000]}") from error
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            model_management.throw_exception_if_processing_interrupted()
            if child.poll() is not None:
                raise RuntimeError(f"生图服务退出（{child.returncode}）。")
            result = api(server, "/history/" + job).get(job)
            if result is not None:
                if result["status"]["status_str"] != "success":
                    raise RuntimeError("独立生图任务失败，请检查工作台日志。")
                break
            time.sleep(1)
        else:
            raise TimeoutError("H3 Image 运行超时。")
    frames = []
    for stem in ("first", "last"):
        matches = list((directory / "output").glob(f"{stem}_*.png"))
        if len(matches) != 1:
            raise RuntimeError(f"独立生图应输出恰好一张 {stem} PNG。")
        frames.append(matches[0])
    return frames
