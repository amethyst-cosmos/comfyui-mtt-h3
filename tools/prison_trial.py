"""Compile and assemble the same two-shot H3 trial on S4000 and Ascend."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


DEFAULT_PROJECT = Path(__file__).resolve().parents[1] / "example_projects/许七安醒来10秒.json"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
IMAGE = "minimax_h3_image_pruned_int8_convrot.safetensors"
IMAGE_VAE = "minimax_h3_image_vae_fp16.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
MUSA_HD = "576×1024 首尾帧试验档"
FPS = 24
RAW_FRAMES = 124
SIZE = (576, 1024)


class Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, node_type, **inputs):
        node_id = str(len(self.nodes) + 1)
        self.nodes[node_id] = {"class_type": node_type, "inputs": inputs}
        return [node_id, 0]


def cann_image(graph, model, clip, vae, prompt, seed, reference=None, size=SIZE):
    ref = {} if reference is None else {"images.image_1": reference}
    cond = graph.add("MiniMaxH3ImageConditioning", clip=clip, vae=vae, prompt=prompt,
                     width=size[0], height=size[1], **ref)
    negative = graph.add("ConditioningZeroOut", conditioning=cond)
    sampled = graph.add("KSampler", model=model, positive=cond, negative=negative,
                        latent_image=[cond[0], 1], seed=seed, steps=6, cfg=1.0,
                        sampler_name="euler", scheduler="simple", denoise=1.0)
    return graph.add("VAEDecode", samples=sampled, vae=vae)


def cann_video(graph, clip, model, vae, audio_vae, first, last, prompt, seed):
    cond = graph.add("MiniMaxH3ImageToVideo", clip=clip, vae=vae, prompt=prompt,
                     width=SIZE[0], height=SIZE[1], length=RAW_FRAMES, first_frame=first, last_frame=last)
    sampler = graph.add("KSamplerSelect", sampler_name="res_multistep")
    sigmas = graph.add("BasicScheduler", model=model, scheduler="simple", steps=8, denoise=1.0)
    noise = graph.add("RandomNoise", noise_seed=seed)
    guider = graph.add("BasicGuider", model=model, conditioning=cond)
    samples = graph.add("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                        sigmas=sigmas, latent_image=[cond[0], 1])
    picture = graph.add("VAEDecode", vae=vae, samples=samples)
    audio = graph.add("VAEDecodeAudio", vae=audio_vae, samples=samples)
    return graph.add("CreateVideo", images=picture, audio=audio, fps=FPS)


def build(project, backend, *, automatic=False, save_film=False):
    if (len(project["shots"]) != 2 or project["fps"] != FPS or project["seconds_per_shot"] != 5
            or tuple(project["video_size"]) != SIZE or tuple(project["image_size"]) != SIZE):
        raise ValueError("此试片固定为两镜、每镜 5 秒、576×1024、24 fps。")
    graph = Graph()
    if backend == "s4000":
        def image(prompt, seed, reference=None):
            ref = {} if reference is None else {"reference": reference}
            return graph.add("MttS4000H3Image", prompt=prompt, seed=seed, width=SIZE[0], height=SIZE[1], **ref)
    else:
        if automatic:
            clip = graph.add("MttAscendH3Clip", clip_name=CLIP)
        else:
            clip = graph.add("CLIPLoader", clip_name=CLIP, type="minimax", device="npu:1")
            image_model = graph.add("UNETLoader", unet_name=IMAGE, weight_dtype="default")
            image_vae = graph.add("VAELoader", vae_name=IMAGE_VAE)
        video_weight = ("minimax_h3_fl2va_pruned_bf16.safetensors" if backend == "910c-bf16"
                        else "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
        video_base = graph.add("UNETLoader", unet_name=video_weight, weight_dtype="default")
        video_model = graph.add("LoraLoaderModelOnly", model=video_base, lora_name=TURBO, strength_model=1.0)
        video_vae = graph.add("VAELoader", vae_name=VIDEO_VAE)
        audio_vae = graph.add("VAELoader", vae_name=AUDIO_VAE)

        def image(prompt, seed, reference=None):
            return cann_image(graph, image_model, clip, image_vae, prompt, seed, reference)

    anchor = None
    slices = []
    for index, shot in enumerate(project["shots"], 1):
        prefix = f"prison-10s/shot-{index:02d}"
        style = "Live-action film with legible facial features, fabric and straw. No text or captions."
        first_prompt = "\n".join((shot["first"], project["character"], project["cell"], style))
        if anchor is not None:
            first_prompt = ("Edit <Picture 1> to the new camera and posture below. "
                            "Keep the same face, hairpin, clothes, room and lighting.\n" + first_prompt)
        last_prompt = ("Edit <Picture 1>. Keep the same actor, camera, room, clothes and light. "
                       "Change only the described movement. " + shot["last"] + " " + style)
        if backend != "s4000" and automatic:
            ref = {} if anchor is None else {"reference": anchor}
            first = graph.add("MttAscendH3ImagePair", first_prompt=first_prompt, last_prompt=last_prompt,
                              seed=shot["seed"], width=SIZE[0], height=SIZE[1], **ref)
            tail = [first[0], 1]
        else:
            first = image(first_prompt, shot["seed"], anchor)
            tail = image(last_prompt, shot["seed"] + 10, first)
        if anchor is None:
            anchor = first
        graph.add("SaveImage", images=first, filename_prefix=f"{prefix}-first")
        graph.add("SaveImage", images=tail, filename_prefix=f"{prefix}-tail")
        motion = "\n".join((shot["motion"], project["cell"], style, "Audio: " + shot["sound"]))
        dialogue = shot.get("dialogue", "")
        if dialogue:
            motion += "\n" + project["voice"]
        if backend == "s4000":
            video = graph.add("MttS4000H3Shot", profile=MUSA_HD, prompt=motion, dialogue=dialogue,
                              seed=shot["seed"], width=SIZE[0], height=SIZE[1], first=first, tail=tail)
        else:
            if dialogue:
                motion += f"\n角色用普通话清楚说一次：「{dialogue}」。说完闭嘴。"
            video = cann_video(graph, clip, video_model, video_vae, audio_vae, first, tail, motion, shot["seed"])
        graph.add("SaveVideo", video=video, filename_prefix=prefix, format="mp4")
        if save_film:
            slices.append(graph.add("Video Slice", video=video, start_time=0, duration=5, strict_duration=True))
    if save_film:
        joined = graph.add("ConcatenateVideo", codec="auto",
                           **{f"videos.video{index}": part for index, part in enumerate(slices)})
        graph.add("SaveVideo", video=joined, filename_prefix="prison-10s/许七安醒来10秒", format="mp4")
    return graph.nodes


def keep_outputs(graph, output_type):
    pending = [key for key, node in graph.items() if node["class_type"] == output_type]
    selected = set()
    while pending:
        key = pending.pop()
        if key in selected:
            continue
        selected.add(key)
        pending.extend(value[0] for value in graph[key]["inputs"].values()
                       if isinstance(value, list) and len(value) == 2 and value[0] in graph)
    return {key: node for key, node in graph.items() if key in selected}


def prepare_phase(graph, phase):
    if phase == "images":
        return keep_outputs(graph, "SaveImage")
    if phase == "video":
        anchors = {}
        for node in list(graph.values()):
            if node["class_type"] == "SaveImage":
                source = node["inputs"]["images"][0]
                filename = node["inputs"]["filename_prefix"] + ".png"
                key = str(len(graph) + 1)
                graph[key] = {"class_type": "LoadImage", "inputs": {"image": filename}}
                anchors[source] = [key, 0]
        for node in graph.values():
            if node["class_type"] == "MiniMaxH3ImageToVideo":
                for field in ("first_frame", "last_frame"):
                    node["inputs"][field] = anchors[node["inputs"][field][0]]
        return keep_outputs(graph, "SaveVideo")
    return graph


def stage_images(image_run, input_root):
    targets = []
    for index in (1, 2):
        for point in ("first", "tail"):
            stem = f"shot-{index:02d}-{point}"
            sources = list((image_run / "renders").rglob(f"*{stem}_*.png"))
            if len(sources) != 1:
                raise ValueError(f"应恰有一张已验收的 {stem} PNG。")
            targets.append((sources[0], input_root / "prison-10s" / f"{stem}.png"))
    for source, destination in targets:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def ffprobe(path, frames):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)
    ], text=True, timeout=45))
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), {})
    if (video.get("width"), video.get("height")) != SIZE or int(video.get("nb_read_frames", 0)) != frames:
        raise ValueError(f"{path.name} 的尺寸或帧数异常。")
    if Fraction(video["avg_frame_rate"]) != FPS or not audio:
        raise ValueError(f"{path.name} 缺少音轨或帧率异常。")
    return {"width": SIZE[0], "height": SIZE[1], "frames": frames, "fps": FPS,
            "duration_s": float(data["format"]["duration"]),
            "audio_codec": audio["codec_name"], "audio_channels": audio.get("channels")}


def assemble(directory):
    renders = directory / "renders"
    shots = []
    for index in (1, 2):
        clips = list(renders.rglob(f"*shot-{index:02d}_*.mp4"))
        if len(clips) != 1:
            raise ValueError(f"第 {index} 镜应恰有一条命名正确的 MP4。")
        ffprobe(clips[0], RAW_FRAMES)
        shots.append(clips[0])
    output = directory / "许七安醒来10秒.mp4"
    pending = output.with_suffix(".partial.mp4")
    filters = []
    for index in (0, 1):
        filters.extend([
            f"[{index}:v]trim=end_frame=120,setpts=PTS-STARTPTS[v{index}]",
            f"[{index}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=end=5,asetpts=PTS-STARTPTS[a{index}]",
        ])
    filters.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-y"]
    for part in shots:
        cmd.extend(["-i", str(part)])
    cmd.extend(["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(pending)])
    subprocess.run(cmd, check=True, timeout=120)
    result = ffprobe(pending, 240)
    if abs(result["duration_s"] - 10.0) > 1 / FPS or result["audio_channels"] != 2:
        raise ValueError("成片时长或双声道校验失败。")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(pending),
                    "-f", "null", "-"], check=True, timeout=60)
    checksum = hashlib.sha256()
    with pending.open("rb") as media:
        for chunk in iter(lambda: media.read(1024 * 1024), b""):
            checksum.update(chunk)
    result["sha256"] = checksum.hexdigest()
    pending.replace(output)
    (directory / "delivery.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("directory", type=Path)
    prepare.add_argument("--backend", required=True, choices=("s4000", "910c-bf16", "910c-int8"))
    prepare.add_argument("--project", default=DEFAULT_PROJECT, type=Path)
    prepare.add_argument("--phase", choices=("all", "images", "video"), default="all")
    stage = sub.add_parser("stage-images")
    stage.add_argument("image_run", type=Path)
    stage.add_argument("input_root", type=Path)
    export = sub.add_parser("assemble")
    export.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.backend == "s4000" and args.phase != "all":
            parser.error("分阶段启动适用于 910C。")
        project = json.loads(args.project.read_text(encoding="utf-8"))
        target = args.directory / "shots" / "001-prison-10s.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        graph = prepare_phase(build(project, args.backend, automatic=args.phase == "all"), args.phase)
        data = json.dumps(graph, ensure_ascii=False, indent=2) + "\n"
        if target.exists() and target.read_text(encoding="utf-8") != data:
            parser.error("该目录已有不同的任务图，请为新试跑另建目录。")
        if not target.exists():
            target.write_text(data, encoding="utf-8")
        print(target)
    elif args.command == "stage-images":
        stage_images(args.image_run, args.input_root)
    else:
        assemble(args.directory)


if __name__ == "__main__":
    main()
