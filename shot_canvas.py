"""Build editable ComfyUI canvases from user inputs and supported device settings."""

from . import cann_graph, image_runtime, runtime


class Canvas:
    def __init__(self):
        self.nodes = []
        self.links = []

    def add(self, kind, title, position, size, *, inputs=(), outputs=(), widgets=(), models=()):
        properties = {"Node name for S&R": kind}
        if models:
            properties["models"] = list(models)
        node = {
            "id": len(self.nodes) + 1, "type": kind, "title": title,
            "pos": list(position), "size": list(size), "flags": {}, "mode": 0,
            "order": len(self.nodes), "properties": properties,
            "inputs": [{"name": name, "type": kind, "link": None} for name, kind in inputs],
            "outputs": [{"name": name, "type": kind, "links": None} for name, kind in outputs],
            "widgets_values": list(widgets),
        }
        self.nodes.append(node)
        return node

    def connect(self, source, target, target_slot=0, source_slot=0):
        output = source["outputs"][source_slot]
        socket = target["inputs"][target_slot]
        link = len(self.links) + 1
        if output["type"] != socket["type"] or socket["link"] is not None:
            raise ValueError("工作流节点的接线无效。")
        output["links"] = [*(output["links"] or []), link]
        socket["link"] = link
        self.links.append([link, source["id"], source_slot, target["id"], target_slot, output["type"]])

    def workflow(self, scale):
        return {"last_node_id": len(self.nodes), "last_link_id": len(self.links), "nodes": self.nodes,
                "links": self.links, "groups": [], "config": {},
                "extra": {"ds": {"scale": scale, "offset": [32, 32]}}, "version": 0.4}

    def load_image(self, point, position):
        return self.add("LoadImage", f"加载{point}", position, (290, 315),
                        outputs=(("IMAGE", "IMAGE"), ("MASK", "MASK")), widgets=("", "image"))

    def save_image(self, point, position, name):
        return self.add("SaveImage", f"保存{point} PNG", position, (285, 210),
                        inputs=(("images", "IMAGE"),), widgets=(name,))

    def save_video(self, position, name):
        return self.add("SaveVideo", "保存有声 MP4", position, (325, 420),
                        inputs=(("video", "VIDEO"),), outputs=(("video", "VIDEO"),),
                        widgets=(name, "mp4", "auto"))


def musa_canvas(profile, source, motion, dialogue, first_prompt, last_prompt):
    width, height, _, needs_tail = runtime.PROFILES[profile]
    canvas = Canvas()

    def frame(point, x, y, prompt, seed):
        if source == "existing":
            return canvas.load_image(point, (x, y))
        image = canvas.add("MttS4000H3Image", f"H3 Image · {point}", (x, 50), (365, 250),
                           inputs=(("reference", "IMAGE"),), outputs=(("IMAGE", "IMAGE"),),
                           widgets=(prompt, seed, "fixed", *image_runtime.SIZE))
        name = "first" if point == "首帧" else "last"
        canvas.connect(image, canvas.save_image(point, (x, 350), f"h3-musa/{name}"))
        return image

    first = frame("首帧", 30, 50, first_prompt, 2026112301)
    sockets = [("first", "IMAGE"), ("tail", "IMAGE")]
    x = 860 if source == "generate" and needs_tail else 465
    video = canvas.add("MttS4000H3Shot", "H3 · 生成有声镜头", (x, 50), (420, 390),
                       inputs=sockets, outputs=(("有声视频", "VIDEO"),),
                       widgets=(profile, motion, dialogue, 20261121211, "fixed", width, height))
    canvas.connect(first, video)
    if needs_tail:
        tail = frame("尾帧", 445 if source == "generate" else 30, 410,
                     "Edit <Picture 1>. " + last_prompt, 2026112311)
        if source == "generate":
            canvas.connect(first, tail)
        canvas.connect(tail, video, target_slot=1)
    canvas.connect(video, canvas.save_video((x + 470, 50), "h3-musa/shot"))
    return canvas.workflow(0.75 if not needs_tail else 0.62)


def model_file(directory, filename, repository="Comfy-Org/MiniMax-H3"):
    location = filename if directory == "loras" else f"{directory}/{filename}"
    return {"directory": directory, "name": filename,
            "url": f"https://huggingface.co/{repository}/resolve/main/{location}"}


def ascend_weights(profile):
    return {
        "image": (("diffusion_models", cann_graph.IMAGE), ("vae", cann_graph.IMAGE_VAE)),
        "video": (("text_encoders", cann_graph.CLIP), ("diffusion_models", cann_graph.VIDEO_MODELS[profile]),
                  ("vae", cann_graph.VIDEO_VAE), ("vae", cann_graph.AUDIO_VAE), ("loras", cann_graph.TURBO)),
    }


def ascend_canvas(profile, source, motion, dialogue, first_prompt, last_prompt):
    canvas = Canvas()
    add = canvas.add
    connect = canvas.connect

    if source == "generate":
        first = last = add("MttAscendH3ImagePair", "H3 Image · 独立生成首尾帧", (30, 65), (360, 300),
                           inputs=(("reference", "IMAGE"),), outputs=(("首帧", "IMAGE"), ("尾帧", "IMAGE")),
                           widgets=(first_prompt, "Edit <Picture 1>. " + last_prompt,
                                    2026112301, "fixed", *cann_graph.SIZE))
        for slot, point in enumerate(("首帧", "尾帧")):
            name = "first" if slot == 0 else "last"
            save = canvas.save_image(point, (30, 420 + 245 * slot), f"h3-cann/{name}")
            connect(first, save, source_slot=slot)
    else:
        first = canvas.load_image("首帧", (30, 65))
        last = canvas.load_image("尾帧", (30, 430))

    clip = add("MttAscendH3Clip", "H3 文本编码 · npu:1", (480, 65), (330, 95),
               outputs=(("CLIP", "CLIP"),), widgets=(cann_graph.CLIP,),
               models=(model_file("text_encoders", cann_graph.CLIP),))
    filename = cann_graph.VIDEO_MODELS[profile]
    model = add("UNETLoader", "H3 FL2VA", (480, 195), (330, 95),
                outputs=(("MODEL", "MODEL"),), widgets=(filename, "default"),
                models=(model_file("diffusion_models", filename),))
    vae = add("VAELoader", "视频 VAE", (480, 325), (330, 65),
              outputs=(("VAE", "VAE"),), widgets=(cann_graph.VIDEO_VAE,),
              models=(model_file("vae", cann_graph.VIDEO_VAE),))
    audio_vae = add("VAELoader", "音频 VAE", (480, 430), (330, 65),
                    outputs=(("VAE", "VAE"),), widgets=(cann_graph.AUDIO_VAE,),
                    models=(model_file("vae", cann_graph.AUDIO_VAE),))
    lora = add("LoraLoaderModelOnly", "Turbo LoRA", (880, 420), (345, 95),
               inputs=(("model", "MODEL"),), outputs=(("MODEL", "MODEL"),), widgets=(cann_graph.TURBO, 1),
               models=(model_file("loras", cann_graph.TURBO, "lightx2v/Minimax-h3-Turbo"),))
    connect(model, lora)
    if dialogue:
        motion += f"\n角色用普通话清楚说一次：「{dialogue}」。说完闭嘴。"
    conditioning = add("MiniMaxH3ImageToVideo", "镜头动作、运镜与对白", (880, 65), (345, 290),
                       inputs=(("clip", "CLIP"), ("vae", "VAE"), ("first_frame", "IMAGE"), ("last_frame", "IMAGE")),
                       outputs=(("positive", "CONDITIONING"), ("LATENT", "LATENT")),
                       widgets=(motion, *cann_graph.SIZE, cann_graph.FRAMES))
    for slot, node in enumerate((clip, vae, first, last)):
        connect(node, conditioning, target_slot=slot, source_slot=int(slot == 3 and source == "generate"))

    noise = add("RandomNoise", "视频种子", (1295, 65), (230, 90),
                outputs=(("NOISE", "NOISE"),), widgets=(2026112303, "fixed"))
    sampler = add("KSamplerSelect", "视频采样器", (1295, 185), (230, 65),
                  outputs=(("SAMPLER", "SAMPLER"),), widgets=("res_multistep",))
    scheduler = add("BasicScheduler", "八步调度", (1295, 285), (230, 110),
                    inputs=(("model", "MODEL"),), outputs=(("SIGMAS", "SIGMAS"),), widgets=("simple", 8, 1))
    guider = add("BasicGuider", "视频引导", (1295, 435), (230, 70),
                 inputs=(("model", "MODEL"), ("conditioning", "CONDITIONING")), outputs=(("GUIDER", "GUIDER"),))
    connect(lora, scheduler)
    connect(lora, guider)
    connect(conditioning, guider, target_slot=1)
    samples = add("SamplerCustomAdvanced", "H3 视频与音频采样", (1600, 65), (275, 140),
                  inputs=(("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                          ("sigmas", "SIGMAS"), ("latent_image", "LATENT")),
                  outputs=(("output", "LATENT"), ("denoised_output", "LATENT")))
    for slot, node in enumerate((noise, guider, sampler, scheduler, conditioning)):
        connect(node, samples, target_slot=slot, source_slot=int(slot == 4))

    video = add("VAEDecode", "解码画面", (1960, 65), (225, 70),
                inputs=(("samples", "LATENT"), ("vae", "VAE")), outputs=(("IMAGE", "IMAGE"),))
    audio = add("VAEDecodeAudio", "解码原声", (1960, 175), (225, 70),
                inputs=(("samples", "LATENT"), ("vae", "VAE")), outputs=(("AUDIO", "AUDIO"),))
    connect(samples, video)
    connect(samples, audio)
    connect(vae, video, target_slot=1)
    connect(audio_vae, audio, target_slot=1)
    movie = add("CreateVideo", "合成有声镜头", (1960, 295), (225, 150),
                inputs=(("images", "IMAGE"), ("audio", "AUDIO")), outputs=(("VIDEO", "VIDEO"),),
                widgets=(cann_graph.FPS, 8, "sRGB", "auto", "none"))
    connect(video, movie)
    connect(audio, movie, target_slot=1)
    connect(movie, canvas.save_video((2265, 65), "h3-cann/shot"))
    return canvas.workflow(0.43)
