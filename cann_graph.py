"""API graph building blocks used by the Ascend image worker."""

CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
IMAGE = "minimax_h3_image_pruned_int8_convrot.safetensors"
IMAGE_VAE = "minimax_h3_image_vae_fp16.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
VIDEO_MODELS = {
    "cann-int8": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "cann-bf16": "minimax_h3_fl2va_pruned_bf16.safetensors",
}
SIZE = (576, 1024)
FPS = 24
FRAMES = 124


class Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, node_type, **inputs):
        node_id = str(len(self.nodes) + 1)
        self.nodes[node_id] = {"class_type": node_type, "inputs": inputs}
        return [node_id, 0]


def cann_image(graph, model, clip, vae, prompt, seed, reference, size):
    references = {} if reference is None else {"images.image_1": reference}
    condition = graph.add("MiniMaxH3ImageConditioning", clip=clip, vae=vae, prompt=prompt,
                          width=size[0], height=size[1], **references)
    negative = graph.add("ConditioningZeroOut", conditioning=condition)
    sampled = graph.add("KSampler", model=model, positive=condition, negative=negative,
                        latent_image=[condition[0], 1], seed=seed, steps=6, cfg=1.0,
                        sampler_name="euler", scheduler="simple", denoise=1.0)
    return graph.add("VAEDecode", samples=sampled, vae=vae)
