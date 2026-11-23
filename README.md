<h1 align="center">MTT H3 有声短片工作台</h1>

<p align="center">在摩尔线程 S4000 或昇腾 910C 上，用 MiniMax H3 从文字、参考图生成有声短片。</p>

<p align="center">
  <img alt="单张 MTT S4000 48 GB" src="https://img.shields.io/badge/GPU-MTT%20S4000%2048%20GB-334155?style=flat-square">
  <img alt="MUSA 原生推理" src="https://img.shields.io/badge/runtime-MUSA%20native-0f766e?style=flat-square">
</p>

<p align="center">
  <a href="#无限画布与长剧续跑">无限画布与长剧续跑</a> ·
  <a href="#h3-image-生图与有声镜头">H3 Image 生图</a> ·
  <a href="#安装与运行">安装与运行</a> ·
  <a href="#实卡实验与成本">实卡实验与成本</a> ·
  <a href="#机型与入口路线">机型与入口路线</a> ·
  <a href="#冷启动发布门槛">冷启动发布门槛</a>
</p>

同一 S4000 镜头的历史原生对照中，**四步配置把进程耗时从 6 分 04 秒降到 4 分 03 秒，缩短约 33%**。中文工作台把关键帧、动作、台词和宽高集中到一套可编辑画布，生成后可在画布内审听模型原声。

> [!NOTE]
> 克隆仓库主分支可使用开机诊断、S4000 原生生图和 910C 自动分阶段工作流。运行需有匹配设备的推理环境和模型权重。

## H3 Image 生图与有声镜头

### 开机后从「H3 短片」开始

用 `start.sh` 启动后，在 ComfyUI 左侧菜单选择 **H3 短片 → 新建 H3 镜头**。工作台会预选检测到的设备档位。选择画质和关键帧来源，填入镜头动作和可选对白，再点「打开工作流」。从文字生图时，填写首帧描述及高清档所需的尾帧变化。使用已有关键帧时，在打开的画布上从「加载首帧」「加载尾帧」选择自己的图片。

检查画布、调整提示词和种子，点击 ComfyUI 的「运行」。S4000 通过 MUSA 运行 H3 Image 和 H3 视频。910C 先用独立进程做生图，再将首尾帧交给原声视频节点。

| 设备 | 画质档 | 关键帧来源 |
| :--- | :--- | :--- |
| S4000 | 448×800 快速首帧、576×1024 高清首尾帧试验档 | H3 Image 生成或使用已有关键帧 |
| 昇腾 910C | 576×1024 INT8、576×1024 剪枝 BF16 | H3 Image 生成或使用已有首尾帧 |

「需要配置」会展开具体检查项。S4000 检查设备、推理程序、视频工具与模型文件。910C 检查两颗逻辑 NPU、当前设备、文本编码器配置，以及所选流程需要的权重。两边都检查输出目录是否可写。缺项补齐后可点「重新检测」。这些是启动前检查。S4000 生成时还会验证权重指纹，整镜清晰度、口型、算子命中和峰值显存取决于实跑。填写实例小时价可查看历史纯视频推理的费用估算，生图、冷启动和排队时间另计。

`start.sh` 默认在 6008 端口提供启动诊断，6006 端口提供工作台。通过 SSH 访问时可同时转发两个端口，并把 `H3_COMFY_PUBLIC_URL` 指向用户实际访问的 ComfyUI 地址。诊断页会显示「打开 ComfyUI」。工作台会尝试识别 910C，也可以设 `H3_WORKBENCH_BACKEND=s4000` 或 `H3_WORKBENCH_BACKEND=910c` 明确指定。

### S4000 · 从文生图到有声镜头

在「新建 H3 镜头」选择「H3 Image 生成关键帧」。默认首帧走文生图，高清档的尾帧引用首帧做编辑。生成的 PNG 和有声 MP4 都保存在 ComfyUI 输出目录。画布打开后可修改提示词、尺寸、种子及节点接线。也可以把「加载图像」接到首帧节点的 `reference`，并在提示词里写 `<Picture 1>` 做定妆图编辑。

在共享模型库准备 `diffusion_models/minimax_h3_image_pruned-Q8_0.gguf`、`vae/minimax_h3_image_vae_fp16.safetensors`，以及带视觉编码器的 `text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf`。视频正在使用同版本编码器时可直接复用。用下面的命令核对完整 SHA-256，并链接到 `MTT_H3_MODELS`。可在无卡模式执行。

```bash
python3 image_runtime.py --shared /path/to/shared/models
```

默认用运行时目录里的 `sd-cli`。需要指定另一份兼容的 MUSA 程序时设置 `MTT_H3_IMAGE_ENGINE`。ComfyUI 仍以 CPU 模式启动，生图和有声视频分别由独立的 MUSA 进程运行。ConvRot INT8 在已测的 S4000 运行时首步算子失败，`MTT_H3_IMAGE_FORMAT=convrot-int8` 仅用于兼容性实验。 [Q8 生图主模型](https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF) · [图像 VAE](https://modelscope.cn/models/Comfy-Org/MiniMax-H3-Image) · [原生生图与参考编辑说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/models/minimax_h3_image.md)

### 昇腾生图权重

在 910C 上使用工作台的 H3 Image 前，将以下文件放入 ComfyUI 的模型目录。 [ComfyUI H3 Image 官方教程](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image)

| 模型目录 | 文件 |
| :--- | :--- |
| `models/diffusion_models/` | `minimax_h3_image_pruned_int8_convrot.safetensors` |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `models/vae/` | `minimax_h3_image_vae_fp16.safetensors` |

### 昇腾 910C · 同一张画布出首尾帧和原声视频

从工作台菜单打开的 910C 画布包含独立生图节点。它先释放视频服务缓存的 NPU 权重，在一个只监听本机的进程中使用分割注意力生成首尾帧。生图进程关闭后，两张图直接送给使用 PyTorch 注意力的 FL2VA 视频节点。已有首尾帧时可以在菜单选择「使用已有关键帧」，直接走视频。已有角色定妆图可以接到生图节点的 `reference`，在首帧提示词使用 `<Picture 1>`。

用已有 CANN/TorchNPU 环境启动，文本编码在 `npu:1`，扩散与解码在 `npu:0`。视频默认使用剪枝 BF16 与八步 Turbo LoRA。除上表生图权重外，还需在 ComfyUI 的模型路径提供以下文件。已落在持久盘的权重可以用 `COMFYUI_MODEL_PATHS` 指向现有 Extra Model Paths YAML，或通过软链接复用。

| 模型目录 | 文件 |
| :--- | :--- |
| `models/diffusion_models/` | `minimax_h3_fl2va_pruned_bf16.safetensors` |
| `models/loras/` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors`、`minimax_h3_audio_vae_fp32.safetensors` |

实卡已通过 576×1024 两阶段推理，工作台自动衔接另用本地流程验证。工作流保留 H3 原生声轨，建议听完每段再决定剪点。

## 无限画布与长剧续跑

在「H3 短片」菜单创建需要的镜头。也可以打开自己以前保存的 ComfyUI 画布继续编辑。

1. **只跑一个输出。** 选中「保存关键帧」或某个「有声视频」输出节点，在「运行」菜单使用「执行到选定输出节点」。同时选多个输出时，ComfyUI 会运行这些输出及它们的上游。整图运行会排队所有启用的输出。
2. **增加镜头。** 框选镜头的加载图片和视频节点，一起复制粘贴，或从节点库添加同样的节点。然后给新镜头换素材、动作和台词。各场可摆成单独一行，按住空格拖动画布。修改模型、采样配置或接线前，可另存一份画布。
3. **长剧分批跑。** 每镜导出一份 API JSON。导出前可将该镜头与所需上游一起复制到空白工作流，只保留该镜头需要的输出节点。将 API JSON 放入同一项目的 `shots/` 并编号，例如 `001-opening.json`、`002-wakeup.json`。镜头文件和项目记录宜放在持久盘，引用的图片放在目标 ComfyUI 的输入目录。

在 Linux 或 macOS 上用 Python 3 运行队列。它通过自托管 ComfyUI 的队列接口工作。每次完成一镜就回收素材到 `renders/`，保存回执到 `receipts/`。重跑同一命令会跳过已完成项。按 Ctrl+C 会停止提交新镜头，当前镜头仍由 ComfyUI 处理，恢复后会接回其历史或队列。需要控制租卡时间时，可以给每次运行设镜头数上限。

```bash
python3 tools/queue_project.py /path/to/episode --check
python3 tools/queue_project.py /path/to/episode --server http://127.0.0.1:6006 --limit 10
```

如果一镜失败，先检查回执，再用 `--retry-failed` 重提失败镜头。如果重启了 ComfyUI，且先前任务的历史和队列都查不到，确认旧任务已停止且没有可用输出后，用 `--retry-missing 007-shot-name` 指定重提。修改已提交的图或素材时，给修订镜头分配新的编号，保留旧回执供核对。

五分钟成片可以分成约六十个五秒镜头，按场次分批展开放在画布上。单镜时长和计算量仍由实际模型和显卡决定。输出保持分镜文件，剪辑时按编号取用。

模型权重可以集中在数据盘上，再把已缓存的文件软链接到对应模型目录。原生生图节点读取上表的 ComfyUI 权重，S4000 视频节点读取 `MTT_H3_MODELS`（默认是运行时下的 `models/`）。选择的本地节点和加载器须支持相应权重格式。自托管推理只调用自己的 ComfyUI，不需要第三方生图或视频 API 账号。

## 安装与运行

**准备环境。** 需要单张可见的 MTT S4000（48 GB）、可运行的 ComfyUI、Python 3.11 或更新版本、匹配的 MUSA 运行库、推理程序和 FFmpeg（含 FFprobe）。画布以 CPU 模式启动，由独立 MUSA 进程推理。模型目录需要约 43.86 GB 加四步 LoRA 的空间。程序与模型路径见下方运行时清单。

**安装节点并启动。** 在已有上述环境的机器上执行。按需将 `COMFYUI_DIR` 和 `COMFYUI_PYTHON` 指向现有 ComfyUI 及其 Python 环境。

```bash
git clone https://github.com/amethyst-cosmos/comfyui-mtt-h3.git mtt-s4000-h3
cd mtt-s4000-h3

export COMFYUI_DIR=/root/ComfyUI
"${COMFYUI_PYTHON:-python3}" runtime.py && bash start.sh
```

已有 910C CANN 镜像时使用相同脚本，指定 ComfyUI、Python 和持久盘模型路径。工作台的文本加载器会选第二颗逻辑 NPU。

```bash
export COMFYUI_DIR=/path/to/ComfyUI
export COMFYUI_PYTHON=/path/to/venv/bin/python
export COMFYUI_MODEL_PATHS=/path/to/model_paths.yaml
export COMFYUI_OUTPUT=/path/to/persistent/output
H3_WORKBENCH_BACKEND=910c bash start.sh
```

**打开工作台。** 默认访问 `http://127.0.0.1:6006`。从「H3 短片」新建自己的镜头，或打开已有的 ComfyUI 工作流。S4000 两个档位均为 124 帧、24 fps，每个镜头约 5.17 秒。

<details>
<summary><strong>展开运行时文件清单与预检命令</strong></summary>

```text
/root/mtt-h3-runtime/
  sd-cli
  native_flash_bridge.so
  sd-cli-chunk4096        # 首尾帧试验档
  trace_bridge.so        # 首尾帧试验档
  models/
    diffusion_models/minimax_h3_fl2va_pruned-Q8_0.gguf
    text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
    vae/minimax_h3_sol_vae_fp16.safetensors
    vae/minimax_h3_audio_vae_fp32.safetensors
    loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
```

使用 `MTT_H3_RUNTIME` 指定程序目录，`MTT_H3_MODELS` 指定已挂载的权重目录。基础档使用 `sd-cli` 和原生 muDNN Flash 桥接库。高清试验档还需要前馈和 QKV 投影按最多 4,096 token 分块的程序及跟踪库。

| 命令 | 用途 |
| :--- | :--- |
| `python3 runtime.py` | 检查单张可见 S4000、基础档运行文件及模型指纹 |
| `python3 runtime.py --files-only` | 无卡时检查基础档文件大小、首尾指纹及 LoRA SHA-256 |
| `python3 runtime.py --detail` | 检查首尾帧试验档，也可加 `--files-only` |
| `bash start.sh` | 挂载本节点并启动 ComfyUI |

使用 `COMFYUI_OUTPUT` 修改输出位置，使用 `COMFYUI_PORT` 修改端口。默认只监听本机，采用实例私有服务映射时可设置 `COMFYUI_LISTEN=0.0.0.0`。

</details>

## 实卡实验与成本

### 同一镜头的八步和四步

2026 年 11 月的原生 S4000 对照固定首帧、提示词、种子和输出规格。四步组使用四步 Turbo LoRA。

| 单张 MTT S4000 · 448×800 · 124 帧 | 八步 | 四步 Turbo |
| :--- | ---: | ---: |
| 推理进程耗时 | 364.3 秒 | **243.5 秒** |
| 采样阶段 | 265.7 秒 | **141.4 秒** |
| 完整解码与音轨 | 通过 | 通过 |

这条样片里，两版人物外观和衣料纹理接近，四步版的双手贴头动作约晚 0.75 秒。两组均在完整写出后发生清理阶段异常退出，文件核验通过。

### 两种画质档的原生基线

| 档位 | S4000 历史原生结果 | ComfyUI 封装 |
| :--- | :--- | :--- |
| **448×800** · 首帧 | 两条有声样片，约 **4 分钟落盘**（240.6–242.9 秒） | 桌面链路已验证，待实卡冷启动 |
| **576×1024** · 首尾帧 | 一条分块诊断样片，约 **8 分钟落盘**（481.0 秒），黑屏检测为 0 秒 | 48 GiB 实例通过启动诊断与两镜有声导出，单镜原生 481.6–482.6 秒 |

### 单条算力成本

按历史记录的单卡小时价折算，每条均为 124 帧、24 fps。保留 NVIDIA 参考记录，方便同时比较**小时价、等待时间和输出规格**。

| 方案 | 输出规格 | 历史费率 | 记录耗时 | 折算每条 |
| :--- | :--- | ---: | ---: | ---: |
| **S4000 · H3 Q8 四步** | 448×800 · 首帧有声 | ¥1.86/小时 | 243.5 秒 | **¥0.13** |
| S4000 · H3 Q8 八步 | 448×800 · 同镜头有声 | ¥1.86/小时 | 364.3 秒 | ¥0.19 |
| S4000 · H3 Q8 四步分块 | 576×1024 · 首尾帧有声 | ¥1.86/小时 | 482.0 秒 | ¥0.25 |
| RTX PRO 6000 · H3 Q8 四步 | 576×1024 · 首帧有声 | ¥8.36/小时 | 256.5 秒 | ¥0.60 |
| RTX PRO 6000 · HyperFlow Flash INT8 | 864×480 · 首尾帧无声 | ¥6.98/小时 | 56.7 秒 | ¥0.11 |

折算公式为「记录耗时（秒）÷ 3,600 × 小时价」。在同镜头 S4000 对照中，四步约省 **¥0.06/条**。跨卡记录的工作流、分辨率或音频配置有差异，HyperFlow 一行采用 ComfyUI 任务耗时，其余为原生进程耗时。实例实际按开关机时间计费，还要考虑准备、空转、重试和存储。 [AutoDL 计费说明](https://www.autodl.com/docs/price/)

## 自选尺寸与输出

- **宽高独立设置。** 每边为 32 的倍数，最长边 1,024 像素。`0` 沿用档位默认值。基础档面积上限 358,400 像素，可试 800×448 和 576×576。首尾帧档上限 589,824 像素，可试 1024×576 和 768×768。自选尺寸已验证桌面参数链路，待实卡验收。
- **构图与原声。** 自选尺寸按目标画幅居中裁切关键帧。节点检查解码、分辨率、帧数、音轨和近乎全黑的输出。可在画布内播放带原声的 MP4，台词清晰度与口型以审片为准。
- **每次运行有记录。** 每个镜头在输出目录下独立保存原片、预览、引擎日志和运行回执。`VIDEO` 端口可接后续节点。

## 机型与入口路线

| 机型 | 状态 | 适配工作 |
| :--- | :--- | :--- |
| **MTT S4000（48 GB）** | 当前源码预览版的目标机型 | 576×1024 的 H3 Image 文生图、参考编辑、有声视频和 10 秒导出已在实卡跑通。镜像空盘安装和其余自选尺寸仍需验收。 |
| **MTT S5000** | 后续适配候选，待实机 | 按目标主机的驱动和 MUSA SDK 单独构建推理程序，复验 Flash 注意力、模型加载与有声输出。在相同镜头上建立耗时和显存基线，再评估更高分辨率、时长和吞吐。 [摩尔线程 S5000 容器指南](https://docs.mthreads.com/musa-sdk/version-5.2.0/docker_guide/) |
| **MTT S3000 与其他 MTT 机型** | 待逐型号验证 | 根据可用硬件和工具链确定能运行的档位，每个机型单独记录验收结果。 |

**当前入口是 ComfyUI 画布。** 视频节点把关键帧、提示词、台词、种子和尺寸交给原生 MUSA 进程，保留视频和运行回执。H3 Image 节点也可独立保存关键帧 PNG。

**规划中的 Agent 任务入口。** 先在 S4000 上验证可编程任务接口。Agent 可提交关键帧、提示词、台词、种子和尺寸，凭任务 ID 查询进度、取消任务、取回视频和验收结果。ComfyUI 与 Agent 入口计划共用推理后端、机型校验和媒体验收。若分发平台要求 `config.yaml`，按该平台的接入规范另行提供。

## 冷启动发布门槛

- 在干净的单张 S4000 上，用相同素材、种子和参数走原生命令与 ComfyUI 两条入口。依次验 448×800 首帧、576×1024 首尾帧，再验 800×448 和 576×576。
- 记录启动和模型加载时间、出片耗时、峰值显存与主机内存。核对尺寸、124 帧、音轨、黑屏和结束后的进程状态。遇到明显回退时复查日志和输出。
- 模型及衍生权重的使用和分发遵循上游许可。 [MiniMax H3 社区许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
