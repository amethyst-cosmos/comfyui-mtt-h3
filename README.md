# MiniMax H3 摩尔线程 S4000 中文工作流

这是面向单张 MTT S4000 的源码预览版。在 ComfyUI 上传首帧、填输出宽高、写动作和可选台词，再点运行。节点调用独立的 MUSA 推理程序，完成技术检查后原声预览。VIDEO 输出可继续接到后续工作流。

## 当前验证范围

| 档位 | 历史原生命令 | 本封装 |
|---|---|---|
| 448×800，首帧，124 帧，24 fps | S4000 两条约 4 分钟出片 | 桌面流程验证，待 S4000 冷启动复验 |
| 576×1024，首尾帧，124 帧，24 fps | S4000 分块诊断版本出片一条 | 桌面流程验证，待 S4000 冷启动复验 |
| 自选尺寸 | 待实卡测试 | 已验证尺寸输入、参数传递和输出检查 |
| S3000 与其他 GPU | 待专项适配 | 当前机型检查会拦截 |

本项目使用 MiniMax H3 FL2VA 系列的 Q8_0 转换权重，配 Q4_K_M 文本编码器、SOL 视频解码器、音频 VAE 及四步 Turbo LoRA。S4000 原生 muDNN Flash 由既有桥接库调用。高分辨率配置采用前馈与 QKV 投影每块最多 4,096 token 的运行版本并保留数值跟踪。采样设置与各自的历史成功调用对齐。

## 安装到已经准备好的运行环境

使用已有 ComfyUI 环境。当前节点注册在 ComfyUI 0.45.3、Python 3.13 CPU 模式验证过。把本仓库放入 `custom_nodes/mtt-s4000-h3`，或用启动脚本创建软链。启动脚本只加载本节点，ComfyUI 负责画布、输入和预览，独立进程负责 MUSA 推理。

默认路径如下。可以通过 `MTT_H3_RUNTIME` 和 `MTT_H3_MODELS` 覆盖根目录。

```text
/root/mtt-h3-runtime/
  sd-cli
  native_flash_bridge.so
  sd-cli-chunk4096
  trace_bridge.so
  models/
    diffusion_models/minimax_h3_fl2va_pruned-Q8_0.gguf
    text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
    vae/minimax_h3_sol_vae_fp16.safetensors
    vae/minimax_h3_audio_vae_fp32.safetensors
    loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
```

```bash
python3 runtime.py --files-only
python3 runtime.py --files-only --detail
bash start.sh
```

在 S4000 实例可用时，运行 `python3 runtime.py` 将检查 **单张可见 S4000**。镜像中需已有匹配的 MUSA 开发运行库及 FFmpeg、FFprobe。基础档依赖清单不需要高清二进制或 trace 库。

默认监听本机 6006。AutoDL 私有实例采用已核实的服务映射时，可设置 `COMFYUI_LISTEN=0.0.0.0`。可用 `COMFYUI_DIR`、`COMFYUI_PYTHON`、`COMFYUI_OUTPUT` 对接具体的基础镜像。打开 ComfyUI 的自定义节点模板或直接拖入 `example_workflows/` 下的 JSON。两套模板共约 4.7 KB，包含画布设置，无模型数据。

## 自选尺寸与运行排查

- 宽和高分别填写，`0` 表示该边沿用档位默认值。正方形、横向和纵向均可输入。每边须为 32 的倍数，最长边不超过 1,024 像素。
- 基础档使用一张首帧，面积上限为 358,400 像素，可填 800×448 或 576×576。首尾帧档需要两张图，面积上限为 589,824 像素，可填 1024×576 或 768×768。面积上限取自对应档位历史样片。所有自选尺寸均属于待实卡验证的试验配置。
- 自选尺寸的关键帧按目标画幅等比例缩放并居中裁切。若构图有严格要求，先在画布中调好关键帧比例。两组历史默认尺寸继续沿用原生程序的预处理。提示词由用户直接输入。
- 只要帧数、分辨率、音轨、解码与黑屏检查通过，就在节点内预览保留原声的 MP4。台词是否逐字准确、口型是否同步由用户审片。
- 每次运行保留一个独立目录，含原片、预览、引擎日志和结果凭据。错误带日志路径。ComfyUI 中断或超过时限会停止对应推理进程组。
- 视频主模型、文本编码器和两个 VAE 共 43.86 GB，另有 LoRA。用数据盘挂载或经核对后接 AutoDL 公共模型软链，检查长度、首尾指纹以及 LoRA SHA-256。基础环境与权重是两个交付项。

## 冷启动发布门槛

用相同输入、种子、模型与原生 GPU，分别执行原生命令和 ComfyUI 入口。记录从开机到可点运行的时间、出片时间、GPU 峰值显存、进程及系统内存、可播放音视频和退出情况。先验 448×800，再验 576×1024。遇到 OOM、黑屏、缺音轨或明显回退时，停留在本地候选阶段。S3000 单独检查预期显存、驱动和注意力内核后再排入验证。

本仓库是画布与运行适配源码候选。冷启动验证使用独立目录，生产运行时文件、权重来源和干净镜像待归档核对。发行使用 MiniMax H3 或衍生权重时，按社区许可证处理标识、分发地域与第三方许可。

## AutoDL Art 交付与激励

1. **原生命令作为参考入口。** 先在普通 AutoDL 的 S4000 实例复验原生命令和本工作流，再从干净系统盘制作环境镜像。系统盘保存 ComfyUI、MUSA 运行依赖、推理程序和启动脚本，模型目录使用挂载。检查挂载能否在另一账号或新实例复现。
2. **GitHub 作为源码入口。** 仓库保存节点代码、工作流、版本说明和冷启动验证结果。AutoDL Art 的「发布 → 镜像」表单依次填写 GitHub 地址与镜像名称、导入镜像、写镜像 README。发布页审核说明要求 `/root` 中有项目 Git 仓库，并可运行仓库内的简单示例。 [镜像文档](https://autodl.art/docs/image/)
3. **应用作为部署入口。** 向 AutoDL 确认对应镜像可用的应用发布入口、S4000 调度及服务端口，再提交一键部署材料。本次查看的 Art 创建面板未列出 S4000，同时普通 AutoDL 有独立摩尔线程机型文档。 [Art 应用广场](https://www.autodl.art/app/market) · [普通 AutoDL 摩尔线程文档](https://api.autodl.com/docs/mt_gpu/)
4. **激励按通过审核的交付类型进入。** 已审核上架的镜像可在创作激励入口申请，公开规则默认 ¥0.05/小时，原创代码或原创算法加成由平台审核。应用审核上架后默认参与，公开规则默认 ¥0.10/小时。提现门槛 ¥100，平台按有效使用时长结算。 [创作激励规则](https://autodl.art/docs/reward_rule/)

已上架产品的包装参考

- HyperFlow 把中文工作流放进 ComfyUI，给了示例和模型公共仓软链，用户从 6006 进入画布。 [应用页](https://www.autodl.art/app/market/343)
- Pikaflex 把环境与模型拆开。页面给出的镜像约 19.25 GB，较大的模型通过公共库及另行下载就位。 [应用页](https://www.autodl.art/app/market/337)
- LianG 则用 6008 作为普通制作入口，6006 留给专业 ComfyUI 调试。同一产品可以保留专业画布。 [应用页](https://www.autodl.art/app/market/323)
- 可查看已公开的 MiniMax H3 AutoDL 项目如何组织脚本、工作流与权重依赖。 [GitHub 示例](https://github.com/zw2029181901/autodl-minimax-h3-comfyui)

[MiniMax H3 社区许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) · [AutoDL Art 公共模型](https://www.autodl.art/docs/app_model/)
