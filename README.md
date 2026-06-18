# DeepVoice Chat

本项目是语音信号处理课程大作业代码部分：使用兼容 OpenAI Chat Completions 的对话 API 生成文本对话内容和本轮语气提示词，再接入本地运行的 OpenBMB VoxCPM 开源语音合成模型，实现可调音色的语音聊天、单句合成、数据预处理和 LoRA 微调配置生成。默认推荐 DeepSeek，因为它在角色扮演和语气规划上比较适合这个任务；其他支持 `/chat/completions` 的模型服务也可以替换。

`VoxCPM/` 是上游源码目录，已经加入 `.gitignore`，不会被提交到本仓库。

## 1. 环境

推荐环境：

- Python 3.10-3.12，本项目默认使用 Python 3.11
- uv 0.11+
- PyTorch 2.5+
- CUDA 12+ 可选
- 播放音频可用 `ffplay`

本机检查结果：

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM
- RAM: 13GB
- CPU: Intel i7-12650H

上游 VoxCPM README 给出的显存需求约为：VoxCPM2 8GB、VoxCPM1.5 6GB、VoxCPM-0.5B 5GB。因此这台机器不适合在 GPU 上舒适运行 VoxCPM2。对话模型只走 API，不占本地显存；语音模型由用户按机器选择：

- 8GB+ 显存或 CPU 慢速演示：使用 `openbmb/VoxCPM2`，支持自然语言音色设计、参考音频克隆和 48kHz 输出。
- 6GB 左右显存：可尝试 `openbmb/VoxCPM1.5`，更稳但不支持 VoxCPM2 的全部音色设计能力。
- 4GB 显存/本机优先跑通：尝试 CPU 推理，或使用 `openbmb/VoxCPM-0.5B` 做低资源演示。
- 展示代码流程：先用 `--llm-backend echo` 验证链路，再接入 DeepSeek API。

## 2. 安装

```bash
uv python install 3.11
uv sync --extra dev
uv pip install -e VoxCPM
uv run voiceagent doctor
```

设置对话 API Key。项目不会自动读取 `.env`，可以把它作为本地记录，然后在 shell 中 `export`。默认 API base 是 DeepSeek，也可以替换为任何 OpenAI-compatible `/chat/completions` 服务：

```bash
cp .env.example .env
export VOICEAGENT_CHAT_API_KEY="sk-your-key-here"
export VOICEAGENT_CHAT_API_BASE="https://api.deepseek.com"
```

如需提前下载本地语音模型：

```bash
uv run huggingface-cli download openbmb/VoxCPM2 \
  --local-dir models/VoxCPM2
```

国内网络可用 ModelScope 下载 VoxCPM：

```bash
uv run python - <<'PY'
from modelscope import snapshot_download
snapshot_download("OpenBMB/VoxCPM2", local_dir="models/VoxCPM2")
PY
```

## 3. 运行

### 单句语音合成

```bash
uv run voiceagent synth \
  --tts-model models/VoxCPM2 \
  --tts-device cpu \
  --text "你好，这是本地语音聊天系统的测试。" \
  --voice "young female voice, warm and natural tone, medium pace, slight smile" \
  --output outputs/demo.wav
```

低资源替代：

```bash
uv run voiceagent synth \
  --tts-model openbmb/VoxCPM-0.5B \
  --tts-device cpu \
  --prompt-audio VoxCPM/examples/reference_speaker.wav \
  --prompt-text "This is a reference speaker example." \
  --text "你好，这是低资源模式下的声音克隆测试。" \
  --output outputs/demo_05b.wav
```

### 语音聊天

对话 API 每轮返回三部分：

- `spoken_text`: 真正朗读出来的回复。
- `voice_prompt`: 给 VoxCPM 的本轮语气提示词，例如 `relaxed tone, slightly slow pace, brief pauses before key suggestions`。
- `dialogue_state`: 不朗读、不传给 TTS 的连续状态摘要，用来给下一轮保留角色心境、关系进展、用户偏好和表达基线。

默认请求使用 Chat Completions 的 tool/function call：工具名是 `emit_voice_chat_turn`，参数由 JSON Schema 约束为上面三个字符串字段。这样比只在 prompt 里要求 JSON 更稳定；代码会读取工具调用参数并在本地校验。

`--voice` 是固定基础音色，`--reference-audio` 是固定参考说话人，对话模型只负责生成本轮语气，不会覆盖用户选择的音色。连续对话时系统会把整段历史都传回对话模型，包括之前的用户消息、之前朗读的 `spoken_text`、之前使用的 `voice_prompt` 和之前的 `dialogue_state`，并复用同一个参考音频和基础音色，从而尽量保持语义、语气和声音人设连续。

API 一般不会把上一轮隐藏的 `think` 自动暴露给下一轮，也不应依赖隐藏思考做状态传递。系统提示词参考了 DeepSeek V4 角色沉浸指令的思路：如果模型有私有推理或内部角色规划过程，会要求它以角色第一人称规划情绪和表达方式。但程序明确禁止输出 chain-of-thought、`<think>` 标签或内心独白；需要跨轮保存的信息必须写进 `dialogue_state`，最终只接受 `emit_voice_chat_turn` 工具调用参数。

```bash
uv run voiceagent chat \
  --llm-backend api \
  --chat-model deepseek-chat \
  --tts-model models/VoxCPM2 \
  --tts-device cpu \
  --voice "young male voice, clear and calm timbre, slightly slow pace" \
  --output-dir outputs/chat \
  --play
```

兼容服务替换示例：

```bash
uv run voiceagent chat \
  --llm-backend api \
  --chat-api-base https://your-compatible-endpoint/v1 \
  --chat-api-key "$YOUR_API_KEY" \
  --chat-model your-chat-model \
  --tts-model models/VoxCPM2 \
  --tts-device cpu
```

使用参考音频固定说话人：

```bash
uv run voiceagent chat \
  --llm-backend api \
  --tts-model models/VoxCPM2 \
  --tts-device cpu \
  --reference-audio data/raw/my_voice/reference.wav \
  --voice "preserve the reference speaker identity, natural and friendly delivery" \
  --output-dir outputs/chat
```

无模型链路测试：

```bash
uv run voiceagent chat \
  --llm-backend echo \
  --tts-model models/VoxCPM2 \
  --tts-device cpu \
  --voice "young female voice, gentle and natural delivery" \
  --output-dir outputs/chat
```

输出示例：

```json
{
  "assistant_text": "可以，我们先从一个短句测试开始。",
  "voice_prompt": "natural and confident tone, medium pace, slightly rising ending",
  "dialogue_state": "I am guiding the user through a first short test with a calm, confident attitude.",
  "audio_path": "outputs/chat/turn_001.wav",
  "audio_duration_sec": 3.6
}
```

## 4. 数据预处理

准备 `metadata.csv`：

```csv
audio,text
sample001.wav,你好，这是第一条训练语音。
sample002.wav,这是第二条训练语音。
```

生成 VoxCPM 训练 manifest：

```bash
uv run voiceagent preprocess \
  --metadata data/raw/metadata.csv \
  --audio-root data/raw/wavs \
  --processed-dir data/processed/wavs_16k \
  --output data/manifests/train.jsonl
```

manifest 格式：

```json
{"audio": "data/processed/wavs_16k/sample001_16000.wav", "text": "你好，这是第一条训练语音。", "duration": 3.2}
```

## 5. LoRA 微调

生成配置：

```bash
uv run voiceagent lora-config \
  --pretrained-path models/VoxCPM2 \
  --train-manifest data/manifests/train.jsonl \
  --output configs/voxcpm_lora.yaml
```

运行上游训练脚本：

```bash
uv run python VoxCPM/scripts/train_voxcpm_finetune.py \
  --config configs/voxcpm_lora.yaml
```

微调完成后推理：

```bash
uv run voiceagent synth \
  --tts-model models/VoxCPM2 \
  --lora-path checkpoints/finetune_lora/lora_weights.ckpt \
  --text "这是加载 LoRA 后的语音。"
```

## 6. 代码结构

- `src/voiceagent/llm.py`: OpenAI-compatible Chat Completions 结构化对话封装，默认用 tool/function-call JSON Schema 输出朗读文本、语气提示词和连续状态；默认推荐 DeepSeek，另保留本地 Hugging Face/echo 后端用于备用测试。
- `src/voiceagent/tts.py`: VoxCPM 合成、音色描述、参考音频克隆封装。
- `src/voiceagent/pipeline.py`: 文本对话到语音输出的端到端流水线。
- `src/voiceagent/preprocess.py`: CSV/JSONL 到 VoxCPM JSONL manifest 的预处理。
- `src/voiceagent/cli.py`: 命令行入口。
- `paper/`: LaTeX 论文源码和科研图提示词。

## 7. 验证

```bash
uv run pytest
uv run ruff check src tests
```

模型端到端验证需要对话 API Key 和本地 VoxCPM 权重。4GB 显存机器上建议先用 CPU 或低资源语音模型验证。

## 8. 数据来源

课程实验可以使用以下两类数据：

- 自录数据：5-10 分钟同一说话人的普通话语音，按句切分并人工转写。
- 开源数据：AISHELL-3、BZNSYP 等中文 TTS/说话人语音数据集。使用时需遵守各数据集许可证。

本仓库不提交原始音频、模型权重和上游 VoxCPM 源码。
