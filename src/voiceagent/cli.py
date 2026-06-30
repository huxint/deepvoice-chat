from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .bench import format_markdown_table, run_benchmark
from .llm import EchoChat, OpenAICompatibleChat
from .pipeline import VoiceChatAgent
from .preprocess import build_manifest
from .tts import VoiceStyle, VoxCPMSynthesizer


DEFAULT_CHAT_API_MODEL = "deepseek-chat"
DEFAULT_TTS_MODEL = "openbmb/VoxCPM2"


def _add_tts_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL, help="VoxCPM model id or local path")
    parser.add_argument("--tts-device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=6)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--voice", default="", help="Fixed base voice/timbre control for local VoxCPM")
    parser.add_argument("--reference-audio", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--prompt-text", default=None)


def _build_tts(args: argparse.Namespace) -> VoxCPMSynthesizer:
    return VoxCPMSynthesizer(
        args.tts_model,
        device=args.tts_device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        lora_path=args.lora_path,
    )


def _build_style(args: argparse.Namespace) -> VoiceStyle:
    return VoiceStyle(
        description=args.voice,
        reference_audio=args.reference_audio,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
    )


def cmd_synth(args: argparse.Namespace) -> None:
    synth = _build_tts(args)
    meta = synth.synthesize_to_file(args.text, args.output, style=_build_style(args))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_chat(args: argparse.Namespace) -> None:
    if args.llm_backend == "echo":
        llm = EchoChat()
    else:
        llm = OpenAICompatibleChat(
            model=args.chat_model,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    agent = VoiceChatAgent(
        chat_backend=llm,
        synthesizer=_build_tts(args),
        style=_build_style(args),
        output_dir=Path(args.output_dir),
        play=args.play,
    )

    print("输入内容后回车开始语音回复；输入 /exit 退出。")
    while True:
        user_text = input("你> ").strip()
        if user_text in {"/exit", "/quit"}:
            break
        if not user_text:
            continue
        result = agent.run_turn(user_text)
        print(f"助手> {result.assistant_text}")
        if result.voice_prompt:
            print(f"语气> {result.voice_prompt}")
        if result.dialogue_state:
            print(f"状态> {result.dialogue_state}")
        print(f"音频> {result.audio_path} ({result.audio_duration_sec:.2f}s)")


def cmd_preprocess(args: argparse.Namespace) -> None:
    items = build_manifest(
        args.metadata,
        args.output,
        audio_root=args.audio_root,
        processed_dir=args.processed_dir,
        target_sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        copy_without_resample=args.copy_without_resample,
    )
    print(f"Wrote {len(items)} samples to {args.output}")


def cmd_bench(args: argparse.Namespace) -> None:
    synth = _build_tts(args)
    results = run_benchmark(
        synth,
        args.text,
        timesteps_list=args.timesteps,
        repeats=args.repeats,
        style=_build_style(args),
        output_dir=args.bench_audio_dir,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_markdown_table(results))
    print(f"\nWrote {output}")


def cmd_lora_config(args: argparse.Namespace) -> None:
    template = Path("VoxCPM/conf/voxcpm_v2/voxcpm_finetune_lora.yaml")
    if template.exists():
        text = template.read_text(encoding="utf-8")
    else:
        text = (
            "pretrained_path: /path/to/VoxCPM2/\n"
            "train_manifest: /path/to/train.jsonl\n"
            "val_manifest: null\n"
            "batch_size: 1\n"
            "grad_accum_steps: 8\n"
            "num_iters: 500\n"
            "save_path: checkpoints/finetune_lora\n"
            "tensorboard: logs/finetune_lora\n"
        )

    replacements = {
        "/path/to/VoxCPM2/": args.pretrained_path,
        "/path/to/train.jsonl": args.train_manifest,
        "/path/to/checkpoints/finetune_lora": args.save_path,
        "/path/to/logs/finetune_lora": args.tensorboard,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")


def _system_ram_gb() -> float | None:
    """Best-effort total system RAM in GB. Returns None if it cannot be read."""

    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        text = meminfo.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return round(float(line.split()[1]) / 1024**2, 2)
    return None


def recommend_model(*, has_cuda: bool, vram_gb: float | None) -> dict[str, str]:
    """Recommend a VoxCPM model and device from detected hardware.

    The VRAM thresholds follow the upstream VoxCPM2 README requirement table:
    VoxCPM2 ~8GB, VoxCPM1.5 ~6GB, VoxCPM-0.5B ~5GB. A GPU below ~5GB cannot
    hold any tier comfortably, so we fall back to CPU inference.
    """

    if has_cuda and vram_gb is not None:
        if vram_gb >= 8:
            return {"model": "openbmb/VoxCPM2", "device": "cuda", "note": "质量优先，显存充足"}
        if vram_gb >= 6:
            return {"model": "openbmb/VoxCPM1.5", "device": "cuda", "note": "显存中等"}
        if vram_gb >= 5:
            return {
                "model": "openbmb/VoxCPM-0.5B",
                "device": "cuda",
                "note": "显存偏紧，建议关闭 denoiser/优化",
            }
    return {
        "model": "openbmb/VoxCPM-0.5B",
        "device": "cpu",
        "note": "显存不足以舒适运行 GPU 推理；CPU 速度优先用 0.5B，质量优先可用 VoxCPM2（很慢）",
    }


def cmd_doctor(args: argparse.Namespace) -> None:
    del args
    import torch

    has_cuda = torch.cuda.is_available()
    vram_gb: float | None = None
    checks: dict[str, object] = {
        "local_voxcpm_clone": Path("VoxCPM").exists(),
        "ffplay": shutil.which("ffplay") is not None,
        "cuda_available": has_cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "system_ram_gb": _system_ram_gb(),
    }
    if has_cuda:
        checks["gpu_name"] = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        checks["gpu_memory_gb"] = vram_gb

    checks["recommendation"] = recommend_model(has_cuda=has_cuda, vram_gb=vram_gb)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voiceagent")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="Generate one TTS wav")
    synth.add_argument("--text", required=True)
    synth.add_argument("--output", default="outputs/synth.wav")
    _add_tts_args(synth)
    synth.set_defaults(func=cmd_synth)

    chat = sub.add_parser("chat", help="Interactive chat-completions API + local VoxCPM chat")
    chat.add_argument("--llm-backend", choices=["api", "echo"], default="api")
    chat.add_argument("--chat-model", "--deepseek-model", dest="chat_model", default=DEFAULT_CHAT_API_MODEL)
    chat.add_argument("--max-new-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.add_argument("--output-dir", default="outputs/chat")
    chat.add_argument("--play", action="store_true")
    _add_tts_args(chat)
    chat.set_defaults(func=cmd_chat)

    bench = sub.add_parser("bench", help="Measure VoxCPM synthesis speed (RTF) across inference steps")
    bench.add_argument("--text", default="你好，这是本地语音合成的基准测试句子。")
    bench.add_argument("--timesteps", type=int, nargs="+", default=[4, 6, 10, 20])
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--output", default="outputs/bench.json")
    bench.add_argument("--bench-audio-dir", default="outputs/bench")
    _add_tts_args(bench)
    bench.set_defaults(func=cmd_bench)

    prep = sub.add_parser("preprocess", help="Build VoxCPM JSONL manifest")
    prep.add_argument("--metadata", required=True, help="CSV/JSONL with audio,text columns")
    prep.add_argument("--output", required=True)
    prep.add_argument("--audio-root", default=None)
    prep.add_argument("--processed-dir", default=None)
    prep.add_argument("--sample-rate", type=int, default=16_000)
    prep.add_argument("--min-duration", type=float, default=0.3)
    prep.add_argument("--max-duration", type=float, default=30.0)
    prep.add_argument("--copy-without-resample", action="store_true")
    prep.set_defaults(func=cmd_preprocess)

    lora = sub.add_parser("lora-config", help="Create a VoxCPM LoRA config from local paths")
    lora.add_argument("--pretrained-path", required=True)
    lora.add_argument("--train-manifest", required=True)
    lora.add_argument("--save-path", default="checkpoints/finetune_lora")
    lora.add_argument("--tensorboard", default="logs/finetune_lora")
    lora.add_argument("--output", default="configs/voxcpm_lora.yaml")
    lora.set_defaults(func=cmd_lora_config)

    doctor = sub.add_parser("doctor", help="Inspect runtime prerequisites")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
