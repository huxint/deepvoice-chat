from __future__ import annotations

import time
from pathlib import Path
from statistics import mean

from .tts import VoiceStyle


def run_benchmark(
    synthesizer,
    text: str,
    *,
    timesteps_list: list[int],
    repeats: int = 1,
    style: VoiceStyle | None = None,
    output_dir: str | Path = "outputs/bench",
) -> list[dict[str, float | int]]:
    """Measure VoxCPM synthesis speed across inference-step settings.

    For each value in ``timesteps_list`` the synthesizer is run ``repeats``
    times. We record wall-clock synthesis time and the generated audio
    duration, then report the real-time factor ``RTF = synth_time / duration``
    (lower is better). The synthesizer is loaded once and reused, so the model
    download/load cost is excluded from the per-setting timing.
    """

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not timesteps_list:
        raise ValueError("timesteps_list must not be empty")

    style = style or VoiceStyle()
    out_dir = Path(output_dir)
    results: list[dict[str, float | int]] = []

    for steps in timesteps_list:
        synth_times: list[float] = []
        duration = 0.0
        for run_idx in range(repeats):
            wav_path = out_dir / f"bench_ts{steps}_r{run_idx + 1}.wav"
            start = time.perf_counter()
            meta = synthesizer.synthesize_to_file(
                text,
                wav_path,
                style=style,
                inference_timesteps=steps,
            )
            synth_times.append(time.perf_counter() - start)
            duration = float(meta["duration_sec"])

        avg_synth = mean(synth_times)
        rtf = avg_synth / duration if duration > 0 else float("inf")
        results.append(
            {
                "inference_timesteps": steps,
                "repeats": repeats,
                "audio_duration_sec": round(duration, 3),
                "avg_synth_sec": round(avg_synth, 3),
                "rtf": round(rtf, 3),
            }
        )

    return results


def format_markdown_table(results: list[dict[str, float | int]]) -> str:
    """Render benchmark results as a Markdown table ready to paste into the paper."""

    header = (
        "| inference_timesteps | audio_duration_sec | avg_synth_sec | RTF (lower=better) |\n"
        "| --- | --- | --- | --- |"
    )
    rows = [
        f"| {r['inference_timesteps']} | {r['audio_duration_sec']} | "
        f"{r['avg_synth_sec']} | {r['rtf']} |"
        for r in results
    ]
    return "\n".join([header, *rows])
