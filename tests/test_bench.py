from voiceagent.bench import format_markdown_table, run_benchmark
from voiceagent.tts import VoiceStyle


class FakeSynth:
    """Duck-typed synthesizer that fakes a fixed audio duration."""

    def __init__(self, duration_sec: float = 2.0):
        self.duration_sec = duration_sec
        self.calls = []

    def synthesize_to_file(self, text, output_path, *, style, inference_timesteps):
        self.calls.append((text, str(output_path), inference_timesteps))
        return {"duration_sec": self.duration_sec}


def test_run_benchmark_reports_one_row_per_timesteps(tmp_path):
    synth = FakeSynth(duration_sec=2.0)
    results = run_benchmark(
        synth,
        "测试",
        timesteps_list=[4, 10],
        repeats=2,
        style=VoiceStyle(description="年轻女声"),
        output_dir=tmp_path,
    )

    assert [r["inference_timesteps"] for r in results] == [4, 10]
    assert all(r["repeats"] == 2 for r in results)
    assert all(r["audio_duration_sec"] == 2.0 for r in results)
    # rtf = synth_time / duration; synth_time >= 0 so rtf is finite and non-negative
    assert all(r["rtf"] >= 0 for r in results)
    # repeats=2 across two timesteps settings => 4 synth calls
    assert len(synth.calls) == 4
    assert {c[2] for c in synth.calls} == {4, 10}


def test_run_benchmark_rejects_bad_args(tmp_path):
    synth = FakeSynth()
    import pytest

    with pytest.raises(ValueError):
        run_benchmark(synth, "x", timesteps_list=[], output_dir=tmp_path)
    with pytest.raises(ValueError):
        run_benchmark(synth, "x", timesteps_list=[4], repeats=0, output_dir=tmp_path)


def test_format_markdown_table_has_header_and_rows():
    results = [
        {"inference_timesteps": 4, "audio_duration_sec": 2.0, "avg_synth_sec": 5.0, "rtf": 2.5},
    ]
    table = format_markdown_table(results)
    assert "inference_timesteps" in table
    assert "RTF" in table
    assert "| 4 | 2.0 | 5.0 | 2.5 |" in table
