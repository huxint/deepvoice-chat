from voiceagent.cli import build_parser, recommend_model


def test_chat_backend_choices_drop_local():
    parser = build_parser()
    args = parser.parse_args(["chat", "--llm-backend", "echo"])
    assert args.llm_backend == "echo"

    # The local backend was removed; argparse should reject it.
    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--llm-backend", "local"])


def test_bench_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["bench", "--timesteps", "4", "10", "--repeats", "2"])
    assert args.timesteps == [4, 10]
    assert args.repeats == 2
    assert args.func.__name__ == "cmd_bench"


def test_recommend_model_by_vram():
    assert recommend_model(has_cuda=True, vram_gb=8.0)["model"] == "openbmb/VoxCPM2"
    assert recommend_model(has_cuda=True, vram_gb=8.0)["device"] == "cuda"
    assert recommend_model(has_cuda=True, vram_gb=6.0)["model"] == "openbmb/VoxCPM1.5"
    assert recommend_model(has_cuda=True, vram_gb=5.0)["model"] == "openbmb/VoxCPM-0.5B"
    assert recommend_model(has_cuda=True, vram_gb=5.0)["device"] == "cuda"


def test_recommend_model_falls_back_to_cpu_on_low_vram():
    rec = recommend_model(has_cuda=True, vram_gb=4.0)
    assert rec["device"] == "cpu"
    rec_nogpu = recommend_model(has_cuda=False, vram_gb=None)
    assert rec_nogpu["device"] == "cpu"
