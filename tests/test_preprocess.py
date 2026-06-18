import csv
import wave
from pathlib import Path

from voiceagent.preprocess import build_manifest


def _write_wav(path: Path, sample_rate: int = 16_000, seconds: float = 0.5) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"\x00\x00" * frames)


def test_build_manifest_from_csv(tmp_path):
    audio = tmp_path / "a.wav"
    _write_wav(audio)
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerow({"audio": "a.wav", "text": "hello"})

    output = tmp_path / "train.jsonl"
    items = build_manifest(metadata, output)

    assert len(items) == 1
    assert output.read_text(encoding="utf-8").count("\n") == 1
