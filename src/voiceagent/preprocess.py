from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestItem:
    audio: str
    text: str
    duration: float | None = None
    ref_audio: str | None = None
    dataset_id: int | None = None


def _read_metadata(path: Path) -> Iterable[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:
        return None


def _copy_or_resample(src: Path, dst: Path, target_sample_rate: int) -> None:
    import librosa
    import soundfile as sf

    dst.parent.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(src, sr=target_sample_rate, mono=True)
    peak = max(abs(float(audio.max(initial=0.0))), abs(float(audio.min(initial=0.0))), 1e-8)
    if peak > 1.0:
        audio = audio / peak
    sf.write(dst, audio, target_sample_rate)


def build_manifest(
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    audio_root: str | Path | None = None,
    processed_dir: str | Path | None = None,
    target_sample_rate: int = 16_000,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    copy_without_resample: bool = False,
) -> list[ManifestItem]:
    """Build a VoxCPM-compatible JSONL manifest from CSV or JSONL metadata.

    Required columns are ``audio`` and ``text``. Optional columns are
    ``ref_audio`` and ``dataset_id``. Relative audio paths are resolved against
    ``audio_root`` when provided, otherwise against the metadata file directory.
    """

    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    base_dir = Path(audio_root) if audio_root else metadata_path.parent
    processed = Path(processed_dir) if processed_dir else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items: list[ManifestItem] = []
    for idx, row in enumerate(_read_metadata(metadata_path), start=1):
        text = str(row.get("text", "")).strip()
        raw_audio = str(row.get("audio", "")).strip()
        if not text or not raw_audio:
            continue

        src_audio = Path(raw_audio)
        if not src_audio.is_absolute():
            src_audio = base_dir / src_audio
        if not src_audio.exists():
            raise FileNotFoundError(f"Line {idx}: missing audio file {src_audio}")

        manifest_audio = src_audio
        if processed:
            manifest_audio = processed / f"{src_audio.stem}_{target_sample_rate}.wav"
            if copy_without_resample:
                manifest_audio.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_audio, manifest_audio)
            else:
                _copy_or_resample(src_audio, manifest_audio, target_sample_rate)

        duration = _duration(manifest_audio)
        if duration is not None and not (min_duration <= duration <= max_duration):
            continue

        ref_audio = str(row.get("ref_audio", "")).strip() or None
        dataset_id_raw = str(row.get("dataset_id", "")).strip()
        dataset_id = int(dataset_id_raw) if dataset_id_raw else None
        items.append(
            ManifestItem(
                audio=str(manifest_audio),
                text=text,
                duration=duration,
                ref_audio=ref_audio,
                dataset_id=dataset_id,
            )
        )

    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            obj: dict[str, str | float | int] = {"audio": item.audio, "text": item.text}
            if item.duration is not None:
                obj["duration"] = round(item.duration, 3)
            if item.ref_audio:
                obj["ref_audio"] = item.ref_audio
            if item.dataset_id is not None:
                obj["dataset_id"] = item.dataset_id
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return items
