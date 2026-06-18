from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceStyle:
    description: str = ""
    reference_audio: str | None = None
    prompt_audio: str | None = None
    prompt_text: str | None = None


def apply_voice_control(text: str, control: str | None) -> str:
    control = (control or "").strip()
    clean_text = " ".join(text.strip().split())
    if not control:
        return clean_text
    if clean_text.startswith("(") or clean_text.startswith("（"):
        return clean_text
    return f"({control}){clean_text}"


class VoxCPMSynthesizer:
    """Lazy VoxCPM wrapper for design, cloning, and controllable synthesis."""

    def __init__(
        self,
        model_path_or_id: str,
        *,
        device: str = "auto",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        no_denoiser: bool = True,
        no_optimize: bool = True,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        lora_path: str | None = None,
    ):
        self.model_path_or_id = model_path_or_id
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.no_denoiser = no_denoiser
        self.no_optimize = no_optimize
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.lora_path = lora_path
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        from voxcpm import VoxCPM

        path = Path(self.model_path_or_id)
        if path.exists():
            self._model = VoxCPM(
                voxcpm_model_path=str(path),
                enable_denoiser=not self.no_denoiser,
                optimize=not self.no_optimize,
                device=self.device,
                lora_weights_path=self.lora_path,
            )
        else:
            self._model = VoxCPM.from_pretrained(
                self.model_path_or_id,
                load_denoiser=not self.no_denoiser,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                optimize=not self.no_optimize,
                device=self.device,
                lora_weights_path=self.lora_path,
            )

    @property
    def sample_rate(self) -> int:
        if self._model is None:
            raise RuntimeError("Model is not loaded yet")
        return int(self._model.tts_model.sample_rate)

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        *,
        style: VoiceStyle | None = None,
    ) -> dict[str, float | int | str]:
        self.load()
        assert self._model is not None

        import soundfile as sf

        style = style or VoiceStyle()
        target_text = apply_voice_control(text, style.description)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        wav = self._model.generate(
            text=target_text,
            prompt_wav_path=style.prompt_audio,
            prompt_text=style.prompt_text,
            reference_wav_path=style.reference_audio,
            cfg_value=self.cfg_value,
            inference_timesteps=self.inference_timesteps,
        )
        sf.write(output, wav, self.sample_rate)
        return {
            "path": str(output),
            "sample_rate": self.sample_rate,
            "duration_sec": float(len(wav) / self.sample_rate),
        }
