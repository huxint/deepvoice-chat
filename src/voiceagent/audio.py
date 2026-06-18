from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def play_audio(path: str | Path) -> bool:
    """Play an audio file with ffplay when it is available.

    Returns True when a player was launched successfully. The function is
    intentionally small because playback is platform-dependent and should not
    be part of the model pipeline correctness.
    """

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    ffplay = shutil.which("ffplay")
    if not ffplay:
        return False

    subprocess.run(
        [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", str(audio_path)],
        check=False,
    )
    return True
