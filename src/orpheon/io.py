from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .signal import Signal

_SUPPORTED_AUDIO_SUFFIXES = {".wav"}


def load_audio(path: str | Path) -> Signal:
    src = Path(path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")
    if src.suffix.lower() not in _SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(
            f"Unsupported audio format '{src.suffix}'. Supported formats: {sorted(_SUPPORTED_AUDIO_SUFFIXES)}"
        )

    samples, srate = sf.read(src, always_2d=True, dtype="float64")
    if samples.size == 0:
        raise ValueError(f"Audio file contains no samples: {src}")

    return Signal.from_samples(
        samples=np.ascontiguousarray(samples, dtype=np.float64),
        srate=int(srate),
        provenance={
            "source_path": str(src.resolve()),
            "loader": "soundfile.read",
            "format": src.suffix.lower().lstrip("."),
        },
    )