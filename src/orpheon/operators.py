from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft

from .signal import Signal


@dataclass(frozen=True)
class SpectrumResult:
    frequencies: NDArray[np.float64]
    times: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScalarResult:
    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


def spectrum(
    signal: Signal,
    n_fft: int = 1024,
    hop_length: int = 512,
    window: str = "hann",
) -> SpectrumResult:
    if n_fft <= 0:
        raise ValueError(f"n_fft must be positive, got {n_fft}.")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}.")

    mono = signal.samples.mean(axis=1)
    noverlap = max(0, n_fft - hop_length)

    freqs, times, zxx = stft(
        mono,
        fs=signal.srate,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(zxx)
    return SpectrumResult(
        frequencies=freqs,
        times=times,
        magnitude=magnitude,
        metadata={
            "operator": "spectrum",
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window": window,
            "input_channels": signal.channels,
            "channel_mix": "mean",
            "provenance": signal.provenance,
        },
    )


def rms(signal: Signal) -> ScalarResult:
    value = float(np.sqrt(np.mean(np.square(signal.samples, dtype=np.float64))))
    return ScalarResult(
        name="rms",
        value=value,
        metadata={
            "operator": "rms",
            "provenance": signal.provenance,
        },
    )
