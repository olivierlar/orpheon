from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

from .io import load_audio
from .operators import ScalarResult, SpectrumResult, rms, spectrum

T = TypeVar("T")


@dataclass(frozen=True)
class ComputeNode(Generic[T]):
    _compute: Callable[[], T]

    def compute(self) -> T:
        return self._compute()


@dataclass(frozen=True)
class AudioInput:
    path: Path

    def spectrum(self, n_fft: int = 1024, hop_length: int = 512) -> ComputeNode[SpectrumResult]:
        return ComputeNode(lambda: spectrum(load_audio(self.path), n_fft=n_fft, hop_length=hop_length))

    def rms(self) -> ComputeNode[ScalarResult]:
        return ComputeNode(lambda: rms(load_audio(self.path)))


@dataclass(frozen=True)
class Operators:
    def audio(self, path: str | Path) -> AudioInput:
        return AudioInput(path=Path(path))


op = Operators()
