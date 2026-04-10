from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Signal:
    """In-memory audio signal with reproducibility metadata."""

    samples: NDArray[np.float64]
    srate: int
    channels: int
    duration: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_samples(
        cls,
        samples: NDArray[np.float64],
        srate: int,
        provenance: dict[str, Any] | None = None,
    ) -> "Signal":
        if samples.ndim != 2:
            raise ValueError(
                f"Signal samples must be a 2D array shaped (frames, channels), got ndim={samples.ndim}."
            )
        if srate <= 0:
            raise ValueError(f"Sample rate must be positive, got {srate}.")

        channels = int(samples.shape[1])
        duration = float(samples.shape[0] / srate)
        return cls(
            samples=samples,
            srate=int(srate),
            channels=channels,
            duration=duration,
            provenance=provenance or {},
        )