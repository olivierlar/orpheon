from .api import op
from .operators import ScalarResult, SpectrumResult
from .signal import Signal

audio = op.audio

__all__ = ["op", "Signal", "SpectrumResult", "ScalarResult", "audio"]
