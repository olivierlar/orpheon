from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from orpheon.api import op
from orpheon.io import load_audio


@pytest.fixture()
def sine_wav(tmp_path):
    path = tmp_path / "fixture.wav"
    srate = 8000
    t = np.arange(0, 0.25, 1.0 / srate, dtype=np.float64)
    samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float64)
    sf.write(path, samples, srate)
    return path


def test_loader_smoke(sine_wav):
    signal = load_audio(sine_wav)
    assert signal.srate == 8000
    assert signal.channels == 1
    assert signal.samples.ndim == 2
    assert signal.provenance["format"] == "wav"


def test_spectrum_shape_and_metadata(sine_wav):
    result = op.audio(sine_wav).spectrum(n_fft=256, hop_length=128).compute()
    assert result.magnitude.shape[0] == 129
    assert result.magnitude.shape[1] == result.times.shape[0]
    assert result.frequencies.shape[0] == 129
    assert result.metadata["operator"] == "spectrum"
    assert result.metadata["n_fft"] == 256


def test_rms_deterministic(sine_wav):
    first = op.audio(sine_wav).rms().compute().value
    second = op.audio(sine_wav).rms().compute().value
    assert first == pytest.approx(second, abs=1e-15)
