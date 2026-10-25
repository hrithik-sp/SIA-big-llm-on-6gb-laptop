"""Has the speaker finished their turn? Smart Turn v3.2 (pipecat-ai, BSD-2-Clause) listens to how a sentence
ends (intonation, a trailing "and...", "um") instead of only waiting for silence.

A Whisper-Tiny encoder with a classifier head, 8M parameters, int8 ONNX (8.7 MB), CPU only. Input: the last
8 s of the turn as Whisper log-mel features (80 bins, 10 ms hop), computed here with numpy exactly as
transformers' WhisperFeatureExtractor(chunk_length=8) does (checked against it), so transformers isn't needed.
Output: the probability that the turn is complete.
"""
import os
from pathlib import Path

import numpy as np

LLM = Path(os.environ.get("LLM_HOME") or Path.home() / "llm")
MODEL = LLM / "models" / "voice" / "smart-turn-v3" / "smart-turn-v3.2-cpu.onnx"
RATE = 16000
SECONDS = 8
N_FFT, HOP, MELS = 400, 160, 80
_S = {"sess": None, "mel": None}


def available():
    return MODEL.exists()


def _hz_to_mel(f):  # Slaney's scale (linear below 1 kHz, logarithmic above), as librosa and transformers use
    f = np.asarray(f, dtype=np.float64)
    mel = 3.0 * f / 200.0
    log = f >= 1000.0
    return np.where(log, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    log = m >= 15.0
    return np.where(log, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)


def _mel_filters():
    if _S["mel"] is None:
        bins = np.linspace(0, RATE / 2, N_FFT // 2 + 1)
        pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(RATE / 2), MELS + 2))
        diff = np.diff(pts)
        slopes = pts[:, None] - bins[None, :]
        down = -slopes[:-2] / diff[:-1, None]
        up = slopes[2:] / diff[1:, None]
        fb = np.maximum(0, np.minimum(down, up))
        fb *= (2.0 / (pts[2:MELS + 2] - pts[:MELS]))[:, None]  # Slaney area normalisation
        _S["mel"] = fb  # (80, 201)
    return _S["mel"]


def features(audio):
    """float32 16 kHz mono -> (1, 80, 800) log-mel, the last 8 s, normalised and right-padded like the reference."""
    x = np.asarray(audio, dtype=np.float32)[-SECONDS * RATE:]
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # do_normalize, over the real samples only
    x = np.pad(x, (0, SECONDS * RATE - len(x))).astype(np.float64)  # padding value 0, on the right
    x = np.pad(x, N_FFT // 2, mode="reflect")  # centred frames
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2  # (801, 201)
    log = np.log10(np.maximum(_mel_filters() @ power.T, 1e-10))[:, :-1]  # (80, 800)
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)[None]


def complete_probability(audio):
    if _S["sess"] is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _S["sess"] = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])
    return float(_S["sess"].run(None, {"input_features": features(audio)})[0].reshape(-1)[0])
