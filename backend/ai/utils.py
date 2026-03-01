"""
Shared audio utilities for Modal AI inference components.

Handles:
  - WebM (Opus/Vorbis container from browser MediaRecorder) → PCM WAV conversion via ffmpeg
  - Waveform loading and resampling with torchaudio
  - Normalisation helpers

Both speech_to_text and audio_anomaly depend on this module.
"""

import io
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F_audio


# ---------------------------------------------------------------------------
# WebM → WAV conversion
# ---------------------------------------------------------------------------

def webm_to_wav_bytes(webm_bytes: bytes) -> bytes:
    """
    Convert raw WebM audio bytes (as produced by the browser MediaRecorder API
    using Opus or Vorbis codec) to PCM WAV bytes using ffmpeg.

    Args:
        webm_bytes: Raw bytes of a .webm audio file.

    Returns:
        PCM WAV bytes (16-bit, mono, original sample rate preserved).

    Raises:
        RuntimeError: If ffmpeg is not available or conversion fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(webm_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".webm", ".wav")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",                  # overwrite output
                "-i", tmp_in_path,     # input
                "-ac", "1",            # mono
                "-f", "wav",           # output format
                tmp_out_path,
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg conversion failed: {exc.stderr.decode(errors='replace')}"
        ) from exc
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)

    try:
        with open(tmp_out_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        Path(tmp_out_path).unlink(missing_ok=True)

    return wav_bytes


# ---------------------------------------------------------------------------
# Waveform loading
# ---------------------------------------------------------------------------

def load_waveform(
    audio_bytes: bytes,
    target_sr: int,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load audio bytes (WAV, FLAC, MP3 — anything torchaudio supports) into a
    float32 waveform tensor, optionally resampled and converted to mono.

    Args:
        audio_bytes: Raw bytes of a supported audio file (WAV recommended).
        target_sr:   Desired sample rate in Hz.
        mono:        If True, average channels to produce a 1-channel tensor.

    Returns:
        Tensor of shape (1, n_samples) on CPU, dtype float32, sample rate = target_sr.

    Raises:
        ValueError: If torchaudio cannot decode the bytes.
    """
    try:
        buf = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buf)  # (C, T)
    except Exception as exc:
        raise ValueError(f"Failed to decode audio: {exc}") from exc

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

    # Resample if needed
    if sr != target_sr:
        waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=target_sr)

    return waveform  # (1, T)


# ---------------------------------------------------------------------------
# Clip helpers
# ---------------------------------------------------------------------------

def pad_or_crop(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    """
    Deterministically pad (zero, at the end) or crop (from the start) a
    waveform to exactly *target_samples* samples.

    Args:
        waveform:       Tensor of shape (1, T).
        target_samples: Desired number of samples.

    Returns:
        Tensor of shape (1, target_samples).
    """
    length = waveform.shape[-1]
    if length < target_samples:
        pad = target_samples - length
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif length > target_samples:
        waveform = waveform[:, :target_samples]
    return waveform


def normalise_waveform(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Peak-normalise waveform to [-1, 1]."""
    peak = waveform.abs().max().clamp(min=eps)
    return waveform / peak
