"""Audio loading / resampling helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_audio(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Decode an audio file. Returns (audio[channels, frames] float32, samplerate)."""
    path = str(path)
    errors = []
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)
    except Exception as e:  # noqa: BLE001
        errors.append(f"soundfile: {e}")
    try:
        from pedalboard.io import AudioFile

        with AudioFile(path) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
        return np.ascontiguousarray(audio, dtype=np.float32), int(sr)
    except Exception as e:  # noqa: BLE001
        errors.append(f"pedalboard: {e}")
    ff = _ffmpeg_exe()
    if ff:
        try:
            return _load_with_ffmpeg(ff, path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"ffmpeg: {e}")
    else:
        errors.append("ffmpeg: not available (pip install imageio-ffmpeg)")
    raise RuntimeError(f"Could not decode {path}:\n  " + "\n  ".join(errors))


def _ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _load_with_ffmpeg(ff: str, path: str) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "decoded.wav")
        subprocess.run(
            [ff, "-v", "error", "-y", "-i", path, "-vn", "-c:a", "pcm_f32le", out],
            check=True,
            capture_output=True,
        )
        data, sr = sf.read(out, dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), int(sr)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample audio[channels, frames]."""
    if sr_in == sr_out:
        return audio
    out = soxr.resample(audio.T, sr_in, sr_out, quality="HQ")
    return np.ascontiguousarray(np.atleast_2d(out.T), dtype=np.float32)


def match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1:
        return np.repeat(audio, channels, axis=0)
    mono = audio.mean(axis=0, keepdims=True)
    return np.repeat(mono, channels, axis=0)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]


def load_mono_resampled_stream(path: str | os.PathLike, sr_out: int, block_seconds: float = 60.0) -> np.ndarray:
    """Read a (possibly very long) file as mono at sr_out without holding the full-rate audio in memory."""
    with sf.SoundFile(str(path)) as f:
        sr_in = f.samplerate
        rs = soxr.ResampleStream(sr_in, sr_out, 1, dtype="float32", quality="HQ")
        parts = []
        block = int(block_seconds * sr_in)
        while True:
            data = f.read(block, dtype="float32", always_2d=True)
            last = data.shape[0] < block
            mono = data.mean(axis=1).astype(np.float32)
            parts.append(rs.resample_chunk(mono, last=last))
            if last:
                break
    return np.concatenate(parts).astype(np.float32)


def load_track_for_render(path: str | os.PathLike, sr: int, channels: int) -> np.ndarray:
    audio, track_sr = load_audio(path)
    audio = resample(audio, track_sr, sr)
    return match_channels(audio, channels)


AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a", ".mp4", ".aac", ".alac", ".wma"}


def is_audio_file(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTENSIONS
