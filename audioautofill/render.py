"""Render the fill audio for each jump and stream the rebuilt set to disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_io import resample, to_mono
from .features import ANALYSIS_SR, HOP, logmel, ncc, parabolic_peak

HOP_SECONDS = HOP / ANALYSIS_SR
from .matching import Jump


@dataclass
class Fill:
    jump: Jump
    track_name: str
    so: int = 0  # recording sample where we splice out to the fill
    si: int = 0  # recording sample where we splice back in
    o1: int = 0  # rendered index = recording index + o1, before the jump
    o2: int = 0  # rendered index = recording index + o2, after the jump
    g1: float = 1.0
    g2: float = 1.0
    debug: dict = field(default_factory=dict)
    rho1: float = 0.0  # waveform correlation at splice-out
    rho2: float = 0.0
    q1: float = 0.0  # spectral match quality at splice-out
    q2: float = 0.0
    rendered: np.ndarray | None = None
    accepted: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def inserted_samples(self) -> int:
        return (self.si + self.o2) - (self.so + self.o1) - (self.si - self.so)


def _read(rec: sf.SoundFile, start: int, n: int) -> np.ndarray:
    start = max(0, start)
    rec.seek(start)
    return rec.read(n, dtype="float32", always_2d=True).T


def _align_wave(x: np.ndarray, y: np.ndarray, k_guess: int, search: int) -> tuple[int, float]:
    """Find k such that y[:, k:k+len(x)] best matches x. Returns (k, normalized correlation)."""
    xm = to_mono(x).astype(np.float64)
    n = xm.size
    lo = max(0, k_guess - search)
    hi = min(y.shape[1], k_guess + search + n)
    ym = to_mono(y[:, lo:hi]).astype(np.float64)
    if ym.size < n:
        return k_guess, 0.0
    xm = xm - xm.mean()
    nfft = 1 << int(ym.size + n).bit_length()
    num = np.fft.irfft(np.conj(np.fft.rfft(xm, nfft)) * np.fft.rfft(ym, nfft), nfft)[: ym.size - n + 1]
    c1 = np.cumsum(np.concatenate([[0.0], ym]))
    c2 = np.cumsum(np.concatenate([[0.0], ym * ym]))
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    den = np.sqrt(np.maximum(s2 - s1 * s1 / n, 1e-12)) * np.sqrt((xm * xm).sum() + 1e-12)
    c = num / den
    k = int(np.argmax(c))
    return lo + k, float(c[k])


ENV_RATE = 2000  # envelope samples per second (0.5 ms resolution)
_BANDS = [(20, 150), (150, 1000), (1000, 5000), (5000, 16000)]


def _envelopes(x: np.ndarray, sr: int) -> np.ndarray:
    """Log amplitude envelopes in a few frequency bands, at ENV_RATE. Phase-insensitive, so it
    aligns audio that went through different time stretchers. Returns (bands, n)."""
    m = to_mono(x).astype(np.float64)
    n = m.size
    spec = np.fft.rfft(m)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    step = sr / ENV_RATE
    out_n = int(n / step)
    idx = (np.arange(out_n) * step).astype(np.int64)
    win = max(1, int(0.003 * sr))
    kernel = np.ones(win) / win
    envs = []
    for lo, hi in _BANDS:
        band = np.fft.irfft(np.where((freqs >= lo) & (freqs < hi), spec, 0), n)
        env = np.sqrt(np.convolve(band * band, kernel, mode="same"))
        envs.append(np.log(env[idx] + 1e-4))
    return np.asarray(envs, dtype=np.float32)


def _align_env(x: np.ndarray, y: np.ndarray, k_guess: int, search: int, sr: int) -> tuple[float, float]:
    """Like _align_wave but on band envelopes; returns (k (fractional samples), score)."""
    n = x.shape[1]
    lo = max(0, k_guess - search)
    hi = min(y.shape[1], k_guess + search + n)
    if hi - lo < n:
        return float(k_guess), 0.0
    ex = _envelopes(x, sr)
    ey = _envelopes(y[:, lo:hi], sr)
    c = ncc(ex, ey)
    if c.size < 3:
        return float(k_guess), 0.0
    k = int(np.argmax(c))
    if k == 0 or k == c.size - 1:  # no real peak inside the search range
        return float(k_guess), 0.0
    kf = parabolic_peak(c, k)
    return lo + kf * sr / ENV_RATE, float(c[k])


def _align_spec(x: np.ndarray, y: np.ndarray, k_guess: int, search: int, sr: int) -> int:
    """Align chunk x within y near k_guess using log-mel features (phase-insensitive)."""
    n = x.shape[1]
    lo = max(0, k_guess - search)
    hi = min(y.shape[1], k_guess + search + n)
    if hi - lo < n:
        return k_guess
    fx = logmel(resample(to_mono(x)[None], sr, ANALYSIS_SR)[0])
    fy = logmel(resample(to_mono(y[:, lo:hi])[None], sr, ANALYSIS_SR)[0])
    c = ncc(fx, fy)
    if c.size < 3:
        return k_guess
    k = int(np.argmax(c))
    if k == 0 or k == c.size - 1:
        return k_guess
    return int(round(lo + parabolic_peak(c, k) * HOP_SECONDS * sr))


def _spectral_quality(x: np.ndarray, y: np.ndarray, sr: int) -> float:
    fx = logmel(resample(to_mono(x)[None], sr, ANALYSIS_SR)[0])
    fy = logmel(resample(to_mono(y)[None], sr, ANALYSIS_SR)[0])
    n = min(fx.shape[1], fy.shape[1])
    c = ncc(fx[:, :n], fy[:, :n])
    return float(c[0]) if c.size else 0.0


def _gain(x: np.ndarray, y: np.ndarray, rho: float) -> float:
    yy = float((y.astype(np.float64) ** 2).sum())
    if yy < 1e-12:
        return 1.0
    if rho > 0.9:
        return float((x.astype(np.float64) * y).sum() / yy)
    return float(np.sqrt((x.astype(np.float64) ** 2).sum() / yy))


def plan_fill(
    jump: Jump,
    track_name: str,
    track_audio: np.ndarray,
    rec: sf.SoundFile,
    margin_s: float = 0.25,
    chunk_s: float = 3.0,
    pad_s: float = 2.0,
    min_quality: float = 0.6,
) -> Fill:
    import pedalboard

    sr = rec.samplerate
    f = Fill(jump=jump, track_name=track_name, notes=list(jump.notes))
    r1, r2 = jump.ratio, jump.r_post
    so_t = jump.time - margin_s
    si_t = jump.time + margin_s
    f.so = int(round(so_t * sr))
    f.si = int(round(si_t * sr))

    p_out = jump.a_pre + r1 * so_t
    p_in = jump.a_post + r2 * si_t
    seg_start = p_out - pad_s - chunk_s
    i0 = int(np.floor(seg_start * sr))
    i1 = int(np.ceil((p_in + pad_s + chunk_s) * sr))
    seg = np.zeros((track_audio.shape[0], i1 - i0), dtype=np.float32)
    s0, s1 = max(0, i0), min(track_audio.shape[1], i1)
    if s1 <= s0:
        f.reason = "jump positions fall outside the track file"
        return f
    seg[:, s0 - i0 : s1 - i0] = track_audio[:, s0:s1]
    seg_start = i0 / sr

    # Playback rate per input sample: constant, or a linear ramp between the two sides' tempos
    # across the skipped section when they differ.
    pos = seg_start + np.arange(seg.shape[1]) / sr
    if abs(r2 - r1) > 1e-6:
        rate = r1 + (r2 - r1) * np.clip((pos - p_out) / max(p_in - p_out, 1e-6), 0.0, 1.0)
    else:
        rate = np.full(seg.shape[1], r1)
    # rendered sample index reached at each input sample
    out_idx = np.concatenate([[0.0], np.cumsum(1.0 / rate)])[:-1]

    if abs(r1 - 1.0) > 1e-4 or abs(r2 - 1.0) > 1e-4:
        # high_quality=False selects Rubber Band's R2 engine. The R3 engine (high_quality=True)
        # drifts by up to ~75 ms over 40 s inside the output (only the total length is exact),
        # which would make fills the wrong length; R2 keeps timing to ~1 ms.
        sf_arg = r1 if abs(r2 - r1) <= 1e-6 else rate.astype(np.float64)
        rendered = pedalboard.time_stretch(seg, sr, stretch_factor=sf_arg, high_quality=False)
        rendered = np.ascontiguousarray(rendered, dtype=np.float32)
    else:
        rendered = seg
    f.rendered = rendered

    def guess(a: float, r: float, n: int) -> int:
        """rendered index - recording index, for recording sample n on line p = a + r t."""
        k = (a + r * n / sr - seg_start) * sr
        return int(round(np.interp(k, np.arange(out_idx.size), out_idx) - n))

    o1_guess = guess(jump.a_pre, r1, f.so)
    o2_guess = guess(jump.a_post, r2, f.si)
    n_chunk = int(chunk_s * sr)
    search = int(0.015 * sr)  # around the chunk's own spectral estimate

    def align(x, k_guess):
        """Spectral alignment of this exact chunk (~3 ms), then band envelopes (sub-ms) near it,
        then snap to the waveform peak if the waveforms agree."""
        k_guess = _align_spec(x, rendered, k_guess, int(0.12 * sr), sr)
        ke, se = _align_env(x, rendered, k_guess, search, sr)
        if se < 0.5:
            return k_guess, 0.0, se
        kw, rho = _align_wave(x, rendered, int(round(ke)), int(0.002 * sr))
        if rho >= 0.8:
            return kw, rho, se
        return int(round(ke)), rho, se

    x1 = _read(rec, f.so - n_chunk, n_chunk)
    k1, f.rho1, e1 = align(x1, f.so - n_chunk + o1_guess)
    x2 = _read(rec, f.si, n_chunk)
    k2, f.rho2, e2 = align(x2, f.si + o2_guess)
    f.o1 = k1 - (f.so - n_chunk)
    f.o2 = k2 - f.si
    f.debug = {"o1_minus_guess_ms": round((f.o1 - o1_guess) / sr * 1000, 2),
               "o2_minus_guess_ms": round((f.o2 - o2_guess) / sr * 1000, 2),
               "envelope_scores": [round(e1, 3), round(e2, 3)]}
    if min(e1, e2) < 0.5:
        f.notes.append(f"weak envelope match at a splice ({min(e1, e2):.2f}); used spectral alignment")

    y1 = rendered[:, f.so - n_chunk + f.o1 : f.so + f.o1]
    y2 = rendered[:, f.si + f.o2 : f.si + n_chunk + f.o2]
    if y1.shape[1] != x1.shape[1] or y2.shape[1] != x2.shape[1]:
        f.reason = "fill render too short (alignment ran off the edge)"
        return f
    f.g1 = _gain(x1, y1, f.rho1)
    f.g2 = _gain(x2, y2, f.rho2)
    f.q1 = _spectral_quality(x1, y1, sr)
    f.q2 = _spectral_quality(x2, y2, sr)

    expected = o2_guess - o1_guess
    if abs(f.inserted_samples - expected) > 0.1 * sr:
        f.reason = f"fill length disagrees with jump size ({f.inserted_samples / sr:.2f}s vs {expected / sr:.2f}s)"
        return f
    if min(f.q1, f.q2) < min_quality:
        f.reason = f"recording doesn't match the original track closely enough at the splice (quality {min(f.q1, f.q2):.2f})"
        return f
    if f.g1 <= 0 or f.g2 <= 0 or max(f.g1, f.g2) / min(f.g1, f.g2) > 2.0:
        f.reason = f"inconsistent volume before/after jump (gain {f.g1:.2f} vs {f.g2:.2f})"
        return f
    f.accepted = True
    return f


def _xfade(a: np.ndarray, b: np.ndarray, correlated: bool) -> np.ndarray:
    n = a.shape[1]
    x = (np.arange(n, dtype=np.float32) + 0.5) / n
    if correlated:
        wa, wb = 1.0 - x, x
    else:
        wa, wb = np.cos(x * np.pi / 2), np.sin(x * np.pi / 2)
    return a * wa + b * wb


def write_output(
    rec_path: Path,
    out_path: Path,
    fills: list[Fill],
    xfade_s: float = 0.03,
    block_s: float = 30.0,
) -> dict:
    """Stream the recording to out_path, inserting accepted fills. Returns stats incl. label positions."""
    fills = sorted([f for f in fills if f.accepted], key=lambda f: f.so)
    stats = {"clipped_samples": 0, "labels": []}
    with sf.SoundFile(str(rec_path)) as rec:
        sr, ch = rec.samplerate, rec.channels
        ext = out_path.suffix.lower().lstrip(".")
        same_container = {"wav": "WAV", "wave": "WAV", "aif": "AIFF", "aiff": "AIFF", "flac": "FLAC"}.get(ext) == rec.format
        kw = {"subtype": rec.subtype} if same_container else {}
        h = int(xfade_s * sr / 2)
        block = int(block_s * sr)
        written = 0

        with sf.SoundFile(str(out_path), "w", samplerate=sr, channels=ch, **kw) as out:

            def emit(x: np.ndarray):
                nonlocal written
                over = int((np.abs(x) > 1.0).sum())
                if over:
                    stats["clipped_samples"] += over
                    x = np.clip(x, -1.0, 1.0)
                out.write(np.ascontiguousarray(x.T))
                written += x.shape[1]

            def copy(a: int, b: int):
                pos = a
                while pos < b:
                    n = min(block, b - pos)
                    x = _read(rec, pos, n)
                    if x.shape[1] == 0:
                        break
                    emit(x)
                    pos += x.shape[1]

            cursor = 0
            for f in fills:
                if f.so - h < cursor:
                    f.accepted = False
                    f.reason = "overlaps the previous fill"
                    continue
                R = f.rendered
                copy(cursor, f.so - h)
                fill_start_out = written
                corr = min(f.rho1, f.rho2) > 0.8
                # splice out
                a = _read(rec, f.so - h, 2 * h)
                b = R[:, f.so - h + f.o1 : f.so + h + f.o1] * f.g1
                emit(_xfade(a, b, corr))
                # body with gain ramp g1 -> g2
                k0, k1 = f.so + h + f.o1, f.si - h + f.o2
                body = R[:, k0:k1]
                if body.shape[1] > 0:
                    g = np.linspace(f.g1, f.g2, body.shape[1], dtype=np.float32)
                    emit(body * g)
                # splice in
                a = R[:, f.si - h + f.o2 : f.si + h + f.o2] * f.g2
                b = _read(rec, f.si - h, 2 * h)
                emit(_xfade(a, b, corr))
                cursor = f.si + h
                stats["labels"].append((fill_start_out / sr, written / sr, f))
            copy(cursor, rec.frames)
        stats["output_seconds"] = written / sr
        stats["input_seconds"] = rec.frames / sr
    return stats
