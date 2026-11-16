"""Tone matching: make rendered track audio sound like the recording (EQ, master colouring, level).

A per-band level ratio between the recording and the rendered original is only meaningful in bands
where both actually have energy (e.g. not the bass during a breakdown), so untrustworthy bands are
filled in from their neighbours, the curve is smoothed and clamped, and a flat curve becomes a plain
scalar gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import to_mono

N_BANDS = 16
F_LO, F_HI = 40.0, 16000.0
CENTERS = np.geomspace(F_LO, F_HI, N_BANDS)
_EDGES = np.sqrt(CENTERS[:-1] * CENTERS[1:])
EDGES = np.concatenate([[F_LO / 1.5], _EDGES, [F_HI * 1.3]])


@dataclass
class Tone:
    gain_db: np.ndarray  # (N_BANDS,) recording minus rendered, per band
    flat: bool  # True: just a scalar gain (median of gain_db)
    trusted: np.ndarray  # which bands were measured (the rest are filled in)

    @property
    def scalar_db(self) -> float:
        return float(np.median(self.gain_db))

    def describe(self) -> str:
        if self.flat:
            return f"gain {self.scalar_db:+.1f} dB"
        lo = self.gain_db[:5].mean() - self.scalar_db
        hi = self.gain_db[-5:].mean() - self.scalar_db
        return f"gain {self.scalar_db:+.1f} dB, low {lo:+.1f} / high {hi:+.1f} dB relative"


def _band_power(x: np.ndarray, sr: int, nfft: int = 4096) -> np.ndarray:
    m = to_mono(x).astype(np.float64)
    if m.size < nfft:
        m = np.pad(m, (0, nfft - m.size))
    win = np.hanning(nfft)
    hop = nfft // 2
    frames = np.lib.stride_tricks.sliding_window_view(m, nfft)[::hop]
    psd = (np.abs(np.fft.rfft(frames * win, axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    return np.array([psd[(f >= EDGES[i]) & (f < EDGES[i + 1])].sum() + 1e-20 for i in range(N_BANDS)])


def estimate_tone(rec: np.ndarray, rendered: np.ndarray, sr: int, dyn_range_db: float = 35.0,
                  clamp_db: float = 9.0, flat_tol_db: float = 0.75) -> Tone:
    """rec and rendered: time-aligned (channels, n) excerpts of the same music."""
    pr = 10 * np.log10(_band_power(rec, sr))
    py = 10 * np.log10(_band_power(rendered, sr))
    trusted = (py >= py.max() - dyn_range_db) & (pr >= pr.max() - dyn_range_db)
    d = pr - py
    if trusted.sum() < 3:
        return Tone(np.full(N_BANDS, float(np.median(d))), True, trusted)
    idx = np.arange(N_BANDS)
    d = np.interp(idx, idx[trusted], d[trusted])  # fill untrusted bands from neighbours
    d = np.convolve(np.pad(d, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # smooth
    med = float(np.median(d[trusted]))
    d = np.clip(d, med - clamp_db, med + clamp_db)
    flat = float(np.abs(d - med).max()) < flat_tol_db
    return Tone(np.full(N_BANDS, med) if flat else d, flat, trusted)


def apply_tone_ramp(y: np.ndarray, sr: int, t0: Tone, t1: Tone, block: int = 4096) -> np.ndarray:
    """Apply a tone curve that moves linearly from t0 (start of y) to t1 (end of y)."""
    n = y.shape[1]
    if n == 0:
        return y
    if t0.flat and t1.flat:
        g = 10 ** (np.linspace(t0.scalar_db, t1.scalar_db, n, dtype=np.float64) / 20)
        return (y * g).astype(np.float32)
    hop = block // 2
    win = np.hanning(block + 1)[:-1]  # periodic: 50% overlap sums to 1
    pad = block
    yp = np.pad(y.astype(np.float64), ((0, 0), (pad, pad + block)))
    out = np.zeros_like(yp)
    f = np.fft.rfftfreq(block, 1.0 / sr)
    logf = np.log(np.clip(f, F_LO / 2, None))
    logc = np.log(CENTERS)
    curve0 = np.interp(logf, logc, t0.gain_db)
    curve1 = np.interp(logf, logc, t1.gain_db)
    for s in range(0, yp.shape[1] - block + 1, hop):
        frac = np.clip((s + block / 2 - pad) / max(n, 1), 0.0, 1.0)
        g = 10 ** (((1 - frac) * curve0 + frac * curve1) / 20)
        spec = np.fft.rfft(yp[:, s : s + block] * win, axis=1)
        out[:, s : s + block] += np.fft.irfft(spec * g, block, axis=1)
    return out[:, pad : pad + n].astype(np.float32)
