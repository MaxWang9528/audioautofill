"""Spectral features and normalized cross-correlation used for alignment."""

from __future__ import annotations

import numpy as np

ANALYSIS_SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 48
FINE_FPS = ANALYSIS_SR / HOP  # ~86.13 frames/s, fine frame j is centred at j / FINE_FPS seconds

COARSE_T = 8  # fine frames pooled per coarse frame
COARSE_B = 2  # mel bands pooled per coarse band
COARSE_FPS = FINE_FPS / COARSE_T  # ~10.77 frames/s
# coarse frame j is centred at j / COARSE_FPS + COARSE_T0 seconds
COARSE_T0 = (COARSE_T - 1) / 2 / FINE_FPS


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 30.0, fmax: float = 10000.0) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = mel_pts[i], mel_pts[i + 1], mel_pts[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
        s = fb[i].sum()
        if s > 0:
            fb[i] /= s
    return fb


_FB = _mel_filterbank(ANALYSIS_SR, N_FFT, N_MELS)
_WIN = np.hanning(N_FFT).astype(np.float32)


def logmel(y: np.ndarray, chunk_frames: int = 16384) -> np.ndarray:
    """Log-mel spectrogram (dB) of mono audio at ANALYSIS_SR. Returns (N_MELS, n_frames) float32."""
    y = np.asarray(y, dtype=np.float32)
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad))
    n_frames = 1 + max(0, (yp.size - N_FFT)) // HOP
    out = np.empty((N_MELS, n_frames), dtype=np.float32)
    frames_view = np.lib.stride_tricks.sliding_window_view(yp, N_FFT)[::HOP]
    for s in range(0, n_frames, chunk_frames):
        fr = frames_view[s : s + chunk_frames] * _WIN
        spec = np.fft.rfft(fr, axis=1)
        power = (spec.real**2 + spec.imag**2).astype(np.float32)
        out[:, s : s + fr.shape[0]] = _FB @ power.T
    out = 10.0 * np.log10(out + 1e-10)
    floor = out.max() - 80.0
    np.maximum(out, floor, out=out)
    return out


def coarse_from_fine(fine: np.ndarray) -> np.ndarray:
    b, n = fine.shape
    nc = n // COARSE_T
    x = fine[:, : nc * COARSE_T].reshape(b, nc, COARSE_T).mean(axis=2)
    x = x[: (b // COARSE_B) * COARSE_B].reshape(b // COARSE_B, COARSE_B, nc).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample_time(feat: np.ndarray, start: float, step: float, count: int) -> np.ndarray:
    """Sample feat[:, x] at fractional frame indices x = start + step * arange(count) (linear interp).

    Indices outside the feature range are clamped to the edges.
    """
    n = feat.shape[1]
    x = start + step * np.arange(count, dtype=np.float64)
    x = np.clip(x, 0.0, n - 1.000001)
    i0 = np.floor(x).astype(np.int64)
    frac = (x - i0).astype(np.float32)
    return feat[:, i0] * (1.0 - frac) + feat[:, i0 + 1] * frac


def _next_pow2(n: int) -> int:
    return 1 << max(1, int(n - 1).bit_length())


def sliding_denominator(F: np.ndarray, L: int) -> np.ndarray:
    """For each offset k: sqrt(sum_b var-like energy of F[b, k:k+L]) with per-band mean removed."""
    Fd = F.astype(np.float64)
    c1 = np.cumsum(np.pad(Fd, ((0, 0), (1, 0))), axis=1)
    c2 = np.cumsum(np.pad(Fd * Fd, ((0, 0), (1, 0))), axis=1)
    s1 = c1[:, L:] - c1[:, :-L]
    s2 = c2[:, L:] - c2[:, :-L]
    return np.sqrt(np.maximum((s2 - s1 * s1 / L).sum(axis=0), 1e-9))


def ncc(W: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of patch W (B, L) at every valid offset in F (B, M).

    Each band is mean-removed over the patch, so per-band gain/EQ offsets don't matter.
    Returns array of length M - L + 1 with values in [-1, 1].
    """
    B, L = W.shape
    M = F.shape[1]
    if M < L:
        return np.zeros(0)
    W0 = W - W.mean(axis=1, keepdims=True)
    wn = np.sqrt((W0.astype(np.float64) ** 2).sum())
    if wn < 1e-9:
        return np.zeros(M - L + 1)
    n = _next_pow2(M)
    num = np.fft.irfft((np.conj(np.fft.rfft(W0, n, axis=1)) * np.fft.rfft(F, n, axis=1)).sum(axis=0), n)[: M - L + 1]
    den = sliding_denominator(F, L)
    # Where F is (nearly) constant - e.g. clamped past the end of a track - there's nothing to
    # correlate with; report no match instead of a huge ratio.
    ok = den > 1e-3 * max(float(np.median(den)), 1e-6)
    return np.clip(np.where(ok, num / (den * wn), 0.0), -1.0, 1.0)


def parabolic_peak(c: np.ndarray, k: int) -> float:
    if 0 < k < len(c) - 1:
        a, b, d = c[k - 1], c[k], c[k + 1]
        den = a - 2 * b + d
        if abs(den) > 1e-12:
            return k + 0.5 * (a - d) / den
    return float(k)


def framewise_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-frame correlation across bands between two (B, N) feature matrices."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = (Xc * Yc).sum(axis=0)
    den = np.sqrt((Xc * Xc).sum(axis=0) * (Yc * Yc).sum(axis=0)) + 1e-9
    return num / den
