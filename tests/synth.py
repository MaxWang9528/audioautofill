"""Synthetic music + DJ-set simulator for testing audioautofill.

The simulated "DJ software" uses its own WSOLA time stretcher (not Rubber Band), so the
app's fills never come from the same algorithm as the recording - like rekordbox's key lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 44100
SCALE = np.array([0, 2, 3, 5, 7, 8, 10])  # natural minor


def _midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m) - 69) / 12)


def _env(n, attack, decay, sr=SR):
    t = np.arange(n) / sr
    e = np.exp(-t / max(decay, 1e-4))
    a = int(attack * sr)
    if a > 0:
        e[:a] *= np.linspace(0, 1, a)
    return e


def _place(buf, start, sig, pan=0.0):
    if start >= buf.shape[1]:
        return
    n = min(sig.size, buf.shape[1] - start)
    lg, rg = np.sqrt(0.5 - pan / 2), np.sqrt(0.5 + pan / 2)
    buf[0, start : start + n] += sig[:n] * lg
    buf[1, start : start + n] += sig[:n] * rg


def make_track(seed: int, bpm: float, bars: int = 96, loopy: bool = False) -> np.ndarray:
    """A techno/house-ish stereo track. Every 8-bar phrase gets its own random patterns,
    unless loopy=True, in which case only two phrase patterns alternate (hard case)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(bars * 4 * beat * SR) + SR
    buf = np.zeros((2, n), dtype=np.float64)
    root = 36 + int(rng.integers(0, 12))
    n_phr = bars // 8
    pattern_seeds = rng.integers(0, 2**31, size=n_phr)
    if loopy:
        pattern_seeds = np.where(np.arange(n_phr) % 4 == 3, pattern_seeds[1], pattern_seeds[0])
    # arrangement: which layers play in each phrase
    layers = []
    for ph in range(n_phr):
        if ph < 2:
            layers.append({"kick", "hat"} | ({"bass"} if ph == 1 else set()))
        elif ph % 6 == 4:
            layers.append({"pad", "lead"})  # breakdown
        else:
            layers.append({"kick", "hat", "bass", "clap"} | ({"pad"} if ph % 2 else {"lead"}) | ({"perc"} if ph % 3 == 0 else set()))

    # each "producer" uses different drum sounds
    k_base, k_sweep, k_rate, k_decay = 40 + 20 * rng.random(), 70 + 90 * rng.random(), 12 + 14 * rng.random(), 0.07 + 0.15 * rng.random()
    hat_decay, hat_hp = 0.008 + 0.03 * rng.random(), int(rng.integers(1, 4))
    clap_decay = 0.02 + 0.06 * rng.random()
    kick_n = int(0.3 * SR)
    tk = np.arange(kick_n) / SR
    kick = np.sin(2 * np.pi * (k_base * tk + (k_sweep / k_rate) * (1 - np.exp(-k_rate * tk)))) * _env(kick_n, 0.001, k_decay)

    for ph in range(n_phr):
        prng = np.random.default_rng(pattern_seeds[ph])
        L = layers[ph]
        bass_pat = prng.choice(SCALE, size=16) * (prng.random(16) < 0.7)
        bass_on = prng.random(16) < 0.6
        lead_pat = prng.choice(SCALE, size=32) + 12 * prng.integers(1, 3, size=32)
        lead_on = prng.random(32) < 0.45
        chord = root + 24 + prng.choice(SCALE, size=3, replace=False)
        hat_amp = 0.05 + 0.1 * prng.random(16)
        perc_on = prng.random(16) < 0.3
        for bar in range(8):
            b0 = (ph * 8 + bar) * 4 * beat
            for st in range(16):
                t = b0 + st * beat / 4
                s = int(t * SR)
                if "kick" in L and st % 4 == 0:
                    _place(buf, s, 0.9 * kick)
                if "hat" in L and st % 4 == 2:
                    m = int(0.05 * SR)
                    h = np.diff(rng.standard_normal(m + hat_hp), n=hat_hp) * _env(m, 0.0005, hat_decay) * hat_amp[st] * 2 / hat_hp
                    _place(buf, s, h, pan=0.3)
                if "clap" in L and st in (4, 12):
                    m = int(0.15 * SR)
                    c = rng.standard_normal(m) * _env(m, 0.002, clap_decay) * 0.25
                    _place(buf, s, c, pan=-0.1)
                if "perc" in L and perc_on[st]:
                    m = int(0.08 * SR)
                    tt = np.arange(m) / SR
                    p = np.sin(2 * np.pi * (300 + 200 * prng.random()) * tt) * _env(m, 0.0005, 0.03) * 0.2
                    _place(buf, s, p, pan=-0.5)
                if "bass" in L and bass_on[st]:
                    m = int(beat / 4 * SR * 0.9)
                    tt = np.arange(m) / SR
                    f0 = _midi_hz(root + bass_pat[st])
                    sig = sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in range(1, 6))
                    _place(buf, s, 0.22 * sig * _env(m, 0.003, 0.12))
                if "lead" in L and st % 2 == 0 and lead_on[(bar % 2) * 16 + st]:
                    i = (bar % 2) * 16 + st
                    m = int(beat / 2 * SR * 0.95)
                    tt = np.arange(m) / SR
                    f0 = _midi_hz(root + 24 + lead_pat[i])
                    sig = np.sin(2 * np.pi * f0 * tt) + 0.3 * np.sin(2 * np.pi * 2 * f0 * tt)
                    _place(buf, s, 0.12 * sig * _env(m, 0.005, 0.15), pan=0.2)
            if "pad" in L:
                m = int(4 * beat * SR)
                tt = np.arange(m) / SR
                sig = sum(np.sin(2 * np.pi * _midi_hz(c) * tt + rng.random() * 6) for c in chord)
                e = np.minimum(1, tt / 0.3) * np.minimum(1, (tt[-1] - tt) / 0.2)
                _place(buf, int(b0 * SR), 0.05 * sig * e, pan=-0.2)
    buf *= 0.5 / max(1e-9, np.abs(buf).max())
    return buf.astype(np.float32)


# --------------------------------------------------------------------------- WSOLA "key lock"


def wsola(x: np.ndarray, pos: np.ndarray, frame: int = 2048, tol: int = 384) -> np.ndarray:
    """Time-stretch x[ch, n] so that output sample i plays source position pos[i] (samples).

    pos can encode any (time-varying) tempo; pitch is preserved."""
    hs = frame // 4
    n_out = pos.size
    ch = x.shape[0]
    win = np.hanning(frame).astype(np.float64)
    out = np.zeros((ch, n_out + frame), dtype=np.float64)
    norm = np.zeros(n_out + frame)
    xm = x.mean(axis=0)
    xpad = np.pad(x, ((0, 0), (frame + tol, frame + tol)))
    xmp = np.pad(xm, (frame + tol, frame + tol))
    off = frame + tol
    prev_src = None
    dec = 4
    for o in range(0, n_out, hs):
        nominal = int(round(pos[o])) - frame // 2
        if prev_src is None:
            src = nominal
        else:
            target = xmp[off + prev_src + hs : off + prev_src + hs + frame : dec]
            lo = nominal - tol
            cand = xmp[off + lo : off + lo + frame + 2 * tol : dec]
            c = np.correlate(cand, target, mode="valid")
            src = lo + int(np.argmax(c)) * dec
        seg = xpad[:, off + src : off + src + frame]
        start = o - frame // 2
        a, b = max(0, start), start + frame
        sa = a - start
        out[:, a:b] += seg[:, sa:] * win[sa:]
        norm[a:b] += win[sa:]
        prev_src = src
    norm = np.maximum(norm, 1e-3)
    return (out[:, :n_out] / norm[:n_out]).astype(np.float32)


# --------------------------------------------------------------------------- set simulation


@dataclass
class DeckPlay:
    track: int
    start: float  # truth-time (s) when the deck starts, at track position 0
    end: float  # truth-time when it stops
    rate: callable  # rate(truth_time_array) -> tempo ratio
    gain: callable  # gain(truth_time_array) -> linear gain
    jumps: list[tuple[float, float]] = field(default_factory=list)  # (truth_time, skipped track seconds)


def render_set(tracks: list[np.ndarray], decks: list[DeckPlay], use_wsola: bool = True):
    """Returns (truth_audio, recording_audio, truth_jumps) where truth_jumps lists
    dicts with recording time, track, from/to positions."""
    total = max(d.end for d in decks)
    n = int(total * SR) + 1
    truth = np.zeros((2, n), dtype=np.float32)
    for d in decks:
        i0, i1 = int(d.start * SR), int(d.end * SR)
        t = np.arange(i0, i1) / SR
        r = d.rate(t)
        pos = np.concatenate([[0.0], np.cumsum(r[:-1])])  # source samples
        src = tracks[d.track]
        pos = np.minimum(pos, src.shape[1] - 1)
        if use_wsola:
            y = wsola(src, pos)
        else:
            y = src[:, np.round(pos).astype(np.int64)]
        truth[:, i0:i1] += y * d.gain(t).astype(np.float32)

    # cut spans out of the truth to make the jumped recording
    cuts = []
    for d in decks:
        for tj, skip in d.jumps:
            r = float(d.rate(np.array([tj]))[0])
            cuts.append((tj, skip / r, d))
    cuts.sort(key=lambda c: c[0])
    pieces, rec_jumps = [], []
    cursor, removed = 0, 0.0
    fade = int(0.004 * SR)
    for tj, dur, d in cuts:
        a, b = int(round(tj * SR)), int(round((tj + dur) * SR))
        piece = truth[:, cursor:a].copy()
        # micro crossfade into the new position, like a DJ app does on a jump
        w = np.linspace(0, 1, fade, dtype=np.float32)
        piece[:, -fade:] = piece[:, -fade:] * (1 - w) + truth[:, b - fade : b] * w
        pieces.append(piece)
        r = float(d.rate(np.array([tj]))[0])
        # track position at the jump (constant tempo assumed around jumps)
        tt = np.arange(int(d.start * SR), a) / SR
        p_before = float(np.sum(d.rate(tt))) / SR
        rec_jumps.append({
            "rec_time": tj - removed,
            "track": d.track,
            "from": p_before,
            "to": p_before + dur * r,
            "inserted": dur,
            "truth_time": tj,
        })
        removed += dur
        cursor = b
    pieces.append(truth[:, cursor:].copy())
    rec = np.concatenate(pieces, axis=1)
    return truth, rec, rec_jumps


def const(v):
    return lambda t: np.full(np.shape(t), v, dtype=np.float64)


def ramp(v0, v1, t0, t1):
    return lambda t: np.interp(t, [t0, t1], [v0, v1])


def envelope(points):
    xs, ys = zip(*points)
    return lambda t: np.interp(t, xs, ys)
