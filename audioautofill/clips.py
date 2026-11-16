"""Clip mode: build a full set from one recording per transition.

Each clip starts with the outgoing track playing alone (settled) and ends with the incoming track
playing alone. The tool identifies the track + position + tempo at the head and tail of every clip,
then renders the music in between consecutive clips from the original track file and splices it in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_io import is_audio_file, load_mono_resampled_stream, load_track_for_render
from .features import ANALYSIS_SR, FINE_FPS, coarse_from_fine, framewise_similarity, logmel, resample_time
from .matching import (
    Chain,
    Track,
    absent_level,
    ambiguous_copy_offset,
    build_chains,
    coarse_match,
    estimate_ratio,
    fine_offset,
    measure_ratio,
    reanchor,
)
from .render import _align_env, _align_spec, _align_wave, _spectral_quality, _xfade
from .tone import Tone, apply_tone_ramp, estimate_tone

HEAD_WINDOW = 8.0  # seconds at each clip edge that are verified as single-track and used
EDGE_GUARD = 0.5  # don't use the very first/last moments of a file
CHUNK = 3.0  # recording audio used to align each splice
MIN_BRIDGE = 1.0  # shortest fill between clips (track seconds)


class ClipError(Exception):
    pass


def natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def find_clip_files(sources: list[Path], exclude: set[Path]) -> list[Path]:
    files: list[Path] = []
    for s in sources:
        if s.is_dir():
            files += sorted((p for p in s.iterdir() if p.is_file() and p.suffix.lower() in (".wav", ".aif", ".aiff", ".flac")),
                            key=natural_key)
        else:
            files.append(s)
    return [f for f in files if f.resolve() not in exclude]


@dataclass
class End:
    """One end of a clip: the single track playing there and how it maps to the clip's time."""

    track: int  # index into the global track list
    a: float  # track position = a + r * t   (t = seconds into the clip)
    r: float
    window: tuple[float, float]  # verified single-track span (clip seconds)
    splice: float  # where we splice (clip seconds)
    score: float
    ambiguous: float = 0.0  # >0: a copy this many seconds away matches about as well
    notes: list[str] = field(default_factory=list)  # warnings worth reading
    debug: list[str] = field(default_factory=list)  # details, shown with -v

    def pos(self, t: float) -> float:
        return self.a + self.r * t


@dataclass
class Clip:
    path: Path
    sr: int
    channels: int
    frames: int
    subtype: str
    fmt: str
    head: End | None = None
    tail: End | None = None

    @property
    def duration(self) -> float:
        return self.frames / self.sr


# ----------------------------------------------------------------------------- analysis


def _onset_strength(fine: np.ndarray) -> np.ndarray:
    d = np.diff(fine, axis=1, prepend=fine[:, :1])
    return np.maximum(d, 0).sum(axis=0)


def _pick_splice(flux: np.ndarray, lo: float, hi: float) -> float:
    """Just before the strongest transient in [lo, hi] (clip seconds): the hit masks the crossfade."""
    j0, j1 = int(np.ceil(lo * FINE_FPS)), int(np.floor(hi * FINE_FPS))
    if j1 <= j0:
        return (lo + hi) / 2
    j = j0 + int(np.argmax(flux[j0:j1]))
    return max(lo, j / FINE_FPS - 0.02)


def _analyze_end(which: str, clip: Clip, fine: np.ndarray, cm, local: list[Track], chains: dict[int, list[Chain]],
                 flux: np.ndarray, limit: float | None = None) -> tuple[End, Chain]:
    """limit: for the head, when the incoming track becomes audible; for the tail, when the
    outgoing track stops being audible (clip seconds). The edge window stays clear of it."""
    dur = clip.duration
    L = cm.win_seconds
    # 1) which chain sits at this edge
    best = None
    for li, cs in chains.items():
        for c in cs:
            if which == "head":
                if c.start > 3.0:
                    continue
                sel = [s for t, s in zip(c.t, c.s) if t <= HEAD_WINDOW]
            else:
                if c.end + L < dur - 3.0:
                    continue
                sel = [s for t, s in zip(c.t, c.s) if t + L >= dur - HEAD_WINDOW]
            if sel and (best is None or np.mean(sel) > best[0]):
                best = (float(np.mean(sel)), li, c)
    if best is None:
        raise ClipError(f"{clip.path.name}: couldn't identify the track playing at the {which} "
                        f"(is it in the track list, and is it playing alone there?)")
    _, li, chain = best
    tr = local[li]
    notes: list[str] = []
    debug: list[str] = []

    # 2) verified window at the edge, clear of the other track. Besides the other track's own
    #    playback (limit), stop where this track's frame-by-frame match starts dropping - a quiet
    #    incoming track or a fader/EQ move shows up there.
    drop = _match_drop(which, tr, fine, chain, dur, L)
    if drop is not None:
        limit = drop if limit is None else (min(limit, drop) if which == "head" else max(limit, drop))
    if which == "head":
        hi = min(HEAD_WINDOW, dur / 2 - 0.5, limit if limit is not None else np.inf)
        win = (EDGE_GUARD, hi)
        side = (EDGE_GUARD, min(chain.end + L / 2, 16.0, hi if limit is not None else np.inf))
        line = chain.predict_before
    else:
        lo = max(dur - HEAD_WINDOW, dur / 2 + 0.5, limit if limit is not None else -np.inf)
        win = (lo, dur - EDGE_GUARD)
        side = (max(chain.start + L / 2, dur - 16.0, lo if limit is not None else -np.inf), dur - EDGE_GUARD)
        line = chain.predict_after
    if win[1] - win[0] < CHUNK + 1.5:
        when = (f"the next track comes in about {limit:.1f} s into the clip" if which == "head" and limit is not None
                else f"the previous track is still audible until about {limit:.1f} s" if limit is not None else "")
        raise ClipError(f"{clip.path.name}: the clean {which} is too short ({max(0.0, win[1] - win[0]):.1f} s"
                        + (f"; {when}" if when else "") + "). Leave at least 8 s with only one track playing at each end.")

    # 3) tempo near this edge, then the exact offset over the window
    r0 = chain.slope_start() if which == "head" else chain.slope_end()
    tc = (win[0] + win[1]) / 2
    r, _ = estimate_ratio(tr, fine, [(win[0], win[1], line)], r0, span=0.012)
    a, score = fine_offset(tr.fine, fine, win[0], win[1], r, line(tc) - r * tc, search=0.6)
    precise = lambda t, a=a, r=r: a + r * t  # noqa: E731
    r_new, used = measure_ratio(tr, fine, [(side[0], side[1], precise)], r, region=3.0, min_span=6.0)
    if not used:
        r_new, _ = estimate_ratio(tr, fine, [(side[0], side[1], precise)], r, span=0.0015)
        debug.append(f"{which}: tempo measured from a short section ({side[1] - side[0]:.1f} s)")
    # keep the position at the window centre, re-fit the offset for the new tempo
    a, score = fine_offset(tr.fine, fine, win[0], win[1], r_new, precise(tc) - r_new * tc, search=0.1)
    r = r_new

    if score < 0.65:
        small = (win[0], win[0] + 5.0) if which == "head" else (win[1] - 5.0, win[1])
        _, sc = fine_offset(tr.fine, fine, small[0], small[1], r, a, search=0.05)
        if sc >= 0.65:
            win = small
            notes.append("only the outermost 5 s look like the track alone")
        else:
            raise ClipError(f"{clip.path.name}: the {which} doesn't match '{tr.name}' cleanly (score {score:.2f}); "
                            f"leave at least 8 s with only that track playing, unchanged, at the {which}")
    # steadiness inside the window (informational)
    j0, j1 = int(win[0] * FINE_FPS), int(win[1] * FINE_FPS)
    if j1 - j0 > 20:
        Y = resample_time(tr.fine, (a + r * j0 / FINE_FPS) * FINE_FPS, r, j1 - j0)
        fs = np.convolve(framewise_similarity(fine[:, j0:j1], Y), np.ones(43) / 43, mode="valid")
        if fs.size and fs.min() < np.median(fs) - 0.4:
            debug.append(f"{which}: match varies within the edge window (min {fs.min():.2f}, median {np.median(fs):.2f})")

    # 4) splice point: just before a transient, early in the head / late in the tail
    if which == "head":
        lo, hi = 0.75, min(win[1] - CHUNK - 0.2, 3.5)
    else:
        lo, hi = max(win[0] + CHUNK + 0.2, dur - 3.5), dur - 0.75
    if hi <= lo:
        raise ClipError(f"{clip.path.name}: the clean {which} is too short ({win[1] - win[0]:.1f} s); "
                        f"leave at least 8 s with only one track playing at each end")
    splice = _pick_splice(flux, lo, hi)

    amb = ambiguous_copy_offset(chain, tr, fine, L, near_jump_end=(which == "tail"), margin=0.02)
    if amb >= 16.0:
        notes.append(f"this part of '{tr.name}' repeats: a spot {amb:.0f} s away sounds the same, so the fill here "
                     f"could be that much too long or short (it will still sound seamless)")
    elif amb > 0:
        debug.append(f"{which}: section repeats ({amb:.1f} s away matches about as well)")
    return End(track=tr.index, a=a, r=r, window=win, splice=splice, score=score, ambiguous=amb, notes=notes,
               debug=debug), chain


def _match_drop(which: str, tr: Track, fine: np.ndarray, chain: Chain, dur: float, L: float,
                span: float = 16.0, drop: float = 0.1) -> float | None:
    """Clip time where the edge track's match falls clearly (and lastingly) below its level at the
    very edge - a second track coming in, or a fader/EQ move. Sliding 1 s windows, 0.25 s apart."""
    if which == "head":
        t0, t1 = EDGE_GUARD, min(dur - EDGE_GUARD, span)
    else:
        t0, t1 = max(EDGE_GUARD, dur - span), dur - EDGE_GUARD
    if t1 - t0 < 3.0:
        return None
    r = chain.slope_start() if which == "head" else chain.slope_end()
    line = chain.predict_before if which == "head" else chain.predict_after
    ref_lo, ref_hi = (t0, t0 + 2.0) if which == "head" else (t1 - 2.0, t1)
    tcr = (ref_lo + ref_hi) / 2
    a, _ = fine_offset(tr.fine, fine, ref_lo, ref_hi, r, line(tcr) - r * tcr, search=0.6)
    starts = np.arange(t0, t1 - 1.0 + 1e-9, 0.25)
    if which == "tail":
        starts = starts[::-1]
    sc = np.array([fine_offset(tr.fine, fine, s0, s0 + 1.0, r, a, search=0.03)[1] for s0 in starts])
    ref = float(np.median(sc[:4]))
    below = sc < ref - drop
    for i in range(len(sc) - 5):
        if below[i : i + 6].all():  # sustained for 1.5 s
            return float(starts[i]) if which == "head" else float(starts[i] + 1.0)
    return None


def _same_playback_extent(cs: list[Chain], ref: Chain, earliest: bool) -> float:
    """Earliest start (or latest end) among a track's chains that continue `ref`'s playback line."""
    best = ref.start if earliest else ref.end
    for c in cs:
        if c is ref:
            continue
        t = c.t[0] if earliest else c.t[-1]
        p = c.p[0] if earliest else c.p[-1]
        pred = ref.predict_before(t) if earliest else ref.predict_after(t)
        if abs(p - pred) <= 1.0 + 0.01 * abs(t - (ref.start if earliest else ref.end)):
            best = min(best, c.start) if earliest else max(best, c.end)
    return best


def analyze_clip(path: Path, tracks: list[Track], candidates: list[int], ratios: np.ndarray) -> Clip:
    info = sf.info(str(path))
    clip = Clip(path, info.samplerate, info.channels, info.frames, info.subtype, info.format)
    fine = logmel(load_mono_resampled_stream(path, ANALYSIS_SR))
    local = [tracks[i] for i in candidates]
    cm = coarse_match(coarse_from_fine(fine), local, ratios)
    L = cm.win_seconds
    chains = {}
    for li, t in enumerate(local):
        cs = build_chains(cm, li)
        for c in cs:
            reanchor(c, t, fine, L)
        chains[li] = cs
    flux = _onset_strength(fine)
    head, head_chain = _analyze_end("head", clip, fine, cm, local, chains, flux)
    tail, tail_chain = _analyze_end("tail", clip, fine, cm, local, chains, flux)
    hl, tl = candidates.index(head.track), candidates.index(tail.track)
    if hl != tl:
        # keep each edge clear of the other track: the incoming one is audible from (about) the start
        # of its playback in this clip, the outgoing one until (about) the end of its playback
        incoming_from = _same_playback_extent(chains[tl], tail_chain, earliest=True)
        outgoing_until = _same_playback_extent(chains[hl], head_chain, earliest=False) + L / 2
        head, _ = _analyze_end("head", clip, fine, cm, local, chains, flux, limit=incoming_from - 1.0)
        tail, _ = _analyze_end("tail", clip, fine, cm, local, chains, flux, limit=outgoing_until)
    clip.head, clip.tail = head, tail
    return clip


# ----------------------------------------------------------------------------- rendering


@dataclass
class Bridge:
    body: np.ndarray  # rendered audio incl. half a crossfade of overlap at each spliced end
    notes: list[str]
    info: dict


def _read(path: Path, start: int, n: int) -> np.ndarray:
    with sf.SoundFile(str(path)) as f:
        start = max(0, start)
        f.seek(start)
        return f.read(n, dtype="float32", always_2d=True).T


def render_bridge(track_audio: np.ndarray, sr: int, start: tuple[Clip, End] | None, end: tuple[Clip, End] | None,
                  h: int, engine: str = "r2") -> Bridge:
    """Track audio from `start` (a clip's tail splice) to `end` (the next clip's head splice).
    start=None: from the beginning of the track (intro). end=None: to the end of the track (outro)."""
    import pedalboard

    notes: list[str] = []
    n_track = track_audio.shape[1]
    s_end = start[1] if start else None
    e_end = end[1] if end else None
    r1 = s_end.r if s_end else e_end.r
    r2 = e_end.r if e_end else r1
    p_out = s_end.pos(s_end.splice) if s_end else 0.0
    p_in = e_end.pos(e_end.splice) if e_end else n_track / sr
    pad_pre = ((s_end.splice - s_end.window[0]) * r1 + 1.0) if s_end else 0.0
    pad_post = ((e_end.window[1] - e_end.splice) * r2 + 1.0) if e_end else 0.0
    seg_start = p_out - pad_pre
    i0 = int(np.floor(seg_start * sr))
    i1 = int(np.ceil((p_in + pad_post) * sr))
    seg = np.zeros((track_audio.shape[0], i1 - i0), dtype=np.float32)
    a0, a1 = max(0, i0), min(n_track, i1)
    seg[:, a0 - i0 : a1 - i0] = track_audio[:, a0:a1]
    seg_start = i0 / sr
    pos = seg_start + np.arange(seg.shape[1]) / sr
    if abs(r2 - r1) > 1e-6:
        rate = r1 + (r2 - r1) * np.clip((pos - p_out) / max(p_in - p_out, 1e-6), 0.0, 1.0)
        if abs(r2 - r1) > 0.003:
            notes.append(f"tempo differs between the clips (x{r1:.4f} vs x{r2:.4f}); the fill ramps between them")
    else:
        rate = np.full(seg.shape[1], r1)
    out_idx = np.concatenate([[0.0], np.cumsum(1.0 / rate)])[:-1]
    if abs(r1 - 1) > 1e-4 or abs(r2 - 1) > 1e-4:
        sfac = r1 if abs(r2 - r1) <= 1e-6 else rate.astype(np.float64)
        y = np.ascontiguousarray(pedalboard.time_stretch(seg, sr, stretch_factor=sfac, high_quality=(engine == "r3")),
                                 dtype=np.float32)
    else:
        y = seg

    def k_at(p: float) -> float:
        return float(np.interp((p - seg_start) * sr, np.arange(out_idx.size), out_idx))

    n_chunk = int(CHUNK * sr)

    def align(x: np.ndarray, k_guess: int) -> tuple[int, float, float]:
        k = _align_spec(x, y, k_guess, int(0.12 * sr), sr)
        ke, se = _align_env(x, y, k, int(0.015 * sr), sr)
        if se < 0.5:
            return k, 0.0, se
        kw, rho = _align_wave(x, y, int(round(ke)), int(0.002 * sr))
        return (kw if rho >= 0.8 else int(round(ke))), rho, se

    info: dict = {}
    if s_end:
        so = int(round(s_end.splice * sr))
        x = _read(start[0].path, so - n_chunk, n_chunk)
        kg = int(round(k_at(p_out))) - n_chunk
        k, rho1, e1 = align(x, kg)
        k_out = k + n_chunk
        w = int((s_end.splice - s_end.window[0]) * sr)
        xr = _read(start[0].path, so - w, w)
        tone1 = estimate_tone(xr, y[:, max(0, k_out - w) : k_out], sr)
        q1 = _spectral_quality(x, y[:, k : k + n_chunk], sr)
        info.update(out_shift_ms=round((k - kg) / sr * 1000, 2), out_env=round(e1, 3), out_quality=round(q1, 3))
    else:
        k_out, rho1, q1, tone1 = 0, 0.0, 1.0, None
    if e_end:
        si = int(round(e_end.splice * sr))
        x = _read(end[0].path, si, n_chunk)
        kg = int(round(k_at(p_in)))
        k_in, rho2, e2 = align(x, kg)
        w = int((e_end.window[1] - e_end.splice) * sr)
        xr = _read(end[0].path, si, w)
        tone2 = estimate_tone(xr, y[:, k_in : k_in + w], sr)
        q2 = _spectral_quality(x, y[:, k_in : k_in + n_chunk], sr)
        info.update(in_shift_ms=round((k_in - kg) / sr * 1000, 2), in_env=round(e2, 3), in_quality=round(q2, 3))
        expected = k_at(p_in) - k_at(p_out) if s_end else None
        if expected is not None and abs((k_in - k_out) - expected) > 0.1 * sr:
            notes.append(f"alignment disagrees with the expected fill length by {((k_in - k_out) - expected) / sr:+.2f} s")
    else:
        k_in, rho2, q2, tone2 = y.shape[1], 0.0, 1.0, None
    tone1 = tone1 or tone2
    tone2 = tone2 or tone1
    for q, where in ((q1, "start"), (q2, "end")):
        if q < 0.5:
            notes.append(f"recording and original don't match well at the {where} of this fill (quality {q:.2f}) - listen to it")
    rhos = ([rho1] if s_end else []) + ([rho2] if e_end else [])
    info.update(tone_start=tone1.describe(), tone_end=tone2.describe(), seconds=round((k_in - k_out) / sr, 3),
                correlated=bool(rhos) and min(rhos) > 0.8)
    lo = k_out - h if s_end else 0
    hi = k_in + h if e_end else y.shape[1]
    if lo < 0 or hi > y.shape[1]:
        raise ClipError("fill render too short for its alignment (internal error)")
    body = apply_tone_ramp(y[:, lo:hi], sr, tone1, tone2)
    if not e_end:  # natural end of the track: tiny fade so it doesn't click
        n = min(body.shape[1], int(0.02 * sr))
        body[:, -n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return Bridge(body=body, notes=notes, info=info)


def plan_bridges(clips: list[Clip], tracks: list[Track]) -> list[str]:
    """Check continuity between clips and fix up splice points where clips overlap.
    Returns notes; raises ClipError for anything that can't be rendered sensibly."""
    notes = []
    for i in range(len(clips) - 1):
        A, B = clips[i], clips[i + 1]
        if A.tail.track != B.head.track:
            lines = [f"  {c.path.name}: starts with '{tracks[c.head.track].name}', ends with '{tracks[c.tail.track].name}'"
                     for c in clips]
            raise ClipError(f"{A.path.name} ends with '{tracks[A.tail.track].name}' but the next clip, {B.path.name}, "
                            f"starts with '{tracks[B.head.track].name}'. Is a clip missing or out of order?\n" + "\n".join(lines))
        p_out, p_in = A.tail.pos(A.tail.splice), B.head.pos(B.head.splice)
        amb = max(A.tail.ambiguous, B.head.ambiguous)
        if amb >= 16.0:
            notes.append(f"WARNING: the fill in '{tracks[A.tail.track].name}' between {A.path.name} and {B.path.name} "
                         f"starts or ends in a section that repeats; it may be about {amb:.0f} s too long or short "
                         f"(seamless either way, but listen to whether the track's structure is right)")
        if p_in - p_out >= MIN_BRIDGE:
            continue
        # Overlap: the clips both contain the same stretch of the track. Move a splice point so a
        # short fill joins them, if the verified windows allow it.
        t_new = (p_out + MIN_BRIDGE - B.head.a) / B.head.r
        if t_new + CHUNK + 0.2 <= B.head.window[1]:
            B.head.splice = t_new
            notes.append(f"{A.path.name} and {B.path.name} overlap in '{tracks[B.head.track].name}'; joined with a short fill")
            continue
        t_new = (p_in - MIN_BRIDGE - A.tail.a) / A.tail.r
        if t_new - CHUNK - 0.2 >= A.tail.window[0]:
            A.tail.splice = t_new
            notes.append(f"{A.path.name} and {B.path.name} overlap in '{tracks[A.tail.track].name}'; joined with a short fill")
            continue
        raise ClipError(
            f"{B.path.name} starts at {p_in:.1f} s into '{tracks[B.head.track].name}', which is before where "
            f"{A.path.name} ends ({p_out:.1f} s). That's more overlap than the clips' clean edges allow - or one of "
            f"them was matched to the wrong copy of a repeated section"
            + (f" (repeats noted: {A.tail.ambiguous:.1f} s / {B.head.ambiguous:.1f} s)" if A.tail.ambiguous or B.head.ambiguous else "")
            + ".")
    return notes


def assemble(clips: list[Clip], tracks: list[Track], out_path: Path, intro: bool, outro: bool,
             xfade_s: float = 0.03, engine: str = "r2", log=print) -> dict:
    sr, ch = clips[0].sr, clips[0].channels
    for c in clips:
        if c.sr != sr:
            raise ClipError(f"{c.path.name} is {c.sr} Hz but {clips[0].path.name} is {sr} Hz; record all clips the same way")
    h = int(xfade_s * sr / 2)
    ext_fmt = {".wav": "WAV", ".aif": "AIFF", ".aiff": "AIFF", ".flac": "FLAC"}.get(out_path.suffix.lower())
    kw = {"subtype": clips[0].subtype} if ext_fmt == clips[0].fmt else {}
    stats = {"labels": [], "bridges": [], "clipped_samples": 0}
    cache: dict[int, np.ndarray] = {}

    def track_audio(ti: int) -> np.ndarray:
        if ti not in cache:
            cache.clear()
            cache[ti] = load_track_for_render(tracks[ti].path, sr, ch)
        return cache[ti]

    with sf.SoundFile(str(out_path), "w", samplerate=sr, channels=ch, **kw) as out:
        written = 0

        def emit(x: np.ndarray):
            nonlocal written
            over = int((np.abs(x) > 1.0).sum())
            if over:
                stats["clipped_samples"] += over
                x = np.clip(x, -1.0, 1.0)
            out.write(np.ascontiguousarray(x.T))
            written += x.shape[1]

        def emit_clip(c: Clip, a: int, b: int, block: int = 30 * sr):
            pos = a
            while pos < b:
                x = _read(c.path, pos, min(block, b - pos))
                if x.shape[1] == 0:
                    break
                emit(x)
                pos += x.shape[1]

        def do_bridge(start, end, label):
            ti = (start[1] if start else end[1]).track
            br = render_bridge(track_audio(ti), sr, start, end, h, engine)
            t0 = written / sr
            body = br.body
            if start:
                c, e = start
                si = int(round(e.splice * sr))
                emit(_xfade(_read(c.path, si - h, 2 * h), body[:, : 2 * h], br.info["correlated"]))
                body = body[:, 2 * h :]
            if end:
                c, e = end
                so = int(round(e.splice * sr))
                emit(body[:, : -2 * h])
                emit(_xfade(body[:, -2 * h :], _read(c.path, so - h, 2 * h), br.info["correlated"]))
            else:
                emit(body)
            stats["labels"].append((t0, written / sr, label))
            stats["bridges"].append({"label": label, "start": round(t0, 3), "end": round(written / sr, 3), **br.info, "notes": br.notes})
            for n in br.notes:
                log(f"      ! {n}")
            return br

        def fmt(p):
            m, s = divmod(p, 60)
            return f"{int(m)}:{s:05.2f}"

        # intro
        c0 = clips[0]
        if intro:
            t = tracks[c0.head.track]
            label = f"fill {t.name} 0:00.00-{fmt(c0.head.pos(c0.head.splice))}"
            log(f"  intro: {label}")
            do_bridge(None, (c0, c0.head), label)
            cursor = int(round(c0.head.splice * sr)) + h
        else:
            cursor = 0
        for i, c in enumerate(clips):
            t_clip0 = written / sr
            last = i == len(clips) - 1
            if not last or outro:
                si = int(round(c.tail.splice * sr))
                emit_clip(c, cursor, si - h)
            else:
                emit_clip(c, cursor, c.frames)
            stats["labels"].append((t_clip0, written / sr, f"clip {c.path.name}"))
            if not last:
                nxt = clips[i + 1]
                t = tracks[c.tail.track]
                label = f"fill {t.name} {fmt(c.tail.pos(c.tail.splice))}-{fmt(nxt.head.pos(nxt.head.splice))}"
                log(f"  {label}")
                do_bridge((c, c.tail), (nxt, nxt.head), label)
                cursor = int(round(nxt.head.splice * sr)) + h
            elif outro:
                t = tracks[c.tail.track]
                label = f"fill {t.name} {fmt(c.tail.pos(c.tail.splice))}-end"
                log(f"  outro: {label}")
                do_bridge((c, c.tail), None, label)
        stats["output_seconds"] = written / sr
    return stats
