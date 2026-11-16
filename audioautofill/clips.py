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


def display(p: Path) -> str:
    """Clip name for messages: rekordbox gives every recording the same file name in its own folder."""
    return f"{p.parent.name}/{p.name}"


def natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


CLIP_EXTS = (".wav", ".aif", ".aiff", ".flac")


def find_clip_files(sources: list[Path], exclude: set[Path]) -> list[Path]:
    """Clips in set order. A folder of clips is sorted by name; a folder whose clips sit in
    subfolders (rekordbox puts every recording in its own folder, all with similar names) is
    sorted by recording time instead."""
    files: list[Path] = []
    for s in sources:
        if s.is_dir():
            flat = [p for p in s.iterdir() if p.is_file() and p.suffix.lower() in CLIP_EXTS]
            nested = [p for p in s.rglob("*") if p.is_file() and p.suffix.lower() in CLIP_EXTS and p.parent != s]
            if nested:
                files += sorted(flat + nested, key=lambda p: p.stat().st_mtime)
            else:
                files += sorted(flat, key=natural_key)
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
    skipped: dict = field(default_factory=dict)  # edges that couldn't be used (first head / last tail only)

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


def _edge_track_and_chain(which: str, clip: Clip, cm, chains: dict[int, list[Chain]], limit: float | None):
    """The track playing at this edge, and its chain that runs into (head) / out of (tail) the
    transition. Fragments before a beat jump right after pressing record, or after a jump just
    before stopping, are other chains of the same track and are skipped."""
    dur = clip.duration
    L = cm.win_seconds
    edge = 15.0
    best = None
    for li, cs in chains.items():
        if not any((c.start <= 4.0) if which == "head" else (c.end + L >= dur - 4.0) for c in cs):
            continue  # this track isn't playing at the very edge
        sel = [s for c in cs for t, s in zip(c.t, c.s) if (t <= edge if which == "head" else t + L >= dur - edge)]
        if len(sel) >= 2 and (best is None or np.mean(sel) * min(1.0, len(sel) / 6) > best[0]):
            best = (float(np.mean(sel)) * min(1.0, len(sel) / 6), li)
    if best is None:
        raise ClipError(f"{display(clip.path)}: couldn't identify the track playing at the {which} "
                        f"(is it in the track list, and is it playing alone there?)")
    li = best[1]
    cs = chains[li]
    # Of that track's chains on this side of the transition, take the one with the most match
    # evidence: a fragment before/after a beat jump at the very edge is short, and so is a stray
    # match to another copy of a repeating section.
    if which == "head":
        bound = limit if limit is not None else dur * 0.6
        pool = [c for c in cs if c.start < bound]
        key = lambda c: (sum(c.s), c.end)  # noqa: E731
    else:
        bound = limit if limit is not None else dur * 0.4
        pool = [c for c in cs if c.end + L > bound]
        key = lambda c: (sum(c.s), -c.start)  # noqa: E731
    chain = max(pool or cs, key=key)
    return li, chain


def _clean_run(which: str, tr: Track, fine: np.ndarray, a: float, r: float, lo: float, hi: float,
               win: float = 2.0, step: float = 0.25, drop: float = 0.08, level_tol: float = 0.5,
               min_len: float = CHUNK + 1.5):
    """Along the playback line p = a + r t, the stretch of [lo, hi] where the recording is clearly
    this track alone. Returns (start, end) or None. Head: the earliest such stretch (most settled,
    furthest from the transition); tail: the latest."""
    if hi - lo < win:
        return None
    starts = np.arange(lo, hi - win + 1e-9, step)
    sc = np.array([fine_offset(tr.fine, fine, t, t + win, r, a, search=0.03)[1] for t in starts])
    # On real rekordbox recordings a single track alone matches at ~0.96-0.98 with its level steady
    # to +-0.1 dB; a second track, fader/EQ move or beat jump drops the match and/or shifts the level.
    # Reference: the outer edge (clean by how the clips are recorded) - unless that is a fragment
    # before a beat jump, in which case the bulk of the stretch sets it.
    outer4 = sc[:4] if which == "head" else sc[-4:]
    ref = max(float(np.percentile(sc, 90)), float(np.median(outer4)))
    good = sc >= ref - drop
    # Level: alone, the recording is the original times a constant gain. Another track coming in
    # (or a fader/EQ move) changes the level relative to the original even when the match stays
    # decent, so each window's level must stay close to the outer edge's.
    j0 = int(lo * FINE_FPS)
    n = int((hi - lo) * FINE_FPS)
    Y = resample_time(tr.fine, (a + r * j0 / FINE_FPS) * FINE_FPS, r, n)
    X = fine[:, j0 : j0 + n]
    n = min(n, X.shape[1])
    pw = lambda M: 10 * np.log10(np.sum(10 ** (M[:, :n].astype(np.float64) / 10), axis=0) + 1e-12)  # noqa: E731
    d = pw(X) - pw(Y)
    k = int(win * FINE_FPS)
    lvl = np.array([np.median(d[int((t - lo) * FINE_FPS) : int((t - lo) * FINE_FPS) + k]) for t in starts])
    outer = np.nonzero(good)[0]
    if outer.size:
        edge_idx = outer[:4] if which == "head" else outer[-4:]
        ref_lvl = float(np.median(lvl[edge_idx]))
        good &= np.abs(lvl - ref_lvl) <= level_tol
    # close short gaps (up to 1 s) - real music dips briefly (vocals, sparse bars)
    gap = int(1.0 / step)
    i = 0
    while i < len(good):
        if not good[i]:
            j = i
            while j < len(good) and not good[j]:
                j += 1
            if 0 < i and j < len(good) and j - i <= gap:
                good[i:j] = True
            i = j
        else:
            i += 1
    runs, i = [], 0
    while i < len(good):
        if good[i]:
            j = i
            while j + 1 < len(good) and good[j + 1]:
                j += 1
            runs.append((float(starts[i]), float(starts[j] + win)))
            i = j + 1
        else:
            i += 1
    runs = [rn for rn in runs if rn[1] - rn[0] >= min_len]
    if not runs:
        return None
    return min(runs, key=lambda rn: rn[0]) if which == "head" else max(runs, key=lambda rn: rn[1])


def _analyze_end(which: str, clip: Clip, fine: np.ndarray, cm, local: list[Track], chains: dict[int, list[Chain]],
                 flux: np.ndarray, limit: float | None = None) -> tuple[End, Chain]:
    """limit: for the head, when the incoming track becomes audible; for the tail, when the
    outgoing track stops being audible (clip seconds). The edge stays clear of it."""
    dur = clip.duration
    L = cm.win_seconds
    li, chain = _edge_track_and_chain(which, clip, cm, chains, limit)
    tr = local[li]
    notes: list[str] = []
    debug: list[str] = []

    # 1) the playback line of that chain
    r0 = chain._fit(at_end=(which == "tail"), span=1e9)[0]
    line = chain.predict_before if which == "head" else chain.predict_after
    # the whole playback along this line in the clip (it can be split into several chains)
    c_lo = max(EDGE_GUARD, _same_playback_extent(chains[li], chain, earliest=True))
    c_hi = min(dur - EDGE_GUARD, _same_playback_extent(chains[li], chain, earliest=False) + L)
    if which == "head" and limit is not None:
        c_hi = min(c_hi, limit)
    if which == "tail" and limit is not None:
        c_lo = max(c_lo, limit)
    if which == "head":  # the start of a chain can be a mixed window: look a bit before it too
        c_lo = max(EDGE_GUARD, c_lo - L / 2)
    else:
        c_hi = min(dur - EDGE_GUARD, c_hi + L / 2)
    tc = (c_lo + c_hi) / 2
    r, _ = estimate_ratio(tr, fine, [(c_lo, c_hi, line)], r0, span=0.012)
    a, _ = fine_offset(tr.fine, fine, c_lo, c_hi, r, line(tc) - r * tc, search=0.6)

    # 2) the clean stretch along that line
    run = _clean_run(which, tr, fine, a, r, c_lo, c_hi)
    if run is None:
        when = (f" - the next track comes in about {limit:.1f} s into the clip" if which == "head" and limit is not None
                else f" - the previous track is audible until about {limit:.1f} s" if limit is not None else "")
        raise ClipError(f"{display(clip.path)}: couldn't find a clean {which} of at least {CHUNK + 1.5:.1f} s with only "
                        f"'{tr.name}' playing{when}. Leave at least 8 s with only one track playing, unchanged, at each end.")
    lo, hi = run
    if which == "head" and lo > 2.0:
        notes.append(f"ignoring the first {lo:.1f} s of the clip (they don't run into the transition - a beat jump?)")
    if which == "tail" and dur - hi > 2.0:
        notes.append(f"ignoring the last {dur - hi:.1f} s of the clip (they don't follow on from the transition - a beat jump?)")

    # 3) exact tempo and offset over the clean stretch
    precise = lambda t, a=a, r=r: a + r * t  # noqa: E731
    r_new, used = measure_ratio(tr, fine, [(lo, hi, precise)], r, region=3.0, min_span=6.0)
    if not used:
        r_new, _ = estimate_ratio(tr, fine, [(lo, hi, precise)], r, span=0.0015)
        debug.append(f"{which}: tempo measured from a short section ({hi - lo:.1f} s)")
    tm = (lo + hi) / 2
    a, score = fine_offset(tr.fine, fine, lo, hi, r_new, precise(tm) - r_new * tm, search=0.1)
    r = r_new
    if score < 0.65:
        raise ClipError(f"{display(clip.path)}: the {which} doesn't match '{tr.name}' cleanly (score {score:.2f}); "
                        f"leave at least 8 s with only that track playing, unchanged, at the {which}")

    # 4) splice: just before a transient, at the outer end of the clean stretch (keeps the most
    #    real recording), with CHUNK seconds of clean audio on the inner side for alignment
    if which == "head":
        s_lo, s_hi = lo + 0.25, min(lo + 3.0, hi - CHUNK - 0.2)
    else:
        s_lo, s_hi = max(hi - 3.0, lo + CHUNK + 0.2), hi - 0.25
    if s_hi <= s_lo:
        raise ClipError(f"{display(clip.path)}: the clean {which} is too short ({hi - lo:.1f} s); "
                        f"leave at least 8 s with only one track playing at each end")
    splice = _pick_splice(flux, s_lo, s_hi)
    win = (lo, min(hi, splice + HEAD_WINDOW)) if which == "head" else (max(lo, splice - HEAD_WINDOW), hi)

    amb = ambiguous_copy_offset(chain, tr, fine, L, near_jump_end=(which == "tail"), margin=0.02)
    if amb >= 16.0:
        notes.append(f"this part of '{tr.name}' repeats: a spot {amb:.0f} s away sounds the same, so the fill here "
                     f"could be that much too long or short (it will still sound seamless)")
    elif amb > 0:
        debug.append(f"{which}: section repeats ({amb:.1f} s away matches about as well)")
    debug.append(f"{which}: clean stretch {lo:.2f}-{hi:.2f} s")
    return End(track=tr.index, a=a, r=r, window=win, splice=splice, score=score, ambiguous=amb, notes=notes,
               debug=debug), chain


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


def analyze_clip(path: Path, tracks: list[Track], candidates: list[int], ratios: np.ndarray,
                 need_head: bool = True, need_tail: bool = True) -> Clip:
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

    def edge(which, need):
        try:
            return _edge_track_and_chain(which, clip, cm, chains, None)
        except ClipError as e:
            if need:
                raise
            clip.skipped[which] = str(e)
            return None, None

    hl, head_chain = edge("head", need_head)
    tl, tail_chain = edge("tail", need_tail)
    lim_head = lim_tail = None
    if hl is not None and tl is not None and hl != tl:
        # keep each edge clear of the other track: the incoming one is audible from (about) the start
        # of its playback in this clip, the outgoing one until (about) the end of its playback
        lim_head = _same_playback_extent(chains[tl], tail_chain, earliest=True) - 1.0
        lim_tail = _same_playback_extent(chains[hl], head_chain, earliest=False) + L / 2
    head = tail = None
    for which, idx, lim, need in (("head", hl, lim_head, need_head), ("tail", tl, lim_tail, need_tail)):
        if idx is None:
            continue
        try:
            end, _ = _analyze_end(which, clip, fine, cm, local, chains, flux, limit=lim)
        except ClipError as e:
            if need:
                raise
            clip.skipped[which] = str(e)
            continue
        if which == "head":
            head = end
        else:
            tail = end
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
            lines = [f"  {display(c.path)}: starts with '{tracks[c.head.track].name}', ends with '{tracks[c.tail.track].name}'"
                     for c in clips]
            raise ClipError(f"{display(A.path)} ends with '{tracks[A.tail.track].name}' but the next clip, {display(B.path)}, "
                            f"starts with '{tracks[B.head.track].name}'. Is a clip missing or out of order?\n" + "\n".join(lines))
        p_out, p_in = A.tail.pos(A.tail.splice), B.head.pos(B.head.splice)
        amb = max(A.tail.ambiguous, B.head.ambiguous)
        if amb >= 16.0:
            notes.append(f"WARNING: the fill in '{tracks[A.tail.track].name}' between {display(A.path)} and {display(B.path)} "
                         f"starts or ends in a section that repeats; it may be about {amb:.0f} s too long or short "
                         f"(seamless either way, but listen to whether the track's structure is right)")
        if p_in - p_out >= MIN_BRIDGE:
            continue
        # Overlap: the clips both contain the same stretch of the track. Move a splice point so a
        # short fill joins them, if the verified windows allow it.
        t_new = (p_out + MIN_BRIDGE - B.head.a) / B.head.r
        if t_new + CHUNK + 0.2 <= B.head.window[1]:
            B.head.splice = t_new
            notes.append(f"{display(A.path)} and {display(B.path)} overlap in '{tracks[B.head.track].name}'; joined with a short fill")
            continue
        t_new = (p_in - MIN_BRIDGE - A.tail.a) / A.tail.r
        if t_new - CHUNK - 0.2 >= A.tail.window[0]:
            A.tail.splice = t_new
            notes.append(f"{display(A.path)} and {display(B.path)} overlap in '{tracks[A.tail.track].name}'; joined with a short fill")
            continue
        raise ClipError(
            f"{display(B.path)} starts at {p_in:.1f} s into '{tracks[B.head.track].name}', which is before where "
            f"{display(A.path)} ends ({p_out:.1f} s). That's more overlap than the clips' clean edges allow - or one of "
            f"them was matched to the wrong copy of a repeated section"
            + (f" (repeats noted: {A.tail.ambiguous:.1f} s / {B.head.ambiguous:.1f} s)" if A.tail.ambiguous or B.head.ambiguous else "")
            + ".")
    return notes


def assemble(clips: list[Clip], tracks: list[Track], out_path: Path, intro: bool, outro: bool,
             xfade_s: float = 0.03, engine: str = "r2", log=print) -> dict:
    sr, ch = clips[0].sr, clips[0].channels
    for c in clips:
        if c.sr != sr:
            raise ClipError(f"{display(c.path)} is {c.sr} Hz but {display(clips[0].path)} is {sr} Hz; record all clips the same way")
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
            stats["labels"].append((t_clip0, written / sr, f"clip {display(c.path)}"))
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
