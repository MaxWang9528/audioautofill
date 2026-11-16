"""Locate tracks inside the recording and find beat jumps.

Model: while a deck plays, the recording time t maps to track position p by p = a + r * t,
where r is the tempo ratio (played BPM / original BPM). A beat jump keeps r but makes `a`
step forward. We find those steps per track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .features import (
    COARSE_FPS,
    COARSE_T0,
    FINE_FPS,
    _next_pow2,
    framewise_similarity,
    ncc,
    parabolic_peak,
    resample_time,
    sliding_denominator,
)


@dataclass
class Track:
    index: int
    path: Path
    fine: np.ndarray  # (B, N) log-mel at FINE_FPS, original speed
    coarse: np.ndarray  # (Bc, Nc) at COARSE_FPS
    duration: float

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass
class CoarseMatches:
    """Top candidate alignments of every recording window against every track.

    Music repeats (bars, phrases), so one window often matches several track positions about
    equally well; we keep several candidates and let a path search pick the consistent ones.
    """

    times: np.ndarray  # (W,) recording time of each window start (s)
    win_seconds: float
    score: np.ndarray  # (W, T, C) similarity of candidate (-1 = none)
    ratio: np.ndarray  # (W, T, C) tempo ratio of candidate
    pos: np.ndarray  # (W, T, C) track position (s) aligned with window start


# ----------------------------------------------------------------------------- coarse

N_CAND = 8
PEAKS_PER_RATIO = 6
NMS_SECONDS = 0.35


def _local_peaks(c: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest local maxima of every row of c (w, m). Shape (w, n)."""
    m = c.shape[1]
    masked = np.full_like(c, -np.inf)
    if m >= 3:
        inner = (c[:, 1:-1] > c[:, :-2]) & (c[:, 1:-1] >= c[:, 2:])
        masked[:, 1:-1] = np.where(inner, c[:, 1:-1], -np.inf)
    masked[:, 0] = c[:, 0]
    masked[:, -1] = c[:, -1]
    n = min(n, m)
    idx = np.argpartition(-masked, n - 1, axis=1)[:, :n]
    return idx


def _nms(scores: np.ndarray, pos: np.ndarray, ratio: np.ndarray, keep: int):
    """Per window, drop candidates within NMS_SECONDS of a better one. Inputs (w, n)."""
    w = scores.shape[0]
    out_s = np.full((w, keep), -1.0, dtype=np.float32)
    out_p = np.zeros((w, keep))
    out_r = np.ones((w, keep), dtype=np.float32)
    order = np.argsort(-scores, axis=1)
    for i in range(w):
        kept = 0
        for j in order[i]:
            s = scores[i, j]
            if not np.isfinite(s) or s <= -1:
                break
            if kept and np.any(np.abs(out_p[i, :kept] - pos[i, j]) < NMS_SECONDS):
                continue
            out_s[i, kept], out_p[i, kept], out_r[i, kept] = s, pos[i, j], ratio[i, j]
            kept += 1
            if kept == keep:
                break
    return out_s, out_p, out_r


def coarse_match(
    rec_coarse: np.ndarray,
    tracks: list[Track],
    ratios: np.ndarray,
    win_seconds: float = 6.0,
    hop_seconds: float = 1.5,
    progress: Callable[[float], None] | None = None,
) -> CoarseMatches:
    B, Nrec = rec_coarse.shape
    L = int(round(win_seconds * COARSE_FPS))
    H = max(1, int(round(hop_seconds * COARSE_FPS)))
    starts = np.arange(0, max(1, Nrec - L + 1), H)
    nW, nT = len(starts), len(tracks)

    # Bank of every (track, ratio): track features re-timed to recording-time frames.
    entries = []
    max_m = L
    for ti, t in enumerate(tracks):
        n = t.coarse.shape[1]
        for r in ratios:
            m = int(np.floor((n - 1) / r)) + 1
            if m >= L:
                entries.append((ti, float(r), m))
                max_m = max(max_m, m)
    nfft = _next_pow2(max_m)
    nf = nfft // 2 + 1
    K = len(entries)
    bank = np.empty((nf, B, K), dtype=np.complex64)
    dens = []
    for k, (ti, r, m) in enumerate(entries):
        Fr = resample_time(tracks[ti].coarse, 0.0, r, m)
        bank[:, :, k] = np.fft.rfft(Fr, nfft, axis=1).T
        dens.append(sliding_denominator(Fr, L).astype(np.float32))

    score = np.full((nW, nT, N_CAND), -1.0, dtype=np.float32)
    ratio = np.ones((nW, nT, N_CAND), dtype=np.float32)
    pos = np.zeros((nW, nT, N_CAND), dtype=np.float64)

    chunk = max(1, int(1.5e8 / (max(K, 1) * nfft * 8)))
    for c0 in range(0, nW, chunk):
        idx = starts[c0 : c0 + chunk]
        W = np.stack([rec_coarse[:, s : s + L] for s in idx])  # (w, B, L)
        W0 = W - W.mean(axis=2, keepdims=True)
        wn = np.sqrt((W0.astype(np.float64) ** 2).sum(axis=(1, 2))) + 1e-9
        Wf = np.conj(np.fft.rfft(W0, nfft, axis=2)).astype(np.complex64)  # (w, B, nf)
        prod = np.matmul(Wf.transpose(2, 0, 1), bank)  # (nf, w, K)
        corr = np.fft.irfft(prod.transpose(1, 2, 0), nfft, axis=2)  # (w, K, nfft)
        rows = np.arange(len(idx))[:, None]
        per_track: dict[int, list] = {}
        for k, (ti, r, m) in enumerate(entries):
            valid = m - L + 1
            c = corr[:, k, :valid] / (dens[k][None, :] * wn[:, None])
            pk = _local_peaks(c, PEAKS_PER_RATIO)
            per_track.setdefault(ti, []).append((c[rows, pk], r * pk / COARSE_FPS + COARSE_T0, np.full(pk.shape, r)))
        sl = slice(c0, c0 + len(idx))
        for ti, parts in per_track.items():
            s_all = np.concatenate([p[0] for p in parts], axis=1)
            p_all = np.concatenate([p[1] for p in parts], axis=1)
            r_all = np.concatenate([p[2] for p in parts], axis=1)
            score[sl, ti], pos[sl, ti], ratio[sl, ti] = _nms(s_all, p_all, r_all, N_CAND)
        if progress:
            progress(min(1.0, (c0 + len(idx)) / nW))

    times = starts / COARSE_FPS + COARSE_T0
    return CoarseMatches(times=times, win_seconds=L / COARSE_FPS, score=score, ratio=ratio, pos=pos)


# ----------------------------------------------------------------------------- chains


@dataclass
class Chain:
    """A run of windows whose track positions follow one continuous line p = a + r t."""

    track: int
    t: list[float] = field(default_factory=list)
    p: list[float] = field(default_factory=list)
    r: list[float] = field(default_factory=list)
    s: list[float] = field(default_factory=list)

    def add(self, t, p, r, s):
        self.t.append(float(t))
        self.p.append(float(p))
        self.r.append(float(r))
        self.s.append(float(s))

    def _fit(self, at_end: bool, span: float = 20.0):
        t = np.asarray(self.t)
        p = np.asarray(self.p)
        if at_end:
            sel = t >= t[-1] - span
        else:
            sel = t <= t[0] + span
        t, p = t[sel], p[sel]
        if len(t) >= 3 and t[-1] - t[0] >= 4.0:
            slope, icpt = np.polyfit(t, p, 1)
            return float(slope), float(icpt)
        rr = float(np.median(np.asarray(self.r)[sel]))
        tr = t[-1] if at_end else t[0]
        pr = p[-1] if at_end else p[0]
        return rr, pr - rr * tr

    def predict_after(self, t: float) -> float:
        slope, icpt = self._fit(at_end=True)
        return icpt + slope * t

    def predict_before(self, t: float) -> float:
        slope, icpt = self._fit(at_end=False)
        return icpt + slope * t

    def slope_end(self) -> float:
        return self._fit(at_end=True)[0]

    def slope_start(self) -> float:
        return self._fit(at_end=False)[0]

    @property
    def start(self) -> float:
        return self.t[0]

    @property
    def end(self) -> float:
        return self.t[-1]


def absent_level(cm: CoarseMatches, track: int) -> float:
    """Similarity a window gets from a track that isn't playing (noise floor), per track."""
    best = cm.score[:, track, 0]
    return float(np.clip(np.percentile(best, 30) + 0.1, 0.35, 0.6))


def build_chains(
    cm: CoarseMatches,
    track: int,
    jump_penalty: float = 1.2,
    enter_penalty: float = 0.6,
    tol: float = 0.25,
    min_windows: int = 2,
) -> list[Chain]:
    """Viterbi path through the per-window candidates of one track.

    States per window: each candidate alignment, plus "absent". Staying on a consistent line
    is free, switching lines (a jump) costs jump_penalty, entering/leaving costs enter_penalty.
    Each maximal consistent run on the best path becomes a Chain.
    """
    S = cm.score[:, track]  # (W, C)
    P = cm.pos[:, track]
    R = cm.ratio[:, track]
    nW, C = S.shape
    if nW == 0:
        return []
    absent = absent_level(cm, track)
    em = np.concatenate([np.where(S > -1, S, -np.inf), np.full((nW, 1), absent)], axis=1)  # (W, C+1)
    V = em[0].copy()
    back = np.zeros((nW, C + 1), dtype=np.int16)
    for w in range(1, nW):
        dt = cm.times[w] - cm.times[w - 1]
        rbar = (R[w - 1][:, None] + R[w][None, :]) / 2  # (C_prev, C_cur)
        d = P[w][None, :] - (P[w - 1][:, None] + rbar * dt)
        trans = np.where(np.abs(d) <= tol, 0.0, -jump_penalty)  # cand -> cand
        full = np.empty((C + 1, C + 1))
        full[:C, :C] = trans
        full[:C, C] = -enter_penalty  # cand -> absent
        full[C, :C] = -enter_penalty  # absent -> cand
        full[C, C] = 0.0
        tot = V[:, None] + full
        back[w] = np.argmax(tot, axis=0)
        V = tot[back[w], np.arange(C + 1)] + em[w]
    path = np.empty(nW, dtype=np.int64)
    path[-1] = int(np.argmax(V))
    for w in range(nW - 1, 0, -1):
        path[w - 1] = back[w, path[w]]

    chains: list[Chain] = []
    cur: Chain | None = None
    for w in range(nW):
        st = path[w]
        if st == C:
            cur = None
            continue
        t, p, r, s = cm.times[w], P[w, st], R[w, st], S[w, st]
        if cur is not None:
            dt = t - cur.end
            rbar = (cur.r[-1] + r) / 2
            if abs(p - (cur.p[-1] + rbar * dt)) > tol:
                cur = None
        if cur is None:
            cur = Chain(track)
            chains.append(cur)
        cur.add(t, p, r, s)
    return [c for c in chains if len(c.t) >= min_windows]


# ----------------------------------------------------------------------------- jumps


@dataclass
class Jump:
    track: int
    kind: str  # "forward" | "backward"
    ratio: float
    a_pre: float  # p = a_pre + ratio * t before the jump
    a_post: float  # p = a_post + ratio * t after the jump
    time: float  # estimated jump time in the recording (s)
    pre_score: float
    post_score: float
    pre_region: tuple[float, float]
    post_region: tuple[float, float]
    notes: list[str] = field(default_factory=list)
    ambiguous: bool = False
    ratio_post: float | None = None  # tempo ratio after the jump, if it differs (else = ratio)

    @property
    def r_post(self) -> float:
        return self.ratio if self.ratio_post is None else self.ratio_post

    @property
    def skipped_track_seconds(self) -> float:
        return self.pos_after - self.pos_before

    @property
    def pos_before(self) -> float:
        return self.a_pre + self.ratio * self.time

    @property
    def pos_after(self) -> float:
        return self.a_post + self.r_post * self.time


def fine_offset(
    track_fine: np.ndarray,
    rec_fine: np.ndarray,
    t0: float,
    t1: float,
    r: float,
    a_guess: float,
    search: float = 0.6,
) -> tuple[float, float]:
    """Best `a` in p = a + r t over recording span [t0, t1], searching a_guess +- search."""
    j0 = max(0, int(round(t0 * FINE_FPS)))
    j1 = min(rec_fine.shape[1], int(round(t1 * FINE_FPS)))
    if j1 - j0 < 16:
        return a_guess, -1.0
    W = rec_fine[:, j0:j1]
    L = W.shape[1]
    extra = int(np.ceil(2 * search * FINE_FPS / r))
    p_start = (a_guess - search) + r * j0 / FINE_FPS  # track pos (s) aligned with W[:, 0] at shift 0
    F = resample_time(track_fine, p_start * FINE_FPS, r, L + extra)
    c = ncc(W, F)
    if c.size == 0:
        return a_guess, -1.0
    k = int(np.argmax(c))
    kf = parabolic_peak(c, k)
    return (a_guess - search) + r * kf / FINE_FPS, float(c[k])


def refine_jump(track: Track, rec_fine: np.ndarray, A: Chain, B: Chain, win_seconds: float) -> Jump:
    t_a, t_b = A.end, B.start
    notes: list[str] = []

    # Clean single-mapping regions: the last A window starts before the jump, and the first
    # B window ends after it.
    pre_end = t_a
    pre_start = max(A.start, pre_end - 5.0)
    post_start = t_b + win_seconds
    post_end = min(B.end + win_seconds, post_start + 5.0)
    if pre_end - pre_start < 2.0:
        pre_start = max(A.start - 1.0, pre_end - 2.0)
        notes.append("short clean section before jump")
    if post_end - post_start < 2.0:
        post_end = post_start + 2.0
        notes.append("short clean section after jump")

    # Tempo ratio: coarse from chain fits, then refined jointly over long regions on both sides
    # (tempo is steady across a jump).
    r0 = float(np.median([A.slope_end(), B.slope_start()]))
    long_pre = (max(A.start, pre_end - 20.0), pre_end, A.predict_after)
    long_post = (post_start, max(post_end, min(B.end + win_seconds, post_start + 20.0)), B.predict_before)
    r, _ = estimate_ratio(track, rec_fine, [long_pre, long_post], r0, span=0.012)  # coarse
    # Tempo must be the tempo *at* the jump: use the 30 s nearest it on each side (further away the
    # DJ may already be ramping into the next track).
    side_pre = (max(A.start + win_seconds / 2, pre_end - 30.0), pre_end, A.predict_after)
    side_post = (post_start, max(post_start + 4.0, min(B.end + win_seconds / 2, post_start + 30.0)), B.predict_before)
    r, n_used = measure_ratio(track, rec_fine, [side_pre, side_post], r)
    r1 = r2 = r  # tempo ratio before / after the jump
    if n_used == 0:
        r, _ = estimate_ratio(track, rec_fine, [long_pre, long_post], r, span=0.0015)
        notes.append("tempo measured from a short section only")
    else:
        r_pre, n1 = measure_ratio(track, rec_fine, [side_pre], r)
        r_post, n2 = measure_ratio(track, rec_fine, [side_post], r)
        if n1 and n2 and abs(r_pre - r_post) > 0.002:
            # e.g. Sync following a tempo change in the track's beatgrid: the rate changed somewhere
            # in the skipped part. Use each side's own rate; the fill ramps between them.
            notes.append(f"tempo differs across jump ({r_pre:.4f} vs {r_post:.4f}); fill ramps between them")
            r1, r2 = r_pre, r_post

    tc_pre = (pre_start + pre_end) / 2
    tc_post = (post_start + post_end) / 2
    a1, s1 = fine_offset(track.fine, rec_fine, pre_start, pre_end, r1, A.predict_after(tc_pre) - r1 * tc_pre)
    a2, s2 = fine_offset(track.fine, rec_fine, post_start, post_end, r2, B.predict_before(tc_post) - r2 * tc_post)

    # Exact jump time: the split point where the recording stops following mapping 1 and starts
    # following mapping 2.
    j0 = int(round(pre_end * FINE_FPS))
    j1 = min(rec_fine.shape[1], int(round(post_start * FINE_FPS)))
    tau = (pre_end + post_start) / 2
    if j1 - j0 > 2:
        X = rec_fine[:, j0:j1]
        n = j1 - j0
        Y1 = resample_time(track.fine, (a1 + r1 * j0 / FINE_FPS) * FINE_FPS, r1, n)
        Y2 = resample_time(track.fine, (a2 + r2 * j0 / FINE_FPS) * FINE_FPS, r2, n)
        d = framewise_similarity(X, Y1) - framewise_similarity(X, Y2)
        prefix = np.concatenate([[0.0], np.cumsum(d)])
        i = int(np.argmax(prefix))
        tau = (j0 + i) / FINE_FPS

    kind = "forward" if (a2 + r2 * tau) > (a1 + r1 * tau) else "backward"
    return Jump(
        track=track.index,
        kind=kind,
        ratio=r1,
        ratio_post=r2,
        a_pre=a1,
        a_post=a2,
        time=tau,
        pre_score=s1,
        post_score=s2,
        pre_region=(pre_start, pre_end),
        post_region=(post_start, post_end),
        notes=notes,
    )


def same_line(track: Track, rec_fine: np.ndarray, A: Chain, B: Chain, win_seconds: float, eps: float = 0.03) -> bool:
    """True if chains A and B are really one continuous playback.

    Repetitive music (identical bars/phrases) makes the coarse path hop between copies; that is
    not a beat jump. We test both directions: does continuing A explain B's audio as well as B
    does, or does extending B backwards explain A's audio as well as A does?
    """
    # forward: A's line over B's span
    t0 = B.start + win_seconds / 2
    t1 = min(B.end + win_seconds, t0 + 20.0)
    if t1 - t0 < 2.0:
        t0, t1 = B.start, B.end + win_seconds
    tc = (t0 + t1) / 2
    r = A.slope_end()
    _, s_cont = fine_offset(track.fine, rec_fine, t0, t1, r, A.predict_after(tc) - r * tc, search=0.3)
    _, s_new = fine_offset(track.fine, rec_fine, t0, t1, r, B.predict_before(tc) - r * tc, search=0.3)
    if s_cont >= s_new - eps:
        return True
    # backward: B's line over A's span. This only repairs a wrongly-placed start (ambiguous
    # intros), where B's line should do at least as well as A's own - a tie isn't enough
    # tolerance to also swallow real jumps between near-identical sections.
    t1 = A.end + win_seconds / 2
    t0 = max(A.start, t1 - 40.0)
    if t1 - t0 < 2.0:
        t0, t1 = A.start, A.end + win_seconds
    tc = (t0 + t1) / 2
    r = B.slope_start()
    _, s_back = fine_offset(track.fine, rec_fine, t0, t1, r, B.predict_before(tc) - r * tc, search=0.3)
    _, s_own = fine_offset(track.fine, rec_fine, t0, t1, r, A.predict_after(tc) - r * tc, search=0.3)
    return s_back >= s_own


def prune_shadows(chains: dict[int, list[Chain]], order_known: bool, margin: float = 0.03, frac: float = 0.7,
                  min_anchor_seconds: float = 30.0) -> None:
    """Remove chains that are just another track's similar-sounding section.

    A chain is a shadow if, on most of its windows, a different track has a chain that scores
    clearly higher at the same time. With an ordered playlist we also drop chains that sit
    outside the slot between the previous and next tracks' main sections.
    """
    window_best: dict[float, list[tuple[int, float]]] = {}
    for ti, cs in chains.items():
        for c in cs:
            for t, s in zip(c.t, c.s):
                window_best.setdefault(round(t, 3), []).append((ti, s))
    for ti, cs in chains.items():
        keep = []
        for c in cs:
            beaten = 0
            for t, s in zip(c.t, c.s):
                others = [s2 for t2, s2 in window_best.get(round(t, 3), []) if t2 != ti]
                if others and max(others) >= s + margin:
                    beaten += 1
            if beaten < frac * len(c.t):
                keep.append(c)
        chains[ti] = keep

    if order_known:
        # A track can't be heard before the middle of the previous track's main section, nor
        # after the middle of the next one's (transitions happen at the ends).
        # Only tracks with a substantial section act as anchors, so a track that is in the
        # History but wasn't recorded (played before REC, previewed) can't push others around.
        main = {ti: max(cs, key=lambda c: c.end - c.start) for ti, cs in chains.items() if cs}
        anchors = sorted(ti for ti, c in main.items() if c.end - c.start >= min_anchor_seconds)
        mid = {ti: (main[ti].start + main[ti].end) / 2 for ti in anchors}
        for ti in chains:
            prev = [a for a in anchors if a < ti]
            nxt = [a for a in anchors if a > ti]
            lo = mid[prev[-1]] if prev else -np.inf
            hi = mid[nxt[0]] if nxt else np.inf
            is_anchor = ti in mid
            chains[ti] = [c for c in chains[ti] if (is_anchor and c is main[ti]) or (c.end >= lo and c.start <= hi)]


def prune_by_schedule(chains: dict[int, list[Chain]], schedule: dict[int, list[tuple[float, float]]],
                      win_seconds: float, before: float = 90.0, after: float = 90.0) -> None:
    """With a rekordbox .cue we know roughly when each track was the current one. Drop chains of a
    track that don't come near any of its scheduled times (an incoming track is audible a while
    before it becomes current, and the outgoing one a while after)."""
    for ti, cs in chains.items():
        wins = schedule.get(ti)
        if not wins:
            chains[ti] = []  # not in the cue at all: wasn't played
            continue
        chains[ti] = [c for c in cs if any(c.end + win_seconds >= lo - before and c.start <= hi + after for lo, hi in wins)]


def _copy_scores(chain: Chain, track: Track, rec_fine: np.ndarray, win_seconds: float, at_end: bool = False,
                 at_start: bool = False):
    """Match a long excerpt of the chain against the whole track.

    Returns (c, r, p_cur, s_cur): similarity at every track offset (offset k <-> track position
    r*k/FINE_FPS at the excerpt start), the tempo ratio, the chain's current position at the
    excerpt start and the similarity there. None if the chain is too short.
    """
    span = chain.end + win_seconds - chain.start
    if span < 8.0:
        return None
    L = min(span, 40.0 if (at_end or at_start) else 60.0)
    if at_end:
        t1 = chain.end + win_seconds / 2
        t0 = max(chain.start, t1 - L)
    elif at_start:
        t0 = chain.start + win_seconds / 2
        t1 = min(chain.end + win_seconds, t0 + L)
    else:
        mid = (chain.start + chain.end + win_seconds) / 2
        t0, t1 = mid - L / 2, mid + L / 2
    j0, j1 = int(round(t0 * FINE_FPS)), min(rec_fine.shape[1], int(round(t1 * FINE_FPS)))
    W = rec_fine[:, j0:j1]
    r = float(np.median(chain.r)) if len(chain.t) < 3 else chain._fit(at_end=True, span=1e9)[0]
    m = int(np.floor((track.fine.shape[1] - 1) / r)) + 1
    F = resample_time(track.fine, 0.0, r, m)
    c = ncc(W, F)
    if c.size == 0:
        return None
    tc = j0 / FINE_FPS
    p_cur = chain.predict_before(tc) if tc <= chain.start else chain.predict_after(tc) if tc >= chain.end else float(np.interp(tc, chain.t, chain.p))
    k_cur = int(round(p_cur * FINE_FPS / r))
    near = c[max(0, k_cur - int(0.3 * FINE_FPS)) : max(0, k_cur + int(0.3 * FINE_FPS)) + 1]
    s_cur = float(near.max()) if near.size else -1.0
    return c, r, p_cur, s_cur


def reanchor(chain: Chain, track: Track, rec_fine: np.ndarray, win_seconds: float, margin: float = 0.0) -> float:
    """Move a chain onto the best-matching copy of its audio within the whole track.

    Coarse 6 s windows can't tell apart sections that repeat (a 32-bar loop, a second chorus);
    a long fine-resolution excerpt usually can, thanks to arrangement differences. When copies
    are near-identical the coarse choice is a coin flip anyway, so any improvement wins. Returns
    the shift applied (s), 0 if none.
    """
    res = _copy_scores(chain, track, rec_fine, win_seconds)
    if res is None:
        return 0.0
    c, r, p_cur, s_cur = res
    k = int(np.argmax(c))
    shift = r * k / FINE_FPS - p_cur
    if c[k] > s_cur + margin and abs(shift) > 0.3:
        chain.p = [p + shift for p in chain.p]
        return shift
    return 0.0


def reanchor_not_backward(A: Chain, B: Chain, track: Track, rec_fine: np.ndarray, win_seconds: float,
                          min_jump: float, tolerance: float = 0.03) -> bool:
    """A backward step from A to B usually means A sits on the wrong copy of a repeating section
    (DJs beat jump forward). Move A to the best nearly-as-good copy that makes the step not
    backward. Returns True if A was moved."""
    res = _copy_scores(A, track, rec_fine, win_seconds, at_end=True)
    if res is None:
        return False
    c, r, p_cur, s_cur = res
    d0 = B.p[0] - A.predict_after(B.t[0])
    best, best_shift = -np.inf, None
    for k in _local_peaks(c[None, :], 64)[0]:
        if c[k] < s_cur - tolerance or c[k] <= best:
            continue
        shift = r * k / FINE_FPS - p_cur
        if d0 - shift >= -min_jump and abs(shift) > 0.3:
            best, best_shift = float(c[k]), shift
    if best_shift is None:
        return False
    A.p = [p + best_shift for p in A.p]
    return True


def ambiguous_copy_offset(chain: Chain, track: Track, rec_fine: np.ndarray, win_seconds: float,
                          near_jump_end: bool, margin: float = 0.01) -> float:
    """Largest |offset| (s) to another copy of the same audio that matches the long excerpt next to
    the jump about as well as the chosen position (within `margin`). 0 = unambiguous."""
    res = _copy_scores(chain, track, rec_fine, win_seconds, at_end=near_jump_end, at_start=not near_jump_end)
    if res is None:
        return 0.0
    c, r, p_cur, s_cur = res
    k_cur = p_cur * FINE_FPS / r
    worst = 0.0
    for k in _local_peaks(c[None, :], 64)[0]:
        off = abs(k - k_cur) * r / FINE_FPS
        if off > 1.0 and c[k] >= s_cur - margin:
            worst = max(worst, off)
    return worst


def measure_ratio(track: Track, rec_fine: np.ndarray, sides, r0: float, region: float = 4.0) -> tuple[float, int]:
    """Tempo ratio from how the measured line offset drifts over time.

    sides: [(t_lo, t_hi, line_fn)], each one continuous playback (the two sides of a jump share
    the tempo but not the offset). At ratio r0, the best offset a of p = a + r0 t measured in short
    regions drifts by (r_true - r0) per second, so a robust straight-line fit with one shared slope
    and a per-side intercept gives r_true. Returns (ratio, number of regions used).
    """
    r = r0
    used = 0
    for _ in range(2):
        rows = []
        for si, (lo, hi, fn) in enumerate(sides):
            t = lo
            while t + region <= hi + 1e-9:
                tc = t + region / 2
                a, sc = fine_offset(track.fine, rec_fine, t, t + region, r, fn(tc) - r * tc, search=0.15)
                if sc > 0.5:
                    rows.append((si, tc, a))
                t += region
        if len(rows) < 3:
            break
        si_arr = np.array([x[0] for x in rows])
        t_arr = np.array([x[1] for x in rows])
        a_arr = np.array([x[2] for x in rows])
        keep = np.ones(len(rows), dtype=bool)
        beta = 0.0
        for _ in range(3):
            X = np.column_stack([t_arr] + [(si_arr == k).astype(float) for k in range(len(sides))])
            sol, *_ = np.linalg.lstsq(X[keep], a_arr[keep], rcond=None)
            beta = sol[0]
            resid = np.abs(X @ sol - a_arr)
            new_keep = resid < max(0.012, 3 * np.median(resid[keep]))
            if new_keep.sum() < 3 or (new_keep == keep).all():
                break
            keep = new_keep
        if np.ptp(t_arr[keep]) < 8.0:  # not enough time span to measure drift
            break
        r = r + beta
        used = int(keep.sum())
    return float(r), used


def estimate_ratio(track: Track, rec_fine: np.ndarray, regions, r0: float, span: float = 0.008) -> tuple[float, float]:
    """Tempo ratio maximizing the summed fine match over regions [(t0, t1, line_fn)]."""
    grid = r0 + np.arange(-span, span + 1e-9, 0.0005)
    totals = []
    for r in grid:
        tot = 0.0
        for t0, t1, fn in regions:
            tc = (t0 + t1) / 2
            _, s = fine_offset(track.fine, rec_fine, t0, t1, r, fn(tc) - r * tc, search=0.3)
            tot += s * (t1 - t0)
        totals.append(tot)
    totals = np.asarray(totals)
    k = int(np.argmax(totals))
    kf = parabolic_peak(totals, k)
    return float(r0 - span + 0.0005 * kf), float(totals[k] / sum(t1 - t0 for t0, t1, _ in regions))


def find_jumps(
    cm: CoarseMatches,
    tracks: list[Track],
    rec_fine: np.ndarray,
    order_known: bool = False,
    schedule: dict[int, list[tuple[float, float]]] | None = None,
    min_jump: float = 0.3,
    max_gap: float = 12.0,
    ambiguity_margin: float = 0.01,
) -> tuple[list[Jump], dict[int, list[Chain]]]:
    chains: dict[int, list[Chain]] = {t.index: build_chains(cm, t.index) for t in tracks}
    if schedule is not None:
        prune_by_schedule(chains, schedule, cm.win_seconds)
    prune_shadows(chains, order_known and schedule is None)
    for t in tracks:
        for c in chains[t.index]:
            reanchor(c, t, rec_fine, cm.win_seconds)
    jumps: list[Jump] = []
    for t in tracks:
        merged: list[Chain] = []
        for B in sorted(chains[t.index], key=lambda c: c.start):
            if merged and B.start - merged[-1].end <= max_gap:
                A = merged[-1]
                d = B.p[0] - A.predict_after(B.t[0])
                if d <= -min_jump and reanchor_not_backward(A, B, t, rec_fine, cm.win_seconds, min_jump):
                    d = B.p[0] - A.predict_after(B.t[0])
                if abs(d) < min_jump or same_line(t, rec_fine, A, B, cm.win_seconds):
                    # Same playback. Keep whichever line is better supported (longer), re-projecting
                    # the other chain's windows onto it.
                    keep, other = (A, B) if len(A.t) >= len(B.t) else (B, A)
                    projected = [
                        (tt, keep.predict_after(tt) if tt > keep.end else keep.predict_before(tt), rr, ss)
                        for tt, rr, ss in zip(other.t, other.r, other.s)
                    ]
                    pts = sorted(list(zip(keep.t, keep.p, keep.r, keep.s)) + projected)
                    new = Chain(t.index)
                    for p_ in pts:
                        new.add(*p_)
                    merged[-1] = new
                    continue
            merged.append(B)
        chains[t.index] = merged
        for A, B in zip(merged, merged[1:]):
            if 0 < B.start - A.end <= max_gap:
                j = refine_jump(t, rec_fine, A, B, cm.win_seconds)
                if abs(j.skipped_track_seconds) < min_jump:
                    continue
                # Leaving a jump unfilled is itself an error of the full jump length (the
                # recording stays seamless, just short). Filling from the wrong copy of a repeated
                # section errs by that copy's offset. So only refuse when the offset could be
                # comparable to the jump itself.
                off = max(ambiguous_copy_offset(A, t, rec_fine, cm.win_seconds, True, ambiguity_margin),
                          ambiguous_copy_offset(B, t, rec_fine, cm.win_seconds, False, ambiguity_margin))
                if off >= 0.5 * abs(j.skipped_track_seconds):
                    j.ambiguous = True
                    j.notes.append(f"repeated section: a copy {off:.1f}s away matches about as well")
                elif off > 0:
                    j.notes.append(f"section repeats; fill length may be off by up to {off:.1f}s (sounds seamless either way)")
                jumps.append(j)
    jumps.sort(key=lambda j: j.time)
    return jumps, chains


def other_tracks_active(chains: dict[int, list[Chain]], track: int, t: float, win_seconds: float, margin: float = 1.0) -> list[int]:
    """Tracks (other than `track`) that look audible around time t."""
    active = []
    for ti, cs in chains.items():
        if ti == track:
            continue
        for c in cs:
            lo = c.start + 0.25 * win_seconds
            hi = c.end + 0.75 * win_seconds
            if lo - margin <= t <= hi + margin:
                active.append(ti)
                break
    return active
