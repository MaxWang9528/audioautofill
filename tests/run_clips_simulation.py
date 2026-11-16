"""End-to-end test of `audioautofill clips` on a simulated set with known ground truth.

A continuous 6-track set is rendered (WSOLA key lock standing in for rekordbox, a master tempo
ramp during one transition). Clips are then cut out of it the way a DJ would record them:
  clip1: transition 1                 clip2: transitions 2 AND 3 (short track in between)
  clip3: transition 4 (no cue)        clip4: transition 5, its head overlapping clip3's tail
clip2 is EQ'd (low shelf -6 dB).
Each clip gets its own level; head/tail lengths are random (not on beats). Track 2 is loop-heavy
and sits at a fill. The tool's output is compared with the true set.

    python tests/run_clips_simulation.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from audioautofill.audio_io import resample, to_mono  # noqa: E402
from audioautofill.features import ANALYSIS_SR, FINE_FPS, logmel, ncc  # noqa: E402
from run_simulation import write_track  # noqa: E402
from synth import SR, DeckPlay, envelope, make_track, render_set  # noqa: E402

OUT = ROOT / "tests" / "_out" / "clips"
# What you'd hear: timing must be continuous across every join and the fills' tempo must match the
# true tempo closely. A fill's tempo is measured from the clip edges (often only ~10 s long, and the
# simulated key lock jitters by +-8.7 ms), so a fill can come out ~0.1% off - inaudible, but over a
# 2-minute fill the set drifts ~100 ms from the true performance. That drift is reported, with a
# loose limit.
JOIN_TIMING_TOL_MS = 20
TEMPO_TOL = 0.0015
DRIFT_TOL_MS = 250
JOIN_TOL_DB = 3.0


def set_bpm_fn(points):
    xs, ys = zip(*points)
    return lambda t: np.interp(t, xs, ys)


def time_at_position(start, rate_fn, target_pos, step=0.005):
    t, p = start, 0.0
    while p < target_pos:
        p += float(rate_fn(np.array([t]))[0]) * step
        t += step
    return t


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*"):
        if f.is_file():
            f.unlink()
    specs = [(601, 124.0, 96, "wav", False), (602, 126.0, 96, "mp3", True), (603, 122.0, 44, "flac", False),
             (604, 125.0, 96, "wav", False), (605, 127.0, 44, "mp3", False), (606, 123.0, 80, "wav", False)]
    tracks_dir = OUT / "tracks"
    tracks_dir.mkdir(exist_ok=True)
    paths, audio = [], []
    for i, (seed, bpm, bars, fmt, loopy) in enumerate(specs):
        p, a = write_track(tracks_dir, f"T{i + 1}", make_track(seed, bpm, bars, loopy), fmt)
        paths.append(p)
        audio.append(a)
    blend = 16
    bar_at = lambda bpm: 4 * 60.0 / bpm  # noqa: E731

    # master tempo: 124, ramping to 126 during transition 4
    decks, starts, ends = [], [], []
    t_start = 0.0
    ramp = None
    for i, (seed, bpm, bars, fmt, loopy) in enumerate(specs):
        starts.append(t_start)
        if i + 1 < len(specs):
            # incoming deck starts when this one is (blend + 4) bars from its end
            tmp_master = set_bpm_fn(ramp or [(0, 124.0), (1e6, 124.0)])
            rate = lambda t, b=bpm, m=tmp_master: m(t) / b  # noqa: E731
            target = (bars - blend - 4) * 4 * 60.0 / bpm
            nxt = time_at_position(t_start, rate, target)
            if i == 3:  # transition 4 (deck 4 -> deck 5): ramp the master tempo during the blend
                ramp = [(0, 124.0), (nxt, 124.0), (nxt + blend * bar_at(124.0), 126.0), (1e6, 126.0)]
            t_start = nxt
    master = set_bpm_fn(ramp)
    for i, (seed, bpm, bars, fmt, loopy) in enumerate(specs):
        rate = lambda t, b=bpm: master(t) / b  # noqa: E731
        s = starts[i]
        level = [0.8, 0.75, 0.85, 0.8, 0.78, 0.82][i]
        if i + 1 < len(specs):
            e = starts[i + 1] + blend * bar_at(float(master(starts[i + 1])))
            pts = ([(s, level)] if i == 0 else [(s, 0.0), (s + blend / 2 * bar_at(124), level)]) + \
                  [(starts[i + 1] + blend / 2 * bar_at(124), level), (e, 0.0)]
        else:
            e = time_at_position(s, rate, audio[i].shape[1] / SR - 0.01)
            pts = [(s, 0.0), (s + blend / 2 * bar_at(126), level)]
        ends.append(e)
        decks.append(DeckPlay(i, s, e, rate, envelope(pts)))
    truth, _, _ = render_set(audio, decks, use_wsola=True)
    sf.write(OUT / "truth.wav", truth.T, SR, subtype="PCM_24")

    # transitions k: from deck k+1 start to deck k end
    trans = [(starts[k + 1], ends[k]) for k in range(len(specs) - 1)]
    rng = np.random.default_rng(3)
    hl = lambda: float(rng.uniform(9.0, 15.0))  # noqa: E731
    h3_tail = 10.0
    t5_solo = trans[4][0] - trans[3][1]
    c4_head = t5_solo - h3_tail + 4.5  # overlaps clip 3's tail by 4.5 s
    clips_spec = [
        ("01 REC clip.wav", trans[0][0] - 7.0, trans[0][1] + hl(), 0.9, False, [0, 1]),  # short head
        ("02 REC clip.wav", trans[1][0] - hl(), trans[2][1] + hl(), 0.7, True, [1, 2, 3]),
        ("03 REC clip.wav", trans[3][0] - hl(), trans[3][1] + h3_tail, 0.85, False, None),
        ("04 REC clip.wav", trans[4][0] - c4_head, trans[4][1] + 7.0, 0.8, False, [4, 5]),  # short tail
    ]
    clip_dir = OUT / "clips"
    clip_dir.mkdir(exist_ok=True)
    for f in clip_dir.glob("*"):
        f.unlink()
    for name, a, b, gain, eq, cue_tracks in clips_spec:
        x = truth[:, int(a * SR) : int(b * SR)] * gain
        if eq:  # low shelf cut: -6 dB below ~150 Hz
            X = np.fft.rfft(x, axis=1)
            f = np.fft.rfftfreq(x.shape[1], 1 / SR)
            q = (f / 150.0) ** 4
            X *= (0.5 + 0.5 * q / (1 + q))[None, :]
            x = np.fft.irfft(X, x.shape[1], axis=1).astype(np.float32)
        sf.write(clip_dir / name, x.T, SR, subtype="PCM_24")
        if cue_tracks:
            lines = ['REM RECORDED_BY "rekordbox-dj"', f'FILE "{name}" WAVE']
            for k, ti in enumerate(cue_tracks):
                lines += [f"\tTRACK {k + 1:02d} AUDIO", f'\t\tFILE "{paths[ti]}" WAVE', f"\t\tINDEX 01 00:00:{k * 10:02d}"]
            (clip_dir / name).with_suffix(".cue").write_text("\n".join(lines), encoding="utf-8")
    # a copy of the set where the first clip's head is far too short (3 s): must fail clearly
    short_dir = OUT / "clips_short_head"
    short_dir.mkdir(exist_ok=True)
    for f in short_dir.glob("*"):
        f.unlink()
    for f in clip_dir.glob("*"):
        (short_dir / f.name).write_bytes(f.read_bytes())
    # T1 alone for 3 s, then T2 cut in at full level (no fade): the head is unusable
    r1 = 124.0 / specs[0][1]
    r2 = 124.0 / specs[1][1]
    p1 = 100.0
    dk = [DeckPlay(0, 0.0, 20.0, lambda t: np.full(np.shape(t), r1), lambda t: np.full(np.shape(t), 0.8)),
          DeckPlay(1, 3.0, 40.0, lambda t: np.full(np.shape(t), r2), lambda t: np.full(np.shape(t), 0.8))]
    src = [audio[0][:, int(p1 * SR):], audio[1]]
    short_truth, _, _ = render_set(src, dk, use_wsola=True)
    sf.write(short_dir / clips_spec[0][0], short_truth.T, SR, subtype="PCM_24")
    (short_dir / clips_spec[0][0]).with_suffix(".cue").unlink(missing_ok=True)
    for f in list(short_dir.glob("*")):
        if f.name != clips_spec[0][0]:
            f.unlink()
    m3u = OUT / "all_tracks.m3u8"
    m3u.write_text("#EXTM3U\n" + "".join(f"{p}\n" for p in paths), encoding="utf-8")
    return clip_dir, m3u, clips_spec, master, specs


def run_tool(args, log_name):
    log_p = OUT / log_name
    with open(log_p, "w", encoding="utf-8") as lf:
        rc = subprocess.run([sys.executable, "-m", "audioautofill", "clips", *args], stdout=lf, stderr=subprocess.STDOUT,
                            cwd=str(ROOT)).returncode
    return rc, log_p.read_text(encoding="utf-8")


def feature_offset(a, b, t, span=3.0, search=0.3):
    i0, i1 = int((t - span / 2) * SR), int((t + span / 2) * SR)
    s = int(search * SR)
    if i0 - s < 0 or i1 + s > min(a.shape[1], b.shape[1]):
        return np.nan, 0.0
    fa = logmel(resample(to_mono(a[:, i0:i1])[None], SR, ANALYSIS_SR)[0])
    fb = logmel(resample(to_mono(b[:, i0 - s : i1 + s])[None], SR, ANALYSIS_SR)[0])
    c = ncc(fa, fb)
    k = int(np.argmax(c))
    return (k / FINE_FPS) - search, float(c[k])


def band_db(x):
    bands = [(30, 150), (150, 600), (600, 2500), (2500, 8000), (8000, 16000)]
    m = to_mono(x)
    spec = np.abs(np.fft.rfft(m * np.hanning(m.size))) ** 2
    f = np.fft.rfftfreq(m.size, 1 / SR)
    return np.array([10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-12) for lo, hi in bands])


def main():
    t0 = time.time()
    clip_dir, m3u, spec, master, track_specs = build()
    print(f"simulated in {time.time() - t0:.1f}s")
    ok = True

    # 1) out-of-order clips must fail loudly
    files = sorted(clip_dir.glob("*.wav"))
    rc, log = run_tool([str(files[0]), str(files[2]), str(files[1]), str(files[3]), "-t", str(m3u),
                        "-o", str(OUT / "wrong_order.wav")], "log_wrong_order.txt")
    good_error = rc != 0 and "out of order" in log
    print(f"out-of-order clips rejected with a clear message: {good_error}")
    ok &= good_error

    rc, log = run_tool([str(OUT / "clips_short_head"), "-t", str(m3u), "-o", str(OUT / "short.wav")], "log_short.txt")
    good_error = rc != 0 and "too short" in log
    print(f"too-short head rejected with a clear message: {good_error}")
    if not good_error:
        print(log[-600:])
    ok &= good_error

    # 2) the real run
    out_p = OUT / "set_full.wav"
    t0 = time.time()
    rc, log = run_tool([str(clip_dir), "-t", str(m3u), "-o", str(out_p)], "log.txt")
    print(log)
    print(f"exit {rc} in {time.time() - t0:.1f}s")
    if rc != 0:
        print("FAIL")
        return 1
    # clean clips must not raise warnings (except the genuinely repetitive loop-heavy track T2)
    warns = [ln.strip() for ln in log.splitlines() if ln.strip().startswith("!")]
    unexpected = [w for w in warns if not ("repeats" in w and "'T2'" in w)]
    print(f"warnings: {len(warns)} ({len(unexpected)} unexpected)")
    for w in unexpected:
        print(f"  unexpected: {w}")
    ok &= not unexpected
    out, _ = sf.read(out_p, dtype="float32", always_2d=True)
    truth, _ = sf.read(OUT / "truth.wav", dtype="float32", always_2d=True)
    out, truth = out.T, truth.T
    dl = (out.shape[1] - truth.shape[1]) / SR * 1000
    print(f"length: output {out.shape[1] / SR:.3f}s vs truth {truth.shape[1] / SR:.3f}s ({dl:+.1f} ms)")
    ok &= abs(dl) <= DRIFT_TOL_MS

    offs = []
    for t in np.arange(5.0, min(out.shape[1], truth.shape[1]) / SR - 5.0, 10.0):
        o, c = feature_offset(truth, out, t)
        if not np.isnan(o) and c > 0.5:
            offs.append((t, o * 1000))
    worst = max(offs, key=lambda x: abs(x[1]))
    print(f"overall drift vs the true performance: worst {worst[1]:+.1f} ms at {worst[0]:.0f}s, "
          f"median |offset| {np.median([abs(o) for _, o in offs]):.1f} ms")
    ok &= abs(worst[1]) <= DRIFT_TOL_MS

    # tempo used at each clip edge vs the true tempo there
    report = json.loads(out_p.with_suffix(".autofill.json").read_text())
    worst_tempo = 0.0
    for (name, a, b, gain, eq, cue), c in zip(spec, report["clips"]):
        for w in ("head", "tail"):
            e = c[w]
            ti = int(e["track"][1:]) - 1
            true_r = float(master(a + e["splice"])) / track_specs[ti][1]
            err = e["tempo_ratio"] / true_r - 1
            worst_tempo = max(worst_tempo, abs(err))
            print(f"  {name} {w}: tempo x{e['tempo_ratio']:.5f} vs true x{true_r:.5f} ({err * 100:+.3f}%)")
    print(f"worst tempo error: {worst_tempo * 100:.3f}%")
    ok &= worst_tempo <= TEMPO_TOL

    # timing continuity across each join: offset vs truth just before and just after
    worst_jump = 0.0
    for b in report["bridges"]:
        for t in (b["start"], b["end"]):
            o1, c1 = feature_offset(truth, out, t - 1.6, span=2.5)
            o2, c2 = feature_offset(truth, out, t + 1.6, span=2.5)
            if np.isnan(o1) or np.isnan(o2) or min(c1, c2) < 0.5:
                continue
            jump = (o2 - o1) * 1000
            worst_jump = max(worst_jump, abs(jump))
            print(f"  join at {t:7.2f}s: timing step {jump:+.1f} ms")
    print(f"worst timing step at a join: {worst_jump:.1f} ms")
    ok &= worst_jump <= JOIN_TIMING_TOL_MS

    # tone/level continuity across every join: change across the join in the output, minus the
    # natural change at the same moment in the true set
    W = int(2.0 * SR)
    excess_all = []
    for b in report["bridges"]:
        for t in (b["start"], b["end"]):
            i = int(t * SR)
            if i - W < 0 or i + W > min(out.shape[1], truth.shape[1]):
                continue
            d_out = band_db(out[:, i : i + W]) - band_db(out[:, i - W : i])
            d_tru = band_db(truth[:, i : i + W]) - band_db(truth[:, i - W : i])
            ex = float(np.abs(d_out - d_tru).max())
            excess_all.append(ex)
            print(f"  join at {t:7.2f}s ({b['label'][:40]}): worst band mismatch {ex:.1f} dB")
    print(f"worst join tone/level mismatch: {max(excess_all):.1f} dB")
    ok &= max(excess_all) <= JOIN_TOL_DB
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
