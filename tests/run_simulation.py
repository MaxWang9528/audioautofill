"""End-to-end tests on simulated DJ sets with known ground truth.

    python tests/run_simulation.py            # all scenarios
    python tests/run_simulation.py full_set   # one scenario

Each scenario writes tracks, a beat-jumped recording and the true continuous set to
tests/_out/<scenario>/, runs audioautofill, and compares the result with the truth.
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
from synth import SR, DeckPlay, const, envelope, make_track, ramp, render_set  # noqa: E402

OUT = ROOT / "tests" / "_out"

# The simulated key lock (WSOLA) lets its timing wander by up to +-8.7 ms, so the "truth" itself
# jitters locally. Each fill is aligned to the recording at both splices, so it inherits that
# jitter as a tiny length difference spread over the whole fill (e.g. 20 ms over 30 s = 0.07%
# tempo), which is inaudible. Tolerances are set at that level, not at sample accuracy.
INSERT_TOL_MS = 30
TIMELINE_TOL_MS = 40


def beats(bpm, n):
    return n * 60.0 / bpm


def write_track(d: Path, name: str, audio: np.ndarray, fmt: str, sr_file: int = SR) -> tuple[Path, np.ndarray]:
    """Write a track file and return (path, the audio a DJ app would decode from it, at SR)."""
    p = d / f"{name}.{fmt}"
    kw = {"format": "MP3"} if fmt == "mp3" else {}
    sf.write(p, resample(audio, SR, sr_file).T, sr_file, **kw)
    decoded, sr = sf.read(p, dtype="float32", always_2d=True)
    return p, resample(decoded.T, sr, SR)


def beat_time(deck: DeckPlay, track_bpm: float, after: float) -> float:
    """First beat of a constant-tempo deck at/after truth time `after`."""
    r = float(deck.rate(np.array([after]))[0])
    period = 60.0 / track_bpm / r
    return deck.start + np.ceil((after - deck.start) / period - 1e-9) * period


def build_set(d: Path, specs: list[dict], set_bpm: float, jumps: dict, blend_bars: int = 16,
              bass_swap: bool = False, use_wsola: bool = True, unplayed: tuple = ()):
    """Generic simulated set.

    specs: [{seed, bpm, bars, fmt, name?, loopy?, sr?}] in play order.
    jumps: {track_index: [(bars after the track's solo section starts, beats skipped)]}.
    bass_swap: incoming track sits at low volume for the whole blend, then swaps in one bar.
    unplayed: [(index in the m3u8, spec)] tracks in the history that never play (e.g. played
      before REC was pressed, or previewed).
    """
    bar = 4 * 60.0 / set_bpm
    paths, audio = [], []
    for i, sp in enumerate(specs):
        p, a = write_track(d, sp.get("name", f"track{i + 1}"), make_track(sp["seed"], sp["bpm"], sp["bars"], sp.get("loopy", False)),
                           sp["fmt"], sp.get("sr", SR))
        paths.append(p)
        audio.append(a)
    decks, start = [], 0.0
    for i, sp in enumerate(specs):
        r = set_bpm / sp["bpm"]
        end_play = start + (audio[i].shape[1] / SR - 1.0) / r
        last = i == len(specs) - 1
        level = 0.7 + 0.2 * ((sp["seed"] * 7) % 10) / 10
        if i == 0:
            pts = [(start, level)]
        elif bass_swap:
            pts = [(start, 0.0), (start + bar, 0.15), (start + (blend_bars - 1) * bar, 0.15), (start + blend_bars * bar, level)]
        else:
            pts = [(start, 0.0), (start + blend_bars / 2 * bar, level)]
        if last:
            deck_end, next_start = end_play, None
        else:
            next_start = end_play - (blend_bars + 4) * bar
            fade = [(next_start + (blend_bars - 1) * bar, level)] if bass_swap else [(next_start + blend_bars / 2 * bar, level)]
            pts += fade + [(next_start + blend_bars * bar, 0.0)]
            deck_end = next_start + blend_bars * bar
        deck = DeckPlay(i, start, deck_end, const(r), envelope(pts))
        solo = start + (blend_bars * bar if i > 0 else 0.0)
        for k, skip in jumps.get(i, []):
            tj = beat_time(deck, sp["bpm"], solo + k * bar)
            assert next_start is None or tj + 8 * bar < next_start, f"jump too close to next transition (track {i + 1})"
            deck.jumps.append((tj, beats(sp["bpm"], skip)))
        decks.append(deck)
        start = next_start
    truth, rec, tj = render_set(audio, decks, use_wsola=use_wsola)
    listed = list(paths)
    for pos, sp in sorted(unplayed, key=lambda u: u[0]):
        p, _ = write_track(d, sp.get("name", f"unplayed{pos}"), make_track(sp["seed"], sp["bpm"], sp["bars"]), sp["fmt"])
        listed.insert(pos, p)
    return listed, truth, rec, tj


# ----------------------------------------------------------------------------- scenarios


def scenario_single_r1(d: Path):
    """One track at original tempo, one 64-beat jump. The fill must null against truth."""
    bpm = 120.0
    p, a = write_track(d, "track1", make_track(1, bpm, bars=64), "wav")
    dur = a.shape[1] / SR - 1.0
    deck = DeckPlay(0, 0.0, dur, const(1.0), const(0.8), jumps=[(beats(bpm, 48), beats(bpm, 64))])
    truth, rec, tj = render_set([a], [deck], use_wsola=False)
    return [p], truth, rec, tj, {"null": True}


def scenario_full_set(d: Path):
    """Four tracks (wav/mp3/flac, one very repetitive), WSOLA key lock, transitions,
    a recorded tempo ramp and five beat jumps."""
    specs = [(11, 122.0, "wav", False), (22, 125.0, "mp3", False), (33, 128.0, "flac", True), (44, 124.0, "wav", False)]
    paths, audio = [], []
    for i, (seed, bpm, fmt, loopy) in enumerate(specs):
        p, a = write_track(d, f"track{i + 1}", make_track(seed, bpm, bars=96, loopy=loopy), fmt)
        paths.append(p)
        audio.append(a)
    set_bpm1, set_bpm2 = 124.0, 126.0
    bar = lambda bpm: 4 * 60.0 / bpm  # noqa: E731

    decks, t = [], 0.0
    # track1: plays from 0, jumps 64 beats, transition into track2 for the last 16 bars
    b1, r1 = specs[0][1], set_bpm1 / specs[0][1]
    len1 = (96 * bar(b1)) / r1
    t2_start = len1 - 16 * bar(set_bpm1) - 4 * bar(set_bpm1)
    decks.append(DeckPlay(0, 0.0, t2_start + 16 * bar(set_bpm1), const(r1),
                          envelope([(0, 0.8), (t2_start + 8 * bar(set_bpm1), 0.8), (t2_start + 16 * bar(set_bpm1), 0.0)]),
                          jumps=[]))
    # track2: fades in, solo, jump 32 beats, later jump 64 beats, then tempo ramp, transition to track3
    b2 = specs[1][1]
    r2a, r2b = set_bpm1 / b2, set_bpm2 / b2
    t2_solo = t2_start + 16 * bar(set_bpm1)
    ramp_t0, ramp_t1 = t2_solo + 120.0, t2_solo + 128.0
    len2_pos = 96 * bar(b2)
    # time when track2 reaches its end (approx, ramp handled numerically below)
    pos_at_ramp_end = (ramp_t1 - t2_start) * r2a + 4.0 * (r2b - r2a)
    t2_end = ramp_t1 + (len2_pos - pos_at_ramp_end) / r2b
    t3_start = t2_end - 20 * bar(set_bpm2)
    decks.append(DeckPlay(1, t2_start, t3_start + 16 * bar(set_bpm2), ramp(r2a, r2b, ramp_t0, ramp_t1),
                          envelope([(t2_start, 0.0), (t2_start + 8 * bar(set_bpm1), 0.7), (t3_start + 8 * bar(set_bpm2), 0.7), (t3_start + 16 * bar(set_bpm2), 0.0)])))
    # track3 (loopy): solo jump 128 beats, transition to track4
    b3, r3 = specs[2][1], set_bpm2 / specs[2][1]
    len3 = 96 * bar(b3) / r3
    t4_start = t3_start + len3 - 20 * bar(set_bpm2)
    decks.append(DeckPlay(2, t3_start, t4_start + 16 * bar(set_bpm2), const(r3),
                          envelope([(t3_start, 0.0), (t3_start + 8 * bar(set_bpm2), 0.9), (t4_start + 8 * bar(set_bpm2), 0.9), (t4_start + 16 * bar(set_bpm2), 0.0)])))
    # track4: fades in, one 64-beat jump, plays out
    b4, r4 = specs[3][1], set_bpm2 / specs[3][1]
    len4 = 96 * bar(b4) / r4
    decks.append(DeckPlay(3, t4_start, t4_start + len4 - 1.0, const(r4),
                          envelope([(t4_start, 0.0), (t4_start + 8 * bar(set_bpm2), 0.75)])))

    def beat_time(deck, track_bpm, after):
        """First beat of the deck at/after truth time `after` (constant tempo)."""
        r = float(deck.rate(np.array([after]))[0])
        period = 60.0 / track_bpm / r
        k = np.ceil((after - deck.start) / period)
        return deck.start + k * period

    # jumps (truth time, skipped track seconds) - all in solo sections, >= 8 bars from transitions
    decks[0].jumps = [(beat_time(decks[0], b1, 40.0), beats(b1, 64))]
    decks[1].jumps = [(beat_time(decks[1], b2, t2_solo + 25.0), beats(b2, 32)),
                      (beat_time(decks[1], b2, t2_solo + 70.0), beats(b2, 64))]
    decks[2].jumps = [(beat_time(decks[2], b3, t3_start + 16 * bar(set_bpm2) + 30.0), beats(b3, 128))]
    decks[3].jumps = [(beat_time(decks[3], b4, t4_start + 16 * bar(set_bpm2) + 20.0), beats(b4, 64))]

    truth, rec, tj = render_set(audio, decks, use_wsola=True)
    return paths, truth, rec, tj, {"null": False, "loopy": {2}}


def scenario_jump_in_mix(d: Path):
    """A jump while two tracks are audible must be left alone (not filled)."""
    bpm = 124.0
    a_p, a = write_track(d, "trackA", make_track(5, bpm, bars=48), "wav")
    b_p, b = write_track(d, "trackB", make_track(6, bpm, bars=48), "wav")
    bar = 4 * 60.0 / bpm
    tb = 24 * bar
    deckA = DeckPlay(0, 0.0, tb + 16 * bar, const(1.0), envelope([(0, 0.8), (tb + 8 * bar, 0.8), (tb + 16 * bar, 0.0)]),
                     jumps=[(tb + 4 * bar, beats(bpm, 32))])  # jump in the middle of the blend
    deckB = DeckPlay(1, tb, tb + 40 * bar, const(1.0), envelope([(tb, 0.0), (tb + 8 * bar, 0.8)]))
    truth, rec, tj = render_set([a, b], [deckA, deckB], use_wsola=False)
    return [a_p, b_p], truth, rec, tj, {"null": False, "expect_unfilled": True}


def scenario_history_extras(d: Path):
    """Real-library details: tracks in the History that were never recorded (one before the set,
    one previewed mid-set), a non-ASCII filename with brackets, a 48 kHz FLAC, an MP3."""
    specs = [
        {"seed": 101, "bpm": 123.0, "bars": 96, "fmt": "wav"},
        {"seed": 102, "bpm": 126.0, "bars": 96, "fmt": "flac", "sr": 48000, "name": "Ünder Pressure (Extended Mix) [Kölsch Rec]"},
        {"seed": 103, "bpm": 125.0, "bars": 96, "fmt": "mp3", "loopy": True},
        {"seed": 104, "bpm": 127.0, "bars": 96, "fmt": "wav"},
    ]
    jumps = {0: [(12, 64)], 1: [(10, 32)], 2: [(16, 64)], 3: [(12, 64)]}
    unplayed = [(0, {"seed": 201, "bpm": 124.0, "bars": 64, "fmt": "wav", "name": "warmup track (played before REC)"}),
                (3, {"seed": 202, "bpm": 125.0, "bars": 64, "fmt": "mp3", "name": "previewed track"})]
    paths, truth, rec, tj = build_set(d, specs, 125.0, jumps, unplayed=unplayed)
    return paths, truth, rec, tj, {"null": False, "loopy": {2}}


def scenario_long_blend(d: Path):
    """32-bar bass-swap blends (incoming track quiet until the swap), then an early jump."""
    specs = [{"seed": 301, "bpm": 124.0, "bars": 112, "fmt": "wav"},
             {"seed": 302, "bpm": 126.0, "bars": 112, "fmt": "wav"},
             {"seed": 303, "bpm": 125.0, "bars": 112, "fmt": "wav"}]
    jumps = {1: [(10, 64)], 2: [(10, 32)]}
    paths, truth, rec, tj = build_set(d, specs, 125.0, jumps, blend_bars=32, bass_swap=True)
    return paths, truth, rec, tj, {"null": False}


def scenario_scale(d: Path):
    """About an hour of set (truth) from 16 tracks with two jumps each - checks speed and memory."""
    rng = np.random.default_rng(7)
    fmts = ["wav", "mp3", "flac", "aiff"]
    specs = [{"seed": 400 + i, "bpm": float(np.round(120 + 10 * rng.random(), 1)), "bars": 128, "fmt": fmts[i % 4],
              "loopy": i % 5 == 4} for i in range(16)]
    jumps = {i: [(8, 64), (40, 64)] for i in range(16)}
    paths, truth, rec, tj = build_set(d, specs, 126.0, jumps)
    return paths, truth, rec, tj, {"null": False, "loopy": {i for i, sp in enumerate(specs) if sp["loopy"]}}


SCENARIOS = {
    "single_r1": scenario_single_r1,
    "full_set": scenario_full_set,
    "jump_in_mix": scenario_jump_in_mix,
    "history_extras": scenario_history_extras,
    "long_blend": scenario_long_blend,
    "scale": scenario_scale,
}


# ----------------------------------------------------------------------------- evaluation


def feature_offset(a: np.ndarray, b: np.ndarray, t: float, span: float = 3.0, search: float = 0.3) -> tuple[float, float]:
    """Time offset (s) of b relative to a around time t, via fine features."""
    i0, i1 = int((t - span / 2) * SR), int((t + span / 2) * SR)
    s = int(search * SR)
    if i0 - s < 0 or i1 + s > min(a.shape[1], b.shape[1]):
        return np.nan, 0.0
    fa = logmel(resample(to_mono(a[:, i0:i1])[None], SR, ANALYSIS_SR)[0])
    fb = logmel(resample(to_mono(b[:, i0 - s : i1 + s])[None], SR, ANALYSIS_SR)[0])
    c = ncc(fa, fb)
    k = int(np.argmax(c))
    return (k / FINE_FPS) - search, float(c[k])


def run(name: str) -> bool:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    t0 = time.time()
    paths, truth, rec, truth_jumps, opts = SCENARIOS[name](d)
    rec_p, truth_p, out_p = d / "recording.wav", d / "truth.wav", d / "output.wav"
    sf.write(rec_p, rec.T, SR, subtype="PCM_24")
    sf.write(truth_p, truth.T, SR, subtype="PCM_24")
    m3u = d / "history.m3u8"
    m3u.write_text("#EXTM3U\n" + "".join(f"#EXTINF:0,{p.stem}\n{p}\n" for p in paths), encoding="utf-8")
    print(f"simulated in {time.time() - t0:.1f}s: recording {rec.shape[1] / SR:.1f}s, truth {truth.shape[1] / SR:.1f}s, {len(truth_jumps)} jumps")

    rec_len = rec.shape[1]
    del truth, rec  # keep the test process small while the tool runs
    t0 = time.time()
    log_p = d / "autofill_log.txt"
    with open(log_p, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen([sys.executable, "-m", "audioautofill", str(rec_p), "-t", str(m3u), "-o", str(out_p)],
                                stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        peak = 0
        try:
            import psutil

            ps = psutil.Process(proc.pid)
            while proc.poll() is None:
                # the venv python.exe on Windows is a launcher; the real interpreter is a child
                try:
                    procs = [ps, *ps.children(recursive=True)]
                except psutil.Error:
                    procs = []
                for q in procs:
                    try:
                        mi = q.memory_info()
                        peak = max(peak, getattr(mi, "peak_wset", 0) or mi.rss)
                    except psutil.Error:
                        pass
                time.sleep(0.1)
        except ImportError:
            pass
        rc = proc.wait()
    print(log_p.read_text(encoding="utf-8"))
    print(f"audioautofill exit {rc} in {time.time() - t0:.1f}s" + (f", peak memory {peak / 2**20:.0f} MB" if peak else ""))
    if rc != 0:
        return False

    report = json.loads(out_p.with_suffix(".autofill.json").read_text())
    out, _ = sf.read(out_p, dtype="float32", always_2d=True)
    out = out.T
    truth_q, _ = sf.read(truth_p, dtype="float32", always_2d=True)
    truth_q = truth_q.T
    ok = True

    # jump detection
    found = [j for j in report["jumps"] if j["kind"] == "forward"]
    skipped_ok = []
    warned_ms = []
    print("\njump detection (truth vs found):")
    for tj in truth_jumps:
        m = [j for j in found if abs(j["time"] - tj["rec_time"]) < 1.0]
        if not m:
            print(f"  MISSED  t={tj['rec_time']:.2f} track{tj['track'] + 1} {tj['from']:.2f}->{tj['to']:.2f}")
            # A jump during a blend may go undetected; that's acceptable only if nothing is spliced.
            # In a loop-heavy track the tool may place it wrongly; acceptable only if it reported
            # an ambiguous (unfilled) jump in that track.
            flagged = any(j["track"] == f"track{tj['track'] + 1}" and j["status"] == "skipped" and "repeats" in (j.get("reason") or "")
                          for j in found)
            if tj["track"] in opts.get("loopy", ()):
                print("          (loop-heavy track: " + ("reported as ambiguous, left as recorded)" if flagged
                      else "jump between identical copies is invisible - known limitation, left as recorded)"))
                skipped_ok.append(tj)
                continue
            ok = ok and bool(opts.get("expect_unfilled"))
            continue
        j = m[0]
        e_from = (j["from"] - tj["from"]) * 1000
        e_ins = (j.get("inserted_seconds", 0) - tj["inserted"]) * 1000
        print(f"  t={tj['rec_time']:7.2f} (found {j['time']:7.2f})  track pos {tj['from']:.2f}->{tj['to']:.2f}  "
              f"from err {e_from:+.1f}ms  inserted err {e_ins:+.1f}ms  status={j['status']}"
              + (f" ({j.get('reason')})" if j["status"] != "filled" else ""))
        if opts.get("expect_unfilled"):
            if j["status"] == "filled":
                print("  FAIL: this jump should have been left alone")
                ok = False
        elif j["status"] != "filled":
            # Skipping is acceptable only for the deliberately loop-heavy tracks, and only when
            # the tool says why (repeated section) - never silently.
            if tj["track"] in opts.get("loopy", ()) and "repeats" in (j.get("reason") or ""):
                skipped_ok.append(tj)
            else:
                ok = False
        elif abs(e_ins) > INSERT_TOL_MS:
            # acceptable only if the tool itself warned that the section repeats by about that much
            bound = max([float(n.split("up to ")[1].split("s")[0]) for n in j.get("notes", []) if "may be off by up to" in n] or [0])
            if abs(e_ins) <= bound * 1000 + INSERT_TOL_MS:
                print(f"          (tool warned: repeated section, may be off by up to {bound:.1f}s)")
                warned_ms.append(e_ins)
            else:
                ok = False
    extra = [j for j in found if not any(abs(j["time"] - tj["rec_time"]) < 1.0 for tj in truth_jumps)]
    for j in extra:
        print(f"  FALSE POSITIVE at {j['time']:.2f}: {j}")
        ok = ok and j["status"] != "filled"

    if opts.get("expect_unfilled"):
        print(f"\noutput length {out.shape[1] / SR:.3f}s (recording {rec_len / SR:.3f}s)")
        same = out.shape[1] == rec_len
        print("output unchanged from recording:", same)
        return ok and same

    if skipped_ok:
        print(f"\n{len(skipped_ok)} jump(s) in loop-heavy tracks correctly reported as ambiguous and left as recorded")
    dl = (out.shape[1] - truth_q.shape[1]) / SR * 1000 + sum(tj["inserted"] for tj in skipped_ok) * 1000 - sum(warned_ms)
    print(f"\nlength: output {out.shape[1] / SR:.3f}s vs truth {truth_q.shape[1] / SR:.3f}s  ({dl:+.1f} ms)")
    if abs(dl) > INSERT_TOL_MS * max(1, len(truth_jumps)):
        ok = False

    if opts.get("null"):
        n = min(out.shape[1], truth_q.shape[1])
        err = np.abs(out[:, :n] - truth_q[:, :n])
        peak = float(err.max())
        print(f"null test: max |output - truth| = {peak:.2e} ({20 * np.log10(peak + 1e-12):.1f} dBFS)")
        ok = ok and peak < 1e-3

    # timeline drift: output should line up with truth everywhere
    offs = []
    t_end = min(out.shape[1], truth_q.shape[1]) / SR - 5.0
    if skipped_ok or warned_ms:  # after an unfilled/warned gap the timelines legitimately diverge
        firsts = [tj["truth_time"] for tj in skipped_ok]
        firsts += [tj["truth_time"] for tj in truth_jumps for j in found
                   if abs(j["time"] - tj["rec_time"]) < 1.0 and any("may be off by up to" in n for n in j.get("notes", []))]
        t_end = min(t_end, min(firsts) - 5.0)
    for t in np.arange(5.0, t_end, 10.0):
        o, c = feature_offset(truth_q, out, t)
        if not np.isnan(o) and c > 0.5:
            offs.append((t, o * 1000, c))
    worst = max(offs, key=lambda x: abs(x[1])) if offs else (0, 0, 0)
    print(f"timeline check at {len(offs)} points: worst offset {worst[1]:+.1f} ms at {worst[0]:.0f}s; "
          f"median |offset| {np.median([abs(o[1]) for o in offs]) if offs else 0:.1f} ms")
    if offs and abs(worst[1]) > TIMELINE_TOL_MS:
        ok = False
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    names = sys.argv[1:] or list(SCENARIOS)
    results = {n: run(n) for n in names}
    print("\n" + "\n".join(f"{n}: {'PASS' if r else 'FAIL'}" for n, r in results.items()))
    sys.exit(0 if all(results.values()) else 1)
