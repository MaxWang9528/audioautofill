"""Command line entry point.

    python -m audioautofill recording.wav -t history.m3u8 -o full_set.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import __version__
from .audio_io import load_audio, load_mono_resampled_stream, load_track_for_render, resample, to_mono
from .features import ANALYSIS_SR, coarse_from_fine, logmel
from .matching import Track, coarse_match, find_jumps, other_tracks_active
from .playlist import collect_tracks
from .render import plan_fill, write_output


def fmt_t(sec: float) -> str:
    sign = "-" if sec < 0 else ""
    sec = abs(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(int(m), 60)
    return f"{sign}{h}:{m:02d}:{s:05.2f}" if h else f"{sign}{m}:{s:05.2f}"


class Log:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.t0 = time.time()

    def __call__(self, msg: str = ""):
        print(msg, flush=True)

    def stage(self, msg: str):
        print(f"[{time.time() - self.t0:6.1f}s] {msg}", flush=True)

    def debug(self, msg: str):
        if self.verbose:
            print("  [debug] " + msg, flush=True)

    def progress(self, label: str):
        last = [-1]

        def cb(frac: float):
            pct = int(frac * 100)
            if pct // 10 != last[0] // 10 or pct == 100:
                last[0] = pct
                print(f"\r  {label}: {pct:3d}%", end="\n" if pct >= 100 else "", flush=True)

        return cb


MAX_TRACKS_UNORDERED = 60


def analyze_track(i: int, path: Path) -> Track:
    audio, sr = load_audio(path)
    y = resample(to_mono(audio)[None], sr, ANALYSIS_SR)[0]
    fine = logmel(y)
    return Track(index=i, path=path, fine=fine, coarse=coarse_from_fine(fine), duration=audio.shape[1] / sr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.stdout.isatty():  # redirected output on Windows defaults to cp1252 and can't print every name
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if argv and argv[0] == "clips":
        from .cli_clips import main as clips_main

        return clips_main(argv[1:])
    if argv and argv[0] == "jumps":
        argv = argv[1:]
    ap = argparse.ArgumentParser(
        prog="audioautofill",
        description="Rebuild a full DJ set from a recording where the boring parts were skipped with beat jumps.",
    )
    ap.add_argument("recording", type=Path, help="the recorded set (WAV/AIFF/FLAC)")
    ap.add_argument("-t", "--tracks", action="append",
                    help="what was played: rekordbox's .cue for the recording (default: the .cue next to it), "
                         "a .m3u8 History export, a folder of tracks, or a track file (repeatable)")
    ap.add_argument("-s", "--search", action="append", type=Path, default=[],
                    help="folder to look in for tracks whose listed path doesn't exist, e.g. a USB export's "
                         "Contents folder (repeatable)")
    ap.add_argument("-o", "--output", type=Path, help="output file (default: <recording>_filled.wav)")
    ap.add_argument("--dry-run", action="store_true", help="only analyze and report; don't write audio")
    ap.add_argument("--max-tempo-change", type=float, default=0.10,
                    help="largest tempo difference from the original track to search for (default 0.10 = +-10%%)")
    ap.add_argument("--crossfade-ms", type=float, default=30.0, help="crossfade length at each splice (default 30)")
    ap.add_argument("--min-quality", type=float, default=0.6,
                    help="minimum spectral match at a splice to accept a fill, 0..1 (default 0.6)")
    ap.add_argument("--allow-many-tracks", action="store_true",
                    help=f"allow more than {MAX_TRACKS_UNORDERED} tracks without an ordered playlist")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)
    if not sys.stdout.isatty():  # redirected output on Windows defaults to cp1252 and can't print every name
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    log = Log(args.verbose)

    rec_path: Path = args.recording
    if not rec_path.exists():
        log(f"Recording not found: {rec_path}")
        return 2
    out_path: Path = args.output or rec_path.with_name(rec_path.stem + "_filled.wav")
    report_path = out_path.with_suffix(".autofill.json")
    labels_path = out_path.with_suffix(".labels.txt")

    try:
        info = sf.info(str(rec_path))
    except Exception as e:  # noqa: BLE001
        log(f"Can't read {rec_path}: {e}\nRecord your set as WAV (or AIFF/FLAC).")
        return 2
    log(f"Recording: {rec_path.name}  ({fmt_t(info.duration)}, {info.samplerate} Hz, {info.channels} ch, {info.subtype})")

    if not args.tracks:
        cue = rec_path.with_suffix(".cue")
        if not cue.exists():
            log("No track list given. Pass the recording's .cue (rekordbox writes one next to the WAV), "
                "an .m3u8 History export, or a folder with -t.")
            return 2
        args.tracks = [str(cue)]
        log(f"Track list: {cue.name}")
    tl = collect_tracks(args.tracks, args.search)
    track_paths, missing = tl.paths, tl.missing
    for old, new, how in tl.relocated:
        log.debug(f"found {old.name} at {new} ({how})")
    if tl.relocated:
        log(f"  {len(tl.relocated)} track(s) not at their listed path were found in the search folder(s)")
    for m in missing:
        log(f"  ! missing track file: {m}" + ("" if args.search else "   (use --search <folder> to look for moved files)"))
    if not track_paths:
        log("No track files found.")
        return 2
    log(f"Tracks: {len(track_paths)}" + (f" ({len(missing)} missing)" if missing else ""))
    order_known = tl.ordered
    if not order_known and len(track_paths) > MAX_TRACKS_UNORDERED and not args.allow_many_tracks:
        log(f"That's {len(track_paths)} tracks. Searching a whole library is slow, uses a lot of memory and "
            "invites false matches.\nExport the session's rekordbox History as .m3u8 and pass that with -t, "
            "or use --allow-many-tracks.")
        return 2

    # ---- analysis
    log.stage("Analyzing recording...")
    rec_fine = logmel(load_mono_resampled_stream(rec_path, ANALYSIS_SR))
    rec_coarse = coarse_from_fine(rec_fine)

    log.stage("Analyzing tracks...")
    tracks: list[Track] = []
    for p in track_paths:
        try:
            tracks.append(analyze_track(len(tracks), p))
            log.debug(f"{p.name}: {fmt_t(tracks[-1].duration)}")
        except Exception as e:  # noqa: BLE001
            log(f"  ! skipping {p.name}: {e}")
    if not tracks:
        return 2

    log.stage("Locating tracks in the recording...")
    steps = int(round(args.max_tempo_change / 0.02))
    ratios = 1.0 + 0.02 * np.arange(-steps, steps + 1)
    cm = coarse_match(rec_coarse, tracks, ratios, progress=log.progress("matching"))

    log.stage("Finding beat jumps...")
    schedule = None
    if tl.schedule is not None:
        schedule = {t.index: tl.schedule.get(t.path, []) for t in tracks}
    jumps, chains = find_jumps(cm, tracks, rec_fine, order_known=order_known, schedule=schedule)

    log("\nTimeline (where each track was heard):")
    spans = []
    for t in tracks:
        for c in chains[t.index]:
            spans.append((c.start, c.end + cm.win_seconds, t, c))
    spans.sort(key=lambda s: s[0])
    for a, b, t, c in spans:
        log(f"  {fmt_t(a)} - {fmt_t(b)}  {t.name}  (track {fmt_t(c.p[0])} -> {fmt_t(c.p[-1] + cm.win_seconds * c.r[-1])}, tempo x{np.median(c.r):.2f})")
    unseen = [t.name for t in tracks if not chains[t.index]]
    if unseen:
        log("  not found in recording: " + ", ".join(unseen))

    # ---- plan fills
    log("")
    log.stage(f"Beat jumps found: {len(jumps)}")
    fills = []
    report_jumps = []
    track_cache: dict[int, np.ndarray] = {}
    with sf.SoundFile(str(rec_path)) as rec:
        for j in jumps:
            t = tracks[j.track]
            entry = {
                "time": round(j.time, 3),
                "track": t.name,
                "kind": j.kind,
                "from": round(j.pos_before, 3),
                "to": round(j.pos_after, 3),
                "tempo_ratio": round(j.ratio, 5) if j.ratio_post is None else [round(j.ratio, 5), round(j.r_post, 5)],
                "match_scores": [round(j.pre_score, 3), round(j.post_score, 3)],
                "notes": list(j.notes),
            }
            report_jumps.append(entry)
            head = f"  @{fmt_t(j.time)}  {t.name}: {fmt_t(j.pos_before)} -> {fmt_t(j.pos_after)}  (x{j.ratio:.4f}" + (f" -> x{j.r_post:.4f}" if j.ratio_post is not None and abs(j.r_post - j.ratio) > 1e-6 else "") + ")"
            if j.kind == "backward":
                entry["status"] = "skipped"
                entry["reason"] = "backward jump (retake) - not supported yet, left as recorded"
                log(head + "\n      ! backward jump (retake?) - left as recorded")
                continue
            if j.ambiguous:
                entry["status"] = "skipped"
                entry["reason"] = ("this part of the track repeats almost identically, so the amount skipped can't be "
                                   "determined - left as recorded (re-record this jump, or jump from a more distinctive spot)")
                log(head + f"\n      ! skipped: {entry['reason']}")
                continue
            busy = other_tracks_active(chains, j.track, j.time, cm.win_seconds)
            if busy:
                entry["status"] = "skipped"
                entry["reason"] = "another track is audible at the jump: " + ", ".join(tracks[b].name for b in busy)
                log(head + f"\n      ! skipped: {entry['reason']}")
                continue
            if j.track not in track_cache:
                track_cache.clear()  # keep memory bounded; jumps are sorted by time so tracks rarely repeat
                track_cache[j.track] = load_track_for_render(t.path, rec.samplerate, rec.channels)
            f = plan_fill(j, t.name, track_cache[j.track], rec, min_quality=args.min_quality)
            entry.update({
                "status": "filled" if f.accepted else "skipped",
                "inserted_seconds": round(f.inserted_samples / rec.samplerate, 3),
                "gain": [round(f.g1, 3), round(f.g2, 3)],
                "splice_correlation": [round(f.rho1, 3), round(f.rho2, 3)],
                "splice_quality": [round(f.q1, 3), round(f.q2, 3)],
                "alignment": f.debug,
            })
            entry["notes"] = f.notes
            if not f.accepted:
                entry["reason"] = f.reason
                log(head + f"\n      ! skipped: {f.reason}")
            else:
                log(head + f"\n      fill {f.inserted_samples / rec.samplerate:.2f}s, gain {f.g1:.2f}/{f.g2:.2f}, "
                    f"match {min(f.q1, f.q2):.2f}")
            for n in f.notes:
                log(f"      note: {n}")
            fills.append((f, entry))

    report = {
        "recording": str(rec_path),
        "output": None if args.dry_run else str(out_path),
        "tracks": [str(t.path) for t in tracks],
        "missing_tracks": [str(m) for m in missing],
        "timeline": [{"start": round(a, 2), "end": round(b, 2), "track": t.name} for a, b, t, _ in spans],
        "jumps": report_jumps,
    }

    if args.dry_run:
        report_path.write_text(json.dumps(report, indent=2))
        log(f"\nDry run - report written to {report_path}")
        return 0

    log("")
    log.stage("Writing output...")
    stats = write_output(rec_path, out_path, [f for f, _ in fills], xfade_s=args.crossfade_ms / 1000)
    for f, entry in fills:
        if entry.get("status") == "filled" and not f.accepted:
            entry["status"] = "skipped"
            entry["reason"] = f.reason
    with open(labels_path, "w", encoding="utf-8") as lf:
        for a, b, f in stats["labels"]:
            jj = f.jump
            lf.write(f"{a:.6f}\t{b:.6f}\tfill {f.track_name} {fmt_t(jj.pos_before)}-{fmt_t(jj.pos_after)}\n")
    report["output_seconds"] = round(stats["output_seconds"], 3)
    report["input_seconds"] = round(stats["input_seconds"], 3)
    report["clipped_samples"] = stats["clipped_samples"]
    report_path.write_text(json.dumps(report, indent=2))

    n_filled = len(stats["labels"])
    log("")
    log.stage(f"Done: {n_filled} gap(s) filled. {fmt_t(stats['input_seconds'])} recorded -> {fmt_t(stats['output_seconds'])} set.")
    log(f"  audio:  {out_path}")
    log(f"  labels: {labels_path}  (import into Audacity: File > Import > Labels, to check each splice)")
    log(f"  report: {report_path}")
    if stats["clipped_samples"]:
        log(f"  ! {stats['clipped_samples']} samples clipped in the fills - consider recording with more headroom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
