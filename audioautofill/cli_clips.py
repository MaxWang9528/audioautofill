"""`audioautofill clips`: build a full set from one recording per transition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .clips import ClipError, analyze_clip, assemble, find_clip_files, plan_bridges
from .playlist import collect_tracks


def fmt_t(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m}:{s:05.2f}"


def main(argv: list[str]) -> int:
    from .cli import Log, analyze_track  # shared helpers

    ap = argparse.ArgumentParser(
        prog="audioautofill clips",
        description="Build a full DJ set from separate recordings of each transition. Each clip should start with "
                    "the outgoing track playing alone and end with the incoming track playing alone (a few settled "
                    "seconds at each end). The music between clips is rendered from the original track files.",
    )
    ap.add_argument("clips", nargs="+", type=Path, help="clip files in set order, or a folder of them (sorted by name)")
    ap.add_argument("-t", "--tracks", action="append", default=[],
                    help="extra track list: .cue, .m3u8, folder or file (by default each clip's own .cue is used)")
    ap.add_argument("-s", "--search", action="append", type=Path, default=[],
                    help="folder to look in for tracks whose listed path doesn't exist (e.g. a USB export's Contents)")
    ap.add_argument("-o", "--output", type=Path, help="output file (default: <folder>_full.wav next to the clips' folder)")
    ap.add_argument("--no-intro", action="store_true", help="start at the first clip instead of the first track's beginning")
    ap.add_argument("--no-outro", action="store_true", help="stop at the end of the last clip instead of playing the last track out")
    ap.add_argument("--engine", choices=["r2", "r3"], default="r2", help="time-stretch engine for fills (default r2)")
    ap.add_argument("--max-tempo-change", type=float, default=0.10)
    ap.add_argument("--crossfade-ms", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true", help="analyze and report only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    log = Log(args.verbose)

    first = args.clips[0]
    if args.output:
        out_path = args.output
    elif first.is_dir():
        out_path = first.parent / f"{first.name}_full.wav"
    else:
        out_path = first.with_name(first.stem + "_full.wav")
    exclude = {out_path.resolve()}
    files = find_clip_files(args.clips, exclude)
    missing_files = [f for f in files if not f.exists()]
    if missing_files or not files:
        log("Clip file(s) not found: " + ", ".join(str(f) for f in missing_files) if missing_files else "No clips found.")
        return 2
    log(f"Clips ({len(files)}, in this order):")
    for f in files:
        log(f"  {f.name}")

    # Track lists: each clip's own .cue, plus anything given with -t.
    per_clip_sources = []
    all_sources = list(args.tracks)
    for f in files:
        cue = f.with_suffix(".cue")
        per_clip_sources.append([str(cue)] if cue.exists() else [])
        if cue.exists():
            all_sources.append(str(cue))
    if not all_sources:
        log("No track list: put each clip's rekordbox .cue next to it, or pass -t (a .m3u8, a folder of tracks...).")
        return 2
    tl = collect_tracks(all_sources, args.search)
    for m in tl.missing:
        log(f"  ! missing track file: {m}" + ("" if args.search else "   (use --search <folder>)"))
    if tl.relocated:
        log(f"  {len(tl.relocated)} track(s) found via the search folder(s)")
    if not tl.paths:
        log("No track files found.")
        return 2

    log.stage(f"Analyzing {len(tl.paths)} tracks...")
    tracks = []
    for p in tl.paths:
        try:
            tracks.append(analyze_track(len(tracks), p))
        except Exception as e:  # noqa: BLE001
            log(f"  ! skipping {p.name}: {e}")
    index_of = {t.path: t.index for t in tracks}
    steps = int(round(args.max_tempo_change / 0.02))
    ratios = 1.0 + 0.02 * np.arange(-steps, steps + 1)

    log.stage("Analyzing clips...")
    clips = []
    try:
        for f, src in zip(files, per_clip_sources):
            cands = list(range(len(tracks)))
            if src:
                own = collect_tracks(src, args.search).paths
                own_idx = [index_of[p] for p in own if p in index_of]
                if len(own_idx) >= 2:
                    cands = own_idx
            c = analyze_clip(f, tracks, cands, ratios)
            clips.append(c)
            for which, e in (("head", c.head), ("tail", c.tail)):
                log(f"  {f.name} {which}: {tracks[e.track].name} at {fmt_t(e.pos(e.splice))} (x{e.r:.4f}), "
                    f"splice {e.splice:.2f}s into clip, match {e.score:.2f}")
                for n in e.notes:
                    log(f"      ! {n}")
                for n in e.debug:
                    log.debug(n)
        notes = plan_bridges(clips, tracks)
        for n in notes:
            log(f"  note: {n}")
    except ClipError as e:
        log(f"\nCan't build the set: {e}")
        return 1

    report = {
        "clips": [{"file": str(c.path),
                   **{w: {"track": tracks[e.track].name, "position": round(e.pos(e.splice), 3), "tempo_ratio": round(e.r, 5),
                          "splice": round(e.splice, 3), "window": [round(x, 2) for x in e.window], "match": round(e.score, 3),
                          "notes": e.notes} for w, e in (("head", c.head), ("tail", c.tail))}} for c in clips],
    }
    report_path = out_path.with_suffix(".autofill.json")
    if args.dry_run:
        report_path.write_text(json.dumps(report, indent=2))
        log(f"\nDry run - report written to {report_path}")
        return 0

    log("")
    log.stage("Rendering the set...")
    try:
        stats = assemble(clips, tracks, out_path, intro=not args.no_intro, outro=not args.no_outro,
                         xfade_s=args.crossfade_ms / 1000, engine=args.engine, log=log)
    except ClipError as e:
        log(f"\nCan't build the set: {e}")
        return 1
    labels_path = out_path.with_suffix(".labels.txt")
    with open(labels_path, "w", encoding="utf-8") as lf:
        for a, b, label in stats["labels"]:
            lf.write(f"{a:.6f}\t{b:.6f}\t{label}\n")
    report.update(bridges=stats["bridges"], output_seconds=round(stats["output_seconds"], 3),
                  clipped_samples=stats["clipped_samples"])
    report_path.write_text(json.dumps(report, indent=2))
    log("")
    log.stage(f"Done: {len(clips)} clips + {len(stats['bridges'])} fills -> {fmt_t(stats['output_seconds'])} set.")
    log(f"  audio:  {out_path}")
    log(f"  labels: {labels_path}  (Audacity: File > Import > Labels)")
    log(f"  report: {report_path}")
    if stats["clipped_samples"]:
        log(f"  ! {stats['clipped_samples']} samples clipped - consider recording with more headroom")
    return 0
