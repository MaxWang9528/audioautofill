"""Track list input: rekordbox recording .cue sheets, .m3u/.m3u8 playlists, or plain folders."""

from __future__ import annotations

import os
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from .audio_io import is_audio_file

_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def normalize_path(entry: str, base_dir: Path) -> Path:
    """Turn a playlist entry into a path that exists on *this* OS where possible.

    Handles file:// URLs, Windows drive paths when running under WSL (C:\\x -> /mnt/c/x),
    and paths relative to the playlist file.
    """
    p = entry.strip().strip('"')
    if p.lower().startswith("file://"):
        p = urllib.parse.unquote(p[7:])
        if p.lower().startswith("localhost/"):
            p = p[len("localhost"):]
        if re.match(r"^/[A-Za-z]:[\\/]", p):  # file:///C:/...
            p = p[1:]
    m = _DRIVE_RE.match(p)
    if m:
        if os.name == "nt":
            return Path(p)
        drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    if os.name != "nt":
        p = p.replace("\\", "/")
    path = Path(p)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def parse_m3u(path: Path) -> list[Path]:
    tracks = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tracks.append(normalize_path(line, path.parent))
    return tracks


@dataclass
class CueEntry:
    path: Path
    start: float  # seconds into the recording


def parse_cue(path: Path) -> list[CueEntry]:
    """rekordbox writes a .cue next to each recording: one TRACK per track played, with the
    track's file and the time (INDEX 01, mm:ss:frames at 75 frames/s) it came in."""
    entries: list[CueEntry] = []
    in_track = False
    cur_file: str | None = None
    text = _read_text(path)
    raw: list[tuple[str, tuple[int, int, int]]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("TRACK "):
            in_track = True
            cur_file = None
        elif s.upper().startswith("FILE ") and in_track:
            m = re.match(r'FILE\s+"(.*)"', s, re.I) or re.match(r"FILE\s+(\S+)", s, re.I)
            if m:
                cur_file = m.group(1)
        elif s.upper().startswith("INDEX 01") and in_track and cur_file:
            m = re.match(r"INDEX\s+01\s+(\d+):(\d+):(\d+)", s, re.I)
            if m:
                raw.append((cur_file, tuple(int(g) for g in m.groups())))
    # Standard cue sheets use MM:SS:FF (75 frames/s), but rekordbox writes HH:MM:SS.
    hms = "rekordbox" in text.lower() or any(c >= 75 for _, (_, _, c) in raw)
    for f, (a, b, c) in raw:
        sec = a * 3600 + b * 60 + c if hms else a * 60 + b + c / 75.0
        entries.append(CueEntry(normalize_path(f, path.parent), float(sec)))
    return entries


class FileIndex:
    """Finds a track by file name under some folders, for playlists whose paths are stale
    (e.g. files copied by rekordbox's USB export, which reorganizes and sometimes truncates names)."""

    def __init__(self, dirs: list[Path]):
        self.by_name: dict[str, list[Path]] = {}
        for d in dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    p = Path(root) / f
                    if is_audio_file(p):
                        self.by_name.setdefault(f.lower(), []).append(p)

    def find(self, missing: Path) -> tuple[Path | None, str]:
        name = missing.name.lower()
        hits = self.by_name.get(name, [])
        if len(hits) == 1:
            return hits[0], "same name"
        if len(hits) > 1:
            same_parent = [h for h in hits if h.parent.name.lower() == missing.parent.name.lower()]
            return (same_parent or hits)[0], "same name (several copies; picked one)"
        stem, ext = os.path.splitext(name)
        cands = []
        for k, ps in self.by_name.items():
            kstem, kext = os.path.splitext(k)
            if kext == ext and len(kstem) >= 12 and stem.startswith(kstem):  # truncated by the export
                cands.extend((len(kstem), p) for p in ps)
        if cands:
            cands.sort(key=lambda c: -c[0])
            return cands[0][1], "truncated name"
        return None, ""


@dataclass
class TrackList:
    paths: list[Path] = field(default_factory=list)
    missing: list[Path] = field(default_factory=list)
    relocated: list[tuple[Path, Path, str]] = field(default_factory=list)
    ordered: bool = False
    # From a .cue: when each track was the current track, as [(start, end)] in recording seconds.
    schedule: dict[Path, list[tuple[float, float]]] | None = None


def collect_tracks(sources: list[str], search_dirs: list[Path] | None = None) -> TrackList:
    """Resolve the track sources. Order is preserved, duplicates removed."""
    out = TrackList(ordered=all(Path(s).suffix.lower() in (".m3u", ".m3u8", ".cue") for s in sources))
    index = FileIndex(search_dirs) if search_dirs else None
    seen: dict[str, Path] = {}
    cue_entries: list[tuple[Path, float]] = []

    def resolve(c: Path) -> Path | None:
        if c.exists():
            return c
        if index:
            hit, how = index.find(c)
            if hit:
                out.relocated.append((c, hit, how))
                return hit
        return None

    for src in sources:
        sp = Path(src)
        if sp.is_dir():
            candidates = sorted(p for p in sp.iterdir() if p.is_file() and is_audio_file(p))
        elif sp.suffix.lower() in (".m3u", ".m3u8"):
            candidates = parse_m3u(sp)
        elif sp.suffix.lower() == ".cue":
            entries = parse_cue(sp)
            candidates = [e.path for e in entries]
            cue_entries.extend((e.path, e.start) for e in entries)
        else:
            candidates = [sp]
        for c in candidates:
            key = str(c).lower()
            if key in seen:
                continue
            r = resolve(c)
            if r is None:
                out.missing.append(c)
                seen[key] = None  # type: ignore[assignment]
                continue
            seen[key] = r
            if r not in out.paths:
                out.paths.append(r)

    if cue_entries:
        out.schedule = {}
        starts = [s for _, s in cue_entries]
        for i, (c, s) in enumerate(cue_entries):
            r = seen.get(str(c).lower())
            if r is None:
                continue
            end = starts[i + 1] if i + 1 < len(starts) else float("inf")
            out.schedule.setdefault(r, []).append((s, end))
    return out
