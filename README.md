# audioautofill

Record a DJ set by beat-jumping through the parts where you aren't doing anything, then let
this tool rebuild the full-length set by filling every skipped section with the original track
audio, time-stretched to the tempo you were playing at and volume-matched to your recording.

## How it works

1. Every few seconds of the recording is matched against every track (at a range of tempos)
   to work out which track is playing and where in that track you are.
2. A path search follows each track's position through the recording. Normal playback is a
   straight line (position advances at your tempo); a beat jump is a sudden step forward.
   Repeating bars don't fool it: a step only counts as a jump if the audio after it does *not*
   also match a simple continuation of the old position.
3. For each jump, the tempo ratio and both positions are refined precisely, the missing
   section is rendered from the track file (Rubber Band time-stretch, pitch preserved), aligned
   to the recording on both sides, gain-matched, and spliced in with short crossfades.
4. Anything uncertain is left exactly as recorded and listed in the report.

## Setup (Windows)

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -e .
```

Optional, for .m4a/AAC tracks: `.venv\Scripts\pip install -e .[ffmpeg]`

## Clip mode (one recording per transition) - recommended

Record each transition as its own file instead of one long recording with beat jumps:

1. Get the outgoing track playing alone, settled (tempo, EQ, FX where you'll leave them), and
   start recording.
2. After **at least ~5 s (8+ is better)**, do the transition.
3. Once the incoming track is playing alone and settled, keep recording **another 5-8+ s**, then stop.
4. Load the next track, get to your next transition point (beat jump, cue, whatever) and repeat.

rekordbox writes a `.cue` next to each recording, which tells the tool which tracks are in it.

```
.venv\Scripts\python -m audioautofill clips E:\TESTS\my_set -s E:\Contents -o my_set_full.wav
```

The clips are used in file-name order. The tool identifies the track, position and tempo at the
head and tail of every clip, and renders the music between consecutive clips from the original
files - time-stretched to your tempo and tone/level-matched to your recording (EQ and master
colouring included). Before the first clip the first track plays from its start (`--no-intro` to
skip); after the last clip the last track plays to its end (`--no-outro`). A clip may contain more
than one transition, and consecutive clips may overlap.

It stops with an explanation instead of guessing if clips are out of order or missing, if an
edge isn't a single settled track, or if consecutive clips can't be joined.

## Beat-jump mode (one long recording)

### Usage

```
.venv\Scripts\python -m audioautofill "E:\TESTS\01 REC-2026-11-14.wav" -s E:\Contents -o my_set_full.wav
```

- **Track list:** by default the `.cue` file rekordbox writes next to each recording is used. It
  lists exactly the tracks played, in order, with times - the best input. Alternatively pass `-t`
  with a `.m3u8` History export, a folder of tracks, or single files (repeatable).
- `-s/--search <folder>`: where to look for tracks whose listed path doesn't exist (e.g. a USB
  export's `Contents` folder - files are found by name, including names the export truncated).
- `--dry-run` analyzes and writes the report without rendering audio.
- `--max-tempo-change 0.16` if you play tracks more than ±10% from their original tempo.

Outputs next to `-o`:

- the rebuilt set (same sample rate / bit depth as the recording),
- `*.labels.txt` - an Audacity label track marking every fill (File > Import > Labels), to
  quickly listen to each splice,
- `*.autofill.json` - timeline of which track was heard where, every jump found, and why any
  were skipped.

## Recording tips

- Record to WAV in rekordbox.
- Keep **Quantize** on and **Master Tempo** (key lock) on; don't use Key Shift.
- Only beat jump when **one track is audible** - finish the blend, pull the old channel down,
  then jump. Jumps during a blend are left alone (not filled).
- Leave a few bars (8 is plenty) of untouched playback before and after each jump, and don't
  touch EQ/FX/loops/tempo right around a jump.
- Export the rekordbox **History** for the session as `.m3u8` (right-click the history list >
  Export a playlist to a file).

## Limitations (v1)

- **Jumps during a blend** (two tracks audible) aren't filled and usually aren't even detected -
  that part of the set just stays shorter. Only jump when one track is playing.
- **Repeated sections:** if the music right around a jump repeats almost identically elsewhere
  in the track (long loops, a second identical drop), the tool may not be able to tell how far
  you jumped. If the uncertainty is big it leaves the jump as recorded and says so in the report;
  if it's small (a bar or two of identical drums) it fills anyway and notes it - that sounds
  seamless either way. A jump between *perfectly* identical sections can't be seen at all.
- **Backward jumps** (re-takes) are detected and reported but not spliced yet.
- Tracks must be local files (not streaming). Thresholds were tuned on synthetic music and may
  need adjusting after the first real recordings.

## Tests

`python tests/run_simulation.py [scenario ...]` generates synthetic tracks and simulated DJ sets
(using a different time-stretch algorithm than the app, standing in for rekordbox's key lock),
runs the tool, and compares the result with the true uncut set:

| scenario | what it checks |
|---|---|
| `single_r1` | original tempo: output nulls against the truth sample-for-sample |
| `full_set` | 4 tracks, wav/mp3/flac, key lock, transitions, tempo ramp, a loop-heavy track |
| `jump_in_mix` | a jump during a blend is left alone |
| `history_extras` | unplayed tracks in the History, non-ASCII filename, 48 kHz file |
| `long_blend` | 32-bar bass-swap blends followed by early jumps |
| `scale` | ~55 min set, 16 tracks, 32 jumps: speed and memory |

Current results: all pass; 37 of 38 fills within ±23 ms of the true length (mean 5.6 ms), the
other one flagged by the tool as a repeated section (off by exactly one identical bar). A 39-minute
recording with 16 tracks takes about 2 minutes and under 1 GB of memory.
