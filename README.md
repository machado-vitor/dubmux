# dubmux

Graft a dubbed audio track from one release of a film or episode onto a better
video release of the same title, in sync, without re-encoding the video.

Typical case: you have a high-quality remux with only the original audio, and a
lower-quality release that carries the dub you want. dubmux measures how the two
timelines relate, re-times the dub to match, muxes it into a copy of the good
file, and then **checks the finished file** before keeping it.

## What it handles

- **Constant delay** — different intros, logos or padding. Fixed with a plain
  mkvmerge sync, no audio touched.
- **Speed drift** — PAL (25 fps) vs NTSC/film (23.976 fps) releases run ~4.3%
  apart. Detected and corrected with `atempo` before muxing.
- **Different edits** — scenes, bumpers or act breaks present in only one
  release. The offset curve is measured over the whole runtime, split into
  segments, each boundary is binary-searched to the real cut, and the dub is
  rebuilt piece by piece (the approach Sushi popularised).
- **Anchor tracks** — when the donor also carries the original-language track
  (most dub releases do), measurement runs on original-vs-original with
  GCC-PHAT, which gives a needle-sharp correlation peak instead of the fuzzy
  match you get comparing a dub against a different performance.

A result that fails its own sync check is deleted, never left behind.

## Requirements

- Python 3.9+ with `numpy` and `scipy`
- `ffmpeg` / `ffprobe`
- `mkvmerge` (MKVToolNix)

```sh
brew install ffmpeg mkvtoolnix       # macOS
sudo apt install ffmpeg mkvtoolnix   # Debian/Ubuntu

pip install git+https://github.com/machado-vitor/dubmux
# or, from a checkout: pip install numpy scipy
```

## Usage

Inspect first — `analyze` never writes anything:

```sh
dubmux analyze --target REMUX.mkv --donor DUBBED.mkv
```

Then mux:

```sh
dubmux mux --target REMUX.mkv --donor DUBBED.mkv --out REMUX.dub.mkv \
    --lang por --title "Brazilian Portuguese"
```

Useful options (`dubmux --help` for all):

| Option | Meaning |
|---|---|
| `--donor-lang`, `--target-lang` | which audio track to read on each side |
| `--delay-ms N` / `--speed F` | skip detection and force a value |
| `--no-anchor` | measure dub vs original even if a shared track exists |
| `--force-piecewise` / `--no-piecewise` | always / never resync per segment |
| `--accept-ms N` | worst residual the final check accepts (default 100) |
| `--default-track` | mark the grafted track as default |

### Whole seasons

`batch_season` pairs episodes by `SxxEyy` (or a leading episode number) and runs
dubmux on each. Every episode is built into a temp file and replaces the
original only after its sync check passes, so an interrupted run always leaves
a playable library. Always look at the pairing with `--dry-run` first:

```sh
batch_season --target-dir "Show/Season 01" --donor-dir dubs/ --season 1 --dry-run
batch_season --target-dir "Show/Season 01" --donor-dir dubs/ --season 1
```

Episodes that already have a track in `--lang` are skipped, so re-running is
safe.

## How the sync check works

With an anchor, the exact time transform that was shipped is applied to the
donor's original track and correlated against the target's original track:
same recording on both sides, so the residual is measured to a few ms. Without
an anchor, the dub is measured against the original inside the output file.
Either way the verdict line reads `worst residual N ms — OK` or
`STILL OUT OF SYNC`.

## License

MIT
