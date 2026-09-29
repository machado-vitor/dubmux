#!/usr/bin/env python3
"""batch_season - graft dubbed audio onto every episode of a season.

Pairs target episodes (SxxEyy in the filename) with donor files by episode
number, runs dubmux for each, and only replaces the original once dubmux's own
verify step reports the dub is in the file and in sync.

Nothing is overwritten in place: each episode is built into a temp file in the
same directory and moved over the original only on success, so an interrupted
or failed run always leaves a playable library behind.

  batch_season.py --target-dir DIR --donor-dir DIR [--season 1] [--dry-run] [--only 3,7]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# dubmux.py sits next to this file both in a checkout and once installed, and
# runs on the same interpreter (the one that has numpy/scipy).
PYTHON = sys.executable
DUBMUX = os.path.join(HERE, "dubmux.py")
VIDEO_EXT = (".mkv", ".mp4", ".avi", ".m4v")

# dubmux prints this when the finished file measures in sync; it is the only
# evidence that matters, so the batch trusts nothing else.
OK_MARK = "— OK"


def episode_number(name, season=None):
    """Episode number from a filename, or None.

    Handles both library naming (S01E07) and the numbered files BR release
    groups ship ("07 - Title.mp4"). A leading number is only accepted when it
    is the whole first token, so "1080p" or a year cannot be read as an episode.
    """
    m = re.search(r"[Ss](\d{1,2})[Ee](\d{1,3})", name)
    if m:
        if season is not None and int(m.group(1)) != season:
            return None
        return int(m.group(2))
    m = re.match(r"\s*(\d{1,3})\s*[-_. ]", name)
    if m:
        return int(m.group(1))
    return None


def index_dir(path, season=None):
    out = {}
    for fn in sorted(os.listdir(path)):
        if not fn.lower().endswith(VIDEO_EXT) or fn.startswith("."):
            continue
        ep = episode_number(fn, season)
        if ep is None:
            continue
        out.setdefault(ep, os.path.join(path, fn))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Run dubmux over every episode of a season, replacing each "
                    "original only after its sync check passes.")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--donor-dir", required=True)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--lang", default="por")
    ap.add_argument("--title", default="Brazilian Portuguese")
    ap.add_argument("--donor-lang", default="por")
    ap.add_argument("--only", default=None, help="comma-separated episode list, e.g. 3,7,12")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default="batch_report.json",
                    help="JSON report path (default: ./batch_report.json)")
    args = ap.parse_args()

    targets = index_dir(args.target_dir, args.season)
    donors = index_dir(args.donor_dir)
    only = {int(x) for x in args.only.split(",")} if args.only else None

    eps = sorted(set(targets) & set(donors))
    if only:
        eps = [e for e in eps if e in only]
    missing = sorted(set(targets) - set(donors))
    print(f"{len(targets)} targets, {len(donors)} donors, {len(eps)} pairs")
    if missing:
        print(f"no donor: {missing}")
    for e in eps:
        print(f"  E{e:02d}: {os.path.basename(targets[e])[:55]}"
              f"  <-  {os.path.basename(donors[e])[:40]}")
    if args.dry_run:
        return

    report, t0 = [], time.time()
    for e in eps:
        tgt, don = targets[e], donors[e]
        # already dubbed? skip, so re-running the batch is safe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream_tags=language", "-of", "csv=p=0", tgt],
            capture_output=True, text=True).stdout
        if args.lang[:2] in probe:
            print(f"\n=== E{e:02d}: already has {args.lang}, skipping")
            report.append({"ep": e, "status": "already_dubbed"})
            continue

        tmp = os.path.join(os.path.dirname(tgt), f".dubbing_E{e:02d}.mkv")
        print(f"\n=== E{e:02d} " + "=" * 50, flush=True)
        started = time.time()
        proc = subprocess.run(
            [PYTHON, DUBMUX, "mux", "--target", tgt, "--donor", don,
             "--donor-lang", args.donor_lang, "--out", tmp,
             "--lang", args.lang, "--title", args.title],
            capture_output=True, text=True)
        out = proc.stdout
        # keep the useful lines, drop mkvmerge's progress spam
        for line in out.splitlines():
            if not line.startswith(("Progress", "'")) and line.strip():
                print("   " + line[:110], flush=True)

        took = time.time() - started
        ok = proc.returncode == 0 and OK_MARK in out and os.path.exists(tmp)
        resid = re.search(r"worst residual\s+(\d+)\s*ms", out)
        entry = {"ep": e, "seconds": round(took),
                 "residual_ms": int(resid.group(1)) if resid else None,
                 "segments": out.count("  segment ")}
        if ok:
            os.replace(tmp, tgt)          # atomic within the same filesystem
            entry["status"] = "ok"
            print(f"   ==> E{e:02d} DONE in {took/60:.1f} min "
                  f"(residual {entry['residual_ms']} ms)", flush=True)
        else:
            if os.path.exists(tmp):
                os.unlink(tmp)
            entry["status"] = "failed"
            entry["error"] = (proc.stderr or out).strip().splitlines()[-1][:200] \
                if (proc.stderr or out).strip() else "no output"
            print(f"   ==> E{e:02d} FAILED: {entry['error']}", flush=True)
        report.append(entry)
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)

    good = [r for r in report if r["status"] == "ok"]
    bad = [r for r in report if r["status"] == "failed"]
    print(f"\n{'='*60}\n{len(good)} ok, {len(bad)} failed, "
          f"{len(report)-len(good)-len(bad)} skipped, "
          f"total {(time.time()-t0)/60:.0f} min")
    if good:
        worst = max(r["residual_ms"] or 0 for r in good)
        print(f"worst residual among ok: {worst} ms")
    if bad:
        print("failures: " + ", ".join(f"E{r['ep']:02d} ({r['error'][:60]})" for r in bad))
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
