#!/usr/bin/env python3
"""dubmux - graft a dubbed audio track from a donor release onto a better video release.

Measures the offset between donor and target audio with FFT cross-correlation on
several probe windows spread across the runtime, so a constant delay is told apart
from progressive drift (PAL/NTSC speed mismatch). Then muxes the donor track into a
copy of the target with mkvmerge (stream copy - no re-encode, no quality loss).

Usage:
  dubmux.py analyze  --target T.mkv --donor D.mkv [--donor-lang por]
  dubmux.py mux      --target T.mkv --donor D.mkv --out OUT.mkv [--lang por]
                     [--title "Brazilian Portuguese"] [--delay-ms N] [--force]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.signal import correlate, correlation_lags

SR = 8000               # analysis sample rate; plenty for speech alignment
WINDOW_S = 90           # seconds of audio per probe window
DEFAULT_PROBES = 5
ACCEPT_MS = 100         # lip-sync error a viewer starts to notice
# Spread allowed before a SINGLE delay is declared enough. This must stay below
# ACCEPT_MS: with a spread of S, the best a single delay can do is about S/2 at
# each extreme, so a loose value here green-lights a mux that verify then
# rejects. It was 250 ms once and shipped a +216 ms episode.
DRIFT_TOL_MS = 120
PAL_RATIO = 25.0 / 23.976023976
# Correlation feature. "phat" (GCC-PHAT: whitened cross-spectrum on the raw
# waveform) is used when BOTH sides are the same recording (anchor mode): it
# gives one needle-sharp peak, so confidence goes from ~1.5 to ~5 on a movie
# mix where the energy envelope of music + effects is too smooth to lock.
# Different performances (dub vs original, no anchor) share no waveform, only
# speech rhythm, so they keep the envelope. Set by main() after pick_anchor.
MATCH_MODE = {"v": "envelope"}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(path):
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration", "-show_streams", "-of", "json", path]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    audio = []
    for s in streams:
        if s.get("codec_type") != "audio":
            continue
        tags = s.get("tags") or {}
        audio.append({
            "index": s["index"],
            "audio_index": len(audio),
            "codec": s.get("codec_name"),
            "channels": s.get("channels"),
            "lang": (tags.get("language") or "und").lower(),
            "title": tags.get("title") or "",
            "bitrate": s.get("bit_rate"),
        })
    video = [s for s in streams if s.get("codec_type") == "video"]
    fps = None
    if video:
        rate = video[0].get("r_frame_rate") or "0/1"
        num, _, den = rate.partition("/")
        try:
            fps = float(num) / float(den or 1)
        except ZeroDivisionError:
            fps = None
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "audio": audio,
        "fps": fps,
    }


def pick_anchor(info_t, info_d, dub_track, target_track):
    """Find a track present in BOTH files to measure against.

    This is the single biggest accuracy win available, and it comes from the
    community (the dual-anchor pipelines): a BR/DE dub release almost always
    carries the ORIGINAL track next to the dubbed one. Cross-correlating
    original-vs-original compares the SAME recording, so the peak is sharp
    (confidence 4-10 instead of 2-3) and the offset curve becomes a clean step
    function. Matching a dubbed voice against the original compares different
    performances and only agrees via rhythm and effects, which is what produced
    ±100 ms of scatter and fake segment boundaries.

    Release groups frequently leave that second track tagged `und` instead of
    `eng`. Refusing to use it because of a missing tag throws away the whole
    advantage, so an untagged non-dub track counts as a candidate: the
    confidence check downstream is what actually decides whether it matched,
    and a wrong guess simply fails that check instead of producing a bad file.

    The time transform measured on the anchor is then applied to the DUB, since
    both sit on the donor's own timeline.
    """
    t_langs = {t["lang"] for t in info_t["audio"] if t["lang"] != "und"}
    others = [t for t in info_d["audio"]
              if t["audio_index"] != dub_track["audio_index"]]
    # 1) same language named on both sides
    cands = [t for t in others if t["lang"] in t_langs and t["lang"] != "und"]
    if cands:
        best = next((c for c in cands if c["lang"] == target_track["lang"]),
                    cands[0])
        tgt = next((t for t in info_t["audio"] if t["lang"] == best["lang"]),
                   target_track)
        return best, tgt
    # 2) untagged extra track: almost always the original, tag simply missing
    untagged = [t for t in others if t["lang"] == "und"]
    if untagged:
        return untagged[0], target_track
    return None, None


def mkvmerge_audio_ids(path):
    """mkvmerge track IDs of the audio tracks, in order.

    mkvmerge's -a/--sync/--language take its OWN track ID, which counts ALL
    tracks in the file, not audio-only. In an MP4 the video is id 0, so passing
    the audio-relative index makes mkvmerge say "a track with the ID 0 was
    requested but not found" (a WARNING, exit code 0) and silently produce an
    output with no dub at all. Always translate through this.
    """
    res = run(["mkvmerge", "-J", path])
    if res.returncode not in (0, 1):
        raise RuntimeError(f"mkvmerge -J failed: {res.stderr[:300]}")
    data = json.loads(res.stdout)
    return [t["id"] for t in data.get("tracks", []) if t.get("type") == "audio"]


def pick_track(info, lang=None, want_pt=False):
    """Choose which audio stream to read. Explicit lang wins, then Portuguese, then first."""
    tracks = info["audio"]
    if not tracks:
        return None
    if lang:
        for t in tracks:
            if t["lang"].startswith(lang[:2]):
                return t
    if want_pt:
        for t in tracks:
            if t["lang"] in ("por", "pt", "pob") or "portug" in t["title"].lower() \
               or "dublad" in t["title"].lower():
                return t
    return tracks[0]


def tempo_chain(pre, speed):
    """Filter chain: exact trim of `pre` INPUT seconds, then speed correction."""
    chain = [f"atrim=start={pre:.4f}", "asetpts=PTS-STARTPTS"]
    if abs(speed - 1.0) > 1e-6:
        chain.append(f"atempo={speed:.6f}")
    return ",".join(chain)


def read_window(path, audio_index, start, dur, speed=1.0):
    """Decode one mono window to a float32 numpy array.

    `speed` > 1 plays the donor faster; used to undo a PAL/NTSC speed mismatch.
    The seek is scaled too, so `start` always means "this instant of the movie".

    Seeking is done in two stages — a fast input seek to a few seconds early,
    then an exact output seek. Input seek alone lands on the nearest packet
    boundary, and since codecs have different frame sizes (DTS vs AC3) that
    rounding shows up as a fake constant delay of tens of milliseconds.
    """
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as fh:
        tmp = fh.name
    try:
        pre = min(5.0, start * speed)
        # The exact trim happens INSIDE the filter chain, before atempo. An
        # output-side -ss counts OUTPUT seconds, i.e. after atempo has already
        # stretched the audio, so it cut `pre` output seconds where `pre` input
        # seconds were meant: a PAL donor (4.3%) came out up to 213 ms off,
        # depending on how far from the file start the window was.
        cmd = ["ffmpeg", "-v", "error",
               "-ss", f"{start * speed - pre:.3f}", "-i", path,
               "-map", f"0:a:{audio_index}",
               "-filter:a", tempo_chain(pre, speed),
               "-t", f"{dur:.3f}"]
        cmd += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "-y", tmp]
        res = run(cmd)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip()[:300])
        return np.fromfile(tmp, dtype=np.float32)
    finally:
        os.unlink(tmp)


def envelope(sig):
    """Speech-ish energy envelope: rectify, smooth, remove DC, normalise."""
    if sig.size == 0:
        return sig
    env = np.abs(sig)
    win = SR // 50                      # 20 ms smoothing
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.convolve(env, kernel, mode="same")
    env -= env.mean()
    norm = np.linalg.norm(env)
    return env / norm if norm else env


def offset_at(target, t_idx, donor, d_idx, start, dur, search_s, guess=0.0,
              speed=1.0):
    """Offset in seconds to add to the donor so it lines up with the target.

    `guess` centres the search window: with progressive drift the offset at the
    end of a film can be a minute, far outside any sane fixed window, so each
    probe searches around what the previous ones predict.
    """
    raw_t = read_window(target, t_idx, start, dur)
    donor_start = start - guess
    pad = min(search_s, donor_start)
    raw_d = read_window(donor, d_idx, donor_start - pad,
                        dur + 2 * search_s, speed)
    if raw_t.size < SR or raw_d.size < SR:
        return None, 0.0
    if MATCH_MODE["v"] == "phat":
        tgt, don = raw_t, raw_d
        n = 1 << int(np.ceil(np.log2(don.size + tgt.size)))
        cross = np.fft.rfft(don, n) * np.conj(np.fft.rfft(tgt, n))
        cross /= np.abs(cross) + 1e-12
        corr = np.fft.irfft(cross, n)[:don.size - tgt.size + 1]
    else:
        tgt, don = envelope(raw_t), envelope(raw_d)
        corr = correlate(don, tgt, mode="valid", method="fft")
    lags = correlation_lags(don.size, tgt.size, mode="valid")
    best = int(np.argmax(corr))
    peak = float(corr[best])
    # Confidence = how much the winning lag beats the best RIVAL lag. Comparing
    # against the mean/std of the whole surface is not discriminative: a wrong
    # match also towers over the baseline. Comparing against the best peak
    # outside a 1 s guard band is, because a genuine match has exactly one spike.
    guard = SR  # 1 s on each side
    rival = np.concatenate([corr[:max(0, best - guard)], corr[best + guard:]])
    ref = float(np.max(rival)) if rival.size else 0.0
    # A non-positive rival max means the surface is degenerate (near-silence or
    # a window too short to have rivals) — that is NOT high confidence, so floor
    # it at zero instead of letting the ratio blow up to infinity.
    conf = peak / ref if ref > 0 and peak > 0 else 0.0
    donor_pos = (donor_start - pad) + lags[best] / SR
    return start - donor_pos, conf


def analyze(target, donor, t_idx, d_idx, probes, search_s, speed=1.0):
    info_t = probe(target)
    dur = info_t["duration"]
    # keep probes away from logos/credits at either end
    lo, hi = dur * 0.12, dur * 0.88
    points = np.linspace(lo, hi, probes)
    results = []
    for p in points:
        # predict this probe's offset by extrapolating the confident ones so far,
        # so a drifting donor stays inside the search window
        conf_pts = [(r["at"], r["offset"]) for r in results if r["conf"] >= 1.35]
        if len(conf_pts) >= 2:
            xs, ys = zip(*conf_pts)
            slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
            guess = float(slope * p + intercept)
        elif conf_pts:
            guess = conf_pts[-1][1]
        else:
            guess = 0.0
        off, conf = offset_at(target, t_idx, donor, d_idx, float(p), WINDOW_S,
                              search_s, guess, speed)
        if off is None:
            continue
        results.append({"at": float(p), "offset": off, "conf": conf})
        print(f"  {p/60:6.1f} min  ->  {off*1000:+9.0f} ms   (confidence {conf:5.1f})",
              flush=True)
    return results


def find_speed(target, donor, t_idx, d_idx, info_t, info_d, search_s):
    """Pick the donor playback speed that makes the offset stop drifting.

    A dub ripped from a PAL disc runs 4.27% fast; no fixed delay can fix that,
    so the speed has to be found BEFORE measuring the delay. Two probes far
    apart are enough: the right speed keeps both offsets equal.

    CAUTION, this is subtle: two probes cannot tell a genuine speed difference
    from a STEP between them. An episode whose offset jumps at an act break
    looks exactly like a slow drift to a two-point fit, and "correcting" it
    applies a real stretch to audio that had none, leaving every segment
    slightly wrong AND the steps still there. So a fitted speed is only
    accepted when a third, middle probe agrees with the straight line: a step
    breaks that test, a true speed error does not.

    Named ratios (PAL and its inverse) are exempt — they are physical, not
    fitted, and a mid probe is not needed to justify them.
    """
    dur = info_t["duration"]
    a, b = dur * 0.18, dur * 0.82
    mid = dur * 0.5
    cands = [1.0, PAL_RATIO, 1 / PAL_RATIO]
    if info_t["duration"] > 0 and info_d["duration"] > 0:
        ratio = info_d["duration"] / info_t["duration"]
        if 0.9 < ratio < 1.1 and abs(ratio - 1) > 0.002:
            cands.append(ratio)
    # measured drift at nominal speed -> exact candidate, only if it is drift
    oa, ca = offset_at(target, t_idx, donor, d_idx, a, WINDOW_S, search_s, 0.0, 1.0)
    if oa is not None and ca >= 1.2:
        ob, cb = offset_at(target, t_idx, donor, d_idx, b, WINDOW_S, search_s, oa, 1.0)
        om, cm = offset_at(target, t_idx, donor, d_idx, mid, WINDOW_S, search_s,
                           oa, 1.0)
        if ob is not None and cb >= 1.2:
            slope = (ob - oa) / (b - a)
            if 1e-6 < abs(slope) < 0.05:
                predicted = oa + slope * (mid - a)
                is_drift = (om is not None and cm >= 1.2
                            and abs(om - predicted) * 1000 <= 40)
                if is_drift:
                    cands.append(1.0 - slope)
                else:
                    print(f"  ({(ob-oa)*1000:+.0f} ms of change is not drift: "
                          f"mid-runtime probe is {abs((om or 0)-predicted)*1000:.0f} ms "
                          f"off the line — a step, not a speed error)")
    scored = []
    for sp in cands:
        oa, ca = offset_at(target, t_idx, donor, d_idx, a, WINDOW_S, search_s, 0.0, sp)
        if oa is None or ca < 1.2:
            continue
        ob, cb = offset_at(target, t_idx, donor, d_idx, b, WINDOW_S, search_s, oa, sp)
        if ob is None or cb < 1.2:
            continue
        spread = abs(ob - oa) * 1000
        print(f"  speed {sp:.6f}: offset varies {spread:7.0f} ms between "
              f"{a/60:.0f} and {b/60:.0f} min  (confidence {min(ca, cb):.1f})")
        scored.append({"sp": sp, "spread": spread, "conf": min(ca, cb), "oa": oa})
    if not scored:
        return 1.0, 1e9

    # Rank by spread, but a candidate is only accepted after the middle probe
    # confirms it (see docstring). Rejecting the winner must fall through to
    # the NEXT best candidate, not to 1.0: on a genuinely PAL donor the fitted
    # ratio can edge out the true PAL ratio on two points and, if rejection
    # meant giving up, the real answer sitting second in the list was thrown
    # away and every segment came out seconds wrong.
    named = (PAL_RATIO, 1 / PAL_RATIO)
    for cand in sorted(scored, key=lambda x: x["spread"]):
        sp = cand["sp"]
        # Confidence gates everything. A named ratio is exempt from the
        # step-vs-speed test, NOT from having to match: on a donor that is not
        # PAL at all, every candidate scores ~1.2 and the PAL ratio can still
        # win on spread, which then stretches the audio by 4% for nothing.
        # Demand a genuinely good lock before honouring any non-nominal speed.
        if abs(sp - 1.0) < 1e-6:
            return sp, cand["spread"]
        # A physical ratio whose two far-apart probes land on the SAME offset
        # (within 40 ms over ~15 min) is proof on its own: a wrong candidate
        # gives random offsets, not two equal ones. Without an anchor (dub vs
        # original) confidence tops out around 1.6, so the 2.0 gate below
        # alone threw away genuine PAL donors and left them seconds off.
        is_named = any(abs(sp - r) < 1e-9 for r in named)
        if is_named and cand["spread"] <= 40 and cand["conf"] >= 1.35:
            return sp, cand["spread"]
        if cand["conf"] < 2.0:
            print(f"  (ignoring {sp:.6f}: confidence {cand['conf']:.1f} "
                  f"too low to justify stretching the audio)")
            continue
        if any(abs(sp - r) < 1e-9 for r in named):
            return sp, cand["spread"]        # physical ratio, confidently matched
        om, cm = offset_at(target, t_idx, donor, d_idx, mid, WINDOW_S, search_s,
                           cand["oa"], sp)
        if om is None or cm < 1.2 or abs(om - cand["oa"]) * 1000 <= 60:
            return sp, cand["spread"]
        print(f"  (discarding {sp:.6f}: ends match but the middle "
              f"is {abs(om-cand['oa'])*1000:.0f} ms off — a step)")
    return 1.0, 1e9


def summarise(results, dur):
    good = [r for r in results if r["conf"] >= 1.35]
    verdict = {"usable": False, "delay_ms": None, "note": "", "speed_ratio": 1.0}
    if len(good) < 2:
        verdict["note"] = ("Could not match the audio: the two tracks are probably "
                           "not from the same cut of the film.")
        return verdict
    offs = np.array([r["offset"] for r in good])
    spread_ms = (offs.max() - offs.min()) * 1000
    median_ms = float(np.median(offs) * 1000)
    verdict["spread_ms"] = spread_ms
    verdict["delay_ms"] = median_ms
    if spread_ms <= DRIFT_TOL_MS:
        verdict["usable"] = True
        verdict["note"] = f"Constant delay of {median_ms:+.0f} ms. Can be muxed directly."
        return verdict
    # drift: fit offset against position to get the speed ratio. Same sign rule
    # as find_speed — the corrective speed is BELOW one when the offset grows.
    xs = np.array([r["at"] for r in good])
    slope, _ = np.polyfit(xs, offs, 1)
    ratio = 1.0 - slope
    verdict["speed_ratio"] = float(ratio)
    if abs(ratio - PAL_RATIO) < 0.002 or abs(ratio - 1 / PAL_RATIO) < 0.002:
        verdict["note"] = (f"PAL/NTSC drift (factor {ratio:.5f}). The audio speed must "
                           "be corrected before muxing.")
    else:
        verdict["note"] = (f"The offset changes across the runtime ({spread_ms:.0f} ms of "
                           f"spread, factor {ratio:.5f}) — different edits.")
    return verdict


def measure_curve(target, donor, t_idx, d_idx, dur, search_s, speed=1.0,
                  step_s=20.0, win_s=45.0):
    """Dense offset-vs-time curve over the whole runtime.

    Used to tell a genuine STEP (different edit: a scene, bumper or act break
    only one release carries) apart from drift. A step cannot be fixed by any
    single delay or speed, so it has to be measured before deciding.

    Window length is the noise knob. Without an anchor the scatter at 25 s is
    ~±100 ms and only 90 s brings it near 15 ms. Measuring on an ANCHOR track
    (same recording on both sides) the correlation peak is sharp, so 45 s is
    already quiet — which is what makes a 20 s step affordable and lets real
    act-break steps be located to within a few seconds.

    Probes run from 1% to 99%: dubs are frequently re-timed right at the head
    and tail (different distributor logos, cut credits), and those are exactly
    the regions that a conservative 3%-97% sweep leaves unmeasured.
    """
    pts = []
    guess = 0.0
    # First probe has no prior: give it a wide search so a dub that starts
    # seconds off (different logos/openings) is found at all. Later probes
    # inherit the previous offset and can stay narrow.
    first = True
    for t in np.arange(dur * 0.01, dur * 0.99, step_s):
        off, conf = offset_at(target, t_idx, donor, d_idx, float(t), win_s,
                              max(search_s, 90.0) if first else search_s,
                              guess, speed)
        first = False
        if off is None:
            continue
        if conf >= 1.35:
            guess = off
            pts.append({"at": float(t), "offset": off, "conf": conf})
    return pts


def robust_local_offsets(pts, dur, block_s=15.0, half_win_s=40.0,
                         outlier_ms=350.0):
    """Offset for each block of the runtime, estimated from its neighbourhood.

    Segmenting on a hard tolerance was wrong: a real dub does not move in clean
    plateaus, it wanders (re-timed per reel, per act, sometimes per scene), and
    probe noise of tens of ms then invents or hides boundaries depending on the
    threshold. Taking a MEDIAN over a sliding neighbourhood needs no threshold:
    it follows genuine movement and ignores single bad probes, which is what
    makes this work on episodes where plateau detection failed.

    `half_win_s` trades noise rejection against step fidelity: too wide and a
    step gets smeared across its neighbourhood, leaving the blocks on both
    sides wrong (a 75 s window left a +406 ms error at one act break). With an
    anchor the per-probe noise is small, so the window can stay narrow.
    """
    if not pts:
        return []
    at = np.array([p["at"] for p in pts])
    off = np.array([p["offset"] for p in pts])
    # Outlier rejection must be LOCAL. Comparing each probe to the GLOBAL
    # median assumes the offset is roughly constant, which is exactly what is
    # false here: an episode can legitimately step by seconds across its
    # runtime, and a global test then throws away whole true plateaus at the
    # ends and collapses everything into one bogus segment. Compare each probe
    # against its own neighbours instead, where only genuine mis-locks stand out.
    order = np.argsort(at)
    at, off = at[order], off[order]
    keep = np.ones(at.size, dtype=bool)
    for i in range(at.size):
        near = np.abs(at - at[i]) <= half_win_s
        near[i] = False
        if near.sum() < 2:
            continue
        local_med = np.median(off[near])
        local_mad = np.median(np.abs(off[near] - local_med))
        tol = max(outlier_ms / 1000.0, 6 * local_mad + 0.05)
        keep[i] = abs(off[i] - local_med) <= tol
    at, off = at[keep], off[keep]
    if at.size == 0:
        return []
    blocks = []
    for start in np.arange(0.0, dur, block_s):
        centre = start + block_s / 2
        sel = np.abs(at - centre) <= half_win_s
        if not sel.any():                      # edges: fall back to nearest
            sel = np.abs(at - centre) <= half_win_s * 3
        if not sel.any():
            blocks.append({"start": float(start), "offset": float(np.median(off))})
            continue
        blocks.append({"start": float(start),
                       "offset": float(np.median(off[sel]))})
    return blocks


def refine_boundary(target, donor, t_idx, d_idx, lo, hi, off_before, off_after,
                    search_s, speed=1.0, win_s=12.0, iters=4):
    """Binary-search the exact instant where the offset jumps.

    Block granularity puts a boundary within one block (±15 s); a splice placed
    that far from the true cut leaves a stretch of audio carrying the WRONG
    offset, which is heard as a lip-sync error even though every segment value
    is correct. Short windows are too noisy to trust on their own, so this asks
    a cheaper question instead: at instant t, does a short probe match the
    BEFORE offset or the AFTER offset? That comparison stays reliable at 12 s
    because both hypotheses are known.
    """
    for _ in range(iters):
        if hi - lo < 2.0:
            break
        mid = (lo + hi) / 2
        tgt = envelope(read_window(target, t_idx, mid, win_s))
        if tgt.size < SR:
            break
        scores = []
        for off in (off_before, off_after):
            don = envelope(read_window(donor, d_idx, mid - off, win_s, speed))
            n = min(tgt.size, don.size)
            scores.append(float(np.dot(tgt[:n], don[:n])) if n >= SR else -1.0)
        if scores[0] >= scores[1]:
            lo = mid           # still on the old offset
        else:
            hi = mid           # already on the new one
    return (lo + hi) / 2


def segment_curve(pts, dur, tol_ms=25.0, block_s=15.0):
    """Merge the per-block estimates into runs that share one offset.

    Merging keeps the number of splices small (each splice is a real change in
    the dub's timing), while the per-block median underneath means a boundary
    only appears where the offset genuinely moved.

    `tol_ms` is tight because with an anchor track the measurement scatter is a
    few ms and real steps are tens to hundreds of ms — the two do not overlap.
    Without an anchor the scatter alone reaches ±100 ms and no threshold
    separates them, which is the real reason to always prefer an anchor.
    """
    blocks = robust_local_offsets(pts, dur, block_s=block_s)
    if not blocks:
        return []
    segs = [{"start": blocks[0]["start"], "offset": blocks[0]["offset"],
             "probes": 1}]
    for b in blocks[1:]:
        if abs(b["offset"] - segs[-1]["offset"]) * 1000 <= tol_ms:
            # absorb: keep the running mean so a slow ramp does not bias to the
            # first block's value
            n = segs[-1]["probes"]
            segs[-1]["offset"] = (segs[-1]["offset"] * n + b["offset"]) / (n + 1)
            segs[-1]["probes"] = n + 1
        else:
            segs.append({"start": b["start"], "offset": b["offset"], "probes": 1})
    for i, s in enumerate(segs):
        s["end"] = segs[i + 1]["start"] if i + 1 < len(segs) else None
    return segs


def build_piecewise_audio(donor, d_idx, segs, dur, out_path, speed=1.0,
                          channels=None, quiet=False):
    """Re-time the donor track segment by segment so every part lands right.

    This is what Sushi does: when the two releases are edited differently, each
    stretch of audio gets its own shift and the pieces are butted together.
    Cuts fall at act breaks, where both sides are near-silent, so the splice is
    inaudible.
    """
    tmpdir = tempfile.mkdtemp(prefix="dubmux_pw_")
    parts, listing = [], os.path.join(tmpdir, "list.txt")
    try:
        for i, s in enumerate(segs):
            seg_start, seg_end = s["start"], s["end"] if s["end"] else dur
            length = seg_end - seg_start
            if length <= 0.01:
                continue
            src = (seg_start - s["offset"]) * speed
            part = os.path.join(tmpdir, f"p{i:03d}.wav")
            lead = 0.0
            if src < 0:                      # donor starts later than target
                lead, src = -src / speed, 0.0
            pre = min(5.0, src)
            # trim before atempo, see read_window
            cmd = ["ffmpeg", "-v", "error", "-ss", f"{src - pre:.3f}", "-i", donor,
                   "-map", f"0:a:{d_idx}"]
            filt = [tempo_chain(pre, speed)]
            if lead > 0.001:
                filt.append(f"adelay=delays={int(lead*1000)}:all=1")
            # pad the tail so the part is ALWAYS exactly `length` long: parts are
            # butted end to end, so any part that comes up short slides every
            # later segment early by that much.
            filt.append("apad")
            cmd += ["-filter:a", ",".join(filt)]
            # -t counts OUTPUT seconds, i.e. after atempo/adelay — so it is the
            # target-timeline length, never the source amount.
            cmd += ["-t", f"{length:.3f}", "-c:a", "pcm_s16le", "-y", part]
            res = run(cmd)
            if res.returncode != 0:
                raise RuntimeError(res.stderr[:300])
            got = probe(part)["duration"]
            if abs(got - length) > 0.05:
                raise RuntimeError(f"segment {i} came out {got:.2f}s, expected {length:.2f}s")
            parts.append(part)
            if not quiet:
                print(f"  segment {seg_start/60:5.1f}-{seg_end/60:5.1f} min  "
                      f"offset {s['offset']*1000:+6.0f} ms")
        with open(listing, "w") as fh:
            for p in parts:
                fh.write(f"file '{p}'\n")
        # stereo dubs are already lossy; FLAC avoids a second generation of loss
        if out_path.endswith(".wav"):
            codec = ["-c:a", "pcm_s16le"]
        else:
            codec = (["-c:a", "flac"] if (channels or 2) <= 2
                     else ["-c:a", "ac3", "-b:a", "640k"])
        res = run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                   "-i", listing, "-ac", str(channels or 2)] + codec +
                  ["-y", out_path])
        if res.returncode != 0:
            raise RuntimeError(res.stderr[:300])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path


def mux(target, donor, out, d_idx, delay_ms, lang, title, default=False,
        speed=1.0, workdir=None, channels=None):
    if not shutil.which("mkvmerge"):
        sys.exit("mkvmerge not found (install mkvtoolnix)")
    donor_src = donor
    tmp_audio = None
    if abs(speed - 1.0) > 1e-6:
        # Speed has to be corrected by re-encoding the audio; mkvmerge can only
        # shift, not stretch. Only the dub is touched — video stays stream-copied.
        tmp_audio = os.path.join(workdir or os.path.dirname(out) or ".",
                                 ".dubmux_speed.mka")
        # A stereo dub is usually already a lossy 128-192k AAC/AC3; re-encoding
        # it to lossy again stacks a second generation of loss for nothing, and
        # FLAC stereo costs little. Multichannel stays AC3 for player compat.
        src_ch = channels or 2
        codec = ["-c:a", "flac"] if src_ch <= 2 else ["-c:a", "ac3", "-b:a", "640k"]
        print(f"correcting audio speed (factor {speed:.6f}, {codec[1]})...")
        res = run(["ffmpeg", "-v", "error", "-i", donor, "-map", f"0:a:{d_idx}",
                   "-filter:a", f"atempo={speed:.6f}"] + codec +
                  ["-y", tmp_audio])
        if res.returncode != 0:
            sys.exit(f"ffmpeg failed: {res.stderr[:300]}")
        donor_src, d_idx = tmp_audio, 0
    # Translate the audio-relative index into mkvmerge's own track ID.
    audio_ids = mkvmerge_audio_ids(donor_src)
    if d_idx >= len(audio_ids):
        sys.exit(f"donor has no audio track a:{d_idx}")
    donor_track_id = audio_ids[d_idx]
    cmd = ["mkvmerge", "-o", out, target,
           "--language", f"{donor_track_id}:{lang}",
           "--track-name", f"{donor_track_id}:{title}",
           "--sync", f"{donor_track_id}:{int(round(delay_ms))}",
           "--default-track-flag", f"{donor_track_id}:{'yes' if default else 'no'}",
           "-a", str(donor_track_id), "-D", "-S", "-B", "-M", "--no-chapters",
           donor_src]
    print(" ".join(cmd))
    res = subprocess.run(cmd)
    if tmp_audio and os.path.exists(tmp_audio):
        os.unlink(tmp_audio)
    if res.returncode not in (0, 1):
        return res.returncode
    # mkvmerge exits 0 even when it only WARNED that the requested track was
    # missing, so trust the output file, not the exit code.
    out_langs = [t["lang"] for t in probe(out)["audio"]]
    if not any(l.startswith(lang[:2]) for l in out_langs):
        sys.exit(f"mux did not add the {lang} track: output has {out_langs}")
    print(f"ok: output audio tracks {out_langs}")
    return 0


def verify_transform(target, donor, segs, dur, t_idx, d_idx, speed=1.0,
                     probes=5, search_s=15.0, accept_ms=ACCEPT_MS):
    """Prove the time transform is right, by applying it to the ANCHOR.

    The dub and the anchor share the donor's timeline, so a transform that puts
    the anchor exactly on top of the target's own anchor puts the dub there
    too. Building a re-timed anchor and measuring it costs one extra pass and
    is the only check that is both precise (same-recording correlation) and
    actually about the transform being shipped.

    Measuring the dubbed track against the original inside the output would be
    cheaper, but it carries ±100 ms of performance scatter and would reject
    correct files — a test that fails on good output is worse than no test.
    """
    # Unique temp name: several dubmux runs can be in flight at once (batch in
    # the background while a single episode is retested), and a fixed name in
    # /tmp makes them delete each other's file mid-read — which surfaces as a
    # bogus "No such file or directory" on an unrelated episode.
    fd, tmp = tempfile.mkstemp(prefix=".dubmux_verify_", suffix=".wav")
    os.close(fd)
    try:
        build_piecewise_audio(donor, d_idx, segs, dur, tmp, speed,
                              channels=1, quiet=True)
        worst, n = 0.0, 0
        for p in np.linspace(dur * 0.1, dur * 0.9, probes):
            off, conf = offset_at(target, t_idx, tmp, 0, float(p), 60.0,
                                  search_s)
            if off is None or conf < 1.35:
                continue
            n += 1
            worst = max(worst, abs(off * 1000))
            print(f"  {p/60:6.1f} min  residual {off*1000:+6.0f} ms  "
                  f"(confidence {conf:4.1f})")
        if not n:
            print("  could not measure the residual")
            return False
        ok = worst <= accept_ms
        print(f"  ==> worst residual {worst:.0f} ms — "
              f"{'OK' if ok else 'STILL OUT OF SYNC'}")
        return ok
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def verify(out_path, target, lang, probes=5, search_s=15.0, accept_ms=ACCEPT_MS):
    """Structural check on the finished file: is the dub actually in there?

    Timing is proven by verify_transform (anchor path) or by measuring dub vs
    original here when there is no anchor.
    """
    info = probe(out_path)
    tracks = info["audio"]
    dub = next((t for t in tracks if t["lang"].startswith(lang[:2])), None)
    if dub is None:
        print(f"  FAILED: no {lang} track in the output ({[t['lang'] for t in tracks]})")
        return False
    ref = next((t for t in tracks if t["audio_index"] != dub["audio_index"]), None)
    if ref is None:
        print("  (no original track in the output to compare against)")
        return True
    dur = info["duration"]
    worst, vals = 0.0, []
    for p in np.linspace(dur * 0.1, dur * 0.9, probes):
        off, conf = offset_at(out_path, ref["audio_index"], out_path,
                              dub["audio_index"], float(p), 60.0, search_s)
        if off is None or conf < 1.35:
            continue
        vals.append(off * 1000)
        worst = max(worst, abs(off * 1000))
        print(f"  {p/60:6.1f} min  residual {off*1000:+6.0f} ms  (confidence {conf:4.1f})")
    if not vals:
        print("  could not measure the residual (tracks too different)")
        return False
    # ~100 ms is around where lip-sync error starts being noticed; audio LATE
    # (positive) is tolerated slightly less than early, but keep one threshold.
    ok = worst <= accept_ms
    print(f"  ==> worst residual {worst:.0f} ms — {'OK' if ok else 'STILL OUT OF SYNC'}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Graft a dubbed audio track from a donor release onto a better "
                    "video release, measuring and correcting delay, drift and edits.")
    ap.add_argument("action", choices=["analyze", "mux"])
    ap.add_argument("--target", required=True, help="the good release (its video is kept)")
    ap.add_argument("--donor", required=True, help="the release the dubbed audio is taken from")
    ap.add_argument("--target-lang", default=None)
    ap.add_argument("--donor-lang", default=None)
    ap.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    ap.add_argument("--search", type=float, default=30.0,
                    help="search window in seconds on each side")
    ap.add_argument("--out")
    ap.add_argument("--lang", default="por")
    ap.add_argument("--title", default="Brazilian Portuguese",
                    help="track name of the grafted audio")
    ap.add_argument("--delay-ms", type=float, default=None)
    ap.add_argument("--speed", type=float, default=None,
                    help="donor speed factor (default: detect)")
    ap.add_argument("--default-track", action="store_true")
    ap.add_argument("--no-piecewise", action="store_true",
                    help="do not fall back to per-segment resync when the offset varies")
    ap.add_argument("--force-piecewise", action="store_true",
                    help="go straight to per-segment resync")
    ap.add_argument("--no-anchor", action="store_true",
                    help="do not use a shared track as the measurement anchor")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--accept-ms", type=float, default=ACCEPT_MS,
                    help="maximum residual accepted by the final check")
    args = ap.parse_args()

    info_t, info_d = probe(args.target), probe(args.donor)
    t = pick_track(info_t, args.target_lang)
    d = pick_track(info_d, args.donor_lang, want_pt=True)
    if not t or not d:
        sys.exit("one of the files has no audio track")

    print(f"target: {os.path.basename(args.target)}")
    print(f"        {info_t['duration']/60:.1f} min, {(info_t['fps'] or 0):.3f} fps, "
          f"track a:{t['audio_index']} {t['lang']} {t['codec']} {t['channels']}ch")
    print(f"donor : {os.path.basename(args.donor)}")
    print(f"        {info_d['duration']/60:.1f} min, "
          f"{(info_d['fps'] or 0):.3f} fps, track a:{d['audio_index']} {d['lang']} "
          f"{d['codec']} {d['channels']}ch")
    dur_diff = abs(info_t["duration"] - info_d["duration"])
    print(f"duration difference: {dur_diff:.1f} s")

    # Measure on a language both files share (see pick_anchor); fall back to
    # dub-vs-original only when the donor carries no shared track.
    a_don, a_tgt = (None, None) if args.no_anchor else \
        pick_anchor(info_t, info_d, d, t)
    if a_don:
        m_t_idx, m_d_idx = a_tgt["audio_index"], a_don["audio_index"]
        MATCH_MODE["v"] = "phat"
        print(f"anchor: {a_don['lang']} in both "
              f"(target a:{m_t_idx} vs donor a:{m_d_idx}) — measuring on it (GCC-PHAT)")
    else:
        m_t_idx, m_d_idx = t["audio_index"], d["audio_index"]
        print("anchor: no shared track, measuring dub vs original")

    speed = args.speed
    if speed is None:
        print("\nsearching for the donor's playback speed:")
        speed, spread = find_speed(args.target, args.donor, m_t_idx,
                                   m_d_idx, info_t, info_d, args.search)
        print(f"==> chosen speed: {speed:.6f}")

    print("\nmeasuring the offset:")
    results = analyze(args.target, args.donor, m_t_idx, m_d_idx,
                      args.probes, args.search, speed)
    v = summarise(results, info_t["duration"])
    if abs(speed - 1.0) > 1e-6 and v["usable"]:
        v["note"] += f" (with the audio sped up by {speed:.6f})"
    print("\n==> " + v["note"])

    if args.action == "mux":
        if not args.out:
            sys.exit("--out is required")
        delay = args.delay_ms if args.delay_ms is not None else v["delay_ms"]

        def do_piecewise():
            """Per-segment resync: the fix when one delay cannot serve the whole
            runtime (different edits, dub re-timed per act)."""
            print("\nmeasuring the full curve for per-segment resync:")
            pts = measure_curve(args.target, args.donor, m_t_idx,
                                m_d_idx, info_t["duration"],
                                args.search, speed)
            segs = segment_curve(pts, info_t["duration"])
            if not segs:
                sys.exit("could not match the audio, not even per segment")
            # Pin each boundary to the real cut instead of a block edge.
            for i in range(1, len(segs)):
                prev, cur = segs[i - 1], segs[i]
                exact = refine_boundary(args.target, args.donor, m_t_idx,
                                        m_d_idx, max(prev["start"],
                                                     cur["start"] - 20.0),
                                        cur["start"] + 20.0,
                                        prev["offset"], cur["offset"],
                                        args.search, speed)
                cur["start"] = exact
                prev["end"] = exact
            print(f"\n{len(segs)} segment(s):")
            # unique per run, for the same reason as in verify_transform
            fd_pw, tmp_pw = tempfile.mkstemp(
                prefix=".dubmux_piecewise_", suffix=".mka",
                dir=os.path.dirname(args.out) or ".")
            os.close(fd_pw)
            build_piecewise_audio(args.donor, d["audio_index"], segs,
                                  info_t["duration"], tmp_pw, speed,
                                  d.get("channels"))
            # the re-timed track is already in target time: no shift, no speed
            rc = mux(args.target, tmp_pw, args.out, 0, 0.0, args.lang,
                     args.title, args.default_track, 1.0,
                     channels=d.get("channels"))
            if os.path.exists(tmp_pw):
                os.unlink(tmp_pw)
            if rc != 0:
                return rc
            print("\nchecking the result:")
            if a_don:
                # anchor path: prove the transform, then confirm the track landed
                ok = verify_transform(args.target, args.donor, segs,
                                      info_t["duration"], m_t_idx, m_d_idx,
                                      speed, accept_ms=args.accept_ms)
                langs = [x["lang"] for x in probe(args.out)["audio"]]
                if args.lang[:2] not in " ".join(langs):
                    print(f"  FAILED: no {args.lang} track in the output ({langs})")
                    ok = False
            else:
                ok = verify(args.out, args.target, args.lang,
                            accept_ms=args.accept_ms)
            if not ok and os.path.exists(args.out):
                # Never leave a file that failed its own check where a caller
                # (or a person) could pick it up as a finished result.
                os.unlink(args.out)
                print("  rejected output deleted")
            return 0 if ok else 1

        # With an anchor the measurement is precise enough that the piecewise
        # transform is both correct and cheap to trust, and it degenerates to a
        # single shift when the curve has one step — so there is no reason to
        # gamble on the single-delay path first.
        if a_don and not args.no_piecewise:
            sys.exit(do_piecewise())

        # Single delay first when the measurement says one delay fits: it is
        # far cheaper and touches the audio least.
        if v["usable"] and not args.force_piecewise:
            print(f"\nmuxing with delay {delay:+.0f} ms")
            rc = mux(args.target, args.donor, args.out, d["audio_index"], delay,
                     args.lang, args.title, args.default_track, speed,
                     channels=d.get("channels"))
            if rc == 0:
                print("\nchecking the result:")
                if verify(args.out, args.target, args.lang, accept_ms=args.accept_ms):
                    sys.exit(0)
                # The prediction was optimistic. Do NOT ship it: measuring the
                # finished file is the authority, so fall back instead of
                # leaving a file that verify just rejected.
                if args.no_piecewise:
                    sys.exit("a single delay was not enough and piecewise is disabled")
                print("\na single delay was not enough — falling back to per-segment resync")
                sys.exit(do_piecewise())
            sys.exit(rc)

        if args.no_piecewise:
            if delay is None or not args.force:
                sys.exit("refusing to mux: alignment is not reliable "
                         "(use --force to insist)")
            print(f"\nmuxing with delay {delay:+.0f} ms (--force)")
            rc = mux(args.target, args.donor, args.out, d["audio_index"], delay,
                     args.lang, args.title, args.default_track, speed,
                     channels=d.get("channels"))
            if rc == 0:
                print("\nchecking the result:")
                verify(args.out, args.target, args.lang,
                       accept_ms=args.accept_ms)
            sys.exit(rc)

        sys.exit(do_piecewise())


if __name__ == "__main__":
    main()
