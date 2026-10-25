#!/usr/bin/env python3
"""
Turns vocal recordings into a clean training set for an RVC voice model (see docs/TRAINING.md).

    python3 tools/training/prepare_dataset.py vocals/ more_vocals/take3.wav --out dataset --name "My Voice"

For every input file (folders are searched recursively; anything ffmpeg can decode works):

  1. decode, mix to mono (and warn when the two channels of a stereo file differ: a double-tracked or
     stereo-effected vocal is not a single voice), remove DC, high-pass at 60 Hz
  2. measure the noise floor, the singing level, clipping, how periodic the voice is (a single voice is
     much more periodic than stacked harmonies or doubles) and how fast phrase ends decay (reverb)
  3. slice at silences into clips of 3 - 10 s (short phrases are joined, long pauses inside a clip are
     shortened, long phrases are cut at their quietest point)
  4. drop clips that are clipped, noisy, polyphonic/doubled or reverberant, and files that are wet overall
  5. trim silence (keeping 80 ms), loudness-normalise (EBU R128 / BS.1770, default -20 LUFS, peaks at most
     -1 dBFS), resample (default 40 kHz, --sr) and write 16-bit WAV files
  6. write report.json and report.md: minutes per voice, clips, pitch range, and every rejected second
     with the reason

Several voices in one run: --manifest voices.csv with the columns `path,voice` (a path may be a glob);
each voice gets its own sub-folder (cluster_singers.py writes such a file). --zip also writes
<voice>.zip for uploading to the training notebook.

Requirements: Python 3.9+, numpy, scipy, soundfile; ffmpeg on the PATH for MP3/AAC/... input.
Only use recordings you made yourself or have explicit permission to use for training a voice model.
"""

import argparse
import csv
import fnmatch
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aif", ".aiff", ".wma", ".webm"}
ANALYSIS_SR = 16000
HOP_S = 0.01            # analysis frame hop (10 ms)

# Rejection thresholds (calibrated on dry vs. wet and single vs. stacked commercial vocal stems; see
# docs/TRAINING.md). All can be changed on the command line.
DEFAULTS = dict(
    min_len=3.0, min_keep=1.5, max_len=10.0, min_snr=20.0, max_clip_pct=0.05, min_periodicity=0.85,
    min_file_periodicity=0.80, max_tail_ms=300.0, max_file_tail_ms=200.0, target_lufs=-20.0, peak_dbfs=-1.0,
)


def log(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------------------------------------- I/O

def decode(path):
    """Returns (float32 array [channels, samples], sample rate)."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.T.copy(), sr
    except Exception:
        pass
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("cannot decode %s: install ffmpeg (or convert it to WAV)" % path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=sample_rate,channels", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
    try:
        sr, channels = [int(v) for v in probe.stdout.strip().split(",")[:2]]
    except ValueError:
        raise RuntimeError("cannot read %s: %s" % (path, probe.stderr.strip() or "not an audio file"))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    return x.reshape(-1, channels).T.copy(), sr


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(x, sr_out // g, sr_in // g, window=("kaiser", 8.0)).astype(np.float32)


def find_inputs(paths, includes, excludes):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in sorted(names)]
        else:
            files += sorted(glob.glob(p)) or [p]

    def matches(name, patterns):
        return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in patterns)

    result = []
    for f in files:
        name = os.path.basename(f)
        if os.path.splitext(name)[1].lower() not in AUDIO_EXTENSIONS:
            continue
        if includes and not matches(name, includes):
            continue
        if excludes and matches(name, excludes):
            continue
        result.append(f)
    return sorted(dict.fromkeys(result))


# ------------------------------------------------------------------------------------------ analysis

def frame_db(x, sr, hop_s=HOP_S, win_s=0.03):
    hop = int(round(sr * hop_s))
    win = int(round(sr * win_s))
    n = max(1, int(math.ceil(len(x) / hop)))
    padded = np.pad(x, (win // 2, win + hop))
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    rms = np.sqrt(np.mean(padded[idx] ** 2, axis=1) + 1e-20)
    return 20 * np.log10(rms + 1e-12)


def yin_frames(x16, hop=160, win=512, fmin=60.0, fmax=1100.0):
    """Vectorised YIN on 16 kHz audio. Returns (f0 Hz, aperiodicity 0..1) per 10 ms frame."""
    tau_max = int(ANALYSIS_SR / fmin)
    tau_min = int(ANALYSIS_SR / fmax)
    length = win + tau_max
    n = max(1, int(math.ceil(len(x16) / hop)))
    padded = np.pad(x16, (win // 2, length + hop))
    f0 = np.zeros(n, np.float32)
    ap = np.ones(n, np.float32)
    nfft = 1 << int(math.ceil(math.log2(length + win)))
    for start in range(0, n, 2048):                         # in blocks to bound memory
        stop = min(n, start + 2048)
        idx = np.arange(start, stop)[:, None] * hop + np.arange(length)[None, :]
        fr = padded[idx]
        fr = fr - fr.mean(axis=1, keepdims=True)
        a = np.fft.rfft(fr[:, :win], nfft)
        b = np.fft.rfft(fr, nfft)
        r = np.fft.irfft(np.conj(a) * b, nfft)[:, :tau_max + 1]
        sq = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
        e0 = sq[:, win][:, None]
        taus = np.arange(tau_max + 1)
        et = sq[:, taus + win] - sq[:, taus]
        d = np.maximum(e0 + et - 2 * r, 0)
        cum = np.cumsum(d[:, 1:], axis=1)
        dn = np.ones_like(d)
        dn[:, 1:] = d[:, 1:] * np.arange(1, tau_max + 1)[None, :] / np.maximum(cum, 1e-12)
        seg = dn[:, tau_min:tau_max + 1]
        # first dip below 0.15 (YIN's absolute threshold), else the global minimum
        below = seg < 0.15
        first = np.where(below.any(axis=1), below.argmax(axis=1), seg.argmin(axis=1))
        # walk to the local minimum after the first dip
        best = first.copy()
        for _ in range(8):
            nxt = np.minimum(best + 1, seg.shape[1] - 1)
            better = seg[np.arange(len(seg)), nxt] < seg[np.arange(len(seg)), best]
            best = np.where(better, nxt, best)
        val = seg[np.arange(len(seg)), best]
        tau = best + tau_min
        f0[start:stop] = np.where(val < 0.35, ANALYSIS_SR / tau, 0.0)
        ap[start:stop] = np.clip(seg.min(axis=1), 0, 1)
    return f0, ap


def k_weighting(sr):
    """BS.1770 K-weighting as two biquads (high shelf + high pass), valid for any sample rate."""
    def biquad(kind, g_db, q, fc):
        a_ = 10 ** (g_db / 40)
        w0 = 2 * math.pi * fc / sr
        alpha = math.sin(w0) / (2 * q)
        c = math.cos(w0)
        if kind == "shelf":
            b = [a_ * ((a_ + 1) + (a_ - 1) * c + 2 * math.sqrt(a_) * alpha),
                 -2 * a_ * ((a_ - 1) + (a_ + 1) * c),
                 a_ * ((a_ + 1) + (a_ - 1) * c - 2 * math.sqrt(a_) * alpha)]
            a = [(a_ + 1) - (a_ - 1) * c + 2 * math.sqrt(a_) * alpha,
                 2 * ((a_ - 1) - (a_ + 1) * c),
                 (a_ + 1) - (a_ - 1) * c - 2 * math.sqrt(a_) * alpha]
        else:
            b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
            a = [1 + alpha, -2 * c, 1 - alpha]
        return np.array(b) / a[0], np.array(a) / a[0]
    return [biquad("shelf", 4.0, 1 / math.sqrt(2), 1500.0), biquad("hp", 0.0, 0.5, 38.0)]


def integrated_lufs(x, sr):
    y = x.astype(np.float64)
    for b, a in k_weighting(sr):
        y = signal.lfilter(b, a, y)
    block, step = int(0.4 * sr), int(0.1 * sr)
    if len(y) < block:
        return -70.0
    starts = np.arange(0, len(y) - block + 1, step)
    power = np.array([np.mean(y[s:s + block] ** 2) for s in starts])
    loud = -0.691 + 10 * np.log10(power + 1e-20)
    gated = power[loud > -70]
    if len(gated) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(gated)) - 10
    gated = power[(loud > -70) & (loud > rel)]
    return float(-0.691 + 10 * np.log10(np.mean(gated) + 1e-20)) if len(gated) else -70.0


def clipped_mask(x, threshold=0.985, min_run=3):
    """Samples inside runs of >= min_run consecutive samples at |x| >= threshold * full scale."""
    hot = np.abs(x) >= threshold
    if not hot.any():
        return hot
    edges = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    mask = np.zeros_like(hot)
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            mask[s:e] = True
    return mask


def release_tails(db, active, level, floor):
    """For every phrase end, the time (ms) the level keeps decaying from 10 dB to 35 dB below the phrase's
    final note (or to 6 dB above the noise floor, or until it stops falling by 2 dB per 100 ms, e.g. a
    breath). Dry singing: ~40-200 ms; reverb rings for 300 ms to several seconds."""
    tails = []
    n = len(db)
    ends = np.where(active[:-1] & ~active[1:])[0]
    for e in ends:
        start = max(0, e - 40)
        note = db[start:e + 1].max()
        if note < level - 25:                         # a quiet breath, not a sung phrase end
            continue
        hi = note - 10
        lo = max(note - 35, floor + 6)
        if lo >= hi:
            continue
        i = start + int(np.argmax(db[start:e + 1]))
        while i < n and db[i] > hi:
            i += 1
        j = i
        limit = min(n, i + 150)                       # look at most 1.5 s ahead
        restarted = False
        while j < limit and db[j] > lo:
            if db[j] > hi + 3:                        # the next phrase starts before the tail ended
                restarted = True
                break
            if j - i >= 10 and db[j - 10] - db[j] < 2.0:
                break                                 # a plateau (breath, room tone), not a decaying ring
            j += 1
        if restarted or (j >= limit and limit < i + 150):
            continue
        tails.append((e, (j - i) * HOP_S * 1000.0))   # 1500 = still ringing after 1.5 s
    return tails


# --------------------------------------------------------------------------------------- processing

def plan_clips(active, min_len, max_len, min_gap_s=0.25, keep_gap_s=0.3, quiet=None):
    """Groups active frames into clips of min_len..max_len seconds.
    Returns a list of lists of (start, end) frame ranges that are concatenated into one clip."""
    fps = int(round(1 / HOP_S))
    n = len(active)
    # phrases = active runs, merged across gaps shorter than min_gap
    edges = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
    starts, ends = list(np.where(edges == 1)[0]), list(np.where(edges == -1)[0])
    phrases = []
    for s, e in zip(starts, ends):
        if phrases and s - phrases[-1][1] < min_gap_s * fps:
            phrases[-1][1] = e
        else:
            phrases.append([s, e])
    pad = int(0.08 * fps)
    phrases = [[max(0, s - pad), min(n, e + pad)] for s, e in phrases]

    # split phrases longer than max_len at their quietest 100 ms
    split = []
    for s, e in phrases:
        while e - s > max_len * fps:
            lo, hi = s + int(min_len * fps), s + int(max_len * fps)
            window = quiet[lo:hi] if quiet is not None else np.zeros(hi - lo)
            cut = lo + int(np.argmin(np.convolve(window, np.ones(10) / 10, mode="same")))
            split.append([s, cut])
            s = cut
        split.append([s, e])

    # join consecutive phrases into clips (pauses between them shortened to keep_gap)
    clips, cur, cur_len = [], [], 0.0
    gap = keep_gap_s
    for s, e in split:
        length = (e - s) / fps
        extra = length + (gap if cur else 0)
        if cur and cur_len + extra > max_len:
            clips.append(cur)
            cur, cur_len = [], 0.0
            extra = length
        cur.append((s, e))
        cur_len += extra
        if cur_len >= min_len and cur_len >= 0.6 * max_len:
            clips.append(cur)
            cur, cur_len = [], 0.0
    if cur:
        clips.append(cur)
    return clips


def assemble(x, sr, pieces, gap_s=0.3, fade_s=0.01):
    hop = sr * HOP_S
    fade = int(fade_s * sr)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    out = []
    for k, (s, e) in enumerate(pieces):
        seg = x[int(s * hop):int(e * hop)].copy()
        if len(seg) > 2 * fade:
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        if k:
            out.append(np.zeros(int(gap_s * sr), np.float32))
        out.append(seg)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80] or "clip"


def process_file(path, args, out_dir, voice, taken=None):
    rep = {"file": path, "voice": voice, "status": "ok", "notes": [], "rejected_s": {}}
    try:
        data, sr = decode(path)
    except Exception as e:
        rep.update(status="error", notes=[str(e)])
        return rep, []

    rep["sample_rate"] = sr
    rep["channels"] = int(data.shape[0])
    rep["duration_s"] = round(data.shape[1] / sr, 2)
    if data.shape[0] > 1:
        l, r = data[0].astype(np.float64), data[1].astype(np.float64)
        denom = math.sqrt(float(np.dot(l, l) * np.dot(r, r))) + 1e-12
        corr = float(np.dot(l, r) / denom)
        rep["stereo_correlation"] = round(corr, 3)
        if corr < 0.9:
            rep["notes"].append("left/right differ (correlation %.2f): stereo double or effect? mixed to mono" % corr)
    x = data.mean(axis=0).astype(np.float64)
    x -= x.mean()
    peak_in = float(np.abs(x).max()) if len(x) else 0.0
    clip_mask_full = clipped_mask(data.reshape(-1) if data.shape[0] == 1 else np.abs(data).max(axis=0))
    sos = signal.butter(4, args.highpass, btype="highpass", fs=sr, output="sos")
    x = signal.sosfilt(sos, x).astype(np.float32)

    db = frame_db(x, sr)
    nonsilent = db > -100
    nz = db[nonsilent]
    if len(nz) < 50:
        rep.update(status="rejected", notes=rep["notes"] + ["silent"])
        return rep, []
    level = float(np.percentile(nz, 95))
    # Noise floor: only measurable in real pauses. Gated/edited stems have digitally silent gaps (and
    # fades next to them), short phrases have no pauses at all; then the floor is "unknown".
    near_silence = np.convolve((~nonsilent).astype(float), np.ones(31), mode="same") > 0
    candidates = db[nonsilent & ~near_silence]
    if np.mean(~nonsilent) > 0.05:
        gaps = candidates[candidates < level - 25]     # gated stem: real pauses only
        floor_known = len(gaps) >= 100
        floor = float(np.median(gaps)) if floor_known else 0.0
    else:
        floor_known = len(candidates) >= 500
        floor = float(np.percentile(candidates, 2)) if floor_known else 0.0
    floor = max(floor, -90.0) if floor_known else max(level - 60.0, -90.0)
    threshold = max(floor + 12, level - 40)
    active = db > threshold
    # remove isolated blips < 60 ms
    act = active.astype(np.int8)
    edges = np.diff(np.concatenate([[0], act, [0]]))
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if e - s < 6:
            active[s:e] = False

    x16 = resample(x, sr, ANALYSIS_SR)
    f0, ap = yin_frames(x16)
    m = min(len(f0), len(db))
    f0, ap, db, active = f0[:m], ap[:m], db[:m], active[:m]
    strong = active & (db > level - 20)
    voiced = strong & (f0 > 0)

    tails = release_tails(db, active, level, floor)
    tail_ms = [t for _, t in tails]
    rep.update(
        noise_floor_db=round(floor, 1) if floor_known else None, level_db=round(level, 1),
        snr_db=round(level - floor, 1) if floor_known else None,
        active_s=round(float(active.sum()) * HOP_S, 1), voiced_s=round(float(voiced.sum()) * HOP_S, 1),
        periodicity=round(float(1 - np.median(ap[strong])) if strong.any() else 0.0, 3),
        release_tail_ms=round(float(np.median(tail_ms)), 0) if tail_ms else None,
        phrase_ends=len(tail_ms), peak_dbfs=round(20 * math.log10(peak_in + 1e-12), 1),
    )

    reject_file = None
    if floor_known and rep["snr_db"] < args.min_snr:
        reject_file = "noisy (noise floor %.0f dB below the singing)" % rep["snr_db"]
    elif tail_ms and len(tail_ms) >= 3 and rep["release_tail_ms"] > args.max_file_tail_ms:
        reject_file = "reverberant / wet (phrase ends ring for %d ms)" % rep["release_tail_ms"]
    elif rep["periodicity"] < args.min_file_periodicity:
        reject_file = "not a single voice (periodicity %.2f: stacked harmonies or doubles?)" % rep["periodicity"]
    if reject_file and not args.keep_all:
        rep["status"] = "rejected"
        rep["notes"].append(reject_file)
        rep["rejected_s"][reject_file.split(" (")[0]] = rep["active_s"]
        return rep, []

    quiet = db.copy()
    plans = plan_clips(active, args.min_len, args.max_len, quiet=quiet)
    fps = int(round(1 / HOP_S))
    hop_samples = sr * HOP_S
    tail_at = {e: t for e, t in tails}
    clips = []
    for k, pieces in enumerate(plans):
        frames = np.zeros(m, bool)
        for s, e in pieces:
            frames[s:e] = True
        length = sum(e - s for s, e in pieces) / fps + 0.3 * (len(pieces) - 1)
        seg_active = frames & active
        seg_strong = frames & strong
        reasons = []
        if length < args.min_keep:
            reasons.append("too short")
        cs = int(pieces[0][0] * hop_samples)
        ce = int(pieces[-1][1] * hop_samples)
        clip_pct = 100.0 * float(clip_mask_full[cs:ce].mean()) if ce > cs else 0.0
        if clip_pct > args.max_clip_pct:
            reasons.append("clipped")
        per = float(1 - np.median(ap[seg_strong])) if seg_strong.sum() > 20 else 1.0
        if per < args.min_periodicity:
            reasons.append("not a single voice")
        seg_tails = [t for e_, t in tail_at.items() if frames[min(e_, m - 1)]]
        if seg_tails and max(seg_tails) > args.max_tail_ms and np.median(seg_tails) > args.max_tail_ms * 0.8:
            reasons.append("reverberant")
        if seg_strong.sum() < 0.3 * fps:
            reasons.append("no singing")
        secs = round(float(seg_active.sum()) * HOP_S, 2)
        if reasons and not args.keep_all:
            key = reasons[0]
            rep["rejected_s"][key] = round(rep["rejected_s"].get(key, 0.0) + secs, 2)
            continue
        audio = assemble(x, sr, pieces)
        lufs = integrated_lufs(audio, sr)
        gain_db = args.target_lufs - lufs
        peak = float(np.abs(audio).max()) + 1e-12
        max_gain_db = args.peak_dbfs - 20 * math.log10(peak)
        if gain_db > max_gain_db:
            gain_db = max_gain_db
        audio = audio * (10 ** (gain_db / 20))
        audio = resample(audio, sr, args.sr)
        audio = np.clip(audio, -1.0, 1.0)
        f0_seg = f0[frames & voiced]
        clips.append(dict(audio=audio, length=len(audio) / args.sr, f0=f0_seg, periodicity=per,
                          lufs=round(lufs + gain_db, 1), index=k))

    base = safe_name(os.path.splitext(os.path.basename(path))[0])
    if taken is not None:                   # same file name in two input folders: keep both
        unique, k = base, 2
        while (out_dir, unique) in taken:
            unique, k = "%s_%d" % (base, k), k + 1
        taken.add((out_dir, unique))
        base = unique
    written = []
    os.makedirs(out_dir, exist_ok=True)
    for c in clips:
        name = "%s_%03d.wav" % (base, c["index"])
        sf.write(os.path.join(out_dir, name), c["audio"], args.sr, subtype="PCM_16")
        written.append(dict(file=name, seconds=round(c["length"], 2), f0=c["f0"]))
        rep.setdefault("clip_list", []).append([name, round(c["length"], 2), round(c["periodicity"], 3)])
    rep["clips"] = len(written)
    rep["kept_s"] = round(sum(w["seconds"] for w in written), 1)
    if not written:
        rep["status"] = "rejected"
    return rep, written


def midi(hz):
    return 69 + 12 * np.log2(np.asarray(hz) / 440.0)


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(m):
    m = int(round(m))
    return "%s%d" % (NOTE_NAMES[m % 12], m // 12 - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", help="audio files, folders or globs")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--name", default="voice", help="voice name = output sub-folder (without --manifest)")
    p.add_argument("--manifest", help="CSV with columns path,voice (paths may be globs, relative to the CSV)")
    p.add_argument("--include", action="append", default=[], help="only file names matching this glob (repeatable)")
    p.add_argument("--exclude", action="append", default=[], help="skip file names matching this glob (repeatable), e.g. '*wet*'")
    p.add_argument("--sr", type=int, default=40000, help="output sample rate (40000 for 40k models, 48000 for 48k)")
    p.add_argument("--min-len", type=float, default=DEFAULTS["min_len"], help="shortest clip in seconds")
    p.add_argument("--min-keep", type=float, default=DEFAULTS["min_keep"],
                   help="keep clips down to this length when a file has no more audio to join (s)")
    p.add_argument("--max-len", type=float, default=DEFAULTS["max_len"], help="longest clip in seconds")
    p.add_argument("--highpass", type=float, default=60.0, help="high-pass cutoff in Hz")
    p.add_argument("--target-lufs", type=float, default=DEFAULTS["target_lufs"])
    p.add_argument("--peak-dbfs", type=float, default=DEFAULTS["peak_dbfs"])
    p.add_argument("--min-snr", type=float, default=DEFAULTS["min_snr"], help="reject files whose noise floor is closer than this (dB)")
    p.add_argument("--max-clip-pct", type=float, default=DEFAULTS["max_clip_pct"], help="reject clips with more clipped samples (%%)")
    p.add_argument("--min-periodicity", type=float, default=DEFAULTS["min_periodicity"],
                   help="reject clips below this (0..1; one dry voice is typically 0.85-0.95)")
    p.add_argument("--min-file-periodicity", type=float, default=DEFAULTS["min_file_periodicity"],
                   help="reject whole files below this median periodicity")
    p.add_argument("--max-tail-ms", type=float, default=DEFAULTS["max_tail_ms"], help="reject clips whose phrase ends ring longer")
    p.add_argument("--max-file-tail-ms", type=float, default=DEFAULTS["max_file_tail_ms"],
                   help="reject whole files whose median phrase end rings longer (wet vocals)")
    p.add_argument("--keep-all", action="store_true", help="measure and report, but keep everything")
    p.add_argument("--zip", action="store_true", help="also write <out>/<voice>.zip for the training notebook")
    p.add_argument("--clean", action="store_true", help="delete existing clips of the voices being written first")
    args = p.parse_args()

    jobs = []   # (path, voice)
    if args.manifest:
        base = os.path.dirname(os.path.abspath(args.manifest))
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("path") or not row.get("voice") or row["voice"].strip().startswith("#"):
                    continue
                pattern = row["path"] if os.path.isabs(row["path"]) else os.path.join(base, row["path"])
                for path in find_inputs([pattern], args.include, args.exclude):
                    jobs.append((path, row["voice"].strip()))
    for path in find_inputs(args.inputs, args.include, args.exclude):
        jobs.append((path, args.name))
    if not jobs:
        sys.exit("no input files (check the paths and --include/--exclude patterns)")

    voices = sorted({v for _, v in jobs})
    if args.clean:
        for v in voices:
            d = os.path.join(args.out, safe_name(v))
            if os.path.isdir(d):
                for name in os.listdir(d):
                    if name.endswith(".wav"):
                        os.remove(os.path.join(d, name))

    reports, per_voice = [], {v: dict(clips=0, seconds=0.0, f0=[], files=0) for v in voices}
    taken = set()
    for i, (path, voice) in enumerate(jobs):
        rep, written = process_file(path, args, os.path.join(args.out, safe_name(voice)), voice, taken)
        reports.append(rep)
        pv = per_voice[voice]
        pv["files"] += 1
        pv["clips"] += len(written)
        pv["seconds"] += sum(w["seconds"] for w in written)
        for w in written:
            pv["f0"].append(w["f0"])
        why = "; ".join(rep["notes"])
        rej = ", ".join("%s %.0f s" % kv for kv in rep["rejected_s"].items())
        log("[%d/%d] %-60s %-8s kept %5.1f s %s%s" % (i + 1, len(jobs), os.path.basename(path)[:60], rep["status"],
                                                    rep.get("kept_s", 0.0), ("| rejected: " + rej) if rej else "",
                                                    (" | " + why) if why else ""))

    summary = {}
    for v, pv in per_voice.items():
        f0 = np.concatenate(pv["f0"]) if pv["f0"] else np.zeros(0)
        entry = dict(files=pv["files"], clips=pv["clips"], minutes=round(pv["seconds"] / 60, 2))
        if len(f0) > 100:
            lo, med, hi = np.percentile(midi(f0), [3, 50, 97])
            entry.update(range_midi=[int(round(lo)), int(round(hi))], median_f0_hz=round(float(np.median(f0)), 1),
                         range_notes="%s - %s (median %s)" % (note_name(lo), note_name(hi), note_name(med)))
        summary[v] = entry
    rejected_total = {}
    for r in reports:
        for k, s in r["rejected_s"].items():
            rejected_total[k] = round(rejected_total.get(k, 0.0) + s, 1)

    os.makedirs(args.out, exist_ok=True)
    report = dict(settings={k: getattr(args, k) for k in ("sr", "min_len", "min_keep", "max_len", "highpass", "target_lufs",
                                                            "peak_dbfs", "min_snr", "max_clip_pct", "min_periodicity", "min_file_periodicity",
                                                            "max_tail_ms", "max_file_tail_ms")},
                  voices=summary, rejected_seconds=rejected_total, files=reports)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=float)

    lines = ["# Dataset report", "", "| Voice | Minutes | Clips | Files | Range (3rd-97th percentile) |", "|---|---|---|---|---|"]
    for v, e in summary.items():
        lines.append("| %s | %.1f | %d | %d | %s |" % (v, e["minutes"], e["clips"], e["files"], e.get("range_notes", "-")))
    lines += ["", "Rejected (seconds of singing): " + (", ".join("%s %.0f" % kv for kv in rejected_total.items()) or "nothing"), "",
              "| File | Voice | Status | Kept s | SNR dB | Periodicity | Tail ms | Rejected | Notes |", "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            os.path.basename(r["file"]), r["voice"], r["status"], r.get("kept_s", 0), r.get("snr_db", ""),
            r.get("periodicity", ""), r.get("release_tail_ms", "") if r.get("release_tail_ms") is not None else "",
            ", ".join("%s %.0f s" % kv for kv in r["rejected_s"].items()), "; ".join(r["notes"])))
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for v, e in summary.items():
        log("%s: %.1f min in %d clips, range %s" % (v, e["minutes"], e["clips"], e.get("range_notes", "?")))
        if e["minutes"] < 10:
            log("  note: under 10 minutes - the voice will work but sound less stable; 20-30+ minutes is better")
        if args.zip and e["clips"]:
            folder = os.path.join(args.out, safe_name(v))
            zpath = os.path.join(args.out, safe_name(v) + ".zip")
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for name in sorted(os.listdir(folder)):
                    if name.endswith(".wav"):
                        z.write(os.path.join(folder, name), name)
            log("  wrote " + zpath)
    log("report: " + os.path.join(args.out, "report.md"))


if __name__ == "__main__":
    main()
