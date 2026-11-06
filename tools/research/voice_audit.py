"""Objective audit of the AI voices on one input vocal: each voice's render (VoxaVersaRender <in> <out> --voice id)
against the dry input.
  words:   word error rate of a speech recognizer (faster-whisper small.en) on the render, against the lyric
           (and the same on the dry input, for reference)
  pitch:   per 10 ms frame where both are voiced, the render's pitch against the input's moved by the render's
           median shift (whole octaves in the register the voice picked): median absolute error (cents), share of
           frames within 50 cents, share an octave or more off; wobble on held notes (std of the render's pitch
           where the input holds within 25 cents for 200 ms, minus the input's own)
  dropouts: share of the input's sung frames where the render is unvoiced or 30 dB down
  range:   the render's pitches against its singer's trained range (voice_assignment.json): share above the top
           (thin, "chipmunk") and below the bottom
  clicks:  10 ms frames whose energy above 6 kHz jumps 15 dB over the surrounding 200 ms with no such jump in the
           input (per minute)
Needs numpy, scipy, soundfile, praat-parselmouth, faster-whisper.
usage: voice_audit.py input.wav "lyric text" voice_assignment.json out.csv render.wav..."""
import sys, os, re, csv, json, math
import numpy as np, soundfile as sf, parselmouth
from scipy.signal import resample_poly, butter, sosfilt

inp, lyric, assignment_path, out_csv, renders = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]

def load(path, rate=16000):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
    g = math.gcd(int(sr), rate); return resample_poly(x, rate // g, int(sr) // g).astype(np.float32)

def pitch(x):
    p = parselmouth.Sound(x.astype(np.float64), 16000).to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=1100)
    f = np.array([p.get_value_at_time((k + 0.5) / 100) or 0.0 for k in range(len(x) // 160)])
    return np.nan_to_num(f)

def frame_db(x, hop=160):
    n = len(x) // hop; e = (x[:n * hop].reshape(n, hop) ** 2).mean(1)
    return 10 * np.log10(e + 1e-10)

def words(text): return re.sub(r"[^a-z' ]", " ", text.lower()).split()

def wer(ref, hyp):
    r, h = words(ref), words(hyp); d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d.copy(), i
        for j, hw in enumerate(h, 1): d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (rw != hw))
    return d[len(h)] / max(1, len(r))

from faster_whisper import WhisperModel
asr = WhisperModel("small.en", device="cpu", compute_type="int8")
def transcribe(x):
    segs, _ = asr.transcribe(x, beam_size=5, language="en", condition_on_previous_text=False)
    return " ".join(s.text for s in segs)

assignment = json.load(open(assignment_path))["packs"]
range_of = {v: p.get("range") for p in assignment.values() for v in p.get("factoryVoices", [])}

def clicks(x, ref):
    sos = butter(4, 6000, "highpass", fs=16000, output="sos")
    hx, hr = frame_db(sosfilt(sos, x)), frame_db(sosfilt(sos, ref)); n = min(len(hx), len(hr)); hx, hr = hx[:n], hr[:n]
    def jumps(h):
        out = np.zeros(len(h), bool)
        for k in range(len(h)):
            around = np.r_[h[max(0, k - 10):max(0, k - 2)], h[k + 3:k + 11]]
            out[k] = len(around) and h[k] - np.median(around) > 15
        return out
    jx, jr = jumps(hx), jumps(hr)
    near = np.array([jr[max(0, k - 3):k + 4].any() for k in range(n)])
    return int((jx & ~near).sum()) / (n / 6000)

x0 = load(inp); f0 = pitch(x0); db0 = frame_db(x0)
sung = (f0 > 0) & (db0[:len(f0)] > np.percentile(db0, 60) - 15)
rows = [dict(voice="(dry input)", wer=round(wer(lyric, transcribe(x0)), 3))]
print(rows[0], flush=True)
for path in renders:
    voice = os.path.basename(path)[:-4]; x = load(path); f = pitch(x); dbx = frame_db(x)
    n = min(len(f), len(f0)); a, b = f0[:n], f[:n]; both = (a > 0) & (b > 0)
    cents = 1200 * np.log2(b[both] / a[both]) if both.any() else np.zeros(1)
    shift = 1200 * round(np.median(cents) / 1200)                       # the register: whole octaves
    err = cents - shift
    # held notes: the input within 25 cents of its 200 ms mean
    held = np.zeros(n, bool); ia = np.where(a > 0, 1200 * np.log2(np.maximum(a, 1) / 440), np.nan)
    for k in range(10, n - 10):
        w = ia[k - 10:k + 10]
        held[k] = np.isfinite(w).all() and np.abs(w - w.mean()).max() < 25
    hb = held & both
    ib = np.where(b > 0, 1200 * np.log2(np.maximum(b, 1) / 440), np.nan)
    def local_std(c, mask):
        vals = [np.std(c[k - 5:k + 5]) for k in np.where(mask)[0] if np.isfinite(c[k - 5:k + 5]).all()]
        return float(np.mean(vals)) if vals else float("nan")
    wobble = local_std(ib, hb) - local_std(ia, hb)
    drop = (sung[:n] & ((b == 0) | (dbx[:n] < np.median(dbx[:n][sung[:n] & (b > 0)]) - 30))).sum() / max(1, sung[:n].sum())
    rng = range_of.get(voice); above = below = float("nan")
    if rng:
        m = 69 + 12 * np.log2(b[b > 0] / 440)
        above, below = float((m > rng[1] + 0.5).mean()), float((m < rng[0] - 0.5).mean())
    row = dict(voice=voice, wer=round(wer(lyric, transcribe(x)), 3), shift_oct=int(shift / 1200),
               pitch_mae=round(float(np.median(np.abs(err))), 1), within50=round(float((np.abs(err) < 50).mean()), 3),
               octave_off=round(float((np.abs(err) > 600).mean()), 3), wobble=round(wobble, 1), dropouts=round(float(drop), 3),
               above_range=round(above, 3), below_range=round(below, 3), clicks_per_min=round(clicks(x, x0), 1),
               singer_range="-" if not rng else "%d-%d" % tuple(rng))
    rows.append(row); print(row, flush=True)
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[1].keys()) if len(rows) > 1 else list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, "") for k in w.fieldnames})
