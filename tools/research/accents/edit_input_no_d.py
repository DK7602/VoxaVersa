"""Round 23: the accent's consonants made in the recording. Round 22's "d" (a quiet murmur and a 12 ms click) was
heard as the sound cutting out; round 21's (a whole "d" with the start of its vowel) was heard, but broke up at the
join (the other voice's pitch). Here each "d" is a whole one again, re-pitched (WORLD) to the note you sang there,
after a short closure, blended into your vowel over 30 ms.
  each "d" (that, this, the): the "th" becomes a short silence, then a voice bar (a low murmur at the vowel's
            pitch, as every real "d" has), then only the 12 ms burst of Kokoro's "d", straight into your own vowel
  "think":  silence, then Kokoro's "t" burst and aspiration (no vowel), into your vowel
  "for":    the r-coloured end of the vowel gets the plain "aw" of its start (WORLD envelope swap)
  "and":    the "d" before "you" becomes a held "n" (an' you)
usage: edit_input_no_d.py take.wav stops_dir out.wav"""
import sys, numpy as np, soundfile as sf, pyworld as pw
from scipy.signal import resample_poly, butter, sosfiltfilt
src, sdir, out = sys.argv[1:4]
x, sr = sf.read(src, dtype="float64"); x = x.mean(1) if x.ndim > 1 else x
y = x.copy(); rng = np.random.default_rng(1)
def load(name):
    s, r = sf.read("%s/%s.wav" % (sdir, name), dtype="float64"); return resample_poly(s, sr, r)
def rms(a): return np.sqrt(np.mean(a ** 2) + 1e-12)
def burst(name, ms):
    s = load(name); j = int(0.004 * sr); n = int(ms / 1000 * sr)        # the files start 4 ms before the burst
    seg = s[: j + n].copy(); seg[-int(0.003 * sr):] *= np.linspace(1, 0, int(0.003 * sr)); return seg
low = butter(4, 450, "lowpass", fs=sr, output="sos")
def xfade(a, b, n):
    r = np.linspace(0, 1, n); return a[:n] * (1 - r) + b[:n] * r
import json
BURSTS = json.load(open("%s/bursts.json" % sdir))
def repitched_d(name, f_target):
    """Kokoro's "a d-vowel" re-synthesised at the singer's note; returns 5 ms before the burst to 70 ms after it."""
    s, r = sf.read("%s/%s_full.wav" % (sdir, name), dtype="float64")
    f0, t = pw.harvest(s, r, frame_period=5.0, f0_floor=60, f0_ceil=600)
    sp = pw.cheaptrick(s, f0, t, r); ap = pw.d4c(s, f0, t, r)
    f0 = np.where(f0 > 0, f_target, 0.0)
    syn = pw.synthesize(f0, sp, ap, r, frame_period=5.0)
    b0 = int(BURSTS[name] * r); seg = syn[b0 - int(0.005 * r):b0 + int(0.070 * r)]
    return resample_poly(seg, sr, r)
def sung_f0(t0):
    a, b = int(t0 * sr), int((t0 + 0.12) * sr)
    f0, _ = pw.harvest(x[a:b], sr, frame_period=5.0, f0_floor=70, f0_ceil=900)
    v = f0[f0 > 0]; return float(np.median(v)) if len(v) else 150.0
# Round 24: a legato "d". Round 23 placed the "d" before the soft start of your vowel (it reaches full strength
# ~40 ms later: d - dip - vowel, heard as a break-up) and put silence in front of it. Now the voice never stops:
# the "th" becomes a voice bar (your coming vowel, low-passed, 10 dB down) at the vowel's pitch, then a short "d"
# (burst + 25 ms, re-pitched), placed right at the strong start of your vowel. "that" keeps your pause before it.
# (kind, sample, closure start, voice bar start, vowel onset)
STOPS = [("t", "t_i_am", 1.84, None, 2.06)]          # the "d"s are done above
for kind, name, c0, vb0, v0 in STOPS:
    a, v = int(c0 * sr), int(v0 * sr); fade = int(0.008 * sr)
    vow = x[v:v + int(0.04 * sr)]
    if kind == "t":
        b = burst(name, 45); b *= 0.5 * rms(vow) / rms(b)                 # burst + aspiration, quieter than the vowel
        start = v - len(b)
        y[a:start] = y[a:start] * np.r_[np.linspace(1, 0, fade), np.zeros(start - a - fade)] + rng.standard_normal(start - a) * 10 ** (-66 / 20)
        y[start:v] = b
    else:
        vb = int(vb0 * sr); b = burst(name, 12); b *= 0.35 * rms(vow) / rms(b)
        # closure: silence; then the voice bar: your vowel's first 40 ms, low-passed, quiet, faded in
        y[a:vb] = y[a:vb] * np.r_[np.linspace(1, 0, fade), np.zeros(vb - a - fade)] + rng.standard_normal(vb - a) * 10 ** (-66 / 20)
        bar_len = v - len(b) - vb
        src_bar = sosfiltfilt(low, np.tile(vow, int(np.ceil(bar_len / len(vow))) + 1))[:bar_len]
        src_bar *= 0.18 * np.linspace(0.3, 1, bar_len)
        y[vb:vb + bar_len] = src_bar
        y[vb + bar_len:v] = b + 0.18 * src_bar[-1] * 0
    # into your own vowel, softly
    n = int(0.006 * sr); y[v:v + n] = xfade(y[v:v + n], x[v:v + n], n)

def envelope_swap(t0, t1, pure0, pure1, sw0, sw1, fill_f0=False):
    """WORLD-resynthesise t0..t1 with the frames sw0..sw1 given the mean envelope of pure0..pure1."""
    a, b = int(t0 * sr), int(t1 * sr); seg = x[a:b]; FP = 5.0
    f0, t = pw.harvest(seg, sr, frame_period=FP, f0_floor=70, f0_ceil=900)
    sp = pw.cheaptrick(seg, f0, t, sr); ap = pw.d4c(seg, f0, t, sr)
    k = lambda s_: int((s_ - t0) * 1000 / FP)
    pure = sp[k(pure0):k(pure1)].mean(0); pap = ap[k(pure0):k(pure1)].mean(0)
    if fill_f0:                      # no unvoiced gap where the "d" was
        vz = np.where(f0 > 0)[0]; f0 = np.interp(np.arange(len(f0)), vz, f0[vz])
    for i in range(k(sw0), k(sw1)):
        w_ = min(1.0, (i - k(sw0)) / 4, (k(sw1) - i) / 4); sp[i] = (1 - w_) * sp[i] + w_ * pure; ap[i] = (1 - w_) * ap[i] + w_ * pap
    syn = pw.synthesize(f0, sp, ap, sr, frame_period=FP)[: len(seg)]
    xf = int(0.015 * sr); c0, c1 = int((sw0 - 0.04 - t0) * sr), int((sw1 + 0.04 - t0) * sr)
    mix = seg.copy(); r = np.linspace(0, 1, xf)
    mix[c0:c0 + xf] = seg[c0:c0 + xf] * (1 - r) + syn[c0:c0 + xf] * r
    mix[c0 + xf:c1 - xf] = syn[c0 + xf:c1 - xf]
    mix[c1 - xf:c1] = syn[c1 - xf:c1] * (1 - r) + seg[c1 - xf:c1] * r
    y[a:b] = mix
envelope_swap(5.08, 5.62, 5.20, 5.30, 5.32, 5.54)                       # for -> fo
envelope_swap(6.46, 7.00, 6.56, 6.63, 6.64, 6.80, fill_f0=True)           # and you -> an' you
sf.write(out, y.astype(np.float32), sr, subtype="FLOAT")
print("edited", out)
