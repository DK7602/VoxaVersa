"""Round 21: the accent's consonants made in the recording, before the AI singer hears it.
  each "th": the hiss becomes silence (the closure), then a real "t" / "d" (Kokoro's, burst + 40 ms of vowel)
  "for":     the r-coloured end of the vowel becomes the plain "aw" of its start (WORLD envelope swap)
usage: edit_input.py take.wav stops_dir out.wav"""
import sys, numpy as np, soundfile as sf, pyworld as pw
from scipy.signal import resample_poly
src, sdir, out = sys.argv[1:4]
x, sr = sf.read(src, dtype="float64"); x = x.mean(1) if x.ndim > 1 else x
y = x.copy()
def load(name):
    s, r = sf.read("%s/%s.wav" % (sdir, name), dtype="float64"); return resample_poly(s, sr, r)
def rms(a): return np.sqrt(np.mean(a ** 2) + 1e-12)
# (sample, closure start, release start, vowel onset)
STOPS = [("t_i_am", 1.84, 1.98, 2.06), ("d_a_am", 2.70, 2.92, 2.99), ("d_i_am", 3.12, 3.26, 3.33), ("d_e_am", 4.15, 4.24, 4.31)]
for name, c0, r0, v0 in STOPS:
    a, r, v = int(c0 * sr), int(r0 * sr), int(v0 * sr)
    fade = int(0.006 * sr)
    w = np.ones(r - a); w[:fade] = np.linspace(1, 0, fade); w[fade:] = 0.0
    y[a:r] *= w                                         # closure: silence (with a faint room noise, as a real one)
    y[a:r] += (1 - w) * np.random.default_rng(1).standard_normal(r - a) * 10 ** (-66 / 20)
    s = load(name)
    vow = rms(x[v:v + int(0.03 * sr)]); s *= vow / rms(s[-int(0.03 * sr):])   # its vowel part as loud as yours
    n = min(len(s), v - r + int(0.03 * sr))
    xf = int(0.02 * sr)                                 # cross-fade into your own vowel
    seg = s[:n].copy()
    y[r:r + n - xf] = seg[:n - xf]
    ramp = np.linspace(0, 1, xf)
    y[r + n - xf:r + n] = seg[n - xf:n] * (1 - ramp) + x[r + n - xf:r + n] * ramp
# "for": the end of the vowel (its r) gets the envelope of the vowel's start
a, b = int(5.08 * sr), int(5.62 * sr)
seg = x[a:b]; FP = 5.0
f0, t = pw.harvest(seg, sr, frame_period=FP, f0_floor=70, f0_ceil=900)
sp = pw.cheaptrick(seg, f0, t, sr); ap = pw.d4c(seg, f0, t, sr)
k = lambda s_: int((s_ - 5.08) * 1000 / FP)
pure = sp[k(5.20):k(5.30)].mean(0)
for i in range(k(5.32), k(5.54)):
    w_ = min(1.0, (i - k(5.32)) / 6); sp[i] = (1 - w_) * sp[i] + w_ * pure
syn = pw.synthesize(f0, sp, ap, sr, frame_period=FP)[: len(seg)]
c0, c1 = int((5.28 - 5.08) * sr), int((5.56 - 5.08) * sr); xf = int(0.015 * sr)
mix = seg.copy(); ramp = np.linspace(0, 1, xf)
mix[c0:c0 + xf] = seg[c0:c0 + xf] * (1 - ramp) + syn[c0:c0 + xf] * ramp
mix[c0 + xf:c1 - xf] = syn[c0 + xf:c1 - xf]
mix[c1 - xf:c1] = syn[c1 - xf:c1] * (1 - ramp) + seg[c1 - xf:c1] * ramp
y[a:b] = mix
sf.write(out, y.astype(np.float32), sr, subtype="FLOAT")
print("edited", out)
