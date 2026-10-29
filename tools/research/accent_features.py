"""Feasibility: the Southern accent made inside the AI singer, on its content features (ContentVec, 50 Hz),
instead of on the audio. Same edits and moments as southern.py (its printed edits file), applied as
  mono  (I, joy): hold the vowel's held-window features through the glide
  raise (friend): + (mean IH - mean EH before a nasal)
  lower (pain):   + (mean EH - mean EY), fading out over the first 60 % of the vowel
  drawl (and):    + w(t) (mean IH - mean AE), a glide peaking at 65 %
then the voice's generator, as try_voice.py does (YIN pitch).
usage: accent_features.py model.onnx contentvec.onnx in.wav edits.txt means.npz transpose out_plain.wav out_accent.wav"""
import sys, os, math, re
import numpy as np, soundfile as sf, onnxruntime as ort
from scipy import signal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from try_voice import resample, coarse, pitch_yin
model, encoder, src, edits_path, means_path, transpose, out_plain, out_acc = sys.argv[1:9]
M = dict(np.load(means_path))
mu = lambda k: M[k] if k in M else None
so = ort.SessionOptions(); so.intra_op_num_threads = 3
enc = ort.InferenceSession(encoder, so, providers=["CPUExecutionProvider"])
gen = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
gin = {i.name for i in gen.get_inputs()}
x, sr = sf.read(src, dtype="float32", always_2d=True); x = x.mean(1)
x16 = resample(x, sr, 16000)
frames = int(math.ceil(len(x16) / 160)); frames += frames % 2
x16 = np.pad(x16, (0, frames * 160 + 80 - len(x16)))
f0 = pitch_yin(x16); f0 = np.pad(f0, (0, max(0, frames - len(f0))))[:frames] * (2 ** (float(transpose) / 12))
feats = enc.run(None, {enc.get_inputs()[0].name: x16[None, None]})[0][0]          # whole file: [T, 768]

held, edits = [], []
for line in open(edits_path):
    m = re.match(r"\s+(\S+) held ([\d.]+)-([\d.]+) s", line)
    if m: held.append((float(m.group(2)), float(m.group(3)))); continue
    p = line.split()
    if len(p) == 6: edits.append((p[0], p[1], float(p[2]), float(p[4])))
acc = feats.copy()
hi = iter(held)
d_raise = mu("IH") - (mu("EH_N") if mu("EH_N") is not None else mu("EH"))
d_lower = mu("EH") - mu("EY")
d_drawl = mu("IH") - mu("AE")
for word, kind, a, b in edits:
    k0, k1 = int(a * 50), min(len(acc), int(b * 50) + 1); m = k1 - k0
    if m < 2: continue
    if kind == "mono":   # hold the steady "ah" just before the glide (the last 100 ms before the final 15 %)
        next(hi)
        g = k0 + int(0.85 * m); j0 = max(k0 + 1, g - 5)
        hold = feats[j0:g].mean(0)
        for k in range(j0, k1 - 1):
            w = min(1.0, (k - j0 + 1) / 3)
            acc[k] = (1 - w) * feats[k] + w * hold
    for i, k in enumerate(range(k0, k1)):
        pos = i / max(1, m - 1); ramp = min(1.0, (i + 1) / 3, (m - i) / 3)
        if kind == "raise": acc[k] = feats[k] + ramp * d_raise
        elif kind == "lower": acc[k] = feats[k] + ramp * max(0.0, 1 - pos / 0.6) * d_lower
        elif kind == "drawl": acc[k] = feats[k] + max(0.0, 1 - abs(pos - 0.65) / 0.25) * d_drawl
print("edits", len(edits), "held", len(held), "| shift sizes: raise %.2f lower %.2f drawl %.2f (mean |feat| %.2f)" % (
    np.linalg.norm(d_raise), np.linalg.norm(d_lower), np.linalg.norm(d_drawl), np.linalg.norm(feats, axis=1).mean()))

def render(F, out):
    rng = np.random.default_rng(0); seg, ctx = 1000, 100; pieces = []; out_sr = None
    for start in range(0, frames, seg):
        a, b = max(0, start - ctx), min(frames, start + seg + ctx); a -= a % 2; b -= (b - a) % 2; n = b - a
        idx = np.minimum((a + np.arange(n)) // 2, len(F) - 1)
        feeds = {"phone": F[idx][None].astype(np.float32), "phone_lengths": np.array([n], np.int64), "ds": np.array([0], np.int64),
                 "rnd": (rng.standard_normal((1, 192, n)) * 0.66666).astype(np.float32)}
        if "pitchf" in gin:
            pf = f0[a:b].astype(np.float32)[None]; feeds.update(pitch=coarse(pf), pitchf=pf)
        y = gen.run(["audio"], feeds)[0].reshape(-1); hop = len(y) // n; out_sr = hop * 100; fade = 5 * hop
        end = min(frames, start + seg); pieces.append(y[(start - a) * hop:min(len(y), (end - a) * hop + fade)])
    o = pieces[0]
    for y in pieces[1:]:
        r = np.linspace(0, 1, fade, dtype=np.float32); o = np.concatenate([o[:-fade], o[-fade:] * (1 - r) + y[:fade] * r, y[fade:]])
    sf.write(out, o[: int(len(x) / sr * out_sr)], out_sr, subtype="PCM_16")
render(feats, out_plain); render(acc, out_acc); print("wrote", out_plain, out_acc)
