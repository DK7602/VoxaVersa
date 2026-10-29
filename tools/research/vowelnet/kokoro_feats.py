"""Kokoro (Apache-2.0) American voices saying lyric lines: ContentVec features (50 Hz) + per-frame labels
(ARPAbet vowels, AY/OY split into the vowel and its glide at 70 %, EH before a nasal, C, SIL), as features.py.
Speech from eight voices widens what the recognizer has heard. Needs kokoro (pip), onnxruntime, scipy.
usage: kokoro_feats.py lines.txt contentvec.onnx out.npz"""
import os, sys, math
import numpy as np, onnxruntime as ort
from scipy.signal import resample_poly
from kokoro import KPipeline
MAP = {"æ": "AE", "ɛ": "EH", "ɪ": "IH", "i": "IY", "ɑ": "AA", "ʌ": "AH", "ə": "AX", "ᵊ": "AX", "ɔ": "AO", "A": "EY", "I": "AY",
       "Y": "OY", "O": "OW", "W": "AW", "u": "UW", "ʊ": "UH", "ɜ": "ER", "ɝ": "ER", "ɚ": "ER"}
NAS = set("nmŋ"); MARKS = set("ˈˌː")
lines = sorted({l.strip() for l in open(sys.argv[1]) if l.strip()})
so = ort.SessionOptions(); so.intra_op_num_threads = 1
enc = ort.InferenceSession(sys.argv[2], so, providers=["CPUExecutionProvider"])
X, Y, G = [], [], []
for voice in ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_michael", "am_adam", "am_eric", "am_liam"]:
    pipe = KPipeline(lang_code="a")
    for li, text in enumerate(lines):
        for r in pipe(text, voice=voice):
            audio = r.audio.numpy().astype(np.float32); d = r.pred_dur.tolist(); unit = len(audio) / sum(d) / 24000
            x16 = resample_poly(audio, 2, 3).astype(np.float32); n = int(math.ceil(len(x16) / 320)); x16 = np.pad(x16, (0, n * 320 + 80 - len(x16)))
            F = enc.run(None, {enc.get_inputs()[0].name: x16[None, None]})[0][0]
            labs = np.full(len(F), "SIL", dtype=object); t = d[0] * unit; ph = list(r.phonemes)
            for i, c in enumerate(ph):
                dur = d[i + 1] * unit; a, b = t, t + dur; t += dur
                if c in MARKS: continue
                if c == " ": lab = None
                elif c in MAP:
                    lab = MAP[c]
                    nxt = next((p for p in ph[i + 1:] if p not in MARKS), "")
                    if lab == "EH" and nxt in NAS: lab = "EH_N"
                elif c.isalpha() or ord(c) > 127: lab = "C"
                else: lab = None
                if lab is None: continue
                for k in range(int(a * 50), min(len(F), int(b * 50) + 1)):
                    tc = (k + 0.5) / 50
                    if a <= tc < b: labs[k] = lab + "_g" if lab in ("AY", "OY") and tc >= a + 0.7 * (b - a) else lab
            X.append(F.astype(np.float16)); Y += list(labs); G += ["kokoro-%s-%02d" % (voice, li)] * len(F)
    print(voice, sum(len(x) for x in X), flush=True)
np.savez_compressed(sys.argv[3], X=np.concatenate(X), Y=np.array(Y), G=np.array(G))
