"""Real voices for the vowel recognizer: LibriSpeech (read speech, CC BY 4.0) utterances with their phone timings
from the LibriSpeech Alignments (L. Lugosch et al., Montreal Forced Aligner, CC BY 4.0, zenodo.org/records/2619474):
ContentVec features (50 Hz) and per-frame labels as features.py's (AH0 is AX; EH before n / m in the same word is
EH_N; the last 30 % of AY / OY their glide). Silence beyond 0.2 s at either end is left out; `every` takes one
utterance in so many. (Aligning with this recognizer itself failed: it was too unsure of real voices and gave
vowels one frame.) Needs numpy, soundfile, onnxruntime.
usage: libri_feats.py contentvec.onnx LibriSpeech/dev-clean textgrid_dir out.npz [every]"""
import sys, os, re, glob, math, numpy as np, soundfile as sf, onnxruntime as ort

so = ort.SessionOptions(); so.intra_op_num_threads = 2
enc = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
every = int(sys.argv[5]) if len(sys.argv) > 5 else 1
VOWELS = set("AA AE AH AO AW AX AY EH ER EY IH IY OW OY UH UW".split())

def tiers(path):
    """{tier name: [(start, end, text)]} of a Praat TextGrid."""
    out, name = {}, None
    for m in re.finditer(r'name = "([^"]*)"|xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', open(path).read()):
        if m.group(1) is not None: name = m.group(1); out[name] = []
        elif name is not None: out[name].append((float(m.group(2)), float(m.group(3)), m.group(4)))
    return out

X, Y, G = [], [], []; n = 0
for flac in sorted(glob.glob(os.path.join(sys.argv[2], "*", "*", "*.flac"))):
    uid = os.path.basename(flac)[:-5]; tg = os.path.join(sys.argv[3], uid + ".TextGrid"); n += 1
    if n % every or not os.path.exists(tg): continue
    t = tiers(tg); phones, words = t.get("phones", []), [w for w in t.get("words", []) if w[2]]
    x, sr = sf.read(flac, dtype="float32")
    T = int(math.ceil(len(x) / 320)); x = np.pad(x, (0, T * 320 + 80 - len(x)))
    F = enc.run(None, {enc.get_inputs()[0].name: x[None, None]})[0][0]
    labs = np.full(len(F), "SIL", dtype=object)
    word_of = lambda a: next((i for i, w in enumerate(words) if w[0] - 1e-3 <= a < w[1] - 1e-3), -1)
    for k, (a, b, p) in enumerate(phones):
        base = p.rstrip("012")
        if base in ("", "sil", "sp", "spn"): continue
        if base in VOWELS:
            lab = "AX" if base == "AH" and p.endswith("0") else base
            nxt = phones[k + 1] if k + 1 < len(phones) else None
            if base == "EH" and nxt is not None and nxt[2] in ("N", "M") and word_of(nxt[0]) == word_of(a): lab = "EH_N"
        else: lab = "C"
        for f in range(int(a * 50), min(len(F), int(b * 50) + 1)):
            tc = (f + 0.5) / 50
            if a <= tc < b: labs[f] = lab + "_g" if lab in ("AY", "OY") and tc >= a + 0.7 * (b - a) else lab
    speech = np.where(labs != "SIL")[0]
    if len(speech) == 0: continue
    keep = slice(max(0, speech[0] - 10), min(len(F), speech[-1] + 11))
    X.append(F[keep].astype(np.float16)); Y += list(labs[keep]); G += ["libri-" + uid] * len(labs[keep])
    if len(X) % 100 == 0: print(len(X), sum(len(a) for a in X), flush=True)
np.savez_compressed(sys.argv[4], X=np.concatenate(X), Y=np.array(Y), G=np.array(G))
print("utterances", len(X), "frames", len(Y))
