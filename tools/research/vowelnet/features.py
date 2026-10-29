"""ContentVec features (50 Hz) + per-frame phone labels for every take rendered by render_takes.sh, from its
melody plan: ARPAbet vowels, the last 30 % of AY / OY as their glide (AY_g, OY_g), EH before a nasal (EH_N),
C (consonant) and SIL. Needs numpy, scipy, soundfile, onnxruntime.
usage: features.py takes_dir contentvec.onnx out.npz"""
import sys, os, glob, csv, math
import numpy as np, soundfile as sf, onnxruntime as ort
from scipy.signal import resample_poly
VOWELS = "AA AE AH AO AW AX AY EH ER EY IH IY OW OY UH UW".split()
NASAL = {"N", "M", "NG"}
def label_rows(plan):
    rows = list(csv.DictReader(open(plan), delimiter="\t"))
    out = []
    for i, r in enumerate(rows):
        ph = r["ph"]
        nxt = rows[i + 1]["ph"] if i + 1 < len(rows) and rows[i + 1]["word"] == r["word"] else ""
        if ph == "EH" and nxt in NASAL: ph = "EH_N"
        lab = ph if ph in VOWELS or ph == "EH_N" else ("SIL" if ph in ("SIL", "BR") else "C")
        out.append((float(r["start"]), float(r["end"]), lab))
    return out
if __name__ == "__main__":
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    enc = ort.InferenceSession(sys.argv[2], so, providers=["CPUExecutionProvider"])
    X, Y, G, T = [], [], [], []
    for wav in sorted(glob.glob(os.path.join(sys.argv[1], "*.wav"))):
        base = os.path.basename(wav)
        if base.startswith("guide-"): continue
        name = base[:-4]; plan = os.path.join(sys.argv[1], "plan-" + name + ".tsv")
        if not os.path.exists(plan): continue
        x, sr = sf.read(wav, dtype="float32", always_2d=True); x = x.mean(1)
        g = math.gcd(int(sr), 16000); x16 = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
        n = int(math.ceil(len(x16) / 320)); x16 = np.pad(x16, (0, n * 320 + 80 - len(x16)))
        F = enc.run(None, {enc.get_inputs()[0].name: x16[None, None]})[0][0]
        labs = np.full(len(F), "SIL", dtype=object)
        for a, b, lab in label_rows(plan):
            for k in range(int(a * 50), min(len(F), int(b * 50) + 1)):
                tc = (k + 0.5) / 50
                if a <= tc < b:
                    # a diphthong's glide (Text to Sing sings it at the end of the note): its last 30 %
                    labs[k] = lab + "_g" if lab in ("AY", "OY") and tc >= a + 0.7 * (b - a) else lab
        X.append(F.astype(np.float16)); Y += list(labs); G += [name] * len(F); T += list(np.arange(len(F)))
        print(name, len(F), flush=True)
    np.savez_compressed(sys.argv[3], X=np.concatenate(X), Y=np.array(Y), G=np.array(G), T=np.array(T))
