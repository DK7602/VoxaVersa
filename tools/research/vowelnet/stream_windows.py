"""The recognizer's input as the plug-in's live stream sees it. The stream runs the content encoder on chunks
(one every 400 ms; 40 frames of new audio + an even 9 + context frames before them, + 80 zero samples), so a
frame's features depend on the chunk: SouthernVowels recognises each 20 ms frame in the first chunk that has
the 2 frames after it, from that chunk's frames t-2..t+2. Whole-file features (features.py) are not what it
sees; a recognizer trained only on them missed "like" and "joy" in the plug-in.

Takes (with plan-<name>.tsv beside them, labelled as features.py) cycle through the stream's contexts (1000 ms:
exports; 600 / 300 / 150 ms: live, by CPU); a real vocal prepared by real_prep.py gets one context for all.
usage: stream_windows.py contentvec.onnx out.npz takes_dir...           (training: W [frames][5*768], Y, G)
       stream_windows.py contentvec.onnx out.npz --real contextMs vocal.wav vocal.npz   (adds F-less W to a copy)"""
import sys, os, glob, math, numpy as np, soundfile as sf, onnxruntime as ort
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import label_rows

so = ort.SessionOptions(); so.intra_op_num_threads = 2
enc = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])

def windows(wav, context_ms):
    """[T][5*768] float16: each encoder frame's window from the chunk that decides it."""
    x, sr = sf.read(wav, dtype="float32", always_2d=True); x = x.mean(1)
    g = math.gcd(int(sr), 16000); x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    n10 = len(x) // 160; T = n10 // 2; chunk = 40 + ((9 + context_ms // 10 + 1) & ~1)
    W = np.zeros((T, 5 * 768), np.float16); done = -1
    for end in range(40, n10 + 80, 40):
        start = end - chunk; a = np.zeros(chunk * 160 + 80, np.float32)
        s0, s1 = max(0, start * 160), min(len(x), end * 160)
        if s1 > s0: a[s0 - start * 160:s1 - start * 160] = x[s0:s1]
        f = enc.run(None, {enc.get_inputs()[0].name: a[None, None]})[0][0]; first = start // 2; E = len(f)
        for t in range(max(done + 1, first), min(T, first + E - 2)):   # as SouthernVowels::apply
            e = t - first
            W[t] = np.concatenate([f[min(max(e + o, 0), E - 1)] for o in range(-2, 3)])
            done = t
    return W

if __name__ == "__main__":
    if sys.argv[3] == "--real":
        ctx, wav, vocal = int(sys.argv[4]), sys.argv[5], sys.argv[6]
        d = dict(np.load(vocal, allow_pickle=True)); W = windows(wav, ctx); n = min(len(W), len(d["F"]))
        d["W"] = W[:n]; d["F"] = d["F"][:n]; d["voiced"] = d["voiced"][:n]
        np.savez(sys.argv[2], **d); print(os.path.basename(wav), ctx, "ms:", n, "frames")
        sys.exit()
    Ws, Y, G = [], [], []
    contexts = [1000, 600, 300, 150]; k = 0
    for takes in sys.argv[3:]:
        for wav in sorted(glob.glob(os.path.join(takes, "*.wav"))):
            name = os.path.basename(wav)[:-4]; plan = os.path.join(takes, "plan-" + name + ".tsv")
            if name.startswith("guide-") or not os.path.exists(plan): continue
            ctx = contexts[k % 4]; k += 1
            W = windows(wav, ctx); labs = np.full(len(W), "SIL", dtype=object)
            for a, b, lab in label_rows(plan):
                for t in range(int(a * 50), min(len(W), int(b * 50) + 1)):
                    tc = (t + 0.5) / 50
                    if a <= tc < b: labs[t] = lab + "_g" if lab in ("AY", "OY") and tc >= a + 0.7 * (b - a) else lab
            keep = np.array([l not in ("SIL",) or t % 3 == 0 for t, l in enumerate(labs)])   # a third of the silence
            Ws.append(W[keep]); Y += list(labs[keep]); G += ["stream%d-%s" % (ctx, name)] * int(keep.sum())
            print(name, ctx, "ms", int(keep.sum()), flush=True)
    np.savez(sys.argv[2], W=np.concatenate(Ws), Y=np.array(Y), G=np.array(G))
