"""Packs a trained vowel recognizer for the plug-in (Resources/neural/southern_vowels.gz, read by SouthernVowels):
"VXVW", version 1, channels, context, hidden, classes (uint32 LE), class names (uint8 length + text), then float16
w1 [hidden][context*channels] (the standardisation folded in), b1, w2 [classes][hidden], b2. With --shifts (vowel
means, as tools/research/accent_phone_means.py makes them: the approved prototype's), version 2 adds step 2's
shifts: their number, then each its name and float16 [channels]: raise = IH - EH_N, lower = EH - EY, drawl = IH - AE.
With features (features.py), checks that the packed float16 network decides as the trained one.
usage: export.py model.pt out.gz [features.npz] [--shifts means.npz]"""
import sys, gzip, struct, numpy as np, torch
args = sys.argv[1:]
means = None
if "--shifts" in args:
    k = args.index("--shifts"); means = dict(np.load(args[k + 1])); del args[k:k + 2]
sys.argv = [sys.argv[0]] + args
m = torch.load(sys.argv[1], weights_only=False); st = m["state"]
W1, B1 = st["0.weight"].numpy().astype(np.float64), st["0.bias"].numpy().astype(np.float64)
W2, B2 = st["3.weight"].numpy().astype(np.float64), st["3.bias"].numpy().astype(np.float64)
mu, sd = m["mu"].astype(np.float64), m["sd"].astype(np.float64)
w1 = W1 / sd[None, :]; b1 = B1 - w1 @ mu
classes = m["classes"]; ctx = 5; ch = W1.shape[1] // ctx
out = bytearray(b"VXVW") + struct.pack("<5I", 2 if means else 1, ch, ctx, W1.shape[0], len(classes))
for c in classes: out += struct.pack("<B", len(c)) + c.encode()
for a in (w1, b1, W2, B2): out += a.astype("<f2").tobytes()
if means:
    shifts = [("raise", means["IH"] - means["EH_N"]), ("lower", means["EH"] - means["EY"]), ("drawl", means["IH"] - means["AE"])]
    out += struct.pack("<I", len(shifts))
    for name, delta in shifts:
        out += struct.pack("<B", len(name)) + name.encode() + np.asarray(delta, np.float64).astype("<f2").tobytes()
        print("shift %s: |delta| %.2f" % (name, np.linalg.norm(delta)))
with open(sys.argv[2], "wb") as raw, gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as f: f.write(bytes(out))   # reproducible
print("classes", classes, "| max |w1| %.1f | %d bytes, %d gzipped" % (np.abs(w1).max(), len(out), len(open(sys.argv[2], "rb").read())))
if len(sys.argv) > 3:   # the packed (fp16) recognizer against the original on some frames
    d = np.load(sys.argv[3], allow_pickle=True); X = d["X"][:4000].astype(np.float32); G = d["G"][:4000]
    T, C = X.shape; Xc = np.zeros((T, ctx * C), np.float32)
    for o, s in enumerate(range(-2, 3)):
        idx = np.clip(np.arange(T) + s, 0, T - 1); same = G[idx] == G; Xc[:, o * C:(o + 1) * C] = np.where(same[:, None], X[idx], X)
    ref = np.maximum(0, ((Xc - mu) / sd) @ W1.T + B1) @ W2.T + B2
    h16 = lambda a: a.astype(np.float16).astype(np.float64)
    pk = np.maximum(0, Xc @ h16(w1).T + h16(b1)) @ h16(W2).T + h16(B2)
    print("packed vs original: same top class on %.2f%% of frames" % (100 * (ref.argmax(1) == pk.argmax(1)).mean()))
