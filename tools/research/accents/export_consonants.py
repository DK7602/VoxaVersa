"""Packs the consonant recognizer (vowelnet/train.py on consonant_feats.py / consonant_stream.py data) and the
accents' consonant templates into Resources/neural/accent_consonants.gz (AccentConsonants::load):
"VXVC", version 1, channels, context, hidden, classes (uint32 each), the class names (uint8 length + text), then
float16 w1 [hidden][context * channels] (the standardisation folded in), b1, w2 [classes][hidden], b2; then the
templates: their number (uint32), each its name (uint8 length + text), its frames (uint32) and float16
[frames][channels]:
  t            Kokoro's "t" (its "a tih", encoded alone): closure, closure, burst, aspiration
  d_a d_e d_i  Kokoro's "d" before "ah" / "eh" / "ih": voice bar x3, burst
  silence      the mean of LibriSpeech's silence frames
  roll, jh, l_light   (with latin_feats.npz, make_latin_templates.py) the Latin accent's rolled "rr" (3 frames),
               "j" (5) and light Spanish "l" (4: its "ala")
Kokoro-82M is Apache 2.0 (docs/THIRD_PARTY.md); the clips come from make_stops_full.py.
usage: export_consonants.py model.pt out.gz kokoro_feats.npz bursts.json libri_cons.npz [latin_feats.npz]
       (kokoro_feats.npz: k_d_a, k_d_e, k_d_i, k_t_i - the clips' ContentVec features, 50 Hz)"""
import sys, gzip, json, struct, numpy as np, torch
model, out_path, kpath, bpath, lpath = sys.argv[1:6]
m = torch.load(model, weights_only=False); st = m["state"]
W1, B1 = st["0.weight"].numpy().astype(np.float64), st["0.bias"].numpy().astype(np.float64)
W2, B2 = st["3.weight"].numpy().astype(np.float64), st["3.bias"].numpy().astype(np.float64)
mu, sd = m["mu"].astype(np.float64), m["sd"].astype(np.float64)
w1 = W1 / sd[None, :]; b1 = B1 - w1 @ mu
classes = [str(c) for c in m["classes"]]; ctx = 5; ch = W1.shape[1] // ctx

K = np.load(kpath); B = json.load(open(bpath))
burst = lambda n: int(round(B[n] * 50))
templates = [("t", K["k_t_i"][burst("t_i") - 2:burst("t_i") + 2])]
for v in ("a", "e", "i"):
    n = "d_" + v; templates.append((n, K["k_" + n][burst(n) - 3:burst(n) + 1]))
L = np.load(lpath); templates.append(("silence", L["X"][L["Y"] == "SIL"].astype(np.float64).mean(0, keepdims=True)))
if len(sys.argv) > 6:      # the Latin accent's: Kokoro's Spanish rolled "rr" ("perro") and the "j" of "a June"
    LT = np.load(sys.argv[6])
    templates += [("roll", LT["k_trill_perro"][18:21]), ("jh", LT["k_jh_june"][28:33]), ("l_light", LT["k_l_ala"][15:19])]

out = bytearray(b"VXVC") + struct.pack("<5I", 1, ch, ctx, W1.shape[0], len(classes))
for c in classes: out += struct.pack("<B", len(c)) + c.encode()
for a in (w1, b1, W2, B2): out += a.astype("<f2").tobytes()
out += struct.pack("<I", len(templates))
for name, frames in templates:
    out += struct.pack("<B", len(name)) + name.encode() + struct.pack("<I", len(frames)) + np.asarray(frames, np.float64).astype("<f2").tobytes()
    print("template %-7s %d frames" % (name, len(frames)))
with open(out_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as f:
    f.write(bytes(out))   # reproducible
print("classes", classes, "| %d bytes, %d gzipped" % (len(out), len(open(out_path, "rb").read())))
