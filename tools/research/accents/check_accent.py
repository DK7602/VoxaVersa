import os, sys, math, glob, os, numpy as np, soundfile as sf, onnxruntime as ort, torch, torch.nn as nn, re
sys.path.insert(0, "/home/user/VoxaVersa/tools/training")
from try_voice import resample
S = os.environ.get("SCRATCH", ".")
enc = ort.InferenceSession(S + "/audit2/home/.config/VoxaVersa/Encoders/contentvec-768.onnx", providers=["CPUExecutionProvider"])
m = torch.load(S + "/vowelnet/final2.pt", weights_only=False); cl = [str(c) for c in m["classes"]]; ci = {c: i for i, c in enumerate(cl)}
net = nn.Sequential(nn.Linear(len(m["mu"]), 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(cl))); net.load_state_dict(m["state"]); net.eval()
from faster_whisper import WhisperModel
asr = WhisperModel("small.en", device="cpu", compute_type="int8")
LY = "faithful friend and life is a marathon your friendship never ends like a runner who never quits laughing and joy and taking pain".split()
def wer(h):
    h = re.sub(r"[^a-z' ]", " ", h.lower()).split(); d = list(range(len(h) + 1))
    for i, r in enumerate(LY, 1):
        p, d[0] = d[:], i
        for j, hw in enumerate(h, 1): d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r != hw))
    return d[len(h)] / len(LY)
CHK = [("faithful ay", 0.08, 0.16, "EY", "EH"), ("pain ay", 13.98, 14.10, "EY", "EH"), ("is", 3.06, 3.46, "IH", "IY"),
       ("quits", 10.12, 10.52, "IH", "IY"), ("runner u", 8.40, 8.68, "AH", "AO"), ("runner -er", 8.78, 9.04, "ER", "AH")]
def post(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x16 = resample(x.mean(1), sr, 16000)
    n = int(math.ceil(len(x16) / 320)); x16p = np.pad(x16, (0, n * 320 + 80 - len(x16)))
    F = enc.run(None, {enc.get_inputs()[0].name: x16p[None, None]})[0][0]; T = len(F)
    W = np.concatenate([F[np.clip(np.arange(T) + s, 0, T - 1)] for s in range(-2, 3)], 1)
    with torch.no_grad(): P = torch.softmax(net(torch.tensor((W - m["mu"]) / m["sd"], dtype=torch.float32)), 1).numpy()
    segs, _ = asr.transcribe(x16, beam_size=5, language="en", condition_on_previous_text=False)
    return P, " ".join(s.text for s in segs)
for v in sys.argv[1:]:
    print("=====", v)
    for ver in ("plain", "light", "full"):
        P, txt = post("r17/%s-%s.wav" % (v, ver))
        cells = []
        for lab, a, b, src, dst in CHK:
            k0, k1 = int(a * 50), int(b * 50); cells.append("%s %s%.2f>%s%.2f" % (lab, src, P[k0:k1, ci[src]].mean(), dst, P[k0:k1, ci[dst]].mean()))
        print("%-5s WER %.2f | %s" % (ver, wer(txt), " | ".join(cells)))
        print("      " + txt.strip())
