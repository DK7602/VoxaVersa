import os, sys, math, json, numpy as np, soundfile as sf, onnxruntime as ort, torch, torch.nn as nn
sys.path.insert(0, "/home/user/VoxaVersa/tools/training")
from try_voice import resample, pitch_yin
S = os.environ.get("SCRATCH", ".")
enc = ort.InferenceSession(S + "/audit2/home/.config/VoxaVersa/Encoders/contentvec-768.onnx", providers=["CPUExecutionProvider"])
x, sr = sf.read(S + "/audit2/excerpt.wav", dtype="float32", always_2d=True); x = x.mean(1); x16 = resample(x, sr, 16000)
frames = int(math.ceil(len(x16) / 160)); frames += frames % 2; x16 = np.pad(x16, (0, frames * 160 + 80 - len(x16)))
f0 = pitch_yin(x16); F = enc.run(None, {enc.get_inputs()[0].name: x16[None, None]})[0][0]; T = len(F)
m = torch.load(S + "/vowelnet/final2.pt", weights_only=False); classes = [str(c) for c in m["classes"]]
net = nn.Sequential(nn.Linear(len(m["mu"]), 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(classes))); net.load_state_dict(m["state"]); net.eval()
W = np.concatenate([F[np.clip(np.arange(T) + s, 0, T - 1)] for s in range(-2, 3)], 1)
with torch.no_grad(): P = torch.softmax(net(torch.tensor((W - m["mu"]) / m["sd"], dtype=torch.float32)), 1).numpy()
e = np.array([10 * np.log10(np.mean(x16[k * 320:(k + 1) * 320] ** 2) + 1e-10) for k in range(T)])
np.savez("diag.npz", P=P, f0=f0, e=e, classes=np.array(classes))
for a, b in [(float(v.split("-")[0]), float(v.split("-")[1])) for v in sys.argv[1:]]:
    print("---- %.2f-%.2f" % (a, b))
    for k in range(int(a * 50), int(b * 50)):
        v = f0[2 * k:2 * k + 2]; top = np.argsort(P[k])[::-1][:3]
        print("%.2f %s %5.1fdB %s" % (k / 50, "V" if (v > 0).any() else "-", e[k], " ".join("%s%.2f" % (classes[i], P[k, i]) for i in top)))
