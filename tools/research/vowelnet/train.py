"""Frame vowel classifier on ContentVec features: 5 frames of context (±40 ms), standardised, MLP 3840 -> 256 -> K.
Regularised for singers it never heard: input noise and dropout, label smoothing. Trains on whole-file features
(features.py, kokoro_feats.py: the windows are gathered here) and on the windows the live stream gives it
(stream_windows.py), all kept as float16. usage: train.py out.pt data.npz[,data2.npz...] [held-out voices...]
(env EPOCHS, NOISE, SMOOTH, THREADS)"""
import sys, os, re, numpy as np, torch, torch.nn as nn
EPOCHS, NOISE, SMOOTH = int(os.environ.get("EPOCHS", 20)), float(os.environ.get("NOISE", 0.5)), float(os.environ.get("SMOOTH", 0.1))
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(int(os.environ.get("THREADS", 2)))
paths = sys.argv[2].split(",")
Y = np.concatenate([np.load(p, allow_pickle=True)["Y"] for p in paths]); G = np.concatenate([np.load(p, allow_pickle=True)["G"] for p in paths])
T = len(Y); W = np.zeros((T, 5 * 768), np.float16); off = 0   # one array, filled part by part (memory)
for path in paths:
    d = np.load(path, allow_pickle=True)
    if "W" in d:
        w = d["W"]; W[off:off + len(w)] = w; off += len(w); del w
    else:   # frames t-2..t+2 of the same take (the take's edge frame repeated)
        X, g = d["X"].astype(np.float16), d["G"]; n, C = X.shape
        for o, s in enumerate(range(-2, 3)):
            idx = np.clip(np.arange(n) + s, 0, n - 1); idx = np.where(g[idx] == g, idx, np.arange(n))
            W[off:off + n, o * C:(o + 1) * C] = X[idx]
        off += n; del X
held = sys.argv[3:]
classes = sorted(set(Y)); ci = {c: i for i, c in enumerate(classes)}; y = np.array([ci[c] for c in Y])
voice = lambda g: re.sub(r"^(dry-|stream\d+-)", "", g)
test = np.array([any(voice(g).startswith(h + "-") for h in held) for g in G]) if held else np.zeros(T, bool)
sample = np.random.choice(np.where(~test)[0], min(60000, int((~test).sum())), replace=False)
mu = W[sample].astype(np.float32).mean(0); sd = W[sample].astype(np.float32).std(0) + 1e-5
window = lambda i: torch.tensor((W[i].astype(np.float32) - mu) / sd)
counts = np.bincount(y[~test], minlength=len(classes)); w = torch.tensor((counts.sum() / (len(classes) * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32)
net = nn.Sequential(nn.Linear(W.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(classes)))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3); lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=SMOOTH)
tr = np.where(~test)[0]; yt = torch.tensor(y)
check = np.union1d(np.where(test)[0], sample)   # accuracy is measured on these: the held-out frames, a sample of the rest
def predict():
    net.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(check), 4096): out.append(torch.softmax(net(window(check[b:b + 4096])), 1))
    return torch.cat(out).numpy()
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr)
    for b in range(0, len(tr), 256):
        i = tr[b:b + 256]; xb = window(i); xb = nn.functional.dropout(xb + NOISE * torch.randn_like(xb), 0.2)
        opt.zero_grad(); l = lossf(net(xb), yt[i]); l.backward(); opt.step()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        P = predict(); p = P.argmax(1); yc, tc = y[check], test[check]
        acc = lambda m: (p[m] == yc[m]).mean() if m.any() else float("nan")
        vow = np.array([c not in ("C", "SIL") for c in Y[check]])
        print("epoch %d: train acc %.3f, held-out acc %.3f (vowel frames %.3f)" % (ep + 1, acc(~tc), acc(tc), acc(tc & vow)), flush=True)
        torch.save(dict(state=net.state_dict(), classes=classes, mu=mu, sd=sd, epoch=ep + 1), sys.argv[1])   # a checkpoint
torch.save(dict(state=net.state_dict(), classes=classes, mu=mu, sd=sd), sys.argv[1])
if test.any():
    for c in ("AY", "AY_g", "OY", "OY_g", "EH_N", "EY", "AE", "AA", "AH", "AW", "OW"):
        if c not in ci: continue
        m = tc & (yc == ci[c]); other = tc & (yc != ci[c]) & vow
        print("%-5s held-out: recall %.2f (frames %d) | false alarms on other vowels %.3f" % (
            c, (P[m].argmax(1) == ci[c]).mean() if m.any() else float("nan"), m.sum(), (P[other].argmax(1) == ci[c]).mean()))
