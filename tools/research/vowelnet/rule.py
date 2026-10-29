"""The plug-in's Southern "ah" decision (Source/neural/SouthernVowels.cpp, Rule) on a whole vocal, for evaluating
and tuning: the recognizer's probabilities per 20 ms frame, then the rule frame by frame with 2 frames of lookahead.
Probabilities are taken relative to P(vowel). In a voiced vowel the glide starts at frame d when d and d + 1 sound
like the glide, or (a glide too short to see) when the vowel ends after d + 1; it is held when the vowel's frames
before d (glide frames aside) average P(I/oy | vowel) >= E0 over at least 3 frames and their last 3 >= N0.
usage: rule.py model.pt vocal.npz   (from real_prep.py) -> the holds"""
import sys, numpy as np, torch, torch.nn as nn

V0, G0, E0, N0 = 0.5, 0.4, 0.4, 0.3   # SouthernVowels::Rule: vowelNeeded, glideNeeded, evidenceNeeded, nucleusNeeded

def load(path):
    m = torch.load(path, weights_only=False)
    net = nn.Sequential(nn.Linear(len(m["mu"]), 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(m["classes"])))
    net.load_state_dict(m["state"]); net.eval(); return net, m["classes"], m["mu"], m["sd"]

def posteriors(net, mu, sd, F=None, W=None):
    """P(class) per frame: of whole-file features F [T][768] (frames t-2..t+2, edges repeated), or of the
    windows W [T][5*768] the live stream gives the recognizer (stream_windows.py)."""
    if W is None:
        T, C = F.shape; W = np.zeros((T, 5 * C), np.float32)
        for o, s in enumerate(range(-2, 3)): W[:, o * C:(o + 1) * C] = F[np.clip(np.arange(T) + s, 0, T - 1)]
    out = []
    with torch.no_grad():
        for b in range(0, len(W), 4096):
            out.append(torch.softmax(net(torch.tensor((W[b:b + 4096].astype(np.float32) - mu) / sd)), 1).numpy())
    return np.concatenate(out)

def holds(P, classes, voiced, V0=V0, G0=G0, E0=E0, N0=N0):
    """(vowel start, glide start, vowel end) of every vowel held; the hold runs from 2 frames before the glide."""
    ci = {c: i for i, c in enumerate(classes)}
    vow = [ci[c] for c in classes if c not in ("C", "SIL")]
    ay = [ci[c] for c in ("AY", "OY") if c in ci]; gl = [ci[c] for c in ("AY_g", "OY_g") if c in ci]
    v = P[:, vow].sum(1); a = P[:, ay].sum(1) / v; g = np.append(P[:, gl].sum(1) / v, [0, 0]); T = len(P)
    isv = np.append(np.asarray(voiced, bool) & (v >= V0), [False, False])
    out, evidence, nucleus, held, start = [], [], [], False, 0
    for d in range(T):
        if not isv[d]: evidence, nucleus, held = [], [], False; continue
        if d == 0 or not isv[d - 1]: start = d
        if held: continue
        if ((g[d] >= G0 and isv[d + 1] and g[d + 1] >= G0) or (isv[d + 1] and not isv[d + 2])) \
                and len(nucleus) >= 3 and np.mean(evidence) >= E0 and np.mean(nucleus[-3:]) >= N0:
            e = d
            while e < T and isv[e]: e += 1
            out.append((start, d, e)); held = True; continue
        if g[d] < G0: evidence.append(a[d] + g[d]); nucleus.append(a[d])
    return out

if __name__ == "__main__":
    net, classes, mu, sd = load(sys.argv[1]); d = np.load(sys.argv[2], allow_pickle=True)
    P = posteriors(net, mu, sd, W=d["W"]) if "W" in d else posteriors(net, mu, sd, d["F"].astype(np.float32))
    for s, g, e in holds(P, classes, d["voiced"]):
        words = [w for w, a, b in d["words"] if a - 0.05 <= s / 50 <= b + 0.05]
        print("%.2f s: vowel from %.2f to %.2f s held from %.2f s  %s" % (g / 50, s / 50, e / 50, (g - 2) / 50, " ".join(words)))
