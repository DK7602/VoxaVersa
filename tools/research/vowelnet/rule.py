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

# The other Southern vowels (step 2), each a shift of the content features by a difference of vowel means, as the
# approved prototype (tools/research/accent_features.py) made them, in shapes the plug-in can decide live:
#   raise (friend, ten: "eh" before n / m towards "ih"): the whole vowel, in and out over 3 frames;
#   lower (pain, taking: "ay" starts lower): from the vowel's start, in over 3 frames, gone after L frames;
#   drawl (and, laughing: "a" with a glide towards "ih"): a 160 ms bump from D0 frames into the vowel.
# A vowel gets at most one; each starts once the vowel so far (at least 3 frames) is that vowel by P(class | vowel).
TR, TL, TD, L, D0 = 0.4, 0.4, 0.4, 8, 5

def shifts(P, classes, voiced, TR=TR, TL=TL, TD=TD, L=L, D0=D0, V0=V0):
    """Per-frame weights (raise, lower, drawl) [T][3], as the plug-in leaves them (2 frames of lookahead; a start
    reaches back 2 frames)."""
    ci = {c: i for i, c in enumerate(classes)}
    vow = [ci[c] for c in classes if c not in ("C", "SIL")]
    v = P[:, vow].sum(1); T = len(P)
    rel = np.stack([P[:, ci["EH_N"]], P[:, ci["EY"]], P[:, ci["AE"]]], 1) / v[:, None]
    isv = np.append(np.asarray(voiced, bool) & (v >= V0), [False, False])
    w = np.zeros((T, 3), np.float32); need = (TR, TL, TD)
    on, start, sums = -1, 0, np.zeros(3)
    bump = lambda k: 0.0 if k < 0 else ((k + 1) / 4 if k < 4 else max(0.0, 1 - (k - 3) / 4))
    for d in range(T):
        if not isv[d]: on, sums = -1, np.zeros(3); continue
        if d == 0 or not isv[d - 1]: start = d
        i = d - start; sums += rel[d]
        out = 1 / 3 if not isv[d + 1] else (2 / 3 if not isv[d + 2] else 1.0)
        if on < 0 and i >= 2:
            means = sums / (i + 1); best = int(np.argmax(means - np.array(need)))
            if means[best] >= need[best]:
                on = best; drawl0 = max(d, start + D0)
                if on == 0:
                    for j in (d - 2, d - 1):
                        if j >= start: w[j, 0] = (j - (d - 2) + 1) / 3
                if on == 1:
                    for j in (d - 2, d - 1):
                        if j >= start: w[j, 1] = min(1, (j - start + 1) / 3) * max(0.0, 1 - (j - start) / L)
        if on == 0: w[d, 0] = out
        elif on == 1: w[d, 1] = min(1, (i + 1) / 3) * max(0.0, 1 - i / L) * out
        elif on == 2: w[d, 2] = bump(d - drawl0) * out
    return w

if __name__ == "__main__":
    net, classes, mu, sd = load(sys.argv[1]); d = np.load(sys.argv[2], allow_pickle=True)
    P = posteriors(net, mu, sd, W=d["W"]) if "W" in d else posteriors(net, mu, sd, d["F"].astype(np.float32))
    for s, g, e in holds(P, classes, d["voiced"]):
        words = [w for w, a, b in d["words"] if a - 0.05 <= s / 50 <= b + 0.05]
        print("%.2f s: vowel from %.2f to %.2f s held from %.2f s  %s" % (g / 50, s / 50, e / 50, (g - 2) / 50, " ".join(words)))
