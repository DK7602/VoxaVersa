"""How the vowel recognizer, run as the plug-in's stream runs it, hears "er" and the "u" of "book" in sung takes, and
how often a rule for a real singer's "er" would fire on each vowel.

A user's "girl", live, came out as "u" (P(u | vowel) 0.25 - 0.6) with "er" second (~0.15): the stream's encoder
hears only ~60 ms after each frame, and the user's voice is not one the recognizer learnt from. On the AI singers'
takes (stream windows from stream_windows.py) "er" is clear (75 % of its frames >= 0.45) and "u" frames rarely reach
0.12. SouthernVowels::AccentRule takes a vowel as "er" once 2 frames in a row are either a clear "er" or a "u" on top
with P(er | vowel) >= 0.10 and its mean over the vowel so far >= 0.12 (from the vowel's 3rd frame); this prints how
often that fires per labelled vowel (at least 2 frames), against the clear-"er" rule alone.
usage: stream_er_uh.py recognizer.pt stream.npz        (stream.npz: stream_windows.py's W, Y, G)"""
import sys, collections, numpy as np, torch, torch.nn as nn

m = torch.load(sys.argv[1], weights_only=False)
cl = [str(c) for c in m["classes"]]; ci = {c: i for i, c in enumerate(cl)}
net = nn.Sequential(nn.Linear(len(m["mu"]), 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(cl)))
net.load_state_dict(m["state"]); net.eval()
z = np.load(sys.argv[2]); W, Y, G = z["W"], z["Y"], z["G"]
P = np.zeros((len(W), len(cl)), np.float32)
with torch.no_grad():
    for a in range(0, len(W), 4096):
        x = (W[a:a + 4096].astype(np.float32) - m["mu"]) / m["sd"]
        P[a:a + 4096] = torch.softmax(net(torch.tensor(x, dtype=torch.float32)), 1).numpy()

vow = [ci[c] for c in cl if c not in ("C", "SIL")]
plain = [ci[c] for c in cl if c not in ("C", "SIL", "AY_g", "OY_g")]
pv = P[:, vow].sum(1); er = P[:, ci["ER"]] / np.maximum(pv, 1e-6); uh = P[:, ci["UH"]] / np.maximum(pv, 1e-6)
top = np.array(plain)[P[:, plain].argmax(1)]; share = P[np.arange(len(P)), top] / np.maximum(pv, 1e-6)

for lab in ["ER", "UH", "AX", "AH", "AO", "OW", "UW", "AA", "EH", "IH", "IY"]:
    k = (Y == lab) & (pv >= 0.5)
    if k.sum():
        tops = collections.Counter(cl[t] for t in top[k]).most_common(3)
        print("%-3s %5d frames  P(er|vowel) median %.2f, >= 0.45 %3.0f %%, >= 0.12 %3.0f %% | P(u|vowel) median %.2f | top %s"
              % (lab, k.sum(), np.median(er[k]), 100 * (er[k] >= 0.45).mean(), 100 * (er[k] >= 0.12).mean(),
                 np.median(uh[k]), tops))

def fires(a, b, heard_as_u):
    """Frames of [a, b) the rule takes as "er" (the plug-in latches after 2 in a row; this counts them)."""
    run, total, n = 0, 0.0, 0
    for t in range(a, b):
        if pv[t] < 0.5:
            run, total = 0, 0.0
            continue
        run += 1; total += er[t]
        clear = cl[top[t]] == "ER" and share[t] >= 0.45
        u = heard_as_u and cl[top[t]] == "UH" and er[t] >= 0.10 and run >= 3 and total / run >= 0.12
        n += clear or u
    return n

segments, a = [], 0
for i in range(1, len(Y) + 1):
    if i == len(Y) or Y[i] != Y[a] or G[i] != G[a]:
        segments.append((a, i, Y[a])); a = i
for name, heard in (("clear \"er\" only", False), ("+ \"er\" heard as \"u\"", True)):
    count = collections.defaultdict(lambda: [0, 0])
    for a, b, lab in segments:
        if lab not in ("SIL", "C") and b - a >= 3:
            count[lab][0] += 1; count[lab][1] += fires(a, b, heard) >= 2
    print("%-22s" % name, " | ".join("%s %d/%d" % (l, count[l][1], count[l][0])
                                     for l in ["ER", "UH", "AX", "UW", "OW", "AO", "AH", "AA", "EH", "IH", "IY", "EY", "AY"]))
