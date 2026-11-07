"""The consonant recognizer's training data as the plug-in's live stream sees it: LibriSpeech utterances (as
consonant_feats.py: every phone labelled, vowels "VOW") through vowelnet/stream_windows.py's windows() - each frame's
5-frame window from the chunk that decides it, the chunk's context cycling through the stream's 1000 / 600 / 300 /
150 ms. Whole-file features alone hear a live "th" as "s" or silence and the "d" of "and" as "t".
usage: consonant_stream.py contentvec.onnx LibriSpeech/dev-clean textgrid_dir out.npz [every] [skip]
       (W [frames][5*768] float16, Y, G; `skip` offsets which utterances `every` takes)"""
import sys, os, re, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vowelnet"))
from stream_windows import windows
VOWELS = set("AA AE AH AO AW AX AY EH ER EY IH IY OW OY UH UW".split())

def tiers(path):
    """{tier name: [(start, end, text)]} of a Praat TextGrid."""
    out, name = {}, None
    for m in re.finditer(r'name = "([^"]*)"|xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', open(path).read()):
        if m.group(1) is not None: name = m.group(1); out[name] = []
        elif name is not None: out[name].append((float(m.group(2)), float(m.group(3)), m.group(4)))
    return out

every = int(sys.argv[5]) if len(sys.argv) > 5 else 1
skip = int(sys.argv[6]) if len(sys.argv) > 6 else 0
contexts = [1000, 600, 300, 150]
Ws, Y, G = [], [], []; n = 0; k = 0
for flac in sorted(glob.glob(os.path.join(sys.argv[2], "*", "*", "*.flac"))):
    uid = os.path.basename(flac)[:-5]; tg = os.path.join(sys.argv[3], uid + ".TextGrid"); n += 1
    if (n + skip) % every or not os.path.exists(tg): continue
    ctx = contexts[k % 4]; k += 1
    W = windows(flac, ctx, sys.argv[1])
    labs = np.full(len(W), "SIL", dtype=object)
    for a, b, p in tiers(tg).get("phones", []):
        base = p.rstrip("012")
        if base in ("", "sil", "sp", "spn"): continue
        for f in range(int(a * 50), min(len(W), int(b * 50) + 1)):
            if a <= (f + 0.5) / 50 < b: labs[f] = "VOW" if base in VOWELS else base
    speech = np.where(labs != "SIL")[0]
    if len(speech) == 0: continue
    keep = slice(max(0, speech[0] - 10), min(len(W), speech[-1] + 11))
    Ws.append(W[keep]); Y += list(labs[keep]); G += ["stream%d-libri-%s" % (ctx, uid)] * len(labs[keep])
    if len(Ws) % 25 == 0: print(len(Ws), sum(len(a) for a in Ws), flush=True)
np.savez(sys.argv[4], W=np.concatenate(Ws), Y=np.array(Y), G=np.array(G))
print("utterances", len(Ws), "frames", len(Y))
