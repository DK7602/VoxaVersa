"""Prototype: a Southern US accent on a sung vocal, before the AI singer. The AI singer sings the vowels it hears,
so the input's vowels are reshaped (WORLD analysis / resynthesis, formants measured with Praat):
  /aI/ (I, my, life, like)      -> [a:]  the glide is dropped: the vowel holds its "ah"
  /OI/ (joy)                    -> [O:]  the same
  /E/ before n, m (friend, end) -> [I]   pin-pen merger
  /eI/ (pain, taking, faithful) -> [EI]  lower start (Southern vowel shift)
  /ae/ (laughing, held "and")   -> [aeIe] drawl: a glide towards "ee" and back
Research prototype (listening audit 10), not part of the plug-in: the target words come from a speech
recognizer's word timings (word_timings.py) and the CMU dictionary; a plug-in version needs its own live vowel
detection. Needs numpy, scipy, soundfile, pyworld.
usage: southern_accent.py in.wav words.json Resources/sing/cmudict.gz out.wav [out-resynth.wav]"""
import sys, json, gzip, re
import numpy as np, soundfile as sf, pyworld as pw

src, words_path, dict_path, out_path = sys.argv[1:5]
x, fs = sf.read(src); x = (x.mean(1) if x.ndim > 1 else x).astype(np.float64)
FP = 5.0
f0, t = pw.harvest(x, fs, frame_period=FP, f0_floor=60, f0_ceil=1000)
sp = pw.cheaptrick(x, f0, t, fs); ap = pw.d4c(x, f0, t, fs)
if len(sys.argv) > 5:
    sf.write(sys.argv[5], pw.synthesize(f0, sp, ap, fs, frame_period=FP).astype(np.float32), fs)
n, bins = sp.shape
freqs = np.arange(bins) * fs / (2 * (bins - 1))
E = 10 * np.log10(sp.sum(1) + 1e-20)

def env_formants(env, fs):
    """F1-F3 as the peaks of a vowel's smooth spectral envelope (WORLD's has no harmonics in it): F1 the
    strongest peak at 250-950 Hz, F2 at F1+250 - 2700 Hz, F3 at F2+300 - 3800 Hz."""
    hz = np.arange(len(env)) * fs / (2 * (len(env) - 1)); L = np.log(env + 1e-30)
    def pk(lo, hi):
        i = np.where((hz >= lo) & (hz <= hi))[0]
        if len(i) < 3: return None
        j = [k for k in i[1:-1] if L[k] >= L[k - 1] and L[k] >= L[k + 1]]
        return hz[max(j, key=lambda k: L[k])] if j else None
    f1 = pk(250, 950)
    if f1 is None: return None
    f2 = pk(max(800, f1 + 250), 2700)
    if f2 is None: return None
    f3 = pk(f2 + 300, 3800)
    return [f1, f2, f3] if f3 is not None else None
def formants(k):
    return env_formants(sp[k], fs)

lines = gzip.open(dict_path, "rt", encoding="latin-1").read().split("\n")
syms = lines[0].split()[2:]
lex = {}
for line in lines[1:]:
    if "\t" in line:
        w, c = line.split("\t"); lex[w] = [syms[ord(ch) - 0x30] for ch in c]

def rule(word, phones, dur):
    """(vowel index, kind) for the stressed vowel this word changes, or None."""
    vow = [i for i, p in enumerate(phones) if p[-1].isdigit()]
    if word == "and" and dur >= 0.25: return 0, "drawl"     # a held "and" is sung with a full "a"
    for vi, i in enumerate(vow):
        p, nxt = phones[i][:-1], (phones[i + 1] if i + 1 < len(phones) else "")
        stressed = phones[i][-1] in "12" or len(vow) == 1
        if not stressed: continue
        if p == "AY": return vi, "mono"
        if p == "OY": return vi, "mono"
        if p == "EH" and nxt in ("N", "M"): return vi, "raise"
        if p == "EY": return vi, "lower"
        if p == "AE": return vi, "drawl"
    return None

def nuclei(a, b):
    """Voiced runs within frames [a, b) at most 12 dB below the loudest voiced frame there."""
    idx = [k for k in range(a, b) if f0[k] > 0]
    if not idx: return []
    top = max(E[k] for k in idx); runs, cur = [], []
    for k in range(a, b):
        if f0[k] > 0 and E[k] >= top - 12: cur.append(k)
        elif cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    merged = []   # a dip of up to 30 ms does not split a vowel
    for r in runs:
        if merged and r[0] - merged[-1][-1] <= 6: merged[-1] = merged[-1] + list(range(merged[-1][-1] + 1, r[-1] + 1))
        else: merged.append(r)
    return [(r[0], r[-1] + 1) for r in merged if len(r) >= 6]

def vowel_formants(k0, k1):
    """F1-F3 of a vowel: medians over its frames (a sung voice's per-frame tracks jump)."""
    v = [f for f in (formants(k) for k in range(k0, k1)) if f and 200 < f[0] < 1100 and 700 < f[1] < 3000 and f[2] > f[1] + 300]
    return None if len(v) < 4 else list(np.median(np.array(v), axis=0))

def warp(k, f, r1, r2, w):
    """Frame k's envelope with F1, F2 (the vowel's, `f`) scaled by r1, r2 (F3 and above kept), by weight w."""
    if f is None or w <= 0: return
    F1, F2, F3 = f
    T1, T2 = F1 * (1 + (r1 - 1) * w), F2 * (1 + (r2 - 1) * w)
    if not (T1 < T2 - 200 and T2 < F3 - 150): return
    src_pts = [0, F1, F2, F3, fs / 2]; dst_pts = [0, T1, T2, F3, fs / 2]
    source_freq = np.interp(freqs, dst_pts, src_pts)        # where each output frequency reads from
    sp[k] = np.exp(np.interp(source_freq, freqs, np.log(sp[k])))

def ramp(i, m, edge):   # 0 -> 1 over `edge` frames at both ends of m frames
    return min(1.0, (i + 1) / edge, (m - i) / edge)

edits = []
for wd in json.load(open(words_path)):
    word = re.sub(r"[^a-z']", "", wd["word"].lower())
    phones = lex.get(word)
    if not phones: continue
    r = rule(word, phones, wd["end"] - wd["start"])
    if r is None: continue
    vi, kind = r
    a, b = int(wd["start"] * 1000 / FP), min(n, int(wd["end"] * 1000 / FP) + 1)
    runs = nuclei(a, b)
    if not runs: continue
    nv = sum(1 for p in phones if p[-1].isdigit())
    n0, n1 = runs[vi] if len(runs) == nv and vi < len(runs) else max(runs, key=lambda s: s[1] - s[0])
    m = n1 - n0
    if kind == "mono":   # hold the colour of its steadiest 40 ms in the first half, keep the level contour
        lo, hi = n0 + max(2, int(0.1 * m)), n0 + max(10, int(0.5 * m))
        d = np.linalg.norm(np.diff(np.log(sp[lo:hi]), axis=0), axis=1)
        best = min(range(max(1, len(d) - 7)), key=lambda i: d[i:i + 8].sum())
        h0 = lo + best; h1 = h0 + 8
        shape = np.log(sp[h0:h1]).mean(0); level = 10 * np.log10(np.exp(shape).sum())
        for k in range(h1 - 6, n1 - 2):
            w = min(1.0, (k - (h1 - 6)) / 8)
            new = shape + (E[k] - level) / 10 * np.log(10)
            sp[k] = np.exp((1 - w) * np.log(sp[k]) + w * new)
    elif kind == "raise":
        f = vowel_formants(n0 + m // 5, n1 - m // 5)
        for i, k in enumerate(range(n0, n1)): warp(k, f, 0.74, 1.13, ramp(i, m, 8))
    elif kind == "lower":
        f = vowel_formants(n0 + m // 10, n0 + m // 2)
        for i, k in enumerate(range(n0, n1)):
            pos = i / max(1, m - 1)
            warp(k, f, 1.22, 0.88, ramp(i, m, 6) * max(0.0, 1 - pos / 0.6))
    elif kind == "drawl":
        f = vowel_formants(n0 + m // 10, n0 + m // 2)
        for i, k in enumerate(range(n0, n1)):
            pos = i / max(1, m - 1)
            w = max(0.0, 1 - abs(pos - 0.65) / 0.25)        # a glide peaking at 65 % of the vowel
            warp(k, f, 0.75, 1.12, w)
    else:
        f = None
    if kind != "mono" and f is None: print("  (no steady formants in %s: unchanged)" % word)
    edits.append(dict(word=word, kind=kind, start=round(n0 * FP / 1000, 3), end=round(n1 * FP / 1000, 3)))

# only the reshaped vowels are resynthesized: everywhere else the vocal is untouched (WORLD alone made the AI
# singer lose words), crossfading 25 ms into and out of each vowel
z = pw.synthesize(f0, sp, ap, fs, frame_period=FP)[:len(x)]
z = np.pad(z, (0, len(x) - len(z)))
mix = np.zeros(len(x)); fade = int(0.025 * fs)
for e in edits:
    a0, b0 = int((e["start"] - 0.02) * fs), int((e["end"] + 0.02) * fs)
    mix[max(0, a0):min(len(x), b0)] = 1.0
mix = np.convolve(mix, np.ones(fade) / fade, mode="same")
y = x * (1 - mix) + z * mix
sf.write(out_path, y.astype(np.float32), fs)
for e in edits: print("%-11s %-6s %6.2f - %5.2f s" % (e["word"], e["kind"], e["start"], e["end"]))
