"""Prototype: Jamaican (reggae), Nigerian (afrobeats) and Spanish (latin) accents made inside the AI singer, on its
content features (ContentVec, 50 Hz), from the known lyric. Each word's sounds are placed on the frames with the
vowel recognizer (Viterbi over the word's CMUdict sounds: vowels by their class, consonants as "C"), then:
  light: "ay" held without its glide (take -> tehk; also "oh"), "th" -> "t" / "d" (+ mean t - mean th)
  full:  light + the accent's vowels: "i" -> "ee" (quits -> queets), "u" in runner -> "o"/"ah", "a" in and -> "ah",
         and (reggae, afrobeats) no "r" after a vowel (never -> neva, your -> yo)
then the voice's generator, as try_voice.py does (YIN pitch).
usage: accent_thick.py model.onnx contentvec.onnx in.wav words.json cmudict.gz recognizer.pt means_libri.npz means2.npz
                  accent transpose out_prefix"""
import sys, math, json, gzip, re
import numpy as np, soundfile as sf, onnxruntime as ort, torch, torch.nn as nn
sys.path.insert(0, "/home/user/VoxaVersa/tools/training")
from try_voice import resample, coarse, pitch_yin
(model, encoder, src, words_path, dict_path, rec_path, mlib_path, mkok_path, accent, transpose, prefix) = sys.argv[1:12]
VL = {k[2:]: v for k, v in np.load(mlib_path).items()}           # vowel classes from LibriSpeech
KK = {k[2:]: v for k, v in np.load(mkok_path).items() if k.startswith("p_")}   # every sound from Kokoro
so = ort.SessionOptions(); so.intra_op_num_threads = 4
enc = ort.InferenceSession(encoder, so, providers=["CPUExecutionProvider"])
gen = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
gin = {i.name for i in gen.get_inputs()}
x, sr = sf.read(src, dtype="float32", always_2d=True); x = x.mean(1)
x16 = resample(x, sr, 16000)
frames = int(math.ceil(len(x16) / 160)); frames += frames % 2
x16 = np.pad(x16, (0, frames * 160 + 80 - len(x16)))
f0 = pitch_yin(x16); f0 = np.pad(f0, (0, max(0, frames - len(f0))))[:frames] * (2 ** (float(transpose) / 12))
F = enc.run(None, {enc.get_inputs()[0].name: x16[None, None]})[0][0]          # [T, 768], 50 Hz
T = len(F)

m = torch.load(rec_path, weights_only=False)
classes = [str(c) for c in m["classes"]]; ci = {c: i for i, c in enumerate(classes)}
net = nn.Sequential(nn.Linear(len(m["mu"]), 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(classes)))
net.load_state_dict(m["state"]); net.eval()
W = np.concatenate([F[np.clip(np.arange(T) + s, 0, T - 1)] for s in range(-2, 3)], 1)
with torch.no_grad():
    P = torch.softmax(net(torch.tensor((W - m["mu"]) / m["sd"], dtype=torch.float32)), 1).numpy()
logP = np.log(P + 1e-6)

lines = gzip.open(dict_path, "rt", encoding="latin-1").read().split("\n")
syms = lines[0].split()[2:]
lex = {}
for line in lines[1:]:
    if "\t" in line:
        w_, c_ = line.split("\t"); lex[w_] = [syms[ord(ch) - 0x30] for ch in c_]

def emit_class(ph):
    p = re.sub(r"\d", "", ph)
    if p in ci and p not in ("C", "SIL"): return [ci[p]] + ([ci["AY_g"]] if p == "AY" else []) + ([ci["OY_g"]] if p == "OY" else [])
    return [ci["C"]]

LONG_OK = {"M", "N", "NG", "L", "R", "W", "Y"}     # consonants a singer can hold

def align_all(words_list):
    """One Viterbi over the whole take: the lyric's sounds in order, optional silence between words. A consonant
    other than a nasal or liquid pays for every frame it stays, so sung vowels are not handed to consonants."""
    st = [("SIL", None, -1, [ci["SIL"]], 0.0)]
    for wi, (key, phones) in enumerate(words_list):
        for pi, p in enumerate(phones):
            cons = not p[-1].isdigit()
            pen = -0.6 if cons and re.sub(r"\d", "", p) not in LONG_OK else 0.0
            st.append((p, wi, pi, emit_class(p), pen))
        st.append(("SIL", None, -1, [ci["SIL"]], 0.0))
    S = len(st)
    em = np.stack([logP[:, idx].max(1) for (_, _, _, idx, _) in st], 1)       # [T, S]
    for j, (p, wi, pi, _, _) in enumerate(st):        # each word stays within 0.2 s of where Whisper heard it
        if wi is not None:
            lo, hi = max(0, int((times[wi][0] - 0.2) * 50)), min(T, int((times[wi][1] + 0.2) * 50))
            em[:lo, j] = -1e9; em[hi:, j] = -1e9
    stay = np.array([x[4] for x in st]); is_sil = np.array([x[0] == "SIL" for x in st])
    D = np.full((T, S), -1e9); B = np.zeros((T, S), np.int32)
    D[0, 0], D[0, 1] = em[0, 0], em[0, 1]
    for t in range(1, T):
        cand_stay = D[t - 1] + stay
        cand_next = np.concatenate([[-1e9], D[t - 1, :-1]])
        # a silence between words may be skipped: from the word's last sound straight to the next word's first
        cand_skip = np.full(S, -1e9); cand_skip[2:] = np.where(is_sil[1:-1], D[t - 1, :-2], -1e9)
        best = np.maximum(np.maximum(cand_stay, cand_next), cand_skip)
        B[t] = np.where(best == cand_stay, 0, np.where(best == cand_next, 1, 2))
        D[t] = best + em[t]
    s = S - 1 if D[T - 1, S - 1] > D[T - 1, S - 2] else S - 2
    path = [s]
    for t in range(T - 1, 0, -1):
        s = s - int(B[t, s]); path.append(s)
    path = path[::-1]
    spans = {}
    for t, s in enumerate(path):
        if st[s][0] != "SIL":
            spans.setdefault(s, [t, t + 1]); spans[s][1] = t + 1
    out = []
    for s in sorted(spans):
        p, wi, pi, _, _ = st[s]
        out.append((words_list[wi][0], p, spans[s][0], spans[s][1], words_list[wi][1], pi))
    return out

# Round 18: the user's line "I think that this is the time for me and you, girl, take it slow", every accent
# change at full strength. Spans read off frames.py (voicing, level, hiss and the recognizer) for this take.
LAM = 2.4
SPANS = [("v_the", 4.31, 4.40, "the"), ("th", 1.86, 2.04, "think"), ("dh", 2.96, 3.02, "that"), ("dh", 3.14, 3.22, "this"), ("dh", 4.14, 4.26, "the"),
         ("v_think", 2.14, 2.44, "think"), ("v_that", 3.02, 3.10, "that"), ("v_this", 3.32, 3.74, "this"),
         ("v_is", 3.96, 4.12, "is"), ("r_for", 5.42, 5.52, "for"), ("v_and", 6.24, 6.62, "and"),
         ("v_girl", 7.88, 8.24, "girl"), ("deglide", 8.50, 8.86, "take"), ("v_it", 9.04, 9.16, "it"), ("deglide", 9.42, 10.32, "slow")]
ACC = {"reggae":    {"v_the": "IY", "v_that": "AA", "v_and": "AA", "v_girl": "AA", "r_for": True},
       "afrobeats": {"v_the": "EH", "v_think": "IY", "v_that": "AA", "v_this": "IY", "v_is": "IY", "v_and": "AA", "v_girl": "EH", "v_it": "IY", "r_for": True},
       "latin":     {"v_the": "EH", "v_think": "IY", "v_that": "AA", "v_this": "IY", "v_is": "IY", "v_and": "AA", "v_girl": "EH", "v_it": "IY", "r_for": True}}[accent]
VOWELS = [c for c in classes if c not in ("C", "SIL", "AY_g", "OY_g") and c in VL]

def ramp(i, n):
    return min(1.0, (i + 1) / 2, (n - i) / 2)

def strong(G, log):
    for kind, a_, b_, word in SPANS:
        a, b = int(round(a_ * 50)), int(round(b_ * 50)); n = b - a
        if kind == "th" or kind == "dh":
            continue                      # done by stops() below
        elif kind == "deglide":     # hold the vowel's start through its glide, nudged towards a pure vowel
            h0, h1 = a + max(1, int(0.15 * n)), a + max(2, int(0.35 * n)); hold = F[h0:h1].mean(0)
            src = "EY" if word != "slow" else "OW"; tgt = "EH" if src == "EY" else "AO"
            hold = hold + 0.8 * (VL[tgt] - VL[src])
            for k in range(h0, b):
                w_ = min(1.0, (k - h0 + 1) / 3); G[k] = (1 - w_) * F[k] + w_ * hold
            log.append("%s: %s held pure" % (word, src))
        elif kind == "r_for" and ACC.get("r_for"):
            hold = G[a - 3:a].mean(0)
            for j, k in enumerate(range(a, b)):
                w_ = min(1.0, (j + 1) / 2); G[k] = (1 - w_) * G[k] + w_ * hold
            log.append("%s: r dropped" % word)
        elif kind in ACC:
            tgt = ACC[kind]
            lam = 3.0 if n <= 12 else LAM               # short vowels (that, this, is, it) pushed harder
            for j, k in enumerate(range(a, b)):        # from the vowel actually sung in this frame
                src = max(VOWELS, key=lambda c: P[k, ci[c]])
                if src != tgt: G[k] = G[k] + lam * max(0.6, ramp(j, n)) * (VL[tgt] - VL[src])
            log.append("%s: vowel -> %s" % (word, tgt))
    return G

def render(Fx, out, f0x=None):
    f0x = f0 if f0x is None else f0x
    rng = np.random.default_rng(0); seg, ctx = 1000, 100; pieces = []; out_sr = None
    for start in range(0, frames, seg):
        a, b = max(0, start - ctx), min(frames, start + seg + ctx); a -= a % 2; b -= (b - a) % 2; n = b - a
        idx = np.minimum((a + np.arange(n)) // 2, len(Fx) - 1)
        feeds = {"phone": Fx[idx][None].astype(np.float32), "phone_lengths": np.array([n], np.int64), "ds": np.array([0], np.int64),
                 "rnd": (rng.standard_normal((1, 192, n)) * 0.66666).astype(np.float32)}
        if "pitchf" in gin:
            pf = f0x[a:b].astype(np.float32)[None]; feeds.update(pitch=coarse(pf), pitchf=pf)
        y = gen.run(["audio"], feeds)[0].reshape(-1); hop = len(y) // n; out_sr = hop * 100; fade = 5 * hop
        end = min(frames, start + seg); pieces.append(y[(start - a) * hop:min(len(y), (end - a) * hop + fade)])
    o = pieces[0]
    for y in pieces[1:]:
        r = np.linspace(0, 1, fade, dtype=np.float32); o = np.concatenate([o[:-fade], o[-fade:] * (1 - r) + y[:fade] * r, y[fade:]])
    sf.write(out, o[: int(len(x) / sr * out_sr)], out_sr, subtype="PCM_16")

# Round 20: each "th" rebuilt as a stop. Your "th" frames are unvoiced, and the generator voices only where the pitch
# track is on, so a "d" gets the next vowel's pitch; the closure is silence (the recognizer data's mean silence frame).
STOPS = [("t", 1.84, 2.015, 2.06, "think"), ("d", 2.70, 2.935, 2.99, "that"), ("d", 3.12, 3.27, 3.325, "this"),
         ("d", 4.17, 4.245, 4.30, "the")]
f0_acc = f0.copy()

def stops(G, log):
    """Round 22: the stops are made in the recording (edit_input.py); here only their pitch: none in the closure,
    the next vowel's pitch under a "d"'s voice bar and burst, none under a "t"."""
    for c, s0, r0, v0, word in STOPS:
        a, r, v = int(round(s0 * 50)), int(round(r0 * 50)), int(round(v0 * 50))
        nxt = next((f0[i] for i in range(2 * v, min(len(f0), 2 * v + 20)) if f0[i] > 0), 0.0)
        f0_acc[2 * a:2 * r] = 0.0
        f0_acc[2 * r:2 * v] = nxt if c == "d" else 0.0
        log.append("%s: %s pitch set" % (word, c))
    return G

log = []
G = stops(strong(F.copy(), log), log)
render(G, prefix + "-r22.wav", f0_acc)
print("\n".join(log))
