"""Round 31: a new Latin accent. Round 29's Latin (round 25 A) was heard as "more Asian than Latin": it dropped the
"r" after a vowel as the Jamaican and Nigerian accents do, which with pure vowels reads as East Asian English. A
Spanish speaker keeps the "r" and taps (or rolls) it, and says "you" as "jou". On top of accent_splice.py's Latin
(its vowels, tink / dat / dees / de, an' you) without the dropped "r":
  A (tap):  for - the vowel held, then Kokoro's Spanish tap (its "para") before "me"
  B (roll): for and girl - its rolled "rr" ("perro"; a tap before the "l" of girl was heard as "d" by the read-back)
  you:  Kokoro's "j" ("a June") on the first frames of the "y" (jou)
Times are this take's (the consonant recognizer's read-out of it).
usage: accent_latin.py model.onnx contentvec.onnx take.wav words.json cmudict.gz recognizer.pt means_libri.npz
                       means2.npz latin transpose out_prefix feats.npz bursts.json latin_feats.npz
       (latin_feats.npz: k_tap_para, k_trill_perro, k_jh_june, k_l_ala - make_latin_templates.py's clips' features)
       [--round32]: the "girl" variants of round 32 instead (G1 e + rolled r + light l, G2 the singer's "er" + rolled r +
       light l, G3 e + light l); [--round33]: "and" on G2 (N1 a pure "an", N2 "en", N3 only its "eh" before the n -> "ah")"""
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
       "latin":     {"v_the": "EH", "v_think": "IY", "v_that": "AA", "v_this": "IY", "v_is": "IY", "v_and": "AA", "v_girl": "EH", "v_it": "IY"}}[accent]   # the "r" is kept (tapped below)
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
            lam = 1.8 if n <= 12 else LAM               # short vowels (that, this, is, it) pushed harder
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

# Round 28: the consonants made inside the AI singer, as the plug-in can (no lyric, no edit of the recording): the
# frames of each "th" replaced by Kokoro's own "t" / "d" frames (its "a tih" / "a dah" / "a deh" / "a dih", encoded
# alone; the template for the vowel that follows), placed so the burst falls just before your vowel; before them
# silence (a pause stays a pause) or, for a legato "d", Kokoro's voice bar. Pitch: none in silence and under a "t",
# the next vowel's under a "d". "and you": the "d" frames get the "n" before them (an' you), pitch filled.
KF = np.load(sys.argv[12])
BURST = json.load(open(sys.argv[13]))
SIL = VL["SIL"] if "SIL" in VL else KF["k_t_i"][:5].mean(0)
#        (kind, template, th start, vowel onset, pause before)
CONS = [("t", "t_i", 1.84, 2.06, True), ("d", "d_a", 2.70, 3.035, True), ("d", "d_i", 3.20, 3.325, False),
        ("d", "d_a", 4.15, 4.285, False)]
f0_acc = f0.copy()

def cons(G, log):
    for kind, tpl, s0, v0, pause in CONS:
        a, v = int(round(s0 * 50)), int(round(v0 * 50)); K = KF["k_" + tpl]; b = int(round(BURST[tpl] * 50))
        if kind == "t":
            frames = K[b - 2:b + 2]          # closure, closure, burst, aspiration
        else:
            frames = K[b - 3:b + 1]          # voice bar x3, burst
        start = max(a, v - len(frames))
        for k in range(a, start):
            G[k] = SIL if (pause or kind == "t") else K[b - 3]
        G[start:v] = frames[len(frames) - (v - start):]
        nxt = next((f0[i] for i in range(2 * v, min(len(f0), 2 * v + 20)) if f0[i] > 0), 0.0)
        if kind == "t":
            f0_acc[2 * a:2 * v] = 0.0
        else:
            f0_acc[2 * a:2 * start] = 0.0 if pause else nxt
            f0_acc[2 * start:2 * v] = nxt
        log.append("%s: %s spliced (%s), %d frames" % ({1.84: "think", 2.70: "that", 3.20: "this", 4.15: "the"}[s0], kind, tpl, v - a))
    # an' you: the "d" between "n" and "you" sung as the "n"
    n0, n1, d0, d1 = int(6.56 * 50), int(6.63 * 50), int(6.64 * 50), int(6.80 * 50)
    hold = F[n0:n1].mean(0)
    for j, k in enumerate(range(d0, d1)):
        w_ = min(1.0, (j + 1) / 2, (d1 - k) / 2); G[k] = (1 - w_) * G[k] + w_ * hold
    seg = f0_acc[2 * n0:2 * d1 + 20]; vz = np.where(seg > 0)[0]
    if len(vz): f0_acc[2 * n0:2 * d1 + 20] = np.interp(np.arange(len(seg)), vz, seg[vz])
    log.append("and you: d held as n")
    return G

LF = np.load(sys.argv[14])
TAP = LF["k_tap_para"][17:19]          # the tap: 2 frames (read as a "d"-like flick)
ROLL = LF["k_trill_perro"][18:21]      # the rolled "rr": 3 frames
JH = LF["k_jh_june"][28:33]            # the "j": 4 frames and its move into the vowel

def latin(G, log, r, girl):
    fr = lambda t: int(round(t * 50))
    # for: R at 5.30 - 5.42, "me" from 5.44; the vowel is already r-coloured from ~5.20: the clean "aw" of
    # 5.10 - 5.18 held to the r, which comes at the end
    hold = F[fr(5.10):fr(5.18)].mean(0)
    for k in range(fr(5.20), fr(5.44) - len(r)):
        w_ = min(1.0, (k - fr(5.20) + 1) / 2); G[k] = (1 - w_) * G[k] + w_ * hold
    G[fr(5.44) - len(r):fr(5.44)] = r
    # girl: the vowel to 8.20, "l" from 8.22: the r just before the "l" (a tap there reads as "d": roll only)
    if girl:
        G[fr(8.22) - len(r):fr(8.22)] = r
    # you: "y" from 6.74: the "j" and its move into the vowel
    G[fr(6.74):fr(6.74) + len(JH)] = JH
    log.append("for%s: %d-frame r; you: jou" % (" / girl" if girl else "", len(r)))
    return G

LIGHT_L = LF["k_l_ala"][15:19]         # Spanish light "l": 4 frames (English "girl" ends in a dark one)

def girl(G, log, vowel_kept, roll, light):
    """Round 32 (Male Latin's "girl" did not sound Latin): the r before the "l" (8.22), the "l" 8.22 - 8.30."""
    fr = lambda t: int(round(t * 50))
    if vowel_kept:      # the singer's own "er" vowel, not "e"
        G[fr(7.88):fr(8.22)] = G_plain[fr(7.88):fr(8.22)]
    if roll:
        G[fr(8.22) - len(ROLL):fr(8.22)] = ROLL
    if light:
        for j, k in enumerate(range(fr(8.22), fr(8.32))):
            G[k] = LIGHT_L[min(j, len(LIGHT_L) - 1)]
    log.append("girl: %s, %s, %s" % ("er kept" if vowel_kept else "e", "rolled r" if roll else "no r", "light l" if light else "dark l"))
    return G

log = []
G0 = cons(strong(F.copy(), log), log)
if "--round32" not in sys.argv and "--round33" not in sys.argv:
    render(latin(G0.copy(), log, TAP, False), prefix + "-r31tap.wav", f0_acc)
    render(latin(G0.copy(), log, ROLL, True), prefix + "-r31roll.wav", f0_acc)
elif "--round33" in sys.argv:
    # Round 33: "and" (sung "in-en": IH 6.10 - 6.40, then the "eh" before n 6.42 - 6.58, n 6.60). Latin "and" is a pure "an".
    # On round 32's pick G2 (gurrl: the singer's own "er", rolled r, light l). The accent's "a" -> "ah" swap of "and" is
    # left out here: these replace its vowel.
    ACC.pop("v_girl"); ACC.pop("v_and")
    G_plain = cons(strong(F.copy(), []), [])
    G2 = girl(latin(G_plain.copy(), log, ROLL, False), log, True, True, True)
    fr = lambda t: int(round(t * 50))
    def pure(G, a_, b_, target, lam=1.0):
        """frames a_..b_ sung as one steady vowel: their mean, moved from the classes they sound like to `target`"""
        a, b = fr(a_), fr(b_)
        src = np.mean([VL[max(VOWELS, key=lambda c: P[k, ci[c]])] for k in range(a, b)], 0)
        hold = F[a:b].mean(0) + lam * (VL[target] - src)
        for j, k in enumerate(range(a, b)):
            w_ = min(1.0, (j + 1) / 2); G[k] = (1 - w_) * G[k] + w_ * hold
        return G
    render(pure(G2.copy(), 6.10, 6.60, "AA", 1.4), prefix + "-r33n1.wav", f0_acc)
    render(pure(G2.copy(), 6.10, 6.60, "EH", 1.4), prefix + "-r33n2.wav", f0_acc)
    render(pure(G2.copy(), 6.42, 6.60, "AA", 1.4), prefix + "-r33n3.wav", f0_acc)
    log.append("and: n1 an (6.10-6.60 -> AA), n2 en (-> EH), n3 i-an (6.42-6.60 -> AA)")
else:
    ACC.pop("v_girl"); G_plain = cons(strong(F.copy(), []), [])      # the same accent with the singer's own "girl" vowel
    B = latin(G0.copy(), log, ROLL, False)                           # round 31 B without its "girl" r
    render(girl(B.copy(), log, False, True, True), prefix + "-r32g1.wav", f0_acc)
    render(girl(B.copy(), log, True, True, True), prefix + "-r32g2.wav", f0_acc)
    render(girl(B.copy(), log, False, False, True), prefix + "-r32g3.wav", f0_acc)
print("\n".join(log))
