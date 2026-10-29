"""Mean ContentVec vector per vowel (50 Hz frames, the middle 60 % of each phone), from Kokoro's American voices
saying sentences rich in the vowels the Southern accent changes. usage: accent_phone_means.py contentvec.onnx out.npz"""
import os, sys, math
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import numpy as np, onnxruntime as ort
from scipy.signal import resample_poly
from kokoro import KPipeline
SENT = ["I like my time on the high side of the night", "My wife tried to drive five miles", "Why fight it, I might fly tonight",
        "Ten men sent friends to the end of the bend", "Then again, when the pen went in, he was gentle", "Remember the tender center of the dent",
        "The boy enjoyed the noise of the toy", "Joy and soy and the voice of the oyster",
        "They came to play in the rain and the lane", "Take the same train, the pain will fade away", "Wait for the late great eight",
        "The cat sat back and laughed at the black hat", "Happy Jack had a bad map", "Ask the man in the van and the band",
        "Sit with Jim in the big inn", "His kid is quick with it", "This little bit is in it",
        "Red bed said head met", "Get the best fresh bread", "Hot top father calm pot", "Saw the dog on the long walk"]
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_michael", "am_adam", "am_eric", "am_liam"]
VOW = {"æ": "AE", "ɛ": "EH", "ɪ": "IH", "i": "IY", "ɑ": "AA", "ʌ": "AH", "ɔ": "AO", "A": "EY", "I": "AY", "Y": "OY", "O": "OW", "W": "AW", "u": "UW", "ʊ": "UH"}
NAS = set("nmŋ")
so = ort.SessionOptions(); so.intra_op_num_threads = 3
enc = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
sums, counts = {}, {}
for voice in VOICES:
    pipe = KPipeline(lang_code="a")
    for text in SENT:
        for r in pipe(text, voice=voice):
            audio = r.audio.numpy().astype(np.float32); d = r.pred_dur.tolist(); unit = len(audio) / sum(d) / 24000
            x16 = resample_poly(audio, 2, 3).astype(np.float32)
            n = int(math.ceil(len(x16) / 320)); x16 = np.pad(x16, (0, n * 320 + 80 - len(x16)))
            feats = enc.run(None, {enc.get_inputs()[0].name: x16[None, None]})[0][0]   # [T, 768] at 50 Hz
            t = d[0] * unit
            ph = [c for c in r.phonemes]
            for i, c in enumerate(ph):
                dur = d[i + 1] * unit; a, b = t, t + dur; t += dur
                if c not in VOW: continue
                label = VOW[c]
                nxt = next((p for p in ph[i + 1:] if p not in "ˈˌː"), "")
                if label == "EH" and nxt in NAS: label = "EH_N"
                lo, hi = a + 0.2 * (b - a), b - 0.2 * (b - a)
                for k in range(int(lo * 50), int(hi * 50) + 1):
                    if 0 <= k < len(feats) and lo <= (k + 0.5) / 50 <= hi:
                        sums[label] = sums.get(label, 0) + feats[k]; counts[label] = counts.get(label, 0) + 1
    print(voice, {k: counts[k] for k in sorted(counts)}, flush=True)
np.savez(sys.argv[2], **{k: sums[k] / counts[k] for k in sums}, counts=np.array([counts[k] for k in sorted(counts)]), labels=np.array(sorted(counts)))
