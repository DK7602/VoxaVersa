"""Mean ContentVec vector per phoneme symbol (vowels and consonants), from Kokoro's American voices.
usage: phone_means2.py contentvec.onnx out.npz"""
import os, sys, math
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import numpy as np, onnxruntime as ort
from scipy.signal import resample_poly
from kokoro import KPipeline
SENT = ["Think of the three things that they thought", "This is the thing with the mother and the father",
        "Both of them breathe through their teeth", "The thin path through the north", "Thank them for the other theme",
        "Take ten tight steps to the top", "Dad did a good deed today", "Put the tea down at the door", "Time to tell the truth",
        "Never ever water the runner", "Your four doors are over there", "Her bird heard the word", "Better weather for the summer",
        "They came to play in the rain and the lane", "Take the same train, the pain will fade away", "Wait for the late great eight",
        "Go slow, so the snow won't blow", "Know the road home alone", "Sit with Jim in the big inn", "His kid is quick with it",
        "Run up the hunter's sunny hut", "The cat sat back and laughed at the black hat", "Hot top father calm pot", "Saw the dog on the long walk",
        "Faithful friend and life is a marathon", "Like a runner who never quits"]
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_michael", "am_adam", "am_eric", "am_liam"]
so = ort.SessionOptions(); so.intra_op_num_threads = 4
enc = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
sums, counts = {}, {}
pipe = KPipeline(lang_code="a")
for voice in VOICES:
    for text in SENT:
        for r in pipe(text, voice=voice):
            audio = r.audio.numpy().astype(np.float32); d = r.pred_dur.tolist(); unit = len(audio) / sum(d) / 24000
            x16 = resample_poly(audio, 2, 3).astype(np.float32)
            n = int(math.ceil(len(x16) / 320)); x16 = np.pad(x16, (0, n * 320 + 80 - len(x16)))
            feats = enc.run(None, {enc.get_inputs()[0].name: x16[None, None]})[0][0]
            t = d[0] * unit
            for i, c in enumerate(r.phonemes):
                dur = d[i + 1] * unit; a, b = t, t + dur; t += dur
                if c in " ˈˌː,.": continue
                lo, hi = a + 0.15 * (b - a), b - 0.15 * (b - a)
                ks = [k for k in range(int(lo * 50), int(hi * 50) + 1) if 0 <= k < len(feats) and lo <= (k + 0.5) / 50 <= hi]
                if not ks and 0 <= int((a + b) / 2 * 50) < len(feats): ks = [int((a + b) / 2 * 50)]
                for k in ks:
                    sums[c] = sums.get(c, 0) + feats[k]; counts[c] = counts.get(c, 0) + 1
    print(voice, len(counts), flush=True)
labels = sorted(counts)
print({c: counts[c] for c in labels})
np.savez(sys.argv[2], **{"p_" + c: sums[c] / counts[c] for c in labels}, labels=np.array(labels), counts=np.array([counts[c] for c in labels]))
