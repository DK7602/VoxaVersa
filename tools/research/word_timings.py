"""Word timings of a vocal: faster-whisper small.en with word timestamps -> JSON [{word, start, end}].
usage: words.py in.wav out.json"""
import os, sys, json
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
x, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True); x = x.mean(1)
g = np.gcd(int(sr), 16000); x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
m = WhisperModel("small.en", device="cpu", compute_type="int8")
segs, _ = m.transcribe(x, beam_size=5, language="en", word_timestamps=True, condition_on_previous_text=False)
words = [dict(word=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3)) for s in segs for w in s.words]
json.dump(words, open(sys.argv[2], "w"), indent=1)
for w in words: print("%6.2f %6.2f %s" % (w["start"], w["end"], w["word"]))
