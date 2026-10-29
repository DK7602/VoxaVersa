"""Real vocals for testing the rule: ContentVec features (50 Hz), voicing (Praat pitch, per 20 ms frame) and word
timings (faster-whisper small.en) of each wav -> out_dir/<name>.npz. Needs onnxruntime, praat-parselmouth,
faster-whisper, scipy, soundfile.  usage: real_prep.py contentvec.onnx out_dir wav..."""
import os, sys, math
import numpy as np, soundfile as sf, onnxruntime as ort, parselmouth
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
so = ort.SessionOptions(); so.intra_op_num_threads = 2
enc = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
asr = WhisperModel("small.en", device="cpu", compute_type="int8")
for wav in sys.argv[3:]:
    name = os.path.basename(wav)[:-4]; out = os.path.join(sys.argv[2], name + ".npz")
    if os.path.exists(out): continue
    x, sr = sf.read(wav, dtype="float32", always_2d=True); x = x.mean(1)
    g = math.gcd(int(sr), 16000); x16 = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    n = int(math.ceil(len(x16) / 320)); xp = np.pad(x16, (0, n * 320 + 80 - len(x16)))
    F = enc.run(None, {enc.get_inputs()[0].name: xp[None, None]})[0][0]
    pitch = parselmouth.Sound(x16.astype(np.float64), 16000).to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=1100)
    f0 = np.array([pitch.get_value_at_time((k + 0.5) / 100) or 0.0 for k in range(2 * len(F))])
    f0 = np.nan_to_num(f0); voiced = np.array([f0[2 * k] > 0 or f0[2 * k + 1] > 0 for k in range(len(F))])
    segs, _ = asr.transcribe(x16, beam_size=5, language="en", word_timestamps=True, condition_on_previous_text=False)
    words = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
    np.savez(out, F=F.astype(np.float16), voiced=voiced, words=np.array(words, dtype=object))
    print(name, len(F), " ".join(w for w, _, _ in words), flush=True)
