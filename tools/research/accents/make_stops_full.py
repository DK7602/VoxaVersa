"""Kokoro-82M (Apache 2.0) saying "a dah", "a dih", "a deh", "a tih": the whole clip and its burst time, so the
"d" can be re-pitched (WORLD) to the singer's note before it is spliced in. usage: make_stops_full.py out_dir"""
import os, sys, json, numpy as np, soundfile as sf
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from kokoro import KPipeline
out = sys.argv[1]; pipe = KPipeline(lang_code="a"); info = {}
for text, ph, name in (("a dah", "d", "d_a"), ("a deh", "d", "d_e"), ("a dih", "d", "d_i"), ("a tih", "t", "t_i")):
    for r in pipe(text, voice="am_michael"):
        audio = r.audio.numpy().astype(np.float32); d = r.pred_dur.tolist(); unit = len(audio) / sum(d)
        t = d[0] * unit; spans = []
        for i, c in enumerate(r.phonemes):
            dur = d[i + 1] * unit; spans.append((c, int(t), int(t + dur))); t += dur
        k = next(i for i, s in enumerate(spans) if s[0] == ph)
        lo, hi = spans[k - 1][1], spans[k][2] + int(0.2 * 24000); hop = 120
        e = np.array([10 * np.log10(np.mean(audio[i:i + hop] ** 2) + 1e-10) for i in range(lo, hi, hop)])
        j = next(i for i in range(len(e)) if e[i] > np.min(e[: max(1, len(e) // 3)]) + 15)
        info[name] = (lo + j * hop) / 24000
        sf.write(os.path.join(out, name + "_full.wav"), audio, 24000)
json.dump(info, open(os.path.join(out, "bursts.json"), "w")); print(info)
