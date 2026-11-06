"""Clean "t" and "d" onsets from Kokoro-82M (Apache 2.0): the burst plus the first 40 ms of the vowel after it.
usage: make_stops.py out_dir"""
import os, sys, numpy as np, soundfile as sf
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from kokoro import KPipeline
out = sys.argv[1]; pipe = KPipeline(lang_code="a")
for voice in ("am_michael", "af_heart"):
    for text, ph_want, name in (("a dah", "d", "d_a"), ("a deh", "d", "d_e"), ("a dih", "d", "d_i"), ("a tih", "t", "t_i")):
        for r in pipe(text, voice=voice):
            audio = r.audio.numpy().astype(np.float32); d = r.pred_dur.tolist(); unit = len(audio) / sum(d)
            t = d[0] * unit; spans = []
            for i, c in enumerate(r.phonemes):
                dur = d[i + 1] * unit; spans.append((c, int(t), int(t + dur))); t += dur
            print(voice, text, [(c, round(a / 24000, 3), round(b / 24000, 3)) for c, a, b in spans])
            k = next(i for i, s in enumerate(spans) if s[0] == ph_want)
            # the burst: the first 5 ms frame after the closure whose level jumps 15 dB above the closure's
            lo, hi = spans[k - 1][1], spans[k][2] + int(0.2 * 24000)
            hop = 120; e = np.array([10 * np.log10(np.mean(audio[i:i + hop] ** 2) + 1e-10) for i in range(lo, hi, hop)])
            base = np.min(e[: max(1, len(e) // 3)])
            j = next(i for i in range(len(e)) if e[i] > base + 15)
            a = lo + j * hop - int(0.004 * 24000); b = a + int(0.075 * 24000)
            seg = audio[a:b].copy(); seg[: 48] *= np.linspace(0, 1, 48)
            print("   burst at %.3f s" % (a / 24000))
            sf.write(os.path.join(out, "%s_%s.wav" % (name, voice[:2])), seg, 24000)
