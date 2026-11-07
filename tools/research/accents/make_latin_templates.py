"""The Latin accent's new sounds, spoken by Kokoro-82M (Apache 2.0): the Spanish tapped and trilled r (its Spanish voice saying
"para", "pero", "perro") and the "j" of a Spanish speaker's "you" (its American voice saying "a June"). Writes each clip and,
in sounds.json, where the sound is (seconds; from Kokoro's own phoneme durations).
usage: make_latin_templates.py out_dir"""
import os, sys, json, numpy as np, soundfile as sf
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from kokoro import KPipeline
out = sys.argv[1]; os.makedirs(out, exist_ok=True); info = {}
CLIPS = [("e", "em_alex", "para", "ɾ", "tap_para"), ("e", "em_alex", "pero", "ɾ", "tap_pero"), ("e", "em_alex", "perro", "r", "trill_perro"),
         ("e", "ef_dora", "para", "ɾ", "tap_para_f"), ("a", "am_michael", "a June", "ʤ", "jh_june")]
pipes = {}
for lang, voice, text, ph, name in CLIPS:
    pipes.setdefault(lang, KPipeline(lang_code=lang))
    for r in pipes[lang](text, voice=voice):
        audio = r.audio.numpy().astype(np.float32); d = r.pred_dur.tolist(); unit = len(audio) / sum(d)
        t = d[0] * unit; spans = []
        for i, c in enumerate(r.phonemes):
            dur = d[i + 1] * unit; spans.append((c, t / 24000, (t + dur) / 24000)); t += dur
        hits = [s for s in spans if s[0] == ph or (ph == "ʤ" and s[0] in ("ʤ", "j"))]
        if not hits:
            print(name, "no", ph, "in", r.phonemes); continue
        info[name] = {"phonemes": r.phonemes, "start": hits[0][1], "end": hits[0][2]}
        sf.write(os.path.join(out, name + ".wav"), audio, 24000)
        print(name, r.phonemes, "%.3f-%.3f s" % (hits[0][1], hits[0][2]))
json.dump(info, open(os.path.join(out, "sounds.json"), "w"), indent=1)
