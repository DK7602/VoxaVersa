#!/usr/bin/env python3
"""
Decides which trained singer powers which built-in VoxaVersa voice ("factoryVoices" in voice.json,
docs/NEURAL_ENGINE.md section 5), from measurements of the singers' own recordings.

    python3 tools/training/factory_voices.py measure --dataset datasets/my_singer --gender female --out my_singer.json
    python3 tools/training/factory_voices.py assign --singers a.json b.json ... --out factory_voice_assignment

`measure` describes a singer's natural colour from the cleaned dataset (all clips):
  register     median sung pitch (MIDI) and the 3rd - 97th percentile range
  brightness   spectral tilt of the long-term spectrum of voiced frames (dB/octave, 0.5 - 8 kHz)
               plus half the "air" (8 - 14 kHz energy relative to all, dB)
  breathiness  median aperiodicity of voiced frames (YIN; aspiration noise raises it)
  power        "ring": 2.5 - 4 kHz energy relative to 0.3 - 1 kHz (the singer's formant that makes a voice
               carry), plus a little for a high top note

`assign` reads the built-in voices from Source/voices/FactoryVoices.cpp (genre voices: register from their
range, brightness from tilt / air / presence / ring frequency, breathiness, power from presence, ring, rasp
and breath) and gives every voice of a gender the singer of that gender whose standardised colour is
closest (squared distance of z-scores; register, brightness, breathiness, power), greedily, best matches
first, with two rules:
  * similar genres get different singers while there are enough singers (groups: bright pop, soulful,
    powerful, rhythmic, acoustic - see GROUPS); with fewer singers than a group has voices, reuse is
    allowed but spread out;
  * the load is balanced (each further voice on the same singer costs a little more);
  * brightness and power are compared within a recording corpus (a singer's "corpus" label: microphones and
    processing differ more between vendors than voices do), and a singer's "style" ("classical" for trained
    classical singers, else contemporary) costs STYLE_PENALTY when it differs from the genre's (opera,
    theatre and angelic are classical); all other voices but deep bass get contemporary singers only
    once a gender has at least MIN_CONTEMPORARY of them; a singer's "genres" (optional) favour those genres' voices.
Specialty voices: male-deep-bass, female-angelic and female-smoky-alto are matched the same way
(neutral-whisper gets none: an AI singer cannot whisper). Writes <out>.json (for the training queue) and <out>.md (the table).
Requirements: numpy, scipy, soundfile.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FACTORY_CPP = os.path.join(REPO, "Source", "voices", "FactoryVoices.cpp")

GROUPS = {
    "bright pop": ["pop", "kpop", "latin", "edm", "afrobeats"],
    "soulful": ["rnb", "soul", "jazz", "blues", "lofi"],
    "powerful": ["gospel", "rock", "metal", "opera", "theatre"],
    "rhythmic": ["rap", "trap", "reggae"],
    "acoustic": ["country", "indie"],
    "specialty": ["deep-bass", "angelic", "smoky-alto", "whisper"],
}
WEIGHTS = dict(register=1.0, bright=1.0, breath=0.8, power=0.8)
FEATURES = list(WEIGHTS)
# Genres whose built-in voice is a trained, "legit" classical/theatre sound; all others are contemporary.
CLASSICAL = {"opera", "theatre", "angelic"}
STYLE_PENALTY = 1.5
# Contemporary voices sung by a classically trained singer sound young and "legit" (a user heard Female Pop as a
# child, then Male Rock as country): every voice that is not classical (opera, theatre, angelic) or a specialty
# that suits a trained voice (deep bass) gets contemporary singers only, once a gender has enough of them
# to share these voices (fewer would put one voice on most of the gender's avatars; the style penalty decides
# until then). The classical singers also train less true to themselves (similarity 0.16 - 0.49 against 0.60 -
# 0.77 for the studio and band singers).
CLASSICAL_FRIENDLY = CLASSICAL | {"deep-bass"}
MIN_CONTEMPORARY = 3
# A singer's "genres" (heard by the owner, e.g. a military-band singer who sings country) make the built-in
# voices of those genres prefer them.
GENRE_BONUS = 1.0


# -------------------------------------------------------------------------------------------- measure

def measure(dataset, gender, max_seconds=1200):
    import soundfile as sf
    from scipy import signal
    sys.path.insert(0, HERE)
    from prepare_dataset import yin_frames, frame_db, resample

    clips = sorted(glob.glob(os.path.join(dataset, "*.wav")))
    if not clips:
        raise SystemExit("no clips in " + dataset)
    f0_all, ap_all, spec_sum, n_frames, seconds = [], [], None, 0, 0.0
    for c in clips:
        x, sr = sf.read(c, dtype="float32", always_2d=True)
        x = x.mean(1)
        seconds += len(x) / sr
        x16 = resample(x, sr, 16000)
        f0, ap = yin_frames(x16)
        db = frame_db(x16, 16000)
        m = min(len(f0), len(db))
        f0, ap, db = f0[:m], ap[:m], db[:m]
        voiced = (f0 > 0) & (db > db.max() - 25)
        f0_all.append(f0[voiced])
        ap_all.append(ap[voiced])
        f, t, Z = signal.stft(x, sr, nperseg=2048, noverlap=2048 - int(sr * 0.01))
        P = np.abs(Z) ** 2
        k = min(P.shape[1], m)
        sel = voiced[:k]
        if sel.any():
            s = P[:, :k][:, sel].sum(1)
            spec_sum = s if spec_sum is None else spec_sum + s
            n_frames += int(sel.sum())
        if seconds > max_seconds:
            break
    f0 = np.concatenate(f0_all)
    ap = np.concatenate(ap_all)
    midi = 69 + 12 * np.log2(f0 / 440.0)
    ltas = 10 * np.log10(spec_sum / max(1, n_frames) + 1e-20)
    band = (f >= 500) & (f <= 8000)
    tilt = float(np.polyfit(np.log2(f[band]), ltas[band], 1)[0])           # dB per octave
    P = spec_sum
    def energy(lo, hi):
        return float(P[(f >= lo) & (f < hi)].sum()) + 1e-20
    air = 10 * math.log10(energy(8000, 14000) / energy(0, 14000))
    ring = 10 * math.log10(energy(2500, 4000) / energy(300, 1000))
    lo, med, hi = np.percentile(midi, [3, 50, 97])
    return dict(dataset=os.path.abspath(dataset), gender=gender, seconds=round(seconds, 1),
                median_midi=round(float(med), 2), range_midi=[round(float(lo), 1), round(float(hi), 1)],
                tilt_db_oct=round(tilt, 2), air_db=round(air, 2), ring_db=round(ring, 2),
                aperiodicity=round(float(np.median(ap)), 4),
                register=round(float(med), 2), bright=round(tilt + 0.5 * air, 3),
                breath=round(float(np.median(ap)), 4), power=round(ring + 0.15 * (float(hi) - float(med)), 3))


# ---------------------------------------------------------------------------------- factory voices

def _numbers(text):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?=f?\b)", text)]


def factory_voices(cpp=FACTORY_CPP):
    """Built-in voices with comparable descriptors, parsed from FactoryVoices.cpp."""
    src = open(cpp, encoding="utf-8").read()
    styles = src[src.index("genreStyles[] = {"):src.index("};", src.index("genreStyles[] = {"))]
    colours_src = src[src.index("voiceColours[] = {"):src.index("};", src.index("voiceColours[] = {"))]
    colours = {}
    for m in re.finditer(r'\{\s*"([a-z]+)",([^}]*)\}', colours_src):
        n = _numbers(m.group(2))
        colours[m.group(1)] = dict(ringHz=n[2], ringDb=n[3], breathScale=n[6], raspAdd=n[7])
    voices = []
    for m in re.finditer(r'\{\s*"([^"]+)",\s*"([a-z]+)",\s*0x[0-9a-f]+,\s*0x[0-9a-f]+,\s*"[^"]*",\s*"[^"]*",\s*\{[^}]*\},([^}]*)\}', styles):
        genre, idp, rest = m.group(1), m.group(2), m.group(3)
        n = _numbers(rest)
        tilt, warmth, presence, air, nasal, breath, rasp = n[0:7]
        male_lo, male_hi, female_lo, female_hi = n[23], n[24], n[26], n[27]
        c = colours.get(idp, dict(ringHz=3000.0, ringDb=0.0, breathScale=1.0, raspAdd=0.0))
        for gender, lo, hi in (("male", male_lo, male_hi), ("female", female_lo, female_hi)):
            b = min(1.0, (breath + (0.05 if gender == "female" else 0.0)) * c["breathScale"])
            r = min(1.0, rasp * (1.0 if gender == "male" else 0.8) + c["raspAdd"])
            voices.append(dict(id="%s-%s" % (gender, idp), genre=genre, group=group_of(idp), gender=gender,
                               register=(lo + hi) / 2, bright=tilt + 0.5 * air + 0.5 * presence + (c["ringHz"] - 2500) / 1000,
                               breath=b, power=0.6 * presence + 0.5 * c["ringDb"] + 3 * r - 3 * b))
    # specialty voices: range and colour from their set-up lambdas
    for m in re.finditer(r'makeSpecialty \("([a-z-]+)",\s*"([^"]+)",\s*"[^"]*",\s*VoiceGender::(\w+)(.*?)\}\)\);', src, re.S):
        vid, name, gender, body = m.group(1), m.group(2), m.group(3).lower(), m.group(4)
        if vid in ("neutral-robot", "neutral-choir", "neutral-whisper"):   # singers of their own, or none
            continue
        def val(key, default=0.0):
            mm = re.search(r"t\.%s\s*=\s*(-?\d+(?:\.\d+)?)f?" % key, body)
            return float(mm.group(1)) if mm else default
        lo, hi = val("rangeLowMidi", 48), val("rangeHighMidi", 72)
        b = val("breathiness")
        voices.append(dict(id=vid, genre=name, group="specialty", gender=gender, register=(lo + hi) / 2,
                           bright=val("tiltDb") + 0.5 * val("airDb") + 0.5 * val("presenceDb") + (val("ringHz", 3000) - 2500) / 1000,
                           breath=b, power=0.6 * val("presenceDb") + 0.5 * val("ringDb") + 3 * val("rasp") - 3 * b))
    if len(voices) < 40:
        raise SystemExit("could not parse the built-in voices from %s (found %d)" % (cpp, len(voices)))
    return voices


def group_of(idp):
    for g, members in GROUPS.items():
        if idp in members:
            return g
    return idp


# ----------------------------------------------------------------------------------------- assign

def zscores(items, keys):
    out = []
    stats = {}
    for k in keys:
        vals = np.array([it[k] for it in items], float)
        mu, sd = float(vals.mean()), float(vals.std())
        stats[k] = (mu, sd if sd > 1e-9 else 1.0)
    for it in items:
        out.append({k: (it[k] - stats[k][0]) / stats[k][1] if len(items) > 1 else 0.0 for k in keys})
    return out


def corpus_normalise(singers):
    """Brightness, air and ring depend on the microphone and processing as much as on the voice: remove each
    corpus' mean (singers with the same "corpus" label, e.g. one sample-pack vendor) so only the differences
    between singers count, then add back the overall mean."""
    out = [dict(s) for s in singers]
    for k in ("bright", "power"):
        overall = float(np.mean([s[k] for s in singers]))
        for c in {s.get("corpus") for s in singers}:
            members = [s for s in out if s.get("corpus") == c]
            mu = float(np.mean([s[k] for s in members]))
            for s in members:
                s[k] = s[k] - mu + overall
    return out


def assign(singers, voices, fixed=None):
    """{voice id: singer name} plus the cost table. `fixed` ({voice id: singer name}) keeps those voices with
    their singers (e.g. ones the owner already likes); they count toward each singer's load."""
    singers = corpus_normalise(singers)
    fixed = {v: n for v, n in (fixed or {}).items() if any(s["name"] == n for s in singers)}
    result, costs = dict(fixed), {v: 0.0 for v in fixed}
    for gender in ("male", "female"):
        sg = [s for s in singers if s["gender"] == gender]
        vs = [v for v in voices if v["gender"] == gender]
        if not sg:
            continue
        zs, zv = zscores(sg, FEATURES), zscores(vs, FEATURES)
        load = {s["name"]: 0 for s in sg}
        groups_used = {}
        for v in vs:
            if v["id"] in fixed:
                load[fixed[v["id"]]] += 1
                groups_used[(v["group"], fixed[v["id"]])] = groups_used.get((v["group"], fixed[v["id"]]), 0) + 1
        contemporary_first = sum(1 for s in sg if s.get("style", "contemporary") != "classical") >= MIN_CONTEMPORARY
        todo = [i for i in range(len(vs)) if vs[i]["id"] not in fixed]
        while todo:
            best = None
            for i in todo:
                for j, s in enumerate(sg):
                    d = sum(WEIGHTS[k] * (zv[i][k] - zs[j][k]) ** 2 for k in FEATURES)
                    group = vs[i]["group"]
                    used = groups_used.get((group, s["name"]), 0)
                    group_size = sum(1 for v in vs if v["group"] == group)
                    if group != "specialty" and used:
                        d += 4.0 * used if len(sg) >= group_size else 1.5 * used
                    d += 0.8 * load[s["name"]]
                    idp = vs[i]["id"].split("-", 1)[1]
                    style = s.get("style")
                    if style and (style == "classical") != (idp in CLASSICAL):
                        d += STYLE_PENALTY
                    if contemporary_first and style == "classical" and idp not in CLASSICAL_FRIENDLY:
                        d += 100.0
                    if idp in (s.get("genres") or ()):
                        d -= GENRE_BONUS
                    if best is None or d < best[0]:
                        best = (d, i, j)
            d, i, j = best
            name = sg[j]["name"]
            result[vs[i]["id"]] = name
            costs[vs[i]["id"]] = round(d, 3)
            load[name] += 1
            groups_used[(vs[i]["group"], name)] = groups_used.get((vs[i]["group"], name), 0) + 1
            todo.remove(i)
    # neutral-whisper gets no singer: an AI singer cannot whisper, the voice is VoxaVersa's DSP whisper
    return result, costs


def note(m):
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    m = int(round(m))
    return "%s%d" % (names[m % 12], m // 12 - 1)


def write_table(singers, voices, result, costs, out):
    by_voice = {v["id"]: v for v in voices}
    data = dict(assignment=result, costs=costs, singers=singers,
                factory_voices={s["name"]: sorted(v for v, n in result.items() if n == s["name"]) for s in singers})
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    lines = ["# Built-in voices powered by trained singers", "",
             "Generated by `tools/training/factory_voices.py assign` from measurements of each singer's dataset "
             "(see docs/TRAINING.md). Re-run when more singers finish.", "",
             "## Singers", "", "| Singer | Gender | Minutes | Median note | Range | Tilt dB/oct | Air dB | Ring dB | Aperiodicity | Built-in voices |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(singers, key=lambda s: (s["gender"], s["name"])):
        lines.append("| %s | %s | %.1f | %s | %s - %s | %.1f | %.1f | %.1f | %.3f | %s |" % (
            s["name"], s["gender"], s["seconds"] / 60, note(s["median_midi"]), note(s["range_midi"][0]), note(s["range_midi"][1]),
            s["tilt_db_oct"], s["air_db"], s["ring_db"], s["aperiodicity"], ", ".join(data["factory_voices"][s["name"]]) or "-"))
    lines += ["", "## Built-in voices", "", "| Voice | Genre group | Register | Singer | Match cost |", "|---|---|---|---|---|"]
    for vid in sorted(result, key=lambda v: (by_voice.get(v, {}).get("gender", "z"), by_voice.get(v, {}).get("group", ""), v)):
        v = by_voice.get(vid, {})
        lines.append("| %s | %s | %s | %s | %s |" % (vid, v.get("group", "-"), note(v["register"]) if v else "-", result[vid], costs.get(vid)))
    unassigned = sorted(v["id"] for v in voices if v["id"] not in result)
    if unassigned:
        lines += ["", "Without a trained singer yet (DSP engine): " + ", ".join(unassigned)]
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--dataset", required=True)
    m.add_argument("--gender", required=True, choices=["male", "female"])
    m.add_argument("--name")
    m.add_argument("--corpus", help="recording source label (e.g. vendor or dataset)")
    m.add_argument("--style", choices=["contemporary", "classical"], default="contemporary")
    m.add_argument("--out", required=True)
    a = sub.add_parser("assign")
    a.add_argument("--singers", nargs="+", required=True, help="JSON files written by 'measure'")
    a.add_argument("--factory-cpp", default=FACTORY_CPP)
    a.add_argument("--out", required=True, help="output path without extension (.json and .md are written)")
    args = p.parse_args()
    if args.cmd == "measure":
        r = measure(args.dataset, args.gender)
        r["name"] = args.name or os.path.basename(os.path.abspath(args.dataset))
        r["corpus"], r["style"] = args.corpus, args.style
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
        print(json.dumps(r))
    else:
        singers = [json.load(open(s, encoding="utf-8")) for s in args.singers]
        voices = factory_voices(args.factory_cpp)
        result, costs = assign(singers, voices)
        write_table(singers, voices, result, costs, args.out)
        print("assigned %d of %d built-in voices to %d singers -> %s.md" % (len(result), len(voices), len(singers), args.out))


if __name__ == "__main__":
    main()
