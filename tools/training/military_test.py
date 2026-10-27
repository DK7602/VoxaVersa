#!/usr/bin/env python3
"""
The military-band singer test: training sets for one male and one female singer from recordings of US
military band singers (Air Force and Navy pop, rock, country, R&B and jazz groups). Works of federal
employees made as part of their duties are not eligible for US copyright; the recordings are published on
Wikimedia Commons with that statement (tools/training/military_sources.csv lists them with their pages).
The singers are not named in VoxaVersa (their right to their own voice is not waived), and many songs are
covers: only the voice is used.

Runs on the Colab GPU as a background job of colab_queue.py (docs/COLAB.md); every stage is resumable:

  fetch     downloads the songs (~90 files, ~6 h of music) with a polite, identified client
  separate  vocals out of the band (BS-Roformer, viperx ep 317), then reverb out (Mel-Band-Roformer, anvuew):
            band mixes put reverb on the voice, and prepare_dataset.py rejects reverberant clips (and clips with
            backing harmonies, so mostly the solo verses remain)
  clean     prepare_dataset.py: 3 - 10 s clips without harmonies, noise or reverb
  singers   cluster_singers.py groups the songs by singer (ECAPA); inside the chosen singer's songs, clips that
            don't sound like them (another member singing a verse) are dropped
  pick      the singer of each gender with the most clean minutes (at least --min-minutes) becomes a training set
            in <work>/data/datasets_military/, listed in <work>/data/voices_military.json for the queue;
            <work>/military/report.md (with 20 s previews) says what was found
  more      --singers "2, 4, 5 male theatre, 6 male country": more of the previewed singers become training
            sets (Band Male 2, ...), by their preview number; male / female for the unclear ones, and genres
            they suit (the built-in voices of those genres prefer them). Works after a new Colab session too:
            the clips are cut and grouped again and each preview is matched to its singer by voice print.

    military_test.py run --work W --rvc /content/rvc --scratch /content/scratch [--limit N --seconds S]

Requirements: the RVC WebUI checkout (its tools/pymss runs the separation models), torch with CUDA, numpy,
scipy, soundfile, speechbrain (cluster_singers.py).
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, "military_sources.csv")
USER_AGENT = "VoxaVersa-voice-research/0.1 (https://github.com/DK7602/VoxaVersa; public-domain training data test)"
MODELS = {
    "vocals": ("bs_roformer", "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.yaml",
               "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
               "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "vocals"),
    "dereverb": ("mel_band_roformer", "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "dereverb_mel_band_roformer_anvuew.yaml",
                 "https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/"
                 "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "noreverb"),
}
MALE_MAX_F0, FEMALE_MIN_F0 = 200.0, 260.0   # cluster_singers.py's gender guess from the median sung pitch


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def status(work, text):
    path = os.path.join(work, "colab_military_status.txt")
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def done(folder, stage):
    return os.path.exists(os.path.join(folder, "." + stage + "_done"))


def mark(folder, stage):
    open(os.path.join(folder, "." + stage + "_done"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


def download(url, dst, tries=6):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst + ".part", "wb") as out:
                shutil.copyfileobj(r, out, 1 << 20)
            os.replace(dst + ".part", dst)
            return True
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                log("  %s: HTTP %d" % (os.path.basename(dst), e.code))
                return False
            wait = min(600, int(e.headers.get("Retry-After") or 0) or 15 * 2 ** attempt)
            log("  %s: HTTP %d, retrying in %d s" % (os.path.basename(dst), e.code, wait))
            time.sleep(wait)
        except (urllib.error.URLError, OSError) as e:
            log("  %s: %s" % (os.path.basename(dst), e))
            time.sleep(10 * (attempt + 1))
    return False


# ------------------------------------------------------------------------------------------------- stages

def fetch(args, raw, vocals):
    """Downloads the songs not separated yet (the downloads live on the session's local disk, the separated
    voices on Drive: after a disconnect only the songs still to separate are fetched again)."""
    rows = list(csv.DictReader(open(args.sources, encoding="utf-8")))[: args.limit or None]
    os.makedirs(raw, exist_ok=True)
    failed = []
    for i, r in enumerate(rows):
        dst = os.path.join(raw, r["file"])
        if os.path.exists(dst) or os.path.exists(os.path.join(vocals, os.path.splitext(r["file"])[0] + ".flac")):
            continue
        status(args.work, "military-band singers: downloading song %d of %d" % (i + 1, len(rows)))
        if not download(r["url"], dst):
            failed.append(r["file"])
        time.sleep(2)   # Wikimedia's etiquette for bulk downloads
    got = len(rows) - len(failed)
    log("downloaded %d of %d songs" % (got, len(rows)))
    if got < max(1, len(rows) // 3):
        raise RuntimeError("Wikimedia refused most downloads (%d of %d failed); run all again later" % (len(failed), len(rows)))


def separator(rvc, weights, key):
    sys.path.insert(0, os.path.join(rvc, "tools"))
    import torch
    from pymss import MSSeparator
    model_type, ckpt, config, url, stem = MODELS[key]
    path = os.path.join(weights, ckpt)
    if not os.path.exists(path):
        log("downloading the %s model" % key)
        if not download(url, path):
            raise RuntimeError("could not download " + url)
    if not os.path.exists(os.path.join(weights, config)):
        raise RuntimeError("missing %s in %s (the RVC WebUI checkout ships it)" % (config, weights))
    cuda = torch.cuda.is_available()
    sep = MSSeparator(model_type=model_type, model_path=path, config_path=os.path.join(weights, config),
                      device="cuda" if cuda else "cpu", device_ids=[0], output_format="wav", use_tta=False,
                      store_dirs={}, audio_params={"wav_bit_depth": "FLOAT"}, logger=None, debug=False,
                      progress_callback=None,
                      # batch 1: the GPU trains voices at the same time
                      inference_params={"batch_size": 1, "chunk_size": 352800, "overlap_size": 176400,
                                        "standardize": False, "normalize": False, "use_amp": cuda,
                                        "model_dtype": "auto", "cuda_attention_backend": "default"})
    return sep, stem


def separate(args, raw, vocals):
    import soundfile as sf
    sys.path.insert(0, os.path.join(args.rvc, "tools"))
    from pymss import load_audio
    os.makedirs(vocals, exist_ok=True)
    weights = os.path.join(args.rvc, "assets", "pymss_weights")
    files = sorted(f for f in glob.glob(os.path.join(raw, "*")) if not f.endswith(".part"))
    todo = [f for f in files if not os.path.exists(os.path.join(vocals, os.path.splitext(os.path.basename(f))[0] + ".flac"))]
    if not todo:
        return
    passes = [separator(args.rvc, weights, "vocals"), separator(args.rvc, weights, "dereverb")]
    for i, f in enumerate(todo):
        status(args.work, "military-band singers: separating the voice from the band, song %d of %d"
                          % (len(files) - len(todo) + i + 1, len(files)))
        t = time.time()
        try:
            audio, sr = load_audio(f, sr=44100, mono=False)
        except Exception as e:
            log("  %s: cannot decode (%s)" % (os.path.basename(f), e))
            continue
        if args.seconds:
            audio = audio[..., : int(args.seconds * sr)]
        for sep, stem in passes:
            out = sep.separate(audio, pbar=False)[stem]
            audio = np.asarray(out, dtype=np.float32)
            audio = audio if audio.shape[0] in (1, 2) else audio.T
        # mono FLAC: prepare_dataset.py mixes to mono anyway, and it keeps the copy on Drive small (~1 GB)
        dst = os.path.join(vocals, os.path.splitext(os.path.basename(f))[0] + ".flac")
        sf.write(dst + ".part.flac", audio.mean(axis=0), sr, subtype="PCM_16")
        os.replace(dst + ".part.flac", dst)
        log("  %s: %.0f s in %.0f s" % (os.path.basename(f), audio.shape[-1] / sr, time.time() - t))


def clean(args, vocals, clips):
    status(args.work, "military-band singers: cutting clean clips")
    subprocess.run([sys.executable, os.path.join(HERE, "prepare_dataset.py"), vocals, "--out", clips, "--name", "pool",
                    "--clean"], check=True)
    if not glob.glob(os.path.join(clips, "pool", "*.wav")):
        raise RuntimeError("no clean solo singing survived the checks (see %s)" % os.path.join(clips, "report.md"))


def singers(args, clips, clusters):
    status(args.work, "military-band singers: telling the singers apart")
    cmd = [sys.executable, os.path.join(HERE, "cluster_singers.py"), os.path.join(clips, "pool"),
           "--report", os.path.join(clips, "report.json"), "--out", clusters, "--threads", "4"]
    if args.ecapa and os.path.isdir(args.ecapa):
        os.makedirs(clusters, exist_ok=True)
        if not os.path.exists(os.path.join(clusters, "ecapa_model")):
            shutil.copytree(args.ecapa, os.path.join(clusters, "ecapa_model"))
    subprocess.run(cmd, check=True)


def clip_embeddings(clusters):
    z = np.load(os.path.join(clusters, "embeddings.npz"), allow_pickle=False)
    return {str(k).split("|")[0]: (z["emb"][i], float(z["dur"][i]), float(z["f0"][i])) for i, k in enumerate(z["keys"])}


def gender_directions(data, clusters, per_voice=12):
    """Mean speaker embeddings of the male and of the female singers already in the training data. Pitch alone
    can't tell: male pop leads often sing around 300 Hz. Returns {} when there are no references."""
    import soundfile as sf
    sys.path.insert(0, HERE)
    from cluster_singers import MODEL, load_16k
    voices = []
    for name in ("voices.json", "voices_military.json"):
        try:
            voices += json.load(open(os.path.join(data, name), encoding="utf-8"))["voices"]
        except (OSError, ValueError):
            pass
    refs = {"male": [], "female": []}
    for v in voices:
        folder = os.path.join(data, v["dataset"])
        if v.get("gender") in refs and os.path.isdir(folder) and not v["dataset"].startswith("datasets_military/"):
            refs[v["gender"]] += sorted(glob.glob(os.path.join(folder, "*.wav")))[:per_voice]
    if not refs["male"] or not refs["female"]:
        return {}
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(source=MODEL, savedir=os.path.join(clusters, "ecapa_model"), run_opts={"device": "cpu"})
    out = {}
    for g, paths in refs.items():
        E = []
        for p in paths:
            with torch.no_grad():
                e = model.encode_batch(torch.from_numpy(load_16k(p))[None]).squeeze().numpy()
            E.append(e / (np.linalg.norm(e) + 1e-9))
        m = np.mean(E, axis=0)
        out[g] = m / np.linalg.norm(m)
    log("gender references: %d male and %d female clips" % (len(refs["male"]), len(refs["female"])))
    return out


def describe(clusters, data):
    """Every singer the clustering found: gender, clean minutes, and the clips that sound like them (a clip goes
    when it sounds more like another singer, or far less like this one than usual: another member singing a
    verse of the song)."""
    result = json.load(open(os.path.join(clusters, "clusters.json"), encoding="utf-8"))
    emb = clip_embeddings(clusters)
    by_stem = {}
    for path in emb:
        by_stem.setdefault(os.path.basename(path).rsplit("_", 1)[0], []).append(path)

    groups = [(c, [p for s in c["stems"] for p in by_stem.get(s["stem"], [])]) for c in result["clusters"]
              if c["cluster"] != "unassigned"]
    groups = [(c, m) for c, m in groups if m]
    centres = []
    for c, members in groups:
        m = np.mean([emb[p][0] for p in members], axis=0)
        centres.append(m / np.linalg.norm(m))
    directions = gender_directions(data, clusters)

    report = []
    for k, (c, members) in enumerate(groups):
        E = np.stack([emb[p][0] for p in members])
        own = E @ centres[k]
        other = np.max(np.stack([E @ centres[j] for j in range(len(groups)) if j != k]), axis=0) if len(groups) > 1 \
            else np.full(len(members), -1.0)
        keep = [p for p, o, x in zip(members, own, other) if o >= x + 0.02 and o >= float(np.median(own)) - 0.15]
        minutes = sum(emb[p][1] for p in keep) / 60
        if directions:
            lean = float(centres[k] @ directions["male"] - centres[k] @ directions["female"])
            gender = "male" if lean > 0.03 else ("female" if lean < -0.03 else None)
        else:
            lean = None
            gender = "male" if 0 < c["median_f0"] < MALE_MAX_F0 else ("female" if c["median_f0"] > FEMALE_MIN_F0 else None)
        report.append(dict(cluster=c["cluster"], gender=gender or "unclear", minutes=round(minutes, 1),
                           songs=len(c["stems"]), median_f0=c["median_f0"], confidence=(c.get("confidence") or "").split(" ")[0],
                           lean=lean, dropped_clips=len(members) - len(keep), keep=keep, centre=centres[k]))
    return report


def voice_entry(gender, number, r, genres=()):
    name = "military_%s_%d" % (gender, number)
    return {
        "name": name, "dataset": "datasets_military/" + name,
        "display_name": "Band %s %d" % (gender.capitalize(), number), "gender": gender,
        "genre": "Pop", "styles": "Clean,Powerful", "cluster": r["cluster"], "genres": list(genres),
        "license": "Public domain recordings: works of US military band members made as part of their duties "
                   "(not eligible for US copyright), via Wikimedia Commons; the singer is not named",
        "tests": [["tests/light_male_25s.wav", 0 if gender == "male" else 12],
                  ["tests/notmine_female_25s.wav", -12 if gender == "male" else 0]],
        "description": "RVC v2 voice trained on %.0f min of a US military band %s singer (public domain recordings)"
                       % (r["minutes"], gender),
        "listed": False, "corpus": "military", "style": "contemporary"}


def copy_clips(data, voice, r):
    dst = os.path.join(data, voice["dataset"])
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for p in r["keep"]:
        shutil.copy2(p, dst)


def save_voices(data, voices):
    with open(os.path.join(data, "voices_military.json.tmp"), "w", encoding="utf-8") as f:
        json.dump({"voices": voices}, f, indent=1)
    os.replace(os.path.join(data, "voices_military.json.tmp"), os.path.join(data, "voices_military.json"))


def pick(args, clips, clusters, data):
    """The singer of each gender with the most clean minutes."""
    import soundfile as sf
    report = describe(clusters, data)
    chosen = {}
    for r in report:
        if r["gender"] != "unclear" and r["minutes"] >= args.min_minutes and r["gender"] not in chosen:
            chosen[r["gender"]] = r

    # preview: 20 s of each singer with at least 3 minutes
    previews = os.path.join(args.work, "military", "previews")
    os.makedirs(previews, exist_ok=True)
    for r in report:
        if r["minutes"] < 3:
            continue
        audio, sr, total = [], 40000, 0.0
        for p in r["keep"]:
            x, sr = sf.read(p, dtype="float32")
            audio.append(x)
            total += len(x) / sr
            if total >= 20:
                break
        sf.write(os.path.join(previews, "%s_%s.wav" % (r["cluster"], r["gender"])), np.concatenate(audio), sr)

    voices = []
    for gender, r in sorted(chosen.items()):
        voices.append(voice_entry(gender, 1, r))
        copy_clips(data, voices[-1], r)
    save_voices(data, voices)

    lines = ["# Military-band singers", "",
             "Songs downloaded, voices separated from the band and de-reverbed, cut into clean clips and grouped "
             "by singer. Gender comes from comparing each singer's voice print with the male and female singers "
             "already in training (lean: + male, - female); check the previews by ear "
             "(*My Drive > VoxaVersa > queue > military > previews*).", "",
             "| Singer | Gender | Clean minutes | Songs | Median pitch (Hz) | Male/female lean | Confidence | Clips dropped (another voice) | Training |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in report:
        trained = next((g for g, c in chosen.items() if c is r), None)
        lines.append("| %s | %s | %.1f | %d | %.0f | %s | %s | %d | %s |" % (
            r["cluster"], r["gender"], r["minutes"], r["songs"], r["median_f0"],
            "-" if r["lean"] is None else "%+.2f" % r["lean"], r["confidence"], r["dropped_clips"],
            "yes: Band %s 1" % trained.capitalize() if trained else ""))
    missing = [g for g in ("male", "female") if g not in chosen]
    if missing:
        lines += ["", "No %s singer with at least %.0f clean minutes was found." % (" or ".join(missing), args.min_minutes)]
    open(os.path.join(args.work, "military", "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log("\n".join(lines))
    return voices


GENRES = {"pop", "kpop", "latin", "edm", "afrobeats", "rnb", "soul", "jazz", "blues", "lofi", "gospel", "rock", "metal",
          "opera", "theatre", "rap", "trap", "reggae", "country", "indie"}
ALIASES = {"r&b": "rnb", "rb": "rnb", "theater": "theatre", "broadway": "theatre", "k-pop": "kpop", "lo-fi": "lofi",
           "hip-hop": "rap", "hiphop": "rap"}
MORE_MIN_MINUTES = 5.0   # thinner than the --min-minutes singers, but RVC still learns a voice from it


def parse_singers(spec):
    """ "2, 4, 5 male theatre, 6 male country" -> [{"number": 2, "gender": None, "genres": []}, ...]"""
    wanted = []
    for part in spec.replace(";", ",").split(","):
        words = part.lower().split()
        if not words:
            continue
        number = int("".join(ch for ch in words[0] if ch.isdigit()) or 0)
        if not number:
            raise ValueError("'%s': start with the singer's number from the previews" % part.strip())
        gender, genres = None, []
        for w in words[1:]:
            w = ALIASES.get(w, w)
            if w in ("male", "man", "m"):
                gender = "male"
            elif w in ("female", "woman", "f"):
                gender = "female"
            elif w in GENRES:
                genres.append(w)
            else:
                log("  singer %d: '%s' is not a gender or a genre VoxaVersa has - ignored" % (number, w))
        wanted.append({"number": number, "gender": gender, "genres": genres})
    return wanted


def preview_of(work, number):
    found = glob.glob(os.path.join(work, "military", "previews", "singer_%02d_*.wav" % number))
    return found[0] if found else None


def match_previews(work, clusters, report, wanted):
    """The singer behind each wanted preview, by voice print (a new session groups the clips again, and its
    numbering may differ from the previews')."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    sys.path.insert(0, HERE)
    from cluster_singers import MODEL, load_16k
    model = EncoderClassifier.from_hparams(source=MODEL, savedir=os.path.join(clusters, "ecapa_model"), run_opts={"device": "cpu"})
    C = np.stack([r["centre"] for r in report])
    found = {}
    for w in wanted:
        path = preview_of(work, w["number"])
        if not path:
            continue
        with torch.no_grad():
            e = model.encode_batch(torch.from_numpy(load_16k(path))[None]).squeeze().numpy()
        sims = C @ (e / (np.linalg.norm(e) + 1e-9))
        order = np.argsort(sims)[::-1]
        log("  preview %d: closest singer %s (similarity %.2f, next %.2f)" % (
            w["number"], report[order[0]]["cluster"], sims[order[0]], sims[order[1]] if len(order) > 1 else 0))
        if sims[order[0]] >= 0.3:
            found[w["number"]] = report[order[0]]
    return found


def add_singers(args, vocals, clips, clusters, data, spec):
    """Trains more of the previewed singers (--singers); the ones already in training stay as they are."""
    wanted = parse_singers(spec)
    status(args.work, "military-band singers: preparing the extra singers (%s)" % spec)
    regrouped = not os.path.exists(os.path.join(clusters, "clusters.json"))
    if regrouped:   # a new Colab session: the clips and the grouping were on the old session's disk
        clean(args, vocals, clips)
        singers(args, clips, clusters)
    report = describe(clusters, data)
    if regrouped:
        by_number = match_previews(args.work, clusters, report, wanted)
    else:
        by_number = {int(r["cluster"].rsplit("_", 1)[-1]): r for r in report if r["cluster"].rsplit("_", 1)[-1].isdigit()}

    voices = json.load(open(os.path.join(data, "voices_military.json"), encoding="utf-8"))["voices"]

    def trained_as(r):
        clipnames = {os.path.basename(p) for p in r["keep"]}
        for v in voices:
            if v.get("cluster") == r["cluster"] and not regrouped:
                return v
            have = set(os.listdir(os.path.join(data, v["dataset"]))) if os.path.isdir(os.path.join(data, v["dataset"])) else set()
            if clipnames and len(clipnames & have) > 0.3 * len(clipnames):
                return v
        return None

    lines = ["", "## More singers (asked for: %s)" % spec, ""]
    for w in wanted:
        r = by_number.get(w["number"])
        if r is None:
            lines.append("* singer %d: not found%s" % (w["number"], "" if preview_of(args.work, w["number"]) else
                                                       " (no preview with that number)"))
            continue
        already = trained_as(r)
        if already:
            if w["genres"] and not already.get("genres"):
                already["genres"] = w["genres"]
            lines.append("* singer %d: already in training as %s" % (w["number"], already["display_name"]))
            continue
        gender = w["gender"] or (r["gender"] if r["gender"] != "unclear" else None)
        if gender is None:
            lines.append("* singer %d: male or female? Add it after the number (e.g. \"%d male\")" % (w["number"], w["number"]))
            continue
        if r["minutes"] < MORE_MIN_MINUTES:
            lines.append("* singer %d: only %.1f clean minutes - too little to train (at least %.0f)"
                         % (w["number"], r["minutes"], MORE_MIN_MINUTES))
            continue
        number = 1 + max([int(v["name"].rsplit("_", 1)[-1]) for v in voices if v["gender"] == gender] or [0])
        v = voice_entry(gender, number, r, w["genres"])
        copy_clips(data, v, r)
        voices.append(v)
        lines.append("* singer %d: %s, %.1f clean minutes%s%s" % (
            w["number"], v["display_name"], r["minutes"], (", suits " + ", ".join(w["genres"])) if w["genres"] else "",
            " (under %.0f minutes: may sound thinner)" % args.min_minutes if r["minutes"] < args.min_minutes else ""))
    save_voices(data, voices)
    with open(os.path.join(args.work, "military", "report.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))
    with open(os.path.join(args.work, "military", ".more_singers"), "w", encoding="utf-8") as f:
        f.write(normalise(spec))
    return voices


def normalise(spec):
    return ",".join(" ".join(part.lower().split()) for part in spec.replace(";", ",").split(",") if part.strip())


def more_wanted(work, spec):
    """True when --singers asks for something the last run didn't."""
    try:
        done_spec = open(os.path.join(work, "military", ".more_singers"), encoding="utf-8").read()
    except OSError:
        done_spec = ""
    return bool(spec.strip()) and normalise(spec) != done_spec


def cmd_run(args):
    scratch = os.path.join(args.scratch, "military")
    raw, vocals, clips, clusters = (os.path.join(scratch, d) for d in ("raw", "vocals", "clips", "clusters"))
    state = os.path.join(args.work, "military")
    data = os.path.join(args.work, "data")
    os.makedirs(state, exist_ok=True)
    # the separated voices are the expensive part: kept on Drive, so a disconnect never repeats them
    vocals = os.path.join(state, "vocals")
    try:
        if not done(state, "separate"):
            fetch(args, raw, vocals)
            separate(args, raw, vocals)
            mark(state, "separate")
            shutil.rmtree(raw, ignore_errors=True)
        if not done(state, "pick"):
            clean(args, vocals, clips)
            singers(args, clips, clusters)
            voices = pick(args, clips, clusters, data)
            mark(state, "pick")
        else:
            voices = json.load(open(os.path.join(data, "voices_military.json")))["voices"]
        if more_wanted(args.work, args.singers):
            voices = add_singers(args, vocals, clips, clusters, data, args.singers)
        names = ", ".join(v["display_name"] for v in voices) or "none"
        status(args.work, "military-band singers: ready (%s in the queue; report in queue/military/report.md)" % names)
    except BaseException as e:
        traceback.print_exc()
        status(args.work, "military-band singers: failed (%s) - training continues with the other singers; details in "
                          "My Drive/VoxaVersa/queue/colab_logs/military.log" % str(e).strip()[-160:])
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--work", required=True, help="the queue folder on Google Drive")
    r.add_argument("--rvc", required=True, help="RVC WebUI checkout (tools/pymss, assets/pymss_weights)")
    r.add_argument("--scratch", default="/content/scratch", help="local (not Drive) working folder")
    r.add_argument("--ecapa", default="/content/ecapa_model", help="speaker model folder, if already downloaded")
    r.add_argument("--min-minutes", type=float, default=10.0, help="least clean audio for a singer to be trained")
    r.add_argument("--singers", default="", help='more previewed singers to train, e.g. "2, 4, 5 male theatre, 6 male country"')
    r.add_argument("--sources", default=SOURCES, help="CSV with the columns file,url (default: military_sources.csv)")
    r.add_argument("--limit", type=int, default=0, help="(testing) only the first N songs")
    r.add_argument("--seconds", type=float, default=0.0, help="(testing) only the first S seconds of each song")
    args = p.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
