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


def pick(args, clips, clusters, data):
    """The singer of each gender with the most clean minutes; their clips that sound like someone else go."""
    import soundfile as sf
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

    chosen, report = {}, []
    for k, (c, members) in enumerate(groups):
        E = np.stack([emb[p][0] for p in members])
        own = E @ centres[k]
        other = np.max(np.stack([E @ centres[j] for j in range(len(groups)) if j != k]), axis=0) if len(groups) > 1 \
            else np.full(len(members), -1.0)
        # a clip stays when it sounds more like this singer than like any other one and is not far below the
        # singer's typical similarity (another member singing a verse of the song)
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
                           lean=lean, dropped_clips=len(members) - len(keep), keep=keep))
        if gender and minutes >= args.min_minutes and gender not in chosen:
            chosen[gender] = report[-1]

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
        name = "military_%s_1" % gender
        dst = os.path.join(data, "datasets_military", name)
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        for p in r["keep"]:
            shutil.copy2(p, dst)
        voices.append({
            "name": name, "dataset": "datasets_military/" + name,
            "display_name": "Band %s 1" % gender.capitalize(), "gender": gender,
            "genre": "Pop", "styles": "Clean,Powerful",
            "license": "Public domain recordings: works of US military band members made as part of their duties "
                       "(not eligible for US copyright), via Wikimedia Commons; the singer is not named",
            "tests": [["tests/light_male_25s.wav", 0 if gender == "male" else 12],
                      ["tests/notmine_female_25s.wav", -12 if gender == "male" else 0]],
            "description": "RVC v2 voice trained on %.0f min of a US military band %s singer (public domain recordings)"
                           % (r["minutes"], "male" if gender == "male" else "female"),
            "listed": False, "corpus": "military", "style": "contemporary"})
    with open(os.path.join(data, "voices_military.json.tmp"), "w", encoding="utf-8") as f:
        json.dump({"voices": voices}, f, indent=1)
    os.replace(os.path.join(data, "voices_military.json.tmp"), os.path.join(data, "voices_military.json"))

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
    r.add_argument("--sources", default=SOURCES, help="CSV with the columns file,url (default: military_sources.csv)")
    r.add_argument("--limit", type=int, default=0, help="(testing) only the first N songs")
    r.add_argument("--seconds", type=float, default=0.0, help="(testing) only the first S seconds of each song")
    args = p.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
