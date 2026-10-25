#!/usr/bin/env python3
"""
Trains an RVC v2 voice model for VoxaVersa with a pinned copy of the official RVC WebUI
(RVC-Project/Retrieval-based-Voice-Conversion-WebUI, MIT licence) - from the command line, on a GPU
(Google Colab: see VoxaVersa_Voice_Training.ipynb) or, slowly, on a CPU.

    python3 tools/training/rvc_train.py setup --rvc-dir rvc                     # once: code + base models
    python3 tools/training/rvc_train.py train --rvc-dir rvc --dataset my_voice.zip --name my_voice --epochs 300
    python3 tools/training/rvc_train.py export --rvc-dir rvc --name my_voice --out my_voice.onnx

`train` runs the whole pipeline and can be re-run at any time: every stage skips work that is done, and
training continues from the last checkpoint (saved every --save-every epochs).

  1. preprocess  RVC's slicer: 3.7 s pieces at the model rate + 16 kHz copies (train/preprocess.py)
  2. f0          RMVPE pitch per 10 ms. Unvoiced frames stay 0 Hz (coarse pitch 1) like the RVC
                 releases up to 2.2 and every community model, because that is what VoxaVersa feeds for
                 breaths and consonants. (RVC 2.3 interpolates the pitch through unvoiced gaps instead;
                 --f0-unvoiced interpolate selects that.)
  3. features    ContentVec/HuBERT layer-12 features (768-d, 20 ms) with RVC's own extractor
                 (train/dataset/extract_hubert_feature.py, transformers model assets/hubert_base)
  4. filelist    filelist.txt + config.json like the WebUI's "Train model" button (2 silence samples added)
  5. train       train/train.py, fine-tuning the pretrained v2 base (f0G40k/f0D40k or 48k)

`export` writes the generator as ONNX in exactly the format VoxaVersa loads (export_onnx.py).

Python 3.11 or 3.12 with the packages from VoxaVersa_Voice_Training.ipynb (torch, torchaudio, transformers
4.49, librosa 0.10, ...). Only train on recordings you made or have explicit permission to use.
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

RVC_REPO = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
RVC_COMMIT = "81eed5e8f68b6bed1789f682fe78cdd324495afc"     # main, 2026-08-04 (release 2.3.260718 + fixes)
HF_REPO = "lj1995/VoiceConversionWebUI"                       # MIT
HF_REVISION = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"      # 2026-08-01
ASSETS = {
    "hubert_base/config.json": "assets/hubert_base/config.json",
    "hubert_base/preprocessor_config.json": "assets/hubert_base/preprocessor_config.json",
    "hubert_base/pytorch_model.bin": "assets/hubert_base/pytorch_model.bin",
    "rmvpe.pt": "assets/rmvpe/rmvpe.pt",
    "pretrained_v2/f0G40k.pth": "assets/pretrained_v2/f0G40k.pth",
    "pretrained_v2/f0D40k.pth": "assets/pretrained_v2/f0D40k.pth",
    "pretrained_v2/f0G48k.pth": "assets/pretrained_v2/f0G48k.pth",
    "pretrained_v2/f0D48k.pth": "assets/pretrained_v2/f0D48k.pth",
    "mute.zip": ".downloads/mute.zip",
}
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")


def log(msg):
    print("[rvc_train] " + msg, flush=True)


def run(cmd, cwd, env=None):
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode != 0:
        sys.exit("command failed (exit code %d): %s" % (p.returncode, " ".join(cmd)))


SITECUSTOMIZE = """# written by VoxaVersa rvc_train.py --loader-workers: caps the data loader processes of RVC's
# train.py (it starts 4, each holding ~0.4 GB) on machines with little memory.
import os
_n = os.environ.get("VOXAVERSA_LOADER_WORKERS")
if _n:
    import torch.utils.data as _data
    _init = _data.DataLoader.__init__
    def _capped(self, *a, **k):
        if k.get("num_workers", 0) > int(_n):
            k["num_workers"] = int(_n)
            if int(_n) == 0:
                k.pop("prefetch_factor", None)
                k["persistent_workers"] = False
            elif k.get("prefetch_factor"):
                k["prefetch_factor"] = min(k["prefetch_factor"], 2)
        _init(self, *a, **k)
    _data.DataLoader.__init__ = _capped
"""


def rvc_env(rvc_dir, threads=None, device="auto", loader_workers=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = rvc_dir + os.pathsep + env.get("PYTHONPATH", "")
    if loader_workers is not None:
        hook_dir = os.path.join(rvc_dir, ".voxaversa")
        os.makedirs(hook_dir, exist_ok=True)
        with open(os.path.join(hook_dir, "sitecustomize.py"), "w") as f:
            f.write(SITECUSTOMIZE)
        env["PYTHONPATH"] = hook_dir + os.pathsep + env["PYTHONPATH"]
        env["VOXAVERSA_LOADER_WORKERS"] = str(loader_workers)
    # The scripts are started as train/<script>.py; without this, Python would put train/ first on the
    # module path and train/train.py would shadow the 'train' package (needs Python 3.11+).
    env["PYTHONSAFEPATH"] = "1"
    env["RVC_CUDA_GRAPH"] = "0"
    env.setdefault("rmvpe_root", "assets/rmvpe")
    env.setdefault("weight_root", "assets/weights")
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        # glibc keeps freed tensor memory in its heap when the shapes change from step to step; on the
        # CPU this made RSS grow past 6 GB at batch 2. A fixed mmap threshold returns it (stable ~4 GB).
        env.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")
        env.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")
        env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


# ------------------------------------------------------------------------------------------- setup

def download(url, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        from urllib.request import urlopen
        with urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    except Exception:
        if shutil.which("curl") is None:
            raise
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", tmp, url], check=True)
    os.replace(tmp, dest)


def cmd_setup(args):
    rvc = os.path.abspath(args.rvc_dir)
    if not os.path.isdir(os.path.join(rvc, ".git")):
        os.makedirs(rvc, exist_ok=True)
        run(["git", "init", "-q"], rvc)
        run(["git", "remote", "add", "origin", args.rvc_repo], rvc)
    have = subprocess.run(["git", "rev-parse", "HEAD"], cwd=rvc, capture_output=True, text=True).stdout.strip()
    if have != args.rvc_commit:
        run(["git", "fetch", "-q", "--depth", "1", "origin", args.rvc_commit], rvc)
        run(["git", "checkout", "-q", "FETCH_HEAD"], rvc)
    log("RVC WebUI at %s (%s)" % (rvc, args.rvc_commit[:10]))

    wanted = [k for k in ASSETS if args.sr in k or "pretrained" not in k]
    for remote in wanted:
        dest = os.path.join(rvc, ASSETS[remote])
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            continue
        local = os.path.join(args.assets_from, remote) if args.assets_from else None
        if local and os.path.isfile(local):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(local, dest)
            log("copied " + remote)
        else:
            log("downloading " + remote)
            download("https://huggingface.co/%s/resolve/%s/%s" % (HF_REPO, HF_REVISION, remote), dest)
    mute = os.path.join(rvc, "logs", "mute")
    if not os.path.isdir(mute):
        with zipfile.ZipFile(os.path.join(rvc, ASSETS["mute.zip"])) as z:
            z.extractall(os.path.join(rvc, "logs"))
    for d in ("assets/weights", "assets/indices"):
        os.makedirs(os.path.join(rvc, d), exist_ok=True)
    log("setup complete")


# ------------------------------------------------------------------------------------------- train

def stage_dataset(dataset, exp_dir):
    """Returns a folder with the dataset's audio files (unzipping archives into the experiment)."""
    if os.path.isdir(dataset):
        files = [f for f in glob.glob(os.path.join(dataset, "**", "*"), recursive=True) if f.lower().endswith(AUDIO_EXT)]
        if not files:
            sys.exit("no audio files in " + dataset)
        if all(os.path.dirname(f) == os.path.abspath(dataset).rstrip("/") or os.path.dirname(f) == dataset.rstrip("/") for f in files):
            return os.path.abspath(dataset)
        src = files
    elif zipfile.is_zipfile(dataset):
        src = None
    else:
        sys.exit("dataset must be a folder or a .zip of audio files: " + dataset)
    flat = os.path.join(exp_dir, "dataset")
    if os.path.isdir(flat) and os.listdir(flat):
        return flat
    os.makedirs(flat, exist_ok=True)
    if src is None:
        with zipfile.ZipFile(dataset) as z:
            for info in z.infolist():
                name = os.path.basename(info.filename)
                if info.is_dir() or not name.lower().endswith(AUDIO_EXT) or name.startswith(".") \
                        or "__MACOSX" in info.filename:
                    continue
                name = info.filename.strip("/").replace("/", "_")
                with z.open(info) as fi, open(os.path.join(flat, name), "wb") as fo:
                    shutil.copyfileobj(fi, fo)
    else:
        for f in src:   # sub/folder/x.wav -> sub_folder_x.wav (RVC's preprocess reads one flat folder)
            rel = os.path.relpath(f, dataset).replace(os.sep, "_")
            shutil.copy2(f, os.path.join(flat, rel))
    log("dataset: %d files" % len(os.listdir(flat)))
    return flat


def stage_f0(rvc, exp_dir, device, unvoiced, threads):
    """RMVPE f0 for every 16 kHz piece -> 2a_f0 (coarse 1..255) and 2b-f0nsf (Hz)."""
    import numpy as np
    sys.path.insert(0, rvc)
    cwd = os.getcwd()
    os.chdir(rvc)
    try:
        import torch
        if threads:
            torch.set_num_threads(threads)
        from infer.audio import load_audio
        from infer.rmvpe import RMVPE
        src = os.path.join(exp_dir, "1_16k_wavs")
        out_c, out_f = os.path.join(exp_dir, "2a_f0"), os.path.join(exp_dir, "2b-f0nsf")
        os.makedirs(out_c, exist_ok=True)
        os.makedirs(out_f, exist_ok=True)
        names = sorted(n for n in os.listdir(src) if n.endswith(".wav"))
        todo = [n for n in names if not (os.path.exists(os.path.join(out_c, n + ".npy"))
                                         and os.path.exists(os.path.join(out_f, n + ".npy")))]
        if not todo:
            return
        use_cuda = device != "cpu" and torch.cuda.is_available()
        model = RMVPE("assets/rmvpe/rmvpe.pt", is_half=use_cuda, device="cuda" if use_cuda else "cpu")
        mel_min, mel_max = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
        for i, n in enumerate(todo):
            x = load_audio(os.path.join(src, n), 16000)
            f0 = np.asarray(model.infer_from_audio(x, thred=0.03), dtype=np.float64)
            voiced = f0 > 0
            if not voiced.any():
                log("no pitch in %s - skipped" % n)
                continue
            if unvoiced == "interpolate":
                f0[~voiced] = np.interp(np.where(~voiced)[0], np.where(voiced)[0], f0[voiced])
            mel = 1127 * np.log(1 + f0 / 700)
            mel[mel > 0] = (mel[mel > 0] - mel_min) * 254 / (mel_max - mel_min) + 1
            mel[mel <= 1] = 1
            mel[mel > 255] = 255
            np.save(os.path.join(out_f, n + ".npy"), f0, allow_pickle=False)
            np.save(os.path.join(out_c, n + ".npy"), np.rint(mel).astype(np.int64), allow_pickle=False)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                log("f0 %d/%d" % (i + 1, len(todo)))
    finally:
        os.chdir(cwd)


def write_filelist(rvc, exp_dir, sr, speaker_id=0):
    gt = os.path.join(exp_dir, "0_gt_wavs")
    fe = os.path.join(exp_dir, "3_feature768")
    f0c = os.path.join(exp_dir, "2a_f0")
    f0f = os.path.join(exp_dir, "2b-f0nsf")

    def stems(d, suffix):
        return {n[:-len(suffix)] for n in os.listdir(d) if n.endswith(suffix)}

    names = stems(gt, ".wav") & stems(fe, ".npy") & stems(f0c, ".wav.npy") & stems(f0f, ".wav.npy")
    if not names:
        sys.exit("no complete training items (preprocess/f0/features failed?)")
    lines = ["%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%d" % (gt, n, fe, n, f0c, n, f0f, n, speaker_id)
             for n in sorted(names)]
    mute = os.path.join(rvc, "logs", "mute")
    for _ in range(2):
        lines.append("%s/0_gt_wavs/mute%s.wav|%s/3_feature768/mute.npy|%s/2a_f0/mute.wav.npy|%s/2b-f0nsf/mute.wav.npy|%d"
                     % (mute, sr, mute, mute, mute, speaker_id))
    random.Random(1234).shuffle(lines)
    with open(os.path.join(exp_dir, "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        src = os.path.join(rvc, "configs", "v1" if sr == "40k" else "v2", sr + ".json")
        shutil.copy(src, config_path)
    return len(names)


def cmd_train(args):
    rvc = os.path.abspath(args.rvc_dir)
    if not os.path.isfile(os.path.join(rvc, ASSETS["rmvpe.pt"])):
        sys.exit("run 'setup' first")
    py = args.python or sys.executable
    exp_dir = os.path.join(rvc, "logs", args.name)
    os.makedirs(exp_dir, exist_ok=True)
    sr_hz = {"32k": 32000, "40k": 40000, "48k": 48000}[args.sr]
    env = rvc_env(rvc, args.threads, args.device, args.loader_workers)
    workers = str(args.workers or max(1, min(4, os.cpu_count() or 1)))

    with open(os.path.join(exp_dir, "voxaversa_train.json"), "w") as f:
        json.dump(dict(vars(args), rvc_commit=RVC_COMMIT, hf_revision=HF_REVISION), f, indent=1)

    gt = os.path.join(exp_dir, "0_gt_wavs")
    if not (os.path.isdir(gt) and os.listdir(gt)):
        data = stage_dataset(args.dataset, exp_dir)
        log("1/5 preprocess")
        run([py, "train/preprocess.py", data, str(sr_hz), workers, exp_dir, "False", "3.7"], rvc, env)
    log("2/5 f0 (RMVPE, unvoiced = %s)" % args.f0_unvoiced)
    run([py, os.path.abspath(__file__), "_f0", rvc, exp_dir, args.device, args.f0_unvoiced, str(args.threads or 0)], rvc, env)
    log("3/5 features (HuBERT/ContentVec layer 12)")
    use_cuda = args.device != "cpu" and _cuda_available()
    if use_cuda:
        run([py, "train/dataset/extract_hubert_feature.py", "cuda:0", "1", "0", "0", exp_dir, "v2", "true"], rvc, env)
    else:
        run([py, "train/dataset/extract_hubert_feature.py", "cpu", "1", "0", exp_dir, "v2", "false"], rvc, env)
    n = write_filelist(rvc, exp_dir, args.sr)
    log("4/5 filelist: %d pieces (%.1f min)" % (n, n * 3.7 / 60 * 0.85))
    log("5/5 training %d epochs (batch %d, checkpoint every %d)" % (args.epochs, args.batch, args.save_every))
    pre = os.path.join("assets", "pretrained_v2")
    cmd = [py, "train/train.py", "-e", args.name, "-sr", args.sr, "-f0", "1", "-bs", str(args.batch),
           "-te", str(args.epochs), "-se", str(args.save_every),
           "-pg", os.path.join(pre, "f0G%s.pth" % args.sr), "-pd", os.path.join(pre, "f0D%s.pth" % args.sr),
           "-l", "1", "-c", "1" if (use_cuda and args.cache_gpu) else "0", "-sw", "1", "-v", "v2"]
    if use_cuda:
        cmd[cmd.index("-te"):cmd.index("-te")] = ["-g", "0"]
    run(cmd, rvc, env)
    final = os.path.join(rvc, "assets", "weights", args.name + ".pth")
    log("done: %s" % (final if os.path.exists(final) else latest_weights(rvc, args.name)))


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def latest_weights(rvc, name):
    """The newest small model RVC saved for an experiment (final <name>.pth or <name>_e<E>_s<S>.pth)."""
    final = os.path.join(rvc, "assets", "weights", name + ".pth")
    if os.path.exists(final):
        return final
    cands = glob.glob(os.path.join(rvc, "assets", "weights", name + "_e*_s*.pth"))
    if not cands:
        return None

    def epoch(p):
        try:
            return int(os.path.basename(p).rsplit("_e", 1)[1].split("_s")[0])
        except ValueError:
            return -1
    return max(cands, key=epoch)


def cmd_export(args):
    rvc = os.path.abspath(args.rvc_dir)
    pth = args.pth or latest_weights(rvc, args.name)
    if not pth:
        sys.exit("no trained model found for '%s' in %s/assets/weights" % (args.name, rvc))
    sys.path.insert(0, HERE)
    import export_onnx
    export_onnx.export(pth, args.out, rvc_dir=rvc, check=not args.no_check)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_f0":                  # internal: f0 stage in its own process
        rvc, exp_dir, device, unvoiced, threads = sys.argv[2:7]
        stage_f0(rvc, exp_dir, device, unvoiced, int(threads) or None)
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="get the pinned RVC code and the base models")
    s.add_argument("--rvc-dir", required=True)
    s.add_argument("--rvc-repo", default=RVC_REPO)
    s.add_argument("--rvc-commit", default=RVC_COMMIT)
    s.add_argument("--sr", default="40k", choices=["40k", "48k"], help="which pretrained base to download")
    s.add_argument("--assets-from", help="folder with already downloaded files (same layout as the HF repo)")
    t = sub.add_parser("train", help="preprocess, extract and train (resumable)")
    t.add_argument("--rvc-dir", required=True)
    t.add_argument("--dataset", required=True, help="folder or .zip from prepare_dataset.py")
    t.add_argument("--name", required=True, help="experiment name (letters, digits, _ and -)")
    t.add_argument("--sr", default="40k", choices=["40k", "48k"])
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--save-every", type=int, default=10, help="checkpoint + small model every N epochs")
    t.add_argument("--device", default="auto", choices=["auto", "cpu"])
    t.add_argument("--threads", type=int, default=None, help="CPU threads (default: all)")
    t.add_argument("--workers", type=int, default=None, help="preprocessing processes")
    t.add_argument("--loader-workers", type=int, default=None,
                   help="limit RVC's data loader processes (default 4, ~0.4 GB each) on machines with little RAM")
    t.add_argument("--cache-gpu", action="store_true", help="keep the dataset in GPU memory (small datasets)")
    t.add_argument("--f0-unvoiced", default="zero", choices=["zero", "interpolate"])
    t.add_argument("--python", help="Python interpreter for the RVC scripts (default: this one)")
    e = sub.add_parser("export", help="write the trained generator as ONNX for VoxaVersa")
    e.add_argument("--rvc-dir", required=True)
    e.add_argument("--name", help="experiment name (uses the newest saved model)")
    e.add_argument("--pth", help="a specific RVC model .pth instead")
    e.add_argument("--out", required=True)
    e.add_argument("--no-check", action="store_true", help="skip the ONNX Runtime comparison")
    args = p.parse_args()
    {"setup": cmd_setup, "train": cmd_train, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
