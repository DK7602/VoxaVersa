#!/usr/bin/env python3
"""
Runs VoxaVersa's training queue (train_queue.py) on a Google Colab GPU, with everything that has to survive a
disconnect on Google Drive. Used by tools/training/VoxaVersa_Queue_Colab.ipynb (step-by-step: docs/COLAB.md);
each step is safe to repeat.

    colab_queue.py setup --work W --rvc /content/rvc --python PY     RVC + base models; RVC's logs/ (the
                                                                     experiments) linked into W; ONNX encoder
    colab_queue.py data  --work W --kit KIT.zip [--vocalset]         training data kit -> W/data; VocalSet
                                                                     downloaded and cleaned -> W/data
    colab_queue.py queue --work W --rvc ... --python ... [--vocalset] [--precision mixed|full] [--batch 8]
                                                                     writes W/queue.json for train_queue.py

The data kit (VoxaVersa_Colab_Data.zip) holds the cleaned datasets that are not public, the test recordings
and voices.json (the queue's voices with paths relative to the kit). VocalSet is public (CC BY 4.0) and is
fetched with fetch_vocalset.py and cleaned with prepare_dataset.py exactly as for the CPU queue (bit-identical).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print("[colab_queue] " + msg, flush=True)


def link_logs(rvc, work):
    """RVC keeps experiments in <rvc>/logs: point it at W/rvc_logs so checkpoints live on Drive."""
    logs, store = os.path.join(rvc, "logs"), os.path.join(work, "rvc_logs")
    os.makedirs(store, exist_ok=True)
    if os.path.islink(logs):
        if os.path.realpath(logs) == os.path.realpath(store):
            return
        os.remove(logs)
    elif os.path.isdir(logs):
        for name in os.listdir(logs):          # e.g. mute/ from setup: keep what Drive does not have yet
            dst = os.path.join(store, name)
            if not os.path.exists(dst):
                shutil.move(os.path.join(logs, name), dst)
        shutil.rmtree(logs)
    os.symlink(store, logs)
    log("RVC experiments are kept in " + store)


def cmd_setup(args):
    os.makedirs(args.work, exist_ok=True)
    subprocess.run([args.python, os.path.join(HERE, "rvc_train.py"), "setup", "--rvc-dir", args.rvc, "--sr", "40k"],
                   check=True)
    link_logs(args.rvc, args.work)
    enc = os.path.join(args.work, "contentvec-768.onnx")
    if not os.path.exists(enc):
        log("exporting the content encoder (once, ~2 minutes)")
        subprocess.run([args.python, os.path.join(HERE, "export_encoder.py"), "--model",
                        os.path.join(args.rvc, "assets", "hubert_base"), "--out", enc + ".part.onnx"], check=True)
        os.replace(enc + ".part.onnx", enc)
    log("setup complete")


def cmd_data(args):
    data = os.path.join(args.work, "data")
    marker = os.path.join(data, ".kit_extracted")
    if not os.path.exists(marker):
        if not os.path.exists(args.kit):
            sys.exit("Training data kit not found: %s\nUpload VoxaVersa_Colab_Data.zip to that place in your "
                     "Google Drive (see docs/COLAB.md)." % args.kit)
        log("unpacking the training data kit")
        os.makedirs(data, exist_ok=True)
        with zipfile.ZipFile(args.kit) as z:
            for item in z.infolist():
                name = item.filename.split("/", 1)[1] if "/" in item.filename else item.filename
                if not name or item.is_dir():
                    continue
                dst = os.path.join(data, name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(item) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
        open(marker, "w").write("ok")
    voices = json.load(open(os.path.join(data, "voices.json")))["voices"]
    for v in voices:
        if v["dataset"].startswith("datasets/") and not os.path.isdir(os.path.join(data, v["dataset"])):
            sys.exit("The data kit has no %s - get a fresh VoxaVersa_Colab_Data.zip" % v["dataset"])
    log("data kit ready: %d voices in voices.json" % len(voices))

    if args.vocalset:
        out = os.path.join(data, "datasets_vocalset")
        if os.path.exists(os.path.join(out, "report.json")):
            log("VocalSet already prepared")
            return
        raw = os.path.join(args.scratch, "vocalset")
        manifest = os.path.join(args.scratch, "vocalset.csv")
        log("downloading VocalSet (CC BY 4.0, 6 GB archive), ~5 - 20 minutes")
        archive = os.path.join(args.scratch, "VocalSet1-2.zip")
        subprocess.run([sys.executable, os.path.join(HERE, "fetch_vocalset.py"), "--out", raw, "--manifest", manifest,
                        "--zip", archive, "--workers", "4"], check=True)
        os.remove(archive)
        log("cleaning VocalSet with prepare_dataset.py, ~15 minutes")
        tmp = out + ".part"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([args.python, os.path.join(HERE, "prepare_dataset.py"), "--manifest", manifest, "--out", tmp],
                       check=True)
        shutil.rmtree(out, ignore_errors=True)
        os.replace(tmp, out)
        shutil.rmtree(raw, ignore_errors=True)
        log("VocalSet ready")


def cmd_queue(args):
    data = os.path.join(args.work, "data")
    voices = json.load(open(os.path.join(data, "voices.json")))["voices"]
    if not args.vocalset:
        voices = [v for v in voices if not v["dataset"].startswith("datasets_vocalset/")]
    if args.only:
        voices = [v for v in voices if v["name"] in args.only]
    q = {
        "rvc_dir": args.rvc, "python": args.python, "device": "cuda", "amp": args.precision == "mixed",
        "batch": args.batch, "threads": 2, "max_threads": 2, "ckpt_every": args.ckpt_every,
        "min_epochs": 100, "max_epochs": 300, "eval_every": 10, "patience": 3,
        "encoder": os.path.join(args.work, "contentvec-768.onnx"), "data_root": data,
        "install": False, "cleanup_finished": True, "ecapa_dir": "/content/ecapa_model", "voices": voices,
    }
    path = os.path.join(args.work, "queue.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(q, f, indent=1)
    os.replace(tmp, path)
    log("queue.json: %d voices, batch %d, %s precision -> %s" % (len(voices), args.batch, args.precision, path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--work", required=True)
    s.add_argument("--rvc", required=True)
    s.add_argument("--python", required=True)
    d = sub.add_parser("data")
    d.add_argument("--work", required=True)
    d.add_argument("--kit", required=True)
    d.add_argument("--vocalset", action="store_true")
    d.add_argument("--python", default=sys.executable, help="interpreter with numpy/scipy/soundfile (prepare_dataset.py)")
    d.add_argument("--scratch", default="/content/scratch", help="local (not Drive) folder for the raw download")
    q = sub.add_parser("queue")
    q.add_argument("--work", required=True)
    q.add_argument("--rvc", required=True)
    q.add_argument("--python", required=True)
    q.add_argument("--vocalset", action="store_true")
    q.add_argument("--precision", default="mixed", choices=["mixed", "full"])
    q.add_argument("--batch", type=int, default=8)
    q.add_argument("--ckpt-every", type=int, default=5)
    q.add_argument("--only", nargs="*", help="train only these voices (names from voices.json)")
    args = p.parse_args()
    {"setup": cmd_setup, "data": cmd_data, "queue": cmd_queue}[args.cmd](args)


if __name__ == "__main__":
    main()
