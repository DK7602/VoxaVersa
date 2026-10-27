#!/usr/bin/env python3
"""
Runs VoxaVersa's training queue (train_queue.py) on a Google Colab GPU, with everything that has to survive a
disconnect on Google Drive. Used by tools/training/VoxaVersa_Queue_Colab.ipynb (step-by-step: docs/COLAB.md);
each step is safe to repeat.

    colab_queue.py setup --work W --rvc /content/rvc --python PY     RVC + base models; RVC's logs/ (the
                                                                     experiments) linked into W; ONNX encoder
    colab_queue.py data  --work W --kit KIT.zip [--vocalset] [--military]
                                                                     training data kit -> W/data; VocalSet
                                                                     downloaded and cleaned -> W/data; the
                                                                     military-band singers (military_test.py)
    colab_queue.py queue --work W --rvc ... --python ... [--vocalset] [--precision mixed|full] [--batch 8]
                                                                     writes W/queue.json for train_queue.py

The data kit (VoxaVersa_Colab_Data.zip) holds the cleaned datasets that are not public, the test recordings
and voices.json (the queue's voices with paths relative to the kit). VocalSet is public (CC BY 4.0) and is
fetched with fetch_vocalset.py and cleaned with prepare_dataset.py exactly as for the CPU queue (bit-identical).
"""

import argparse
import glob
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
    pitch = os.path.join(args.work, "fcpe.onnx")
    if not os.path.exists(pitch):
        log("exporting the AI pitch tracker (once, ~1 minute)")
        p = subprocess.run([args.python, os.path.join(HERE, "export_pitch.py"), "--out", pitch + ".part.onnx"])
        if p.returncode == 0:
            os.replace(pitch + ".part.onnx", pitch)
        else:
            log("the AI pitch tracker could not be exported (training is not affected); VoxaVersa works without it")
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
        if os.path.exists(os.path.join(data, "datasets_vocalset", ".ready")):
            log("VocalSet already prepared")
        elif vocalset_job_running(args.work):
            log("VocalSet is still being prepared in the background (progress: step 6)")
        else:
            # In the background, so training starts at once with the singers in the kit; the queue picks the
            # VocalSet singers up when they are ready (and a failure here never blocks training).
            os.makedirs(os.path.join(args.work, "colab_logs"), exist_ok=True)
            logf = open(os.path.join(args.work, "colab_logs", "vocalset.log"), "a")
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "vocalset", "--work", args.work,
                                  "--python", args.python, "--scratch", args.scratch],
                                 stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            with open(os.path.join(args.work, "colab_logs", "vocalset.pid"), "w") as f:
                f.write(str(p.pid))
            set_vocalset_status(args.work, "starting the download")
            log("VocalSet (20 free singers) is being downloaded and cleaned in the background, ~30 - 60 minutes; "
                "training starts now with the singers in the kit")

    if args.military:
        if os.path.exists(os.path.join(args.work, "military", ".pick_done")):
            log("military-band singers already prepared (report: queue/military/report.md)")
        elif job_running(args.work, "military"):
            log("military-band singers are still being prepared in the background (progress: step 6)")
        else:
            start_job(args, "military", ["--rvc", args.rvc])
            log("military-band singers (public-domain US Air Force and Navy band recordings) are being downloaded, "
                "separated from the band and sorted by singer in the background, ~1 - 2 hours; the best male and "
                "female singer then go to the front of the training queue")


def start_job(args, name, extra=()):
    """Runs `colab_queue.py <name>` in the background, logging to colab_logs/<name>.log."""
    os.makedirs(os.path.join(args.work, "colab_logs"), exist_ok=True)
    logf = open(os.path.join(args.work, "colab_logs", name + ".log"), "a")
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), name, "--work", args.work, "--python", args.python,
                          "--scratch", args.scratch] + list(extra),
                         stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    with open(os.path.join(args.work, "colab_logs", name + ".pid"), "w") as f:
        f.write(str(p.pid))


def job_running(work, name):
    try:
        pid = int(open(os.path.join(work, "colab_logs", name + ".pid")).read().strip())
        os.kill(pid, 0)
        return b"colab_queue.py" in open("/proc/%d/cmdline" % pid, "rb").read()
    except (OSError, ValueError):
        return False


def cmd_military(args):
    """Background job: the military-band singer test (military_test.py), then its singers join the queue."""
    p = subprocess.run([sys.executable, os.path.join(HERE, "military_test.py"), "run", "--work", args.work,
                        "--rvc", args.rvc, "--scratch", args.scratch])
    if p.returncode != 0:
        sys.exit(p.returncode)
    qargs = load(os.path.join(args.work, "colab_queue_args.json"))
    if qargs:
        write_queue(argparse.Namespace(**qargs))
        subprocess.run([qargs["python"], os.path.join(HERE, "train_queue.py"), "resume", "--work", args.work])


def vocalset_job_running(work):
    return job_running(work, "vocalset")


def set_vocalset_status(work, text):
    """One line shown in the training status (QUEUE_STATUS.md / notebook step 6)."""
    path = os.path.join(work, "colab_vocalset_status.txt")
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def cmd_vocalset(args):
    """Background job: download + clean VocalSet, then add its singers to queue.json."""
    import threading
    import traceback
    work, data = args.work, os.path.join(args.work, "data")
    out = os.path.join(data, "datasets_vocalset")
    raw = os.path.join(args.scratch, "vocalset")
    local = os.path.join(args.scratch, "datasets_vocalset")
    manifest = os.path.join(args.scratch, "vocalset.csv")
    stage = {"text": "downloading"}
    done = threading.Event()

    def step(cmd):
        """Runs a tool, copying its output to this job's log; fails with the tool's last message."""
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        last = []
        for line in p.stdout:
            print(line, end="", flush=True)
            if line.strip():
                last = (last + [line.strip()])[-3:]
        if p.wait() != 0:
            raise RuntimeError(" | ".join(last) or "exit code %d" % p.returncode)

    def progress():
        while not done.wait(30):
            n = len(glob.glob(os.path.join(raw, "*", "*", "*", "*.wav")))
            c = len(glob.glob(os.path.join(local, "*", "*.wav")))
            part = os.path.join(args.scratch, "VocalSet1-2.zip.part")
            if stage["text"] == "downloading" and os.path.exists(part):
                detail = "%.1f of 6.0 GB" % (os.path.getsize(part) / 1e9)
            elif stage["text"] == "downloading":
                detail = "%d of ~2900 recordings unpacked" % n
            else:
                detail = "%d clips so far" % c
            set_vocalset_status(work, "%s (%s)" % (stage["text"], detail))

    threading.Thread(target=progress, daemon=True).start()
    try:
        log("downloading VocalSet")
        archive = os.path.join(args.scratch, "VocalSet1-2.zip")
        step([sys.executable, os.path.join(HERE, "fetch_vocalset.py"), "--out", raw, "--manifest", manifest,
              "--zip", archive, "--workers", "4"] + (["--singers"] + args.singers if args.singers else []))
        if os.path.exists(archive):
            os.remove(archive)
        stage["text"] = "cleaning"
        log("cleaning VocalSet")
        shutil.rmtree(local, ignore_errors=True)
        step([args.python, os.path.join(HERE, "prepare_dataset.py"), "--manifest", manifest, "--out", local])
        stage["text"] = "copying to Google Drive"
        log("copying to Google Drive")
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(local, out)
        open(os.path.join(out, ".ready"), "w").write("ok")
        done.set()
        shutil.rmtree(raw, ignore_errors=True)
        qargs = load(os.path.join(work, "colab_queue_args.json"))
        if qargs:
            write_queue(argparse.Namespace(**qargs))
            subprocess.run([qargs["python"], os.path.join(HERE, "train_queue.py"), "resume", "--work", work])
        set_vocalset_status(work, "ready - its singers are in the queue")
        log("VocalSet ready")
    except BaseException as e:
        done.set()
        traceback.print_exc()
        set_vocalset_status(work, "failed (%s) - training continues with the other singers; details in "
                                  "My Drive/VoxaVersa/queue/colab_logs/vocalset.log" % str(e).strip()[-160:])
        sys.exit(1)


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def cmd_queue(args):
    with open(os.path.join(args.work, "colab_queue_args.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "cmd"}, f)
    write_queue(args)
    # voices/voice_assignment.json for the singers finished so far (current names, built-in voices)
    p = subprocess.run([args.python, os.path.join(HERE, "train_queue.py"), "assign", "--work", args.work],
                       capture_output=True, text=True)
    log((p.stdout.strip().splitlines() or ["voice assignment: nothing to do"])[-1] if p.returncode == 0
        else "voice assignment not updated: " + (p.stderr or p.stdout).strip()[-300:])


def alternate_genders(pending, started, state):
    """Male and female singers take turns (the gender with fewer finished singers first), so both sides of
    the built-in voices get more trained singers as the queue goes on, instead of all of one gender first."""
    finished = lambda g: sum(1 for v in started if v.get("gender") == g and state.get(v["name"], {}).get("status") == "done")
    by_gender = {g: [v for v in pending if v.get("gender") == g] for g in ("male", "female")}
    order = ["male", "female"] if finished("male") <= finished("female") else ["female", "male"]
    result = []
    while by_gender["male"] or by_gender["female"]:
        for g in order:
            if by_gender[g]:
                result.append(by_gender[g].pop(0))
    return result + [v for v in pending if v.get("gender") not in ("male", "female")]


def presentation(v):
    """How a singer appears in VoxaVersa. VocalSet singers keep their names (the dataset's CC BY 4.0 credit);
    the data kit's own singers are named after their voice only: "Studio Female A", "Studio Female Blend"."""
    if v["dataset"].startswith("datasets_vocalset/"):
        return v
    gender = v.get("gender", "")
    tokens = v["name"].split("_")
    rest = tokens[tokens.index(gender) + 1:] if gender in tokens else tokens[-1:]
    label = " ".join(t.upper() if len(t) == 1 else t.capitalize() for t in rest)
    return dict(v, display_name=" ".join(x for x in ("Studio", gender.capitalize(), label) if x),
                description="RVC v2 voice trained on dry studio vocals",
                license="Trained from licence-free studio vocals in the owner's library")


def write_queue(args):
    data = os.path.join(args.work, "data")
    voices = json.load(open(os.path.join(data, "voices.json")))["voices"]
    vocalset_ready = os.path.exists(os.path.join(data, "datasets_vocalset", ".ready"))

    def available(v):
        if v["dataset"].startswith("datasets_vocalset/"):
            return args.vocalset and vocalset_ready and os.path.isdir(os.path.join(data, v["dataset"]))
        return os.path.isdir(os.path.join(data, v["dataset"]))

    voices = [presentation(v) for v in voices if available(v)]
    # the military-band singers (military_test.py), once prepared: a test, so they go before the other new singers
    military = [v for v in (load(os.path.join(data, "voices_military.json")) or {}).get("voices", [])
                if os.path.isdir(os.path.join(data, v["dataset"]))]
    voices += military
    if args.only:
        voices = [v for v in voices if v["name"] in args.only]
    # voices already started keep going first, so adding singers later never interrupts one
    state = load(os.path.join(args.work, "queue_state.json")) or {}
    started = [v for v in voices if state.get(v["name"], {}).get("status") not in (None, "pending")]
    first = [v for v in voices if v in military and v not in started]
    voices = started + first + alternate_genders([v for v in voices if v not in started and v not in first], started, state)
    q = {
        "rvc_dir": args.rvc, "python": args.python, "device": "cuda", "amp": args.precision == "mixed",
        "batch": args.batch, "threads": 2, "max_threads": 2, "ckpt_every": args.ckpt_every,
        "min_epochs": 100, "max_epochs": 300, "eval_every": 10, "patience": 3,
        "encoder": os.path.join(args.work, "contentvec-768.onnx"), "data_root": data,
        "install": False, "cleanup_finished": True, "ecapa_dir": "/content/ecapa_model",
        "status_extra": [os.path.join(args.work, "colab_vocalset_status.txt"),
                         os.path.join(args.work, "colab_military_status.txt")], "voices": voices,
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
    d.add_argument("--military", action="store_true", help="also the military-band singer test (military_test.py)")
    d.add_argument("--rvc", default="/content/rvc")
    d.add_argument("--python", default=sys.executable, help="interpreter with numpy/scipy/soundfile (prepare_dataset.py)")
    d.add_argument("--scratch", default="/content/scratch", help="local (not Drive) folder for the raw download")
    vs = sub.add_parser("vocalset", help="(internal) background VocalSet download + cleaning")
    vs.add_argument("--work", required=True)
    vs.add_argument("--python", default=sys.executable)
    vs.add_argument("--scratch", default="/content/scratch")
    vs.add_argument("--singers", nargs="*", help="(testing) only these VocalSet singers")
    mj = sub.add_parser("military", help="(internal) background military-band singer test")
    mj.add_argument("--work", required=True)
    mj.add_argument("--python", default=sys.executable)
    mj.add_argument("--scratch", default="/content/scratch")
    mj.add_argument("--rvc", default="/content/rvc")
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
    {"setup": cmd_setup, "data": cmd_data, "queue": cmd_queue, "vocalset": cmd_vocalset, "military": cmd_military}[args.cmd](args)


if __name__ == "__main__":
    main()
