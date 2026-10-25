#!/usr/bin/env python3
"""
Trains a queue of VoxaVersa voices one after another on a CPU (or GPU), unattended and restart-proof.

    python3 tools/training/train_queue.py resume --work WORK      # start, or do nothing if already running
    python3 tools/training/train_queue.py status --work WORK
    python3 tools/training/train_queue.py stop   --work WORK      # checkpoint and stop

WORK/queue.json describes the voices (see `example_queue()` and docs/TRAINING.md); everything the queue
produces stays in WORK: experiments (in the RVC checkout's logs/), queue_state.json, QUEUE_STATUS.md
(heartbeat: voice, epoch, s/step, ETA, evaluations), queue_logs/, voices/<voice>/ (ONNX, .vxpack, report),
listen/<voice>/ (before/after MP3s).

`resume` is idempotent: if the supervisor is alive it only prints the status; otherwise it starts it
detached. After a crash, OOM kill or container restart the current voice continues from its last
checkpoint (written atomically after every epoch, and on `stop`).

Per voice:
  1. prepare  rvc_train.py train --prepare-only (RVC slicing, RMVPE f0 with unvoiced = 0, ContentVec
              features); ~4 % of the pieces are held out for validation
  2. train    the RVC v2 training step (same models, losses and hyper-parameters as RVC's train.py,
              fine-tuned from the pretrained 40k base), in one process with: the discriminator in bf16
              autocast on CPUs with AMX/AVX512-BF16 (the generator stays fp32: bf16 there raised the mel
              loss by ~19 % at identical weights), 2 threads, or 4 while the rest of the machine is idle,
              a memory cap (RLIMIT_AS; an allocation failure halves the batch instead of OOM-killing the box),
              oom_score_adj 1000 (if the machine runs out of memory anyway, the kernel kills this resumable
              process first), and a lock per voice so two trainers can never write the same checkpoints
  3. evaluate every `eval_every` epochs (the trainer saves and exits, the supervisor evaluates and then
              resumes training, so the two are never in memory together): validation mel distance through
              the inference path (held-out pieces, fixed noise) and speaker similarity (SpeechBrain ECAPA)
              of conversions of other singers' recordings to the voice's own recordings
  4. stop     at `max_epochs`, or after `min_epochs` once neither the validation mel distance (by more than
              0.5 %) nor the speaker similarity (by more than 0.005) improved for `patience` evaluations.
              The kept checkpoint: lowest validation mel distance among the evaluations within 0.01 of the
              best similarity.
  5. finalize ONNX export (export_onnx.py), .vxpack (package_voice.py), install into VoxaVersa's user
              folder, renders through VoxaVersa (VoxaVersaRender, if configured) and before/after MP3s with
              measured similarity and pitch accuracy.

On a GPU (Google Colab: tools/training/VoxaVersa_Queue_Colab.ipynb, docs/COLAB.md) set "device": "cuda" in
queue.json: same steps, stop rule and checkpoint choice; training uses RVC's standard fp16 mixed precision
("amp": false for fp32), the features are extracted in full precision as on the CPU, and without
VoxaVersaRender the listening files are made with try_voice.py (the same ONNX path as the plug-in's offline
engine). Relative "dataset" / "tests" paths are resolved against "data_root" (default: the work folder).

Requirements: the venv used for rvc_train.py plus speechbrain (evaluation); ffmpeg for MP3s.
"""

import argparse
import glob
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

DEFAULTS = dict(min_epochs=100, max_epochs=300, eval_every=10, patience=3, batch=2, threads=2, max_threads=4,
                mem_limit_gb=8.0, d_bf16=True, holdout=0.04, amp=True, ckpt_every=1)
# On a GPU ("device": "cuda"): no memory cap / thread governor / bf16; "amp" = RVC's standard fp16 mixed precision
# (GradScaler, losses in fp32); "ckpt_every" = full G/D checkpoint every N epochs (always before an evaluation
# and on stop) - on Google Drive each one is ~1 GB to write.

EXIT_DONE, EXIT_MEMORY, EXIT_STOPPED, EXIT_EVAL, EXIT_BUSY = 0, 3, 4, 5, 6

CHILD_ENV = {
    # glibc keeps freed tensor memory when shapes change from step to step; this keeps RSS at ~4 GB (batch 2)
    "MALLOC_MMAP_THRESHOLD_": "131072", "MALLOC_TRIM_THRESHOLD_": "131072", "MALLOC_ARENA_MAX": "2",
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
    "OMP_WAIT_POLICY": "PASSIVE",   # idle OpenMP threads sleep instead of spinning on a shared machine
}


def take_lock(path):
    """An exclusive lock held until the process exits (also when it is killed); None if someone has it."""
    import errno
    import fcntl
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno not in (errno.EAGAIN, errno.EACCES):   # e.g. Google Drive: no locks - carry on without
            return f
        f.close()
        return None
    return f


def voice_lock(w, cfg, name):
    exp = w.exp_dir(cfg, name)
    os.makedirs(exp, exist_ok=True)
    lock = take_lock(os.path.join(exp, "voxaversa.lock"))
    if lock is None:
        print("[%s] another process is working on %s - exiting" % (now(), name), flush=True)
        sys.exit(EXIT_BUSY)
    return lock


def be_first_oom_victim():
    """If the machine runs out of memory, the kernel should kill this (resumable) process, not other work."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def on_gpu(cfg):
    return str(cfg.get("device", "cpu")).lower() in ("cuda", "gpu")


def limit_memory(cfg, v):
    """CPU only: a CUDA process reserves far more address space than it uses, so RLIMIT_AS would break it."""
    if on_gpu(cfg):
        return
    import resource
    limit = int(v["mem_limit_gb"] * 1e9)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


class Work:
    def __init__(self, work, queue=None):
        self.dir = os.path.abspath(work)
        self.queue_path = os.path.abspath(queue) if queue else os.path.join(self.dir, "queue.json")
        self.state_path = os.path.join(self.dir, "queue_state.json")
        self.status_md = os.path.join(self.dir, "QUEUE_STATUS.md")
        self.live_path = os.path.join(self.dir, "queue_live.json")
        self.pid_path = os.path.join(self.dir, "queue.pid")
        self.logs = os.path.join(self.dir, "queue_logs")
        os.makedirs(self.logs, exist_ok=True)

    def cfg(self):
        cfg = load_json(self.queue_path)
        if cfg is None:
            sys.exit("missing or invalid %s (see example_queue() in %s)" % (self.queue_path, __file__))
        root = os.path.abspath(cfg.get("data_root", self.dir))
        for v in cfg["voices"]:
            for k, d in DEFAULTS.items():
                v.setdefault(k, cfg.get(k, d))
            v["dataset"] = os.path.join(root, v["dataset"])
            v["tests"] = [[os.path.join(root, t[0]), t[1]] for t in v.get("tests", [])]
        return cfg

    def voice(self, name):
        for v in self.cfg()["voices"]:
            if v["name"] == name:
                return v
        sys.exit("unknown voice " + name)

    def state(self):
        return load_json(self.state_path, {})

    def update_state(self, name, **kw):
        st = self.state()
        st.setdefault(name, {}).update(kw)
        save_json(self.state_path, st)
        return st

    def exp_dir(self, cfg, name):
        return os.path.join(cfg["rvc_dir"], "logs", name)


# ------------------------------------------------------------------------------------------- status

def fmt_h(seconds):
    if seconds is None or seconds != seconds:
        return "?"
    h = seconds / 3600
    return "%.1f h" % h if h < 48 else "%.1f days" % (h / 24)


def write_status(w, note=None):
    cfg, st = w.cfg(), w.state()
    live = load_json(w.live_path, {}) or {}
    lines = ["# Voice training queue", "", "Updated %s UTC%s" % (now(), (" - " + note) if note else ""), ""]
    if live and live.get("voice"):
        v = live
        lines += ["## Now: %s" % v["voice"], "",
                  "| | |", "|---|---|",
                  "| Epoch | %s of min %s / max %s (step %s/%s in this epoch) |" % (v.get("epoch"), v.get("min_epochs"), v.get("max_epochs"), v.get("step"), v.get("steps_per_epoch")),
                  ("| Speed | %.2f s/step, batch %s, %s |" % (v.get("s_per_step", 0), v.get("batch"), v["device"])
                   if str(v.get("device", "CPU")).startswith("GPU") else
                   "| Speed | %.2f s/step, batch %s, %s threads, D in %s |" % (v.get("s_per_step", 0), v.get("batch"), v.get("threads"), "bf16" if v.get("d_bf16") else "fp32")),
                  "| ETA | min epochs in %s, max epochs in %s |" % (fmt_h(v.get("eta_min_s")), fmt_h(v.get("eta_max_s"))),
                  "| Memory | %.1f GB RSS%s |" % (v.get("rss_gb", 0), (" (cap %.1f GB address space)" % v["mem_limit_gb"]) if v.get("mem_limit_gb") else ""),
                  "| Last losses | %s |" % v.get("losses", "-"),
                  "| Heartbeat | %s UTC (pid %s) |" % (v.get("time"), v.get("pid")), ""]
    lines += ["## Queue", "", "| # | Voice | Status | Epoch | Best eval (epoch: val mel, similarity) | Last eval |", "|---|---|---|---|---|---|"]
    for i, v in enumerate(cfg["voices"]):
        s = st.get(v["name"], {})
        evals = s.get("evals", [])
        best = s.get("best")
        last = evals[-1] if evals else None
        lines.append("| %d | %s | %s | %s | %s | %s |" % (
            i + 1, v["name"], s.get("status", "pending"), s.get("epoch_done", "-"),
            ("e%d: %.3f, %.3f" % (best["epoch"], best["val_mel"], best["sim"])) if best else "-",
            ("e%d: %.3f, %.3f" % (last["epoch"], last["val_mel"], last["sim"])) if last else "-"))
    done = [v["name"] for v in cfg["voices"] if st.get(v["name"], {}).get("status") == "done"]
    if done:
        lines += ["", "## Finished voices", ""]
        for name in done:
            s = st[name]
            lines.append("* **%s**: %s - %s" % (name, s.get("final_summary", ""), s.get("pack", "")))
    lines += ["", "Resume (idempotent): `%s %s resume --work %s`" % (sys.executable, os.path.abspath(__file__), w.dir)]
    tmp = w.status_md + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, w.status_md)


# --------------------------------------------------------------------------------------- supervisor

def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return b"train_queue.py" in f.read()
    except OSError:
        return True


def running_pid(w):
    try:
        pid = int(open(w.pid_path).read().strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def cmd_resume(args):
    w = Work(args.work, args.queue)
    pid = running_pid(w)
    if pid:
        print("queue already running (pid %d) - status: %s" % (pid, w.status_md))
        return
    w.cfg()   # validate
    log = open(os.path.join(w.logs, "supervisor.log"), "a")
    cmd = [sys.executable, os.path.abspath(__file__), "run", "--work", w.dir]
    if args.queue:
        cmd += ["--queue", w.queue_path]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print("started queue supervisor (pid %d); status: %s" % (p.pid, w.status_md))


def cmd_stop(args):
    w = Work(args.work, args.queue)
    pid = running_pid(w)
    if not pid:
        print("queue is not running")
        return
    os.kill(pid, signal.SIGTERM)
    print("asked the queue (pid %d) to checkpoint and stop" % pid)


def cmd_status(args):
    w = Work(args.work, args.queue)
    write_status(w)
    print(open(w.status_md).read())
    pid = running_pid(w)
    print("supervisor:", ("running, pid %d" % pid) if pid else "not running")


def sub(cfg, w, voice, cmd, *opts):
    q = ["--queue", w.queue_path] if w.queue_path != os.path.join(w.dir, "queue.json") else []
    return [cfg["python"], os.path.abspath(__file__), cmd, voice, "--work", w.dir, *q, *opts]


def cmd_run(args):
    w = Work(args.work, args.queue)
    lock = take_lock(os.path.join(w.dir, "queue.lock"))
    if lock is None:
        print("another supervisor is running")
        return
    with open(w.pid_path, "w") as f:
        f.write(str(os.getpid()))
    child = {"p": None}
    stopping = {"flag": False}
    log_files = {}

    def on_term(signum, frame):
        stopping["flag"] = True
        if child["p"] is not None and child["p"].poll() is None:
            child["p"].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    print("[%s] supervisor started (pid %d)" % (now(), os.getpid()), flush=True)
    try:
        while not stopping["flag"]:
            cfg = w.cfg()
            st = w.state()
            todo = [v for v in cfg["voices"] if st.get(v["name"], {}).get("status") not in ("done", "failed")]
            if not todo:
                write_status(w, "queue complete")
                print("[%s] queue complete" % now(), flush=True)
                break
            v = todo[0]
            name = v["name"]
            s = st.get(name, {})
            if name not in log_files:
                log_files[name] = open(os.path.join(w.logs, name + ".log"), "a")
            logf = log_files[name]

            def step(stage, *extra):
                print("[%s] %s: %s" % (now(), name, stage), flush=True)
                child["p"] = subprocess.Popen(sub(cfg, w, name, *extra), stdout=logf, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL, env=dict(os.environ, **CHILD_ENV))
                rc = child["p"].wait()
                child["p"] = None
                return rc

            def evaluate_pending():
                """Evaluates saved checkpoints that have no result yet; returns (stop?, reason)."""
                exp = w.exp_dir(cfg, name)
                st_v = w.state().get(name, {})
                have = {e["epoch"] for e in st_v.get("evals", [])}
                for small in sorted(glob.glob(os.path.join(exp, "voxaversa_ckpt", name + "_e*.pth")),
                                    key=lambda p_: int(p_.rsplit("_e", 1)[1].split(".")[0])):
                    epoch = int(small.rsplit("_e", 1)[1].split(".")[0])
                    if epoch in have or stopping["flag"]:
                        continue
                    res = load_json(small + ".eval.json")
                    if not res:
                        rc_ = step("evaluate epoch %d" % epoch, "_eval", "--pth", small, "--epoch", str(epoch))
                        res = load_json(small + ".eval.json")
                        if rc_ != 0 or not res:
                            print("[%s] %s: evaluation of epoch %d failed (rc %s)" % (now(), name, epoch, rc_), flush=True)
                            fails = w.state()[name].get("eval_failures", 0) + 1
                            w.update_state(name, eval_failures=fails)
                            if fails < 3:
                                return False, ""
                            res = None
                    if res:
                        evals = sorted([e for e in w.state()[name].get("evals", []) if e["epoch"] != epoch] + [res],
                                       key=lambda e: e["epoch"])
                        best = pick_best(evals)
                        w.update_state(name, evals=evals, best=best, eval_failures=0)
                        print("[%s] %s: e%d val mel %.4f, similarity %.3f (best e%d)" % (
                            now(), name, epoch, res["val_mel"], res["sim"], best["epoch"]), flush=True)
                        prune_small(os.path.join(exp, "voxaversa_ckpt"), name, best["epoch"], epoch)
                        write_status(w)
                    have.add(epoch)
                evals = w.state()[name].get("evals", [])
                done_epoch = w.state()[name].get("epoch_done", 0)
                if done_epoch >= v["max_epochs"]:
                    return True, "reached max_epochs %d" % v["max_epochs"]
                return should_stop(evals, v)

            if s.get("status") in (None, "pending", "preparing"):
                w.update_state(name, status="preparing")
                rc = step("prepare", "_prepare")
                if rc != 0:
                    fails = w.update_state(name, prepare_failures=s.get("prepare_failures", 0) + 1)[name]["prepare_failures"]
                    if fails >= 3:
                        w.update_state(name, status="failed", error="prepare failed (see queue_logs/%s.log)" % name)
                    time.sleep(30)
                    continue
                w.update_state(name, status="training")
            if w.state()[name].get("status") == "training":
                stop_now, why = evaluate_pending()
                if stopping["flag"]:
                    break
                if stop_now:
                    w.update_state(name, status="finalizing", stop_reason=why, crashes=0)
                    print("[%s] %s: training finished (%s)" % (now(), name, why), flush=True)
                    continue
                rc = step("train", "_train")
                if stopping["flag"] or rc == EXIT_STOPPED:
                    break
                if rc == EXIT_EVAL:
                    w.update_state(name, crashes=0)
                    continue
                if rc == EXIT_MEMORY or rc in (-9, 137):
                    s = w.state()[name]
                    ooms = s.get("ooms", 0) + 1
                    batch = s.get("batch", v["batch"])
                    if rc == EXIT_MEMORY or ooms >= 2:
                        batch = max(1, batch // 2)
                        ooms = 0
                    w.update_state(name, ooms=ooms, batch=batch)
                    print("[%s] %s: out of memory (rc %s) - batch %d" % (now(), name, rc, batch), flush=True)
                    time.sleep(60)
                    continue
                if rc == EXIT_DONE:
                    continue      # max epochs reached: evaluate_pending() finalizes on the next round
                if rc == EXIT_BUSY:
                    time.sleep(60)    # a trainer from an earlier supervisor is still running
                    continue
                crashes = w.state()[name].get("crashes", 0) + 1
                w.update_state(name, crashes=crashes)
                print("[%s] %s: trainer exited with %s (crash %d)" % (now(), name, rc, crashes), flush=True)
                if crashes >= 6:
                    w.update_state(name, status="failed", error="trainer crashed %d times" % crashes)
                time.sleep(min(600, 30 * crashes))
                continue
            if w.state()[name].get("status") == "finalizing":
                rc = step("finalize", "_finalize")
                if stopping["flag"]:
                    break
                if rc == 0:
                    w.update_state(name, status="done", finished=now())
                else:
                    fails = w.state()[name].get("finalize_failures", 0) + 1
                    w.update_state(name, finalize_failures=fails)
                    if fails >= 3:
                        w.update_state(name, status="failed", error="finalize failed (see queue_logs/%s.log)" % name)
                    time.sleep(30)
            save_json(w.live_path, {})
            write_status(w)
    finally:
        write_status(w, "supervisor stopped" if stopping["flag"] else None)
        try:
            if int(open(w.pid_path).read().strip()) == os.getpid():
                os.remove(w.pid_path)
        except (OSError, ValueError):
            pass
        print("[%s] supervisor exit" % now(), flush=True)


# ------------------------------------------------------------------------------------------ prepare

def rvc_imports(rvc):
    """Puts the RVC checkout on the path and quiets its DEBUG logging."""
    import logging
    if rvc not in sys.path:
        sys.path.insert(0, rvc)
    os.chdir(rvc)
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("numba", "matplotlib", "PIL", "speechbrain", "torch"):
        logging.getLogger(name).setLevel(logging.WARNING)


def cmd_prepare(args):
    be_first_oom_victim()
    w = Work(args.work, args.queue)
    cfg, v = w.cfg(), w.voice(args.voice)
    lock = voice_lock(w, cfg, v["name"])
    exp = w.exp_dir(cfg, v["name"])
    if not os.path.exists(os.path.join(exp, "filelist.txt")):
        gpu = on_gpu(cfg)
        rc = subprocess.call([cfg["python"], os.path.join(HERE, "rvc_train.py"), "train", "--rvc-dir", cfg["rvc_dir"],
                              "--dataset", v["dataset"], "--name", v["name"], "--device", "auto" if gpu else "cpu",
                              "--threads", str(v["threads"]), "--workers", "2", "--prepare-only"]
                             + (["--full-precision"] if gpu else []))
        if rc != 0:
            sys.exit(rc)
    split_holdout(exp, v["holdout"])
    prepare_eval(cfg, w, v, exp)


def split_holdout(exp, fraction):
    train_l, val_l = os.path.join(exp, "filelist_train.txt"), os.path.join(exp, "filelist_val.txt")
    if os.path.exists(train_l) and os.path.exists(val_l):
        return
    lines = [l for l in open(os.path.join(exp, "filelist.txt"), encoding="utf-8").read().splitlines() if l.strip()]
    real = sorted(l for l in lines if "/logs/mute/" not in l)
    mute = [l for l in lines if "/logs/mute/" in l]
    n_val = max(4, min(12, int(round(len(real) * fraction))))
    step = max(1, len(real) // n_val)
    val = real[step // 2::step][:n_val]
    vs = set(val)
    train = [l for l in real if l not in vs] + mute
    random.Random(1234).shuffle(train)
    with open(train_l, "w", encoding="utf-8") as f:
        f.write("\n".join(train))
    with open(val_l, "w", encoding="utf-8") as f:
        f.write("\n".join(val))


def prepare_eval(cfg, w, v, exp):
    """Caches what the evaluations need: ContentVec features + RMVPE f0 of the test recordings, and the
    speaker-embedding centroid of the voice's own clips."""
    out = os.path.join(exp, "voxaversa_eval")
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, "ready.json")
    if os.path.exists(done):
        return
    import numpy as np
    import torch
    torch.set_num_threads(v["threads"])
    rvc = cfg["rvc_dir"]
    rvc_imports(rvc)
    from infer.hubert import load_hubert_model, extract_hubert_features
    from infer.rmvpe import RMVPE
    hubert = load_hubert_model("cpu", False)
    rmvpe = RMVPE(os.path.join(rvc, "assets/rmvpe/rmvpe.pt"), is_half=False, device="cpu")
    tests = []
    for i, (path, transpose) in enumerate(v["tests"]):
        x16 = load_audio(path, 16000)[: 16000 * 30]
        with torch.no_grad():
            feats = extract_hubert_features(hubert, torch.from_numpy(x16)[None], "v2").squeeze(0).numpy()
        f0 = np.asarray(rmvpe.infer_from_audio(x16, thred=0.03), dtype=np.float32)
        np.savez(os.path.join(out, "test_%d.npz" % i), feats=feats.astype(np.float32), f0=f0, transpose=float(transpose))
        tests.append(dict(path=path, transpose=transpose))
    ecapa = load_ecapa(w)
    clips = sorted(glob.glob(os.path.join(v["dataset"], "*.wav")))
    random.Random(0).shuffle(clips)
    cent = np.mean([embed(ecapa, load_audio(c, 16000)) for c in clips[:40]], 0)
    np.save(os.path.join(out, "target_centroid.npy"), cent / np.linalg.norm(cent))
    save_json(done, dict(tests=tests, clips=len(clips)))


def load_audio(path, sr):
    import numpy as np
    import soundfile as sf
    from scipy import signal as sps
    x, s = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if s != sr:
        g = math.gcd(s, sr)
        x = sps.resample_poly(x, sr // g, s // g).astype(np.float32)
    return x


def load_ecapa(w):
    from speechbrain.inference.speaker import EncoderClassifier
    # "ecapa_dir": SpeechBrain links its files into this folder, which Google Drive can't hold (Colab: local disk)
    savedir = os.path.expanduser(w.cfg().get("ecapa_dir") or os.path.join(w.dir, "ecapa_model"))
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir,
                                          run_opts={"device": "cpu"})


def embed(model, x16, chunk=6.0):
    """Mean ECAPA embedding over the non-silent 6 s chunks of a 16 kHz signal (unit length)."""
    import numpy as np
    import torch
    n = int(chunk * 16000)
    es = []
    for i in range(0, max(1, len(x16) - n // 2), n):
        part = x16[i:i + n]
        if len(part) < 16000 or float(np.sqrt(np.mean(part ** 2))) < 0.003:
            continue
        with torch.no_grad():
            e = model.encode_batch(torch.from_numpy(part)[None]).squeeze().numpy()
        es.append(e / np.linalg.norm(e))
    if not es:
        with torch.no_grad():
            e = model.encode_batch(torch.from_numpy(x16)[None]).squeeze().numpy()
        es = [e / np.linalg.norm(e)]
    m = np.mean(es, 0)
    return m / np.linalg.norm(m)


# -------------------------------------------------------------------------------------------- train

def rss_gb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        return 0.0


class ThreadGovernor:
    """`low` threads normally, `high` while the other processes leave enough cores idle: measures the CPU time
    used by everything else since the last call (/proc/stat minus this process), so it reacts within a
    minute and is not fooled by its own load."""

    def __init__(self, low, high):
        self.low, self.high, self.prev = low, high, None
        self.tick = os.sysconf("SC_CLK_TCK")

    def _sample(self):
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:9]]
        with open("/proc/self/stat") as f:
            p = f.read().rsplit(")", 1)[1].split()
        return sum(v) - v[3] - v[4], int(p[11]) + int(p[12]), time.time()

    def choose(self, current):
        try:
            s = self._sample()
        except (OSError, ValueError, IndexError):
            return current
        if self.prev is None:
            self.prev = s
            return current
        (busy, own, t), (pb, po, pt) = s, self.prev
        if t - pt < 20:
            return current
        self.prev = s
        others = max(0.0, ((busy - pb) - (own - po)) / self.tick / (t - pt))     # cores used by others
        free = (os.cpu_count() or 4) - others
        if free >= self.high:
            return self.high
        if free < self.high - 0.7:
            return self.low
        return current


def save_small(net_g, hps, path, epoch):
    """Inference model in RVC's format (process_ckpt.savee), loadable by export_onnx.py and the WebUI."""
    import torch
    from collections import OrderedDict
    opt = OrderedDict()
    opt["weight"] = {k: v.detach().half().cpu() for k, v in net_g.state_dict().items() if "enc_q" not in k}
    m, d = hps["model"], hps["data"]
    opt["config"] = [d["filter_length"] // 2 + 1, 32, m["inter_channels"], m["hidden_channels"], m["filter_channels"],
                     m["n_heads"], m["n_layers"], m["kernel_size"], m["p_dropout"], m["resblock"],
                     m["resblock_kernel_sizes"], m["resblock_dilation_sizes"], m["upsample_rates"],
                     m["upsample_initial_channel"], m["upsample_kernel_sizes"], m["spk_embed_dim"], m["gin_channels"],
                     d["sampling_rate"]]
    opt["info"] = "%depoch" % epoch
    opt["sr"] = "%dk" % (d["sampling_rate"] // 1000)
    opt["f0"] = 1
    opt["version"] = "v2"
    torch.save(opt, path + ".tmp")
    os.replace(path + ".tmp", path)


def save_ckpt(model, optim, lr, epoch, path):
    import torch
    torch.save({"model": model.state_dict(), "iteration": epoch, "optimizer": optim.state_dict(), "learning_rate": lr},
               path + ".tmp")
    os.replace(path + ".tmp", path)


def should_stop(evals, v):
    """(stop?, reason). See the module docstring."""
    if not evals:
        return False, ""
    last = evals[-1]["epoch"]
    if last >= v["max_epochs"]:
        return True, "reached max_epochs %d" % v["max_epochs"]
    if last < v["min_epochs"] or len(evals) <= v["patience"]:
        return False, ""
    recent, before = evals[-v["patience"]:], evals[:-v["patience"]]
    best_mel = min(e["val_mel"] for e in before)
    best_sim = max(e["sim"] for e in before)
    mel_better = min(e["val_mel"] for e in recent) < best_mel * (1 - 0.005)
    sim_better = max(e["sim"] for e in recent) > best_sim + 0.005
    if not mel_better and not sim_better:
        return True, "plateau: no gain in validation mel (>0.5 %%) or similarity (>0.005) for %d evaluations" % v["patience"]
    return False, ""


def pick_best(evals):
    top = max(e["sim"] for e in evals)
    cands = [e for e in evals if e["sim"] >= top - 0.01]
    return min(cands, key=lambda e: e["val_mel"])


def cmd_train(args):
    w = Work(args.work, args.queue)
    cfg, v = w.cfg(), w.voice(args.voice)
    lock = voice_lock(w, cfg, v["name"])
    limit_memory(cfg, v)
    be_first_oom_victim()

    import numpy as np
    import torch
    import torch.nn.functional as F
    rvc = cfg["rvc_dir"]
    rvc_imports(rvc)
    from infer.module import commons
    from infer.module.models import SynthesizerTrnMs768NSFsid, MultiPeriodDiscriminatorV2
    from train.data_utils import DistributedBucketSampler, TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid
    from train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
    from train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
    from train import utils as rutils
    from types import SimpleNamespace

    name = v["name"]
    exp = w.exp_dir(cfg, name)
    st = w.state().get(name, {})
    batch = st.get("batch", v["batch"])
    hps = load_json(os.path.join(exp, "config.json"))
    ns = lambda d: SimpleNamespace(**{k: ns(x) if isinstance(x, dict) else x for k, x in d.items()})
    H = ns(hps)
    threads = v["threads"]
    torch.set_num_threads(threads)
    gpu = on_gpu(cfg)
    if gpu and not torch.cuda.is_available():
        sys.exit("queue.json says \"device\": \"cuda\" but PyTorch sees no GPU")
    dev = torch.device("cuda" if gpu else "cpu")
    amp = gpu and bool(v["amp"])
    d_bf16 = (not gpu) and v["d_bf16"] and torch.backends.mkldnn.is_available() \
        and "avx512" in str(torch.backends.cpu.get_cpu_capability()).lower()

    train_set = TextAudioLoaderMultiNSFsid(os.path.join(exp, "filelist_train.txt"), H.data)
    sampler = DistributedBucketSampler(train_set, batch, [100, 200, 300, 400, 500, 600, 700, 800, 900],
                                       num_replicas=1, rank=0, shuffle=True)
    loader = torch.utils.data.DataLoader(train_set, num_workers=2 if gpu else 1, shuffle=False, pin_memory=gpu,
                                         collate_fn=TextAudioCollateMultiNSFsid(), batch_sampler=sampler,
                                         persistent_workers=True, prefetch_factor=2)
    net_g = SynthesizerTrnMs768NSFsid(H.data.filter_length // 2 + 1, H.train.segment_size // H.data.hop_length,
                                      **hps["model"], is_half=False, sr=H.data.sampling_rate)
    net_d = MultiPeriodDiscriminatorV2(H.model.use_spectral_norm)
    net_g.to(dev)    # before the optimizers and the checkpoint: Adam's state follows its parameters' device
    net_d.to(dev)
    optim_g = torch.optim.AdamW(net_g.parameters(), H.train.learning_rate, betas=H.train.betas, eps=H.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), H.train.learning_rate, betas=H.train.betas, eps=H.train.eps)

    g_path, d_path = os.path.join(exp, "G_2333333.pth"), os.path.join(exp, "D_2333333.pth")
    if os.path.exists(g_path) and os.path.exists(d_path):
        _, _, _, ep_d = rutils.load_checkpoint(d_path, net_d, optim_d)
        _, _, _, ep_g = rutils.load_checkpoint(g_path, net_g, optim_g)
        epoch_done = min(ep_d, ep_g)
        if st.get("partial_epoch") == epoch_done:     # saved mid-epoch by `stop`: run that epoch again
            epoch_done -= 1
        print("[%s] resumed from epoch %d" % (now(), epoch_done), flush=True)
    else:
        pre = os.path.join(rvc, "assets", "pretrained_v2")
        sr_k = "%dk" % (H.data.sampling_rate // 1000)
        net_g.load_state_dict(torch.load(os.path.join(pre, "f0G%s.pth" % sr_k), map_location=dev)["model"])
        net_d.load_state_dict(torch.load(os.path.join(pre, "f0D%s.pth" % sr_k), map_location=dev)["model"])
        epoch_done = 0
        print("[%s] starting from the pretrained %s base" % (now(), sr_k), flush=True)
    w.update_state(name, epoch_done=epoch_done, partial_epoch=None)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=H.train.lr_decay, last_epoch=epoch_done - 1)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=H.train.lr_decay, last_epoch=epoch_done - 1)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda s, f: stop.__setitem__("flag", True))
    ckpt_dir = os.path.join(exp, "voxaversa_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    seg = H.train.segment_size // H.data.hop_length
    ema, t_last, beat = None, time.time(), 0.0
    governor = None if gpu else ThreadGovernor(v["threads"], v["max_threads"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    steps_per_epoch = len(loader)
    try:
        for epoch in range(epoch_done + 1, v["max_epochs"] + 1):
            sampler.set_epoch(epoch)
            net_g.train()
            net_d.train()
            for i, batch_data in enumerate(loader):
                if gpu:
                    batch_data = [t.to(dev, non_blocking=True) for t in batch_data]
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = batch_data
                # GPU: RVC's own mixed-precision recipe (train/train.py); CPU: fp32 G, optional bf16 D
                with torch.autocast("cuda", enabled=amp) if gpu else torch.autocast("cpu", enabled=False):
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                    mel = spec_to_mel_torch(spec, H.data.filter_length, H.data.n_mel_channels, H.data.sampling_rate,
                                            H.data.mel_fmin, H.data.mel_fmax)
                y_mel = commons.slice_segments(mel, ids_slice, seg)
                y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1), H.data.filter_length, H.data.n_mel_channels,
                                                  H.data.sampling_rate, H.data.hop_length, H.data.win_length,
                                                  H.data.mel_fmin, H.data.mel_fmax)
                wave = commons.slice_segments(wave, ids_slice * H.data.hop_length, H.train.segment_size)
                with (torch.autocast("cuda", enabled=amp) if gpu
                      else torch.autocast("cpu", dtype=torch.bfloat16, enabled=d_bf16)):
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc, _, _ = discriminator_loss([t.float() for t in y_d_hat_r], [t.float() for t in y_d_hat_g])
                optim_d.zero_grad()
                scaler.scale(loss_disc).backward()
                scaler.step(optim_d)
                with (torch.autocast("cuda", enabled=amp) if gpu
                      else torch.autocast("cpu", dtype=torch.bfloat16, enabled=d_bf16)):
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                loss_mel = F.l1_loss(y_mel.float(), y_hat_mel) * H.train.c_mel
                loss_kl = kl_loss(z_p.float(), logs_q.float(), m_p.float(), logs_p.float(), z_mask) * H.train.c_kl
                loss_fm = feature_loss([[t.float() for t in a] for a in fmap_r], [[t.float() for t in a] for a in fmap_g])
                loss_gen, _ = generator_loss([t.float() for t in y_d_hat_g])
                loss_all = loss_gen + loss_fm + loss_mel + loss_kl
                optim_g.zero_grad()
                scaler.scale(loss_all).backward()
                scaler.step(optim_g)
                scaler.update()

                t = time.time()
                dt, t_last = t - t_last, t
                ema = dt if ema is None else 0.9 * ema + 0.1 * dt
                if governor is not None and i % 3 == 0:
                    new = governor.choose(threads)
                    if new != threads:
                        threads = new
                        torch.set_num_threads(threads)
                if t - beat > 30 or i + 1 == steps_per_epoch:
                    beat = t
                    left_epoch = steps_per_epoch - i - 1
                    eta = lambda target: (left_epoch + max(0, target - epoch) * steps_per_epoch) * ema
                    live = dict(voice=name, epoch=epoch, step=i + 1, steps_per_epoch=steps_per_epoch, s_per_step=ema,
                                batch=batch, threads=threads, d_bf16=d_bf16, min_epochs=v["min_epochs"],
                                max_epochs=v["max_epochs"], eta_min_s=eta(v["min_epochs"]), eta_max_s=eta(v["max_epochs"]),
                                rss_gb=rss_gb(), mem_limit_gb=0.0 if gpu else v["mem_limit_gb"], time=now(), pid=os.getpid(),
                                device="GPU %s%s" % (torch.cuda.get_device_name(0), ", fp16 mixed precision" if amp else "") if gpu else "CPU",
                                losses="disc %.3f gen %.3f fm %.3f mel %.3f kl %.3f" % (
                                    float(loss_disc), float(loss_gen), float(loss_fm), float(loss_mel), float(loss_kl)))
                    save_json(w.live_path, live)
                    write_status(w)
                if stop["flag"]:
                    break
            if stop["flag"]:
                lr = optim_g.param_groups[0]["lr"]
                save_ckpt(net_d, optim_d, lr, epoch, d_path)
                save_ckpt(net_g, optim_g, lr, epoch, g_path)
                w.update_state(name, partial_epoch=epoch)
                print("[%s] stopped during epoch %d (checkpoint saved)" % (now(), epoch), flush=True)
                return EXIT_STOPPED
            sched_g.step()
            sched_d.step()
            lr = optim_g.param_groups[0]["lr"]
            evaluating = epoch % v["eval_every"] == 0 or epoch == v["max_epochs"]
            if evaluating or epoch % max(1, int(v["ckpt_every"])) == 0:
                save_ckpt(net_d, optim_d, lr, epoch, d_path)
                save_ckpt(net_g, optim_g, lr, epoch, g_path)
                w.update_state(name, epoch_done=epoch, batch=batch)
            print("[%s] epoch %d done (%.2f s/step, %d threads, mel %.3f)" % (now(), epoch, ema, threads, float(loss_mel)), flush=True)
            if evaluating:
                # the supervisor evaluates while this process is gone (never both in memory)
                save_small(net_g, hps, os.path.join(ckpt_dir, "%s_e%d.pth" % (name, epoch)), epoch)
                print("[%s] epoch %d: checkpoint for evaluation saved" % (now(), epoch), flush=True)
                return EXIT_EVAL
        return EXIT_DONE
    except (MemoryError, RuntimeError) as e:
        if isinstance(e, MemoryError) or "alloc" in str(e).lower() or "memory" in str(e).lower():
            print("[%s] out of memory: %s" % (now(), e), flush=True)
            return EXIT_MEMORY
        raise


def prune_small(ckpt_dir, name, best_epoch, last_epoch):
    keep = {best_epoch, last_epoch}
    for p in glob.glob(os.path.join(ckpt_dir, name + "_e*.pth")):
        e = int(os.path.basename(p).rsplit("_e", 1)[1].split(".")[0])
        if e not in keep:
            os.remove(p)


# --------------------------------------------------------------------------------------------- eval

def cmd_eval(args):
    w = Work(args.work, args.queue)
    cfg, v = w.cfg(), w.voice(args.voice)
    lock = voice_lock(w, cfg, v["name"])
    limit_memory(cfg, v)
    be_first_oom_victim()
    import numpy as np
    import torch
    torch.set_num_threads(v["threads"])
    rvc = cfg["rvc_dir"]
    sys.path.insert(0, HERE)
    import export_onnx
    net, info = export_onnx.load_model(args.pth, rvc)
    dev = torch.device("cuda" if on_gpu(cfg) and torch.cuda.is_available() else "cpu")
    net.to(dev)
    rvc_imports(rvc)
    from train.data_utils import TextAudioLoaderMultiNSFsid
    from train.mel_processing import mel_spectrogram_torch
    exp = w.exp_dir(cfg, v["name"])
    hps = load_json(os.path.join(exp, "config.json"))
    from types import SimpleNamespace
    ns = lambda d: SimpleNamespace(**{k: ns(x) if isinstance(x, dict) else x for k, x in d.items()})
    H = ns(hps)
    sid = torch.LongTensor([0]).to(dev)

    def mel(y):
        return mel_spectrogram_torch(y, H.data.filter_length, H.data.n_mel_channels, H.data.sampling_rate,
                                     H.data.hop_length, H.data.win_length, H.data.mel_fmin, H.data.mel_fmax)

    val = TextAudioLoaderMultiNSFsid(os.path.join(exp, "filelist_val.txt"), H.data)
    dists = []
    with torch.no_grad():
        for k in range(len(val)):
            spec, wav, phone, pitch, pitchf, _ = val[k]
            torch.manual_seed(k)
            y = net.infer(phone[None].to(dev), torch.LongTensor([phone.shape[0]]).to(dev), pitch[None].to(dev),
                          pitchf[None].to(dev), sid)[0][0].float().cpu()
            n = min(y.shape[-1], wav.shape[-1])
            dists.append(float(torch.mean(torch.abs(mel(y[:, :n]) - mel(wav[:, :n])))))
    ecapa = load_ecapa(w)
    cent = np.load(os.path.join(exp, "voxaversa_eval", "target_centroid.npy"))
    sims = []
    from scipy import signal as sps
    lo, hi = 1127 * math.log(1 + 50 / 700), 1127 * math.log(1 + 1100 / 700)
    for path in sorted(glob.glob(os.path.join(exp, "voxaversa_eval", "test_*.npz"))):
        z = np.load(path)
        feats, f0 = z["feats"], z["f0"] * 2 ** (float(z["transpose"]) / 12)
        n = min(len(f0), 2 * len(feats))
        n -= n % 2
        phone = torch.from_numpy(np.repeat(feats, 2, axis=0)[:n])[None]
        f0 = f0[:n].astype(np.float32)
        mel_f0 = 1127 * np.log(1 + f0 / 700)
        coarse = np.clip(np.rint(np.where(f0 > 0, (mel_f0 - lo) * 254 / (hi - lo) + 1, 1)), 1, 255).astype(np.int64)
        torch.manual_seed(0)
        with torch.no_grad():
            y = net.infer(phone.to(dev), torch.LongTensor([n]).to(dev), torch.from_numpy(coarse)[None].to(dev),
                          torch.from_numpy(f0)[None].to(dev), sid)[0]
        y = y.reshape(-1).float().cpu().numpy()
        g = math.gcd(info["sample_rate"], 16000)
        y16 = sps.resample_poly(y, 16000 // g, info["sample_rate"] // g).astype(np.float32)
        sims.append(float(np.dot(embed(ecapa, y16), cent)))
    res = dict(epoch=args.epoch, val_mel=float(np.mean(dists)), sim=float(np.mean(sims)), sims=[round(s, 4) for s in sims],
               time=now())
    save_json(args.pth + ".eval.json", res)
    print("eval", res, flush=True)


# ----------------------------------------------------------------------------------------- finalize

def mp3(x, sr, path):
    import numpy as np
    peak = float(np.abs(x).max()) or 1.0
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-c:a", "libmp3lame", "-q:a", "2", path], input=(x * (0.89 / peak)).astype(np.float32).tobytes(),
                   check=True)


def cmd_finalize(args):
    be_first_oom_victim()
    w = Work(args.work, args.queue)
    cfg, v = w.cfg(), w.voice(args.voice)
    lock = voice_lock(w, cfg, v["name"])
    import numpy as np
    name = v["name"]
    st = w.state()[name]
    exp = w.exp_dir(cfg, name)
    best = st.get("best") or {"epoch": st.get("epoch_done")}
    small = os.path.join(exp, "voxaversa_ckpt", "%s_e%d.pth" % (name, best["epoch"]))
    if not os.path.exists(small):
        small = os.path.join(exp, "G_2333333.pth")
    outdir = os.path.join(w.dir, "voices", name)
    os.makedirs(outdir, exist_ok=True)
    sys.path.insert(0, HERE)
    import export_onnx
    onnx_path = os.path.join(outdir, "%s_e%d.onnx" % (name, best["epoch"]))
    export_onnx.export(small, onnx_path, rvc_dir=cfg["rvc_dir"], check=True)
    pack_id = v.get("id", name.replace("_", "-"))
    pack = os.path.join(outdir, pack_id + ".vxpack")
    rng = load_json(os.path.join(os.path.dirname(v["dataset"].rstrip("/")), "report.json"), {}) or {}
    vrange = rng.get("voices", {}).get(os.path.basename(v["dataset"].rstrip("/")), {}).get("range_midi")
    cmd = [sys.executable, os.path.join(REPO, "tools", "neural", "package_voice.py"), "--model", onnx_path,
           "--name", v.get("display_name", name), "--id", pack_id, "--out", pack, "--gender", v.get("gender", "neutral"),
           "--genre", v.get("genre", "Custom"), "--styles", v.get("styles", ""),
           "--description", v.get("description", "") + " (epoch %d)" % best["epoch"], "--author", v.get("author", ""),
           "--license", v.get("license", ""), "--sample-rate", "40000", "--encoder", "contentvec-768", "--i-have-the-rights"]
    if vrange:
        cmd += ["--range", str(vrange[0]), str(vrange[1])]
    subprocess.run(cmd, check=True)
    user = os.path.expanduser(cfg.get("voxaversa_user_dir", "~/.config/VoxaVersa"))
    inst = os.path.join(user, "Voices", pack_id)
    if cfg.get("install", True):     # false (Colab): only the .vxpack, imported in VoxaVersa later
        shutil.rmtree(inst, ignore_errors=True)
        os.makedirs(inst)
        with zipfile.ZipFile(pack) as z:
            z.extractall(inst)
    if cfg.get("install", True) and cfg.get("encoder"):
        enc_dst = os.path.join(user, "Encoders", "contentvec-768.onnx")
        if not os.path.exists(enc_dst):
            os.makedirs(os.path.dirname(enc_dst), exist_ok=True)
            shutil.copy(cfg["encoder"], enc_dst)

    lines = ["# %s" % v.get("display_name", name), "", "Best checkpoint: epoch %s (%s)" % (best["epoch"], st.get("stop_reason", "")), ""]
    if st.get("evals"):
        lines += ["| Epoch | Validation mel distance | Similarity to the voice |", "|---|---|---|"]
        lines += ["| %d | %.4f | %.3f |" % (e["epoch"], e["val_mel"], e["sim"]) for e in st["evals"]]
        lines.append("")
    summary = "epoch %s" % best["epoch"]
    render = cfg.get("render")
    listen = os.path.join(w.dir, "listen", name)
    os.makedirs(listen, exist_ok=True)
    use_render = bool(render and os.path.exists(render))
    encoder = cfg.get("encoder")

    def convert(path, transpose, out):
        """The test recording through the finished voice: VoxaVersa itself when VoxaVersaRender is configured,
        else try_voice.py (the same ONNX generator + encoder path as VoxaVersa's offline engine)."""
        if use_render:
            env = dict(os.environ, VOXAVERSA_AI_THREADS=str(v["threads"]))
            p = subprocess.run([render, path, out, "--voice", "user-" + pack_id, "--set", "aiEngine=1",
                                "--set", "pitch=%d" % transpose], capture_output=True, text=True, env=env)
            speed = next((l.strip() for l in p.stdout.splitlines() if l.strip().startswith("voice ") and "real time" in l), "?")
            return p.returncode == 0 and os.path.exists(out), speed.split("|")[-1].strip(), (p.stderr or p.stdout)[-200:]
        t0 = time.time()
        p = subprocess.run([cfg["python"], os.path.join(HERE, "try_voice.py"), "--model", onnx_path, "--encoder", encoder,
                            "--input", path, "--out", out, "--transpose", str(transpose), "--rvc-dir", cfg["rvc_dir"]],
                           capture_output=True, text=True)
        return p.returncode == 0 and os.path.exists(out), "try_voice.py %.0f s" % (time.time() - t0), (p.stderr or p.stdout)[-200:]

    if use_render or (encoder and os.path.exists(encoder)):
        rvc_imports(cfg["rvc_dir"])
        import torch
        torch.set_num_threads(v["threads"])
        ecapa = load_ecapa(w)
        cent = np.load(os.path.join(exp, "voxaversa_eval", "target_centroid.npy"))
        from infer.rmvpe import RMVPE
        rmvpe = RMVPE(os.path.join(cfg["rvc_dir"], "assets/rmvpe/rmvpe.pt"), is_half=False, device="cpu")
        lines += ["| Input | Shift | Similarity to the voice (input itself) | Pitch error median | Within 50 cents | Voiced kept | Render speed |",
                  "|---|---|---|---|---|---|---|"]
        sims = []
        for i, (path, transpose) in enumerate(v["tests"]):
            base = "%d_%s" % (i + 1, os.path.splitext(os.path.basename(path))[0])
            out = os.path.join(outdir, base + "_converted.wav")
            ok, speed, err = convert(path, transpose, out)
            if not ok:
                lines.append("| %s | %+d | conversion failed: %s | | | | |" % (base, transpose, err))
                continue
            xi, xo = load_audio(path, 16000), load_audio(out, 16000)
            n = min(len(xi), len(xo))
            xi, xo = xi[:n], xo[:n]
            si, so = float(np.dot(embed(ecapa, xi), cent)), float(np.dot(embed(ecapa, xo), cent))
            sims.append(so)
            fi = np.asarray(rmvpe.infer_from_audio(xi, thred=0.03)) * 2 ** (transpose / 12)
            fo = np.asarray(rmvpe.infer_from_audio(xo, thred=0.03))
            m = min(len(fi), len(fo))
            both = (fi[:m] > 0) & (fo[:m] > 0)
            c = np.abs(1200 * np.log2(fo[:m][both] / fi[:m][both])) if both.any() else np.array([0.0])
            lines.append("| %s | %+d | %.3f (%.3f) | %.1f cents | %.0f %% | %.0f %% | %s |" % (
                base, transpose, so, si, float(np.median(c)), 100 * float(np.mean(c < 50)),
                100 * both.sum() / max(1, (fi[:m] > 0).sum()), speed))
            a = load_audio(path, 44100)[: 44100 * 20]
            b = load_audio(out, 44100)[: 44100 * 20]
            mp3(a, 44100, os.path.join(listen, base + "_1_before.mp3"))
            mp3(b, 44100, os.path.join(listen, base + "_2_after.mp3"))
            gap = np.zeros(int(0.6 * 44100), np.float32)
            mp3(np.concatenate([a / (np.abs(a).max() or 1), gap, b / (np.abs(b).max() or 1)]), 44100,
                os.path.join(listen, base + "_3_before_then_after.mp3"))
        if sims:
            summary += ", similarity of conversions %.3f" % float(np.mean(sims))
    # built-in voices: measure this singer, re-assign all finished singers, update every installed pack
    assignment = apply_factory_voices(w, cfg, name)
    mine = assignment.get("factory_voices", {}).get(name, [])
    lines += ["", "Built-in voices powered by this singer: %s (see %s)" % (", ".join(mine) or "none",
                                                                       os.path.join(w.dir, "factory_assignment.md"))]
    if mine and render and os.path.exists(render):
        path, transpose = v["tests"][0]
        out = os.path.join(outdir, "via_%s.wav" % mine[0])
        p = subprocess.run([render, path, out, "--voice", mine[0], "--set", "aiEngine=1", "--set", "pitch=%d" % transpose],
                           capture_output=True, text=True, env=dict(os.environ, VOXAVERSA_AI_THREADS=str(v["threads"])))
        if p.returncode == 0 and os.path.exists(out):
            xo = load_audio(out, 16000)
            sim = float(np.dot(embed(ecapa, xo), cent))
            ai = next((l.strip() for l in p.stdout.splitlines() if "voice model" in l), "AI model not reported")
            lines.append("Check through the built-in voice %s (its DSP colour on top): similarity %.3f; %s" % (mine[0], sim, ai))
            b = load_audio(out, 44100)[: 44100 * 20]
            mp3(b, 44100, os.path.join(listen, "%s_via_builtin_%s.mp3" % (os.path.splitext(os.path.basename(path))[0], mine[0])))
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    w.update_state(name, pack=pack, final_summary=summary, report=os.path.join(outdir, "report.md"), factory_voices=mine)
    if cfg.get("cleanup_finished"):
        # keep the .vxpack and the chosen small checkpoint, drop what only training needs (~1.8 GB per voice)
        for f in ["G_2333333.pth", "D_2333333.pth"] + glob.glob(os.path.join(exp, "voxaversa_ckpt", name + "_e*.pth")):
            if os.path.exists(os.path.join(exp, f)) and os.path.abspath(os.path.join(exp, f)) != os.path.abspath(small):
                os.remove(os.path.join(exp, f))
        for d in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
            shutil.rmtree(os.path.join(exp, d), ignore_errors=True)
        if os.path.exists(onnx_path):
            os.remove(onnx_path)     # it is inside the .vxpack
    print("finalized", name, summary, flush=True)


def apply_factory_voices(w, cfg, current):
    """Measures `current`'s timbre, assigns the built-in voices over all finished singers (+ current) and
    writes "factoryVoices" / "listed" into every installed pack and its .vxpack."""
    sys.path.insert(0, HERE)
    import factory_voices as fv
    tdir = os.path.join(w.dir, "timbre")
    os.makedirs(tdir, exist_ok=True)
    st = w.state()
    names = [v["name"] for v in cfg["voices"] if st.get(v["name"], {}).get("status") == "done" or v["name"] == current]
    singers = []
    for v in cfg["voices"]:
        if v["name"] not in names:
            continue
        tpath = os.path.join(tdir, v["name"] + ".json")
        t = load_json(tpath)
        if not t:
            t = fv.measure(v["dataset"], v["gender"])
            t["name"] = v["name"]
            save_json(tpath, t)
        t["corpus"], t["style"] = v.get("corpus"), v.get("style", "contemporary")
        singers.append(t)
    voices = fv.factory_voices()
    result, costs = fv.assign(singers, voices)
    data = fv.write_table(singers, voices, result, costs, os.path.join(w.dir, "factory_assignment"))
    user = os.path.expanduser(cfg.get("voxaversa_user_dir", "~/.config/VoxaVersa"))
    for v in cfg["voices"]:
        if v["name"] not in names:
            continue
        pack_id = v.get("id", v["name"].replace("_", "-"))
        fvs = data["factory_voices"].get(v["name"], [])
        listed = bool(v.get("listed", True)) or not fvs   # a pack that powers no built-in voice stays visible
        manifest = os.path.join(user, "Voices", pack_id, "voice.json")
        j = load_json(manifest)
        if j is not None:
            j["factoryVoices"] = fvs
            j["listed"] = listed
            save_json(manifest, j)
        pack = os.path.join(w.dir, "voices", v["name"], pack_id + ".vxpack")
        if os.path.exists(pack):
            tmp = pack + ".tmp"
            with zipfile.ZipFile(pack) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zout:
                for item in zin.infolist():
                    if item.filename == "voice.json":
                        jj = json.loads(zin.read(item))
                        jj["factoryVoices"] = fvs
                        jj["listed"] = listed
                        zout.writestr("voice.json", json.dumps(jj, indent=2, ensure_ascii=False))
                    else:
                        zout.writestr(item, zin.read(item))
            os.replace(tmp, pack)
    return data


def example_queue():
    return {
        "rvc_dir": "/path/to/rvc", "python": "/path/to/venv/bin/python", "device": "cpu",
        "render": "/path/to/VoxaVersaRender (optional)", "encoder": "/path/to/contentvec-768.onnx",
        "voices": [{"name": "my_voice", "dataset": "/path/to/datasets/my_voice", "display_name": "My Voice",
                    "gender": "female", "genre": "Pop", "styles": "Smooth", "license": "Personal use only",
                    "tests": [["/path/to/other_singer_dry.wav", 0]], "min_epochs": 100, "max_epochs": 300}],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["resume", "run", "status", "stop", "example", "_prepare", "_train", "_eval", "_finalize"])
    p.add_argument("voice", nargs="?")
    p.add_argument("--work", help="work folder (queue.json, state, status, logs, results)")
    p.add_argument("--queue", help="queue file (default WORK/queue.json)")
    p.add_argument("--pth")
    p.add_argument("--epoch", type=int)
    args = p.parse_args()
    if args.cmd == "example":
        print(json.dumps(example_queue(), indent=1))
        return
    if not args.work:
        sys.exit("--work is required")
    fn = {"resume": cmd_resume, "run": cmd_run, "status": cmd_status, "stop": cmd_stop, "_prepare": cmd_prepare,
          "_train": cmd_train, "_eval": cmd_eval, "_finalize": cmd_finalize}[args.cmd]
    rc = fn(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
