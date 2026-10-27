"""
The steps of VoxaVersa_Queue_Colab.ipynb, kept in the repository so that the notebook picks up fixes with a
`git pull` instead of a new notebook file. Every command's output is streamed into the notebook (Colab does
not show what a subprocess prints on its own).
"""

import glob
import json
import os
import subprocess
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = "/content/rvc-env/bin/python"
RVC = "/content/rvc"


def run(cmd, show=True, tail=80):
    """Runs a shell command, printing its output as it comes; raises SystemExit with the last lines on failure."""
    p = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                         errors="replace")
    lines = []
    for line in p.stdout:
        lines.append(line)
        if show:
            print(line, end="", flush=True)
    p.wait()
    if p.returncode != 0:
        if not show:
            print("".join(lines[-tail:]))
        raise SystemExit("Step failed (see the messages above):\n" + cmd)
    return "".join(lines)


def install():
    """Python 3.12 with the same package versions as the CPU trainer (PyTorch 2.4.1, here with CUDA 12.1)."""
    if not os.path.exists(PY):
        print("Installing (about 3 - 5 minutes)...", flush=True)
        run("pip install -q uv", show=False)
        run("uv venv -q --python 3.12 /content/rvc-env", show=False)
        run("uv pip install -q --python %s torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121"
            % PY, show=False)
        run("uv pip install -q --python %s numpy==1.26.4 scipy==1.17.1 librosa==0.10.2.post1 numba==0.68.0 "
            "soundfile==0.14.0 av==15.1.0 ffmpeg-python==0.2.0 praat-parselmouth==0.4.7 transformers==4.49.0 "
            "huggingface_hub==0.36.2 tensorboard==2.21.0 matplotlib==3.11.2 scikit-learn==1.9.1 tqdm "
            "onnx==1.17.0 onnxruntime==1.20.1 speechbrain==1.0.3" % PY, show=False)
    # the AI pitch tracker's export (export_pitch.py); also for runtimes installed before it was added
    if subprocess.run([PY, "-c", "import torchfcpe"], capture_output=True).returncode != 0:
        run("uv pip install -q --python %s torchfcpe==0.0.4 einops==0.8.2 local-attention==1.11.2" % PY, show=False)
    out = run("%s -c \"import torch, speechbrain; print(torch.__version__, torch.cuda.is_available())\"" % PY, show=False)
    print("PyTorch", out.strip().splitlines()[-1])
    if not out.strip().endswith("True"):
        raise SystemExit("PyTorch can't see the GPU - Runtime > Change runtime type > T4 GPU, then Run all.")
    print("OK - software installed")


def setup(work):
    run("%s %s/colab_queue.py setup --work %s --rvc %s --python %s" % (PY, TOOLS, work, RVC, PY))
    print("OK - trainer ready")


def data(work, kit, vocalset=True, precision="mixed", batch=8, military=True):
    flag = "--vocalset" if vocalset else ""
    run("%s %s/colab_queue.py data --work %s --kit '%s' --python %s --rvc %s %s %s"
        % (PY, TOOLS, work, kit, PY, RVC, flag, "" if military else "--no-military"))
    run("%s %s/colab_queue.py queue --work %s --rvc %s --python %s --precision %s --batch %d %s"
        % (PY, TOOLS, work, RVC, PY, precision, batch, flag))
    print("OK - training data ready")


def train(work, every=120):
    from IPython.display import clear_output
    queue = TOOLS + "/train_queue.py"
    print(run("%s %s resume --work %s" % (PY, queue, work), show=False))
    while True:
        time.sleep(every)
        st = run("%s %s status --work %s" % (PY, queue, work), show=False)
        clear_output(wait=True)
        print(time.strftime("%H:%M:%S UTC"), "\n")
        print(st.split("Resume (idempotent)")[0])
        if ("queue complete" in st or "supervisor: not running" in st) and not background_jobs(work):
            try:
                print("--- supervisor log ---\n" + open(work + "/queue_logs/supervisor.log").read()[-1500:])
            except OSError:
                pass
            break


def background_jobs(work):
    """VocalSet or the military-band test still preparing singers: they add them to the queue and restart it."""
    running = []
    for name in ("vocalset", "military"):
        try:
            pid = int(open(os.path.join(work, "colab_logs", name + ".pid")).read().strip())
            os.kill(pid, 0)
            running.append(name)
        except (OSError, ValueError):
            pass
    return running


def results(work):
    from IPython.display import Audio, Markdown, display
    path = os.path.join(work, "queue_state.json")
    state = json.load(open(path)) if os.path.exists(path) else {}
    done = [n for n, s in state.items() if s.get("status") == "done"]
    print("%d voices finished. The .vxpack files are in My Drive/VoxaVersa/queue/voices/<voice>/" % len(done))
    for name in done:
        s = state[name]
        display(Markdown("**%s** - %s - powers: %s" % (name, s.get("final_summary", ""),
                                                       ", ".join(s.get("factory_voices", [])) or "-")))
        for mp3 in sorted(glob.glob(os.path.join(work, "listen", name, "*_3_before_then_after.mp3"))):
            print(os.path.basename(mp3))
            display(Audio(mp3))
    table = os.path.join(work, "factory_assignment.md")
    if os.path.exists(table):
        display(Markdown(open(table).read()))
    report = os.path.join(work, "military", "report.md")
    if os.path.exists(report):
        display(Markdown(open(report).read()))
        for wav in sorted(glob.glob(os.path.join(work, "military", "previews", "*.wav"))):
            print(os.path.basename(wav))
            display(Audio(wav))
