"""
Trains VoxaVersa singers on a free Kaggle GPU, headless: the same queue as on Colab (colab_queue.py +
train_queue.py, see docs/COLAB.md), with the work folder in /kaggle/working so Kaggle keeps it as the run's output.

Pushed as a Kaggle "script" kernel with GPU and internet on, and the private training-data dataset attached
(the data kit VoxaVersa_Colab_Data.zip, plus any words-test recordings). Settings come from the JSON in
VOXAVERSA_KAGGLE (the push tool writes it at the top of the kernel source):

  {"repo": "https://github.com/DK7602/VoxaVersa", "rev": "<commit>",
   "data": "/kaggle/input/<dataset>",        the attached dataset
   "only": ["cymatics_female_blend"],        voices from the kit's voices.json to train
   "words_tests": {"cymatics_female_blend": [["/kaggle/input/<dataset>/user_excerpt.wav", 12, "lyric"]]},
   "military": {"like": ["3 female=bf3_ref_a.wav,bf3_ref_b.wav"]},
                                             optional: rebuild the military-band singers' audio (military_test.py:
                                             download, separate, clean, group) and find singers by reference takes
   "resume": "/kaggle/input/<previous run>", optional: an earlier run's output, to continue from its checkpoints
   "hours": 11}                              stop (with checkpoints) before Kaggle's 12-hour limit

Output (/kaggle/working/queue): queue_state.json and QUEUE_STATUS.md (evaluations: validation mel, singer
similarity, words missed), voices/<voice>/ (ONNX, .vxpack), listen/<voice>/ (before/after MP3s).
"""

import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

START = time.time()
CFG = json.loads(os.environ.get("VOXAVERSA_KAGGLE") or globals().get("VOXAVERSA_KAGGLE", "{}"))
TMP = "/kaggle/tmp"
WORK = "/kaggle/working/queue"
os.environ["VOXAVERSA_PY"] = TMP + "/rvc-env/bin/python"
os.environ["VOXAVERSA_RVC"] = TMP + "/rvc"


def sh(cmd):
    print("+ " + cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    os.makedirs(TMP, exist_ok=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working %s | tail -2; ls -R /kaggle/input | head -20" % TMP)
    repo = TMP + "/VoxaVersa"
    if not os.path.isdir(repo):
        sh("git clone -q %s %s" % (CFG.get("repo", "https://github.com/DK7602/VoxaVersa"), repo))
    sh("cd %s && git fetch -q origin %s 2>/dev/null; git checkout -q %s" % (repo, CFG["rev"], CFG["rev"]))
    tools = repo + "/tools/training"
    sys.path.insert(0, tools)
    import colab_steps as cs

    # an earlier run's output: continue from its checkpoints and evaluations
    if CFG.get("resume") and not os.path.exists(WORK):
        prev = glob.glob(os.path.join(CFG["resume"], "**", "queue_state.json"), recursive=True)
        if prev:
            print("resuming from " + os.path.dirname(prev[0]), flush=True)
            shutil.copytree(os.path.dirname(prev[0]), WORK, symlinks=True)

    cs.install()
    cs.setup(WORK)

    # the data kit: Kaggle may have unpacked the zip; colab_queue.py wants the zip
    kit = glob.glob(os.path.join(CFG["data"], "**", "VoxaVersa_Colab_Data.zip"), recursive=True)
    if kit:
        kit = kit[0]
    else:
        folder = os.path.dirname(glob.glob(os.path.join(CFG["data"], "**", "voices.json"), recursive=True)[0])
        kit = shutil.make_archive(TMP + "/VoxaVersa_Colab_Data", "zip", os.path.dirname(folder),
                                  os.path.basename(folder))
    def find(path):     # a file of the attached dataset, wherever Kaggle mounted it
        if os.path.exists(path):
            return path
        hits = glob.glob(os.path.join(CFG["data"], "**", os.path.basename(path)), recursive=True)
        return hits[0] if hits else path

    py, q = cs.PY, tools + "/colab_queue.py"
    only = " ".join(CFG.get("only", []))
    cs.run("%s %s data --work %s --kit '%s' --python %s --rvc %s --no-military" % (py, q, WORK, kit, py, cs.RVC))
    if CFG.get("military"):
        likes = []
        for spec in CFG["military"].get("like", []):
            head, files = spec.split("=", 1)
            likes += ["--like", head + "=" + ",".join(find(f) for f in files.split(","))]
        cs.run(" ".join(shlex.quote(a) for a in [py, tools + "/military_test.py", "run", "--work", WORK, "--rvc", cs.RVC,
                                                   "--scratch", TMP + "/scratch", "--ecapa", TMP + "/ecapa_military"] + likes))
    cs.run("%s %s queue --work %s --rvc %s --python %s --precision mixed --batch 8 %s"
           % (py, q, WORK, cs.RVC, py, ("--only " + only) if only else ""))

    queue_path = os.path.join(WORK, "queue.json")
    queue = json.load(open(queue_path))
    for v in queue["voices"]:
        if v["name"] in CFG.get("words_tests", {}):
            v["words_tests"] = [[find(t[0])] + list(t[1:]) for t in CFG["words_tests"][v["name"]]]
    queue["asr_dir"] = TMP + "/asr_model"
    queue["ecapa_dir"] = TMP + "/ecapa_model"
    json.dump(queue, open(queue_path, "w"), indent=1)

    tq = tools + "/train_queue.py"
    print(cs.run("%s %s resume --work %s" % (py, tq, WORK), show=False), flush=True)
    limit = 3600 * float(CFG.get("hours", 11))
    while True:
        time.sleep(120)
        st = cs.run("%s %s status --work %s" % (py, tq, WORK), show=False)
        print(time.strftime("%H:%M:%S"), st.split("Resume (idempotent)")[0][-1500:], flush=True)
        if "queue complete" in st or "supervisor: not running" in st:
            break
        if time.time() - START > limit:
            print("time limit: stopping with checkpoints; push the next run with this one as \"resume\"", flush=True)
            cs.run("%s %s stop --work %s" % (py, tq, WORK), show=False)
            break
    try:
        print(open(os.path.join(WORK, "queue_logs", "supervisor.log")).read()[-3000:])
    except OSError:
        pass


if __name__ == "__main__":
    main()
