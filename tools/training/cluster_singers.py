#!/usr/bin/env python3
"""
Finds out which recordings were sung by the same person, so a voice model is trained on one singer
(or on a deliberate blend) instead of an accidental mix.

    python3 tools/training/prepare_dataset.py vocals/ --out clean --name pool      # clean clips first
    python3 tools/training/cluster_singers.py clean/pool --report clean/report.json --out clusters

Every clip gets a speaker embedding from SpeechBrain's ECAPA-TDNN model trained on VoxCeleb
(speechbrain/spkrec-ecapa-voxceleb, Apache-2.0, downloaded from Hugging Face on first use; runs on the
CPU). The clips of each source file ("stem") are averaged, and stems are clustered by cosine similarity
(average linkage). Because a speaker model trained on speech is less sure about singing, the report
calibrates the scale on your own data:

  * same stem:      the similarity between the two halves of one stem (same singer by definition),
  * other gender:   stems whose median pitch differs by more than 9 semitones and that the user did
                    not put into the same cluster (almost surely different singers).

and states, for every cluster, how tight it is and how far it is from the nearest other cluster.

Outputs (in --out): clusters.md (human readable), clusters.json, manifest.csv (columns path,voice: the
source files per cluster, ready for prepare_dataset.py --manifest after you rename/merge the voices),
and with --copy-to DIR the clips themselves sorted into DIR/<cluster>/.

Requirements: pip install torch torchaudio speechbrain soundfile scipy numpy
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def load_16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        x = signal.resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def f0_median(x):
    """Rough median f0 of a 16 kHz clip (autocorrelation per 40 ms frame, voiced frames only)."""
    hop, win = 320, 640
    f0s = []
    for s in range(0, len(x) - win, hop):
        fr = x[s:s + win] - x[s:s + win].mean()
        e = float(np.dot(fr, fr))
        if e < 1e-4:
            continue
        r = np.correlate(fr, fr, "full")[win - 1:]
        lo, hi = 16000 // 1000, 16000 // 60
        lag = lo + int(np.argmax(r[lo:hi]))
        if r[lag] / (r[0] + 1e-12) > 0.6:
            f0s.append(16000.0 / lag)
    return float(np.median(f0s)) if len(f0s) >= 5 else 0.0


def confidence_label(cohesion, nearest, threshold):
    """'high' when the cluster is clearly tighter than its distance to any other cluster."""
    if nearest is None:
        return "high"
    tight = cohesion if cohesion is not None else 1.0
    gap = tight - nearest
    if nearest < threshold - 0.1 and gap > 0.15:
        return "high"
    if nearest < threshold and gap > 0.05:
        return "medium"
    return "low (may be the same singer as its nearest cluster)"


def stem_of(clip_name):
    return re.sub(r"_\d{3}\.wav$", "", clip_name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="+", help="folders of clips made by prepare_dataset.py")
    p.add_argument("--report", help="prepare_dataset.py report.json (maps clips back to their source files)")
    p.add_argument("--out", required=True, help="output folder for the reports")
    p.add_argument("--threshold", type=float, default=None,
                   help="cosine similarity above which stems are merged (default: calibrated from your data)")
    p.add_argument("--group", metavar="REGEX",
                   help="cluster groups of stems instead of single stems: the regex's first non-empty capture "
                        "group (searched in the stem name) is the group, e.g. the song name; stems that don't "
                        "match stay on their own")
    p.add_argument("--core-seconds", type=float, default=15.0,
                   help="units with less audio are not clustered but attached to the nearest cluster when "
                        "they are clearly closest to it (short phrases give noisy embeddings)")
    p.add_argument("--min-seconds", type=float, default=0.0, help="ignore stems with less audio than this")
    p.add_argument("--threads", type=int, default=2, help="CPU threads for the embedding model")
    p.add_argument("--copy-to", help="copy the clips into <dir>/<cluster>/")
    p.add_argument("--cache", default=None, help="embedding cache file (default: <out>/embeddings.npz)")
    args = p.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    clip_paths = []
    for folder in args.clips:
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".wav"):
                clip_paths.append(os.path.join(folder, name))
    if not clip_paths:
        raise SystemExit("no clips found")

    source_of = {}
    if args.report:
        rep = json.load(open(args.report, encoding="utf-8"))
        for f in rep["files"]:
            for entry in f.get("clip_list", []):
                source_of[entry[0]] = f["file"]

    cache_path = args.cache or os.path.join(args.out, "embeddings.npz")
    cache = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        cache = {k: (z["emb"][i], float(z["dur"][i]), float(z["f0"][i])) for i, k in enumerate(z["keys"])}

    todo = [c for c in clip_paths if c not in cache]
    if todo:
        from speechbrain.inference.speaker import EncoderClassifier
        model = EncoderClassifier.from_hparams(source=MODEL, savedir=os.path.join(args.out, "ecapa_model"),
                                               run_opts={"device": "cpu"})
        for i, c in enumerate(todo):
            x = load_16k(c)
            with torch.no_grad():
                e = model.encode_batch(torch.from_numpy(x)[None]).squeeze().numpy()
            cache[c] = (e / (np.linalg.norm(e) + 1e-9), len(x) / 16000.0, f0_median(x))
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print("embedded %d/%d clips" % (i + 1, len(todo)), flush=True)
        keys = sorted(cache)
        np.savez(cache_path, keys=np.array(keys), emb=np.stack([cache[k][0] for k in keys]),
                 dur=np.array([cache[k][1] for k in keys]), f0=np.array([cache[k][2] for k in keys]))

    # ---- stems
    stems = defaultdict(list)
    for c in clip_paths:
        key = stem_of(os.path.basename(c))
        if args.group:
            m = re.search(args.group, key)
            if m:
                key = next((g for g in m.groups() if g), m.group(0))
        stems[key].append(c)
    names = sorted(s for s in stems if sum(cache[c][1] for c in stems[s]) >= args.min_seconds)
    emb = {}
    halves = []
    info = {}
    for s in names:
        cl = stems[s]
        E = np.stack([cache[c][0] for c in cl])
        w = np.array([cache[c][1] for c in cl])
        m = (E * w[:, None]).sum(0)
        emb[s] = m / np.linalg.norm(m)
        f0s = [cache[c][2] for c in cl if cache[c][2] > 0]
        info[s] = dict(seconds=float(w.sum()), clips=len(cl), f0=float(np.median(f0s)) if f0s else 0.0,
                       sources=sorted({source_of.get(os.path.basename(c), stem_of(os.path.basename(c))) for c in cl}))
        if len(cl) >= 4:
            a, b = E[0::2].mean(0), E[1::2].mean(0)
            halves.append(float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))

    X = np.stack([emb[s] for s in names])
    S = np.clip(X @ X.T, -1, 1)
    n = len(names)

    def semis(a, b):
        return abs(12 * math.log2(a / b)) if a > 0 and b > 0 else 0.0

    other_gender = [S[i, j] for i in range(n) for j in range(i + 1, n)
                    if semis(info[names[i]]["f0"], info[names[j]]["f0"]) > 9]
    same = np.array(halves)
    diff = np.array(other_gender)
    if args.threshold is not None:
        threshold = args.threshold
    elif len(diff) >= 20:
        # just above what clearly different singers reach (99th percentile of the pairs whose pitch is
        # more than 9 semitones apart), within sensible bounds
        threshold = float(np.clip(np.percentile(diff, 99) + 0.03, 0.5, 0.75))
    else:
        threshold = 0.6

    # cluster the units with enough audio, attach the short ones afterwards
    core = [i for i, s in enumerate(names) if info[s]["seconds"] >= args.core_seconds]
    if len(core) < 2:
        core = list(range(n))
    labels = np.zeros(n, int)
    if len(core) >= 2:
        D = 1 - S[np.ix_(core, core)]
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(np.clip(D, 0, 2), checks=False), method="average")
        labels[core] = fcluster(Z, t=1 - threshold, criterion="distance")
    else:
        labels[core] = 1
    clusters = defaultdict(list)
    for i in core:
        clusters[int(labels[i])].append(names[i])
    cents = {}
    for l, members in clusters.items():
        m = np.mean([emb[s] for s in members], axis=0)
        cents[l] = m / np.linalg.norm(m)
    unassigned = []
    for i, s in enumerate(names):
        if i in core:
            continue
        sims = sorted(((float(np.dot(emb[s], c)), l) for l, c in cents.items()), reverse=True)
        best, second = sims[0], (sims[1] if len(sims) > 1 else (-1.0, None))
        if best[0] >= threshold and best[0] - second[0] >= 0.05:
            clusters[best[1]].append(s)
        else:
            unassigned.append(s)
    # the short units that fit no cluster: cluster them among themselves (e.g. a pack of short phrases by
    # one singer); groups of at least 3 become clusters of their own, the rest stays unassigned
    if len(unassigned) >= 3:
        idx = [names.index(s) for s in unassigned]
        D = 1 - S[np.ix_(idx, idx)]
        np.fill_diagonal(D, 0)
        sub = fcluster(linkage(squareform(np.clip(D, 0, 2), checks=False), method="average"),
                       t=1 - threshold, criterion="distance")
        rest = []
        next_label = max(clusters) + 1 if clusters else 1
        for l in sorted(set(sub)):
            members = [unassigned[k] for k in range(len(unassigned)) if sub[k] == l]
            if len(members) >= 3:
                clusters[next_label] = members
                m = np.mean([emb[x] for x in members], axis=0)
                cents[next_label] = m / np.linalg.norm(m)
                next_label += 1
            else:
                rest += members
        unassigned = rest
    if unassigned:
        clusters[-1] = unassigned
    order = sorted((l for l in clusters if l != -1), key=lambda l: -sum(info[s]["seconds"] for s in clusters[l]))
    if unassigned:
        order.append(-1)
        m = np.mean([emb[s] for s in unassigned], axis=0)
        cents[-1] = m / np.linalg.norm(m)

    # confidence is judged against the clusters that have enough audio for a reliable centroid
    minutes_of = {l: sum(info[s]["seconds"] for s in clusters[l]) / 60 for l in order}
    solid = [l for l in order if l != -1 and minutes_of[l] >= 1.0] or [l for l in order if l != -1]

    result = []
    for k, l in enumerate(order):
        members = clusters[l]
        idx = [names.index(s) for s in members]
        cohesion = float(np.mean([S[i, j] for i in idx for j in idx if i < j])) if len(idx) > 1 else None
        nearest = max(((float(np.dot(cents[l], cents[o])), o) for o in solid if o != l), default=(None, None))
        f0s = [info[s]["f0"] for s in members if info[s]["f0"] > 0]
        med = float(np.median(f0s)) if f0s else 0.0
        stems_out = []
        for s in sorted(members):
            own = float(np.dot(emb[s], cents[l]))
            other = max((float(np.dot(emb[s], cents[o])) for o in order if o not in (l, -1)), default=-1.0)
            stems_out.append(dict(stem=s, sources=info[s]["sources"], seconds=round(info[s]["seconds"], 1),
                                  clips=info[s]["clips"], f0=round(info[s]["f0"], 1),
                                  own=round(own, 3), next_best=round(other, 3), margin=round(own - other, 3)))
        result.append(dict(cluster="unassigned" if l == -1 else "singer_%02d" % (k + 1), minutes=round(sum(info[s]["seconds"] for s in members) / 60, 2),
                           stems=stems_out, median_f0=round(med, 1),
                           likely_gender="male?" if 0 < med < 200 else ("female?" if med > 260 else "unclear"),
                           cohesion=None if cohesion is None else round(cohesion, 3),
                           nearest_cluster=None if nearest[1] is None else "singer_%02d" % (order.index(nearest[1]) + 1),
                           confidence=confidence_label(cohesion, nearest[0], threshold) if l != -1 else "n/a",
                           nearest_similarity=None if nearest[0] is None else round(nearest[0], 3)))

    calib = dict(threshold=round(threshold, 3),
                 same_stem_halves=None if not len(same) else [round(float(v), 3) for v in np.percentile(same, [10, 50, 90])],
                 other_gender_pairs=None if not len(diff) else [round(float(v), 3) for v in np.percentile(diff, [50, 90, 99])])
    with open(os.path.join(args.out, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(dict(calibration=calib, clusters=result, stems=names, similarity=S.round(3).tolist()), f, indent=1)

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "voice"])
        for c in result:
            for s in c["stems"]:
                for src in s["sources"]:
                    w.writerow([src, c["cluster"]])

    lines = ["# Singer clusters", "",
             "Cosine similarity of ECAPA speaker embeddings (1 = identical); 'nearest' and the confidence refer to the "
             "clusters with at least 1 minute of audio. Gender is only guessed from the median "
             "sung pitch (male? < 200 Hz, female? > 260 Hz): check it by ear. Calibration on this data: two halves "
             "of the same stem p10/p50/p90 = %s; stems more than 9 semitones apart in pitch (different singers) "
             "p50/p90/p99 = %s. Merge threshold: %.3f." % (calib["same_stem_halves"], calib["other_gender_pairs"], threshold), ""]
    for c in result:
        lines.append("## %s: %.1f min, %d stems, median f0 %.0f Hz (%s)" % (c["cluster"], c["minutes"], len(c["stems"]),
                                                                          c["median_f0"], c["likely_gender"]))
        lines.append("")
        lines.append("Cohesion (mean pairwise similarity) %s; nearest other cluster %s at %s; confidence: %s." % (
            c["cohesion"], c["nearest_cluster"], c["nearest_similarity"], c.get("confidence")))
        lines.append("")
        lines.append("| Unit | Seconds | f0 Hz | Similarity to cluster | Best other | Margin |")
        lines.append("|---|---|---|---|---|---|")
        for s in c["stems"]:
            lines.append("| %s | %.0f | %.0f | %.3f | %.3f | %.3f |" % (s["stem"], s["seconds"], s["f0"], s["own"], s["next_best"], s["margin"]))
        lines.append("")
    with open(os.path.join(args.out, "clusters.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if args.copy_to:
        for c in result:
            d = os.path.join(args.copy_to, c["cluster"])
            os.makedirs(d, exist_ok=True)
            for s in c["stems"]:
                for clip in stems[s["stem"]]:
                    shutil.copy2(clip, d)

    print("calibration:", calib)
    for c in result:
        print("%-10s %5.1f min %3d units f0 %5.0f %-6s cohesion %s nearest %s (%s) confidence %s" % (
            c["cluster"], c["minutes"], len(c["stems"]), c["median_f0"], c["likely_gender"], c["cohesion"],
            c["nearest_cluster"], c["nearest_similarity"], c.get("confidence")))
    print("details:", os.path.join(args.out, "clusters.md"))


if __name__ == "__main__":
    main()
