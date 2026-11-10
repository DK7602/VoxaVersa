#!/usr/bin/env python3
"""
Builds a singer's feature index for VoxaVersa (the "index" of a voice pack, see docs/NEURAL_ENGINE.md): content
features (contentvec-768, as the plug-in's encoder computes them) of the singer's own training recordings, k-means
clustered. The plug-in pulls each frame of the input toward the nearest of them, so the singer sings with her own
sounds instead of the input singer's (RVC's .index; without it a man's input made a woman sound "like a man sped
up").

    # from RVC's training features (logs/<voice>/3_feature768/*.npy)
    python3 tools/neural/build_index.py build --features logs/my_voice/3_feature768 --out my_voice_index.npy

    # from recordings, with the plug-in's encoder
    python3 tools/neural/build_index.py build --audio dataset/my_voice --encoder contentvec-768.onnx --out index.npy

    # an update adding the index to an installed singer (found by her id; everything else she has stays);
    # import the .vxpack as usual
    python3 tools/neural/build_index.py update --id my-voice --index index.npy --out my-voice-index.vxpack

Needs numpy and scikit-learn; --audio also soundfile, scipy and onnxruntime. The index is derived from the
recordings: share it only where the recordings' terms allow (a GTSinger singer's: personal use only).
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import zipfile

import numpy as np

CLUSTERS = 2048          # measured as close to the singer's recordings as 8192, at a quarter of the search time
MIN_FRAMES = 4 * CLUSTERS


def fail(message):
    print("error: " + message, file=sys.stderr)
    sys.exit(1)


def feature_files(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.npy"), recursive=True)) if os.path.isdir(p) else [p]
    return files


def audio_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("wav", "flac", "mp3", "ogg"):
                files += glob.glob(os.path.join(p, "**", "*." + ext), recursive=True)
        else:
            files.append(p)
    return sorted(files)


def encode(paths, encoder, threads=0):
    """contentvec features of the recordings, as the plug-in computes them: [frames, 768] at 50 frames/s, frames
    with sound only (RVC trains on sliced, silence-trimmed audio)."""
    import onnxruntime as ort
    import soundfile as sf
    from scipy.signal import resample_poly
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(encoder, sess_options=options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    rank = len(session.get_inputs()[0].shape)
    out = []
    for i, path in enumerate(paths, 1):
        x, sr = sf.read(path, always_2d=True, dtype="float32")
        x = x.mean(axis=1)
        if sr != 16000:
            g = np.gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
        x = x / max(1e-3, float(np.abs(x).max())) * 0.9
        for start in range(0, len(x), 16000 * 20):
            seg = x[start:start + 16000 * 20]
            if len(seg) < 16000:
                continue
            f = session.run(None, {name: seg[None, None, :] if rank == 3 else seg[None, :]})[0]
            f = f.reshape(-1, f.shape[-1])
            rms = np.sqrt(np.mean(seg[:len(f) * 320].reshape(-1, 320) ** 2, axis=1)) if len(seg) >= len(f) * 320 else None
            out.append(f[rms > 0.01] if rms is not None else f)
        if i % 50 == 0 or i == len(paths):
            print("  %d / %d recordings, %d frames" % (i, len(paths), sum(len(f) for f in out)), flush=True)
    return np.concatenate(out).astype(np.float32)


def cluster(features, clusters=CLUSTERS, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    if len(features) < MIN_FRAMES:
        print("note: only %d frames (%.0f s of singing): the index keeps them all" % (len(features), len(features) / 50))
        return features.astype(np.float16)
    km = MiniBatchKMeans(n_clusters=clusters, batch_size=256 * 32, n_init=1, max_iter=60, random_state=seed,
                         compute_labels=False).fit(features)
    return km.cluster_centers_.astype(np.float16)


def read_pack(pack):
    """voice.json of a .vxpack (zip) or an installed pack folder."""
    if os.path.isdir(pack):
        return json.load(open(os.path.join(pack, "voice.json"), encoding="utf-8"))
    with zipfile.ZipFile(pack) as z:
        name = next((n for n in z.namelist() if n.endswith("voice.json")), None)
        if name is None:
            fail(pack + " has no voice.json")
        return json.loads(z.read(name).decode("utf-8"))


def cmd_build(args):
    if args.features:
        files = feature_files(args.features)
        if not files:
            fail("no .npy feature files in " + ", ".join(args.features))
        feats = np.concatenate([np.load(f).reshape(-1, 768) for f in files]).astype(np.float32)
    else:
        if not args.encoder:
            fail("--audio needs --encoder (contentvec-768.onnx)")
        files = audio_files(args.audio)
        if not files:
            fail("no recordings in " + ", ".join(args.audio))
        feats = encode(files, args.encoder, args.threads)
    print("%d frames (%.1f min of singing) from %d files" % (len(feats), len(feats) / 50 / 60, len(files)), flush=True)
    index = cluster(feats, args.clusters)
    np.save(args.out, index)
    print("Wrote %s: %d x %d, %.1f MB" % (args.out, index.shape[0], index.shape[1], os.path.getsize(args.out) / 1e6))


def cmd_update(args):
    """An update pack: {"update": true, id, name, model: {index}} + the index. VoxaVersa finds the installed singer
    by her id and adds the index to her voice.json; everything else she has stays."""
    if args.pack:
        voice = read_pack(args.pack)
        if not isinstance(voice.get("model"), dict):
            fail(args.pack + " is not an AI singer's pack (no model)")
        sid, name = voice.get("id") or voice.get("name"), voice.get("name") or voice.get("id")
    else:
        sid, name = args.id, args.display_name or args.id
    if not sid:
        fail("--pack or --id is needed")
    index_name = args.name or (sid + "_index.npy")
    model = {"index": index_name}
    if args.rate is not None:
        model["indexRate"] = args.rate
    update = {"update": True, "id": sid, "name": name, "model": model}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        manifest = os.path.join(tmp, "voice.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(update, f, indent=2, ensure_ascii=False)
        with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(manifest, "voice.json")
            z.write(args.index, index_name)
    print("Wrote %s: an update of %s (%s) adding its feature index" % (args.out, name, sid))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="cluster a singer's features into an index")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", nargs="+", help="RVC 3_feature768 folder(s) or .npy files ([frames, 768])")
    src.add_argument("--audio", nargs="+", help="the singer's recordings (folders or files)")
    b.add_argument("--encoder", help="contentvec-768.onnx (with --audio)")
    b.add_argument("--threads", type=int, default=0, help="encoder threads (default: all)")
    b.add_argument("--clusters", type=int, default=CLUSTERS)
    b.add_argument("--out", required=True, help="the index (.npy)")
    u = sub.add_parser("update", help="an update pack adding the index to an installed singer")
    who = u.add_mutually_exclusive_group(required=True)
    who.add_argument("--pack", help="the singer's .vxpack or installed folder (for her id and name)")
    who.add_argument("--id", help="the singer's pack id, e.g. studio-female-a")
    u.add_argument("--display-name", help="her name, for messages (with --id)")
    u.add_argument("--index", required=True)
    u.add_argument("--name", help="the index's file name in the pack (default <id>_index.npy)")
    u.add_argument("--rate", type=float, help="indexRate (default: the plug-in's, 0.75)")
    u.add_argument("--out", required=True, help="the update .vxpack")
    args = p.parse_args()
    (cmd_build if args.cmd == "build" else cmd_update)(args)


if __name__ == "__main__":
    main()
