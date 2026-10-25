#!/usr/bin/env python3
"""
Quick preview of an exported voice: converts a recording with the voice's ONNX generator and the ONNX
content encoder the same way VoxaVersa's offline engine does (16 kHz features, each 20 ms vector used
twice, target pitch per 10 ms, noise 0.66666), so you can listen before importing the pack.

    python3 tools/training/try_voice.py --model my_voice.onnx --encoder contentvec-768.onnx \
        --input dry_vocal.wav --out preview.wav [--transpose -12] [--rvc-dir rvc]

The input's pitch is measured with RMVPE (needs --rvc-dir, the RVC checkout used for training) or, without
it, with a simple YIN tracker; --transpose shifts it in semitones (e.g. +12 for a male input into a female
voice). VoxaVersa itself gets the target pitch from its pitch engine instead.

Requirements: numpy, soundfile, scipy, onnxruntime (+ torch for RMVPE).
"""

import argparse
import math
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal


def resample(x, a, b):
    if a == b:
        return x.astype(np.float32)
    g = math.gcd(int(a), int(b))
    return signal.resample_poly(x, b // g, a // g).astype(np.float32)


def coarse(f0):
    lo, hi = 1127 * math.log(1 + 50 / 700), 1127 * math.log(1 + 1100 / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    c = np.where(f0 > 0, (mel - lo) * 254 / (hi - lo) + 1, 1)
    return np.clip(np.rint(c), 1, 255).astype(np.int64)


def pitch_rmvpe(x16, rvc_dir):
    sys.path.insert(0, os.path.abspath(rvc_dir))
    from infer.rmvpe import RMVPE
    import torch
    model = RMVPE(os.path.join(rvc_dir, "assets/rmvpe/rmvpe.pt"), is_half=False,
                  device="cuda" if torch.cuda.is_available() else "cpu")
    return np.asarray(model.infer_from_audio(x16, thred=0.03), dtype=np.float32)


def pitch_yin(x16):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_dataset import yin_frames
    f0, _ = yin_frames(x16)
    return f0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--transpose", type=float, default=0.0, help="semitones")
    p.add_argument("--rvc-dir", help="RVC checkout (for RMVPE pitch); YIN otherwise")
    p.add_argument("--noise", type=float, default=0.66666)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    enc = ort.InferenceSession(args.encoder, so, providers=providers)
    gen = ort.InferenceSession(args.model, so, providers=providers)
    meta = gen.get_modelmeta().custom_metadata_map
    gen_inputs = {i.name for i in gen.get_inputs()}

    x, sr = sf.read(args.input, dtype="float32", always_2d=True)
    x = x.mean(1)
    x16 = resample(x, sr, 16000)
    frames = int(math.ceil(len(x16) / 160))
    frames += frames % 2
    x16 = np.pad(x16, (0, frames * 160 + 80 - len(x16)))

    f0 = pitch_rmvpe(x16, args.rvc_dir) if args.rvc_dir else pitch_yin(x16)
    f0 = np.pad(f0, (0, max(0, frames - len(f0))))[:frames] * (2 ** (args.transpose / 12))

    # ~10 s segments with 1 s of context on each side, joined with 50 ms crossfades
    seg, ctx = 1000, 100
    out_sr = None
    pieces = []
    rng = np.random.default_rng(0)
    for start in range(0, frames, seg):
        a, b = max(0, start - ctx), min(frames, start + seg + ctx)
        a -= a % 2
        b -= (b - a) % 2
        audio = x16[a * 160: b * 160 + 80][None, None]
        feats = enc.run(None, {enc.get_inputs()[0].name: audio})[0]          # [1, T, 768]
        n = b - a
        idx = np.minimum(np.arange(n) // 2, feats.shape[1] - 1)
        phone = feats[:, idx, :].astype(np.float32)
        feeds = {"phone": phone, "phone_lengths": np.array([n], np.int64), "ds": np.array([0], np.int64),
                 "rnd": (rng.standard_normal((1, 192, n)) * args.noise).astype(np.float32)}
        if "pitchf" in gen_inputs:
            pf = f0[a:b].astype(np.float32)[None]
            feeds.update(pitch=coarse(pf), pitchf=pf)
        y = gen.run(["audio"], feeds)[0].reshape(-1)
        hop = len(y) // n
        out_sr = hop * 100
        fade = 5 * hop                                   # 50 ms, taken from the right context
        end = min(frames, start + seg)
        y = y[(start - a) * hop:min(len(y), (end - a) * hop + fade)]
        pieces.append(y)
    # consecutive pieces overlap by `fade` samples (same time span): crossfade them
    out = pieces[0]
    for y in pieces[1:]:      # every piece but the last ends with `fade` samples of right context
        r = np.linspace(0, 1, fade, dtype=np.float32)
        out = np.concatenate([out[:-fade], out[-fade:] * (1 - r) + y[:fade] * r, y[fade:]])
    out = out[: int(len(x) / sr * out_sr)]
    sf.write(args.out, out, out_sr, subtype="PCM_16")
    voiced = f0[f0 > 0]
    print("wrote %s (%.1f s at %d Hz, %s); target pitch median %.0f Hz" % (
        args.out, len(out) / out_sr, out_sr, meta.get("training", "?"), np.median(voiced) if len(voiced) else 0))


if __name__ == "__main__":
    main()
