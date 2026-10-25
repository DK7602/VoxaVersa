#!/usr/bin/env python3
"""
Exports the content encoder that RVC v2 models are trained with (ContentVec / HuBERT base, layer 12) to
the ONNX format VoxaVersa expects, so voice packs never depend on third-party ONNX files:

    input   source  [1, 1, N]   float32, 16 kHz mono waveform (not normalised, like RVC)
    output  embed   [1, T, 768] float32, the 12th transformer layer, one vector per 20 ms
                                (T = floor((N - 400) / 320) + 1; VoxaVersa pads N by 80 samples)

The weights are the checkpoint the trainer itself uses: the RVC WebUI's assets/hubert_base
(transformers format, published in the MIT-licensed Hugging Face repository lj1995/VoiceConversionWebUI;
it is RVC's hubert_base.pt = ContentVec by auspicious3000, MIT). Training features (RVC's
extract_hubert_feature.py: HubertModel(...).last_hidden_state) and VoxaVersa's inference features are
therefore computed by the same network. The export is verified against PyTorch on real audio.

    python3 tools/training/export_encoder.py --model rvc/assets/hubert_base --out contentvec-768.onnx

Install the result once as <user data>/VoxaVersa/Encoders/contentvec-768.onnx (see docs/TRAINING.md).
Requirements: torch, transformers (4.49 tested), onnx, onnxruntime (for the check), numpy, soundfile.
"""

import argparse
import math
import os
import sys

import numpy as np

HF_REPO = "lj1995/VoiceConversionWebUI"
HF_REVISION = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]


def fetch(model_dir):
    from urllib.request import urlopen
    import shutil
    os.makedirs(model_dir, exist_ok=True)
    for f in FILES:
        dest = os.path.join(model_dir, f)
        if not os.path.exists(dest):
            url = "https://huggingface.co/%s/resolve/%s/hubert_base/%s" % (HF_REPO, HF_REVISION, f)
            print("downloading", url)
            with urlopen(url) as r, open(dest + ".part", "wb") as o:
                shutil.copyfileobj(r, o, length=1 << 20)
            os.replace(dest + ".part", dest)


def load(model_dir):
    import torch
    from transformers import HubertConfig, HubertModel
    config = HubertConfig.from_pretrained(model_dir)
    if getattr(config, "do_stable_layer_norm", False) or config.hidden_size != 768 or config.num_hidden_layers != 12:
        raise SystemExit("unexpected model: VoxaVersa's RVC v2 path needs HuBERT/ContentVec base (768-d, 12 layers)")
    model = HubertModel.from_pretrained(model_dir, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    # fold the positional convolution's weight normalisation into a plain weight
    conv = model.encoder.pos_conv_embed.conv
    try:
        from torch.nn.utils import parametrize
        if parametrize.is_parametrized(conv):
            parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    except ImportError:
        pass
    if hasattr(conv, "weight_g"):
        torch.nn.utils.remove_weight_norm(conv)
    return model


def wrapper(model):
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self, hubert):
            super().__init__()
            self.hubert = hubert

        def forward(self, source):
            x = source.reshape(source.shape[0], source.shape[-1])
            return self.hubert(input_values=x).last_hidden_state

    return Encoder(model).eval()


def test_signal(seconds, seed=0):
    """A sung-vowel-like test signal (harmonics with vibrato and an envelope) plus a little noise."""
    sr = 16000
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(seed)
    f0 = 196 * 2 ** ((np.sin(2 * math.pi * 5.5 * t) * 0.3 + np.floor(t * 1.5) % 5 * 2) / 12)
    phase = 2 * math.pi * np.cumsum(f0) / sr
    x = sum((0.5 / k) * np.sin(k * phase) * (1.0 if k < 6 else 0.4) for k in range(1, 20))
    env = 0.5 + 0.5 * np.sin(2 * math.pi * 0.7 * t) ** 2
    return (0.3 * x * env + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def load_audio_16k(path):
    import soundfile as sf
    from scipy import signal
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        x = signal.resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="transformers HuBERT folder (RVC's assets/hubert_base)")
    p.add_argument("--download", action="store_true", help="download the pinned assets/hubert_base into --model first")
    p.add_argument("--out", required=True, help="output .onnx (e.g. contentvec-768.onnx)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--check-audio", nargs="*", default=[], help="real recordings to compare PyTorch and ONNX on")
    p.add_argument("--compare-with", help="another encoder ONNX to report the difference to (information only)")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    if args.download:
        fetch(args.model)
    model = load(args.model)
    enc = wrapper(model)

    example = torch.from_numpy(test_signal(2.0))[None, None]
    torch.onnx.export(enc, (example,), args.out, input_names=["source"], output_names=["embed"],
                      dynamic_axes={"source": {2: "samples"}, "embed": {1: "frames"}},
                      opset_version=args.opset, do_constant_folding=True)
    import onnx
    m = onnx.load(args.out)
    for k, v in (("model", "ContentVec/HuBERT base (RVC assets/hubert_base, %s@%s)" % (HF_REPO, HF_REVISION[:10])),
                 ("layer", "12"), ("feature_dim", "768"), ("sample_rate", "16000"), ("frame_ms", "20"),
                 ("license", "MIT (ContentVec, RVC WebUI)"), ("exporter", "VoxaVersa export_encoder.py")):
        prop = m.metadata_props.add()
        prop.key, prop.value = k, v
    onnx.save(m, args.out)
    print("wrote %s (%.0f MB)" % (args.out, os.path.getsize(args.out) / 1e6))

    try:
        import onnxruntime as ort
    except ImportError:
        print("note: onnxruntime not installed - skipped the numerical check")
        return
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])
    other = ort.InferenceSession(args.compare_with, so, providers=["CPUExecutionProvider"]) if args.compare_with else None

    signals = [("test signal 0.5 s", test_signal(0.5, 1)), ("test signal 7.3 s", test_signal(7.3, 2))]
    for path in args.check_audio:
        x = load_audio_16k(path)
        signals.append((os.path.basename(path), x[: 16000 * 12]))
    worst = 0.0
    for name, x in signals:
        n = len(x) - len(x) % 320 + 80      # VoxaVersa's framing: N = frames * 320 + 80
        x = np.pad(x, (0, max(0, n - len(x))))[:n].astype(np.float32)
        ref = enc(torch.from_numpy(x)[None, None]).numpy()
        got = sess.run(["embed"], {"source": x[None, None]})[0]
        if ref.shape != got.shape or got.shape[1] != (n - 80) // 320 or got.shape[2] != 768:
            raise SystemExit("check failed for %s: ONNX %s, PyTorch %s" % (name, got.shape, ref.shape))
        err = float(np.abs(ref - got).max())
        rel = float(np.linalg.norm(ref - got) / (np.linalg.norm(ref) + 1e-12))
        cos = float(np.mean(np.sum(ref * got, -1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1) + 1e-12)))
        worst = max(worst, rel)
        line = "%-40s %4d frames: max |onnx - torch| %.2e, relative %.2e, cosine %.6f" % (name[:40], got.shape[1], err, rel, cos)
        if other is not None:
            o = other.run(None, {other.get_inputs()[0].name: x[None, None]})[0]
            if o.shape == got.shape:
                c2 = float(np.mean(np.sum(o * got, -1) / (np.linalg.norm(o, axis=-1) * np.linalg.norm(got, axis=-1) + 1e-12)))
                line += " | vs %s: cosine %.4f" % (os.path.basename(args.compare_with), c2)
        print(line)
    if worst > 1e-3:
        raise SystemExit("check failed: relative error %.2e" % worst)
    print("check OK: ONNX matches PyTorch (worst relative error %.1e)" % worst)


if __name__ == "__main__":
    main()
