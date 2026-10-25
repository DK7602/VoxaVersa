#!/usr/bin/env python3
"""
Exports a trained RVC voice model (.pth from the RVC WebUI, Applio or rvc_train.py) to the ONNX format
VoxaVersa loads (docs/NEURAL_ENGINE.md):

    inputs   phone [1,T,768] float, phone_lengths [1] int64, pitch [1,T] int64 (coarse 1..255),
             pitchf [1,T] float (Hz, 0 = unvoiced), ds [1] int64 (speaker), rnd [1,192,T] float (noise)
    output   audio [1,1,T*hop] float          (hop = sample rate / 100, T = 10 ms frames)

This is the I/O of the RVC WebUI's former "Export Onnx" tab (removed in RVC 2.3), rebuilt on the model
code of the pinned RVC checkout: z_p = (m_p + exp(logs_p) * rnd) * mask, so VoxaVersa controls the noise
(its noiseScale, default 0.66666, multiplies a Gaussian). The sample rate is stored in the model's
metadata ("sampling_rate"). v1 (256-d) and pitch-less models are exported too.

    python3 tools/training/export_onnx.py --rvc-dir rvc --pth rvc/assets/weights/my_voice.pth --out my_voice.onnx

The export is verified with ONNX Runtime (if installed): with the generator's random excitation noise
switched off in both, the ONNX model must reproduce PyTorch to within 1e-3 (max abs, typically ~1e-5);
the real model is then checked for finite output at several lengths.

Requirements: torch, onnx, (onnxruntime for the check) and the RVC WebUI code (--rvc-dir).
"""

import argparse
import math
import os
import sys

import numpy as np


def _import_rvc(rvc_dir):
    rvc_dir = os.path.abspath(rvc_dir)
    if rvc_dir not in sys.path:
        sys.path.insert(0, rvc_dir)
    import torch.nn.functional as F
    from infer.module import attentions, models

    # RVC 2.3 converts sequence lengths to Python ints in the relative attention, which would freeze the
    # traced graph to the example length. These are the same operations without the int() casts.
    def rel_to_abs(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]

    def abs_to_rel(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * length + length * (length - 1)])
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    attentions.MultiHeadAttention._relative_position_to_absolute_position = rel_to_abs
    attentions.MultiHeadAttention._absolute_position_to_relative_position = abs_to_rel
    return models


def from_training_checkpoint(cpt, pth):
    """A G_<step>.pth training checkpoint (+ config.json in the same folder) -> inference model dict,
    like RVC's process_ckpt.savee, so a model can be exported while training is still running."""
    import json
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(pth)), "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit("%s is a training checkpoint but %s is missing" % (pth, cfg_path))
    hps = json.load(open(cfg_path, encoding="utf-8"))
    m, d = hps["model"], hps["data"]
    weight = {k: v for k, v in cpt["model"].items() if "enc_q" not in k}
    config = [d["filter_length"] // 2 + 1, 32, m["inter_channels"], m["hidden_channels"], m["filter_channels"],
              m["n_heads"], m["n_layers"], m["kernel_size"], m["p_dropout"], m["resblock"], m["resblock_kernel_sizes"],
              m["resblock_dilation_sizes"], m["upsample_rates"], m["upsample_initial_channel"],
              m["upsample_kernel_sizes"], m["spk_embed_dim"], m["gin_channels"], d["sampling_rate"]]
    version = "v2" if weight["enc_p.emb_phone.weight"].shape[1] == 768 else "v1"
    f0 = 1 if "enc_p.emb_pitch.weight" in weight else 0
    return {"weight": weight, "config": config, "version": version, "f0": f0,
            "info": "%sepoch (training checkpoint)" % cpt.get("iteration", "?")}


def load_model(pth, rvc_dir):
    import torch
    models = _import_rvc(rvc_dir)
    cpt = torch.load(pth, map_location="cpu", weights_only=False)
    if "weight" not in cpt and "model" in cpt:
        cpt = from_training_checkpoint(cpt, pth)
    if "weight" not in cpt or "config" not in cpt:
        raise SystemExit("%s is not an RVC model (expected an inference model with 'weight' and 'config', e.g. "
                         "assets/weights/<name>.pth, or a G_*.pth checkpoint next to its config.json)" % pth)
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    version = cpt.get("version", "v1")
    f0 = int(cpt.get("f0", 1))
    cls = {("v1", 1): models.SynthesizerTrnMs256NSFsid, ("v1", 0): models.SynthesizerTrnMs256NSFsid_nono,
           ("v2", 1): models.SynthesizerTrnMs768NSFsid, ("v2", 0): models.SynthesizerTrnMs768NSFsid_nono}[(version, f0)]
    net = cls(*config, is_half=False)
    if hasattr(net, "enc_q"):
        del net.enc_q
    missing, unexpected = net.load_state_dict(cpt["weight"], strict=False)
    missing = [k for k in missing if not k.startswith("enc_q")]
    if missing:
        raise SystemExit("model weights incomplete, missing: %s" % ", ".join(missing[:8]))
    net.eval()
    net.remove_weight_norm()
    sr = int(config[-1]) if isinstance(config[-1], int) else {"32k": 32000, "40k": 40000, "48k": 48000}[config[-1]]
    info = dict(version=version, f0=f0, sample_rate=sr, speakers=int(config[-3]),
                channels=768 if version == "v2" else 256, info=str(cpt.get("info", "")))
    return net, info


class OnnxGenerator:
    """Builds the export wrapper lazily (torch is imported only when needed)."""

    @staticmethod
    def create(net, f0):
        import torch

        class WithF0(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, phone, phone_lengths, pitch, pitchf, ds, rnd):
                g = self.net.emb_g(ds).unsqueeze(-1)
                m_p, logs_p, x_mask = self.net.enc_p(phone, pitch, phone_lengths)
                z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
                z = self.net.flow(z_p, x_mask, g=g, reverse=True)
                return self.net.dec(z * x_mask, pitchf, g=g)

        class NoF0(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, phone, phone_lengths, ds, rnd):
                g = self.net.emb_g(ds).unsqueeze(-1)
                m_p, logs_p, x_mask = self.net.enc_p(phone, None, phone_lengths)
                z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
                z = self.net.flow(z_p, x_mask, g=g, reverse=True)
                return self.net.dec(z * x_mask, g=g)

        return (WithF0 if f0 else NoF0)(net).eval()


def example_inputs(info, frames, seed=0, voiced=True):
    rng = np.random.default_rng(seed)
    phone = rng.standard_normal((1, frames, info["channels"])).astype(np.float32) * 0.5
    t = np.arange(frames) / 100.0
    pitchf = (220.0 * 2 ** (np.sin(2 * math.pi * 0.7 * t) * 0.5)).astype(np.float32)[None]
    if not voiced:
        pitchf[:, frames // 3: frames // 2] = 0.0
    mel = 1127 * np.log(1 + pitchf / 700)
    lo, hi = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    coarse = np.where(mel > 0, (mel - lo) * 254 / (hi - lo) + 1, 1)
    pitch = np.clip(np.rint(coarse), 1, 255).astype(np.int64)
    feeds = dict(phone=phone, phone_lengths=np.array([frames], np.int64), pitch=pitch, pitchf=pitchf,
                 ds=np.array([0], np.int64), rnd=(rng.standard_normal((1, 192, frames)) * 0.66666).astype(np.float32))
    if not info["f0"]:
        del feeds["pitch"], feeds["pitchf"]
    return feeds


def _export(module, info, path, opset):
    import torch
    feeds = example_inputs(info, 200)
    names = list(feeds)
    axes = {"phone": {1: "frames"}, "pitch": {1: "frames"}, "pitchf": {1: "frames"}, "rnd": {2: "frames"},
            "audio": {2: "samples"}}
    with torch.no_grad():
        torch.onnx.export(module, tuple(torch.from_numpy(feeds[n]) for n in names), path, input_names=names,
                          output_names=["audio"], dynamic_axes={k: v for k, v in axes.items() if k in names + ["audio"]},
                          opset_version=opset, do_constant_folding=True)


def _add_metadata(path, info):
    import onnx
    m = onnx.load(path)
    for k, v in (("sampling_rate", str(info["sample_rate"])), ("rvc_version", info["version"]),
                 ("f0", str(info["f0"])), ("training", info["info"]), ("exporter", "VoxaVersa export_onnx.py")):
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, path)


def export(pth, out, rvc_dir, opset=17, check=True):
    import torch
    torch.set_grad_enabled(False)
    net, info = load_model(pth, rvc_dir)
    print("loaded %s: RVC %s, %s, %d Hz, %d speaker(s), %s" % (os.path.basename(pth), info["version"],
          "pitch-guided" if info["f0"] else "no pitch", info["sample_rate"], info["speakers"], info["info"]))
    module = OnnxGenerator.create(net, info["f0"])
    _export(module, info, out, opset)
    _add_metadata(out, info)
    print("wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))

    if not check:
        return info
    try:
        import onnxruntime as ort
    except ImportError:
        print("note: onnxruntime not installed - skipped the numerical check")
        return info

    # 1. exact check: the same graph exported with the excitation noise switched off in both
    if info["f0"]:
        import tempfile
        from infer.module import models
        original = models.torch.randn_like

        def quiet_randn_like(x, *a, **k):
            return torch.zeros_like(x)
        models.torch.randn_like = quiet_randn_like
        try:
            with tempfile.TemporaryDirectory() as tmp:
                quiet = os.path.join(tmp, "quiet.onnx")
                _export(module, info, quiet, opset)
                sess = ort.InferenceSession(quiet, providers=["CPUExecutionProvider"])
                for frames, voiced in ((200, True), (71, False), (523, False)):
                    feeds = example_inputs(info, frames, seed=frames, voiced=voiced)
                    ref = module(*[torch.from_numpy(v) for v in feeds.values()]).numpy()
                    got = sess.run(["audio"], feeds)[0]
                    if ref.shape != got.shape:
                        raise SystemExit("ONNX check failed: shape %s vs PyTorch %s" % (got.shape, ref.shape))
                    err = float(np.abs(ref - got).max())
                    print("check (noise off) %4d frames: max |onnx - torch| = %.2e (peak %.3f)" % (frames, err, np.abs(ref).max()))
                    if err > 1e-3:
                        raise SystemExit("ONNX check failed: max error %.2e" % err)
        finally:
            models.torch.randn_like = original

    # 2. the real export: right length, finite, sensible level
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    hop = info["sample_rate"] // 100
    for frames in (32, 77, 300):
        feeds = example_inputs(info, frames, seed=1)
        got = sess.run(["audio"], feeds)[0]
        if got.shape[-1] != frames * hop or not np.isfinite(got).all():
            raise SystemExit("ONNX check failed for %d frames: shape %s" % (frames, got.shape))
    print("check: output %d samples per 10 ms frame (%d Hz), finite at 32/77/300 frames - OK" % (hop, info["sample_rate"]))
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pth", required=True, help="RVC inference model (assets/weights/<name>.pth)")
    p.add_argument("--out", required=True, help="output .onnx")
    p.add_argument("--rvc-dir", required=True, help="RVC WebUI checkout (rvc_train.py setup)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args()
    export(args.pth, args.out, args.rvc_dir, args.opset, not args.no_check)


if __name__ == "__main__":
    main()
