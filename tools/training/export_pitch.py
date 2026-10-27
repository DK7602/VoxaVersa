#!/usr/bin/env python3
"""
Exports the neural pitch tracker of VoxaVersa's AI voices (FCPE, CNChTu/FCPE, MIT; the network RVC's
real-time mode uses) to the ONNX format VoxaVersa expects, spectrogram included, so the plug-in only feeds
audio:

    input   audio       [1, N]  float32, 16 kHz mono, N a multiple of 160
    output  f0          [1, T]  float32, Hz of each 10 ms frame (T = N / 160; frame k is samples
                                [160k, 160k + 160)), decoded with FCPE's local argmax
    output  confidence  [1, T]  float32, the network's peak salience (FCPE calls a frame unvoiced at <= 0.006)

The network is convolution-only, so chunks with ~0.3 s of context on each side give the same result as
the whole clip. The weights are torchfcpe's bundled model (fcpe_c_v001.pt). The export is checked against
torchfcpe on real audio.

    python3 tools/training/export_pitch.py --out fcpe.onnx [--check vocal.wav]

Install the result once as <user data>/VoxaVersa/Encoders/fcpe.onnx (Import Voices... does this).
Requirements: torch, torchfcpe (0.0.4 tested), onnx, onnxruntime (for the check), numpy, soundfile, scipy.
"""

import argparse
import math

import numpy as np

SR, HOP, NFFT, MELS = 16000, 160, 1024, 128


def build():
    import torch
    from torchfcpe import spawn_bundled_infer_model

    infer = spawn_bundled_infer_model("cpu")
    net = infer.model.eval()
    mel = infer.wav2mel.mel_extractor if hasattr(infer.wav2mel, "mel_extractor") else infer.wav2mel
    mel_basis = mel.mel_basis.detach().clone() if hasattr(mel, "mel_basis") else None
    if mel_basis is None:
        from librosa.filters import mel as librosa_mel
        mel_basis = torch.tensor(librosa_mel(sr=SR, n_fft=NFFT, n_mels=MELS, fmin=0, fmax=8000)).float()

    class Tracker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            window = torch.hann_window(NFFT)
            k = torch.arange(NFFT // 2 + 1, dtype=torch.float64)[:, None]
            n = torch.arange(NFFT, dtype=torch.float64)[None, :]
            phase = 2.0 * math.pi * k * n / NFFT
            # a strided convolution with the windowed DFT basis is torch.stft (center=False)
            self.register_buffer("cos", (torch.cos(phase) * window).float()[:, None, :])
            self.register_buffer("sin", (-torch.sin(phase) * window).float()[:, None, :])
            self.register_buffer("mel", mel_basis)
            self.net = net

        def forward(self, audio):
            pad = (NFFT - HOP) // 2
            x = torch.nn.functional.pad(audio[:, None, :], (pad, (NFFT - HOP + 1) // 2), mode="reflect")
            re = torch.nn.functional.conv1d(x, self.cos, stride=HOP)
            im = torch.nn.functional.conv1d(x, self.sin, stride=HOP)
            spec = torch.sqrt(re * re + im * im + 1e-9)
            logmel = torch.log(torch.clamp(torch.matmul(self.mel, spec), min=1e-5)).transpose(1, 2)
            latent = self.net(logmel)                                       # [1, T, 360], sigmoid salience
            confidence, index = torch.max(latent, dim=-1, keepdim=True)
            local = torch.clamp(torch.arange(9, device=index.device) + (index - 4), 0, latent.shape[-1] - 1)
            cents_table = self.net.cent_table[None, None, :].expand(latent.shape[0], latent.shape[1], -1)
            weights = torch.gather(latent, -1, local)
            cents = torch.sum(torch.gather(cents_table, -1, local) * weights, -1) / torch.sum(weights, -1)
            return 10.0 * torch.pow(2.0, cents / 1200.0), confidence[..., 0]

    return Tracker().eval(), infer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fcpe.onnx")
    ap.add_argument("--check", help="a vocal recording to compare ONNX Runtime with torchfcpe on")
    args = ap.parse_args()

    import torch
    tracker, infer = build()
    dummy = torch.zeros(1, HOP * 200)
    torch.onnx.export(tracker, (dummy,), args.out, input_names=["audio"], output_names=["f0", "confidence"],
                      dynamic_axes={"audio": {1: "samples"}, "f0": {1: "frames"}, "confidence": {1: "frames"}},
                      opset_version=17)
    import onnx
    m = onnx.load(args.out)
    for key, value in {"voxaversa_role": "pitch", "sample_rate": str(SR), "hop": str(HOP),
                       "source": "FCPE (CNChTu/FCPE, MIT), torchfcpe bundled fcpe_c_v001"}.items():
        p = m.metadata_props.add(); p.key, p.value = key, value
    onnx.save(m, args.out)
    print("wrote", args.out)

    if args.check:
        import onnxruntime as ort
        import soundfile as sf
        from scipy import signal
        x, sr = sf.read(args.check, dtype="float32", always_2d=True)
        x = signal.resample_poly(x.mean(1), SR, sr).astype(np.float32)
        x = x[: len(x) // HOP * HOP]
        with torch.no_grad():
            ref = infer(torch.from_numpy(x)[None, :, None], sr=SR, decoder_mode="local_argmax", threshold=0.006)
        ref = ref[0, :, 0].numpy()
        f0, conf = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(None, {"audio": x[None]})
        f0 = np.where(conf[0] > 0.006, f0[0], 0.0)
        n = min(len(ref), len(f0))
        both = (ref[:n] > 0) & (f0[:n] > 0)
        cents = np.abs(1200 * np.log2(f0[:n][both] / ref[:n][both]))
        print("frames %d/%d, voicing agrees %.2f%%, pitch error max %.2f cents"
              % (len(f0), len(ref), 100 * np.mean((ref[:n] > 0) == (f0[:n] > 0)), cents.max() if len(cents) else 0))
        if np.mean((ref[:n] > 0) == (f0[:n] > 0)) < 0.995 or (len(cents) and np.percentile(cents, 99) > 1.0):
            raise SystemExit("ONNX export does not match torchfcpe")


if __name__ == "__main__":
    main()
