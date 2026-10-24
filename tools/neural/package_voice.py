#!/usr/bin/env python3
"""
Builds a VoxaVersa voice pack (.vxpack) from an RVC voice model exported to ONNX.

    python3 tools/neural/package_voice.py --model MyVoice.onnx --name "My Voice" \\
        --gender female --genre Pop --styles "Breathy,Bright" --author "Me" \\
        --i-have-the-rights

The model must be an RVC v1/v2 generator exported with the RVC WebUI's "Export ONNX" feature (inputs
phone, phone_lengths, pitch, pitchf, ds, rnd; output audio). If the `onnx` Python package is installed
(`pip install onnx`) the model's inputs/outputs are validated and the feature size (v1: 256, v2: 768),
the matching content encoder and - for exports that carry it - the sample rate are detected.

The pack is a zip archive containing voice.json, the model and optionally a content encoder and an
avatar image. Import it in VoxaVersa via Settings > Import voice pack. See docs/NEURAL_ENGINE.md.

Only package voices you own or have explicit permission from the speaker/singer to use.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import zipfile

RVC_INPUTS = {"phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"}
OPTIONAL_RVC_INPUTS = {"pitch", "pitchf"}
ENCODER_FOR_CHANNELS = {768: "contentvec-768", 256: "contentvec-256"}


def fail(message):
    print("error: " + message, file=sys.stderr)
    sys.exit(1)


def _dims(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None)
    return dims


def inspect_generator(path):
    """Returns a dict with the detected properties, or None if the onnx package is not available."""
    try:
        import onnx
    except ImportError:
        return None

    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = {i.name: i for i in graph.input if i.name not in initializers}
    outputs = [o.name for o in graph.output]

    missing = (RVC_INPUTS - OPTIONAL_RVC_INPUTS) - set(inputs)
    if missing:
        fail("%s is not an RVC ONNX export: missing input(s) %s (found %s)"
             % (path, ", ".join(sorted(missing)), ", ".join(sorted(inputs))))

    unknown = set(inputs) - RVC_INPUTS
    if unknown:
        fail("unsupported model input(s): " + ", ".join(sorted(unknown)))

    has_pitch = "pitch" in inputs
    if has_pitch != ("pitchf" in inputs):
        fail("the model must have both 'pitch' and 'pitchf' inputs, or neither")

    if not outputs:
        fail("the model has no outputs")

    phone_dims = _dims(inputs["phone"])
    if len(phone_dims) != 3:
        fail("input 'phone' must have 3 dimensions [1, frames, channels], found %s" % phone_dims)
    channels = phone_dims[2]

    rnd_dims = _dims(inputs["rnd"])
    if len(rnd_dims) != 3:
        fail("input 'rnd' must have 3 dimensions [1, 192, frames], found %s" % rnd_dims)

    sample_rate = None
    for prop in model.metadata_props:
        if prop.key in ("samplingRate", "sampling_rate", "sample_rate", "sr"):
            try:
                sample_rate = int(prop.value)
            except ValueError:
                pass
        elif prop.key == "metadata":
            try:
                meta = json.loads(prop.value)
                for key in ("samplingRate", "sampling_rate", "sample_rate", "sr"):
                    if key in meta:
                        sample_rate = int(meta[key])
            except (ValueError, TypeError):
                pass

    return {"channels": channels, "f0": has_pitch, "sample_rate": sample_rate, "outputs": outputs}


def inspect_encoder(path, expected_channels):
    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(path, load_external_data=False)
    initializers = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    if not inputs or len(_dims(inputs[0])) not in (2, 3):
        fail("%s does not look like a content encoder (expected a waveform input [1,1,N] or [1,N])" % path)
    out_dims = _dims(model.graph.output[0])
    if len(out_dims) != 3:
        fail("%s: expected a [1, frames, channels] output, found %s" % (path, out_dims))
    if expected_channels and expected_channels not in out_dims[1:]:
        found = [d for d in out_dims[1:] if d]
        if found:
            fail("%s produces %s-dimensional features but the voice model expects %d"
                 % (path, found[0], expected_channels))


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "voice"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="RVC generator exported to ONNX")
    p.add_argument("--name", required=True, help="voice name shown in VoxaVersa")
    p.add_argument("--out", help="output .vxpack file (default: <name>.vxpack)")
    p.add_argument("--id", help="stable identifier (default: derived from the name)")
    p.add_argument("--sample-rate", type=int, help="model output rate: 32000, 40000 or 48000 (detected when possible)")
    p.add_argument("--encoder", help="content encoder id (default: contentvec-768 for v2, contentvec-256 for v1 models)")
    p.add_argument("--encoder-file", help="bundle this content encoder ONNX inside the pack")
    p.add_argument("--gender", choices=["male", "female", "neutral"], default="neutral")
    p.add_argument("--genre", default="Custom")
    p.add_argument("--styles", default="", help="comma separated tags, e.g. 'Breathy,Powerful'")
    p.add_argument("--description", default="")
    p.add_argument("--author", default="")
    p.add_argument("--license", default="", help="terms under which the voice may be used")
    p.add_argument("--range", nargs=2, type=int, metavar=("LOW", "HIGH"), help="comfortable MIDI note range")
    p.add_argument("--formant", type=float, help="DSP formant hint in semitones (male ~ -1, female ~ +1)")
    p.add_argument("--colour", help="accent colour, e.g. '#ff3a7b'")
    p.add_argument("--avatar", help="square PNG/JPEG image")
    p.add_argument("--speaker-id", type=int, default=0, help="speaker index for multi-speaker models")
    p.add_argument("--hop-size", type=int, help="generator samples per 10 ms frame (normally sample-rate / 100)")
    p.add_argument("--noise-scale", type=float, help="scale of the generator noise input (default 0.66666)")
    p.add_argument("--stream-context-ms", type=int, help="real-time left context per chunk (default 300)")
    p.add_argument("--no-validate", action="store_true", help="skip the ONNX validation")
    p.add_argument("--i-have-the-rights", action="store_true",
                   help="confirm that you own this voice or have the speaker's permission to use and share it")
    args = p.parse_args()

    if not args.i_have_the_rights:
        fail("voice packs may only be made from recordings you own or have explicit permission to use.\n"
             "Re-run with --i-have-the-rights to confirm this applies to '%s'." % args.name)

    for path in (args.model, args.encoder_file, args.avatar):
        if path and not os.path.isfile(path):
            fail("file not found: " + path)

    info = None if args.no_validate else inspect_generator(args.model)
    if info is None and not args.no_validate:
        print("note: the 'onnx' package is not installed - skipping model validation (pip install onnx)")

    channels = info["channels"] if info else None
    sample_rate = args.sample_rate or (info["sample_rate"] if info else None)
    if not sample_rate:
        fail("could not detect the model's sample rate - pass --sample-rate (32000, 40000 or 48000)")
    if sample_rate not in (32000, 40000, 48000):
        print("warning: unusual sample rate %d for an RVC model" % sample_rate)

    encoder = args.encoder or ENCODER_FOR_CHANNELS.get(channels, "contentvec-768")
    if channels and args.encoder and args.encoder in ENCODER_FOR_CHANNELS.values() \
            and ENCODER_FOR_CHANNELS.get(channels) not in (None, args.encoder):
        fail("the model uses %d-dimensional features; encoder '%s' does not match" % (channels, args.encoder))

    if args.encoder_file and not args.no_validate:
        inspect_encoder(args.encoder_file, channels)

    model_name = os.path.basename(args.model)
    model_section = {"file": model_name, "encoder": encoder, "sampleRate": sample_rate}
    if args.encoder_file:
        model_section["encoderFile"] = os.path.basename(args.encoder_file)
    if args.hop_size:
        model_section["hopSize"] = args.hop_size
    if args.speaker_id:
        model_section["speakerId"] = args.speaker_id
    if args.noise_scale is not None:
        model_section["noiseScale"] = args.noise_scale
    if args.stream_context_ms:
        model_section["streamContextMs"] = args.stream_context_ms

    voice = {
        "name": args.name,
        "id": args.id or slug(args.name),
        "description": args.description or "AI voice",
        "gender": args.gender,
        "genre": args.genre,
        "styles": [s.strip() for s in args.styles.split(",") if s.strip()],
        "author": args.author,
        "license": args.license,
        "model": model_section,
    }
    if args.range:
        voice["range"] = list(args.range)
    if args.formant is not None:
        voice["formant"] = args.formant
    if args.colour:
        voice["colour"] = args.colour
    if args.avatar:
        voice["avatar"] = "avatar" + os.path.splitext(args.avatar)[1].lower()

    out = args.out or (slug(args.name) + ".vxpack")
    with tempfile.TemporaryDirectory() as tmp:
        manifest = os.path.join(tmp, "voice.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(voice, f, indent=2, ensure_ascii=False)

        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            z.write(manifest, "voice.json")
            z.write(args.model, model_name)
            if args.encoder_file:
                z.write(args.encoder_file, os.path.basename(args.encoder_file))
            if args.avatar:
                z.write(args.avatar, voice["avatar"])

    kind = "unvalidated" if info is None else "RVC %s, %s, %d-d features" % (
        "v2" if channels == 768 else "v1" if channels == 256 else "?",
        "pitch-guided" if info["f0"] else "no pitch input", channels or 0)
    print("Wrote %s (%s, %d Hz, encoder %s)" % (out, kind, sample_rate, model_section.get("encoderFile", encoder)))


if __name__ == "__main__":
    main()
