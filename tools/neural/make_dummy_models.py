#!/usr/bin/env python3
"""
Builds tiny, deterministic ONNX models that mimic the I/O signatures of a ContentVec/HuBERT content
encoder and of an RVC v2 generator ("Export ONNX" from the RVC WebUI). VoxaVersa's unit tests use them
to verify the neural engine end to end without shipping real voice models.

    python3 -m venv .venv && .venv/bin/pip install onnx numpy
    .venv/bin/python tools/neural/make_dummy_models.py --out /tmp/voxaversa-dummy-models

    VOXAVERSA_TEST_ORT_LIB=/path/to/libonnxruntime.so \
    VOXAVERSA_TEST_MODELS_DIR=/tmp/voxaversa-dummy-models \
        ./build/VoxaVersaTests_artefacts/Release/VoxaVersaTests Neural

What the dummy models compute (so tests can check the output analytically):

  encoder_*.onnx   |x| -> Conv1d(kernel 400, stride 320, C channels, gain g_c) i.e. a strided moving
                   average of the rectified input, exactly the framing of HuBERT's feature extractor
                   (T = floor((N - 400) / 320) + 1 frames of 20 ms).
                   Input "source" [1,1,N] (or [1,N] for the rank-2 variant).
                   Output "embed" [1,T,C] (or [1,C,T] for the channels-first variant).

  generator_*.onnx amplitude a[t] = mean_c(phone[t, c]) * k  (k chosen so a sine input of amplitude A
                   yields amplitude ~A), voiced[t] = (pitch[t] > 1) i.e. gated by the *coarse* pitch,
                   y = a * voiced * sin(2 pi cumsum(pitchf) / sr), upsampled by `hop` samples per frame.
                   Inputs phone [1,T,C], phone_lengths [1], pitch [1,T] (int64), pitchf [1,T],
                   ds [1] (int64), rnd [1,192,T]; output "audio" [1,1,T*hop].
                   The "nof0" variant has no pitch inputs and sings a fixed 220 Hz tone.
                   The "fp16" variant takes/returns float16 tensors (as some exporters produce).

A pitch tracker run on the converted output therefore recovers the *target* f0 passed to the engine,
which proves that features, pitch, pitchf and frame timing are threaded through correctly.
"""

import argparse
import json
import math
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17          # ONNX Runtime >= 1.16 supports opset 19
IR_VERSION = 8      # loadable by every ONNX Runtime release since 1.10
WINDOW = 400        # HuBERT/ContentVec receptive field (samples @ 16 kHz)
HOP16K = 320        # 20 ms


def _gains(channels):
    return (1.0 + 0.5 * np.arange(channels, dtype=np.float64) / channels).astype(np.float32)


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="voxaversa-dummy")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_encoder(path, channels=768, channels_first=False, input_rank=3):
    if input_rank == 3:
        source = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 1, "samples"])
    else:
        source = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, "samples"])

    out_shape = [1, channels, "frames"] if channels_first else [1, "frames", channels]
    embed = helper.make_tensor_value_info("embed", TensorProto.FLOAT, out_shape)

    weight = (np.ones((channels, 1, WINDOW), dtype=np.float32) / WINDOW) * _gains(channels)[:, None, None]
    inits = [numpy_helper.from_array(weight, "conv_w")]

    nodes = []
    x = "source"
    if input_rank == 2:
        inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"))
        nodes.append(helper.make_node("Unsqueeze", [x, "axis1"], ["source3"]))
        x = "source3"
    nodes.append(helper.make_node("Abs", [x], ["rect"]))
    nodes.append(helper.make_node("Conv", ["rect", "conv_w"], ["conv"], kernel_shape=[WINDOW], strides=[HOP16K]))
    if channels_first:
        nodes.append(helper.make_node("Identity", ["conv"], ["embed"]))
    else:
        nodes.append(helper.make_node("Transpose", ["conv"], ["embed"], perm=[0, 2, 1]))

    _save(helper.make_graph(nodes, "dummy_content_encoder", [source], [embed], inits), path)


def make_generator(path, sample_rate=40000, channels=768, with_f0=True, fp16=False):
    hop = sample_rate // 100
    ftype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT

    inputs = [helper.make_tensor_value_info("phone", ftype, [1, "frames", channels]),
              helper.make_tensor_value_info("phone_lengths", TensorProto.INT64, [1])]
    if with_f0:
        inputs += [helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "frames"]),
                   helper.make_tensor_value_info("pitchf", ftype, [1, "frames"])]
    inputs += [helper.make_tensor_value_info("ds", TensorProto.INT64, [1]),
               helper.make_tensor_value_info("rnd", ftype, [1, 192, "frames"])]
    audio = helper.make_tensor_value_info("audio", ftype, [1, 1, "samples"])

    # a sine of amplitude A has mean |x| = 2A/pi; the encoder multiplies by mean(gains)
    amp_scale = (math.pi / 2.0) / float(np.mean(_gains(channels).astype(np.float64)))

    inits = [numpy_helper.from_array(np.array(amp_scale, dtype=np.float32), "amp_scale"),
             numpy_helper.from_array(np.array(1.0 / sample_rate, dtype=np.float32), "inv_sr"),
             numpy_helper.from_array(np.array(2.0 * math.pi, dtype=np.float32), "two_pi"),
             numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
             numpy_helper.from_array(np.array(1, dtype=np.int64), "time_axis"),
             numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2"),
             numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
             numpy_helper.from_array(np.array([1, 1, hop], dtype=np.int64), "expand_shape"),
             numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "flat_shape")]

    nodes = []

    def as_float(name):
        if not fp16:
            return name
        nodes.append(helper.make_node("Cast", [name], [name + "_f32"], to=TensorProto.FLOAT))
        return name + "_f32"

    def upsample(name, out):
        nodes.append(helper.make_node("Unsqueeze", [name, "axis2"], [out + "_u"]))         # [1,T,1]
        nodes.append(helper.make_node("Expand", [out + "_u", "expand_shape"], [out + "_e"]))  # [1,T,hop]
        nodes.append(helper.make_node("Reshape", [out + "_e", "flat_shape"], [out]))        # [1,T*hop]

    phone = as_float("phone")
    rnd = as_float("rnd")
    nodes.append(helper.make_node("ReduceMean", [phone], ["amp_raw"], axes=[2], keepdims=0))   # [1,T]
    nodes.append(helper.make_node("Mul", ["amp_raw", "amp_scale"], ["amp"]))

    if with_f0:
        inits.append(numpy_helper.from_array(np.array(1, dtype=np.int64), "one_i64"))
        pitchf = as_float("pitchf")
        nodes.append(helper.make_node("Greater", ["pitch", "one_i64"], ["voiced_b"]))
        nodes.append(helper.make_node("Cast", ["voiced_b"], ["voiced"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Mul", ["amp", "voiced"], ["amp_gated"]))
        upsample(pitchf, "f0_up")
    else:
        inits.append(numpy_helper.from_array(np.array(220.0, dtype=np.float32), "fixed_f0"))
        nodes.append(helper.make_node("Identity", ["amp"], ["amp_gated"]))
        nodes.append(helper.make_node("Mul", ["amp", "zero"], ["amp_zero"]))
        nodes.append(helper.make_node("Add", ["amp_zero", "fixed_f0"], ["f0_frames"]))
        upsample("f0_frames", "f0_up")

    upsample("amp_gated", "amp_up")
    nodes.append(helper.make_node("Mul", ["f0_up", "inv_sr"], ["cycles_step"]))
    nodes.append(helper.make_node("CumSum", ["cycles_step", "time_axis"], ["cycles"]))
    nodes.append(helper.make_node("Mul", ["cycles", "two_pi"], ["phase"]))
    nodes.append(helper.make_node("Sin", ["phase"], ["sine"]))
    nodes.append(helper.make_node("Mul", ["sine", "amp_up"], ["voice"]))

    # Consume the remaining inputs (multiplied by zero) so they stay part of the graph.
    nodes.append(helper.make_node("Cast", ["phone_lengths"], ["len_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", ["ds"], ["ds_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceMean", [rnd], ["rnd_mean"], keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["len_f"], ["len_sum"], keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["ds_f"], ["ds_sum"], keepdims=0))
    nodes.append(helper.make_node("Add", ["len_sum", "ds_sum"], ["aux0"]))
    nodes.append(helper.make_node("Add", ["aux0", "rnd_mean"], ["aux1"]))
    nodes.append(helper.make_node("Mul", ["aux1", "zero"], ["aux"]))
    nodes.append(helper.make_node("Add", ["voice", "aux"], ["voice_out"]))

    if fp16:
        nodes.append(helper.make_node("Unsqueeze", ["voice_out", "axis1"], ["audio_f32"]))
        nodes.append(helper.make_node("Cast", ["audio_f32"], ["audio"], to=TensorProto.FLOAT16))
    else:
        nodes.append(helper.make_node("Unsqueeze", ["voice_out", "axis1"], ["audio"]))

    graph = helper.make_graph(nodes, "dummy_rvc_generator", inputs, [audio], inits)
    _save(graph, path)

    # RVC exports made with w-okada's voice changer carry their parameters as metadata; mimic that.
    model = onnx.load(path)
    meta = {"application": "VoxaVersa dummy", "samplingRate": sample_rate, "f0": with_f0,
            "embChannels": channels}
    entry = model.metadata_props.add()
    entry.key = "metadata"
    entry.value = json.dumps(meta)
    onnx.save(model, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    make_encoder(os.path.join(out, "encoder_768.onnx"), 768)
    make_encoder(os.path.join(out, "encoder_768_cf_rank2.onnx"), 768, channels_first=True, input_rank=2)
    make_encoder(os.path.join(out, "encoder_256.onnx"), 256)
    make_generator(os.path.join(out, "generator_40k.onnx"), 40000, 768, with_f0=True)
    make_generator(os.path.join(out, "generator_48k_nof0.onnx"), 48000, 768, with_f0=False)
    make_generator(os.path.join(out, "generator_32k_fp16.onnx"), 32000, 768, with_f0=True, fp16=True)

    voice = {
        "name": "Dummy Neural Voice",
        "id": "dummy-neural",
        "description": "Test voice built by tools/neural/make_dummy_models.py",
        "gender": "female",
        "genre": "Test",
        "author": "VoxaVersa tests",
        "license": "CC0",
        "model": {
            "file": "generator_40k.onnx",
            "encoder": "dummy-768",
            "encoderFile": "encoder_768.onnx",
            "sampleRate": 40000,
        },
    }
    with open(os.path.join(out, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(voice, f, indent=2)

    print("Dummy models written to", out)


if __name__ == "__main__":
    main()
