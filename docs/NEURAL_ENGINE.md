# VoxaVersa AI voice engine

VoxaVersa can transform a voice with an **AI voice model** instead of (or in addition to) its DSP voice
engine. The AI engine runs voice models in the format of the open-source
[RVC (Retrieval-based Voice Conversion) WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI),
the de-facto standard for singing voice conversion, exported to ONNX and executed with
[ONNX Runtime](https://onnxruntime.ai).

The engine is optional:

* VoxaVersa does **not** link against ONNX Runtime. The runtime is a separate shared library that is
  loaded when it is first needed. If it is missing (or too old), AI voices are unavailable, the UI shows
  why, and everything else keeps working with the DSP engine.
* Voices with an AI model are ordinary voice packs (`voice.json` + files) with an extra `"model"` section.

Contents

1. [Architecture](#1-architecture)
2. [Latency and CPU](#2-latency-and-cpu)
3. [Installing ONNX Runtime](#3-installing-onnx-runtime)
4. [Creating a voice pack](#4-creating-a-voice-pack)
5. [voice.json reference](#5-voicejson-reference)
6. [Integration notes (for developers)](#6-integration-notes-for-developers)
7. [Testing](#7-testing)
8. [Known limitations](#8-known-limitations)

---

## 1. Architecture

```
                 host rate                        16 kHz                          model rate (32/40/48 kHz)        host rate
 input voice ──► band-limited ──► content encoder (ContentVec/HuBERT) ──► features ×2 ──► RVC generator ──► band-limited ──► output
 target f0  ──────────────────────── per 10 ms frame: pitchf (Hz) + coarse pitch (1..255) ──┘                 resampler
```

| Piece | Source | Role |
|---|---|---|
| `OnnxRuntime` | `Source/neural/NeuralRuntime.*`, `NeuralPlatform.*` | Finds and loads the ONNX Runtime library at run time, requests C API **version 16** (works with any ONNX Runtime ≥ 1.16), creates one process-wide environment with a shared, non-spinning thread pool. |
| `NeuralVoiceModel` | `Source/neural/RvcModel.*` | A content encoder + RVC generator pair loaded from a voice pack. Input/output names, ranks, dimensions and element types are discovered from the models. Immutable after loading; can be shared between threads and plugin instances. |
| `NeuralVoiceStream` | `Source/neural/NeuralVoice.cpp` | Real-time conversion with a fixed latency (worker thread + lock-free rings). |
| `NeuralVoiceModel::convert` | `Source/neural/RvcModel.cpp` | Offline conversion of whole clips (Text-to-Sing takes, exports) at best quality. |
| DSP helpers | `Source/neural/NeuralDsp.*` | Windowed-sinc resampler, coarse pitch, f0 framing, SOLA, crossfades, deterministic noise. |
| C API header | `Source/neural/third_party/onnxruntime/` | Unmodified `onnxruntime_c_api.h` from ONNX Runtime v1.20.1 (MIT). |

### The RVC pipeline

1. **Content encoder** (ContentVec / HuBERT exported to ONNX, e.g. `vec-768-layer-12.onnx` for RVC v2,
   `vec-256-layer-9.onnx` for v1). Input: 16 kHz mono waveform, `[1,1,N]` or `[1,N]` (an optional padding
   mask input is fed with zeros). Output: one feature vector per 20 ms, `[1,T,C]` or `[1,C,T]` with
   C = 768 (v2) or 256 (v1). The layout is detected from the model's static dimensions, or from the
   actual output shape if they are dynamic. The input is band-limited to 16 kHz with a Kaiser-windowed
   sinc filter and padded by 80 samples so that HuBERT's 400-sample receptive field yields exactly
   N / 320 frames.
2. Each feature vector is used twice (20 ms → 10 ms frames), as in RVC.
3. **Generator** (the RVC WebUI's ONNX export): inputs `phone [1,T,C]`, `phone_lengths [1]`,
   `pitch [1,T]` (int64, coarse), `pitchf [1,T]` (Hz), `ds [1]` (speaker id), `rnd [1,192,T]` (noise);
   output `audio [1,1,T·hop]`.
   * `pitchf` is the **target pitch** supplied by VoxaVersa (pitch correction, harmonies, MIDI or a
     Text-to-Sing melody) — the engine never estimates pitch itself.
   * Coarse pitch: `mel = 1127·ln(1 + f0/700)`, `coarse = (mel − mel(50)) · 254 / (mel(1100) − mel(50)) + 1`,
     rounded and clipped to 1..255, unvoiced = 1 (identical to RVC).
   * `rnd` is Gaussian noise scaled by `noiseScale` (default 0.66666 like RVC's PyTorch inference). It is
     generated deterministically from the absolute frame index, so overlapping chunks see the same noise.
   * Models without pitch inputs (v1 "nof0" exports) are supported; float16 exports are converted
     transparently; int32 index inputs are accepted too.
4. The generator output (hop = sample rate / 100 samples per frame, verified by a test run at load time)
   is resampled to the host rate with the same band-limited resampler.

### Real-time stream

* The audio thread writes input samples and per-sample target f0 into lock-free ring buffers indexed by
  absolute sample position, and reads already-converted output from a third ring. It never allocates,
  locks or waits (one exception: see *offline bounces* below).
* A worker thread converts **chunks of 0.4 s of new audio** with **1 s of left context**
  (`streamContextMs`) plus a 90 ms tail: the content encoder sees all 1.5 s, the (much more expensive)
  generator only the last 0.69 s (chunk + tail + 200 ms of its own context), as in RVC's real-time mode.
  Measured on a trained voice, 0.3 s of context made the encoder's view of each chunk differ clearly from
  the whole-file view and cost speaker similarity (0.642 vs 0.669 offline); 1 s recovers most of it
  (0.656) and 2 s matches offline (0.672).
  That full context is always used while the host renders offline. **Live** playback uses the longest
  context this computer converts comfortably in real time: when the model loads, chunks with 1 s, 0.6 s,
  0.3 s and 0.15 s of context are timed, and live playback starts with the first one that needs at most
  half of the 200 ms budget. If chunks still keep missing their deadline (3 within 8 s, e.g. a busier
  session), the stream continues with the next shorter one. All shapes are primed at load time, so
  ONNX Runtime reuses its memory plan for each. The generator's cost does not depend on the context, so
  this helps less than it might seem (on a 4-core 2.8 GHz cloud VM the generator takes ~75% of a chunk).
* Consecutive chunks are joined with **SOLA**: the start of the new chunk is searched within 12.5 ms
  (≥ one period of an 80 Hz voice) for the best match with the previous chunk's tail, then crossfaded over
  60 ms with gains normalised for the signals' correlation (aligned → amplitude-complementary,
  unrelated → equal power). This avoids the phase cancellation you get from naively crossfading two
  independently generated waveforms. Exactly 0.4 s is emitted per chunk, so the stream never drifts.
* **Silence detection**: chunks whose input is below −60 dBFS RMS / −40 dBFS peak skip inference.
* If a chunk would be finished too late, it is skipped instead of shifting the output, so the latency
  reported to the host always stays valid; `isRunningInRealTime()` then returns false and
  `getNumMissedChunks()` counts it (the editor then says once a minute that exports are complete).
* **No gaps:** `process()` also returns a per-sample *coverage* (0..1): 1 where converted voice
  continues, 0 where there is none (warming up, after `reset()`, a skipped chunk), and the complement of
  the crossfade gains where the voice fades in from or out to silence. A chunk that arrives partly too
  late, or output that stops at a missing chunk, gets a 5 ms ramp instead of a hard edge. `VoiceChain`
  mixes its DSP lead (aligned to the same latency) in with gain `1 − coverage`, so a computer that
  can't keep up live plays short stretches of the DSP voice instead of dropping out.
* `setModel()` hands the model to the worker, which swaps it between chunks (crossfading from the old
  voice to the new one) and releases the old model on the worker thread.
* `reset()` (audio thread) only bumps a counter; the worker forgets the buffered audio and the next
  latency-worth of output is silence, like a delay line being cleared.
* **Offline renders**: when the host says it renders offline (`setNonRealtime (true)`, i.e. the
  plug-in's `isNonRealtime()`: exports, Render in Place, freeze) or calls the plugin much faster than real
  time (more than 3× real time, measured excluding the time spent waiting), the stream waits for the
  worker instead of dropping chunks (up to 10 s per block when the host says so), so renders are complete
  and at full quality even on a computer that is too slow for live playback. In real time this never
  happens.

### Offline conversion

`NeuralVoiceModel::convert` resamples the clip to 16 kHz, interpolates the target f0 contour onto 10 ms
frames (geometric interpolation between voiced frames, nearest value at voiced/unvoiced boundaries) and
converts it in segments of ~10 s (8.5 – 13 s) with 1 s of real audio as context on both sides. Segment
boundaries are placed at the quietest point within ±1.5 s, each segment is aligned to its predecessor
with SOLA (bounded to ±6 ms around its nominal position, so timing never accumulates errors) and joined
with a 50 ms crossfade. The cancel flag is checked between segments; ONNX Runtime's memory arena is
shrunk after each big inference.

---

## 2. Latency and CPU

The stream's latency depends only on the host sample rate and maximum block size — never on the model —
so the host's delay compensation stays valid when the voice changes:

```
latency = chunk (0.4 s) + crossfade (60 ms) + SOLA search / 2 (6.25 ms) + resampler lookahead (~1 ms)
        + one host block + inference headroom (200 ms) + worker polling interval (2 ms)
```

| Host rate | Block 128 | Block 512 | Block 1024 |
|---|---|---|---|
| 44.1 kHz | 29 648 samples (672 ms) | 30 032 samples (681 ms) | 30 544 samples (693 ms) |
| 48 kHz | 32 259 samples (672 ms) | 32 643 samples (680 ms) | 33 155 samples (691 ms) |

This is typical for RVC-style real-time conversion; the AI engine is meant for tracking/monitoring with
latency compensation and for rendering, not for zero-latency live monitoring (the DSP engine covers that).

**CPU budget.** Every 0.4 s the worker runs the encoder on 1.5 s and the generator on 0.69 s of audio
and must finish within the 200 ms headroom. When a model is loaded it is timed on this computer;
`NeuralVoiceModel::getDescription()` then ends with "real-time load NN%" (time per chunk relative to the
200 ms budget), "live context NNN ms" when live playback needs a shorter context than configured, and
"too slow for live playback on this computer, exports are complete" above 80%. For reference, the community
`vec-768-layer-12.onnx` ContentVec encoder alone takes about 80 ms per chunk with 2 threads on a 2.1 GHz
Xeon cloud VM (≈ 0.08× real time offline); the RVC generator adds to that, so a full-size RVC v2 model
needs a recent desktop-class CPU with 4+ cores for real-time use. On slower machines use a GPU
accelerator (below) or offline rendering, which has no time limit. A longer `streamContextMs` mainly
costs encoder time. ONNX Runtime uses one shared thread pool for all plugin instances (`min(4, physical cores / 2)`
threads by default, without busy-waiting) so the DAW's own threads are not starved.

---

## 3. Installing ONNX Runtime

Download an official release (**1.16 or newer**; 1.20.x recommended) from
<https://github.com/microsoft/onnxruntime/releases> and put the library where VoxaVersa looks for it.

### Where VoxaVersa looks (in this order)

1. The file or folder in the environment variable `VOXAVERSA_ONNXRUNTIME_PATH` (for testing).
2. Next to the plugin / app binary — e.g. `VoxaVersa.vst3/Contents/x86_64-win/`, `.../x86_64-linux/`,
   the `.lv2` folder, the standalone's folder. On macOS also the bundle's `Contents/Frameworks` and
   `Contents/Resources`.
3. VoxaVersa's user data folder: `<user data>/Runtime` (and `<user data>/Runtime/lib`):
   * Windows: `%APPDATA%\VoxaVersa\Runtime`
   * macOS: `~/Library/Application Support/VoxaVersa/Runtime`
   * Linux: `~/.config/VoxaVersa/Runtime`
4. Common install prefixes: `/opt/homebrew/lib`, `/usr/local/lib` (macOS); `/usr/local/lib`,
   `/opt/onnxruntime/lib` (Linux).
5. The system loader's default search (`PATH`, `LD_LIBRARY_PATH`, `ld.so.cache`, `DYLD_*` …).

File names: `onnxruntime.dll`; `libonnxruntime.dylib` or versioned `libonnxruntime.1.20.1.dylib`;
`libonnxruntime.so`, `libonnxruntime.so.1` or `libonnxruntime.so.1.20.1`.

The status shown in the UI is, for example, *"AI engine ready (ONNX Runtime 1.20.1, CPU)"*,
*"ONNX Runtime not found — install ONNX Runtime 1.16 or newer by copying libonnxruntime.so into
~/.config/VoxaVersa/Runtime"* or *"ONNX Runtime 1.15.1 at … is too old (VoxaVersa needs 1.16 or newer)"*.

### Windows

1. Download `onnxruntime-win-x64-1.20.1.zip` (or `-arm64-`).
2. Copy `lib\onnxruntime.dll` to `%APPDATA%\VoxaVersa\Runtime\`.

Note: some Windows versions ship an old `onnxruntime.dll` in `System32` for Windows ML. Because the
folders above are searched first, it is only used if nothing else is found (and is rejected if older
than 1.16).

DirectML (GPU, any DirectX 12 card): take `onnxruntime.dll` **and** `DirectML.dll` from the
`Microsoft.ML.OnnxRuntime.DirectML` NuGet package (`runtimes\win-x64\native\`; a `.nupkg` is a zip file)
and set the accelerator to `directml` (see below).

### macOS

1. Download `onnxruntime-osx-universal2-1.20.1.tgz` and extract it.
2. Copy `lib/libonnxruntime.1.20.1.dylib` to `~/Library/Application Support/VoxaVersa/Runtime/`.
3. Files downloaded with a browser are quarantined; clear the flag:
   `xattr -d com.apple.quarantine ~/Library/Application\ Support/VoxaVersa/Runtime/*.dylib`

Alternatively `brew install onnxruntime` (found in `/opt/homebrew/lib`). The macOS builds include the
CoreML execution provider (accelerator `coreml`).

A standalone app signed with the hardened runtime can only load a user-installed dylib if it has the
`com.apple.security.cs.disable-library-validation` entitlement; otherwise ship ONNX Runtime inside the
bundle (see *Bundling*). DAWs that load third-party plugins allow this already.

### Linux

1. Download `onnxruntime-linux-x64-1.20.1.tgz` (or `-aarch64-`; `-gpu-` for CUDA 12).
2. `mkdir -p ~/.config/VoxaVersa/Runtime && cp -P onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so* ~/.config/VoxaVersa/Runtime/`

### Accelerators and threads

The CPU is used by default. To try a GPU execution provider (when the installed runtime build includes
it), set one of `cpu`, `auto`, `cuda`, `directml`, `coreml`:

* environment variable `VOXAVERSA_AI_ACCELERATOR=auto`, or
* in `<user data>/settings.xml`: `<VoxaVersaSettings> <NeuralEngine accelerator="auto" threads="3"/> … </VoxaVersaSettings>`

`auto` picks CoreML on macOS, DirectML then CUDA on Windows, CUDA on Linux. If the provider can't be
used for a model, VoxaVersa silently falls back to the CPU. `threads` (or `VOXAVERSA_AI_THREADS`) sets the
size of the shared inference thread pool (default `min(4, physical cores / 2)`).

### Bundling ONNX Runtime with your build

```
cmake -S . -B build -DVOXAVERSA_ONNXRUNTIME_DIR=/path/to/onnxruntime-osx-universal2-1.20.1
```

`cmake/BundleOnnxRuntime.cmake` then copies the runtime (plus any provider libraries in the archive)
next to every plugin/app binary after each build — into `Contents/Frameworks` for macOS bundles. Re-sign
macOS bundles afterwards. ONNX Runtime is MIT-licensed; include its `LICENSE` and `ThirdPartyNotices.txt`
when you redistribute it.

---

## 4. Creating a voice pack

Training kit (dataset cleaning, free Colab notebook, ONNX exporters for voices and the content encoder): [TRAINING.md](TRAINING.md).

> **Consent and rights.** Only create voice models from recordings **you made yourself or that you have
> explicit, informed permission to use for this purpose** from the person whose voice it is (and from the
> rights holders of the recordings). A voice is personal: do not clone public figures, other artists or
> anyone who has not agreed to it, do not use a voice to deceive or impersonate, and respect the terms you
> agreed with the speaker when sharing a pack. Many jurisdictions protect a person's voice and likeness.
> Put the terms of use in the pack's `license` field. `tools/neural/package_voice.py` asks you to confirm
> this before it builds a pack.

### Step 1 — Record a dataset

10 – 30 minutes (more is better) of clean, dry, unprocessed singing and/or speech of one person,
covering the range and styles the voice should have. No reverb, no backing track, no clipping, little
noise. Split long recordings into files of a few seconds to a minute.

### Step 2 — Train with the RVC WebUI

Install the [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (it needs an
NVIDIA/AMD GPU or patience) and in the *Train* tab:

* model version **v2**, target sample rate **40k** (32k and 48k work too),
* **pitch guidance (f0) enabled** — VoxaVersa drives the pitch, so f0 models give the best results,
* f0 extraction method `rmvpe` (best quality),
* train until the voice sounds right (typically 200–500 epochs for a small dataset). The trained model
  is written to the WebUI's `weights/` folder as `<name>.pth`.

(The feature `.index` file used by the WebUI's retrieval step is not used by VoxaVersa.)

### Step 3 — Export to ONNX

In the WebUI's **Export Onnx** tab, enter the path of the trained `weights/<name>.pth` as the RVC model
path and an output path such as `MyVoice.onnx`, then export. The result has the inputs `phone`,
`phone_lengths`, `pitch`, `pitchf`, `ds`, `rnd` and the output `audio`. Exports made with other tools
(e.g. the w-okada voice changer) work as long as they keep these names; float16 exports are supported.

### Step 4 — Get a content encoder

The generator needs the content encoder it was trained with, as ONNX: ContentVec layer 12 (768-d) for v2
models, ContentVec layer 9 with projection (256-d) for v1 models. Community exports are commonly named
`vec-768-layer-12.onnx` / `vec-256-layer-9.onnx` (for example in the `NaruseMioShirakana/MoeSS-SUBModel`
repository on Hugging Face — check the license of whatever file you use; that repository is published
under GPL-3.0). You can also export one yourself from the ContentVec/HuBERT checkpoint the RVC WebUI uses
(input: 16 kHz waveform `[1,1,N]`; output: the layer-12 hidden states `[1,T,768]`).

Install it once for all voices as `<user data>/Encoders/contentvec-768.onnx` (the names
`vec-768-layer-12.onnx` and `contentvec-256.onnx` / `vec-256-layer-9.onnx` are recognised too), or ship
it inside the pack and point `"encoderFile"` at it. Encoders are shared in memory between voices that
use the same file.

### Step 5 — Package

```
python3 tools/neural/package_voice.py --model MyVoice.onnx --name "My Voice" \
    --gender female --genre Pop --styles "Breathy,Bright" --range 55 79 \
    --author "Your Name" --license "Personal use only" --i-have-the-rights
```

This validates the model (if `pip install onnx` is available: input names, v1/v2 feature size, sample
rate from the export's metadata when present), writes `voice.json` and creates `my-voice.vxpack` — a zip
archive with `voice.json` at the root. Add `--encoder-file vec-768-layer-12.onnx` to bundle the encoder,
`--sample-rate 40000` if it can't be detected, `--avatar face.png` for a picture.

You can also build the pack by hand: put `voice.json` (see below) and the files in a folder and zip its
contents, renaming the archive to `.vxpack`.

### Step 6 — Import

In VoxaVersa open the settings menu, choose the voice pack import command and select the `.vxpack` (a
folder or `.zip` works too; this calls `VoiceLibrary::importVoicePack`). The pack is copied to
`<user data>/Voices/<pack name>/` and the voice appears in the library. Select it and enable the AI
engine. (Copying the unzipped folder into `<user data>/Voices/` and rescanning works as well.)

---

## 5. voice.json reference

```json
{
  "name": "My Voice",
  "id": "my-voice",
  "description": "Warm pop voice",
  "gender": "female",
  "genre": "Pop",
  "styles": ["Breathy", "Bright"],
  "range": [55, 79],
  "formant": 1.0,
  "breathiness": 0.1,
  "rasp": 0.0,
  "vibratoDepth": 20,
  "vibratoRate": 5.5,
  "colour": "#ff3a7b",
  "avatar": "avatar.png",
  "author": "Your Name",
  "license": "Personal use only",
  "factoryVoices": ["female-pop", "female-latin"],
  "listed": true,
  "model": {
    "file": "MyVoice.onnx",
    "encoder": "contentvec-768",
    "sampleRate": 40000,
    "encoderFile": "vec-768-layer-12.onnx",
    "hopSize": 400,
    "speakerId": 0,
    "noiseScale": 0.66666,
    "streamContextMs": 1000
  }
}
```

Top level (read by the voice library; all optional except `name`):

| Field | Meaning |
|---|---|
| `name`, `id`, `description`, `genre`, `styles` | Presentation and search. `id` defaults to the name. |
| `gender` | `male`, `female` or `neutral` (DSP defaults and filtering). |
| `range` | Comfortable MIDI range `[low, high]` (smart octave placement, Text-to-Sing). |
| `formant`, `breathiness`, `rasp`, `vibratoDepth`, `vibratoRate` | DSP voice traits (used by the DSP engine and for blending). |
| `colour`, `avatar` | Accent colour (`#rrggbb`) and image file in the pack. |
| `author`, `license` | Credits and terms of use — state what the speaker agreed to. |
| `factoryVoices` | Built-in voice ids (e.g. `female-pop`, see `VoxaVersaRender --list-voices`) that use this pack's AI model. Each keeps its own DSP colour, range and expression on top of the model's timbre. If two packs list the same voice, the pack whose folder name sorts first wins. |
| `listed` | `false` hides the pack as a voice of its own (it then only powers the built-in voices it lists). Default `true`. |

`"model"` (AI engine):

| Field | Required | Meaning |
|---|---|---|
| `file` | yes | RVC generator ONNX file, relative to the pack. |
| `encoder` | yes* | Content encoder id, resolved as `<pack>/<id>.onnx` then `<user data>/Encoders/<id>.onnx` (with aliases `contentvec-768` ↔ `vec-768-layer-12`, `contentvec-256` ↔ `vec-256-layer-9`). Defaults to `contentvec-768` / `contentvec-256` by feature size. |
| `sampleRate` | yes* | Generator output rate (32000 / 40000 / 48000). If 0 or missing it is read from the model's metadata or derived from the hop size. |
| `encoderFile` | no | Encoder ONNX shipped in the pack; overrides `encoder`. |
| `hopSize` | no | Generator samples per 10 ms frame (default `sampleRate / 100`); the load fails if the model disagrees. |
| `speakerId` | no | Speaker index (`ds`) for multi-speaker models, default 0. |
| `noiseScale` | no | Scale of the `rnd` noise input, default 0.66666 (0 = deterministic, slightly duller). |
| `streamContextMs` | no | Left context of each real-time chunk, 100 – 3000 ms, default 1000. More context = more consistent timbre between chunks but more encoder CPU; it does not change the latency. 2000 matches offline quality on a fast CPU; 300 is the minimum sensible value for slow machines. |

\* strongly recommended.

---

## 6. Integration notes (for developers)

```cpp
using namespace voxa::neural;

// Any thread, once (cheap if already loaded):
if (! OnnxRuntime::getInstance().ensureLoaded())
    showStatus (OnnxRuntime::getInstance().getStatusMessage());

// Background thread (loading + test runs take from ~50 ms to a few seconds for real models):
juce::String error;
auto model = NeuralVoiceModel::load (profile, error);        // nullptr + error on failure

// prepareToPlay (message thread):
stream.prepare (sampleRate, maxBlockSize);
setLatencySamples (stream.getLatencySamples());              // constant for these settings

// Message thread, any time (also while audio runs):
stream.setModel (model);                                       // nullptr disables the stream

// processBlock (audio thread):
stream.process (input, targetF0PerSample, output, numSamples);
if (! stream.isRunningInRealTime()) { /* e.g. fall back to the DSP voice */ }
// on transport jumps:
stream.reset();

// Offline (worker thread): f0 contour at any frame rate, 0 = unvoiced
model->convert (clip, clipSampleRate, f0, f0FrameRate, rendered, &cancelFlag, error);
```

* `NeuralVoiceModel::load` blocks while the models load and run twice; call it off the message thread.
  The returned model is immutable, thread-safe and can be shared by several streams and `convert` calls.
* `prepare()` stops and restarts the worker; call it only while audio is stopped (as `prepareToPlay`).
* The output is delayed by exactly `getLatencySamples()`; silence (coverage 0) is output until the first
  chunk is ready, after `reset()`, and whenever the worker falls behind live. Mix a fallback in with
  `1 − coverage` (as `VoiceChain` does) to fill it.
* Call `setNonRealtime()` with the host's offline flag before processing, so exports are complete.
* `targetF0Hz` is the desired output pitch for every input sample (0 = unvoiced → breath/consonants).
  For f0-less models it is ignored.
* Internals for diagnostics are reachable through `getImpl()` (`Source/neural/RvcModel.h`,
  `NeuralRuntime.h`), e.g. `model->getImpl().getRealTimeLoad()` or
  `OnnxRuntime::getInstance().getImpl().setAccelerator (Accelerator::automatic)`.

---

## 7. Testing

The unit tests (`Tests/neural/NeuralVoiceTests.cpp`, category `Neural`) always test the DSP parts. The
inference tests need ONNX Runtime and the dummy models built by `tools/neural/make_dummy_models.py`
(tiny ONNX graphs with the exact I/O signatures of a ContentVec encoder and an RVC generator; the dummy
generator sings a sine at `pitchf`, gated by the coarse `pitch`, with the input's envelope, so the
output's pitch and timing can be checked exactly). Without them the tests log "skipped" and pass.

```
python3 -m venv .venv && .venv/bin/pip install onnx numpy
.venv/bin/python tools/neural/make_dummy_models.py --out /tmp/vx-models
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xzf onnxruntime-linux-x64-1.20.1.tgz

cmake --build build --target VoxaVersaTests
VOXAVERSA_TEST_ORT_LIB=$PWD/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so \
VOXAVERSA_TEST_MODELS_DIR=/tmp/vx-models \
    ./build/VoxaVersaTests_artefacts/Release/VoxaVersaTests Neural
```

They check runtime loading, I/O discovery (rank-2/rank-3 inputs, channels-first outputs, float16,
models without f0), error messages for missing/corrupt/mismatched files, offline conversion (length,
sample rate, pitch, unvoiced gaps, long segmented clips, cancellation), constant streaming latency,
alignment of the streamed output with the offline result after the reported latency, hot-swapping
models and resets while streaming, and offline bounces.

---

## 8. Known limitations

* **Latency** is ~0.68 s in real time (see above). It is constant and reported to the host.
* **Feature retrieval** (RVC's `.index` / faiss step) and RVC's `protect` and `rms_mix_rate` options are
  not implemented; the generator receives the raw content features.
* **Pitch** must be supplied by the caller (target f0); the engine does not estimate pitch.
* **GPU providers** (CUDA, DirectML, CoreML) are wired up through ONNX Runtime's generic APIs but have
  only been exercised on CPU builds; any failure falls back to the CPU.
* **Offline joins**: segments of long clips may be shifted by up to ±6 ms around their nominal position
  to align phases (inaudible, never accumulates).
* **Noise** for the generator is deterministic per frame position; re-rendering a clip gives the same
  result.
