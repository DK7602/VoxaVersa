# Training your own AI voices

VoxaVersa's AI engine plays **RVC v2 voice models** (see [NEURAL_ENGINE.md](NEURAL_ENGINE.md)). This guide
explains, in plain language, how to turn recordings of a singer into a VoxaVersa voice pack (`.vxpack`):
what audio you need, how to clean it, how to train on a free Google Colab GPU, and how to install the result.

Contents

1. [The short version](#1-the-short-version)
2. [What audio you need](#2-what-audio-you-need)
3. [Preparing the recordings](#3-preparing-the-recordings)
4. [Training on Google Colab (recommended)](#4-training-on-google-colab-recommended)
5. [Training on your own GPU or a rented one](#5-training-on-your-own-gpu-or-a-rented-one)
6. [Installing the voice in VoxaVersa](#6-installing-the-voice-in-voxaversa)
7. [Troubleshooting](#7-troubleshooting)
8. [Example: the Cymatics vocal packs](#8-example-the-cymatics-vocal-packs)
9. [VocalSet](#9-vocalset)
10. [Licences of the tools and models](#10-licences-of-the-tools-and-models)
11. [How it works (for developers)](#11-how-it-works-for-developers)

---

## 1. The short version

1. Collect **10 - 30+ minutes of dry singing by one person** (no reverb, no backing track, no stacked
   harmonies). Zip the files.
2. Open [`tools/training/VoxaVersa_Voice_Training.ipynb`](../tools/training/VoxaVersa_Voice_Training.ipynb)
   in [Google Colab](https://colab.research.google.com) (*File > Upload notebook*, or open it from GitHub),
   choose *Runtime > Change runtime type > T4 GPU*, fill in the settings and run all cells.
3. After 1.5 - 3 hours you download `your_voice.vxpack` (and, the first time, `contentvec-768.onnx`).
4. In VoxaVersa: put `contentvec-768.onnx` into the *Encoders* folder, *Import voice pack*, pick the voice,
   switch the **AI engine** on.

> **Rights.** Only train voices you own or have explicit, written permission to use, and only from
> recordings whose licence allows AI training. Royalty-free sample packs usually allow use *in songs*, not
> necessarily training a voice model: check the licence (or ask the vendor) first. Never clone a public
> figure or anyone who hasn't agreed. Put the terms in the pack's *licence* field.

## 2. What audio you need

**Amount.** RVC fine-tunes a pretrained model, so it works with little data, but more is clearly better:

| Minutes of clean singing | What to expect |
|---|---|
| 3 - 10 | Recognisable timbre, but unstable: hoarse or metallic notes, problems on notes and vowels the singer didn't sing. |
| 10 - 30 | Usable voice. Most community models are in this range. |
| 30 - 60+ | Stable across the range, clearer consonants, better on fast passages. |

**Quality matters more than quantity.** Ten minutes of clean dry singing beats an hour of reverb-soaked
song stems.

* **One singer.** Mixing singers produces a blend (see below), never one of them.
* **Dry.** No reverb, delay, chorus, doubling, autotune or heavy compression. A little EQ is fine.
  "Wet" stems from sample packs have the effects baked in; the model would learn the reverb and add it to
  everything.
* **Monophonic.** One voice at a time: no stacked harmonies, no double-tracked leads. Single harmony lines
  are fine (they are just more singing).
* **Clean.** No backing track or bleed, no clipping, little room noise. Isolated vocals from finished songs
  (stem separation) work but sound worse than real dry recordings.
* **Lossless if you have it.** WAV/FLAC/AIFF. MP3 works, but its artefacts (especially at lower bit rates)
  end up in the voice; use the original WAVs when you have both.
* **Variety.** Low and high notes, soft and loud, sustained vowels, fast phrases, the styles you want the
  voice to do. The model can only reproduce what it has heard; VoxaVersa can move a voice outside its range,
  but the further outside, the worse it sounds.

**What to record (your own voice or a session singer).** In a quiet, dead room (a closet full of clothes
works), with a decent condenser or dynamic mic 15 - 20 cm away and a pop filter, 24-bit WAV, peaks around
-10 dBFS, no effects: 10 - 20 songs or verses in your genre, plus scales and sustained vowels across the
whole range, a few minutes of soft/breathy singing, and a few minutes of belting. Split long takes into
files of a few minutes.

**What to buy.** Sample packs that contain **dry acapellas and dry single-voice stems of one credited
singer** (many vendors also sell "vocal kits" per singer), *with a licence that allows AI training*. Packs
of short phrases from many singers are less useful: you get a few seconds per singer. Speech datasets
work for timbre but don't teach singing.

**Blend or single singer?** Training on several singers at once gives a *new, original-sounding* voice
(an average of the timbres), which is attractive when you don't want to imitate a real person and when no
single singer has enough audio. The cost: the model has no consistent target, so the timbre can drift
between notes or phrases and pitch accuracy can suffer. Prefer one singer with 15+ minutes; blend only
singers of the same gender and similar style.

## 3. Preparing the recordings

The notebook does this automatically. On your own computer (Python 3.9+, `pip install numpy scipy
soundfile`, and ffmpeg for MP3 input):

```
python3 tools/training/prepare_dataset.py my_vocals/ --out dataset --name my_voice --zip
```

For every file it decodes, mixes to mono, removes DC and rumble (60 Hz high-pass), measures noise,
clipping, how much the singing is a single voice (periodicity) and how long phrase ends ring (reverb),
slices at silences into 3 - 10 s clips, drops what is clipped, noisy, polyphonic or reverberant,
normalises loudness (-20 LUFS, peaks at most -1 dBFS) and writes 40 kHz 16-bit WAVs plus
`report.md`/`report.json`: minutes per voice, the singer's range, and every rejected second with the reason.

Useful options: `--include '*dry*' --exclude '*wet*' --exclude '*acapella*'` (file name patterns,
repeatable), `--sr 48000` (for 48k models), `--min-len`/`--max-len`, `--keep-all` (only measure),
`--manifest voices.csv` (columns `path,voice`: several voices in one run), `--zip` (for the notebook).

**Which files belong to which singer?** When you have recordings from several songs or packs and don't
know who sang what:

```
pip install torch torchaudio speechbrain
python3 tools/training/prepare_dataset.py vocals/ --out clean --name pool
python3 tools/training/cluster_singers.py clean/pool --report clean/report.json --out clusters \
    --group '<regex whose first group is the song name>'
```

It computes speaker embeddings (SpeechBrain ECAPA-TDNN) for every clip, groups stems or songs by
similarity, and writes `clusters.md` (who sang what, how confident) and `manifest.csv` (edit the voice
names, then `prepare_dataset.py --manifest manifest.csv --out datasets --zip`). Speaker models are trained
on speech, so treat the result as strong evidence, not proof: listen to the clusters.

## 4. Training on Google Colab (recommended)

Open `tools/training/VoxaVersa_Voice_Training.ipynb` in Colab. Every cell explains itself; in short:

| Step | What happens | Time |
|---|---|---|
| 1. Check the GPU | Fails clearly if you forgot *Runtime > Change runtime type > T4 GPU*. | seconds |
| 2. Settings | Name, gender, genre, styles, licence; epochs; Google Drive on/off; the rights confirmation. | - |
| 3. Install | A separate Python 3.12 with PyTorch 2.7.1 (CUDA 11.8) and pinned packages, plus VoxaVersa's tools from GitHub. | 3 - 5 min |
| 4. Trainer | The official RVC WebUI at a pinned commit, its v2 base models, RMVPE and ContentVec (about 600 MB). | 1 min |
| 5. Recordings | Upload a zip (or point to a file/folder on Drive); `prepare_dataset.py` cleans it and reports the usable minutes. | 1 - 3 min |
| 6. Train | Slicing, RMVPE pitch, ContentVec features, then training from the pretrained base. Checkpoints go to Drive every 10 epochs. | 1.5 - 3 h |
| 7. Export and listen | ONNX export of the voice (checked against PyTorch) and of the content encoder; a preview of one clip. | 2 min |
| 8. Pack and download | `package_voice.py` writes `voice.json` (name, gender, genre, range from the dataset, ...) and the `.vxpack`. | seconds |

**Epochs.** One epoch = the whole dataset once. With the pretrained base, 200 - 400 epochs suit 10 - 30
minutes of audio; tiny datasets need fewer (100 - 200), larger ones can go to 500+. Too many epochs on a
small dataset *overtrain*: the voice gets harsh and copies the recordings' noise. Step 7 can export from the
latest checkpoint at any time (`USE_LATEST_CHECKPOINT`), so you can listen at, say, epoch 100 and 200.

**Time and cost** (approximate, 2026; check current prices):

| Where | GPU | Speed (15 - 20 min dataset) | 300 epochs | Cost |
|---|---|---|---|---|
| Colab free | T4 16 GB | ~20 - 40 s per epoch | 2 - 3 h | free, but sessions are limited (typically a few GPU hours per day) and disconnect when idle |
| Colab Pro / pay-as-you-go | L4 / A100 | 2 - 5x faster | 30 - 60 min | about 1 - 2 USD in compute units |
| Rented GPU (RunPod, Vast.ai, Lambda...) | RTX 3090 / 4090 | 3 - 5x faster than T4 | 30 - 50 min | about 0.2 - 0.8 USD per hour |
| Your PC | NVIDIA, 8 GB+ | like the rented GPU of the same class | | electricity |
| CPU only | - | ~10 min per epoch for 14 min of audio (4 cores) | days | not practical except for tests |

Free Colab tips: keep the tab open (or use Drive so a disconnect only costs the epochs since the last
checkpoint); if you run out of GPU time, continue the next day: run all cells again with the same voice name.

## 5. Training on your own GPU or a rented one

The notebook is a thin layer over `tools/training/rvc_train.py`, which works on any Linux/Windows machine
with an NVIDIA GPU (Python 3.11 or 3.12):

```
python -m venv venv && venv/bin/pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
venv/bin/pip install numpy==1.26.4 scipy librosa==0.10.2.post1 soundfile av ffmpeg-python praat-parselmouth \
    transformers==4.49.0 tensorboard matplotlib scikit-learn tqdm onnx==1.17.0 onnxruntime==1.20.1
venv/bin/python tools/training/prepare_dataset.py my_vocals/ --out datasets --name my_voice
venv/bin/python tools/training/rvc_train.py setup --rvc-dir rvc
venv/bin/python tools/training/rvc_train.py train --rvc-dir rvc --dataset datasets/my_voice --name my_voice --epochs 300 --batch 8
venv/bin/python tools/training/rvc_train.py export --rvc-dir rvc --name my_voice --out my_voice.onnx
venv/bin/python tools/training/export_encoder.py --model rvc/assets/hubert_base --out contentvec-768.onnx
venv/bin/python tools/training/try_voice.py --model my_voice.onnx --encoder contentvec-768.onnx \
    --input some_dry_vocal.wav --out preview.wav --rvc-dir rvc
python3 tools/neural/package_voice.py --model my_voice.onnx --name "My Voice" --gender female --genre Pop \
    --range 55 79 --license "Personal use only" --i-have-the-rights
```

`train` is resumable: run the same command again after an interruption. `--device cpu --threads 2
--batch 2 --loader-workers 1` trains on a CPU with about 5 GB of RAM (slowly). Models trained with the
RVC WebUI itself, or with forks such as Applio (HiFi-GAN vocoder, v2), export the same way:
`export_onnx.py --rvc-dir rvc --pth their_model.pth --out model.onnx`.

## 6. Installing the voice in VoxaVersa

Once per computer:

1. **ONNX Runtime 1.16+** (1.20.1 recommended): see [NEURAL_ENGINE.md, section 3](NEURAL_ENGINE.md#3-installing-onnx-runtime).
2. **Content encoder**: copy `contentvec-768.onnx` (made by the notebook or `export_encoder.py`) into
   * Windows `%APPDATA%\VoxaVersa\Encoders\`
   * macOS `~/Library/Application Support/VoxaVersa/Encoders/`
   * Linux `~/.config/VoxaVersa/Encoders/`

   All RVC v2 voices share it. (Alternatively tick *INCLUDE_ENCODER_IN_PACK*: each pack then carries its
   own 378 MB copy.)

Per voice: VoxaVersa's settings menu > **Import voice pack** > choose the `.vxpack` (or unzip it into
`<user data>/Voices/<name>/`). Select the voice in the library and switch the **AI engine** on. The status
line shows *"RVC v2 voice model (768-d features, pitch-guided, 40 kHz, encoder contentvec-768.onnx, CPU,
real-time load NN%)"* when it works.

Test renders from the command line (developer build with `-DVOXAVERSA_BUILD_TOOLS=ON`):
`VoxaVersaRender in.wav out.wav --voice user-my-voice --set aiEngine=1 --set pitch=-12`.

## 7. Troubleshooting

| Problem | Cause and fix |
|---|---|
| Notebook step 1: no GPU | *Runtime > Change runtime type > T4 GPU*. If no GPU is available, the free quota is used up: try later or use a paid option. |
| "Less than 3 minutes of usable singing" | Most of your audio was rejected: open `report.md` (the reason is listed per file). Wet/reverb stems, stacked harmonies and noisy recordings are removed on purpose. |
| Training stops after a disconnect | Run all cells again with the same voice name; it resumes from the last checkpoint on Drive. |
| `CUDA out of memory` | Lower `BATCH_SIZE` to 4. |
| The voice is hoarse, metallic or "underwater" | Too little or too noisy data, or too few epochs; listen to `report.md`'s rejected files, add clean singing, train longer. Harsh and noisy at high epochs = overtrained: export an earlier checkpoint. |
| It sounds reverberant | Wet audio in the dataset. Use dry stems only. |
| Mumbled consonants | Typical for small datasets and for MP3 sources; more and cleaner data helps. |
| Out-of-range notes crack | The singer never sang there. Use VoxaVersa's octave/pitch so the voice stays in its range (`range` in `voice.json`). |
| VoxaVersa: "Content encoder 'contentvec-768' not found" | Copy `contentvec-768.onnx` into the *Encoders* folder (section 6). |
| VoxaVersa: "ONNX Runtime not found" | Install ONNX Runtime (section 6). |
| "too slow for real time on this computer" | Expected on slower CPUs: the AI engine still works for offline renders/bounces; for live use set a GPU accelerator ([NEURAL_ENGINE.md](NEURAL_ENGINE.md#accelerators-and-threads)). |
| `package_voice.py`: "not an RVC ONNX export" | Export with `export_onnx.py` (RVC 2.3 removed its own ONNX export). |

## 8. Example: the Cymatics vocal packs

How the first VoxaVersa voices were prepared from the Cymatics vocal sample packs the owner licensed for AI
training (the audio itself is never committed):

```
# all dry single-voice stems; full acapellas are mixes of the part stems, wet/flanged stems have effects
python3 tools/training/prepare_dataset.py infinity/ cymatics/ uploads/ --out clean --name pool \
    --include '*dry*' --exclude '*wet*' --exclude '*flanged*' --exclude '*acapella-dry*' --exclude '*Cymatics_-_Acapella_-_*'
python3 tools/training/cluster_singers.py clean/pool --report clean/report.json --out clusters \
    --group '^Cymatics_-_(?:Hook_)?((?!Deluxe|Euphoria)[^-]+?)_-_|^((?!Cymatics)[A-Za-z_]+?)_(?:lead|background|harmon|ad-libs|adlib|low-voice|hook)'
python3 tools/training/prepare_dataset.py --manifest voices.csv --out datasets --zip
```

Findings (39.8 minutes of clean dry singing from about 200 stems):

* The ten male *Infinity* songs are one singer (song-to-song similarity 0.6 - 0.86, against at most 0.46
  between clearly different singers); the Euphoria acapella *Leave This Place* matches him too: **14.4 min**.
* The five female *Infinity* songs are one singer: **6.6 min**.
* *Blame*, *Found My Way* and most *Deluxe* female phrases are one singer; *On My Way* and *Time* are close
  to her (possibly the same): together **6.3 min**.
* The hooks, the *Light* lead, the *Deluxe* male phrases and the *Euphoria* phrases come from several other
  singers with 0.1 - 2.7 minutes each.
* Datasets: `cymatics_male_a` (14.4 min, one singer), `cymatics_female_a` (6.6), `cymatics_female_b` (6.3)
  and `cymatics_female_blend` (17.2 min, all female singers with at least a minute).

The MP3 transcodes (LAME V5) are usable but the original WAVs are better; with only 6 - 15 minutes per
singer these voices are at the low end of the quality scale. More dry material of the same singers (other
packs credited to them, or the vendor's full vocal sessions) is what would improve them most.

## 9. VocalSet

[VocalSet](https://zenodo.org/records/1442513) is a free dataset of 20 professionally trained singers
(11 male, 9 female) singing vowels, scales, arpeggios, long tones, song excerpts and some spoken text in
many techniques, about 20 - 27 minutes per singer after leaving out techniques that don't suit a voice model
(inhaled singing, lip trills, vocal fry, trills, trillo). It is licensed **CC BY 4.0**, so voices trained on
it may be shared as long as you credit it:

> VocalSet: A Singing Voice Dataset, J. Wilkins, P. Seetharaman, A. Wahl, B. Pardo, ISMIR 2018, CC BY 4.0.

Put that line in the pack's `--description` or `--license` field. Prepare one dataset per singer:

```
python3 tools/training/prepare_dataset.py vocalset/data/male1 --out datasets --name vocalset_male1 --zip
```

VocalSet is mostly classical-style exercises (vowels, scales, arpeggios) with some song excerpts, recorded
dry in a studio: excellent for a clean, stable *timbre* across a wide range, but it does not teach a
pop/R&B/trap *style* (phrasing, ad-libs, consonants). Combine it with genre singing of the same singer if
you can, or use it as the timbre and let VoxaVersa's input performance supply the style.

## 10. Licences of the tools and models

| Component | Used for | Licence |
|---|---|---|
| [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) at commit `81eed5e` (release 2.3.260718 + fixes) | slicing, f0, features, training code, model definitions | MIT |
| Pretrained v2 base (`f0G40k/f0D40k`, `f0G48k/f0D48k`), RMVPE (`rmvpe.pt`), ContentVec/HuBERT (`hubert_base/`, = RVC's `hubert_base.pt`) from [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) at revision `e6d0c1a` | training base, pitch, content features | MIT (repository licence); ContentVec itself is MIT |
| `contentvec-768.onnx` made by `export_encoder.py` | VoxaVersa's content encoder | MIT (derived from the above) |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) + [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | singer clustering only (not shipped) | Apache-2.0 (model trained on VoxCeleb) |
| PyTorch, transformers, ONNX, ONNX Runtime, librosa, NumPy, SciPy | tooling | BSD / Apache-2.0 / MIT / ISC |

Voice packs contain only the trained generator (and optionally the encoder). The **voice** in them is
governed by the rights to the recordings, which is why `package_voice.py` asks you to confirm them.
Do not use or ship the community `vec-768-layer-12.onnx` (GPL-3.0 repository) in packs:
`export_encoder.py` produces the equivalent file (cosine similarity 1.0000 to it on test audio) from MIT
sources.

## 11. How it works (for developers)

* **Training features = inference features.** RVC 2.3 extracts training features with
  `transformers.HubertModel(...).last_hidden_state` from `assets/hubert_base`; `export_encoder.py` exports
  exactly that network (input `source [1,1,N]` at 16 kHz, output `embed [1,T,768]`, one frame per 20 ms)
  and checks ONNX against PyTorch (relative error ~1e-5).
* **Unvoiced frames.** VoxaVersa feeds `pitchf = 0` and coarse pitch 1 for breaths and consonants, like RVC
  up to 2.2 and all community models. RVC 2.3 changed training and inference to interpolate the pitch
  through unvoiced gaps; `rvc_train.py` keeps zeros (`--f0-unvoiced zero`, default) so models match
  VoxaVersa's engine.
* **Generator export.** RVC 2.3 dropped its ONNX export; `export_onnx.py` rebuilds it on the pinned model
  code (inputs `phone, phone_lengths, pitch, pitchf, ds, rnd`, output `audio`, opset 17, dynamic length,
  sample rate in the metadata) and patches two `int()` casts in RVC's relative attention that would
  otherwise freeze the traced length. With the excitation noise switched off on both sides, ONNX matches
  PyTorch within ~1e-4.
* **Pinned versions.** RVC commit `81eed5e8f68b6bed1789f682fe78cdd324495afc`, Hugging Face revision
  `e6d0c1a17da07c33557852f9dfa2bd44cc75737d`, PyTorch 2.7.1 (GPU) / 2.4.1 (tested CPU), transformers 4.49.0,
  NumPy 1.26.4, librosa 0.10.2.post1, onnx 1.17.0, onnxruntime 1.20.1.
