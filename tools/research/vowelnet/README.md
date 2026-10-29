# Vowel recognizer for the country voices' Southern "ah"

`Source/neural/SouthernVowels.*` makes Male and Female Country sing *I, my, time, like* with the Southern "ah"
in Voice Transform: inside the AI singer, a small recognizer finds a vowel that sounds like "I" (or "oy") and
the moment it glides towards "ee", and the generator gets the vowel's "ah" from there to its end. These scripts
build that recognizer (`Resources/neural/southern_vowels.gz`) and check the decision rule, which `rule.py`
mirrors frame for frame.

The recognizer is an MLP (3840 -> 256 -> 21 classes) on five 20 ms frames of ContentVec features (the AI
singers' content encoder): ARPAbet vowels, the glide of AY and OY (`AY_g`, `OY_g`), EH before a nasal
(`EH_N`, for the next accent step), `C` and `SIL`. About 2 MB as float16.

## Data (nothing here is committed but the scripts and lyric lines)

Everything the weights learn from must allow it; sample-pack singers must never be used.

1. **Sung takes** (`render_takes.sh`): Text to Sing takes of `lyrics.txt` (vowel-rich lines) and
   `lyrics_open.txt` (the open vowels of *but, got, saw, down, go*: without them the recognizer took them
   for "I"), sung by AI singers trained on public-domain recordings (the band singers) and on VocalSet
   (CC BY 4.0, Wilkins et al. 2018), with the voices' effects off. The melody plan of each take labels it.
   ```
   HOME=<a home whose VoxaVersa config has the voices> \
     tools/research/vowelnet/render_takes.sh build/VoxaVersaRender_artefacts/Release/VoxaVersaRender takes 0 1 \
     male-pop male-rock male-opera male-theatre male-deep-bass user-vocalset-male3 \
     female-jazz female-rnb female-opera female-angelic female-theatre user-vocalset-female2 male-country female-country
   python3 tools/research/vowelnet/features.py takes contentvec-768.onnx takes.npz
   ```
2. **Speech** (`kokoro_feats.py`): eight American voices of Kokoro-82M (Apache 2.0) saying both line files.
   ```
   python3 tools/research/vowelnet/kokoro_feats.py tools/research/vowelnet/lyrics.txt contentvec-768.onnx kokoro.npz
   python3 tools/research/vowelnet/kokoro_feats.py tools/research/vowelnet/lyrics_open.txt contentvec-768.onnx kokoro_open.npz
   ```
3. **Real singing for testing only**: VocalSet's *Row, Row, Row Your Boat* (`excerpts/straight/*_row_straight.wav`
   from `tools/training/fetch_vocalset.py`), whose "life" is the target and the rest must be left alone, and
   any vocal of your own: `real_prep.py` adds voicing and word timings (faster-whisper).

## Train, check, export

The plug-in's stream runs the content encoder on chunks, so its features are not the whole-file ones:
`stream_windows.py` computes them as the stream does (each take with one of its contexts, 1 s to 150 ms), and
the recognizer learns from both.

```
python3 tools/research/vowelnet/stream_windows.py contentvec-768.onnx stream.npz takes
python3 tools/research/vowelnet/train.py model.pt stream.npz,takes.npz,kokoro.npz,kokoro_open.npz male-theatre user-vocalset-female2
python3 tools/research/vowelnet/real_prep.py contentvec-768.onnx real vocalset/*/excerpts/straight/*_row_straight.wav
for f in real/*.npz; do    # the real vocals as the stream sees them, with its longest and shortest context
  for ms in 1000 150; do python3 tools/research/vowelnet/stream_windows.py contentvec-768.onnx ${f%.npz}-$ms.npz \
    --real $ms vocalset/*/excerpts/straight/$(basename ${f%.npz}).wav $f; done; done
python3 tools/research/vowelnet/evaluate.py model.pt Resources/sing/cmudict.gz \
    --takes takes.npz takes male-theatre,user-vocalset-female2 --real real/*-1000.npz real/*-150.npz [--tune]
python3 tools/research/vowelnet/train.py final.pt stream.npz,takes.npz,kokoro.npz,kokoro_open.npz      # everything
python3 tools/research/vowelnet/export.py final.pt Resources/neural/southern_vowels.gz takes.npz
```

`rule.py` mirrors the plug-in's decision; `VoxaVersaRender --southern features.f32 f0.f32` runs the plug-in's own
code on a file's features in stream-sized chunks, to check they agree.

Test only on singers the recognizer never heard: leave out the *Row* takes of every VocalSet singer whose AI
singer sang takes (with the voices above: Male 1, 3, 4, 8 and Female 1, 2, 6, 9).

## What was learnt

- On real singing the recognizer's probabilities spread out more than on the takes it learnt from (the right
  vowel first, but at 0.3 - 0.5): the rule compares each probability with P(vowel) instead of using it as it
  is. Takes without the voices' effects (`render_takes.sh` turns them off) and more voices narrowed the gap.
- Trained on whole-file features alone, the recognizer caught "life" in the plug-in but missed "like" and "joy":
  the stream's encoder hears only up to 1 s before each frame. Hence `stream_windows.py`.
- Many singers glide into "ee" too briefly for the recognizer to see (a user's "like"): the last 40 ms of a
  vowel that sounded like "I" count as its glide.
- A "pain" sung with an open start looks like "I" at first: the frames just before the glide must still sound
  like "ah".
- On the open vowels the first recognizer (few takes, no open-vowel lines) held the end of *down, but, row,
  merrily*; that is what `lyrics_open.txt` is for.
