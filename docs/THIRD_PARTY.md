# Third-party material

## CMU Pronouncing Dictionary

- **Used for:** English pronunciation in Text-to-Sing (`Source/sing/G2P.cpp`).
- **Source:** <https://github.com/cmusphinx/cmudict> (`cmudict.dict`, retrieved October 2026)
- **Copyright:** © 1993-2015 Carnegie Mellon University. All rights reserved.
- **Licence:** BSD-style (2-clause). The full text is in
  [`Resources/sing/CMUDICT_LICENSE.txt`](../Resources/sing/CMUDICT_LICENSE.txt). Because the
  dictionary is compiled into the binary, this licence text must also ship with the product (for
  example in the About box or the installer's licence notes).
- **Modifications:** only the first pronunciation of each word is kept (variants such as
  `word(2)` and comments are dropped). The phonemes are packed one byte each and the file is
  gzip-compressed into `Resources/sing/cmudict.gz` (126,052 words, about 750 KB). The format is
  described in `Source/sing/G2P.cpp`.

To rebuild the asset from a newer `cmudict.dict`: keep the first entry per word, sort the words
bytewise, write the header line `#VOXALEX 1 <symbols...>` (the 69 stress-marked ARPAbet symbols
in sorted order; symbol *i* is encoded as the byte `0x30 + i`), then one `word<TAB>codes` line
per word, and gzip the result.

## NRL letter-to-sound rules

- **Used for:** pronouncing words that are not in the dictionary (`Source/sing/LetterToSound.cpp`).
- **Source:** H. S. Elovitz, R. Johnson, A. McHugh, J. E. Shore, *Automatic Translation of English
  Text to Phonetics by Means of Letter-to-Sound Rules*, NRL Report 7948, US Naval Research
  Laboratory, 1976.
- **Licence:** a work of the US Government, in the public domain. The rules were re-typed with
  ARPAbet output; no third-party code is included.

## WORLD vocoder (CheapTrick and unvoiced synthesis)

- **Used for:** the Whisper voice (`Whisperizer` in `Source/dsp/VocalFx.cpp`): its spectral envelope
  analysis (CheapTrick) and noise synthesis are ported from WORLD, rewritten to allocate nothing while
  processing.
- **Source:** <https://github.com/mmorise/World> (`cheaptrick.cpp`, `synthesis.cpp`, `common.cpp`,
  `matlabfunctions.cpp`)
- **Copyright:** © 2010 M. Morise. All rights reserved.
- **Licence:** modified BSD (3-clause). Because the port ships in the binary, this notice must also ship
  with the product (for example in the About box or the installer's licence notes):

  > Redistribution and use in source and binary forms, with or without modification, are permitted
  > provided that the following conditions are met: Redistributions of source code must retain the above
  > copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form
  > must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  > documentation and/or other materials provided with the distribution. Neither the name of the M. Morise
  > nor the names of its contributors may be used to endorse or promote products derived from this
  > software without specific prior written permission.
  >
  > THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
  > WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  > PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
  > ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  > LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  > INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  > TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  > ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

- **Modifications:** buffers are allocated once in `prepare()`; frames are analysed as audio arrives
  (half an FFT ahead) instead of over a whole file; only the unvoiced (noise) synthesis is used; JUCE's
  FFT replaces FFTSG.

## Published methods implemented from the literature (no code copied)

- Liljencrants-Fant glottal flow model (Fant, Liljencrants & Lin, 1985) and its Rd
  parameterisation (Fant, 1995).
- Cascade/parallel formant synthesis after Klatt (1980).
- YIN fundamental-frequency estimator (de Cheveigné & Kawahara, 2002).
- Krumhansl-Schmuckler key finding (already part of `Source/core/MusicTheory.cpp`).
- Vowel formant targets derived from Peterson & Barney (1952) and Hillenbrand et al. (1995).

## Voice Library portraits (`Resources/voices/*.jpg`)

The 46 factory-voice headshots and the 15 singer-card portraits (`singer-*.jpg`, for trained singers shown as
voices of their own) are **AI-generated images of people who do not exist**. No photos
of real people were used as input, and no prompt names a real person, artist or celebrity; every
image was reviewed and anything resembling a recognisable person was rejected.

- **Generator:** [`SG161222/Realistic_Vision_V5.1_noVAE`](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE)
  (Stable Diffusion 1.5 fine-tune "Realistic Vision V5.1" by SG161222), Hugging Face revision
  `1e9f017a7b1eaefb63a1900ea6c5953d2739fd21`.
  - **Licence:** CreativeML OpenRAIL-M (as stated on the model card). The licence places no
    restrictions on the generated output beyond its use-based restrictions (Attachment A), so the
    images may be used commercially. The author's CivitAI page (model 4201) additionally lists
    "sell images they generate" as permitted and asks for credit, which this section provides.
- **VAE:** [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse),
  revision `31f26fdeee1355a5c34592e401dd41e45d25a493`, MIT licence.
- **Generation:** offline on CPU with Hugging Face `diffusers`; image-to-image from a neutral
  head-and-shoulders silhouette on a plain mid-grey studio backdrop (so every portrait has the same
  framing), soft key + fill lighting, DPM++ 2M Karras sampler, CFG 5.5. The per-voice prompts
  (genre styling, wardrobe, age, ethnicity) and seeds are kept with the generation scripts, not in
  the product.
- **Post-processing tools (not shipped):** the YuNet face detector from the OpenCV model zoo
  (MIT) to frame each face consistently, and the IS-Net `isnet-general-use` segmentation model
  (Apache-2.0) to cut the person out of the grey backdrop and place them on the shared
  navy/indigo starry backdrop with a thin blue rim light. Images are exported as 256 x 280 JPEGs.

The model weights are not redistributed with VoxaVersa; only the generated images are.

## Training data of the Southern vowel recognizer (`Resources/neural/southern_vowels.gz`)

- **Used for:** the Southern "ah" of the country voices (`Source/neural/SouthernVowels.cpp`). The file holds the
  weights of a small network trained by `tools/research/vowelnet` (no audio).
- **Trained on:** Text to Sing takes sung by AI singers trained on public-domain US military band recordings and
  on **VocalSet** (J. Wilkins, P. Seetharaman, A. Wahl, B. Pardo, *VocalSet: A Singing Voice Dataset*, ISMIR 2018,
  <https://zenodo.org/records/1442513>, CC BY 4.0), and on speech from **Kokoro-82M** (hexgrad,
  <https://huggingface.co/hexgrad/Kokoro-82M>, Apache License 2.0). Its input features come from ContentVec
  (MIT), the content encoder of the AI voices.

## Bundled with the installers (not part of the source code)

| Component | Use | Licence |
|---|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.20.1 (Windows: DirectML build) | runs the AI voice models | MIT |
| [DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML) 1.15.2 (Windows) | GPU acceleration for ONNX Runtime | Microsoft DirectML licence (redistributable) |
| Microsoft Visual C++ Redistributable (Windows installer) | C++ runtime needed by ONNX Runtime | Microsoft Visual Studio redistributable terms |

