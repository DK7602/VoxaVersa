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

## Bundled with the installers (not part of the source code)

| Component | Use | Licence |
|---|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.20.1 (Windows: DirectML build) | runs the AI voice models | MIT |
| [DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML) 1.15.2 (Windows) | GPU acceleration for ONNX Runtime | Microsoft DirectML licence (redistributable) |
| Microsoft Visual C++ Redistributable (Windows installer) | C++ runtime needed by ONNX Runtime | Microsoft Visual Studio redistributable terms |

