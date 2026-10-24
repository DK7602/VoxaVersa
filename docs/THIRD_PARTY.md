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
