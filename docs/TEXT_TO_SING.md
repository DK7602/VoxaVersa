# Text to Sing

VoxaVersa can sing typed lyrics. The `voxa::sing` module (`Source/sing/`) turns text into a
**guide vocal**: a dry, mono, in-tune performance with clear diction and natural pitch
behaviour. The guide then runs through the selected voice's chain like any other input, and
through the voice's neural model when one is installed (see *Quality* below).

```
lyrics ──► parse (words, syllables, melismas, breaths, phrases)
       ──► pronounce (CMU dictionary → fallbacks → letter-to-sound) ──► syllabify
       ──► melody (compose, or fit a MIDI / captured / hummed melody)
       ──► performance plan (phoneme timing, breaths, pitch contour, dynamics)
       ──► formant synthesizer ──► SingResult { audio, notes, f0, key }
```

Public API: `Source/sing/SingEngine.h` (`SingEngine` for offline takes, `LiveSinger` for MIDI-driven
real-time singing).

## Lyrics syntax

| You type | Meaning |
| --- | --- |
| words separated by spaces | one or more syllables per word, split automatically |
| a new line | a new phrase; the singer breathes between phrases |
| `beau-ti-ful` | explicit syllable split (each piece is one or more sung syllables) |
| `_` (or `love_`, `love__`) | melisma: the previous syllable continues onto the next note |
| `[br]` or `[breath]` | an explicit breath / rest |
| `[Chorus]`, `{x2}`, `<...>` | any other tag is ignored |
| `, . ; : ! ? —` | stripped; commas and full stops become phrasing hints (a slightly longer note and a short gap) |
| `21`, `1st`, `1999` | spelled out ("twenty one", "first", "nineteen ninety nine") |
| `don't`, `I'm`, `gonna`, `lovin'`, `'cause`, `don’t` | contractions, slang and curly apostrophes are understood |
| `ooooh`, `yeahhh`, `mmm`, `hmm`, `la`, `na` | vocables; elongated spellings are collapsed; `mmm` is hummed on M |
| `&`, `@`, `%`, `+` | sung as "and", "at", "percent", "plus" |

`SingEngine::countSyllables (lyrics)` returns exactly the number of syllables that a render or the
live singer will sing (melisma marks and breaths do not count), so the UI can show "12 syllables /
14 notes".

### Pronunciation

1. The [CMU Pronouncing Dictionary](THIRD_PARTY.md) (126k words, embedded, decompressed once on first
   use, ~20 ms) gives the pronunciation and lexical stress.
2. Words it lacks are handled by contraction rules (`'s 'll 're 've 'd n't`), g-dropping
   (`nothin'`), elongation collapsing, suffix stripping (`-ing -ed -s -er -ly -ness ...` on a known
   stem), compound splitting (`moonlightdancer`) and finally the NRL letter-to-sound rules.
3. Syllables are formed around vowel nuclei with the maximal-onset principle (`ex-tra` sings as
   *ek-stra*). A one-syllable function word (*the, a, of, to, and, my...*) is treated as unstressed.
4. The singer speaks **General American** connected speech, not the dictionary's word-by-word citation
   forms (those are what make a synthetic singer sound like a non-native speaker):
   - function words take their weak form on short notes that run on into the next word: *to* [tə],
     *for / or / your / are* [ɚ], *of, from, was, at, can, an, as, than, them, but, some, does, has, have,
     had* with a schwa. Long, held or phrase-final notes keep the full vowel, as singers do. *The* is
     [ði] before a vowel;
   - T and D are **flapped** between a vowel (or R) and an unstressed vowel (*better, city, little,
     party*) and at the end of a word before a vowel-initial word (*got it, what about*);
   - *can't you, hold you* keep their T / D, unreleased, before a clean Y (no *cantchu*, *hol-ju*: a user
     heard the coalesced form as sloppy);
   - voiceless stops are aspirated only at the start of stressed syllables and words (not in *happy*);
     final stops are unreleased before another consonant (*good night*), released into a following
     vowel (*hold on*) and released softly - no voiced "-uh" - before a pause;
   - /u/ is fronted after coronals (*you, to, new*), /æ/ is raised before nasals (*hand, can't*), a
     vowel takes on the colour of a following R (*here, dark*: F3 falls) or L (*feel, all*: dark L).

If a made-up word comes out wrong, spell it phonetically (`ny-ooo`) or hyphenate it.

## Melody sources (`SingRequest::melodyMode`)

### Automatic (`MelodyMode::AutoGenerate`)

Composes a new melody for the lyrics, deterministic for a given `seed`:

- **Form:** each lyric line (or `[br]`-separated part) becomes a 2- or 4-bar phrase in 4/4; very
  long lines are split at commas / word boundaries. The take starts in bar 2 (one bar of lead-in
  leaves room for the first breath and pick-up notes), so beat 0 of the take is a bar line.
- **Rhythm:** a dynamic programme places syllables on a 16th grid so that stressed syllables fall
  on strong beats and unstressed ones (and function words) on weak beats or pick-ups, around a
  syllable density set by the voice's `singRhythmBusy` (≈0.7 syllables/beat for ballads and opera,
  ≈2.6 for rap/trap). Phrases end on a long note followed by a rest; commas get a breath. When a line
  ends in a word stressed before its last syllable (*feeling, magic, together*), the stressed syllable
  gets the long note and the weak ending a moderate one (≤ 1.5 beats) - a long note on the ending
  would sound like *fee-LING*.
- **Pitch:** a second-order Viterbi search over scale degrees, guided by an arch contour (a flat
  reciting tone for rap), prefers stepwise motion with occasional leaps (larger leaps are
  "recovered" by a step back), favours chord tones of an implied four-bar chord loop on strong beats
  and repeats earlier rhythms/motifs when lines have the same shape. Antecedent phrases end on 3 or
  5, consequent phrases on 1 or 3, and the song ends on the tonic, approached from above.
- **Key and register:** every note stays in the requested key/scale (Chromatic composes in major)
  and inside the voice's `rangeLowMidi..rangeHighMidi`, with the tessitura centred in the range.
  `keyRoot = -1` picks a key whose tonic sits about a third below the voice's centre note
  (preferring C, D, E♭, E, F, G, A, B♭).

### Provided (`MelodyMode::Provided`)

`SingRequest::melody` comes from one of:

- `SingEngine::melodyFromMidiFile` – the first track that has notes; ticks → beats, the first
  tempo event → `bpmOut`; SMPTE files are converted via seconds.
- `SingEngine::melodyFromSequence` – notes played into the plugin (timestamps in beats).
- `SingEngine::melodyFromAudio` – hum-to-melody: YIN pitch tracking (10 ms hop), stable-pitch
  segmentation (vibrato and scoops don't split notes; silences and re-attacks do), fragments
  under ~80 ms are merged, quantised to a 1/16 grid at the given tempo; leading silence is trimmed
  to one beat.

All three are made monophonic (chords keep the top note; overlaps use last-note priority).
Syllables are assigned to notes in order, `_` holds the previous syllable on the next note, and a
lyric line break or `[br]` makes the singer breathe (shortening a legato note if needed). If there
are **more syllables than notes** the melody repeats in whole bars; if there are **more notes than
syllables** the last syllable carries on through the rest of its phrase as a short melisma and
the remaining notes are left as rests. The melody is moved by whole octaves to fit the voice;
with `keyRoot = -1` the reported key is estimated from the notes. A played melody carries no lyric
stress, so unstressed syllables are sung about 1 dB softer than their note velocity.

## How the voice shapes the singing

| `VoiceTraits` | Effect |
| --- | --- |
| `gender`, `formantSemis` | Male / female / neutral formant tables, scaled by 2^((formantSemis − nominal)/12) with nominal −1 (male), +1 (female), 0 (neutral) |
| `rangeLowMidi`, `rangeHighMidi` | Melody register, octave placement of provided melodies, automatic key |
| `breathiness` | Aspiration noise gated by the glottal cycle (HNR ≈ 29 → 17 → 8 → −1 dB for 0 / 0.4 / 0.6 / 0.9), laxer glottal pulses, louder breaths; above 0.5 the voicing itself fades towards a whisper |
| `rasp` | Period doubling (subharmonic growl), irregular weak pulses, extra jitter/shimmer, slightly pressed phonation |
| `tiltDb` | Glottal pulse shape (brighter = tenser) |
| `vibratoRateHz/DepthCents/OnsetMs` | Vibrato only on held notes (≥ 0.5 s), blooming in their second half and slightly different every cycle; phrase-final notes always, others about half the time and lighter. Short notes are sung straight |
| `scoopCents`, `scoopMs` | How freely the voice bends notes: scoops up into fresh attacks (≈1.8 × scoopCents deep on stressed syllables, less on weak ones), dips into some rising stressed notes, falls through below lower notes and decorates some held notes (soulful styles), and falls off most phrase ends |
| `jitterCents` | Slow pitch drift and cycle-to-cycle jitter (0 = robot-steady) |
| `singPortamentoMs` | Glide time between legato notes (centred just before the beat, with a damped overshoot on leaps) |
| `singLegato` | Gaps between syllables (detached ↔ fully joined) and release times |
| `singRhythmBusy` | Syllable density, rhythmic grid, phrase-final lengths, repeated-note tendency, consonant speed |

The remaining traits (EQ, presence, air, nasal, "vibe" effects) are applied by the voice chain
that processes the guide, as for any other input.

The pitch performance follows contemporary studio lead vocals rather than a synthesizer. A regular
vibrato on every note, gentle attacks and held phrase ends (as earlier versions sang) read as
synthesized, Vocaloid-like singing, which listeners heard as an East Asian singing style even with
native English pronunciation. Measured with RMVPE on R&B lead vocals: about half of their sung time is
steady pitch, attacks start 0.3-3 semitones low, vibrato appears on long notes only, and quick dips and
falls between notes are common.

Velocity sets loudness (up to −16 dB for soft notes) and brightness (glottal tension).

## The synthesizer

- **Source:** Liljencrants-Fant glottal flow derivative, pre-computed for 13 values of Fant's Rd
  (0.3 pressed … 2.7 lax) as band-limited mip-mapped wavetables (half-octave harmonic limits) so
  that nothing aliases at any pitch or sample rate; per-period jitter and shimmer; aspiration noise
  gated by the open phase.
- **Tract:** nasal anti-resonance, then a cascade of 14 resonators (F1-F5 follow the phonemes,
  F6-F14 model the upper tract and act as a higher-pole correction for a natural spectrum above
  4 kHz), plus a gentle high shelf. Bandwidths follow the formant frequencies; F1 is raised above
  f0 on high notes the way trained singers tune it - on sonorants only: an obstruent constriction
  pulls F1 down whatever the pitch, and at soprano pitches that dip is most of what marks a stop,
  flap or fricative transition (nasals stay tuned, or their murmur would vanish on high notes).
- **Vowels:** General American targets after Hillenbrand et al. (1995), slightly more open for
  singing. /eɪ/ and /oʊ/ are real diphthongs (F1 falls 140-170 Hz) - sung as pure [e] and [o] they
  sound foreign; /æ/ is a front vowel (not [a]), /ɔ/ an open one (not [o]).
- **Consonants:** plosives with closure, burst shaped by place of articulation (velar bursts
  follow the neighbouring vowel, with F2 and F3 drawn together - the "velar pinch" that tells /g/
  from /b/ and /d/), aspiration whose formants move from the consonant's locus
  towards the vowel; flaps as a ~28 ms voiced tap (a ~10 dB dip, no burst); final stops with a weak,
  devoiced release; no voice bar before a voiced stop after a pause (English /b d g/ are not
  prevoiced); fricatives through a high-passed two-resonance noise branch (S, SH, F, TH and
  voiced Z, ZH, V, DH mixing voicing); affricates; nasal murmur with an anti-resonance; liquids and
  glides as formant targets, with a velarised (dark) L in codas; HH as aspiration through the next
  vowel.
- **Timing:** onset consonants are sung *before* the beat so the vowel lands on it (20 % shorter in
  unstressed syllables, as in English speech); codas sit at
  the end of the note (sharing the gap with the next onset in legato lines); diphthongs hold their
  first vowel and glide late, completing just before the next consonant (a vowel before a coda R
  or L glides into its colour the same way); detached voices leave small gaps between words but
  never inside a word or where words link (flaps, vowel-initial words); breaths are inserted before
  phrases when there is room.
- **Coarticulation:** articulator targets are interpolated with zero-phase Gaussian smoothing
  (≈50-60 ms formant transitions; fast for amplitudes and bursts). Consonant-vowel transitions are
  anchored at the release so the audible part carries the place cue.
- **Loudness:** vowels are partially equalised (the tract would otherwise make some vowels much
  quieter at some pitches), and the take is normalised to −12 dBFS peak with gentle fades and DC
  removal.

`SingResult::f0Hz` is the sung pitch contour (100 frames/s, 0 when unvoiced) and matches the
audio within a few cents; `notes` are the melody notes with their syllable text (`_` for melisma
continuations); `keyRoot`/`scale` are the key actually used.

### Performance

A 30-second take renders in about 0.1–0.2 s at 48 kHz on one core. `render()` checks the cancel
flag every ~250 ms of audio and reports progress monotonically up to 1.0. `SingEngine` holds no
mutable state, so one instance may render on several threads at once.

## MIDI export

`SingEngine::toMidiFile (result, bpm)` writes one track with the track name, tempo, 4/4 time
signature, the notes and a lyric meta event (type 0x05) on every sung syllable, using the karaoke
convention that a trailing hyphen means the word continues (`Hel-`, `lo`). Melisma notes carry no
lyric.

## Live singing (`LiveSinger`)

Every MIDI note-on sings the next syllable (wrapping at the end; `restartLyrics()` goes back to
the start, e.g. when the transport starts). It is monophonic with last-note priority: a note-on
while another key is held glides (portamento) and articulates the next syllable without a new
attack; releasing the top key falls back to the still-held key on the same vowel; the final
note-off sings the coda and releases. A `_` in the lyrics makes the next note continue the
previous vowel. With no lyrics it sings "ah".

- `setLyrics()` (message thread) parses the text and publishes it through an atomic pointer; the
  audio thread adopts it at the next note-on or render call and hands the old object back through
  a lock-free FIFO, which the next `setLyrics()` (or the destructor) frees. Nothing is allocated,
  locked or freed on the audio thread; `noteOn/noteOff/render/setVoice` are real-time safe.
- Cost: about 0.3 % of one core at 48 kHz (measured by the unit test).
- Because the singer can't know a note is coming, onset consonants are compressed to ≤ 90 ms
  right after the note-on (the vowel arrives that much later than the key press).

## Quality and limitations

Be honest with users about what this is:

- **It is a formant synthesizer.** On its own it sounds clearly synthetic — clean and intelligible
  in the lower and middle registers, but buzzy and "vocoder-like" compared with a real singer. Its
  job in VoxaVersa is to be a well-timed, well-pronounced, in-tune **guide**. When the selected
  voice has a neural model installed, VoxaVersa runs the guide through that model, which replaces
  the timbre with the trained voice's and is what makes the result sound human. Voices without a
  model only get the DSP voice chain, so they will still sound synthetic.
- **Intelligibility** was measured with off-the-shelf speech recognisers on two four-line test
  songs sung by 14 factory voices (auto melodies): Whisper small.en recognises about 98 % of the
  words (every take ≥ 95 %, the soprano included); the weaker base.en model about 90 %, mostly
  because it hallucinates on long sustained notes. Remaining confusions are nasals vs. liquids
  (*night* → *light*) and the odd voiced plosive at a phrase start (*dancing* → *then sing*).
  A live-singer run of "Hello my friend, how are you" is recognised word for word.
- **Accent.** Early versions sang with a recognisably non-native accent (users heard an East Asian
  one, most of all in the female voices): pure [e] and [o]
  for /eɪ oʊ/, an [a]-like /æ/, a very back /u/, citation forms of every function word, no flaps,
  aspirated T in *better*, a voiced "-uh" after final stops, silent gaps between the syllables of a
  word and long notes on weak endings (*fee-LING*). Measured on 20 lines (5 four-line songs, 2
  melody seeds) sung by Female Pop, Female R&B and Male Pop at 80 and 128 bpm, the American English
  rules above roughly doubled the English score of a VoxLingua107 language identifier (P(English)
  of the female guides 0.07-0.29 → 0.16-0.59; Piper US/GB text-to-speech scores ≈ 1.0, so sung
  audio never reaches speech values), raised the US margin of a CommonAccent accent classifier for
  every voice, lowered Whisper base.en WER in 11 of 12 cases and Whisper small.en WER through the
  voice chain from 6-15 % to 3-4 % for the female voices, and raised the stressed/unstressed vowel
  duration ratio from 2.1-2.5 to 2.5-2.7. Female Pop still scores lowest on language ID, mainly
  because its clean, non-breathy source reads as less natural (raising its `breathiness` from 0.2 to
  0.3 lifts the score by ~40 % but costs a little intelligibility, so the voice was left as
  designed).
- **High notes:** above roughly A5 the harmonics are too far apart to define the vowel (true for
  real sopranos too), and formant tuning makes all vowels drift towards "ah".
- **English only.** Other languages are read with English rules. Names and invented words go
  through the letter-to-sound rules and may need phonetic spelling.
- **One pronunciation per word** (the dictionary's first variant): *read* (present/past), *live*,
  *wind* etc. may come out in the wrong form; respell them (*red*, *liv*).
- **Expression** is rule-based: no learned phrasing, no growls, belts, runs or ad-libs beyond what
  the melody, melisma marks and traits describe.
- **Auto melodies** are pleasant and singable but simple: one implied chord loop, 4/4 only, no
  chorus/verse awareness beyond motif reuse between lines of the same length.
- **Live mode** cannot pre-time consonants (see above) and has no lookahead for diphthong timing.
  It shares the vowel and consonant allophones (dark L, soft final stops, fronted /u/, R/L colouring
  on release) but not the rhythm-dependent connected speech (flaps, weak forms, *don'-chu*).

## Development notes

- Unit tests: `./VoxaVersaTests Sing` (G2P, lyrics, melody, MIDI, hum-to-melody, render pitch
  accuracy via YIN, f0 contour, cancellation, LiveSinger behaviour and CPU).
- Demo takes and per-phoneme level statistics:
  `VOXA_SING_RENDER_DIR=/some/dir ./VoxaVersaTests SingDemo` writes WAVs of several factory voices,
  phoneme inventories, breathiness/rasp ladders and a live-singer run.
- Measuring a change (build with `-DVOXAVERSA_BUILD_TOOLS=ON`):
  `VoxaVersaRender --sing lyrics.txt guide.wav --voice female-pop --bpm 80 --segments plan.tsv
  --chain voiced.wav` renders the take with an automatic melody (seed 1 = the plug-in's first
  Generate), its phoneme plan (one row per segment: times, phoneme, stress, word) and the take through
  the voice chain as the plug-in exports it. `--trait formantSemis=0.9` etc. overrides a trait of the
  guide for experiments. Keep renders and models out of the repository.
- Main sources: `Lyrics` (parsing), `G2P` + `LetterToSound` (pronunciation), `Melody` (compose,
  fit), `PitchTracker` (YIN, hum-to-melody), `Articulation` (phoneme targets, voice set-up),
  `VocalSynth` (glottal source and filters), `OfflineRenderer` (performance planning and
  rendering), `LiveSinger.cpp`, `SingEngine.cpp` (public API, MIDI).
