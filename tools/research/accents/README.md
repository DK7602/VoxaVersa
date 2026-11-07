# Accent prototype (Jamaican, Nigerian, Spanish) inside the AI singer

Round 17 of the listening audit: Female and Male Reggae, Afrobeats and Latin sing the user's vocal with an
accent made the way the country voices' Southern "ah" is, on the AI singer's content features (ContentVec, 50 Hz)
before its generator. The AI singer keeps the input's pronunciation, so the accent has to be put into what it hears.

* `phone_means.py` - mean feature vector per sound (vowels and consonants) from Kokoro-82M's American voices
  (Apache 2.0): the "th" -> "t" change adds (mean t - mean th).
* Vowel means come from LibriSpeech frames labelled by the vowel recognizer's data (`../vowelnet`, CC BY 4.0).
* `frames.py` - per-frame voicing, level and the recognizer's top guesses, to place the changes on a take.
* `accent_render.py` - plain, light and full versions of a take through one voice's generator:
  light = pure "ay" (the vowel's start held through its glide) and "th" -> "t"; full = light plus the accent's
  vowels ("i" -> "ee", "u" -> "o"/"ah", "a" -> "ah") and, for Jamaican and Nigerian, no "r" after a vowel.
  The spans are set by hand for the audit clip (read off `frames.py`): the prototype tests how the changes sound;
  finding them automatically in a live stream is the plug-in step, as `SouthernVowels` does for "I".
* `check_accent.py` - the recognizer on the rendered audio (did the singer sing the changed vowel?) and the words a
  speech recognizer still hears.

Round 17 (light and full, on the 14 s audit clip) was heard as no accent at all: a handful of small changes
doesn't read as an accent. `accent_strong.py` is the decisive test (round 18): the user's line "I think that this is
the time for me and you, girl, take it slow", rich in the words accents show on, with every change at 1.8x
(each vowel moved from the class actually sung in that frame), next to no accent.

Round 18 verdict: from "and" on (and -> an, girl -> gyal / gel, take -> tek, slow -> slo) a good accent; before it
(the "th" sounds, the short vowels of that / this / is, the "r" of for) no change heard. Changes to held vowels carry
through the AI singer; a nudge of a short consonant or vowel does not. `accent_stronger.py` (round 19) replaces "th"
with a stop (a closure, then "t" / "d") and pushes the short vowels harder.

Round 19 verdict: still "th" where a "d" belongs. Two causes: two of the four "th" spans were off (the user's "th" in
"that" is a breathy hiss before a short pause, 0.3 s ahead of the vowel), and the generator voices only where the
pitch track is on, so anything put in an unvoiced "th" comes out as hiss. `accent_stops.py` (round 20) rebuilds each
"th" as a stop: silence (closure), then a "t" burst or a voiced "d" release with the next vowel's pitch switched on.
The hiss in those spans drops by 8 - 26 dB.

Round 20 verdict: still "th", and the "r" of "for". Its spectrogram showed why: a gap where the "th" was, but no
burst, so no "t" / "d" to hear. Round 21 makes the consonants in the recording itself, before the AI singer:
`make_stops.py` cuts clean "t" / "d" onsets (burst + 40 ms of vowel) from Kokoro-82M (Apache 2.0), `edit_input.py`
turns each "th" into a closure plus one of them (level-matched, cross-faded into the singer's own vowel) and swaps
the r-coloured end of "for" for its plain "aw" (WORLD envelope); `accent_thick.py` then pushes every vowel change
harder (x2.4, short vowels x3). `spectrogram.py` shows the bursts in the AI singer's output.

Round 21 verdict: much better, but the audio broke up on the d of dat / dis / de, and "and" still ended in d. The
spliced "d" brought 40 ms of the text-to-speech voice's vowel at another pitch, after a long dead closure. Round 22
(`edit_input.py` as it is now): a short closure, a voice bar (the singer's own vowel, low-passed, quiet: the murmur
of a real "d"), only the 12 ms burst, then the singer's own vowel; the "d" of "and" before "you" becomes a held "n"
(WORLD envelope swap, pitch filled through the gap).

Round 22 verdict: the d's unintelligible, the sound just cuts out: a quiet voice bar and a 12 ms click render as
near-silence. Round 23: a whole "d" with the start of its vowel again (`make_stops_full.py` keeps Kokoro's whole
clip and burst time), re-synthesised by WORLD at the note sung there, after a short closure, blended into the
singer's vowel over 30 ms.

Round 23 verdict: still breaks up on the d. Measured (5 ms levels, input against output): the singer's vowel after
"th" starts softly and reaches full strength ~40 ms later, and the "d" sat before that soft start - the AI sang d,
a 9 - 20 dB dip, then the vowel; the silence put before each "d" was another break in a legato line. Round 24
(`edit_input.py` as it is now): no silence (a voice bar at the vowel's pitch, 10 dB down), a short "d" (burst +
25 ms, re-pitched) at the strong start of the vowel; "this" keeps round 23's version, which had no dip. The dips
after the "d" fall to 0 - 8 dB (the male voices' remainder is mostly the 5 ms window catching single pitch cycles).

Round 24 verdict: better (Female Reggae picked it over 23) but the d "warped", most in the voices with the hardest
vowel pushes right after it (x3 on the short vowels of dat / dees / ees / de); Female Reggae, with the fewest,
warped least. Round 25, the last on the d: A = round 24's d with those vowels at x1.8; B = no d change
(`edit_input_no_d.py`: the singer's own "th", with tink, fo and an' you kept). The pick goes into the plug-in.

Round 25: A (the d with gentler vowels) for all six voices - the accent to build into the plug-in.

## In the plug-in: the vowels (step 1)

The prototype knew the lyric; the plug-in decides live, from the vowel recognizer (`SouthernVowels`, now with
every class's mean features: `export.py --class-means`, means from LibriSpeech frames), with 2 frames of lookahead:
`SouthernVowels::AccentRule` (the reggae, Latin and afrobeats voices' `singAccent` 2 - 4). "ay" / "oh" sung pure
(tek, slo), the "a" of that / and -> "ah", "er" -> "ah" / "eh" / "ah" turning to "eh", "i" -> "ee" (Latin,
afrobeats), at round 25 A's pushes (x1.8, x2.4 on held notes). Two things undid them in the full plug-in
(`VoxaVersaRender`, recognizer read-back of the output on the user's take):
- the vowel keeper (VoiceChain), which pulls the AI singer's vowels toward the input's, pulled them back. The
  stream now reports per sample how much the accent changed the vowel (NeuralVoiceStream::process `accentOut`,
  NeuralVoiceModel::convert `accentFrames` for exports) and the keeper lets go there (VowelKeeper `release`):
  reggae "slow": P(oh) 0.35 -> 0.04; Latin "it": P(ee) 0.07 -> 0.30.
- live, the stream's encoder hears ~60 ms after each frame, and the user's "girl" came out as the "u" of "book"
  with "er" second (~0.15) - never a clear "er". A vowel is now also an "er" when "u" is on top with P(er | vowel)
  >= 0.10 and its mean so far >= 0.12, two frames in a row, and stays one to its end
  (`vowelnet/stream_er_uh.py`: on the AI singers' takes 3 of 80 "u" vowels, 1 before): reggae "girl" P(er)
  0.42 -> 0.01, P(ah) 0.44 (the prototype: 0.47); Latin P(er) 0.01, P(eh) 0.16 (0.19); afrobeats P(er) 0.12.

Step 2, the consonants (th -> t / d, fo, an' you), needs a consonant recognizer and edits to the input before the
encoder and the consonant keeper.

## Rhythm prototype (round 26)

`groove_render.py`: the accent A take re-timed toward each genre's groove on the take's own grid (93.3 BPM, eighths
fitted to the vowel onsets), by time-warping the AI singer's content features and pitch frames before the
generator (so the voice and pitch stay natural): each syllable's vowel onset moves to its target, its consonant
shifts rigidly with it, the vowel stretches or shrinks. Reggae: 60 % to the nearest sixteenth, 50 ms laid back,
connected notes cut at 85 % (clipped endings). Afrobeats: off-beat eighths half a sixteenth late (swung), 75 %,
cut at 70 % (bounce). Latin: the first off-beat eighth of each half bar a sixteenth late (dembow 3+3+2), 85 %, legato.

Round 27 (listening): the plug-in's live vowels (VoxaVersaRender on the take) against round 25 A.

## The consonants (step 2)

Round 25 A made tink / dat / dees / de and an' you by editing the recording at known times; live, the plug-in hears
each sound only ~40 ms ahead and cannot re-pitch and splice audio before the encoder in time. `accent_splice.py`
(round 28, C) makes the same sounds inside the AI singer: the "th" frames get Kokoro's own "t" / "d" content
frames, the "d" of "and" the sound before it. Read back against A (ContentVec cosine, A = 1): the "d"s 0.84 - 0.91
for "the" (B, no "d" change: 0.70 - 0.79), "think" 0.90 - 0.95; "and you" differs (A turned the "d" into a vowel
glide, C holds the "n"). Finding the sounds live needs a consonant recognizer: `consonant_feats.py` (LibriSpeech,
every phone labelled).

In the plug-in (`Source/neural/AccentConsonants`, round 29): the consonant recognizer (the vowel recognizer's MLP,
`vowelnet/train.py` on LibriSpeech dev-clean: `consonant_feats.py`, whole files, a third of the utterances, and
`consonant_stream.py`, the windows the live stream computes, a ninth; held-out speakers: th 0.92, dh 0.89, d 0.87,
r 0.84 of frames) and `export_consonants.py` (its weights, Kokoro's "t" / "d" frames, LibriSpeech's mean silence ->
`Resources/neural/accent_consonants.gz`). Its rule runs after the vowels' with the same 2 frames of lookahead: a
"th" gets closure (a "d": voice bar) and its template's last two frames at its last two frames, a "t" no pitch, a
"d" the next vowel's; a "d" / "t" up to 2 frames after an "n" is sung as that "n"; an "r" of at least 4 frames after
a vowel is sung as the vowel from its 2nd frame. The consonant, vowel and r keepers let go where it changed the
sound. On the user's take (recognizer read-back): Render (`VoxaVersaRender --export`, the whole take) finds them
all; live, the stream hears only ~60 ms after each frame and takes the user's "think" for "f" / "s" for its first
half and "this" for "t" (sung as it is), the rest as Render.

Round 29 verdict: Render sounds like the prototype on all six (Reggae, Afrobeats good; Latin iffy, as the prototype:
"more Asian than Latin"); the DAW bounce ("live") iffy on all six, fixed by the stream's offline look-ahead (round 30:
"as good as Render").

## Latin, again (round 31)

Dropping the "r" after a vowel is Jamaican / Nigerian; with the pure vowels it read as East Asian English. A Spanish
speaker keeps the "r" and taps or rolls it, and often says "you" as "jou". `make_latin_templates.py` (Kokoro's
Spanish voice: "para", "pero", "perro"; its American "a June") and `accent_latin.py` (round 28's splice, Latin
without the dropped "r"): A, the "r" of "for" tapped (its vowel held from before it turns r-coloured); B, rolled in
"for" and "girl" (a tap right before the "l" of "girl" read back as "d"); both "jou".

Round 31 verdict: B (rolled r) for both Latin voices; Male Latin's "girl" "doesn't sound Latin". Round 32 ("girl" on
B): G2 "gurrl" (the singer's own "er", rolled r, light Spanish "l") for both; "and doesn't sound Latin" - the user
sings it "in-en". Round 33: N2 "en" (one pure "e") for both. Round 34 (the whole Latin accent through the plug-in):
Render good, the DAW export "quality seems less" - its "j" covered the whole "you" (a hard "dju") and its roll was
short; round 35 (a 60 % "j" on the "y"'s first 2 frames, the roll from the r's 3rd held frame to its end): good.

In the plug-in the Latin accent (singAccent 3) now differs from the Jamaican / Nigerian ones: the r after a vowel
held and rolled (AccentConsonants: Kokoro's Spanish "perro" trill), "y" -> a soft "j", "er" kept with its r rolled
before a light Spanish "l" ("ala"), and the "eh" English sings before "n" / "m" -> a pure "e" (SouthernVowels).
