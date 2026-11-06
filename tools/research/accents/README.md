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
