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
