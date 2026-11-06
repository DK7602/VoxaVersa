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
