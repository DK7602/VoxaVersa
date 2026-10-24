#pragma once

#include "G2P.h"

namespace voxa::sing
{

/** A syllable as it will be sung: display text plus its phonetic make-up. */
struct SungSyllable
{
    juce::String text;            // display text for this syllable, e.g. "beau"
    std::vector<Ph> onset;        // sung before the beat
    Ph nucleus = Ph::AX;          // the sustained sound
    std::vector<Ph> coda;         // sung at the end of the note
    int stress = 0;               // 0 unstressed, 1 primary, 2 secondary (function words are 0)
    int wordIndex = 0;
    int phraseIndex = 0;
    bool wordStart = true;
    bool wordEnd = true;
    bool functionWord = false;
    bool pauseAfter = false;      // punctuation such as a comma follows this syllable
};

struct LyricEvent
{
    enum class Type
    {
        Syllable,     // a new syllable starts (one note)
        Melisma,      // "_": the previous syllable continues onto the next note
        Breath,       // "[br]": explicit breath / rest
        PhraseEnd     // end of a lyric line
    };

    Type type = Type::Syllable;
    int syllable = -1;            // Syllable: its index. Melisma: the syllable being held.
};

/** Parsed lyrics. Lines are phrases. */
struct LyricScript
{
    std::vector<SungSyllable> syllables;
    std::vector<LyricEvent> events;
    juce::StringArray words;      // display words, indexed by SungSyllable::wordIndex
    int numPhrases = 0;

    bool isEmpty() const noexcept   { return syllables.empty(); }
    int getNumSyllables() const noexcept   { return (int) syllables.size(); }
};

/** Lyrics syntax:
      - words separated by spaces; each line is a phrase (a breath is taken between phrases)
      - "beau-ti-ful"  explicit syllable splits
      - "_"            melisma: the previous syllable continues onto the next note ("love _ _", "love__")
      - "[br]"         explicit breath / rest
      - punctuation is stripped (commas etc. become phrasing hints), numbers are spelled out,
        contractions ("don't", "I'm", "gonna", "lovin'") and unicode apostrophes are handled. */
LyricScript parseLyrics (const juce::String& lyrics);

/** "21" -> "twenty one", "1999" (as a year) -> "nineteen ninety nine". */
juce::String spellNumber (juce::int64 number, bool asYear = false);

/** Splits a written word into `numSyllables` display pieces ("beautiful", 3 -> beau|ti|ful). */
juce::StringArray splitWordText (const juce::String& word, int numSyllables);

/** True for unstressed grammatical words (the, a, of, to, and, ...). */
bool isFunctionWord (const juce::String& lowerCaseWord);

} // namespace voxa::sing
