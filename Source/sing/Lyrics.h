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

    // Weak form of a function word, sung on short unaccented notes ("to" [tə], "for" [fɚ], "and" [ən]).
    // The dictionary gives the citation form; singing that on every quick pickup note sounds foreign.
    bool hasWeakForm = false;
    Ph weakNucleus = Ph::AX;
    std::vector<Ph> weakCoda;
    bool weakOnlyBeforeConsonant = false;   // "to" stays [tu] and "and" keeps its D before a vowel
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

/** The English a voice's style sings in (Text to Sing). The pronunciation decides what the AI singer sings, so
    an accent is made here, from the words, as the singers of these styles sing them. */
enum class Accent
{
    American = 0,
    Southern,    // country: "I" as "ah" (not before voiceless consonants), pin-pen merger, singin'
    Jamaican,    // reggae: th as t/d, no r after vowels, "day" as "deh", the as "di", singin', final t/d dropped
    Latin,       // Spanish-accented: i/ee and u/oo merge, a open, z as s, v as b, th as t/d
    Nigerian     // Afrobeats: th as t/d, no r after vowels, a open, "love" with an open o, i/ee and u/oo merge
};

/** Rewrites the sung syllables (and their weak forms) in `accent`. American leaves them as they are. */
void applyAccent (LyricScript& script, Accent accent);

/** "21" -> "twenty one", "1999" (as a year) -> "nineteen ninety nine". */
juce::String spellNumber (juce::int64 number, bool asYear = false);

/** Splits a written word into `numSyllables` display pieces ("beautiful", 3 -> beau|ti|ful). */
juce::StringArray splitWordText (const juce::String& word, int numSyllables);

/** True for unstressed grammatical words (the, a, of, to, and, ...). */
bool isFunctionWord (const juce::String& lowerCaseWord);

/** Fills in the weak form of a one-syllable function word whose dictionary (strong) form is in `s`.
    Returns false (and leaves `s` alone) for words without one. */
bool setWeakForm (const juce::String& lowerCaseWord, SungSyllable& s);

} // namespace voxa::sing
