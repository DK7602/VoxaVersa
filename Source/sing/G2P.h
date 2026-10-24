#pragma once

#include "Phonemes.h"

namespace voxa::sing
{

/** One phonetic syllable: consonant onset, a sung nucleus and a coda. */
struct Syllable
{
    std::vector<Ph> onset;   // consonants sung before the beat
    Ph nucleus = Ph::AX;     // vowel, or a sonorant consonant for vowel-less words ("mm", "hmm")
    std::vector<Ph> coda;    // consonants sung at the end of the note
    int stress = 0;          // 0 unstressed, 1 primary, 2 secondary

    juce::String toString() const;
};

/** English grapheme-to-phoneme conversion: CMU Pronouncing Dictionary first, then morphological
    fallbacks (contractions, suffixes, compounds, elongated interjections) and finally the NRL
    letter-to-sound rules. All functions are thread-safe. */
namespace g2p
{
    /** Exact CMU dictionary lookup of a lower-case word. Loads the dictionary on first use. */
    bool lookupDictionary (const juce::String& lowerCaseWord, PhoneString& result);

    /** Number of words in the embedded dictionary (0 if it failed to load). */
    int dictionarySize();

    /** Letter-to-sound rules (Elovitz et al., NRL 1976) with a simple stress assignment. */
    PhoneString letterToSound (const juce::String& word);

    /** Lower-cases, folds accents and unicode apostrophes and strips everything but a-z and '. */
    juce::String normaliseWord (const juce::String& word);

    /** The complete pipeline. Returns an empty string for words without letters. */
    PhoneString pronounce (const juce::String& word);

    /** Vowel-nucleus syllabification with the maximal-onset principle. Vowel-less words get a
        sonorant (or schwa) nucleus so that every non-empty word has at least one syllable. */
    std::vector<Syllable> syllabify (const PhoneString& phones);

    /** True when `consonants` is a permissible English syllable onset. */
    bool isLegalOnset (const Ph* consonants, int count) noexcept;
}

} // namespace voxa::sing
