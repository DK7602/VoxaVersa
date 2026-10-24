#pragma once

#include <juce_core/juce_core.h>
#include <cstdint>
#include <vector>

namespace voxa::sing
{

/** ARPAbet phoneme set used by the CMU Pronouncing Dictionary, plus a reduced schwa (AX, used for
    unstressed AH0), silence and an inhalation breath. The vowels come first so isVowel() is a compare. */
enum class Ph : std::uint8_t
{
    AA, AE, AH, AO, AW, AX, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW,
    B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, P, R, S, SH, T, TH, V, W, Y, Z, ZH,
    Sil, Breath,
    Count
};

enum class Manner : std::uint8_t
{
    Vowel, Diphthong, Stop, Affricate, Fricative, Aspirate, Nasal, Liquid, Glide, Silence, Breath
};

enum class Place : std::uint8_t
{
    None, Labial, LabioDental, Dental, Alveolar, PostAlveolar, Palatal, Velar, Glottal
};

struct PhonemeInfo
{
    const char* symbol;
    Manner manner;
    Place place;
    bool voiced;
    float onsetMs;   // nominal sung duration as part of a syllable onset
    float codaMs;    // nominal sung duration as part of a syllable coda
    float minMs;     // shortest duration when squeezed into fast passages
};

const PhonemeInfo& phonemeInfo (Ph) noexcept;
const char* phonemeSymbol (Ph) noexcept;

constexpr bool isVowel (Ph p) noexcept           { return p <= Ph::UW; }
inline bool isDiphthong (Ph p) noexcept          { return phonemeInfo (p).manner == Manner::Diphthong; }
inline bool isVoiced (Ph p) noexcept             { return phonemeInfo (p).voiced; }
bool isSonorant (Ph) noexcept;                   // vowels, nasals, liquids and glides
inline bool isConsonant (Ph p) noexcept          { return p >= Ph::B && p <= Ph::ZH; }

/** Parses an ARPAbet symbol ("AA", "zh", ...). Returns Ph::Count if unknown. */
Ph phonemeFromSymbol (const char* text, int length) noexcept;

/** A phoneme with its CMU lexical stress: -1 for consonants, 0 unstressed, 1 primary, 2 secondary. */
struct Phone
{
    Ph ph = Ph::Sil;
    std::int8_t stress = -1;

    bool operator== (const Phone& other) const noexcept   { return ph == other.ph && stress == other.stress; }
};

using PhoneString = std::vector<Phone>;

/** "HH AH0 L OW1" style string, mainly for tests and debugging. */
juce::String toString (const PhoneString&);

/** Parses "HH AH0 L OW1" (AH0 becomes AX). Unknown symbols are skipped. */
PhoneString phonesFromString (const juce::String&);

} // namespace voxa::sing
