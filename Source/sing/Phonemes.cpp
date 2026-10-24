#include "Phonemes.h"

namespace voxa::sing
{

namespace
{
    using M = Manner;
    using P = Place;

    // Durations are for sung (not spoken) English at a moderate tempo. They get compressed when
    // the rhythm leaves no room, never stretched.
    const PhonemeInfo phonemeTable[] = {
        //  symbol  manner        place            voiced onset  coda   min
        { "AA",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "AE",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "AH",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "AO",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "AW",  M::Diphthong, P::None,         true,  0,     0,     60 },
        { "AX",  M::Vowel,     P::None,         true,  0,     0,     30 },
        { "AY",  M::Diphthong, P::None,         true,  0,     0,     60 },
        { "EH",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "ER",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "EY",  M::Diphthong, P::None,         true,  0,     0,     60 },
        { "IH",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "IY",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "OW",  M::Diphthong, P::None,         true,  0,     0,     60 },
        { "OY",  M::Diphthong, P::None,         true,  0,     0,     60 },
        { "UH",  M::Vowel,     P::None,         true,  0,     0,     40 },
        { "UW",  M::Vowel,     P::None,         true,  0,     0,     40 },

        { "B",   M::Stop,      P::Labial,       true,  75,    70,    28 },
        { "CH",  M::Affricate, P::PostAlveolar, false, 125,   125,   50 },
        { "D",   M::Stop,      P::Alveolar,     true,  65,    65,    24 },
        { "DH",  M::Fricative, P::Dental,       true,  50,    55,    20 },
        { "F",   M::Fricative, P::LabioDental,  false, 105,   110,   40 },
        { "G",   M::Stop,      P::Velar,        true,  75,    75,    28 },
        { "HH",  M::Aspirate,  P::Glottal,      false, 80,    40,    35 },
        { "JH",  M::Affricate, P::PostAlveolar, true,  105,   100,   45 },
        { "K",   M::Stop,      P::Velar,        false, 100,   85,    35 },
        { "L",   M::Liquid,    P::Alveolar,     true,  70,    85,    30 },
        { "M",   M::Nasal,     P::Labial,       true,  80,    95,    35 },
        { "N",   M::Nasal,     P::Alveolar,     true,  70,    90,    30 },
        { "NG",  M::Nasal,     P::Velar,        true,  70,    100,   35 },
        { "P",   M::Stop,      P::Labial,       false, 100,   85,    35 },
        { "R",   M::Liquid,    P::Alveolar,     true,  70,    80,    30 },
        { "S",   M::Fricative, P::Alveolar,     false, 125,   130,   45 },
        { "SH",  M::Fricative, P::PostAlveolar, false, 125,   130,   45 },
        { "T",   M::Stop,      P::Alveolar,     false, 90,    75,    30 },
        { "TH",  M::Fricative, P::Dental,       false, 105,   110,   40 },
        { "V",   M::Fricative, P::LabioDental,  true,  75,    80,    30 },
        { "W",   M::Glide,     P::Labial,       true,  75,    50,    35 },
        { "Y",   M::Glide,     P::Palatal,      true,  65,    50,    30 },
        { "Z",   M::Fricative, P::Alveolar,     true,  90,    105,   35 },
        { "ZH",  M::Fricative, P::PostAlveolar, true,  90,    95,    35 },

        { "SIL", M::Silence,   P::None,         false, 0,     0,     0 },
        { "BR",  M::Breath,    P::None,         false, 0,     0,     0 },
    };

    static_assert (sizeof (phonemeTable) / sizeof (phonemeTable[0]) == (size_t) Ph::Count,
                   "phoneme table must match the Ph enum");
}

const PhonemeInfo& phonemeInfo (Ph p) noexcept
{
    const auto index = juce::jlimit (0, (int) Ph::Count - 1, (int) p);
    return phonemeTable[index];
}

const char* phonemeSymbol (Ph p) noexcept
{
    return phonemeInfo (p).symbol;
}

bool isSonorant (Ph p) noexcept
{
    const auto m = phonemeInfo (p).manner;
    return m == Manner::Vowel || m == Manner::Diphthong || m == Manner::Nasal
        || m == Manner::Liquid || m == Manner::Glide;
}

Ph phonemeFromSymbol (const char* text, int length) noexcept
{
    if (length < 1 || length > 3)
        return Ph::Count;

    char upper[4] = {};
    for (int i = 0; i < length; ++i)
        upper[i] = (char) juce::CharacterFunctions::toUpperCase ((juce::juce_wchar) text[i]);

    for (int i = 0; i < (int) Ph::Count; ++i)
        if (std::strcmp (phonemeTable[i].symbol, upper) == 0)
            return (Ph) i;

    return Ph::Count;
}

juce::String toString (const PhoneString& phones)
{
    juce::String s;
    for (const auto& p : phones)
    {
        if (s.isNotEmpty())
            s << ' ';

        if (p.ph == Ph::AX)
            s << "AH0";
        else
            s << phonemeSymbol (p.ph);

        if (p.ph != Ph::AX && isVowel (p.ph))
            s << juce::String (juce::jmax (0, (int) p.stress));
    }
    return s;
}

PhoneString phonesFromString (const juce::String& text)
{
    PhoneString result;
    for (auto token : juce::StringArray::fromTokens (text, " \t", ""))
    {
        token = token.trim();
        if (token.isEmpty())
            continue;

        int stress = -1;
        const auto last = token.getLastCharacter();
        if (last >= '0' && last <= '2')
        {
            stress = (int) (last - '0');
            token = token.dropLastCharacters (1);
        }

        const auto utf8 = token.toStdString();
        auto ph = phonemeFromSymbol (utf8.c_str(), (int) utf8.size());
        if (ph == Ph::Count)
            continue;

        if (isVowel (ph))
        {
            if (stress < 0)
                stress = 1;
            if (ph == Ph::AH && stress == 0)
                ph = Ph::AX;
        }
        else
        {
            stress = -1;
        }

        result.push_back ({ ph, (std::int8_t) stress });
    }
    return result;
}

} // namespace voxa::sing
