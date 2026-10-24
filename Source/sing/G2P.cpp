#include "G2P.h"
#include "VoxaAssets.h"

#include <juce_core/juce_core.h>
#include <algorithm>
#include <array>
#include <cstring>

namespace voxa::sing
{

juce::String Syllable::toString() const
{
    juce::String s;
    for (auto p : onset)
        s << phonemeSymbol (p) << ' ';
    s << '[' << phonemeSymbol (nucleus) << juce::String (stress) << ']';
    for (auto p : coda)
        s << ' ' << phonemeSymbol (p);
    return s;
}

namespace g2p
{

//==============================================================================
namespace
{
    /** The embedded CMU dictionary, decompressed once on first use.

        Asset format (see docs/THIRD_PARTY.md): gzip of
            "#VOXALEX 1 <sym0> <sym1> ...\n" followed by sorted "<word>\t<codes>\n" lines,
        where phoneme symbol i is encoded as the byte 0x30 + i. */
    class CmuDictionary
    {
    public:
        static const CmuDictionary& get()
        {
            static const CmuDictionary instance;   // thread-safe lazy initialisation
            return instance;
        }

        int size() const noexcept   { return (int) entries.size(); }

        bool lookup (const std::string& word, PhoneString& out) const
        {
            if (entries.empty() || word.empty())
                return false;

            auto it = std::lower_bound (entries.begin(), entries.end(), word,
                                        [this] (std::uint32_t offset, const std::string& w)
                                        {
                                            return compare (offset, w) < 0;
                                        });

            if (it == entries.end() || compare (*it, word) != 0)
                return false;

            const auto* text = static_cast<const char*> (data.getData());
            const auto total = data.getSize();
            size_t pos = *it + word.size() + 1;   // skip word and tab

            out.clear();
            while (pos < total && text[pos] != '\n')
            {
                const auto c = (unsigned char) text[pos++];
                if (validCode[c])
                    out.push_back (codes[c]);
            }
            return ! out.empty();
        }

    private:
        CmuDictionary()
        {
            juce::MemoryInputStream compressed (VoxaAssets::cmudict_gz, (size_t) VoxaAssets::cmudict_gzSize, false);
            juce::GZIPDecompressorInputStream gzip (&compressed, false, juce::GZIPDecompressorInputStream::gzipFormat);
            juce::MemoryOutputStream decompressed (data, false);
            decompressed.writeFromInputStream (gzip, -1);
            decompressed.flush();

            const auto* text = static_cast<const char*> (data.getData());
            const auto total = data.getSize();
            if (total < 16 || std::strncmp (text, "#VOXALEX", 8) != 0)
            {
                jassertfalse;   // corrupt asset
                return;
            }

            size_t pos = 0;
            while (pos < total && text[pos] != '\n')
                ++pos;

            parseHeader (juce::String::fromUTF8 (text, (int) pos));

            ++pos;
            entries.reserve (130000);
            while (pos < total)
            {
                entries.push_back ((std::uint32_t) pos);
                while (pos < total && text[pos] != '\n')
                    ++pos;
                ++pos;
            }

            // The packer sorts bytewise, but never trust an asset blindly.
            if (! std::is_sorted (entries.begin(), entries.end(),
                                  [this] (std::uint32_t a, std::uint32_t b) { return compareEntries (a, b) < 0; }))
            {
                std::sort (entries.begin(), entries.end(),
                           [this] (std::uint32_t a, std::uint32_t b) { return compareEntries (a, b) < 0; });
            }
        }

        void parseHeader (const juce::String& header)
        {
            auto tokens = juce::StringArray::fromTokens (header, " ", "");
            for (int i = 2; i < tokens.size(); ++i)
            {
                const auto code = 0x30 + (i - 2);
                if (code > 255)
                    break;

                auto phones = phonesFromString (tokens[i]);
                if (phones.size() == 1)
                {
                    codes[(size_t) code] = phones.front();
                    validCode[(size_t) code] = true;
                }
            }
        }

        int compare (std::uint32_t offset, const std::string& word) const noexcept
        {
            const auto* text = static_cast<const char*> (data.getData()) + offset;
            const auto remaining = data.getSize() - offset;
            size_t i = 0;
            for (;; ++i)
            {
                const bool entryEnded = i >= remaining || text[i] == '\t' || text[i] == '\n';
                const bool wordEnded = i >= word.size();
                if (entryEnded || wordEnded)
                    return entryEnded && wordEnded ? 0 : (entryEnded ? -1 : 1);

                const auto a = (unsigned char) text[i];
                const auto b = (unsigned char) word[i];
                if (a != b)
                    return a < b ? -1 : 1;
            }
        }

        int compareEntries (std::uint32_t a, std::uint32_t b) const noexcept
        {
            const auto* text = static_cast<const char*> (data.getData());
            size_t end = a;
            while (text[end] != '\t' && text[end] != '\n')
                ++end;
            return -compare (b, std::string (text + a, end - a));
        }

        juce::MemoryBlock data;
        std::vector<std::uint32_t> entries;
        std::array<Phone, 256> codes {};
        std::array<bool, 256> validCode {};
    };

    //==============================================================================
    bool lookup (const juce::String& word, PhoneString& out)
    {
        return CmuDictionary::get().lookup (word.toStdString(), out);
    }

    bool endsWith (const juce::String& w, const char* suffix)
    {
        return w.endsWith (suffix);
    }

    Ph lastPh (const PhoneString& p)   { return p.empty() ? Ph::Sil : p.back().ph; }

    void append (PhoneString& p, const char* arpabet)
    {
        for (auto& ph : phonesFromString (arpabet))
            p.push_back (ph);
    }

    bool isOneOf (Ph p, std::initializer_list<Ph> set) noexcept
    {
        return std::find (set.begin(), set.end(), p) != set.end();
    }

    void appendPlural (PhoneString& p)
    {
        const auto last = lastPh (p);
        if (isOneOf (last, { Ph::S, Ph::Z, Ph::SH, Ph::ZH, Ph::CH, Ph::JH }))
            append (p, "IH0 Z");
        else if (isOneOf (last, { Ph::P, Ph::T, Ph::K, Ph::F, Ph::TH }))
            append (p, "S");
        else
            append (p, "Z");
    }

    void appendPast (PhoneString& p)
    {
        const auto last = lastPh (p);
        if (isOneOf (last, { Ph::T, Ph::D }))
            append (p, "IH0 D");
        else if (isOneOf (last, { Ph::P, Ph::K, Ph::F, Ph::S, Ph::SH, Ph::CH, Ph::TH }))
            append (p, "T");
        else
            append (p, "D");
    }

    /** Sung interjections and vocables that the dictionary lacks or pronounces as letter names. */
    bool lookupInterjection (const juce::String& w, PhoneString& out)
    {
        struct Entry { const char* word; const char* phones; };
        static const Entry table[] = {
            { "m", "M" }, { "mm", "M" }, { "hm", "HH M" }, { "hmm", "HH M" },
            { "oo", "UW1" }, { "ooh", "UW1" }, { "ah", "AA1" }, { "aah", "AA1" }, { "aa", "AA1" },
            { "o", "OW1" }, { "oh", "OW1" }, { "ohh", "OW1" }, { "uh", "AH1" }, { "uhh", "AH1" },
            { "la", "L AA1" }, { "na", "N AA1" }, { "da", "D AA1" }, { "ba", "B AA1" }, { "doo", "D UW1" },
            { "yeah", "Y EH1" }, { "yea", "Y EH1" }, { "woah", "W OW1" }, { "whoa", "W OW1" },
            { "ayy", "EY1" }, { "ay", "AY1" }, { "eh", "EH1" }, { "yo", "Y OW1" }, { "ya", "Y AA1" },
            { "imma", "AY1 M AH0" }, { "ima", "AY1 M AH0" }, { "n", "AH0 N" }, { "tryna", "T R AY1 N AH0" },
        };

        for (const auto& e : table)
        {
            if (w == e.word)
            {
                out = phonesFromString (e.phones);
                return true;
            }
        }
        return false;
    }

    juce::String collapseRuns (const juce::String& w, int maxRun)
    {
        juce::String result;
        juce::juce_wchar previous = 0;
        int run = 0;
        for (auto c : w)
        {
            run = (c == previous) ? run + 1 : 1;
            previous = c;
            if (run <= maxRun)
                result += c;
        }
        return result;
    }

    bool pronounceCore (const juce::String& w, PhoneString& out, int depth);

    bool tryApostrophes (const juce::String& w, PhoneString& out, int depth)
    {
        if (! w.containsChar ('\''))
            return false;

        if (w.startsWithChar ('\''))
            return pronounceCore (w.substring (1), out, depth + 1);

        if (endsWith (w, "in'"))
        {
            // g-dropping: lovin' -> loving with a final N instead of NG
            if (lookup (w.dropLastCharacters (1) + "g", out) && lastPh (out) == Ph::NG)
            {
                out.back().ph = Ph::N;
                return true;
            }
        }

        if (w.endsWithChar ('\''))
            return pronounceCore (w.dropLastCharacters (1), out, depth + 1);

        struct Clitic { const char* suffix; const char* afterVowel; const char* afterConsonant; };
        static const Clitic clitics[] = {
            { "n't", "N T", "AH0 N T" },
            { "'ll", "L", "AH0 L" },
            { "'re", "R", "ER0" },
            { "'ve", "V", "AH0 V" },
            { "'d",  "D", "AH0 D" },
            { "'m",  "M", "AH0 M" },
        };

        for (const auto& c : clitics)
        {
            if (! endsWith (w, c.suffix))
                continue;

            const auto base = w.dropLastCharacters ((int) std::strlen (c.suffix));
            if (base.isEmpty() || ! pronounceCore (base, out, depth + 1))
                return false;

            append (out, isVowel (lastPh (out)) ? c.afterVowel : c.afterConsonant);
            return true;
        }

        if (endsWith (w, "'s"))
        {
            const auto base = w.dropLastCharacters (2);
            if (base.isNotEmpty() && pronounceCore (base, out, depth + 1))
            {
                appendPlural (out);
                return true;
            }
        }

        return false;
    }

    bool trySuffixes (const juce::String& w, PhoneString& out)
    {
        enum Kind { Plain, Plural, Past };
        struct Suffix { const char* text; const char* phones; Kind kind; };
        static const Suffix suffixes[] = {
            { "ingly", "IH0 NG L IY0", Plain }, { "ings", "IH0 NG Z", Plain }, { "ing", "IH0 NG", Plain },
            { "in", "IH0 N", Plain }, { "ers", "ER0 Z", Plain }, { "er", "ER0", Plain },
            { "est", "AH0 S T", Plain }, { "ly", "L IY0", Plain }, { "ness", "N AH0 S", Plain },
            { "less", "L AH0 S", Plain }, { "ful", "F AH0 L", Plain }, { "ment", "M AH0 N T", Plain },
            { "able", "AH0 B AH0 L", Plain }, { "ish", "IH0 SH", Plain }, { "ism", "IH0 Z AH0 M", Plain },
            { "ist", "IH0 S T", Plain }, { "ity", "IH0 T IY0", Plain }, { "ize", "AY2 Z", Plain },
            { "y", "IY0", Plain }, { "es", "", Plural }, { "s", "", Plural }, { "ed", "", Past },
            { "d", "", Past },
        };

        for (const auto& s : suffixes)
        {
            const auto suffixLength = (int) std::strlen (s.text);
            if (! endsWith (w, s.text) || w.length() - suffixLength < 2)
                continue;

            const auto base = w.dropLastCharacters (suffixLength);
            juce::StringArray candidates { base, base + "e" };
            const auto n = base.length();
            if (n >= 3 && base[n - 1] == base[n - 2])
                candidates.add (base.dropLastCharacters (1));       // running -> run
            if (base.endsWithChar ('i'))
                candidates.add (base.dropLastCharacters (1) + "y");  // happiness -> happy

            for (const auto& candidate : candidates)
            {
                if (! lookup (candidate, out))
                    continue;

                if (s.kind == Plural)
                    appendPlural (out);
                else if (s.kind == Past)
                    appendPast (out);
                else
                    append (out, s.phones);
                return true;
            }
        }
        return false;
    }

    bool tryCompound (const juce::String& w, PhoneString& out)
    {
        const int n = w.length();
        if (n < 6)
            return false;

        int bestSplit = -1, bestScore = 0;
        PhoneString left, right;
        for (int i = 3; i <= n - 3; ++i)
        {
            const int score = juce::jmin (i, n - i);
            if (score <= bestScore)
                continue;

            if (lookup (w.substring (0, i), left) && lookup (w.substring (i), right))
            {
                bestScore = score;
                bestSplit = i;
            }
        }

        if (bestSplit < 0)
            return false;

        lookup (w.substring (0, bestSplit), left);
        lookup (w.substring (bestSplit), right);

        // The second element of a compound carries secondary stress.
        for (auto& p : right)
            if (p.stress == 1)
                p.stress = 2;

        out = left;
        out.insert (out.end(), right.begin(), right.end());
        return true;
    }

    bool pronounceCore (const juce::String& w, PhoneString& out, int depth)
    {
        if (w.isEmpty() || depth > 3 || ! w.containsAnyOf ("abcdefghijklmnopqrstuvwxyz"))
            return false;

        if (lookupInterjection (w, out) || lookup (w, out))
            return true;

        if (tryApostrophes (w, out, depth))
            return true;

        const auto plain = w.removeCharacters ("'");
        if (plain != w)
            return pronounceCore (plain, out, depth + 1);

        // Elongated words: "sooo", "yeahhh", "ooooh"
        for (int maxRun : { 2, 1 })
        {
            const auto collapsed = collapseRuns (w, maxRun);
            if (collapsed != w && (lookupInterjection (collapsed, out) || lookup (collapsed, out)))
                return true;
        }

        // g-dropping without an apostrophe: "nothin", "somethin"
        if (endsWith (w, "in") && lookup (w + "g", out) && lastPh (out) == Ph::NG)
        {
            out.back().ph = Ph::N;
            return true;
        }

        if (trySuffixes (w, out) || tryCompound (w, out))
            return true;

        out = letterToSound (w);
        return ! out.empty();
    }

    //==============================================================================
    bool isOnsetPair (Ph a, Ph b) noexcept
    {
        if (a == Ph::P || a == Ph::B || a == Ph::F)  return isOneOf (b, { Ph::R, Ph::L, Ph::Y });
        if (a == Ph::T || a == Ph::D || a == Ph::TH) return isOneOf (b, { Ph::R, Ph::W });
        if (a == Ph::K || a == Ph::G)                return isOneOf (b, { Ph::R, Ph::L, Ph::W, Ph::Y });
        if (a == Ph::V || a == Ph::M)                return b == Ph::Y;
        if (a == Ph::SH)                             return b == Ph::R;
        if (a == Ph::HH)                             return b == Ph::Y || b == Ph::W;
        if (a == Ph::S)                              return isOneOf (b, { Ph::P, Ph::T, Ph::K, Ph::M, Ph::N, Ph::L, Ph::W, Ph::F });
        return false;
    }
}

//==============================================================================
bool isLegalOnset (const Ph* c, int count) noexcept
{
    if (count <= 0)
        return true;

    if (count == 1)
        return isConsonant (c[0]) && c[0] != Ph::NG;

    if (count == 2)
        return isOnsetPair (c[0], c[1]);

    if (count == 3)
    {
        if (c[0] != Ph::S)
            return false;

        if (c[1] == Ph::P) return c[2] == Ph::R || c[2] == Ph::L || c[2] == Ph::Y;
        if (c[1] == Ph::T) return c[2] == Ph::R;
        if (c[1] == Ph::K) return c[2] == Ph::R || c[2] == Ph::L || c[2] == Ph::W || c[2] == Ph::Y;
    }
    return false;
}

bool lookupDictionary (const juce::String& lowerCaseWord, PhoneString& result)
{
    return lookup (lowerCaseWord, result);
}

int dictionarySize()
{
    return CmuDictionary::get().size();
}

juce::String normaliseWord (const juce::String& word)
{
    juce::String result;
    result.preallocateBytes ((size_t) word.length() + 4);

    for (auto c : word)
    {
        c = juce::CharacterFunctions::toLowerCase (c);

        if (c >= 'a' && c <= 'z')
        {
            result += c;
            continue;
        }

        switch (c)
        {
            case '\'': case 0x2019: case 0x2018: case 0x02bc: case 0x2032: case '`': case 0x00b4:
                result += '\''; break;
            case 0xe0: case 0xe1: case 0xe2: case 0xe3: case 0xe4: case 0xe5: result += 'a'; break;
            case 0xe7: result += 'c'; break;
            case 0xe8: case 0xe9: case 0xea: case 0xeb: result += 'e'; break;
            case 0xec: case 0xed: case 0xee: case 0xef: result += 'i'; break;
            case 0xf1: result += 'n'; break;
            case 0xf2: case 0xf3: case 0xf4: case 0xf5: case 0xf6: case 0xf8: result += 'o'; break;
            case 0xf9: case 0xfa: case 0xfb: case 0xfc: result += 'u'; break;
            case 0xfd: case 0xff: result += 'y'; break;
            case 0xe6: result += "ae"; break;
            case 0x153: result += "oe"; break;
            case 0xdf: result += "ss"; break;
            default: break;
        }
    }
    return result;
}

PhoneString pronounce (const juce::String& word)
{
    const auto w = normaliseWord (word);
    PhoneString result;
    if (! pronounceCore (w, result, 0))
        result.clear();
    return result;
}

std::vector<Syllable> syllabify (const PhoneString& phones)
{
    std::vector<Syllable> syllables;
    if (phones.empty())
        return syllables;

    std::vector<int> nuclei;
    for (int i = 0; i < (int) phones.size(); ++i)
        if (isVowel (phones[(size_t) i].ph))
            nuclei.push_back (i);

    if (nuclei.empty())
    {
        // "hmm", "mm": hum on the last sonorant. Otherwise add a schwa ("shh", "psst").
        Syllable s;
        s.stress = 1;
        int sonorant = -1;
        for (int i = 0; i < (int) phones.size(); ++i)
            if (isSonorant (phones[(size_t) i].ph))
                sonorant = i;

        if (sonorant >= 0)
        {
            for (int i = 0; i < sonorant; ++i)
                s.onset.push_back (phones[(size_t) i].ph);
            s.nucleus = phones[(size_t) sonorant].ph;
            for (int i = sonorant + 1; i < (int) phones.size(); ++i)
                s.coda.push_back (phones[(size_t) i].ph);
        }
        else
        {
            for (const auto& p : phones)
                s.onset.push_back (p.ph);
            s.nucleus = Ph::AX;
        }
        syllables.push_back (std::move (s));
        return syllables;
    }

    // Boundary k: index of the first phone of syllable k.
    std::vector<int> starts { 0 };
    for (size_t k = 1; k < nuclei.size(); ++k)
    {
        const int prevNucleus = nuclei[k - 1];
        const int nextNucleus = nuclei[k];
        const int firstConsonant = prevNucleus + 1;
        const int numConsonants = nextNucleus - firstConsonant;

        std::array<Ph, 16> cluster {};
        const int n = juce::jmin (numConsonants, (int) cluster.size());
        for (int i = 0; i < n; ++i)
            cluster[(size_t) i] = phones[(size_t) (firstConsonant + i)].ph;

        // Maximal onset: give the next syllable the longest legal tail of the cluster.
        int onsetLength = 0;
        for (int len = juce::jmin (n, 3); len > 0; --len)
        {
            if (isLegalOnset (cluster.data() + (n - len), len))
            {
                onsetLength = len;
                break;
            }
        }
        starts.push_back (nextNucleus - onsetLength);
    }
    starts.push_back ((int) phones.size());

    for (size_t k = 0; k < nuclei.size(); ++k)
    {
        Syllable s;
        const int nucleus = nuclei[k];
        for (int i = starts[k]; i < nucleus; ++i)
            s.onset.push_back (phones[(size_t) i].ph);
        s.nucleus = phones[(size_t) nucleus].ph;
        s.stress = juce::jmax (0, (int) phones[(size_t) nucleus].stress);
        for (int i = nucleus + 1; i < starts[k + 1]; ++i)
            s.coda.push_back (phones[(size_t) i].ph);
        syllables.push_back (std::move (s));
    }
    return syllables;
}

} // namespace g2p
} // namespace voxa::sing
