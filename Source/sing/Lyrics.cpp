#include "Lyrics.h"

namespace voxa::sing
{

namespace
{
    bool isApostrophe (juce::juce_wchar c) noexcept
    {
        return c == '\'' || c == 0x2019 || c == 0x2018 || c == 0x02bc || c == 0x2032 || c == '`' || c == 0x00b4;
    }

    bool isPausePunctuation (juce::juce_wchar c) noexcept
    {
        return c == ',' || c == '.' || c == ';' || c == ':' || c == '!' || c == '?'
            || c == 0x2026 /* ellipsis */ || c == 0x2014 /* em dash */ || c == 0x2013 /* en dash */;
    }

    /** Letters (any script, so accented Latin survives until normalisation), digits, apostrophes,
        hyphens and melisma underscores. Locale-independent. */
    bool isWordCharacter (juce::juce_wchar c) noexcept
    {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
            return true;
        if (isApostrophe (c) || c == '-' || c == '_')
            return true;
        return c >= 0xc0 && c < 0x2000 && c != 0xd7 && c != 0xf7;
    }

    struct RawToken
    {
        enum class Kind { Word, Breath };
        Kind kind = Kind::Word;
        juce::String text;
        bool pauseAfter = false;
    };

    std::vector<RawToken> tokeniseLine (const juce::String& line)
    {
        std::vector<RawToken> tokens;
        juce::String current;

        auto flush = [&]
        {
            if (current.isNotEmpty())
            {
                RawToken t;
                t.text = current;
                tokens.push_back (t);
                current.clear();
            }
        };

        auto markPause = [&]
        {
            flush();
            for (auto it = tokens.rbegin(); it != tokens.rend(); ++it)
            {
                if (it->kind == RawToken::Kind::Word)
                {
                    it->pauseAfter = true;
                    break;
                }
            }
        };

        auto addWord = [&] (const char* w)
        {
            flush();
            RawToken t;
            t.text = w;
            tokens.push_back (t);
        };

        const int n = line.length();
        for (int i = 0; i < n; ++i)
        {
            const auto c = line[i];

            if (c == '[' || c == '{' || c == '<')
            {
                // [br] / [breath] are breaths; any other tag ("[Chorus]", "{x2}") is ignored.
                const juce::juce_wchar close = c == '[' ? ']' : (c == '{' ? '}' : '>');
                int end = i + 1;
                while (end < n && line[end] != close)
                    ++end;

                const auto tag = line.substring (i + 1, end).trim().toLowerCase();
                flush();
                if (tag == "br" || tag == "breath" || tag == "b")
                {
                    RawToken t;
                    t.kind = RawToken::Kind::Breath;
                    tokens.push_back (t);
                }
                i = end;
                continue;
            }

            if (isWordCharacter (c))
            {
                current += c;
                continue;
            }

            if (juce::CharacterFunctions::isWhitespace (c))
            {
                flush();
                continue;
            }

            if (isPausePunctuation (c))
            {
                markPause();
                continue;
            }

            switch (c)
            {
                case '&': addWord ("and"); break;
                case '@': addWord ("at"); break;
                case '%': addWord ("percent"); break;
                case '+': addWord ("plus"); break;
                default:  flush(); break;
            }
        }
        flush();

        // A hyphen surrounded by spaces (or a double hyphen) is a dash, i.e. a pause.
        std::vector<RawToken> cleaned;
        for (auto& t : tokens)
        {
            if (t.kind == RawToken::Kind::Word && t.text.containsOnly ("-"))
            {
                if (! cleaned.empty())
                    cleaned.back().pauseAfter = true;
                continue;
            }
            if (t.kind == RawToken::Kind::Word && t.text.contains ("--"))
            {
                // "love--you": split into two words with a pause between
                auto parts = juce::StringArray::fromTokens (t.text.replace ("--", " "), " ", "");
                for (int p = 0; p < parts.size(); ++p)
                {
                    RawToken w;
                    w.text = parts[p];
                    w.pauseAfter = (p < parts.size() - 1) || t.pauseAfter;
                    cleaned.push_back (w);
                }
                continue;
            }
            cleaned.push_back (t);
        }
        return cleaned;
    }

    const char* const ones[] = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                 "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                                 "eighteen", "nineteen" };
    const char* const tens[] = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

    juce::String toOrdinal (const juce::String& cardinal)
    {
        auto words = juce::StringArray::fromTokens (cardinal, " ", "");
        if (words.isEmpty())
            return cardinal;

        auto& last = words.getReference (words.size() - 1);
        struct Irregular { const char* from; const char* to; };
        static const Irregular irregular[] = { { "one", "first" }, { "two", "second" }, { "three", "third" },
                                               { "five", "fifth" }, { "eight", "eighth" }, { "nine", "ninth" },
                                               { "twelve", "twelfth" } };
        bool done = false;
        for (const auto& ir : irregular)
        {
            if (last == ir.from)
            {
                last = ir.to;
                done = true;
                break;
            }
        }
        if (! done)
            last = last.endsWithChar ('y') ? last.dropLastCharacters (1) + "ieth" : last + "th";

        return words.joinIntoString (" ");
    }

    /** Expands digits inside a token: "21" -> {"twenty", "one"}, "2nite" -> {"two", "nite"}. */
    juce::StringArray expandDigits (const juce::String& token)
    {
        juce::StringArray out;
        int i = 0;
        const int n = token.length();
        while (i < n)
        {
            const bool digit = juce::CharacterFunctions::isDigit (token[i]);
            int j = i;
            while (j < n && juce::CharacterFunctions::isDigit (token[j]) == digit)
                ++j;

            auto run = token.substring (i, j);
            if (digit)
            {
                const auto value = run.getLargeIntValue();
                const bool yearLike = run.length() == 4 && ((value >= 1100 && value <= 1999) || (value >= 2010 && value <= 2099));
                auto spelled = spellNumber (value, yearLike);

                // Ordinals: 1st, 2nd, 3rd, 4th
                const auto rest = token.substring (j).toLowerCase();
                if (rest.startsWith ("st") || rest.startsWith ("nd") || rest.startsWith ("rd") || rest.startsWith ("th"))
                {
                    spelled = toOrdinal (spelled);
                    j += 2;
                }
                out.addTokens (spelled, " ", "");
            }
            else
            {
                out.add (run);
            }
            i = j;
        }
        out.removeEmptyStrings();
        return out;
    }

    struct WordSpec
    {
        juce::StringArray pieces;   // explicit hyphen splits (one piece if none)
        juce::String display;
        int leadingMelisma = 0;
        int trailingMelisma = 0;
        bool pauseAfter = false;
    };

    void appendSyllables (const juce::String& pieceText, const std::vector<Syllable>& phonetic,
                          std::vector<std::pair<juce::String, Syllable>>& out)
    {
        const auto texts = splitWordText (pieceText, (int) phonetic.size());
        for (size_t k = 0; k < phonetic.size(); ++k)
            out.emplace_back (texts[(int) k], phonetic[k]);
    }

    /** Phonetic syllables for one written word with display text per syllable. */
    std::vector<std::pair<juce::String, Syllable>> syllablesForWord (const WordSpec& spec)
    {
        std::vector<std::pair<juce::String, Syllable>> result;

        if (spec.pieces.size() > 1)
        {
            // Prefer the dictionary pronunciation of the joined word when its syllables line up 1:1.
            PhoneString joined;
            if (g2p::lookupDictionary (g2p::normaliseWord (spec.pieces.joinIntoString ("")), joined))
            {
                const auto syl = g2p::syllabify (joined);
                if ((int) syl.size() == spec.pieces.size())
                {
                    for (int k = 0; k < spec.pieces.size(); ++k)
                        result.emplace_back (spec.pieces[k], syl[(size_t) k]);
                    return result;
                }
            }

            for (const auto& piece : spec.pieces)
            {
                const auto phones = g2p::pronounce (piece);
                if (! phones.empty())
                    appendSyllables (piece, g2p::syllabify (phones), result);
            }
            return result;
        }

        const auto phones = g2p::pronounce (spec.display);
        if (! phones.empty())
            appendSyllables (spec.display, g2p::syllabify (phones), result);
        return result;
    }

    bool isVowelLetter (juce::juce_wchar c) noexcept
    {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
}

//==============================================================================
juce::String spellNumber (juce::int64 n, bool asYear)
{
    if (n < 0)
        return "minus " + spellNumber (-n);

    if (asYear && n >= 1100 && n <= 2099 && ! (n >= 2000 && n <= 2009))
    {
        const auto hi = n / 100, lo = n % 100;
        if (lo == 0)
            return spellNumber (hi) + " hundred";
        return spellNumber (hi) + (lo < 10 ? " oh " : " ") + spellNumber (lo);
    }

    if (n < 20)
        return ones[n];

    if (n < 100)
        return juce::String (tens[n / 10]) + (n % 10 != 0 ? juce::String (" ") + ones[n % 10] : juce::String());

    if (n < 1000)
        return juce::String (ones[n / 100]) + " hundred" + (n % 100 != 0 ? " " + spellNumber (n % 100) : juce::String());

    struct Scale { juce::int64 value; const char* name; };
    static const Scale scales[] = { { 1000000000000LL, "trillion" }, { 1000000000LL, "billion" },
                                    { 1000000LL, "million" }, { 1000LL, "thousand" } };

    for (const auto& s : scales)
    {
        if (n >= s.value)
        {
            const auto rest = n % s.value;
            return spellNumber (n / s.value) + " " + s.name + (rest != 0 ? " " + spellNumber (rest) : juce::String());
        }
    }
    return {};
}

bool isFunctionWord (const juce::String& w)
{
    static const juce::StringArray words {
        "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on", "at", "for", "with", "from", "by",
        "as", "into", "onto", "is", "am", "are", "was", "were", "be", "been", "it", "its", "it's", "me", "my",
        "your", "you're", "him", "his", "her", "we", "us", "our", "they", "them", "their", "that", "than", "then",
        "so", "if", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "has", "have", "had", "i'm", "i'll", "i've", "i'd", "we're", "they're", "he's", "she's", "there",
        "this", "these", "those", "some", "just", "such", "upon", "a'", "'n'", "n", "o'", "ya", "da"
    };
    return words.contains (w);
}

juce::StringArray splitWordText (const juce::String& word, int numSyllables)
{
    juce::StringArray result;
    if (numSyllables <= 1 || word.length() < 2)
    {
        result.add (word);
        for (int i = 1; i < numSyllables; ++i)
            result.add ({});
        return result;
    }

    const auto lower = word.toLowerCase();
    const int n = lower.length();

    auto isV = [&] (int i)
    {
        const auto c = lower[i];
        if (isVowelLetter (c))
            return true;
        return c == 'y' && i > 0 && ! isVowelLetter (lower[i - 1]);
    };

    struct Group { int start, end; };
    std::vector<Group> groups;
    for (int i = 0; i < n;)
    {
        if (! isV (i)) { ++i; continue; }
        int j = i + 1;
        while (j < n && isV (j))
            ++j;
        groups.push_back ({ i, j });
        i = j;
    }

    // Silent final "e" / "es" / "ed" (but keep the syllabic "-le" of "little").
    if ((int) groups.size() > numSyllables && ! groups.empty())
    {
        const auto& g = groups.back();
        const bool single = g.end - g.start == 1 && lower[g.start] == 'e';
        const bool atEnd = g.end == n || (g.end == n - 1 && (lower[n - 1] == 's' || lower[n - 1] == 'd'));
        const bool syllabicLe = g.start >= 2 && lower[g.start - 1] == 'l' && ! isVowelLetter (lower[g.start - 2]);
        if (single && atEnd && ! syllabicLe)
            groups.pop_back();
    }

    // Reconcile the number of written vowel groups with the number of sung syllables.
    while ((int) groups.size() > numSyllables && groups.size() >= 2)
    {
        size_t best = 0;
        int bestGap = 1 << 30;
        for (size_t k = 0; k + 1 < groups.size(); ++k)
        {
            const int gap = groups[k + 1].start - groups[k].end;
            if (gap < bestGap)
            {
                bestGap = gap;
                best = k;
            }
        }
        groups[best].end = groups[best + 1].end;
        groups.erase (groups.begin() + (long) best + 1);
    }

    while ((int) groups.size() < numSyllables)
    {
        size_t longest = 0;
        for (size_t k = 1; k < groups.size(); ++k)
            if (groups[k].end - groups[k].start > groups[longest].end - groups[longest].start)
                longest = k;

        if (groups.empty() || groups[longest].end - groups[longest].start < 2)
            break;

        const auto g = groups[longest];
        const int mid = g.start + (g.end - g.start) / 2;
        groups[longest].end = mid;
        groups.insert (groups.begin() + (long) longest + 1, Group { mid, g.end });
    }

    if ((int) groups.size() != numSyllables)
    {
        // Fallback: proportional split by letters.
        int start = 0;
        for (int k = 0; k < numSyllables; ++k)
        {
            const int end = (k == numSyllables - 1) ? n : juce::jmax (start, (int) std::round ((double) n * (k + 1) / numSyllables));
            result.add (word.substring (start, end));
            start = end;
        }
        return result;
    }

    auto isDigraphRight = [&] (int i)
    {
        const auto a = lower[i], b = lower[i + 1];
        return b == 'h' && (a == 't' || a == 's' || a == 'c' || a == 'p' || a == 'w');
    };
    auto isDigraphLeft = [&] (int i)
    {
        const auto a = lower[i], b = lower[i + 1];
        return (a == 'c' && b == 'k') || (a == 'n' && b == 'g') || (a == 'g' && b == 'h');
    };

    std::vector<int> cuts;
    for (size_t k = 0; k + 1 < groups.size(); ++k)
    {
        const int c0 = groups[k].end, c1 = groups[k + 1].start;
        const int consonants = c1 - c0;
        int cut;
        if (consonants <= 0)
            cut = c1;
        else if (consonants == 1)
            cut = c0;                                   // ba-by
        else if (consonants == 2 && isDigraphRight (c0))
            cut = c0;                                   // fa-ther
        else if (isDigraphLeft (c0))
            cut = c0 + 2;                               // sing-er, lock-et
        else
            cut = c0 + 1;                               // hap-py, win-ter
        cuts.push_back (cut);
    }

    int start = 0;
    for (auto cut : cuts)
    {
        result.add (word.substring (start, cut));
        start = cut;
    }
    result.add (word.substring (start));
    return result;
}

//==============================================================================
LyricScript parseLyrics (const juce::String& lyrics)
{
    LyricScript script;

    auto text = lyrics.replace ("\r\n", "\n").replaceCharacter ('\r', '\n');
    auto lines = juce::StringArray::fromLines (text);

    int phrase = 0;
    int syllablesInPhrase = 0;
    int lastSyllable = -1;          // most recent syllable that a melisma may continue

    auto closePhrase = [&] (LyricEvent::Type type)
    {
        if (syllablesInPhrase == 0)
            return;

        LyricEvent e;
        e.type = type;
        script.events.push_back (e);
        ++phrase;
        syllablesInPhrase = 0;
        lastSyllable = -1;
    };

    auto addMelisma = [&] (int count)
    {
        for (int i = 0; i < count && lastSyllable >= 0; ++i)
        {
            LyricEvent e;
            e.type = LyricEvent::Type::Melisma;
            e.syllable = lastSyllable;
            script.events.push_back (e);
        }
    };

    for (const auto& line : lines)
    {
        for (const auto& token : tokeniseLine (line))
        {
            if (token.kind == RawToken::Kind::Breath)
            {
                closePhrase (LyricEvent::Type::Breath);
                continue;
            }

            auto t = token.text;

            // Melisma marks: "_", "__", "love_", "_love"
            int leading = 0, trailing = 0;
            while (t.startsWithChar ('_')) { ++leading; t = t.substring (1); }
            while (t.endsWithChar ('_'))   { ++trailing; t = t.dropLastCharacters (1); }
            t = t.replaceCharacter ('_', '-');

            // Stray hyphens at the ends: "-ing" or "love-"
            while (t.startsWithChar ('-')) t = t.substring (1);
            while (t.endsWithChar ('-'))   t = t.dropLastCharacters (1);

            addMelisma (leading);

            if (t.isEmpty())
            {
                addMelisma (trailing);
                continue;
            }

            // Build the written words (numbers may expand to several words).
            std::vector<WordSpec> specs;
            auto pieces = juce::StringArray::fromTokens (t, "-", "");
            pieces.removeEmptyStrings();

            bool hasDigits = false;
            for (auto c : t)
                hasDigits = hasDigits || juce::CharacterFunctions::isDigit (c);

            if (hasDigits)
            {
                for (const auto& piece : pieces)
                {
                    for (const auto& w : expandDigits (piece))
                    {
                        WordSpec s;
                        s.pieces.add (w);
                        s.display = w;
                        specs.push_back (s);
                    }
                }
            }
            else
            {
                WordSpec s;
                s.pieces = pieces;
                s.display = pieces.joinIntoString ("");
                specs.push_back (s);
            }

            if (specs.empty())
                continue;

            specs.front().leadingMelisma = 0;
            specs.back().trailingMelisma = trailing;
            specs.back().pauseAfter = token.pauseAfter;

            for (const auto& spec : specs)
            {
                const auto sylls = syllablesForWord (spec);
                if (sylls.empty())
                    continue;

                const int wordIndex = script.words.size();
                script.words.add (spec.display);

                const auto normalised = g2p::normaliseWord (spec.display);
                const bool function = isFunctionWord (normalised) && sylls.size() == 1;

                for (size_t k = 0; k < sylls.size(); ++k)
                {
                    SungSyllable s;
                    s.text = sylls[k].first;
                    s.onset = sylls[k].second.onset;
                    s.nucleus = sylls[k].second.nucleus;
                    s.coda = sylls[k].second.coda;
                    s.stress = function ? 0 : sylls[k].second.stress;
                    if (sylls.size() == 1 && ! function)
                        s.stress = 1;
                    s.wordIndex = wordIndex;
                    s.phraseIndex = phrase;
                    s.wordStart = k == 0;
                    s.wordEnd = k + 1 == sylls.size();
                    s.functionWord = function;
                    s.pauseAfter = s.wordEnd && spec.pauseAfter;

                    LyricEvent e;
                    e.type = LyricEvent::Type::Syllable;
                    e.syllable = (int) script.syllables.size();
                    script.events.push_back (e);

                    lastSyllable = e.syllable;
                    script.syllables.push_back (std::move (s));
                    ++syllablesInPhrase;
                }

                addMelisma (spec.trailingMelisma);
            }
        }

        closePhrase (LyricEvent::Type::PhraseEnd);
    }

    script.numPhrases = phrase;
    return script;
}

} // namespace voxa::sing
