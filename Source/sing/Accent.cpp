#include "Lyrics.h"

#include <algorithm>

namespace voxa::sing
{

namespace
{
    bool isVoicelessConsonant (Ph p) noexcept
    {
        return p == Ph::P || p == Ph::T || p == Ph::K || p == Ph::F || p == Ph::TH || p == Ph::S
            || p == Ph::SH || p == Ph::CH || p == Ph::HH;
    }

    void replaceConsonants (std::vector<Ph>& cs, Accent accent)
    {
        for (auto& c : cs)
        {
            if (accent != Accent::Southern)   // th-stopping: Jamaican, Spanish-accented and Nigerian English
            {
                if (c == Ph::TH) c = Ph::T;
                else if (c == Ph::DH) c = Ph::D;
            }
            if (accent == Accent::Latin)      // Spanish has no z or v
            {
                if (c == Ph::Z) c = Ph::S;
                else if (c == Ph::V) c = Ph::B;
            }
        }
    }

    /** No r after a vowel (Jamaican, Nigerian): the coda's r goes, an r-coloured vowel opens. */
    void dropPostVocalicR (Ph& nucleus, std::vector<Ph>& coda, int stress, Accent accent)
    {
        if (nucleus == Ph::ER)
            nucleus = accent == Accent::Nigerian ? (stress > 0 ? Ph::EH : Ph::AA) : (stress > 0 ? Ph::AH : Ph::AX);
        coda.erase (std::remove (coda.begin(), coda.end(), Ph::R), coda.end());
    }

    Ph vowel (Ph v, int stress, Accent accent)
    {
        switch (accent)
        {
            case Accent::Jamaican:
                if (v == Ph::EY) return Ph::EH;
                break;
            case Accent::Latin:
                if (v == Ph::IH) return Ph::IY;
                if (v == Ph::UH) return Ph::UW;
                if (v == Ph::AE) return Ph::AA;
                if (v == Ph::AH && stress > 0) return Ph::AA;
                break;
            case Accent::Nigerian:
                if (v == Ph::IH) return Ph::IY;
                if (v == Ph::UH) return Ph::UW;
                if (v == Ph::AE) return Ph::AA;
                if (v == Ph::AH && stress > 0) return Ph::AO;
                if (v == Ph::EY) return Ph::EH;
                break;
            case Accent::Southern:
            case Accent::American:
                break;
        }
        return v;
    }

    void accentSyllable (Ph& nucleus, std::vector<Ph>& onset, std::vector<Ph>& coda, int stress, Accent accent,
                         bool wordEnd, const juce::String& word)
    {
        replaceConsonants (onset, accent);
        replaceConsonants (coda, accent);
        if (accent == Accent::Jamaican || accent == Accent::Nigerian)
            dropPostVocalicR (nucleus, coda, stress, accent);

        if (accent == Accent::Southern)
        {
            // "I", "time", "my" as "ah" (the glide stays before a voiceless consonant: "right", "nice")
            if (nucleus == Ph::AY && (coda.empty() || ! isVoicelessConsonant (coda.front())))
                nucleus = Ph::AA;
            // pin-pen merger: "ten", "friend", "remember"
            if (nucleus == Ph::EH && ! coda.empty() && (coda.front() == Ph::N || coda.front() == Ph::M))
                nucleus = Ph::IH;
        }
        nucleus = vowel (nucleus, stress, accent);

        // singin', lovin' (Southern, Jamaican)
        if ((accent == Accent::Southern || accent == Accent::Jamaican) && wordEnd && stress == 0
            && ! coda.empty() && coda.back() == Ph::NG && word.endsWithIgnoreCase ("ing"))
            coda.back() = Ph::N;

        // Jamaican: a final t or d after a consonant drops ("and", "just", "world")
        if (accent == Accent::Jamaican && wordEnd && coda.size() >= 2 && (coda.back() == Ph::T || coda.back() == Ph::D)
            && isConsonant (coda[coda.size() - 2]))
            coda.pop_back();
    }
}

void applyAccent (LyricScript& script, Accent accent)
{
    if (accent == Accent::American)
        return;

    for (auto& syl : script.syllables)
    {
        const auto word = juce::isPositiveAndBelow (syl.wordIndex, script.words.size()) ? script.words[syl.wordIndex]
                                                                                         : juce::String();
        std::vector<Ph> noOnset;
        accentSyllable (syl.nucleus, syl.onset, syl.coda, syl.stress, accent, syl.wordEnd, word);
        if (syl.hasWeakForm)
            accentSyllable (syl.weakNucleus, noOnset, syl.weakCoda, 0, accent, syl.wordEnd, word);

        if (accent == Accent::Jamaican && word.equalsIgnoreCase ("the"))   // "di"
        {
            syl.nucleus = Ph::IH;
            syl.weakNucleus = Ph::IH;
        }
    }
}

} // namespace voxa::sing
