#include "sing/Lyrics.h"
#include "sing/SingEngine.h"

namespace voxa::sing
{

/** Lyrics parsing, grapheme-to-phoneme conversion and syllabification. */
class SingTextTests : public juce::UnitTest
{
public:
    SingTextTests() : juce::UnitTest ("Text to Sing: lyrics & G2P", "Sing") {}

    void runTest() override
    {
        beginTest ("Embedded dictionary loads");
        {
            const auto start = juce::Time::getMillisecondCounterHiRes();
            const int words = g2p::dictionarySize();
            logMessage ("dictionary: " + juce::String (words) + " words, first access "
                        + juce::String (juce::Time::getMillisecondCounterHiRes() - start, 1) + " ms");
            expect (words > 120000, "CMU dictionary should have ~126k words");
        }

        beginTest ("G2P of dictionary words");
        expectPronunciation ("hello", "HH AH0 L OW1");
        expectPronunciation ("world", "W ER1 L D");
        expectPronunciation ("beautiful", "B Y UW1 T AH0 F AH0 L");
        expectPronunciation ("Singing", "S IH1 NG IH0 NG");
        expectPronunciation ("tonight", "T AH0 N AY1 T");
        expectPronunciation ("heart", "HH AA1 R T");

        beginTest ("Contractions, slang and apostrophes");
        expectPronunciation ("don't", "D OW1 N T");
        expectPronunciation (juce::String::fromUTF8 ("don\xe2\x80\x99t"), "D OW1 N T");    // unicode apostrophe
        expectPronunciation ("I'm", "AY1 M");
        expectPronunciation ("gonna", "G AA1 N AH0");
        expectPronunciation ("lovin'", "L AH1 V IH0 N");                                 // g-dropping
        expectPronunciation ("yeahhh", "Y EH1");                                         // elongation
        expectPronunciation ("ooooh", "UW1");

        beginTest ("Out-of-vocabulary words");
        {
            PhoneString unused;
            expect (! g2p::lookupDictionary ("blorptastic", unused));
            const auto p = g2p::pronounce ("blorptastic");
            expect (p.size() >= 8);
            expect (p.front().ph == Ph::B && p[1].ph == Ph::L);
            expectEquals ((int) g2p::syllabify (p).size(), 3);

            // Compound of two dictionary words
            expectEquals (toString (g2p::pronounce ("moonlightdancer")), juce::String ("M UW1 N L AY2 T D AE2 N S ER0"));

            // Suffix stripping keeps the stem's dictionary pronunciation
            expect (toString (g2p::pronounce ("flimbering")).startsWith ("F L IH1 M"));

            // Every word that has letters gets a pronunciation
            for (auto w : { "zxqv", "qwrtpsdf", "aeiou", "skrrt", "xyzzy" })
                expect (! g2p::pronounce (w).empty(), w);
        }

        beginTest ("Letter-to-sound rules");
        expectEquals (toString (g2p::letterToSound ("cat")), juce::String ("K AE1 T"));
        expectEquals (toString (g2p::letterToSound ("ship")), juce::String ("SH IH1 P"));
        expectEquals (toString (g2p::letterToSound ("phone")), juce::String ("F OW1 N"));
        expectEquals (toString (g2p::letterToSound ("night")), juce::String ("N AY1 T"));
        expectEquals (toString (g2p::letterToSound ("thing")), juce::String ("TH IH1 NG"));
        expectEquals (toString (g2p::letterToSound ("quick")), juce::String ("K W IH1 K"));

        beginTest ("Maximal-onset syllabification");
        {
            const auto s = g2p::syllabify (phonesFromString ("B Y UW1 T AH0 F AH0 L"));
            expectEquals ((int) s.size(), 3);
            expect (s[0].onset == std::vector<Ph> { Ph::B, Ph::Y } && s[0].nucleus == Ph::UW && s[0].stress == 1);
            expect (s[1].onset == std::vector<Ph> { Ph::T } && s[1].nucleus == Ph::AX);
            expect (s[2].onset == std::vector<Ph> { Ph::F } && s[2].coda == std::vector<Ph> { Ph::L });

            const auto extra = g2p::syllabify (phonesFromString ("EH1 K S T R AH0"));   // ek-stra
            expectEquals ((int) extra.size(), 2);
            expect (extra[0].coda == std::vector<Ph> { Ph::K });
            expect (extra[1].onset == (std::vector<Ph> { Ph::S, Ph::T, Ph::R }));

            const auto singing = g2p::syllabify (phonesFromString ("S IH1 NG IH0 NG"));  // NG is never an onset
            expect (singing[0].coda == std::vector<Ph> { Ph::NG } && singing[1].onset.empty());

            const auto hmm = g2p::syllabify (phonesFromString ("HH M"));                 // vowel-less: hum on M
            expectEquals ((int) hmm.size(), 1);
            expect (hmm[0].nucleus == Ph::M);
        }

        beginTest ("Syllable counting");
        expectEquals (SingEngine::countSyllables ("hello world"), 3);
        expectEquals (SingEngine::countSyllables ("beautiful"), 3);
        expectEquals (SingEngine::countSyllables ("beau-ti-ful"), 3);
        expectEquals (SingEngine::countSyllables ("I love you _ _"), 3);
        expectEquals (SingEngine::countSyllables ("love__"), 1);
        expectEquals (SingEngine::countSyllables ("[br] hello [br] world"), 3);
        expectEquals (SingEngine::countSyllables ("Don't stop, believin'!"), 5);
        expectEquals (SingEngine::countSyllables ("21 guns"), 4);              // twen-ty one guns
        expectEquals (SingEngine::countSyllables ("[Chorus]\nla la la"), 3);   // section tags are ignored
        expectEquals (SingEngine::countSyllables ("hmm mmm"), 2);
        expectEquals (SingEngine::countSyllables ("gonna wanna"), 4);
        expectEquals (SingEngine::countSyllables (""), 0);
        expectEquals (SingEngine::countSyllables ("  ,.!? -- _ [br]  "), 0);

        beginTest ("Lyric events, phrases and display text");
        {
            const auto s = parseLyrics ("Hello world,\nbeau-ti-ful _ night [br] yes");
            juce::StringArray texts;
            for (const auto& syl : s.syllables)
                texts.add (syl.text);
            expectEquals (texts.joinIntoString ("|"), juce::String ("Hel|lo|world|beau|ti|ful|night|yes"));

            using T = LyricEvent::Type;
            const std::vector<T> expected { T::Syllable, T::Syllable, T::Syllable, T::PhraseEnd,
                                            T::Syllable, T::Syllable, T::Syllable, T::Melisma, T::Syllable, T::Breath,
                                            T::Syllable, T::PhraseEnd };
            expectEquals ((int) s.events.size(), (int) expected.size());
            for (size_t i = 0; i < expected.size() && i < s.events.size(); ++i)
                expect (s.events[i].type == expected[i], "event " + juce::String ((int) i));

            expectEquals (s.events[7].syllable, 5);          // the melisma holds "ful"
            expectEquals (s.numPhrases, 3);
            expect (s.syllables[2].pauseAfter);              // comma after "world"
            expectEquals (s.syllables[0].stress, 0);         // hel-LO
            expectEquals (s.syllables[1].stress, 1);
            expectEquals (s.syllables[3].wordIndex, s.syllables[5].wordIndex);
            expect (s.syllables[3].wordStart && ! s.syllables[3].wordEnd && s.syllables[5].wordEnd);

            const auto f = parseLyrics ("the sun");
            expect (f.syllables[0].functionWord && f.syllables[0].stress == 0);
            expect (! f.syllables[1].functionWord && f.syllables[1].stress == 1);
        }

        beginTest ("Numbers are spelled out");
        expectEquals (spellNumber (21), juce::String ("twenty one"));
        expectEquals (spellNumber (100), juce::String ("one hundred"));
        expectEquals (spellNumber (1999, true), juce::String ("nineteen ninety nine"));
        expectEquals (spellNumber (2005, true), juce::String ("two thousand five"));
        expectEquals (spellNumber (1200345), juce::String ("one million two hundred thousand three hundred forty five"));
        expectEquals (parseLyrics ("the 1st time").syllables[1].text, juce::String ("first"));

        beginTest ("Written syllable splitting");
        expectEquals (splitWordText ("beautiful", 3).joinIntoString ("|"), juce::String ("beau|ti|ful"));
        expectEquals (splitWordText ("singing", 2).joinIntoString ("|"), juce::String ("sing|ing"));
        expectEquals (splitWordText ("happy", 2).joinIntoString ("|"), juce::String ("hap|py"));
        expectEquals (splitWordText ("little", 2).joinIntoString ("|"), juce::String ("lit|tle"));
        expectEquals (splitWordText ("Tonight", 2).joinIntoString ("|"), juce::String ("To|night"));
        expectEquals (splitWordText ("love", 1).joinIntoString ("|"), juce::String ("love"));
    }

private:
    void expectPronunciation (const juce::String& word, const juce::String& expected)
    {
        expectEquals (toString (g2p::pronounce (word)), expected, word);
    }
};

static SingTextTests singTextTests;

} // namespace voxa::sing
