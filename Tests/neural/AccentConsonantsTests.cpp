#include "neural/AccentConsonants.h"

#include <algorithm>
#include <cmath>
#include <random>

/*  The reggae / Latin / afrobeats accents' consonants (category "Neural"): the decision rule on handmade recognizer
    output, and the recognizer embedded in the plug-in. */

namespace voxa::neural
{

namespace
{
    using Rule = AccentConsonants::Rule;
    using Frame = AccentConsonants::Frame;
    using D = Rule::Decision;

    Frame sound (float vowel, bool voiced = true)
    {
        Frame f;
        f.voiced = voiced;
        f.vowel = vowel;
        return f;
    }
    Frame vowel()               { return sound (0.9f); }
    Frame th (float p = 0.8f)   { auto f = sound (0.05f, false); f.th = p; return f; }
    Frame dh (float p = 0.8f)   { auto f = sound (0.05f); f.dh = p; return f; }
    Frame n()                   { auto f = sound (0.05f); f.n = 0.8f; return f; }
    Frame d()                   { auto f = sound (0.05f); f.d = 0.8f; return f; }
    Frame r()                   { auto f = sound (0.1f); f.r = 0.8f; return f; }
    Frame silence()             { return sound (0.0f, false); }

    /** The rule's decisions over the frames, as the plug-in makes them (2 frames of lookahead; two silent frames
        follow). */
    std::vector<D> run (std::vector<Frame> frames, bool spanish = false)
    {
        frames.push_back (silence());
        frames.push_back (silence());
        Rule rule;
        rule.spanish = spanish;
        std::vector<D> out;
        for (size_t i = 0; i + 2 < frames.size(); ++i)
            out.push_back (rule.step (frames[i], frames[i + 1], frames[i + 2]));
        return out;
    }

    bool allNone (const std::vector<D>& ds)
    {
        return std::all_of (ds.begin(), ds.end(), [] (const D& x) { return x.kind == D::none; });
    }
}

class AccentConsonantsTests : public juce::UnitTest
{
public:
    AccentConsonantsTests() : juce::UnitTest ("Accent consonants", "Neural") {}

    void runTest() override
    {
        beginTest ("\"th\" -> \"t\" / \"d\": closure, then the template's last two frames at the end of the \"th\"");
        {
            auto think = run ({ silence(), th(), th(), th(), th(), vowel(), vowel() });
            expect (think[0].kind == D::none, "the silence before");
            expect (think[1].kind == D::closure && think[2].kind == D::closure, "closure");
            expect (think[3].kind == D::stop && think[3].frame == 2 && ! think[3].voicedStop, "burst...");
            expect (think[4].kind == D::stop && think[4].frame == 3 && ! think[4].voicedStop, "...and aspiration: tink");
            expect (think[5].kind == D::none, "the vowel is yours");

            auto that = run ({ vowel(), vowel(), dh(), dh(), dh(), vowel() });
            expect (that[2].kind == D::voiceBar && that[2].voicedStop, "legato: a voice bar, not silence");
            expect (that[3].kind == D::stop && that[3].frame == 2 && that[4].frame == 3 && that[4].voicedStop, "dat");

            auto paused = run ({ silence(), dh(), dh(), dh(), vowel() });
            expect (paused[1].kind == D::voiceBar, "after a pause too (its pitch waits for the vowel's)");

            auto the = run ({ vowel(), dh(), vowel() });
            expect (the[1].kind == D::stop && the[1].frame == 3, "a one-frame \"th\": the burst");

            expect (allNone (run ({ vowel(), th (0.3f), vowel() })), "not on a guess");
            Frame mixed = th (0.3f);
            mixed.dh = 0.3f;
            expect (run ({ vowel(), mixed, vowel() })[1].kind == D::stop, "th and dh together count");
        }

        beginTest ("an' you: the \"d\" after \"n\" sung as the \"n\"");
        {
            auto andYou = run ({ vowel(), n(), n(), d(), d(), d(), vowel() });
            expect (andYou[3].kind == D::holdN && andYou[3].weight == 0.5f, "fading in");
            expect (andYou[4].kind == D::holdN && andYou[5].kind == D::holdN && andYou[5].weight == 1.0f, "the whole \"d\"");
            expect (andYou[6].kind == D::none, "into the next word");
            expect (allNone (run ({ vowel(), d(), d(), vowel() })), "a \"d\" not after \"n\" stays");
            expect (run ({ vowel(), n(), sound (0.05f), d(), d(), vowel() })[3].kind == D::holdN, "a frame between them");
            expect (allNone (run ({ n(), sound (0.9f), sound (0.9f), sound (0.9f), d(), vowel() })), "but not a vowel");
            Frame t = sound (0.05f, false);
            t.t = 0.8f;
            expect (run ({ vowel(), n(), t, t, vowel() })[2].kind == D::holdN, "live, the \"d\" of \"and\" is heard as \"t\"");
            std::vector<Frame> longFrames { n() };
            longFrames.insert (longFrames.end(), (size_t) Rule::maxDroppedD + 3, d());
            longFrames.push_back (vowel());
            auto longD = run (longFrames);
            int held = 0;
            for (auto& x : longD) held += x.kind == D::holdN ? 1 : 0;
            expectEquals (held, Rule::maxDroppedD, "a long one is a \"d\" to keep");
        }

        beginTest ("fo: the \"r\" after a vowel sung as the vowel; an \"r\" between vowels stays");
        {
            auto forr = run ({ vowel(), vowel(), vowel(), vowel(), r(), r(), r(), r(), silence() });
            expect (forr[4].kind == D::none, "its first frame passes (not yet sure it is long)");
            expect (forr[5].kind == D::holdVowel && forr[5].weight == 0.5f, "from its 2nd: fading in");
            expect (forr[7].kind == D::holdVowel && forr[7].weight == 1.0f, "to its end");
            expect (forr[8].kind == D::none, "then nothing");
            expect (allNone (run ({ vowel(), vowel(), vowel(), r(), r(), vowel(), vowel() })), "very: a short \"r\" between vowels");
            expect (allNone (run ({ vowel(), vowel(), vowel(), r(), r(), r(), vowel(), vowel() })), "nor a 3-frame one");
            expect (allNone (run ({ silence(), r(), r(), r(), r(), vowel() })), "red: no vowel before");
            expect (allNone (run ({ vowel(), r(), r(), r(), r(), silence() })), "a vowel too short to hold");
        }

        beginTest ("Latin: the \"r\" after a vowel rolled at its end (fo-rr), \"you\" -> \"jou\"");
        {
            auto forr = run ({ vowel(), vowel(), vowel(), vowel(), r(), r(), r(), r(), r(), silence() }, true);
            expect (forr[5].kind == D::holdVowel, "the vowel held");
            expect (forr[7].kind == D::roll && forr[7].frame == 0 && forr[8].kind == D::roll && forr[8].frame == 1,
                    "the rolled rr on the r's last two frames");
            expect (run ({ vowel(), vowel(), vowel(), vowel(), r(), r(), r(), r(), r(), silence() })[8].kind == D::holdVowel,
                    "Jamaican / Nigerian: fo, no roll");
            Frame y = sound (0.1f);
            y.y = 0.8f;
            auto you = run ({ n(), y, y, y, y, y, y, y, vowel() }, true);
            expect (you[1].kind == D::jh && you[1].frame == 0 && you[5].kind == D::jh && you[5].frame == 4, "jou: the j and into the vowel");
            expect (you[6].kind == D::none, "the rest of the y is the singer's");
            expect (allNone (run ({ n(), y, y, vowel() })), "only the Latin accent");

            // gurrl: the singer's "er", rolled on its last two frames before the "l", the "l" light
            Frame er = vowel();
            er.er = true;
            Frame l = sound (0.2f);
            l.l = 0.8f;
            auto girl = run ({ er, er, er, er, er, l, l, l, silence() }, true);
            expect (girl[2].kind == D::none, "the er itself is the singer's");
            expect (girl[3].kind == D::roll && girl[3].frame == 0 && girl[4].kind == D::roll && girl[4].frame == 1, "rr");
            expect (girl[5].kind == D::lightL && girl[5].frame == 0 && girl[7].kind == D::lightL && girl[7].frame == 2, "light l");
            expect (allNone (run ({ vowel(), vowel(), vowel(), l, l, silence() }, true)), "the l of \"all\", \"feel\": as sung");
            Frame erL = l;                 // the vowel rule's "er" runs on into the "l"
            erL.er = true;
            auto live = run ({ er, er, er, er, er, erL, erL, erL, silence() }, true);
            expect (live[4].kind == D::roll && live[4].frame == 1 && live[5].kind == D::lightL && live[6].kind == D::lightL,
                    "the l is the l, not more r");
        }

        beginTest ("The consonant recognizer embedded in the plug-in");
        {
            const auto* consonants = AccentConsonants::get();
            expect (consonants != nullptr, "the embedded data loads");
            if (consonants == nullptr)
                return;
            const int channels = consonants->getChannels();
            expectEquals (channels, 768, "ContentVec features");
            for (int t = Rule::t; t <= Rule::dI; ++t)
                expectEquals (consonants->getTemplateFrames (t), 4, "the t / d templates: 4 frames");
            expectEquals ((int) consonants->getSilence().size(), channels, "the silence frame");
            expect (consonants->hasLatin(), "the Latin accent's rolled rr and j");

            constexpr int encoderFrames = 60, frames = 2 * encoderFrames, skip = 8;
            std::mt19937 random (11);
            std::normal_distribution<float> normal (0.0f, 0.3f);
            std::vector<float> features ((size_t) (encoderFrames * channels)), scratch;
            for (auto& v : features)
                v = normal (random);
            const auto f = consonants->recognise (features.data(), encoderFrames, 10, scratch);
            expect (f.th >= 0.0f && f.dh >= 0.0f && f.vowel >= 0.0f && f.th + f.dh + f.d + f.n + f.r + f.vowel <= 1.0001f,
                    "probabilities");

            // a voiced chunk of random features: whatever it decides, the output is finite and marks what it changed
            std::vector<float> phone ((size_t) ((frames - skip) * channels));
            for (int i = skip; i < frames; ++i)
                std::copy_n (features.data() + (size_t) (i / 2) * (size_t) channels, channels,
                             phone.data() + (size_t) (i - skip) * (size_t) channels);
            const auto original = phone;
            std::vector<float> f0 ((size_t) frames, 180.0f), changed ((size_t) (frames - skip), 0.0f);
            AccentConsonants::State state;
            consonants->apply (features.data(), encoderFrames, channels, f0.data(), frames, skip, 1000, phone.data(), state,
                               scratch, changed.data());
            bool finite = true, marked = true;
            for (auto v : phone)
                finite = finite && std::isfinite (v);
            for (int i = 0; i < frames - skip; ++i)
            {
                const auto* p = phone.data() + (size_t) i * (size_t) channels;
                const bool moved = ! std::equal (p, p + channels, original.data() + (size_t) i * (size_t) channels);
                marked = marked && (! moved || changed[(size_t) i] > 0.0f) && changed[(size_t) i] <= 1.0f;
            }
            expect (finite, "finite features");
            expect (marked, "every changed frame is reported");

            // "think": Kokoro's own "t" (its burst frame) put in for the "th" frames of a voiced take
            const auto& t = consonants->getTemplate (Rule::t);
            auto withT = features;
            for (int e = 20; e < 24; ++e)   // the "t" itself, as a "th" would never be this clear: its 4 frames
                std::copy_n (t.data() + (size_t) (e - 20) * (size_t) channels, channels, withT.data() + (size_t) e * (size_t) channels);
            const auto heard = consonants->recognise (withT.data(), encoderFrames, 22, scratch);
            logMessage ("Kokoro's t burst heard as: th " + juce::String (heard.th, 2) + ", vowel " + juce::String (heard.vowel, 2));
            expectLessThan (heard.th + heard.dh, 0.5f, "a \"t\" is not taken for a \"th\" (it would loop)");
        }
    }
};

static AccentConsonantsTests accentConsonantsTests;

} // namespace voxa::neural
