#include "neural/SouthernVowels.h"

#include <algorithm>
#include <cmath>
#include <random>

/*  The country voices' Southern "ah" (category "Neural"): the decision rule on handmade recognizer output, and the
    recognizer embedded in the plug-in. */

namespace voxa::neural
{

namespace
{
    using Rule = SouthernVowels::Rule;
    using Frame = SouthernVowels::Frame;

    Frame vowel (float iy, float glide = 0.05f)
    {
        Frame f;
        f.voiced = true;
        f.vowel = 0.95f;
        f.iy = iy;
        f.glide = glide;
        return f;
    }

    Frame consonant()
    {
        Frame f;
        f.voiced = true;
        f.vowel = 0.1f;
        return f;
    }

    struct Word
    {
        Word& add (const Frame& f, int count = 1) { frames.insert (frames.end(), (size_t) count, f); return *this; }
        std::vector<Frame> frames;
    };

    /** Runs the rule over a word (followed by a consonant) as the plug-in does, with its lookahead. Returns the frame
        where the glide was found to start (-1: none) and the events. */
    int decide (Rule& rule, Word word, std::vector<Rule::Event>* events = nullptr)
    {
        word.add (consonant(), 1 + Rule::lookahead);
        int started = -1;
        for (size_t d = 0; d + Rule::lookahead < word.frames.size(); ++d)
        {
            const auto event = rule.step (word.frames[d], word.frames[d + 1], word.frames[d + 2]);
            if (event == Rule::Event::glideStarts)
                started = (int) d;
            if (events != nullptr)
                events->push_back (event);
        }
        return started;
    }
}

class SouthernVowelsTests : public juce::UnitTest
{
public:
    SouthernVowelsTests() : juce::UnitTest ("Southern vowels", "Neural") {}

    void runTest() override
    {
        beginTest ("The rule holds an \"I\" from its glide to the end of the vowel");
        {
            Rule rule;
            std::vector<Rule::Event> events;
            expectEquals (decide (rule, Word().add (vowel (0.8f), 5).add (vowel (0.1f, 0.85f), 4), &events), 5,
                          "the glide starts on its first frame");
            for (int d = 6; d < 9; ++d)
                expect (events[(size_t) d] == Rule::Event::holding, "held to the end of the vowel");
            expect (events[9] == Rule::Event::none && ! rule.held, "the consonant ends it");

            rule.reset();
            expectEquals (decide (rule, Word().add (vowel (0.9f), 7)), 5, "a glide too short to see: the vowel's last 40 ms");
        }

        beginTest ("The rule leaves other vowels alone");
        {
            Rule rule;
            expectEquals (decide (rule, Word().add (vowel (0.1f), 6).add (vowel (0.1f, 0.85f), 3)), -1,
                          "a glide after a vowel that is not \"I\" (\"say\", \"see\")");

            rule.reset();
            expectEquals (decide (rule, Word().add (vowel (0.1f), 8)), -1, "\"father\"");

            rule.reset();
            expectEquals (decide (rule, Word().add (vowel (0.9f), 5).add (vowel (0.0f), 5).add (vowel (0.1f, 0.85f), 2)), -1,
                          "\"pain\" sung with an open start");

            rule.reset();
            expectEquals (decide (rule, Word().add (vowel (0.8f), 2).add (vowel (0.1f, 0.85f), 2)), -1,
                          "too little vowel before the glide to hold");

            rule.reset();
            Frame unvoiced = vowel (0.8f);
            unvoiced.voiced = false;
            for (int k = 0; k < 5; ++k)
                rule.step (unvoiced, unvoiced, unvoiced);
            expectEquals (rule.vowelFrames, 0, "whispered or unvoiced frames are not a vowel");
        }

        beginTest ("The rule waits for a real glide");
        {
            Rule rule;
            expectEquals (decide (rule, Word().add (vowel (0.8f), 4).add (vowel (0.1f, 0.85f)).add (vowel (0.8f), 4)), 7,
                          "a single frame of glide is not one: the vowel's end is");

            rule.reset();
            expectEquals (decide (rule, Word().add (vowel (0.8f), 4).add (vowel (0.1f, 0.85f)).add (vowel (0.8f), 3)
                                              .add (vowel (0.1f, 0.85f), 3).add (vowel (0.8f), 2)), 8,
                          "the real glide later in the same vowel");
        }

        beginTest ("Step 2: each shift on its own vowel, in shapes decided live");
        {
            using Shift = SouthernVowels::ShiftRule;
            auto shiftVowel = [] (int which, float p)
            {
                Frame f = vowel (0.0f);
                f.shift[(size_t) which] = p;
                return f;
            };
            // runs the rule over a vowel of `frames` frames (then a consonant): the final weight of each frame
            auto run = [] (const Frame& x, int frames, int& shift)
            {
                std::vector<Frame> v ((size_t) frames, x);
                v.insert (v.end(), 3, consonant());
                std::vector<float> w ((size_t) frames, 0.0f);
                Shift rule;
                shift = -1;
                for (int d = 0; d < frames; ++d)
                {
                    const auto r = rule.step (v[(size_t) d], v[(size_t) d + 1], v[(size_t) d + 2]);
                    w[(size_t) d] = r.weight;
                    for (int j = 0; j < 2; ++j)
                        if (r.before[(size_t) j] >= 0.0f && d - 2 + j >= 0)
                            w[(size_t) (d - 2 + j)] = r.before[(size_t) j];
                    if (r.shift >= 0)
                        shift = r.shift;
                }
                return w;
            };

            int shift = -1;
            auto w = run (shiftVowel (Shift::raise, 0.8f), 8, shift);
            expectEquals (shift, (int) Shift::raise, "\"eh\" before n: raised");
            expectWithinAbsoluteError (w[0], 1.0f / 3.0f, 1.0e-5f, "in over 3 frames");
            expectWithinAbsoluteError (w[1], 2.0f / 3.0f, 1.0e-5f);
            expectWithinAbsoluteError (w[4], 1.0f, 1.0e-5f, "the whole vowel");
            expectWithinAbsoluteError (w[7], 1.0f / 3.0f, 1.0e-5f, "out over 3 frames");

            w = run (shiftVowel (Shift::lower, 0.8f), 12, shift);
            expectEquals (shift, (int) Shift::lower, "\"ay\": lowered");
            expect (w[1] > w[0] && w[2] > w[4] && w[4] > w[6], "from the start, fading");
            expectEquals (w[Shift::lowerFrames], 0.0f, "gone after lowerFrames");

            w = run (shiftVowel (Shift::drawl, 0.8f), 16, shift);
            expectEquals (shift, (int) Shift::drawl, "\"a\": drawled");
            expectEquals (w[Shift::drawlDelay - 1], 0.0f, "not before drawlDelay");
            expectWithinAbsoluteError (w[Shift::drawlDelay + 3], 1.0f, 1.0e-5f, "the bump's peak");
            expectEquals (w[Shift::drawlDelay + 8], 0.0f, "and back");

            run (vowel (0.1f), 10, shift);
            expectEquals (shift, -1, "other vowels are left alone");
            run (shiftVowel (Shift::raise, 0.2f), 10, shift);
            expectEquals (shift, -1, "not on a guess");
        }

        beginTest ("Embedded recognizer");
        {
            const auto* vowels = SouthernVowels::get();
            expect (vowels != nullptr, "southern_vowels.gz is embedded and loads");
            if (vowels == nullptr)
                return;

            const int channels = vowels->getChannels();
            expectEquals (channels, 768, "ContentVec features");

            constexpr int encoderFrames = 60, frames = 2 * encoderFrames, skip = 8;
            std::mt19937 random (7);
            std::normal_distribution<float> normal (0.0f, 0.3f);
            std::vector<float> features ((size_t) (encoderFrames * channels)), scratch;
            for (auto& v : features)
                v = normal (random);

            const auto f = vowels->recognise (features.data(), encoderFrames, 10, scratch);
            expect (f.vowel >= 0.0f && f.vowel <= 1.0001f && f.iy >= 0.0f && f.glide >= 0.0f && f.iy + f.glide <= f.vowel + 1.0e-4f,
                    "probabilities");

            // unvoiced: the generator's features are left exactly as they were
            std::vector<float> phone ((size_t) ((frames - skip) * channels));
            for (int i = skip; i < frames; ++i)
                std::copy_n (features.data() + (size_t) (i / 2) * (size_t) channels, channels,
                             phone.data() + (size_t) (i - skip) * (size_t) channels);
            const auto original = phone;
            const std::vector<float> silence ((size_t) frames, 0.0f);

            SouthernVowels::State state;
            state.reset();
            vowels->apply (features.data(), encoderFrames, channels, silence.data(), frames, skip, 1000, phone.data(), state, scratch);
            expect (phone == original, "nothing held without a voice");
            expectEquals ((int) state.recognised, 500 + encoderFrames - 1 - 2, "every frame with its context recognised");

            expect (vowels->hasMeans(), "the accents' vowel means are embedded (version 3)");
            const auto& k = vowels->getClasses();
            expect (k.AA >= 0 && k.AE >= 0 && k.AH >= 0 && k.AO >= 0 && k.EH >= 0 && k.ER >= 0 && k.EY >= 0 && k.IH >= 0
                    && k.IY >= 0 && k.OW >= 0, "every vowel the accents use");

            // a voiced take through a Jamaican stream: changed only where the rule says
            const std::vector<float> voiced ((size_t) frames, 180.0f);
            SouthernVowels::State jamaican;
            vowels->setAccent (jamaican, SouthernVowels::jamaican);
            expectEquals (jamaican.accentRule.accent, (int) SouthernVowels::jamaican, "the stream runs the accent");
            phone = original;
            vowels->apply (features.data(), encoderFrames, channels, voiced.data(), frames, skip, 1000, phone.data(), jamaican, scratch);
            for (auto v : phone)
                if (! std::isfinite (v)) { expect (false, "finite features"); break; }

            // "girl" (an "er" from encoder frame 20 on: its mean, twice as far again from the mean of all the
            // classes, which the recognizer hears as a clear "er") through a Jamaican stream: "gyal", and the frames
            // it changed reported for the vowel keeper (Southern reports none)
            const auto& er = vowels->getMean (k.ER);
            const auto& aa = vowels->getMean (k.AA);
            std::vector<float> clearEr ((size_t) channels, 0.0f), centre ((size_t) channels, 0.0f);
            int numMeans = 0;
            for (; ! vowels->getMean (numMeans).empty(); ++numMeans)
                for (int c = 0; c < channels; ++c)
                    centre[(size_t) c] += vowels->getMean (numMeans)[(size_t) c];
            for (int c = 0; c < channels; ++c)
                clearEr[(size_t) c] = er[(size_t) c] + 2.0f * (er[(size_t) c] - centre[(size_t) c] / (float) numMeans);
            auto girl = features;
            for (int e = 20; e < encoderFrames; ++e)
                std::copy (clearEr.begin(), clearEr.end(), girl.begin() + (size_t) e * (size_t) channels);
            for (int i = skip; i < frames; ++i)
                std::copy_n (girl.data() + (size_t) (i / 2) * (size_t) channels, channels,
                             phone.data() + (size_t) (i - skip) * (size_t) channels);
            const auto plainGirl = phone;
            std::vector<float> changed ((size_t) (frames - skip), -1.0f);
            SouthernVowels::State girlState;
            vowels->setAccent (girlState, SouthernVowels::jamaican);
            vowels->apply (girl.data(), encoderFrames, channels, voiced.data(), frames, skip, 1000, phone.data(), girlState,
                           scratch, changed.data());
            int reported = 0, agree = 0;
            float towardAA = 0.0f;
            for (int i = 0; i < frames - skip; ++i)
            {
                const auto* p = phone.data() + (size_t) i * (size_t) channels;
                const auto* q = plainGirl.data() + (size_t) i * (size_t) channels;
                const bool moved = ! std::equal (p, p + channels, q);
                expect (changed[(size_t) i] >= 0.0f && changed[(size_t) i] <= 1.0f, "0..1");
                agree += moved == (changed[(size_t) i] > 0.0f) ? 1 : 0;
                if (changed[(size_t) i] > 0.0f)
                {
                    ++reported;
                    for (int c = 0; c < channels; ++c)
                        towardAA += (p[c] - q[c]) * (aa[(size_t) c] - er[(size_t) c]);
                }
            }
            logMessage ("girl: " + juce::String (reported) + " of " + juce::String (frames - skip) + " frames changed");
            expectGreaterThan (reported, 20, "the held \"er\" becomes the accent's vowel");
            expectEquals (agree, frames - skip, "exactly the changed frames are reported");
            expectGreaterThan (towardAA, 0.0f, "towards \"ah\"");

            SouthernVowels::State southernState;
            vowels->setAccent (southernState, SouthernVowels::southern);
            phone = plainGirl;
            vowels->apply (girl.data(), encoderFrames, channels, voiced.data(), frames, skip, 1000, phone.data(), southernState,
                           scratch, changed.data());
            expect (std::all_of (changed.begin(), changed.end(), [] (float v) { return v == 0.0f; }), "Southern reports nothing");
        }

        beginTest ("Accents: \"ay\" and \"oh\" sung pure, the accent's vowels");
        {
            using Accent = SouthernVowels::AccentRule;
            using D = Accent::Decision;
            Accent::Classes k;
            k.AA = 0; k.AE = 1; k.AH = 2; k.AO = 3; k.EH = 4; k.ER = 5; k.EY = 6; k.IH = 7; k.IY = 8; k.OW = 9; k.UH = 10;
            auto sung = [&k] (int cls, float share = 0.6f, float er = -1.0f)
            {
                Frame f;
                f.voiced = true; f.vowel = 0.9f; f.top = (int8_t) cls; f.topShare = share;
                f.er = er >= 0.0f ? er : (cls == k.ER ? share : 0.02f);
                return f;
            };
            auto run = [&] (int accent, std::vector<Frame> frames)
            {
                Accent rule;
                rule.accent = accent;
                rule.classes = k;
                frames.push_back (consonant());
                frames.push_back (consonant());
                std::vector<D> out;
                for (size_t d = 0; d + 1 < frames.size(); ++d)
                    out.push_back (rule.step (frames[d], frames[d + 1]));
                return out;
            };

            auto take = run (SouthernVowels::jamaican, std::vector<Frame> (6, sung (k.EY)));
            expect (take[0].kind == D::none && take[1].kind == D::none, "\"ay\" not touched before it is sure");
            expect (take[2].kind == D::hold && take[2].startsHold && take[2].to == k.EH, "held from its 3rd frame, towards \"eh\"");
            expectEquals (take[2].weight, 0.5f, "fading in");
            expect (take[3].kind == D::hold && take[5].kind == D::hold && take[5].weight == 1.0f, "to the end of the vowel");
            expect (take[6].kind == D::none, "the consonant ends it");
            expect (run (SouthernVowels::nigerian, std::vector<Frame> (4, sung (k.OW)))[2].to == k.AO, "\"oh\": towards \"aw\"");

            auto that = run (SouthernVowels::jamaican, std::vector<Frame> (5, sung (k.AE)));
            expect (that[0].kind == D::swap && that[0].from == k.AE && that[0].to == k.AA, "the \"a\" of that -> \"ah\"");
            expectWithinAbsoluteError (that[0].weight, 0.9f, 1.0e-5f, "in over 2 frames");
            expectWithinAbsoluteError (that[2].weight, 1.8f, 1.0e-5f, "x1.8 (round 25 A)");
            expectWithinAbsoluteError (that[4].weight, 0.9f, 1.0e-5f, "half on the vowel's last frame");
            auto held = run (SouthernVowels::spanish, std::vector<Frame> (20, sung (k.AE)));
            expectWithinAbsoluteError (held[18].weight, 2.4f, 1.0e-5f, "x2.4 on a held note");

            expect (run (SouthernVowels::jamaican, { sung (k.IH), sung (k.IH), sung (k.IH) })[1].kind == D::none, "Jamaican keeps \"i\"");
            expect (run (SouthernVowels::spanish, { sung (k.IH), sung (k.IH), sung (k.IH) })[1].to == k.IY, "Spanish: \"i\" -> \"ee\"");
            expect (run (SouthernVowels::jamaican, { sung (k.ER), sung (k.ER) })[0].to == k.AA, "Jamaican gyal");
            expect (run (SouthernVowels::spanish, { sung (k.ER), sung (k.ER) })[0].to == k.EH, "Spanish \"er\" -> \"eh\"");
            auto girl = run (SouthernVowels::nigerian, std::vector<Frame> (16, sung (k.ER)));
            expect (girl[2].to == k.AH && girl[2].to2 == k.EH && girl[2].blend == 0.0f, "Nigerian neva: \"ah\"");
            expectEquals (girl[14].blend, 1.0f, "turning to \"eh\" on a held note (gel)");

            expect (run (SouthernVowels::nigerian, { sung (k.AE, 0.1f), sung (k.AE, 0.1f) })[0].kind == D::none, "not on a guess");

            std::vector<Frame> fuzzy (6, sung (k.IH, 0.3f));                  // a user's "and", sung near "in"
            fuzzy.insert (fuzzy.end(), 6, sung (k.IH, 0.65f));
            auto in = run (SouthernVowels::spanish, fuzzy);
            bool none = true;
            for (auto& d : in) none = none && d.kind == D::none;
            expect (none, "an \"i\" not clear from its start is left alone");
            expect (run (SouthernVowels::nigerian, { sung (k.IH, 0.3f), sung (k.IH, 0.6f), sung (k.IH, 0.8f) })[1].to == k.IY,
                    "\"it\": clear within its first frames");
            auto you = run (SouthernVowels::jamaican, std::vector<Frame> (6, sung (k.ER, 0.35f)));
            expect (std::all_of (you.begin(), you.end(), [] (const D& d) { return d.kind == D::none; }),
                    "the start of \"you\" heard as a weak \"er\": left alone");

            // the user's "girl" as the stream hears it: the "u" of "book" on top, "er" second (P(er | vowel) ~0.15),
            // then the dark l ("ah"); the "er" is sure from its 2nd such frame on, and stays to the vowel's end
            std::vector<Frame> userGirl { sung (k.AH, 0.5f, 0.05f), sung (k.UH, 0.3f, 0.15f), sung (k.UH, 0.35f, 0.17f),
                                          sung (k.UH, 0.4f, 0.15f), sung (k.UH, 0.45f, 0.08f), sung (k.UH, 0.5f, 0.12f),
                                          sung (k.AA, 0.35f, 0.02f), sung (k.AA, 0.3f, 0.01f) };
            auto gyal = run (SouthernVowels::jamaican, userGirl);
            expect (gyal[2].kind == D::none, "not on its first sign");
            expect (gyal[3].kind == D::swap && gyal[3].from == k.ER && gyal[3].to == k.AA, "gyal: \"er\" -> \"ah\"");
            expect (gyal[4].kind == D::swap && gyal[7].kind == D::swap && gyal[7].from == k.ER, "to the vowel's end");
            expect (gyal[8].kind == D::none, "the consonant ends it");
            expect (run (SouthernVowels::spanish, userGirl)[5].to == k.EH, "Spanish gel");

            auto good = run (SouthernVowels::jamaican, std::vector<Frame> (8, sung (k.UH, 0.5f, 0.05f)));
            expect (std::all_of (good.begin(), good.end(), [] (const D& d) { return d.kind == D::none; }),
                    "\"good\" (a \"u\" with little \"er\"): left alone");
            std::vector<Frame> once (8, sung (k.UH, 0.5f, 0.05f));
            once[3] = sung (k.UH, 0.5f, 0.6f);
            auto look = run (SouthernVowels::jamaican, once);
            expect (std::all_of (look.begin(), look.end(), [] (const D& d) { return d.kind == D::none; }),
                    "one frame that sounds like \"er\" is not enough");
            auto boot = run (SouthernVowels::jamaican, std::vector<Frame> (8, sung (k.AO, 0.5f, 0.2f)));
            expect (std::all_of (boot.begin(), boot.end(), [] (const D& d) { return d.kind == D::none; }),
                    "only when the \"u\" of \"book\" is on top");
            Frame whisper = sung (k.AE);
            whisper.voiced = false;
            expect (run (SouthernVowels::nigerian, { whisper, whisper })[0].kind == D::none, "only voiced vowels");
            expect (run (SouthernVowels::nigerian, { sung (k.AA), sung (k.AA) })[0].kind == D::none, "other vowels alone");
        }
    }
};

static SouthernVowelsTests southernVowelsTests;

} // namespace voxa::neural
