#include "neural/SouthernVowels.h"

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
        }
    }
};

static SouthernVowelsTests southernVowelsTests;

} // namespace voxa::neural
