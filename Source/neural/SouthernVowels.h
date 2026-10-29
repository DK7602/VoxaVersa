#pragma once

#include <juce_core/juce_core.h>
#include <array>
#include <vector>

namespace voxa::neural
{

/** The Southern "ah" of the country voices (listening audits 10 - 13): an "I" or "oy" (I, my, time, like, joy)
    keeps its open vowel instead of gliding to "ee" - "Ah", "mah", "tahm", "lahk", "jaw".

    It works inside the AI singer, on what its content encoder hears, so it adds no latency: a small vowel
    recognizer (an MLP on 5 frames of content features, trained on Text to Sing takes sung by public-domain and
    CC BY singers and on Kokoro's American voices; tools/research/vowelnet) finds a vowel that sounds like "I"/"oy"
    and the moment its glide starts (`Rule`). From 40 ms before the glide to the end of the vowel the generator then
    gets the features of the 60 ms of the vowel before it that sound most like "ah" (not its start: a hold taken
    from next to the "l" sang "luf"), fading in over 40 ms and going straight into the next consonant (a hold that
    stopped short let the glide back in: "like" broke up). The recognizer learnt from the content features as the
    live stream computes them, chunk by chunk, as well as from whole files (offline renders). Not for Text to Sing,
    whose accents are in its own pronunciation. */
class SouthernVowels
{
public:
    /** The recognizer embedded in the plug-in; nullptr when its data is missing or broken. */
    static const SouthernVowels* get();

    /** What the rule needs from the recognizer for one 20 ms frame. */
    struct Frame
    {
        bool voiced = false;
        float vowel = 0.0f;   // probability that it is a vowel (any)
        float iy = 0.0f;      // ... the vowel part of "I" or "oy"
        float glide = 0.0f;   // ... their glide towards "ee"
    };

    /** The decision rule, one frame at a time with two frames of lookahead (separate from the recognizer so it
        can be tested). In a voiced vowel the glide starts at a frame when it and the next sound like the glide, or
        - a glide too short to see - when the vowel ends after the next frame. The vowel is held from there when
        its frames before (glide frames aside) sounded like "I"/"oy" and the last `holdFrames` still sound like
        its "ah" (not the "e" of a "pain" sung with an open start). The probabilities are taken relative to
        P(vowel): on a real singer the recognizer spreads them more than on the voices it learnt from, but still
        ranks the vowels right (tuned on VocalSet singers and a user's vocal: tools/research/vowelnet). */
    struct Rule
    {
        static constexpr float vowelNeeded = 0.5f;       // P(vowel)
        static constexpr float glideNeeded = 0.4f;       // P(glide | vowel)
        static constexpr float evidenceNeeded = 0.4f;    // mean P("I"/"oy" | vowel) over the vowel before its glide
        static constexpr float nucleusNeeded = 0.3f;     // ... over its last `holdFrames` frames
        static constexpr int holdFrames = 3;             // the features held: 60 ms of the vowel
        static constexpr int lookahead = 2;              // frames known after the one decided
        static constexpr int searchFrames = 25;          // how far back the held 60 ms are looked for

        static bool isVowel (const Frame& f) noexcept { return f.voiced && f.vowel >= vowelNeeded; }
        static bool isGlide (const Frame& f) noexcept { return isVowel (f) && f.glide >= glideNeeded * f.vowel; }

        enum class Event { none, glideStarts, holding };

        void reset() noexcept { *this = {}; }
        /** Decides frame `f`, given the two after it. */
        Event step (const Frame& f, const Frame& next, const Frame& afterNext) noexcept;

        int vowelFrames = 0;                       // frames of the vowel so far that are not glide
        double evidence = 0.0;                     // their P("I"/"oy" | vowel), summed
        std::array<float, holdFrames> recent {};   // P("I"/"oy" | vowel) of the last of them
        bool held = false;
    };

    /** Per stream: the rule's state and the last frames (absolute 20 ms frame numbers). */
    struct State
    {
        State() noexcept { reset(); }
        void reset() noexcept;

        Rule rule;
        int64_t recognised = -1;                    // the last frame recognised (decided: `Rule::lookahead` before)
        static constexpr int history = 512;
        std::array<int64_t, history> frameOf {};    // which frame each slot holds
        std::array<Frame, history> frames {};       // the recognizer's output
        std::array<float, history> weight {};       // how much of the hold it gets
        std::array<uint8_t, history> holdIndex {};  // which of the two holds
        std::vector<float> holds[2];
        uint8_t currentHold = 0;
    };

    /** Edits one chunk of generator features in place. `features`: the content encoder's output for the whole
        chunk, time-major [numEncoderFrames][channels] (20 ms frames; frame e covers the 10 ms frames 2e, 2e + 1);
        `f0Frames`: per 10 ms frame, 0 where unvoiced; `phone`: the generator's frames [skip, numFrames), each the
        features of encoder frame frame / 2; `firstFrame`: the absolute 10 ms frame where the chunk starts. */
    void apply (const float* features, int numEncoderFrames, int channels, const float* f0Frames, int numFrames,
                int skip, int64_t firstFrame, float* phone, State& state, std::vector<float>& scratch) const;

    int getChannels() const noexcept { return channels; }

    /** The recognizer's probabilities for one encoder frame (context frames clamped to the chunk). */
    Frame recognise (const float* features, int numEncoderFrames, int frame, std::vector<float>& scratch) const;

private:
    SouthernVowels() = default;
    bool load (const void* data, size_t size);

    int channels = 0, context = 0, hidden = 0, numClasses = 0;
    std::vector<float> w1, b1, w2, b2;   // the standardisation is folded into w1 / b1
    std::vector<int> vowelClasses, iyClasses, glideClasses;
};

} // namespace voxa::neural
