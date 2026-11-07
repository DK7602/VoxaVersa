#pragma once

#include "SouthernVowels.h"

#include <juce_core/juce_core.h>
#include <array>
#include <vector>

namespace voxa::neural
{

/** The consonants of the reggae, Latin and afrobeats voices' accents (listening audit rounds 20 - 28,
    tools/research/accents): "th" sung as "t" / "d" (tink, dat, dis, de), the "d" after "n" dropped (an' you) and
    the "r" after a vowel not sung (fo). Like their vowels (SouthernVowels::AccentRule) it works inside the AI
    singer, on what its content encoder hears, decided live with 2 frames of lookahead, so it adds no latency.

    A small consonant recognizer (an MLP on 5 frames of content features, trained on LibriSpeech's phone
    alignments, whole files and the windows the live stream computes; consonant_feats.py, consonant_stream.py)
    finds the sounds. The generator then gets, for a "th", Kokoro's own "t" / "d" content frames (closure, burst,
    aspiration / voice bar, burst; the last two at the last two frames of the "th") and the pitch a "t" or "d"
    has (none under a "t"; under a "d" the note around it); for the "d" after "n", the "n"; for the "r" after a
    vowel, that vowel. Round 25 A made these by editing the recording at known times, which a live plug-in
    cannot; round 28 (C) checked the same sounds made this way. */
class AccentConsonants
{
public:
    /** The recognizer embedded in the plug-in; nullptr when its data is missing or broken. */
    static const AccentConsonants* get();

    /** What the rule needs from the recognizer for one 20 ms frame. */
    struct Frame
    {
        bool voiced = false;
        float th = 0.0f, dh = 0.0f, d = 0.0f, t = 0.0f, n = 0.0f, r = 0.0f, y = 0.0f, l = 0.0f, vowel = 0.0f;   // probabilities
        bool er = false;   // the vowel rule decided this frame an "er" (SouthernVowels::State::isEr)
    };

    /** The decision rule, one frame at a time with two frames of lookahead (separate from the recognizer so it can
        be tested). */
    struct Rule
    {
        static constexpr float thNeeded = 0.5f;     // P(th) + P(dh)
        // P(d) + P(t) after an "n" (live, the "d" of "and" turns to "t"); P(n) of that "n"
        static constexpr float dNeeded = 0.45f, nNeeded = 0.5f;
        // a sung vowel's P(vowel) and an "r"'s P(r): a vowel shares its frames with the classes around it (~0.4 on
        // the user's take), and the "r" of "for" grows out of it (0.3 -> 0.9)
        static constexpr float vowelNeeded = 0.35f, rNeeded = 0.3f;
        static constexpr int maxDroppedD = 8;       // frames: a longer "d" / "t" after "n" is one to keep
        static constexpr int nWithin = 2;           // the "d" / "t" starts this many frames after the "n" at most
        static constexpr int vowelBeforeR = 3;      // frames of vowel before an "r" that is not sung
        static constexpr float cleanVowelR = 0.15f; // P(r) of the vowel frames such an "r" holds
        // an "r" is held from its 2nd frame when the two after that are "r" too (at least 4 frames: not the short
        // one between vowels of "very")

        // Kokoro's "t" / "d"s (4 frames each), and the Latin accent's Spanish rolled "rr" (2), the "j" of "jou" (5)
        // and the light Spanish "l" (4)
        enum Template : int8_t { t = 0, dA, dE, dI, roll, jh, lightL, numTemplates };
        static constexpr float lNeeded = 0.5f;   // P(l) of the "l" after an "er" (girl) that is sung light
        static constexpr float notYetL = 0.3f;   // P(l) below which a frame of that "er" can take its rolled r
        static constexpr int erBeforeRoll = 3;   // frames of "er" before its rolled r
        static constexpr float yNeeded = 0.5f;   // P(y) of the "y" that becomes "j"

        struct Decision
        {
            enum Kind : uint8_t { none, closure, voiceBar, stop, holdN, holdVowel, roll, jh, lightL } kind = none;
            int8_t frame = 0;          // stop: which of its template's last two frames (2 or 3 of 0..3); roll, jh:
                                       // which of theirs
            bool voicedStop = false;   // a "d" (from "dh"), else a "t"
            float weight = 1.0f;       // holdN / holdVowel: fading in
        };

        void reset() noexcept { *this = {}; }
        /** Decides frame `f`, given the two after it. */
        Decision step (const Frame& f, const Frame& next, const Frame& afterNext) noexcept;

        static bool isTh (const Frame& f) noexcept    { return f.th + f.dh >= thNeeded; }
        static bool isVowel (const Frame& f) noexcept { return f.voiced && f.vowel >= vowelNeeded; }

        int thRun = 0;                // frames of this "th" so far
        float thSum = 0.0f, dhSum = 0.0f;
        bool legato = false;          // the voice ran into this "th" (a "d"'s voice bar sings the note before it)
        bool prevVoiced = false;      // the frame before was voiced
        int sinceN = 99;              // frames since the last "n" (the frame between "n" and "d" is often neither)
        int droppedD = 0;             // frames of the "d" after "n" sung as "n"
        int vowelRun = 0;             // vowel frames just before
        int rRun = 0;
        bool holdingVowel = false;
        bool spanish = false;         // the Latin accent: the "r" after a vowel rolled, "y" -> "j", "gurrl"
        int yRun = 0;
        int erRun = 0, sinceEr = 99, lRun = 0;   // "er" frames in a row; frames since the last; "l" frames
    };

    /** Per stream: the rule's state and the last frames (absolute 20 ms frame numbers). */
    struct State
    {
        State() noexcept { reset(); }
        void reset() noexcept;

        Rule rule;
        int64_t recognised = -1;
        static constexpr int history = 512;
        std::array<int64_t, history> frameOf {};
        std::array<Frame, history> frames {};
        std::array<Rule::Decision, history> decisions {};
        std::array<int8_t, history> stopTemplate {};   // the "d" template for the vowel after it
        std::array<float, history> stopPitch {};       // the pitch under a "d" (0: keep)
        std::array<uint8_t, history> holdIndex {};
        int8_t currentTemplate = Rule::dA;             // the "d" of the current "th"
        std::vector<float> holds[2];                  // the "n" / the vowel a hold sings
        uint8_t currentHold = 0;
        float lastPitch = 0.0f;                        // the last voiced pitch decided
        bool latin = false;                            // the Latin accent's consonants (set by the caller; kept by reset())
    };

    /** Edits one chunk of generator features (and its pitch) in place, as SouthernVowels::apply (same arguments;
        call it after that). `f0Frames` (per 10 ms frame, 0 where unvoiced) gets a "t"'s silence and a "d"'s
        voicing; `changed` (optional, numFrames - skip values) is raised to 1 where a consonant was changed, so the
        plug-in's consonant, vowel and r keepers let the AI singer's own sound through there. `lookAheadFrames`:
        as SouthernVowels::apply. `vowels`: the vowel rule's state for the same chunk (which frames are an "er"). */
    void apply (const float* features, int numEncoderFrames, int channels, float* f0Frames, int numFrames,
                int skip, int64_t firstFrame, float* phone, State& state, std::vector<float>& scratch,
                float* changed = nullptr, int lookAheadFrames = 0, const SouthernVowels::State* vowels = nullptr) const;

    /** The recognizer's probabilities for one encoder frame (context frames clamped to the chunk). */
    Frame recognise (const float* features, int numEncoderFrames, int frame, std::vector<float>& scratch) const;

    int getChannels() const noexcept { return channels; }
    /** A template's frames ([frames][channels]); empty when missing. */
    const std::vector<float>& getTemplate (int which) const noexcept { return templates[(size_t) which]; }
    int getTemplateFrames (int which) const noexcept
    {
        return channels > 0 ? (int) (templates[(size_t) which].size() / (size_t) channels) : 0;
    }
    const std::vector<float>& getSilence() const noexcept { return silence; }
    /** True when the data has the Latin accent's rolled "rr" and "j". */
    bool hasLatin() const noexcept { return getTemplateFrames (Rule::roll) == 2 && getTemplateFrames (Rule::jh) == 5; }

    /** For tests: loads data in the file format (see load()). */
    static std::unique_ptr<AccentConsonants> fromData (const void* data, size_t size);

private:
    AccentConsonants() = default;
    bool load (const void* data, size_t size);

    int channels = 0, context = 0, hidden = 0, numClasses = 0;
    std::vector<float> w1, b1, w2, b2;   // the standardisation is folded into w1 / b1
    int thClass = -1, dhClass = -1, dClass = -1, tClass = -1, nClass = -1, rClass = -1, yClass = -1, lClass = -1, vowelClass = -1;
    std::array<std::vector<float>, Rule::numTemplates> templates;
    std::vector<float> silence;
};

} // namespace voxa::neural
