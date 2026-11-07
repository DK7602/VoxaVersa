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
    /** Voice Transform accents, as FactoryVoices' `singAccent` numbers them. Southern is the country voices' "ah"
        (`Rule`); the others are the reggae, Latin and afrobeats voices' vowels (`AccentRule`). */
    enum Accent { noAccent = 0, southern = 1, jamaican = 2, spanish = 3, nigerian = 4 };

    /** The recognizer embedded in the plug-in; nullptr when its data is missing or broken. */
    static const SouthernVowels* get();

    /** What the rule needs from the recognizer for one 20 ms frame. */
    struct Frame
    {
        bool voiced = false;
        float vowel = 0.0f;   // probability that it is a vowel (any)
        float iy = 0.0f;      // ... the vowel part of "I" or "oy"
        float glide = 0.0f;   // ... their glide towards "ee"
        std::array<float, 3> shift {};   // ... "eh" before n / m, "ay", the "a" of "cat" (ShiftRule's shifts)
        int8_t top = -1;                 // the most likely plain vowel (a class index; glides aside)
        float topShare = 0.0f;           // ... its probability relative to P(vowel)
        float er = 0.0f;                 // P("er" | vowel) (0 when the data has no "er" class)
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

    /** Step 2 of the accent, the approved prototype's other changes: each a shift of the content features by a
        difference of vowel means, in shapes that can be decided live. `raise` ("eh" before n / m towards "ih":
        friend, ten) over the whole vowel, in and out over 3 frames; `lower` (the start of "ay": pain, taking) from
        the vowel's start, in over 3 frames and gone after `lowerFrames`; `drawl` (a glide towards "ih" in the "a"
        of and, laughing) a 160 ms bump from `drawlDelay` frames into the vowel. A vowel gets at most one, once the
        vowel so far (3 frames or more) is that vowel by its mean P(class | vowel). */
    struct ShiftRule
    {
        enum Shift { raise, lower, drawl, numShifts };
        static constexpr std::array<float, numShifts> needed { 0.4f, 0.4f, 0.4f };   // mean P(class | vowel)
        static constexpr int lowerFrames = 8, drawlDelay = 5;

        /** A frame's shift (-1: none) and weight, and the weights it gives the two frames before it (negative:
            as they were). */
        struct Decision
        {
            int shift = -1;
            float weight = 0.0f;
            std::array<float, 2> before { -1.0f, -1.0f };
        };

        void reset() noexcept { *this = {}; }
        /** Decides frame `f`, given the two after it. */
        Decision step (const Frame& f, const Frame& next, const Frame& afterNext) noexcept;

        int shift = -1, index = -1, drawlStart = 0;   // the vowel's shift; frames into the vowel
        std::array<double, numShifts> sums {};        // P(class | vowel) summed over the vowel so far
    };

    /** The Jamaican, Spanish and Nigerian accents of the reggae, Latin and afrobeats voices (listening audit
        rounds 17 - 25, tools/research/accents): their vowels, decided live like the Southern "ah".
        - "ay" and "oh" sung pure (take -> tek, slow -> slo): once a vowel has sounded like "ay" / "oh" for
          `holdAfter` frames, the generator gets those first frames' features (nudged towards "eh" / "aw") to the
          end of the vowel, fading in over 2 frames.
        - each other vowel moved to the accent's (`target`): the content features shifted by the difference of the
          two vowels' means - the "a" of that / and -> "ah" (all three), "er" (girl, never) -> "ah" (Jamaican gyal),
          "eh" (Spanish) or "ah" turning to "eh" on a held note (Nigerian neva / gel), "i" -> "ee" (Spanish and
          Nigerian: dees, ees, eet). The push the audit approved: x1.8, rising to x2.4 on a held note
          (`longVowel` frames on), in over 2 frames and half on the vowel's last frame. */
    struct AccentRule
    {
        static constexpr float vowelNeeded = 0.5f;       // P(vowel)
        static constexpr float classNeeded = 0.25f;      // P(top vowel | vowel), to act on it
        // stricter for the two a singer's own vowels get mistaken for (a user's "and" sung near "in", the start of
        // "you" heard as "er"): an "er" needs this per frame; an "i" must sound this clearly like one within its
        // first `iClearWithin` frames heard as "i" (a user's think, it: yes; and: no)
        static constexpr float erNeeded = 0.45f, iClearNeeded = 0.55f;
        static constexpr int iClearWithin = 3;
        // a real singer's "er" (the user's "girl") the stream's encoder, hearing only ~60 ms after it, takes for the
        // "u" of "book": that on top, with P(er | vowel) at least `erHeard` and its mean over the vowel so far at
        // least `erMeanNeeded` from the vowel's 3rd frame (the AI singers' sung takes: 3 of 80 "u" vowels, 1 with
        // the clear "er" alone; tools/research/vowelnet/stream_er_uh.py). After
        // `erLatch` such frames in a row (or "er" frames) the rest of the vowel is the accent's.
        static constexpr float erHeard = 0.10f, erMeanNeeded = 0.12f;
        static constexpr int erLatch = 2;
        static constexpr int holdAfter = 3;              // frames of "ay" / "oh" before it is held
        static constexpr int longVowel = 12;             // frames: the push grows from here over 4 frames
        static constexpr float shortPush = 1.8f, longPush = 2.4f, pureNudge = 0.8f;

        /** Class indices of the vowels the accents use (-1: missing). */
        struct Classes { int AA = -1, AE = -1, AH = -1, AO = -1, EH = -1, ER = -1, EY = -1, IH = -1, IY = -1, OW = -1, UH = -1; };

        struct Decision
        {
            enum Kind : uint8_t { none, swap, hold } kind = none;
            int8_t from = -1, to = -1, to2 = -1;   // swap: from -> (1 - blend) to + blend to2
            float weight = 0.0f, blend = 0.0f;
            bool startsHold = false;               // hold: take the hold from this and the frames before it
        };

        static bool isVowel (const Frame& f) noexcept { return f.voiced && f.vowel >= vowelNeeded && f.top >= 0; }

        void reset() noexcept
        {
            run = sameRun = holdAt = erFrames = 0;
            last = -1;
            holding = iClear = erVowel = false;
            erSum = 0.0f;
        }
        /** Decides frame `f`, given the one after it. */
        Decision step (const Frame& f, const Frame& next) noexcept;

        int accent = southern;   // which accent the stream runs (State's default: the Southern "ah")
        Classes classes;
        int run = 0, sameRun = 0, last = -1, holdAt = 0;   // vowel frames so far; of its current class
        bool holding = false, iClear = false;
        float erSum = 0.0f;                                // P(er | vowel) summed over the vowel
        int erFrames = 0;                                  // "er"-like frames in a row
        bool erVowel = false;                              // this vowel is an "er" (latched)
    };

    /** Per stream: the rule's state and the last frames (absolute 20 ms frame numbers). */
    struct State
    {
        State() noexcept { reset(); }
        void reset() noexcept;

        Rule rule;
        ShiftRule shiftRule;
        AccentRule accentRule;                      // its `accent` says which accent runs (Southern: `rule`)
        int64_t recognised = -1;                    // the last frame recognised (decided: `Rule::lookahead` before)
        static constexpr int history = 512;
        std::array<int64_t, history> frameOf {};    // which frame each slot holds
        std::array<Frame, history> frames {};       // the recognizer's output
        std::array<float, history> weight {};       // how much of the hold it gets
        std::array<uint8_t, history> holdIndex {};  // which of the two holds
        std::array<int8_t, history> shiftOf {};     // its shift (-1: none)
        std::array<float, history> shiftWeight {};  // and how much of it
        std::array<AccentRule::Decision, history> accentOf {};   // the accent's swap (holds use weight / holdIndex)
        std::vector<float> holds[2];
        uint8_t currentHold = 0;
    };

    /** Edits one chunk of generator features in place. `features`: the content encoder's output for the whole
        chunk, time-major [numEncoderFrames][channels] (20 ms frames; frame e covers the 10 ms frames 2e, 2e + 1);
        `f0Frames`: per 10 ms frame, 0 where unvoiced; `phone`: the generator's frames [skip, numFrames), each the
        features of encoder frame frame / 2; `firstFrame`: the absolute 10 ms frame where the chunk starts.
        `changed` (optional, numFrames - skip values): how much each generator frame was given an accent's vowel
        (0..1; Jamaican, Spanish and Nigerian only), so the plug-in's vowel keeper lets those vowels through.
        `lookAheadFrames`: the chunk's last 10 ms frames are context only (offline look-ahead, not sung): frames
        are recognised up to the sung ones and the rule's lookahead, so each is heard with what follows it. */
    void apply (const float* features, int numEncoderFrames, int channels, const float* f0Frames, int numFrames,
                int skip, int64_t firstFrame, float* phone, State& state, std::vector<float>& scratch,
                float* changed = nullptr, int lookAheadFrames = 0) const;

    /** Sets the stream's accent (an `Accent`); a change resets it. Southern runs `Rule`; Jamaican, Spanish and
        Nigerian run `AccentRule` when the data has the vowel means (version 3), else nothing. */
    void setAccent (State& state, int accent) const noexcept;
    /** True when the data has every class's mean features (version 3: the accents' vowels). */
    bool hasMeans() const noexcept { return ! means.empty(); }
    const AccentRule::Classes& getClasses() const noexcept { return accentClasses; }
    /** A class's mean features (version 3; empty without). */
    const std::vector<float>& getMean (int cls) const noexcept
    {
        static const std::vector<float> none;
        return cls >= 0 && (size_t) cls < means.size() ? means[(size_t) cls] : none;
    }

    int getChannels() const noexcept { return channels; }
    /** True when the data has step 2's shifts. */
    bool hasShifts() const noexcept
    {
        for (const auto& d : deltas)
            if (d.size() != (size_t) channels)
                return false;
        return true;
    }

    /** The recognizer's probabilities for one encoder frame (context frames clamped to the chunk). */
    Frame recognise (const float* features, int numEncoderFrames, int frame, std::vector<float>& scratch) const;

private:
    SouthernVowels() = default;
    bool load (const void* data, size_t size);

    int channels = 0, context = 0, hidden = 0, numClasses = 0;
    std::vector<float> w1, b1, w2, b2;   // the standardisation is folded into w1 / b1
    std::vector<int> vowelClasses, iyClasses, glideClasses, plainVowelClasses;
    AccentRule::Classes accentClasses;
    std::vector<std::vector<float>> means;   // per class (version 3)
    std::array<int, ShiftRule::numShifts> shiftClasses { -1, -1, -1 };   // EH_N, EY, AE
    std::array<std::vector<float>, ShiftRule::numShifts> deltas;         // the shifts (version 2 data)
};

} // namespace voxa::neural
