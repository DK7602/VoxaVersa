#pragma once

#include "Articulation.h"
#include "Melody.h"

#include <atomic>
#include <functional>

namespace voxa::sing
{

/** Turns a lyric-bound score into a sung take.

    Planning: every phoneme gets a time slot. Onset consonants are placed before the beat so the
    vowel lands on it; codas sit at the end of the note; legato / detached articulation follows
    the voice's traits; breaths go before phrases. Then per ~1 ms control frame the articulator
    targets are computed, smoothed with zero-phase Gaussians (coarticulation), combined with a pitch
    contour (portamento, scoops, overshoot, delayed vibrato, drift) and dynamics, and rendered by
    the VocalSynth. */
class OfflineRenderer
{
public:
    OfflineRenderer (const LyricScript&, const Score&, const VoiceTraits&, double bpm, double sampleRate, juce::uint32 seed);

    /** Returns false if cancelled. `progress` receives 0..1. */
    bool render (const std::atomic<bool>* cancel, const std::function<void (float)>& progress);

    struct Segment
    {
        PhoneEvent event;
        double start = 0.0, end = 0.0;  // seconds
        int note = -1;                  // score note whose dynamics apply
        bool nucleus = false;
        float attack = 0.0f;            // nucleus: attack time (0 = voicing continues from before)
        float attackFloor = 0.0f;       // nucleus: level the attack starts from
        float release = 0.0f;           // nucleus: fade-out time at its end (0 = none)
        float level = 1.0f;             // extra gain (used for breaths)
    };

    const std::vector<Segment>& getSegments() const noexcept   { return segments; }
    double getDurationSeconds() const noexcept                 { return durationSeconds; }

    /** Results (valid after a successful render()). */
    juce::AudioBuffer<float> audio;
    std::vector<float> f0Hz;            // 100 frames per second, 0 = unvoiced

private:
    struct TimedNote
    {
        double start = 0.0, end = 0.0;
        float midi = 60.0f;
        float velocity = 0.8f;
        bool legatoFromPrevious = false;
        double glide = 0.0;
        double attackFrom = 0.0;        // start of the onset consonants (scoop region)
        double voiceEnd = 0.0;          // end of voicing for phrase-final notes
        bool phraseEnd = false;
        bool stressed = false;          // first note of a stressed syllable (scoops land on these)
    };

    /** A syllable plus the melisma notes it is held through. */
    struct Occurrence
    {
        int syllable = -1;
        SungSyllable sung;                // the syllable as sung here (connected-speech changes applied)
        int firstNote = 0, lastNote = 0;
        double vowelStart = 0.0, noteEnd = 0.0;
        bool breathBefore = false;
        bool connected = false;           // flows into the next syllable without a rest
        bool onsetTap = false;            // the onset T / D is a flap ("bet-ter")
        bool codaTap = false;             // the final T / D is a flap into the next word ("got it")
        std::vector<float> onset, onsetMin, coda, codaMin;
        double onsetStart = 0.0, vowelEnd = 0.0, codaEnd = 0.0;
    };

    void plan();
    void applyConnectedSpeech (std::vector<Occurrence>&) const;
    void buildPitchTrack (int numFrames, double frameSeconds, std::vector<float>& midiTrack) const;

    const LyricScript& script;
    const Score& score;
    VoiceTraits traits;
    VoiceSetup setup;
    double bpm, sampleRate;
    juce::uint32 seed;

    std::vector<TimedNote> notes;
    std::vector<Segment> segments;
    double durationSeconds = 0.0;
};

} // namespace voxa::sing
