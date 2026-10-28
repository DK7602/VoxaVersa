#pragma once

#include <array>

#include "PitchDetector.h"
#include "Psola.h"
#include "../core/MusicTheory.h"
#include "../voices/VoiceProfile.h"

namespace voxa::dsp
{

/** Turns detected pitch into target pitch: pitch correction, MIDI control, smart register,
    transposition and expressive pitch behaviour (scoops, vibrato, micro jitter). */
class PitchMapper
{
public:
    struct Settings
    {
        float pitchSemis = 0.0f;
        float fineCents = 0.0f;

        bool tuneOn = false;
        int keyRoot = -1;                 // -1 = auto detect
        ScaleType scale = ScaleType::Major;
        float retuneMs = 40.0f;
        float humanize = 0.3f;            // 0..1
        int midiNote = -1;                // held MIDI note controlling the target (-1 = none)

        bool pitchPreserve = true;        // false = move into the target voice's register
        bool expression = true;           // profile scoops/vibrato/jitter
        float styleScale = 1.0f;          // 0..2 scales expressive traits
        float age = 0.0f;                 // -1 young .. +1 mature
        float userVibratoCents = 0.0f;
        float userVibratoRate = 5.5f;
        int sourceGender = 0;             // 0 auto, 1 male, 2 female
        bool sourceIsSynth = false;       // Text-to-Sing guide: no correction / register / profile expression
        bool naturalVibrato = false;      // AI lead: keep the singer's own vibrato and scoops, add none unless correcting
        bool aiRegister = false;          // AI lead: place the octave near traits.aiCentreMidi (its singer's range)
        VoiceTraits traits;               // target voice
    };

    void prepare (double sampleRate, int hopSamples);
    void reset();

    /** Processes one detector frame. `time` is the absolute input position the frame describes. */
    PitchFrame process (const PitchDetector::Frame& frame, juce::int64 time, const Settings& settings) noexcept;

    /** Harmony target shift (semitones relative to the source) for a generic interval. */
    static float harmonyShift (const PitchFrame& frame, HarmonyInterval interval, int keyRoot, ScaleType scale) noexcept;

    int getEffectiveKeyRoot (const Settings& s) const noexcept;
    ScaleType getEffectiveScale (const Settings& s) const noexcept;
    int getDetectedKeyRoot() const noexcept     { return keyKnown ? keyDetector.getKeyRoot() : -1; }
    ScaleType getDetectedScale() const noexcept { return keyDetector.getScale(); }
    int getSourceGenderEstimate() const noexcept { return genderEstimate; }   // 0 unknown, 1 male, 2 female
    /** Gender of the input voice from VoiceGenderEstimator (0 unknown, 1 male, 2 female). */
    void setSourceGenderEstimate (int g) noexcept { genderEstimate = g; }
    float getSourceMedianMidi() const noexcept  { return sourceMedian; }
    int getRegisterOctaves() const noexcept     { return registerOctaves; }

    /** Formant baseline of the source voice in "formant semitones" (male -1, female +1, unknown 0). */
    float getSourceFormantBaseline (int sourceGenderSetting) const noexcept;

private:
    void updateRegister (const Settings&) noexcept;
    float sourcePercentile (float fraction) const noexcept;

    double sr = 48000.0;
    float hopSec = 0.004f;

    // correction state
    float corrState = 0.0f;
    int currentNote = -1;
    bool wasVoiced = false;
    float unvoicedTime = 1.0f, noteTime = 0.0f;

    // expression state
    float scoopOffset = 0.0f;
    float vibPhase = 0.0f, vibRateJitter = 0.0f, vibDepthRamp = 0.0f;
    float jitterState = 0.0f;
    Noise noise;

    // source analysis
    float sourceMedian = 0.0f;
    float sourceWeight = 0.0f;
    std::array<float, 128> sourceNotes {};   // seconds sung on each note (recent minutes)
    float sourceNotesTotal = 0.0f;
    int genderEstimate = 0;
    int registerOctaves = 0;
    bool registerLocked = false;
    int lockedCentre = -1000, lockedSource = -1;
    KeyDetector keyDetector;
    bool keyKnown = false;
    int keyUpdateCounter = 0;
};

} // namespace voxa::dsp
