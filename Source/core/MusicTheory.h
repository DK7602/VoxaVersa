#pragma once

#include <juce_core/juce_core.h>
#include <array>
#include <cmath>
#include <vector>

namespace voxa
{

enum class ScaleType
{
    Major = 0,
    NaturalMinor,
    Chromatic,
    MajorPentatonic,
    MinorPentatonic,
    Blues,
    Dorian,
    Mixolydian,
    HarmonicMinor,
    Phrygian,
    Lydian,
    NumScales
};

/** Generic (scale-aware) harmony intervals used by the harmonizer. */
enum class HarmonyInterval
{
    Off = 0,
    OctaveDown,
    SixthDown,
    FifthDown,
    FourthDown,
    ThirdDown,
    Unison,
    ThirdUp,
    FourthUp,
    FifthUp,
    SixthUp,
    OctaveUp,
    NumIntervals
};

namespace music
{
    inline float midiToHz (float midi) noexcept   { return 440.0f * std::exp2 ((midi - 69.0f) / 12.0f); }
    inline float hzToMidi (float hz) noexcept     { return 69.0f + 12.0f * std::log2 (hz / 440.0f); }
    inline float semitonesToRatio (float st) noexcept { return std::exp2 (st / 12.0f); }
    inline float ratioToSemitones (float r) noexcept  { return 12.0f * std::log2 (r); }

    const juce::StringArray& getKeyNames();      // "C", "C#", ... "B"
    const juce::StringArray& getScaleNames();    // matches ScaleType order
    const juce::StringArray& getHarmonyIntervalNames(); // matches HarmonyInterval order

    /** Ascending semitone offsets (0..11) of the scale relative to its root. */
    const std::vector<int>& scaleSteps (ScaleType);

    bool isInScale (int midiNote, int keyRoot, ScaleType);

    /** Nearest in-scale MIDI note to a (fractional) MIDI pitch. Ties resolve downwards. */
    int nearestScaleNote (float midi, int keyRoot, ScaleType);

    /** Moves `steps` scale degrees away from `midiNote` (which is first snapped into the scale). */
    int diatonicTranspose (int midiNote, int steps, int keyRoot, ScaleType);

    /** The harmony note for `melodyNote` at the given generic interval, respecting the scale.
        Returns -1 for HarmonyInterval::Off. */
    int harmonyNote (int melodyNote, HarmonyInterval, int keyRoot, ScaleType);

    /** "C4", "F#3", ... (MIDI 60 = C4) */
    juce::String noteName (int midiNote);
}

/** Krumhansl-Schmuckler key estimation from a running, decaying pitch-class histogram. */
class KeyDetector
{
public:
    void reset() noexcept;
    void addPitch (float midi, float weight) noexcept;
    void decay (float factor) noexcept;

    int getKeyRoot() const noexcept          { return keyRoot; }
    ScaleType getScale() const noexcept      { return scale; }
    float getConfidence() const noexcept     { return confidence; }
    float getTotalWeight() const noexcept;

    /** Re-estimates the key from the current histogram. Cheap (24 correlations of length 12). */
    void update() noexcept;

private:
    std::array<float, 12> histogram {};
    int keyRoot = 0;
    ScaleType scale = ScaleType::Major;
    float confidence = 0.0f;
};

} // namespace voxa
