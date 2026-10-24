#pragma once

#include "Biquad.h"
#include <array>
#include <complex>
#include <vector>

namespace voxa::dsp
{

/** Estimates whether the input voice is male or female from two cues that survive singing:
    the median fundamental (f0) and the vocal-tract length, read from the 3rd and 4th formants
    (LPC on ~11 kHz audio). Pitch alone fails for men singing high or women singing low.

    Score = ln(f0 / f0Centre) / f0Scale + ln((F3 + F4) / formantCentre) / formantScale; > 0 means female.
    Calibrated on 15 dry pro acapellas (10 male, 5 female). */
class VoiceGenderEstimator
{
public:
    enum class Gender { Unknown = 0, Male, Female };

    void prepare (double sampleRate);
    void reset();
    void process (const float* input, int numSamples) noexcept;

    /** Feeds an accurate f0 estimate (Hz) for a voiced frame from the main pitch detector. */
    void addPitch (float f0Hz) noexcept;

    Gender getGender() const noexcept         { return gender; }
    float getScore() const noexcept           { return score; }
    float getMedianF0() const noexcept        { return medianF0; }
    float getMedianF3F4() const noexcept      { return medianF34; }
    int getNumVoicedFrames() const noexcept   { return totalFrames; }

    /** Re-evaluates the decision now (normally done every 50 analysed frames). */
    void finalise() noexcept { updateDecision(); }

    /** Offline helper: analyses a whole clip. */
    static Gender analyse (const float* input, int numSamples, double sampleRate, float* scoreOut = nullptr);

private:
    void analyseFrame() noexcept;
    void updateDecision() noexcept;
    bool lpc (const float* x, int n, std::array<double, 13>& a) const noexcept;
    int findFormants (const std::array<double, 13>& a, float* formants) const noexcept;

    static constexpr int lpcOrder = 12;
    static constexpr int historySize = 4096;   // ~40 s of voiced frames

    double sr = 48000.0, fs = 12000.0;
    int decimation = 4, decimCounter = 0;
    Biquad aa1, aa2;

    std::vector<float> frame, work;
    int frameLength = 360, hop = 120, writeIndex = 0, hopCounter = 0;

    std::array<float, historySize> f0History {}, f34History {};
    int historyCount = 0, historyWrite = 0, totalFrames = 0, framesSinceDecision = 0;
    int f0Count = 0, f0Write = 0, pitchDecimator = 0;
    float peakRms = 0.0f;
    std::vector<float> scratch;

    Gender gender = Gender::Unknown;
    float score = 0.0f, medianF0 = 0.0f, medianF34 = 0.0f;
    float prevSample = 0.0f;
};

} // namespace voxa::dsp
