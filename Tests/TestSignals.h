#pragma once

#include "dsp/Biquad.h"
#include <functional>
#include <vector>

namespace voxa::test
{

/** A crude but voice-like test signal: glottal pulses through three formant resonators. */
inline std::vector<float> makeVoice (double sr, int numSamples, std::function<float (int)> f0At, float level = 0.3f)
{
    std::vector<float> out ((size_t) numSamples);
    dsp::Biquad f1, f2, f3;
    f1.c = dsp::BiquadCoeffs::bandPass (sr, 700.0, 6.0);
    f2.c = dsp::BiquadCoeffs::bandPass (sr, 1200.0, 8.0);
    f3.c = dsp::BiquadCoeffs::bandPass (sr, 2600.0, 10.0);
    double phase = 0.0;
    for (int i = 0; i < numSamples; ++i)
    {
        const double f0 = f0At (i);
        phase += f0 / sr;
        if (phase >= 1.0) phase -= 1.0;
        // Rosenberg-like pulse: smooth rise, quick fall, closed phase.
        float g = 0.0f;
        if (phase < 0.4)       g = (float) (0.5 * (1.0 - std::cos (juce::MathConstants<double>::pi * phase / 0.4)));
        else if (phase < 0.55) g = (float) std::cos (juce::MathConstants<double>::halfPi * (phase - 0.4) / 0.15);
        const float src = g - 0.3f;
        out[(size_t) i] = level * (0.6f * f1.process (src) + 0.35f * f2.process (src) + 0.15f * f3.process (src) + 0.2f * src);
    }
    return out;
}

/** Pumps the message loop until `condition` is true or the timeout expires. */
inline bool waitFor (std::function<bool()> condition, int timeoutMs)
{
    const auto end = juce::Time::getMillisecondCounter() + (juce::uint32) timeoutMs;
    while (! condition())
    {
        if (juce::Time::getMillisecondCounter() > end)
            return false;
        juce::MessageManager::getInstance()->runDispatchLoopUntil (10);
    }
    return true;
}

} // namespace voxa::test
