#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

namespace voxa::dsp
{

/** RBJ-cookbook biquad coefficients (normalised so a0 == 1). */
struct BiquadCoeffs
{
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;

    static BiquadCoeffs lowPass (double sr, double hz, double q = 0.7071)        { return make (Type::LowPass, sr, hz, q, 0.0); }
    static BiquadCoeffs highPass (double sr, double hz, double q = 0.7071)       { return make (Type::HighPass, sr, hz, q, 0.0); }
    static BiquadCoeffs bandPass (double sr, double hz, double q)                { return make (Type::BandPass, sr, hz, q, 0.0); }
    static BiquadCoeffs peak (double sr, double hz, double q, double gainDb)     { return make (Type::Peak, sr, hz, q, gainDb); }
    static BiquadCoeffs lowShelf (double sr, double hz, double gainDb, double q = 0.7071)  { return make (Type::LowShelf, sr, hz, q, gainDb); }
    static BiquadCoeffs highShelf (double sr, double hz, double gainDb, double q = 0.7071) { return make (Type::HighShelf, sr, hz, q, gainDb); }

private:
    enum class Type { LowPass, HighPass, BandPass, Peak, LowShelf, HighShelf };

    static BiquadCoeffs make (Type type, double sr, double hz, double q, double gainDb)
    {
        hz = juce::jlimit (5.0, sr * 0.49, hz);
        const double w0 = juce::MathConstants<double>::twoPi * hz / sr;
        const double cw = std::cos (w0), sw = std::sin (w0);
        const double alpha = sw / (2.0 * q);
        const double A = std::pow (10.0, gainDb / 40.0);

        double b0 = 1, b1 = 0, b2 = 0, a0 = 1, a1 = 0, a2 = 0;

        switch (type)
        {
            case Type::LowPass:
                b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2;
                a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
                break;
            case Type::HighPass:
                b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2;
                a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
                break;
            case Type::BandPass:
                b0 = alpha; b1 = 0; b2 = -alpha;
                a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
                break;
            case Type::Peak:
                b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A;
                a0 = 1 + alpha / A; a1 = -2 * cw; a2 = 1 - alpha / A;
                break;
            case Type::LowShelf:
            {
                const double s = 2 * std::sqrt (A) * alpha;
                b0 = A * ((A + 1) - (A - 1) * cw + s);
                b1 = 2 * A * ((A - 1) - (A + 1) * cw);
                b2 = A * ((A + 1) - (A - 1) * cw - s);
                a0 = (A + 1) + (A - 1) * cw + s;
                a1 = -2 * ((A - 1) + (A + 1) * cw);
                a2 = (A + 1) + (A - 1) * cw - s;
                break;
            }
            case Type::HighShelf:
            {
                const double s = 2 * std::sqrt (A) * alpha;
                b0 = A * ((A + 1) + (A - 1) * cw + s);
                b1 = -2 * A * ((A - 1) + (A + 1) * cw);
                b2 = A * ((A + 1) + (A - 1) * cw - s);
                a0 = (A + 1) - (A - 1) * cw + s;
                a1 = 2 * ((A - 1) - (A + 1) * cw);
                a2 = (A + 1) - (A - 1) * cw - s;
                break;
            }
        }

        BiquadCoeffs c;
        c.b0 = (float) (b0 / a0); c.b1 = (float) (b1 / a0); c.b2 = (float) (b2 / a0);
        c.a1 = (float) (a1 / a0); c.a2 = (float) (a2 / a0);
        return c;
    }
};

/** Transposed direct form II biquad. */
struct Biquad
{
    BiquadCoeffs c;
    float z1 = 0.0f, z2 = 0.0f;

    void reset() noexcept { z1 = z2 = 0.0f; }

    inline float process (float x) noexcept
    {
        const float y = c.b0 * x + z1;
        z1 = c.b1 * x - c.a1 * y + z2;
        z2 = c.b2 * x - c.a2 * y;
        return y;
    }

    void process (float* data, int n) noexcept
    {
        for (int i = 0; i < n; ++i)
            data[i] = process (data[i]);
    }

    /** Flushes denormals / runaway states (call once per block). */
    void sanitise() noexcept
    {
        if (! std::isfinite (z1) || ! std::isfinite (z2) || std::abs (z1) < 1.0e-20f)
            z1 = z2 = 0.0f;
    }
};

/** One-pole smoother for control values. */
struct Smoothed
{
    float value = 0.0f, target = 0.0f, coeff = 0.0f;

    void prepare (double rateHz, double timeMs) noexcept
    {
        coeff = timeMs <= 0.0 ? 0.0f : (float) std::exp (-1.0 / (rateHz * timeMs * 0.001));
    }
    void reset (float v) noexcept       { value = target = v; }
    void setTarget (float t) noexcept   { target = t; }
    inline float next() noexcept        { value = target + coeff * (value - target); return value; }
};

/** Cheap xorshift white noise in [-1, 1]. */
struct Noise
{
    juce::uint32 state = 0x12345678u;
    inline float next() noexcept
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float) (state & 0xffffff) / (float) 0x7fffff - 1.0f;
    }
};

} // namespace voxa::dsp
