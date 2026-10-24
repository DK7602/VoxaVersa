#pragma once

#include <juce_core/juce_core.h>
#include <array>
#include <cmath>
#include <vector>

namespace voxa::sing
{

//==============================================================================
/** Band-limited Liljencrants-Fant glottal flow-derivative wavetables.

    One period of the LF model is generated for a grid of Rd values (Fant's single "voice quality"
    parameter: 0.3 = pressed/tense .. 2.7 = lax/breathy) and stored as mip-mapped tables, each limited
    to a maximum harmonic count (half-octave spacing), so that the oscillator never aliases. Built
    once (thread-safe) on first use; read-only afterwards. */
class GlottalTables
{
public:
    static constexpr int tableSize = 2048;
    static constexpr int numRd = 13;
    static constexpr int numLevels = 20;
    static constexpr float rdMin = 0.3f;
    static constexpr float rdStep = 0.2f;

    static const GlottalTables& get();

    const float* table (int rdIndex, int level) const noexcept
    {
        return data.data() + ((size_t) rdIndex * numLevels + (size_t) level) * (size_t) (tableSize + 1);
    }

    /** Most detailed level whose highest harmonic stays at or below `maxHarmonic`. */
    static int levelForMaxHarmonic (float maxHarmonic) noexcept;
    static int harmonicsAtLevel (int level) noexcept;

    /** Normalised time (0..1) at which the glottis closes for the given Rd index. */
    float closingInstant (int rdIndex) const noexcept   { return closing[(size_t) rdIndex]; }

private:
    GlottalTables();
    std::vector<float> data;
    std::array<float, numRd> closing {};
};

//==============================================================================
/** Fast deterministic white noise in [-1, 1]. */
struct NoiseSource
{
    juce::uint32 state = 0x12345678u;

    void seed (juce::uint32 s) noexcept   { state = s != 0 ? s : 0x9E3779B9u; }

    inline float next() noexcept
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float) (juce::int32) state * (1.0f / 2147483648.0f);
    }

    /** Roughly unit-variance Gaussian (sum of three uniforms). */
    inline float gaussian() noexcept
    {
        return (next() + next() + next());
    }
};

//==============================================================================
/** Topology-preserving-transform state variable filter (Simper). Stable under fast modulation. */
struct Svf
{
    float a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, k = 1.0f;
    float ic1 = 0.0f, ic2 = 0.0f;
    float bp = 0.0f;

    void reset() noexcept   { ic1 = ic2 = bp = 0.0f; }

    void set (float frequency, float bandwidth, float sampleRate) noexcept
    {
        frequency = juce::jlimit (20.0f, 0.45f * sampleRate, frequency);
        const float g = std::tan (juce::MathConstants<float>::pi * frequency / sampleRate);
        k = juce::jlimit (0.01f, 4.0f, bandwidth / frequency);
        a1 = 1.0f / (1.0f + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;
    }

    /** Returns the low-pass (unity DC gain, resonant peak = Q) output; bp holds the band-pass. */
    inline float lowpass (float x) noexcept
    {
        const float v3 = x - ic2;
        const float v1 = a1 * ic1 + a2 * v3;
        const float v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1 = 2.0f * v1 - ic1;
        ic2 = 2.0f * v2 - ic2;
        bp = v1;
        return v2;
    }

    /** High-pass output (12 dB/octave below the corner). */
    inline float highpass (float x) noexcept
    {
        const float lp = lowpass (x);
        return x - k * bp - lp;
    }

    /** Band-pass normalised to unity gain at the centre frequency. */
    inline float bandpass (float x) noexcept
    {
        lowpass (x);
        return k * bp;
    }
};

//==============================================================================
/** Control-rate synthesis parameters (one frame per ~1 ms). */
struct SynthFrame
{
    static constexpr int numFormants = 14;   // F1-F5 move; F6-F14 model the upper tract

    float f0 = 120.0f;           // Hz
    float av = 0.0f;             // voicing amplitude
    float ah = 0.0f;             // aspiration amplitude (excites the vocal tract)
    float af = 0.0f;             // frication amplitude (parallel noise branch)
    float rd = 1.0f;             // glottal shape
    std::array<float, numFormants> formants { 500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500, 11500, 12500, 13500 };
    std::array<float, numFormants> bandwidths { 80, 100, 140, 250, 300, 450, 600, 800, 850, 900, 950, 1000, 1100, 1200 };
    float nasalZeroHz = 1500.0f;
    float nasality = 0.0f;       // 0..1 depth of the nasal anti-resonance
    std::array<float, 2> fricFreq { 5000, 8000 };
    std::array<float, 2> fricBandwidth { 2000, 3000 };
    std::array<float, 2> fricGain { 1, 0.5f };
    float fricBypass = 0.0f;
    float fricHighpass = 1000.0f;
};

/** Voice-quality settings of the source (per voice, not per frame). */
struct SourceCharacter
{
    float breathiness = 0.0f;    // 0..1 aspiration mixed into voicing
    float rasp = 0.0f;           // 0..1 period-doubling, irregular pulses, extra noise
    float jitter = 0.003f;       // relative period perturbation (std-dev)
    float shimmer = 0.03f;       // relative amplitude perturbation (std-dev)
};

//==============================================================================
/** The formant synthesizer core shared by the offline renderer and the live singer.

    Source: band-limited LF glottal pulses with jitter, shimmer and rasp, plus glottally-gated
    aspiration noise. Tract: nasal anti-resonance followed by a 14-formant cascade (F1-F5 move,
    F6-F14 model the upper tract and act as the "higher-pole correction" that keeps the spectrum
    natural above 4 kHz; resonances above 0.45 fs are skipped). Frication: a parallel two-resonance noise branch with a bypass.
    process() is real-time safe. */
class VocalSynth
{
public:
    void prepare (double sampleRate, juce::uint32 seed);
    void reset() noexcept;
    void setSourceCharacter (const SourceCharacter&) noexcept;

    /** Renders numSamples (replacing the buffer contents), interpolating f0 and the amplitudes
        from the previous frame towards `frame`; filter settings are taken from `frame`. */
    void process (const SynthFrame& frame, float* output, int numSamples) noexcept;

    double getSampleRate() const noexcept   { return sampleRate; }

private:
    void startPeriod (float f0, float rd) noexcept;

    double sampleRate = 48000.0;
    float sr = 48000.0f, invSr = 1.0f / 48000.0f;
    const GlottalTables* tables = nullptr;
    SourceCharacter character;

    // Glottal oscillator
    float phase = 0.0f;
    float periodFreqFactor = 1.0f, periodAmp = 1.0f, closing = 0.6f, rdFrac = 0.0f;
    const float* tableA = nullptr;
    const float* tableB = nullptr;
    bool oddPeriod = false;

    // Interpolated controls
    float lastF0 = 120.0f, lastAv = 0.0f, lastAh = 0.0f, lastAf = 0.0f, lastRd = 1.0f;

    std::array<Svf, SynthFrame::numFormants> cascade;
    int numActiveFormants = SynthFrame::numFormants;
    Svf nasal, fric1, fric2, fricHp;
    float nasalK = 0.0f;
    float aspHpState = 0.0f, aspHpCoeff = 0.95f;
    float shelfState = 0.0f, shelfCoeff = 0.5f, shelfGain = 1.0f;
    float dcX = 0.0f, dcY = 0.0f, dcCoeff = 0.997f;

    NoiseSource noiseA, noiseB, noiseC;
};

} // namespace voxa::sing
