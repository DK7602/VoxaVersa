#include "Watermark.h"
#include "AudioUtils.h"

#include <juce_dsp/juce_dsp.h>
#include <complex>
#include <vector>

namespace voxa
{

namespace
{
    constexpr int patternOrder = 15;
    constexpr int patternLength = 1 << patternOrder;   // samples at the reference rate
    constexpr double referenceRate = 48000.0;
    constexpr double lowestHz = 1000.0, highestHz = 8000.0;
    constexpr int numBands = WatermarkEmbedder::numBands;

    double bandEdge (int k)
    {
        return lowestHz * std::pow (highestHz / lowestHz, (double) k / numBands);
    }

    /** The band-split noise pattern of one watermark kind. Each band is brick-wall band-limited in the
        frequency domain (so the period loops seamlessly) and normalised to unit RMS. Every table has one
        sample of wrap-around padding before and two after, for cubic interpolation. */
    struct Pattern
    {
        static constexpr int stride = patternLength + 3;
        std::vector<float> data;

        const float* band (int k) const noexcept   { return data.data() + (size_t) k * stride + 1; }

        explicit Pattern (juce::uint64 seed)
        {
            juce::dsp::FFT fft (patternOrder);
            std::vector<float> noise ((size_t) patternLength * 2, 0.0f), work ((size_t) patternLength * 2);

            // splitmix64: identical on every platform.
            auto next = [&seed]
            {
                juce::uint64 z = (seed += 0x9E3779B97F4A7C15ull);
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                return z ^ (z >> 31);
            };
            for (int i = 0; i < patternLength; ++i)
                noise[(size_t) i] = (float) ((double) (next() >> 11) * (2.0 / 9007199254740992.0) - 1.0);
            fft.performRealOnlyForwardTransform (noise.data(), true);

            data.assign ((size_t) numBands * stride, 0.0f);
            const double binHz = referenceRate / patternLength;

            for (int k = 0; k < numBands; ++k)
            {
                std::fill (work.begin(), work.end(), 0.0f);
                const int lo = (int) std::ceil (bandEdge (k) / binHz);
                const int hi = (int) std::ceil (bandEdge (k + 1) / binHz);
                for (int b = lo; b < hi; ++b)
                {
                    work[(size_t) (2 * b)] = noise[(size_t) (2 * b)];
                    work[(size_t) (2 * b + 1)] = noise[(size_t) (2 * b + 1)];
                    work[(size_t) (2 * (patternLength - b))] = noise[(size_t) (2 * b)];
                    work[(size_t) (2 * (patternLength - b) + 1)] = -noise[(size_t) (2 * b + 1)];
                }
                fft.performRealOnlyInverseTransform (work.data());

                double energy = 0.0;
                for (int i = 0; i < patternLength; ++i)
                    energy += (double) work[(size_t) i] * work[(size_t) i];
                const float norm = (float) (1.0 / std::sqrt (juce::jmax (1.0e-30, energy / patternLength)));

                float* t = data.data() + (size_t) k * stride + 1;
                for (int i = 0; i < patternLength; ++i)
                    t[i] = work[(size_t) i] * norm;
                t[-1] = t[patternLength - 1];
                t[patternLength] = t[0];
                t[patternLength + 1] = t[1];
            }
        }
    };

    const Pattern& patternFor (WatermarkKind kind)
    {
        static const Pattern ai (0x564F584141490001ull);
        static const Pattern singing (0x564F584153490002ull);
        return kind == WatermarkKind::SynthesizedSinging ? singing : ai;
    }

    inline float cubic (const float* t, int i, float f) noexcept
    {
        const float y0 = t[i - 1], y1 = t[i], y2 = t[i + 1], y3 = t[i + 2];
        const float c1 = 0.5f * (y2 - y0);
        const float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
        const float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);
        return ((c3 * f + c2) * f + c1) * f + y1;
    }
}

//==============================================================================
void WatermarkEmbedder::prepare (double sampleRate)
{
    sr = sampleRate > 0.0 ? sampleRate : 48000.0;
    increment = referenceRate / sr;

    patternFor (WatermarkKind::AiVoice);              // build the tables now, not on the audio thread
    patternFor (WatermarkKind::SynthesizedSinging);

    for (auto& c : channels)
    {
        for (int k = 0; k < numBands; ++k)
        {
            const double lo = bandEdge (k), hi = bandEdge (k + 1);
            const double centre = std::sqrt (lo * hi);
            const auto coeffs = dsp::BiquadCoeffs::bandPass (sr, juce::jmin (centre, 0.45 * sr), 0.65 * centre / (hi - lo));   // two in cascade ~ the band width
            c.band[(size_t) (2 * k)].c = coeffs;
            c.band[(size_t) (2 * k + 1)].c = coeffs;
        }
    }

    msCoeff = (float) std::exp (-1.0 / (0.010 * sr));   // 10 ms level follower
    gainStep = (float) (1.0 / (0.05 * sr));              // 50 ms fades
    reset();
}

void WatermarkEmbedder::reset() noexcept
{
    position = 0.0;
    gain = 0.0f;
    for (auto& c : channels)
    {
        for (auto& b : c.band)
            b.reset();
        c.meanSquare.fill (0.0f);
    }
}

float WatermarkEmbedder::nextMark (Channel& c, float input, const float* const* tables, int index, float frac) noexcept
{
    float mark = 0.0f;
    for (int k = 0; k < numBands; ++k)
    {
        const float v = c.band[(size_t) (2 * k + 1)].process (c.band[(size_t) (2 * k)].process (input));
        auto& ms = c.meanSquare[(size_t) k];
        ms = v * v + msCoeff * (ms - v * v);
        mark += std::sqrt (ms) * cubic (tables[k], index, frac);
    }
    return mark;
}

void WatermarkEmbedder::process (float* left, float* right, int numSamples, WatermarkKind kind) noexcept
{
    if (kind != WatermarkKind::None)
        current = kind;

    const auto& pattern = patternFor (current);
    const float* tables[numBands];
    for (int k = 0; k < numBands; ++k)
        tables[k] = pattern.band (k);

    const float target = kind != WatermarkKind::None ? 1.0f : 0.0f;

    for (int i = 0; i < numSamples; ++i)
    {
        gain = gain < target ? juce::jmin (target, gain + gainStep) : juce::jmax (target, gain - gainStep);

        const int index = (int) position;
        const float frac = (float) (position - index);
        const float g = gain * strength;

        const float markL = nextMark (channels[0], left[i], tables, index, frac);
        left[i] += g * markL;
        if (right != nullptr)
            right[i] += g * nextMark (channels[1], right[i], tables, index, frac);

        position += increment;
        if (position >= (double) patternLength)
            position -= (double) patternLength;
    }

    for (auto& c : channels)
        for (auto& b : c.band)
            b.sanitise();
}

//==============================================================================
WatermarkDetection detectWatermark (const juce::AudioBuffer<float>& audio, double sampleRate)
{
    WatermarkDetection result;
    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0 || sampleRate <= 0.0)
        return result;

    const auto mono = resampleBuffer (toMono (audio), sampleRate, referenceRate);
    const int total = mono.getNumSamples();
    const float* x = mono.getReadPointer (0);

    juce::dsp::FFT fft (patternOrder);
    const int bins = patternLength / 2 + 1;
    const double binHz = referenceRate / patternLength;
    const int lo = (int) std::ceil (lowestHz / binHz), hi = (int) std::ceil (highestHz / binHz);
    std::vector<float> work ((size_t) patternLength * 2);

    // Reference spectra (all bands summed), phase only.
    using Complex = std::complex<float>;
    std::vector<Complex> reference[2];
    const WatermarkKind kinds[2] = { WatermarkKind::AiVoice, WatermarkKind::SynthesizedSinging };
    for (int r = 0; r < 2; ++r)
    {
        const auto& pattern = patternFor (kinds[r]);
        std::fill (work.begin(), work.end(), 0.0f);
        for (int k = 0; k < numBands; ++k)
            juce::FloatVectorOperations::add (work.data(), pattern.band (k), patternLength);
        fft.performRealOnlyForwardTransform (work.data(), true);
        reference[r].assign ((size_t) bins, Complex());
        for (int b = lo; b < hi; ++b)
        {
            const Complex v (work[(size_t) (2 * b)], work[(size_t) (2 * b + 1)]);
            const float m = std::abs (v);
            reference[r][(size_t) b] = m > 1.0e-20f ? std::conj (v) / m : Complex();
        }
    }

    // Phase-transform cross-spectra summed over pattern-length blocks (the pattern repeats, so the
    // correlation peak is at the same lag in every block).
    std::vector<Complex> acc[2] = { std::vector<Complex> ((size_t) bins), std::vector<Complex> ((size_t) bins) };
    for (int start = 0; start < total; start += patternLength)
    {
        const int n = juce::jmin (patternLength, total - start);
        if (n < patternLength / 4 && start > 0)
            break;
        std::fill (work.begin(), work.end(), 0.0f);
        std::copy (x + start, x + start + n, work.begin());
        fft.performRealOnlyForwardTransform (work.data(), true);
        for (int b = lo; b < hi; ++b)
        {
            const Complex v (work[(size_t) (2 * b)], work[(size_t) (2 * b + 1)]);
            const float m = std::abs (v);
            if (m < 1.0e-12f)
                continue;
            const Complex w = v / m;
            acc[0][(size_t) b] += w * reference[0][(size_t) b];
            acc[1][(size_t) b] += w * reference[1][(size_t) b];
        }
    }

    float scores[2] {};
    int peaks[2] {};
    std::vector<float> sorted ((size_t) patternLength);
    for (int r = 0; r < 2; ++r)
    {
        std::fill (work.begin(), work.end(), 0.0f);
        for (int b = 0; b < bins; ++b)
        {
            work[(size_t) (2 * b)] = acc[r][(size_t) b].real();
            work[(size_t) (2 * b + 1)] = acc[r][(size_t) b].imag();
        }
        for (int b = bins; b < patternLength; ++b)
        {
            work[(size_t) (2 * b)] = acc[r][(size_t) (patternLength - b)].real();
            work[(size_t) (2 * b + 1)] = -acc[r][(size_t) (patternLength - b)].imag();
        }
        fft.performRealOnlyInverseTransform (work.data());

        // Robust z-score of the correlation peak (median / MAD of all lags).
        std::copy (work.begin(), work.begin() + patternLength, sorted.begin());
        const auto mid = sorted.begin() + patternLength / 2;
        std::nth_element (sorted.begin(), mid, sorted.end());
        const float median = *mid;
        for (auto& v : sorted)
            v = std::abs (v - median);
        std::nth_element (sorted.begin(), mid, sorted.end());
        const float mad = *mid * 1.4826f;

        const auto peak = std::max_element (work.begin(), work.begin() + patternLength);
        peaks[r] = (int) (peak - work.begin());
        scores[r] = mad > 1.0e-20f ? (*peak - median) / mad : 0.0f;
    }

    result.aiVoiceScore = scores[0];
    result.synthesizedSingingScore = scores[1];
    const int best = scores[1] > scores[0] ? 1 : 0;
    if (scores[best] >= WatermarkDetection::threshold)
    {
        result.kind = kinds[best];
        result.offsetSeconds = peaks[best] / referenceRate;
    }
    return result;
}

juce::String watermarkKindName (WatermarkKind kind)
{
    switch (kind)
    {
        case WatermarkKind::AiVoice:            return "AI voice";
        case WatermarkKind::SynthesizedSinging: return "synthesized singing";
        case WatermarkKind::None:
        default:                                return "none";
    }
}

} // namespace voxa
