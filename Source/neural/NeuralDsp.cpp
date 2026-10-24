#include "NeuralDsp.h"

#include <cmath>
#include <limits>

namespace voxa::neural::dsp
{

namespace
{
    int64_t floorDiv (int64_t a, int64_t b) noexcept
    {
        auto q = a / b;

        if ((a % b != 0) && ((a < 0) != (b < 0)))
            --q;

        return q;
    }

    double besselI0 (double x) noexcept
    {
        double sum = 1.0, term = 1.0;
        const double halfX = 0.5 * x;

        for (int k = 1; k < 64; ++k)
        {
            term *= (halfX / k) * (halfX / k);
            sum += term;

            if (term < sum * 1.0e-12)
                break;
        }

        return sum;
    }

    uint64_t splitMix (uint64_t x) noexcept
    {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    float dot (const float* a, const float* b, int n) noexcept
    {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int i = 0;

        for (; i + 4 <= n; i += 4)
        {
            s0 += a[i] * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
        }

        for (; i < n; ++i)
            s0 += a[i] * b[i];

        return (s0 + s1) + (s2 + s3);
    }

    double energy (const float* a, int n) noexcept
    {
        double e = 0.0;
        for (int i = 0; i < n; ++i)
            e += (double) a[i] * (double) a[i];
        return e;
    }

    bool isVoiced (float f0) noexcept
    {
        return f0 > 0.0f && std::isfinite (f0);
    }
} // namespace

//==============================================================================
int coarsePitch (float f0Hz) noexcept
{
    if (! isVoiced (f0Hz))
        return 1;

    static const float melMin = 1127.0f * std::log (1.0f + f0MinHz / 700.0f);
    static const float melMax = 1127.0f * std::log (1.0f + f0MaxHz / 700.0f);

    const float mel = 1127.0f * std::log (1.0f + f0Hz / 700.0f);
    const float coarse = (mel - melMin) * 254.0f / (melMax - melMin) + 1.0f;

    if (coarse <= 1.0f)
        return 1;

    return (int) juce::jlimit (1L, 255L, std::lrint (coarse));
}

//==============================================================================
void Resampler::prepare (int inputRate, int outputRate)
{
    jassert (inputRate > 0 && outputRate > 0);
    inRate = juce::jmax (1, inputRate);
    outRate = juce::jmax (1, outputRate);

    if (inRate == outRate)
    {
        halfWidth = 0;
        numTaps = 0;
        table.clear();
        return;
    }

    constexpr double zeroCrossings = 16.0;
    constexpr double kaiserBeta = 8.0;

    const double cutoff = 0.46 * (double) juce::jmin (inRate, outRate) / (double) inRate;   // cycles per input sample
    halfWidth = (int) std::ceil (zeroCrossings / (2.0 * cutoff));
    numTaps = 2 * halfWidth;
    table.assign ((size_t) (numPhases + 1) * (size_t) numTaps, 0.0f);

    const double i0Beta = besselI0 (kaiserBeta);

    for (int p = 0; p <= numPhases; ++p)
    {
        auto* row = table.data() + (size_t) p * (size_t) numTaps;
        const double frac = (double) p / numPhases;
        double sum = 0.0;

        for (int j = 0; j < numTaps; ++j)
        {
            // distance (in input samples) between the output instant and tap j
            const double d = frac + (double) (halfWidth - 1 - j);
            const double x = d / (double) halfWidth;
            double h = 0.0;

            if (std::abs (x) < 1.0)
            {
                const double arg = 2.0 * cutoff * d;
                const double sinc = std::abs (arg) < 1.0e-12 ? 1.0 : std::sin (juce::MathConstants<double>::pi * arg)
                                                                     / (juce::MathConstants<double>::pi * arg);
                h = 2.0 * cutoff * sinc * besselI0 (kaiserBeta * std::sqrt (1.0 - x * x)) / i0Beta;
            }

            row[j] = (float) h;
            sum += h;
        }

        if (sum > 0.0)
            for (int j = 0; j < numTaps; ++j)
                row[j] = (float) (row[j] / sum);   // exact unity gain at DC for every phase
    }
}

int64_t Resampler::firstInputFor (int64_t outIndex) const noexcept
{
    if (inRate == outRate)
        return outIndex;

    return floorDiv (outIndex * inRate, outRate) - (halfWidth - 1);
}

int64_t Resampler::lastInputFor (int64_t outIndex) const noexcept
{
    if (inRate == outRate)
        return outIndex;

    return floorDiv (outIndex * inRate, outRate) + halfWidth;
}

void Resampler::process (const float* input, int64_t inStart, int numIn,
                         float* output, int64_t outStart, int numOut) const noexcept
{
    jassert (isPrepared());

    if (inRate == outRate)
    {
        for (int i = 0; i < numOut; ++i)
        {
            const auto index = outStart + i - inStart;
            output[i] = index >= 0 && index < numIn ? input[index] : 0.0f;
        }

        return;
    }

    for (int i = 0; i < numOut; ++i)
    {
        const int64_t numerator = (outStart + i) * (int64_t) inRate;
        const int64_t pos = floorDiv (numerator, outRate);
        const int64_t remainder = numerator - pos * outRate;

        const double phase = (double) remainder * numPhases / (double) outRate;
        const int p = juce::jmin ((int) phase, numPhases - 1);
        const float t = (float) (phase - p);

        const float* rowA = table.data() + (size_t) p * (size_t) numTaps;
        const float* rowB = rowA + numTaps;

        const int64_t base = pos - (halfWidth - 1) - inStart;   // input index of tap 0
        const int j0 = (int) juce::jlimit ((int64_t) 0, (int64_t) numTaps, -base);
        const int j1 = (int) juce::jlimit ((int64_t) 0, (int64_t) numTaps, (int64_t) numIn - base);

        float a = 0.0f, b = 0.0f;

        for (int j = j0; j < j1; ++j)
        {
            const float x = input[base + j];
            a += rowA[j] * x;
            b += rowB[j] * x;
        }

        output[i] = a + t * (b - a);
    }
}

std::vector<float> Resampler::processAll (const float* input, int numIn) const
{
    const auto numOut = (int) (((int64_t) numIn * outRate + inRate - 1) / inRate);
    std::vector<float> result ((size_t) juce::jmax (0, numOut));

    if (numOut > 0)
        process (input, 0, numIn, result.data(), 0, numOut);

    return result;
}

//==============================================================================
void resampleF0 (const std::vector<float>& f0Hz, double frameRate, float* out, int numFrames)
{
    const int n = (int) f0Hz.size();

    auto valueAt = [&] (int index)
    {
        const float v = f0Hz[(size_t) juce::jlimit (0, n - 1, index)];
        return isVoiced (v) ? v : 0.0f;
    };

    for (int i = 0; i < numFrames; ++i)
    {
        if (n == 0 || ! (frameRate > 0.0))
        {
            out[i] = 0.0f;
            continue;
        }

        const double s = (i + 0.5) * frameRate / framesPerSecond - 0.5;

        if (s > n - 0.5)
        {
            out[i] = 0.0f;
            continue;
        }

        const double fl = std::floor (s);
        const int k = (int) fl;
        const auto t = (float) (s - fl);
        const float a = valueAt (k), b = valueAt (k + 1);

        if (a > 0.0f && b > 0.0f)
            out[i] = std::exp (std::log (a) + t * (std::log (b) - std::log (a)));
        else
            out[i] = t < 0.5f ? a : b;
    }
}

float frameF0 (const float* perSampleF0, int numSamples) noexcept
{
    int voiced = 0;
    double sum = 0.0;

    for (int i = 0; i < numSamples; ++i)
    {
        if (isVoiced (perSampleF0[i]))
        {
            ++voiced;
            sum += perSampleF0[i];
        }
    }

    return (voiced > 0 && voiced * 2 >= numSamples) ? (float) (sum / voiced) : 0.0f;
}

float gaussianNoise (int64_t frame, int channel, uint64_t seed) noexcept
{
    const auto key = ((uint64_t) frame << 12) ^ (uint64_t) (channel & 0xfff);
    const auto h1 = splitMix (key ^ splitMix (seed));
    const auto h2 = splitMix (h1);

    constexpr double scale = 1.0 / 9007199254740992.0;   // 2^-53
    const double u1 = ((double) (h1 >> 11) + 0.5) * scale;
    const double u2 = (double) (h2 >> 11) * scale;

    return (float) (std::sqrt (-2.0 * std::log (u1)) * std::cos (juce::MathConstants<double>::twoPi * u2));
}

//==============================================================================
int findBestOffset (const float* reference, const float* candidate, int length, int searchLength,
                    float& correlationOut) noexcept
{
    correlationOut = 0.0f;

    if (length <= 0 || searchLength <= 0)
        return 0;

    const double refEnergy = energy (reference, length);

    if (refEnergy < 1.0e-10)
        return searchLength / 2;

    double windowEnergy = energy (candidate, length);
    int best = searchLength / 2;
    double bestScore = -std::numeric_limits<double>::max(), bestDot = 0.0, bestEnergy = 0.0;

    for (int offset = 0; offset <= searchLength; ++offset)
    {
        if (offset > 0)
        {
            const double leaving = candidate[offset - 1], entering = candidate[offset + length - 1];
            windowEnergy = juce::jmax (0.0, windowEnergy + entering * entering - leaving * leaving);
        }

        const double d = dot (reference, candidate + offset, length);
        const double score = d / std::sqrt (windowEnergy + 1.0e-9);

        if (score > bestScore)
        {
            bestScore = score;
            best = offset;
            bestDot = d;
            bestEnergy = windowEnergy;
        }
    }

    if (bestEnergy > 1.0e-10)
        correlationOut = (float) juce::jlimit (-1.0, 1.0, bestDot / std::sqrt (refEnergy * bestEnergy));

    return best;
}

float correlation (const float* a, const float* b, int length) noexcept
{
    const double ea = energy (a, length), eb = energy (b, length);

    if (ea < 1.0e-10 || eb < 1.0e-10)
        return 0.0f;

    return (float) juce::jlimit (-1.0, 1.0, (double) dot (a, b, length) / std::sqrt (ea * eb));
}

void crossfade (const float* fadingOut, const float* fadingIn, float* dest, int length, float r) noexcept
{
    r = juce::jlimit (0.0f, 1.0f, r);

    for (int i = 0; i < length; ++i)
    {
        const float s = std::sin (juce::MathConstants<float>::halfPi * ((float) i + 0.5f) / (float) length);
        const float wIn = s * s;
        const float wOut = 1.0f - wIn;
        const float norm = 1.0f / std::sqrt (wIn * wIn + wOut * wOut + 2.0f * r * wIn * wOut);
        dest[i] = (fadingOut[i] * wOut + fadingIn[i] * wIn) * norm;
    }
}

bool isSilent (const float* samples, int numSamples) noexcept
{
    if (numSamples <= 0)
        return true;

    float peak = 0.0f;
    double sum = 0.0;

    for (int i = 0; i < numSamples; ++i)
    {
        const float a = std::abs (samples[i]);
        peak = juce::jmax (peak, a);
        sum += (double) a * a;
    }

    const double rms = std::sqrt (sum / numSamples);
    return rms < 0.001 && peak < 0.01;
}

} // namespace voxa::neural::dsp
