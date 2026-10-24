#include "VoiceGenderEstimator.h"
#include "PitchDetector.h"
#include <algorithm>

namespace voxa::dsp
{

// Calibration (see class comment).
static constexpr float f0Centre = 255.0f, f0Scale = 0.22f;
static constexpr float formantCentre = 7000.0f, formantScale = 0.057f;
static constexpr float decisionOffset = 0.1f;   // centres the boundary between the two calibration groups

void VoiceGenderEstimator::prepare (double sampleRate)
{
    sr = sampleRate;
    decimation = juce::jmax (1, (int) std::lround (sr / 11025.0));
    fs = sr / decimation;

    const double cutoff = 0.42 * fs;
    aa1.c = BiquadCoeffs::lowPass (sr, cutoff, 0.5412);
    aa2.c = BiquadCoeffs::lowPass (sr, cutoff, 1.3066);

    frameLength = (int) std::lround (0.03 * fs);
    hop = (int) std::lround (0.01 * fs);
    frame.assign ((size_t) frameLength, 0.0f);
    work.assign ((size_t) frameLength, 0.0f);
    scratch.assign ((size_t) historySize, 0.0f);
    reset();
}

void VoiceGenderEstimator::reset()
{
    aa1.reset();
    aa2.reset();
    std::fill (frame.begin(), frame.end(), 0.0f);
    decimCounter = writeIndex = hopCounter = 0;
    historyCount = historyWrite = totalFrames = framesSinceDecision = 0;
    f0Count = f0Write = pitchDecimator = 0;
    peakRms = 0.0f;
    gender = Gender::Unknown;
    score = medianF0 = medianF34 = 0.0f;
}

void VoiceGenderEstimator::process (const float* input, int numSamples) noexcept
{
    for (int i = 0; i < numSamples; ++i)
    {
        const float y = aa2.process (aa1.process (input[i]));
        if (++decimCounter < decimation)
            continue;
        decimCounter = 0;

        frame[(size_t) writeIndex] = y;
        writeIndex = (writeIndex + 1) % frameLength;

        if (++hopCounter >= hop)
        {
            hopCounter = 0;
            analyseFrame();
        }
    }
    aa1.sanitise();
    aa2.sanitise();
}

bool VoiceGenderEstimator::lpc (const float* x, int n, std::array<double, 13>& a) const noexcept
{
    double r[lpcOrder + 1];
    for (int lag = 0; lag <= lpcOrder; ++lag)
    {
        double sum = 0.0;
        for (int i = 0; i + lag < n; ++i)
            sum += (double) x[i] * x[i + lag];
        r[lag] = sum;
    }
    if (r[0] <= 0.0)
        return false;

    a.fill (0.0);
    a[0] = 1.0;
    double e = r[0];
    double tmp[lpcOrder + 1];

    for (int i = 1; i <= lpcOrder; ++i)
    {
        double acc = r[i];
        for (int j = 1; j < i; ++j)
            acc += a[(size_t) j] * r[i - j];
        const double k = -acc / e;

        for (int j = 0; j <= lpcOrder; ++j)
            tmp[j] = a[(size_t) j];
        for (int j = 1; j < i; ++j)
            a[(size_t) j] = tmp[j] + k * tmp[i - j];
        a[(size_t) i] = k;

        e *= (1.0 - k * k);
        if (e <= 0.0)
            return false;
    }
    return true;
}

int VoiceGenderEstimator::findFormants (const std::array<double, 13>& a, float* formants) const noexcept
{
    // Roots of z^12 + a1 z^11 + ... + a12 (Durand-Kerner).
    using C = std::complex<double>;
    constexpr int n = lpcOrder;
    C roots[n];
    C seed (0.4, 0.9);
    C power (1.0, 0.0);
    for (int k = 0; k < n; ++k)
    {
        power *= seed;
        roots[k] = power;
    }

    auto evaluate = [&a] (C z)
    {
        C v (1.0, 0.0);
        for (int k = 1; k <= n; ++k)
            v = v * z + a[(size_t) k];
        return v;
    };

    for (int iteration = 0; iteration < 200; ++iteration)
    {
        double change = 0.0;
        for (int k = 0; k < n; ++k)
        {
            C denom (1.0, 0.0);
            for (int j = 0; j < n; ++j)
                if (j != k)
                    denom *= (roots[k] - roots[j]);
            if (std::abs (denom) < 1.0e-14)
                denom = C (1.0e-14, 0.0);
            const C delta = evaluate (roots[k]) / denom;
            roots[k] -= delta;
            change = std::max (change, std::abs (delta));
        }
        if (change < 1.0e-10)
            break;
    }

    float candidates[n];
    int count = 0;
    for (const auto& z : roots)
    {
        if (z.imag() <= 0.0)
            continue;
        const double magnitude = std::abs (z);
        if (magnitude <= 0.0 || magnitude >= 1.0)
            continue;
        const double freq = std::arg (z) * fs / juce::MathConstants<double>::twoPi;
        const double bandwidth = -std::log (magnitude) * fs / juce::MathConstants<double>::pi;
        if (bandwidth < 500.0 && freq > 150.0 && freq < 5000.0)
            candidates[count++] = (float) freq;
    }
    std::sort (candidates, candidates + count);
    for (int i = 0; i < juce::jmin (count, 4); ++i)
        formants[i] = candidates[i];
    return count;
}

void VoiceGenderEstimator::analyseFrame() noexcept
{
    const int n = frameLength;
    for (int i = 0; i < n; ++i)
        work[(size_t) i] = frame[(size_t) ((writeIndex + i) % n)];

    double energy = 0.0;
    for (int i = 0; i < n; ++i)
        energy += (double) work[(size_t) i] * work[(size_t) i];
    const float rms = (float) std::sqrt (energy / n);
    peakRms = juce::jmax (rms, peakRms * 0.9995f);
    if (rms < 0.001f || rms < 0.1f * peakRms)
        return;

    // Periodicity (normalised autocorrelation over 70-800 Hz).
    const int lo = juce::jmax (2, (int) (fs / 800.0));
    const int hi = juce::jmin (n - 1, (int) (fs / 70.0));
    double best = -1.0;
    int bestLag = lo;
    for (int lag = lo; lag < hi; ++lag)
    {
        double sum = 0.0;
        for (int i = 0; i + lag < n; ++i)
            sum += (double) work[(size_t) i] * work[(size_t) (i + lag)];
        if (sum > best)
        {
            best = sum;
            bestLag = lag;
        }
    }
    if (best / energy < 0.6)
        return;

    // Pre-emphasis + Hamming window, then LPC formants.
    float prev = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        const float x = work[(size_t) i];
        const float w = 0.54f - 0.46f * std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) (n - 1));
        work[(size_t) i] = (x - 0.97f * prev) * w;
        prev = x;
    }

    std::array<double, 13> a;
    if (! lpc (work.data(), n, a))
        return;

    float formants[4] {};
    if (findFormants (a, formants) < 4)
        return;

    juce::ignoreUnused (bestLag);
    f34History[(size_t) historyWrite] = formants[2] + formants[3];
    historyWrite = (historyWrite + 1) % historySize;
    historyCount = juce::jmin (historyCount + 1, historySize);
    ++totalFrames;

    // Decide quickly (1 s of voice), then re-check every half second.
    if (++framesSinceDecision >= (totalFrames < 300 ? 25 : 50) && totalFrames >= 100)
    {
        framesSinceDecision = 0;
        updateDecision();
    }
}

void VoiceGenderEstimator::addPitch (float f0Hz) noexcept
{
    if (f0Hz <= 0.0f)
        return;
    // Detector frames arrive every ~4 ms: keep one in three so the history spans ~50 s.
    if (++pitchDecimator < 3)
        return;
    pitchDecimator = 0;
    f0History[(size_t) f0Write] = f0Hz;
    f0Write = (f0Write + 1) % historySize;
    f0Count = juce::jmin (f0Count + 1, historySize);
}

void VoiceGenderEstimator::updateDecision() noexcept
{
    if (f0Count < 20 || historyCount < 20)
        return;

    auto median = [this] (const std::array<float, historySize>& values, int count)
    {
        std::copy (values.begin(), values.begin() + count, scratch.begin());
        auto mid = scratch.begin() + count / 2;
        std::nth_element (scratch.begin(), mid, scratch.begin() + count);
        return *mid;
    };

    medianF0 = median (f0History, f0Count);
    medianF34 = median (f34History, historyCount);
    score = std::log (medianF0 / f0Centre) / f0Scale + std::log (medianF34 / formantCentre) / formantScale - decisionOffset;

    // The more voice we've heard, the more evidence it takes to change our mind.
    const float hysteresis = totalFrames < 600 ? 0.15f : 0.6f;
    if (gender == Gender::Unknown)
        gender = score > 0.0f ? Gender::Female : Gender::Male;
    else if (gender == Gender::Male && score > hysteresis)
        gender = Gender::Female;
    else if (gender == Gender::Female && score < -hysteresis)
        gender = Gender::Male;
}

VoiceGenderEstimator::Gender VoiceGenderEstimator::analyse (const float* input, int numSamples, double sampleRate, float* scoreOut)
{
    VoiceGenderEstimator e;
    e.prepare (sampleRate);
    e.process (input, numSamples);
    for (const auto& f : PitchDetector::analyse (input, numSamples, sampleRate, (int) (sampleRate * 0.01)))
        if (f.voiced)
            e.addPitch (f.frequencyHz);
    e.updateDecision();
    if (scoreOut != nullptr)
        *scoreOut = e.score;
    return e.gender;
}

} // namespace voxa::dsp
