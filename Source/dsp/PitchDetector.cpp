#include "PitchDetector.h"
#include <algorithm>

namespace voxa::dsp
{

void PitchDetector::prepare (double sampleRate, float minHz, float maxHz)
{
    sr = sampleRate;
    minFreq = minHz;
    maxFreq = maxHz;

    decimation = juce::jmax (1, (int) std::floor (sr / 16000.0));
    decimatedRate = sr / decimation;

    maxLagD = (int) std::ceil (decimatedRate / minFreq) + 2;
    minLagD = juce::jmax (2, (int) std::floor (decimatedRate / maxFreq));
    windowD = maxLagD;
    hopD = juce::jmax (16, (int) std::round (decimatedRate * 0.004));
    hopFull = hopD * decimation;

    dSize = windowD + maxLagD + 4;
    dBuf.assign ((size_t) dSize, 0.0f);
    frameBuf.assign ((size_t) (windowD + maxLagD), 0.0f);
    diff.assign ((size_t) maxLagD + 2, 0.0f);
    cmnd.assign ((size_t) maxLagD + 2, 1.0f);

    windowFull = juce::jmin (4096, windowD * decimation);
    const int needed = windowFull + (maxLagD + 4) * decimation + 16;
    int size = 1;
    while (size < needed)
        size <<= 1;
    fBuf.assign ((size_t) size, 0.0f);
    fMask = size - 1;

    analysisDelay = (windowD + maxLagD) * decimation / 2;

    const double aaCutoff = juce::jmin (4000.0, 0.4 * decimatedRate);
    aa1.c = BiquadCoeffs::lowPass (sr, aaCutoff, 0.5412);
    aa2.c = BiquadCoeffs::lowPass (sr, aaCutoff, 1.3066);

    reset();
}

void PitchDetector::reset()
{
    std::fill (dBuf.begin(), dBuf.end(), 0.0f);
    std::fill (fBuf.begin(), fBuf.end(), 0.0f);
    dWrite = fWrite = 0;
    decimCounter = hopCounter = 0;
    aa1.reset();
    aa2.reset();
    latest = {};
    lastPeriods[0] = lastPeriods[1] = lastPeriods[2] = 0.0f;
    voicedRun = unvoicedRun = 0;
}

bool PitchDetector::pushSample (float x) noexcept
{
    fBuf[(size_t) (fWrite & fMask)] = x;
    ++fWrite;

    const float y = aa2.process (aa1.process (x));

    if (++decimCounter >= decimation)
    {
        decimCounter = 0;
        dBuf[(size_t) dWrite] = y;
        dWrite = (dWrite + 1) % dSize;

        if (++hopCounter >= hopD)
        {
            hopCounter = 0;
            analyseFrame();
            return true;
        }
    }
    return false;
}

float PitchDetector::refineAtFullRate (float tauFull) const noexcept
{
    const int lo = juce::jmax (2, (int) std::floor (tauFull) - decimation - 1);
    const int hi = (int) std::ceil (tauFull) + decimation + 1;
    const int span = windowFull + hi + 2;
    const int start = fWrite - span;

    auto sample = [this, start] (int i) { return fBuf[(size_t) ((start + i) & fMask)]; };

    float best = 1.0e30f;
    int bestTau = (int) std::round (tauFull);
    float dPrev = 0.0f, dBest = 0.0f, dNext = 0.0f;
    float values[64];
    const int count = juce::jmin (64, hi - lo + 1);

    for (int k = 0; k < count; ++k)
    {
        const int tau = lo + k;
        float d = 0.0f;
        for (int j = 0; j < windowFull; ++j)
        {
            const float e = sample (j) - sample (j + tau);
            d += e * e;
        }
        values[k] = d;
        if (d < best)
        {
            best = d;
            bestTau = tau;
        }
    }

    const int bk = bestTau - lo;
    if (bk <= 0 || bk >= count - 1)
        return tauFull;

    dPrev = values[bk - 1];
    dBest = values[bk];
    dNext = values[bk + 1];
    const float denom = dPrev - 2.0f * dBest + dNext;
    const float delta = denom > 1.0e-12f ? 0.5f * (dPrev - dNext) / denom : 0.0f;
    return (float) bestTau + juce::jlimit (-0.5f, 0.5f, delta);
}

void PitchDetector::analyseFrame() noexcept
{
    const int N = windowD + maxLagD;
    const int start = (dWrite - N + dSize) % dSize;
    for (int i = 0; i < N; ++i)
        frameBuf[(size_t) i] = dBuf[(size_t) ((start + i) % dSize)];

    const float* x = frameBuf.data();

    // Level
    double energy = 0.0;
    for (int i = 0; i < N; ++i)
        energy += (double) x[i] * x[i];
    const float rms = (float) std::sqrt (energy / N);
    latest.levelDb = juce::Decibels::gainToDecibels (rms, -100.0f);

    bool rawVoiced = false;
    float tauD = 0.0f, aperiodicity = 1.0f;

    if (latest.levelDb > gateDb)
    {
        // Difference function via energies + autocorrelation.
        double e0 = 0.0;
        for (int j = 0; j < windowD; ++j)
            e0 += (double) x[j] * x[j];
        double eTau = e0;

        diff[0] = 0.0f;
        for (int tau = 1; tau < maxLagD; ++tau)
        {
            eTau += (double) x[windowD + tau - 1] * x[windowD + tau - 1] - (double) x[tau - 1] * x[tau - 1];
            float r = 0.0f;
            const float* a = x;
            const float* b = x + tau;
            for (int j = 0; j < windowD; ++j)
                r += a[j] * b[j];
            diff[(size_t) tau] = (float) juce::jmax (0.0, e0 + eTau - 2.0 * r);
        }

        // Cumulative mean normalised difference.
        cmnd[0] = 1.0f;
        double running = 0.0;
        for (int tau = 1; tau < maxLagD; ++tau)
        {
            running += diff[(size_t) tau];
            cmnd[(size_t) tau] = running > 0.0 ? (float) (diff[(size_t) tau] * tau / running) : 1.0f;
        }

        constexpr float threshold = 0.15f;
        int tau = -1;
        for (int t = minLagD; t < maxLagD - 1; ++t)
        {
            if (cmnd[(size_t) t] < threshold)
            {
                while (t + 1 < maxLagD - 1 && cmnd[(size_t) t + 1] < cmnd[(size_t) t])
                    ++t;
                tau = t;
                break;
            }
        }

        if (tau < 0)
        {
            float best = 1.0e9f;
            for (int t = minLagD; t < maxLagD - 1; ++t)
                if (cmnd[(size_t) t] < best)
                {
                    best = cmnd[(size_t) t];
                    tau = t;
                }
        }

        if (tau > 0 && tau < maxLagD - 1)
        {
            aperiodicity = cmnd[(size_t) tau];
            const float a = cmnd[(size_t) tau - 1], b = cmnd[(size_t) tau], c = cmnd[(size_t) tau + 1];
            const float denom = a - 2.0f * b + c;
            const float delta = std::abs (denom) > 1.0e-9f ? 0.5f * (a - c) / denom : 0.0f;
            tauD = (float) tau + juce::jlimit (-0.5f, 0.5f, delta);
            rawVoiced = aperiodicity < 0.28f;
        }
    }

    // Voicing hysteresis.
    if (rawVoiced)
    {
        ++voicedRun;
        unvoicedRun = 0;
    }
    else
    {
        ++unvoicedRun;
        voicedRun = 0;
    }

    const bool wasVoiced = latest.voiced;
    const bool nowVoiced = wasVoiced ? unvoicedRun < 2 : voicedRun >= 2;

    if (nowVoiced && rawVoiced)
    {
        float period = refineAtFullRate (tauD * (float) decimation);

        // Median-of-three in the log domain against octave blips.
        lastPeriods[2] = lastPeriods[1];
        lastPeriods[1] = lastPeriods[0];
        lastPeriods[0] = period;
        if (lastPeriods[1] > 0.0f && lastPeriods[2] > 0.0f)
        {
            float sorted[3] = { lastPeriods[0], lastPeriods[1], lastPeriods[2] };
            std::sort (sorted, sorted + 3);
            const float median = sorted[1];
            if (std::abs (std::log2 (period / median)) > 0.4f)
                period = median;
        }

        latest.voiced = true;
        latest.periodSamples = period;
        latest.frequencyHz = (float) (sr / period);
        latest.confidence = juce::jlimit (0.0f, 1.0f, 1.0f - aperiodicity);
    }
    else if (nowVoiced)
    {
        // Hold the previous estimate through a single dropout.
        latest.voiced = latest.periodSamples > 0.0f;
        latest.confidence *= 0.5f;
    }
    else
    {
        latest.voiced = false;
        latest.frequencyHz = 0.0f;
        latest.periodSamples = 0.0f;
        latest.confidence = juce::jlimit (0.0f, 1.0f, 1.0f - aperiodicity);
        lastPeriods[0] = lastPeriods[1] = lastPeriods[2] = 0.0f;
    }
}

std::vector<PitchDetector::Frame> PitchDetector::analyse (const float* input, int numSamples, double sampleRate,
                                                         int hopSamples, float minHz, float maxHz)
{
    PitchDetector d;
    d.prepare (sampleRate, minHz, maxHz);

    struct Timed { int time; Frame frame; };
    std::vector<Timed> timed;
    timed.reserve ((size_t) (numSamples / d.getHopSize() + 8));

    const int delay = d.getAnalysisDelay();
    std::vector<float> padded ((size_t) (numSamples + delay * 2), 0.0f);
    std::copy (input, input + numSamples, padded.begin());

    d.process (padded.data(), (int) padded.size(), [&] (int index, const Frame& f)
    {
        timed.push_back ({ index - delay, f });
    });

    std::vector<Frame> result;
    if (hopSamples <= 0)
        return result;

    const int numFrames = numSamples / hopSamples + 1;
    result.resize ((size_t) numFrames);
    size_t cursor = 0;
    for (int k = 0; k < numFrames; ++k)
    {
        const int t = k * hopSamples;
        while (cursor + 1 < timed.size() && std::abs (timed[cursor + 1].time - t) <= std::abs (timed[cursor].time - t))
            ++cursor;
        if (! timed.empty())
            result[(size_t) k] = timed[cursor].frame;
    }
    return result;
}

} // namespace voxa::dsp
