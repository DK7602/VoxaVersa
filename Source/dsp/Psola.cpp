#include "Psola.h"

namespace voxa::dsp
{

static int nextPowerOfTwo (int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

//==============================================================================
void PsolaAnalyzer::prepare (double sampleRate, float minHz, float maxHz)
{
    sr = sampleRate;
    maxPeriod = (float) (sr / minHz);
    minPeriod = (float) (sr / maxHz);
    unvoicedHop = (float) std::round (sr * 0.0025);   // 2.5 ms hops for noise / transients

    const int size = nextPowerOfTwo ((int) (sr * 2.5));
    ring.assign ((size_t) size, 0.0f);
    lpRing.assign ((size_t) size, 0.0f);
    mask = size - 1;

    marks.assign (8192, {});
    reset();
}

void PsolaAnalyzer::reset()
{
    std::fill (ring.begin(), ring.end(), 0.0f);
    std::fill (lpRing.begin(), lpRing.end(), 0.0f);
    writePos = 0;
    markHead = markCount = 0;
    hasLastMark = false;
    lastMark = {};
    peakLp1.reset();
    peakLp2.reset();
    peakFilterPeriod = 0.0f;
    updatePeakFilter (maxPeriod * 0.5f);
}

void PsolaAnalyzer::updatePeakFilter (float periodSamples)
{
    if (periodSamples <= 0.0f)
        return;
    if (peakFilterPeriod > 0.0f && std::abs (periodSamples / peakFilterPeriod - 1.0f) < 0.15f)
        return;

    peakFilterPeriod = periodSamples;
    const double f0 = sr / periodSamples;
    const double fc = juce::jlimit (250.0, 1600.0, f0 * 2.2);
    peakLp1.c = BiquadCoeffs::lowPass (sr, fc, 0.7071);
    peakLp2.c = peakLp1.c;
}

void PsolaAnalyzer::push (const float* input, int numSamples, const PitchTimeline& timeline, int placementLag)
{
    if (! timeline.isEmpty() && timeline.newest().voiced)
        updatePeakFilter (timeline.newest().periodSamples);

    for (int i = 0; i < numSamples; ++i)
    {
        const float x = input[i];
        ring[(size_t) (writePos & mask)] = x;
        lpRing[(size_t) (writePos & mask)] = peakLp2.process (peakLp1.process (x));
        ++writePos;
    }

    peakLp1.sanitise();
    peakLp2.sanitise();
    placeMarks (timeline, placementLag);
}

void PsolaAnalyzer::addMark (const Mark& m) noexcept
{
    const int capacity = (int) marks.size();
    if (markCount > 0)
    {
        auto& prev = marks[(size_t) ((markHead + markCount - 1) % capacity)];
        const auto distance = (float) (m.pos - prev.pos);
        if (prev.voiced && m.voiced && distance > 0.6f * prev.period && distance < 1.5f * prev.period)
            prev.spacing = distance;
    }

    marks[(size_t) ((markHead + markCount) % capacity)] = m;
    if (markCount < capacity)
        ++markCount;
    else
        markHead = (markHead + 1) % capacity;
    lastMark = m;
    hasLastMark = true;
}

void PsolaAnalyzer::placeMarks (const PitchTimeline& timeline, int placementLag)
{
    const juce::int64 limit = writePos - juce::jmax (0, placementLag) - 2;
    // Oldest position still held in the ring (with room for a grain).
    const juce::int64 oldest = writePos - (juce::int64) ring.size() + (juce::int64) (maxPeriod * 3.0f);

    if (! hasLastMark || lastMark.pos < oldest)
    {
        lastMark = { juce::jmax ((juce::int64) 0, limit - (juce::int64) unvoicedHop), unvoicedHop, 0.0f, false };
        hasLastMark = true;
    }

    int guard = 0;
    while (++guard < 8192)
    {
        PitchFrame frame;
        const double probe = (double) lastMark.pos + (double) (lastMark.voiced ? lastMark.period : unvoicedHop);
        const bool haveFrame = timeline.lookup (probe, frame);
        const bool voiced = haveFrame && frame.voiced && frame.periodSamples >= minPeriod * 0.9f
                            && frame.periodSamples <= maxPeriod * 1.1f;

        if (voiced)
        {
            const float T = frame.periodSamples;
            const double predicted = (double) lastMark.pos + T;
            const float radius = (lastMark.voiced ? 0.22f : 0.5f) * T;
            const auto lo = (juce::int64) std::ceil (predicted - radius);
            const auto hi = (juce::int64) std::floor (predicted + radius);

            if (hi >= limit)
                break;

            juce::int64 best = (juce::int64) std::llround (predicted);
            float bestValue = -1.0e30f;
            for (auto k = juce::jmax (lo, lastMark.pos + (juce::int64) (0.5f * T)); k <= hi; ++k)
            {
                // Mild bias towards the predicted position keeps marks stable on flat peaks.
                const float distance = (float) std::abs ((double) k - predicted) / juce::jmax (1.0f, radius);
                const float v = lpRing[(size_t) (k & mask)] - 0.05f * distance * std::abs (lpRing[(size_t) (k & mask)]);
                if (v > bestValue)
                {
                    bestValue = v;
                    best = k;
                }
            }

            addMark ({ best, T, 0.0f, true });
        }
        else
        {
            const auto next = lastMark.pos + (juce::int64) unvoicedHop;
            if (next >= limit)
                break;
            addMark ({ next, unvoicedHop, 0.0f, false });
        }
    }

    // Drop marks that are too old to be useful (keep ~0.5 s for delayed voices).
    const juce::int64 keepFrom = writePos - (juce::int64) (sr * 0.6) - (juce::int64) maxPeriod * 4;
    const int capacity = (int) marks.size();
    while (markCount > 2 && marks[(size_t) markHead].pos < keepFrom)
    {
        markHead = (markHead + 1) % capacity;
        --markCount;
    }
}

bool PsolaAnalyzer::findNearestMark (double pos, Mark& result) const noexcept
{
    if (markCount == 0)
        return false;

    const int capacity = (int) marks.size();
    auto at = [&] (int i) -> const Mark& { return marks[(size_t) ((markHead + i) % capacity)]; };

    int lo = 0, hi = markCount - 1;
    if (pos <= (double) at (lo).pos) { result = at (lo); return true; }
    if (pos >= (double) at (hi).pos) { result = at (hi); return true; }

    while (hi - lo > 1)
    {
        const int mid = (lo + hi) / 2;
        if ((double) at (mid).pos <= pos) lo = mid; else hi = mid;
    }

    result = (pos - (double) at (lo).pos) <= ((double) at (hi).pos - pos) ? at (lo) : at (hi);
    return true;
}

//==============================================================================
void PsolaVoice::prepare (int maxLatencySamples)
{
    const int size = nextPowerOfTwo (juce::jmax (8192, maxLatencySamples * 4 + 8192));
    acc.assign ((size_t) size, 0.0f);
    mask = size - 1;
    reset (0.0);
}

void PsolaVoice::reset (double startPosition) noexcept
{
    std::fill (acc.begin(), acc.end(), 0.0f);
    nextPos = startPosition;
    lastVoiced = false;
}

void PsolaVoice::addGrain (const PsolaAnalyzer& analyzer, double centre, double markPos, float half,
                           float formantRatio, float gain) noexcept
{
    const auto k0 = (juce::int64) std::ceil (centre - half);
    const auto k1 = (juce::int64) std::floor (centre + half);
    const double invHalf = juce::MathConstants<double>::pi / (double) half;

    for (auto k = k0; k <= k1; ++k)
    {
        const double offset = (double) k - centre;
        const float w = 0.5f + 0.5f * (float) std::cos (offset * invHalf);
        acc[(size_t) (k & mask)] += gain * w * analyzer.interpolate (markPos + offset * formantRatio);
    }
}

void PsolaVoice::read (juce::int64 start, float* output, int numSamples) noexcept
{
    for (int i = 0; i < numSamples; ++i)
    {
        auto& v = acc[(size_t) ((start + i) & mask)];
        output[i] = v;
        v = 0.0f;
    }
}

} // namespace voxa::dsp
