#pragma once

#include "Biquad.h"
#include <array>
#include <vector>
#include <cmath>

namespace voxa::dsp
{

/** Per-analysis-frame pitch information on the absolute input timeline. */
struct PitchFrame
{
    juce::int64 time = 0;          // absolute input sample the frame describes
    float periodSamples = 0.0f;    // 0 when unvoiced
    bool voiced = false;
    float srcMidi = 0.0f;          // detected pitch
    float mainShift = 0.0f;        // semitones applied to the lead voice
    float melodyMidi = 0.0f;       // corrected melody (no expression / offsets) - harmony reference
    float expression = 0.0f;       // vibrato + scoop + jitter (semitones)
    float globalOffset = 0.0f;     // register + pitch knob + fine tune (semitones)
};

/** Ring of recent PitchFrames with interpolated lookup by time. */
class PitchTimeline
{
public:
    static constexpr int capacity = 2048;

    void reset() noexcept { count = 0; head = 0; }

    void push (const PitchFrame& f) noexcept
    {
        frames[(size_t) ((head + count) % capacity)] = f;
        if (count < capacity)
            ++count;
        else
            head = (head + 1) % capacity;
    }

    bool isEmpty() const noexcept { return count == 0; }
    const PitchFrame& newest() const noexcept { return at (count - 1); }

    /** Frame values at `time`, linearly interpolated between neighbouring voiced frames. */
    bool lookup (double time, PitchFrame& out) const noexcept
    {
        if (count == 0)
            return false;

        // Binary search for the last frame with frame.time <= time.
        int lo = 0, hi = count - 1;
        if (time <= (double) at (0).time) { out = at (0); return true; }
        if (time >= (double) at (hi).time) { out = at (hi); return true; }

        while (hi - lo > 1)
        {
            const int mid = (lo + hi) / 2;
            if ((double) at (mid).time <= time) lo = mid; else hi = mid;
        }

        const auto& a = at (lo);
        const auto& b = at (hi);
        if (! (a.voiced && b.voiced))
        {
            out = (time - (double) a.time) < ((double) b.time - time) ? a : b;
            return true;
        }

        const float t = (float) ((time - (double) a.time) / juce::jmax (1.0, (double) (b.time - a.time)));
        out = a;
        out.periodSamples = a.periodSamples + (b.periodSamples - a.periodSamples) * t;
        out.srcMidi = a.srcMidi + (b.srcMidi - a.srcMidi) * t;
        out.mainShift = a.mainShift + (b.mainShift - a.mainShift) * t;
        out.melodyMidi = a.melodyMidi + (b.melodyMidi - a.melodyMidi) * t;
        out.expression = a.expression + (b.expression - a.expression) * t;
        out.globalOffset = a.globalOffset + (b.globalOffset - a.globalOffset) * t;
        return true;
    }

private:
    const PitchFrame& at (int logicalIndex) const noexcept { return frames[(size_t) ((head + logicalIndex) % capacity)]; }

    std::array<PitchFrame, capacity> frames {};
    int head = 0, count = 0;
};

//==============================================================================
/** Input history + pitch-synchronous analysis marks shared by every PSOLA synthesis voice. */
class PsolaAnalyzer
{
public:
    struct Mark
    {
        juce::int64 pos = 0;
        float period = 0.0f;     // detector period at this mark
        float spacing = 0.0f;    // measured distance to the next voiced mark (0 = unknown)
        bool voiced = false;
    };

    void prepare (double sampleRate, float minHz, float maxHz);
    void reset();

    /** Appends input. Marks are placed up to (writePos - placementLag) using periods from the timeline. */
    void push (const float* input, int numSamples, const PitchTimeline& timeline, int placementLag);

    juce::int64 getWritePosition() const noexcept { return writePos; }
    float getMaxPeriod() const noexcept           { return maxPeriod; }
    float getUnvoicedHop() const noexcept         { return unvoicedHop; }

    /** Cubic (Hermite) interpolated input sample at a fractional absolute position. */
    inline float interpolate (double pos) const noexcept
    {
        const double lo = (double) (writePos - (juce::int64) ring.size() + 4);
        const double hi = (double) (writePos - 3);
        pos = juce::jlimit (lo, hi, pos);
        const auto i = (juce::int64) std::floor (pos);
        const float f = (float) (pos - (double) i);
        const float xm1 = ring[(size_t) ((i - 1) & mask)];
        const float x0  = ring[(size_t) (i & mask)];
        const float x1  = ring[(size_t) ((i + 1) & mask)];
        const float x2  = ring[(size_t) ((i + 2) & mask)];
        const float c1 = 0.5f * (x1 - xm1);
        const float c2 = xm1 - 2.5f * x0 + 2.0f * x1 - 0.5f * x2;
        const float c3 = 0.5f * (x2 - xm1) + 1.5f * (x0 - x1);
        return ((c3 * f + c2) * f + c1) * f + x0;
    }

    /** Nearest analysis mark to `pos`. Returns false when no marks exist yet. */
    bool findNearestMark (double pos, Mark& result) const noexcept;

private:
    void placeMarks (const PitchTimeline& timeline, int placementLag);
    void addMark (const Mark& m) noexcept;
    void updatePeakFilter (float periodSamples);

    double sr = 48000.0;
    std::vector<float> ring, lpRing;
    juce::int64 mask = 0, writePos = 0;
    float maxPeriod = 800.0f, minPeriod = 40.0f, unvoicedHop = 120.0f;

    Biquad peakLp1, peakLp2;
    float peakFilterPeriod = 0.0f;

    std::vector<Mark> marks;     // ring buffer of marks
    int markHead = 0, markCount = 0;
    Mark lastMark;
    bool hasLastMark = false;
};

//==============================================================================
/** One PSOLA synthesis voice: places pitch-shifted, formant-scaled grains into an output accumulator. */
class PsolaVoice
{
public:
    struct GrainParams
    {
        float pitchRatio = 1.0f;     // > 1 raises pitch
        float formantRatio = 1.0f;   // > 1 raises formants (smaller / "female" vocal tract)
        float gain = 1.0f;
        float readDelay = 0.0f;      // samples of extra delay (doubling / ensemble)
    };

    void prepare (int maxLatencySamples);
    void reset (double startPosition) noexcept;

    /** Synthesises every grain whose centre lies before `untilPos`.
        `paramsAt (double grainCentre, const PsolaAnalyzer::Mark&)` returns the grain parameters. */
    template <typename ParamFn>
    void synthesise (const PsolaAnalyzer& analyzer, double untilPos, ParamFn&& paramsAt) noexcept
    {
        int guard = 0;
        while (nextPos < untilPos && ++guard < 4096)
        {
            double s = nextPos;
            PsolaAnalyzer::Mark am;

            GrainParams gp = paramsAt (s, PsolaAnalyzer::Mark {});
            const double lookup = s - (double) gp.readDelay;

            if (! analyzer.findNearestMark (lookup, am))
            {
                am.pos = (juce::int64) std::llround (lookup);
                am.period = analyzer.getUnvoicedHop();
                am.voiced = false;
            }

            gp = paramsAt (s, am);

            // Re-synchronise grain placement with the analysis epochs at voicing onsets.
            if (am.voiced && ! lastVoiced)
            {
                const double aligned = (double) am.pos + (double) gp.readDelay;
                if (aligned > s && aligned - s < (double) am.period)
                    s = aligned;
            }

            const float tIn = am.period;
            const float f = juce::jlimit (0.5f, 2.0f, gp.formantRatio);
            float tOut, half, g;

            if (am.voiced)
            {
                const float r = juce::jlimit (0.125f, 8.0f, gp.pitchRatio);
                // Advancing by the measured mark spacing keeps grains phase-locked to the epochs.
                const float advance = am.spacing > 0.0f ? am.spacing : tIn;
                tOut = advance / r;
                half = juce::jmin (tIn / f, 1.25f * tIn);
                g = gp.gain * juce::jlimit (0.25f, 1.6f, std::sqrt (tOut / half));
            }
            else
            {
                tOut = tIn;
                half = tIn;
                g = gp.gain;
            }

            addGrain (analyzer, s, (double) am.pos, half, f, g);

            nextPos = s + (double) juce::jmax (8.0f, tOut);
            lastVoiced = am.voiced;
        }
    }

    /** Reads (and clears) n output samples starting at absolute position `start`. */
    void read (juce::int64 start, float* output, int numSamples) noexcept;

    double getNextGrainPosition() const noexcept { return nextPos; }

private:
    void addGrain (const PsolaAnalyzer&, double centre, double markPos, float half, float formantRatio, float gain) noexcept;

    std::vector<float> acc;
    juce::int64 mask = 0;
    double nextPos = 0.0;
    bool lastVoiced = false;
};

} // namespace voxa::dsp
