#include "PitchTracker.h"

#include <algorithm>

namespace voxa::sing
{

YinPitchTracker::YinPitchTracker (double sr, float minHz, float maxHz, float thresholdToUse)
    : sampleRate (sr), threshold (thresholdToUse)
{
    maxLag = juce::jmax (4, (int) std::ceil (sr / juce::jmax (20.0f, minHz)));
    minLag = juce::jmax (2, (int) std::floor (sr / juce::jmax (minHz + 1.0f, maxHz)));
    windowSize = maxLag;
    difference.resize ((size_t) maxLag + 3);
}

PitchFrame YinPitchTracker::estimate (const float* x)
{
    PitchFrame result;

    double energy = 0.0;
    for (int j = 0; j < windowSize; ++j)
        energy += (double) x[j] * x[j];
    result.rms = (float) std::sqrt (energy / windowSize);

    if (energy <= 1.0e-12)
        return result;

    // Difference function d(tau) and its cumulative-mean-normalised version d'(tau).
    const int lastLag = maxLag + 1;
    difference[0] = 1.0f;
    double runningSum = 0.0;
    for (int tau = 1; tau <= lastLag; ++tau)
    {
        double d = 0.0;
        const float* a = x;
        const float* b = x + tau;
        for (int j = 0; j < windowSize; ++j)
        {
            const float diff = a[j] - b[j];
            d += (double) diff * diff;
        }
        runningSum += d;
        difference[(size_t) tau] = runningSum > 0.0 ? (float) (d * tau / runningSum) : 1.0f;
    }

    int best = -1;
    for (int tau = minLag; tau < lastLag; ++tau)
    {
        if (difference[(size_t) tau] < threshold)
        {
            while (tau + 1 < lastLag && difference[(size_t) tau + 1] < difference[(size_t) tau])
                ++tau;
            best = tau;
            break;
        }
    }

    if (best < 0)
    {
        best = minLag;
        for (int tau = minLag + 1; tau < lastLag; ++tau)
            if (difference[(size_t) tau] < difference[(size_t) best])
                best = tau;
    }

    // Octave check: when a strong 2nd harmonic (e.g. on a formant) makes half the period look
    // periodic enough, the full period is still clearly *more* periodic.
    for (int pass = 0; pass < 2; ++pass)
    {
        const int target = 2 * best;
        if (target + 2 >= lastLag || difference[(size_t) best] < 0.02f)
            break;

        int local = target;
        for (int tau = target - 2; tau <= target + 2; ++tau)
            if (difference[(size_t) tau] < difference[(size_t) local])
                local = tau;

        if (difference[(size_t) local] >= 0.5f * difference[(size_t) best])
            break;
        best = local;
    }

    result.clarity = juce::jlimit (0.0f, 1.0f, 1.0f - difference[(size_t) best]);
    if (difference[(size_t) best] > 0.45f || best <= 1 || best >= lastLag)
        return result;

    // Parabolic interpolation around the minimum.
    const float a = difference[(size_t) best - 1];
    const float b = difference[(size_t) best];
    const float c = difference[(size_t) best + 1];
    const float denominator = a - 2.0f * b + c;
    float shift = 0.0f;
    if (std::abs (denominator) > 1.0e-9f)
        shift = juce::jlimit (-0.5f, 0.5f, 0.5f * (a - c) / denominator);

    result.hz = (float) (sampleRate / ((double) best + shift));
    return result;
}

//==============================================================================
std::vector<PitchFrame> trackPitch (const float* samples, int numSamples, double sampleRate,
                                    double hopSeconds, float minHz, float maxHz)
{
    std::vector<PitchFrame> frames;
    if (samples == nullptr || numSamples <= 0 || sampleRate <= 0.0)
        return frames;

    // Decimate to ~16-24 kHz: plenty for voice pitch, and the difference function is O(n^2).
    const int factor = juce::jmax (1, (int) std::floor (sampleRate / 16000.0));
    const double rate = sampleRate / factor;

    std::vector<float> signal;
    if (factor > 1)
    {
        std::vector<float> filtered (samples, samples + numSamples);
        const double cutoff = 0.4 * rate;
        for (double q : { 0.5412, 1.3066 })
        {
            juce::IIRFilter filter;
            filter.setCoefficients (juce::IIRCoefficients::makeLowPass (sampleRate, cutoff, q));
            filter.processSamples (filtered.data(), numSamples);
        }
        signal.reserve ((size_t) (numSamples / factor + 1));
        for (int i = 0; i < numSamples; i += factor)
            signal.push_back (filtered[(size_t) i]);
    }
    else
    {
        signal.assign (samples, samples + numSamples);
    }

    YinPitchTracker yin (rate, minHz, maxHz);
    const int frameSize = yin.getFrameSize();
    std::vector<float> frame ((size_t) frameSize);

    const int n = (int) signal.size();
    const double hop = hopSeconds * rate;
    const int numFrames = (int) std::ceil ((double) n / hop);
    frames.reserve ((size_t) numFrames);

    const int shortHalf = juce::jmax (1, (int) std::round (0.005 * rate));
    for (int i = 0; i < numFrames; ++i)
    {
        const int centre = (int) std::round (i * hop);
        const int start = centre - frameSize / 2;
        for (int j = 0; j < frameSize; ++j)
        {
            const int k = start + j;
            frame[(size_t) j] = (k >= 0 && k < n) ? signal[(size_t) k] : 0.0f;
        }
        auto result = yin.estimate (frame.data());

        double sum = 0.0;
        for (int k = juce::jmax (0, centre - shortHalf); k < juce::jmin (n, centre + shortHalf); ++k)
            sum += (double) signal[(size_t) k] * signal[(size_t) k];
        result.energy = (float) std::sqrt (sum / (2 * shortHalf));
        frames.push_back (result);
    }
    return frames;
}

//==============================================================================
namespace
{
    float median (std::vector<float> v)
    {
        if (v.empty())
            return 0.0f;
        const auto mid = v.begin() + (long) v.size() / 2;
        std::nth_element (v.begin(), mid, v.end());
        return *mid;
    }

    struct Segment
    {
        int first = 0, last = 0;
        std::vector<float> pitches;
        std::vector<float> levels;
        bool rearticulated = false;

        float pitch() const
        {
            // Median of the central part: ignores scoops into and falls out of the note.
            const int n = (int) pitches.size();
            const int trim = n >= 6 ? n / 5 : 0;
            return median (std::vector<float> (pitches.begin() + trim, pitches.end() - trim));
        }
    };
}

std::vector<Note> transcribeMelody (const float* samples, int numSamples, double sampleRate, double bpm)
{
    std::vector<Note> notes;
    constexpr double hop = 0.01;
    const auto frames = trackPitch (samples, numSamples, sampleRate, hop);
    if (frames.empty())
        return notes;

    float maxRms = 0.0f;
    for (const auto& f : frames)
        maxRms = juce::jmax (maxRms, f.energy);
    if (maxRms <= 1.0e-5f)
        return notes;

    const float silenceThreshold = juce::jmax (1.0e-4f, maxRms * 0.03f);
    const int n = (int) frames.size();

    std::vector<bool> voiced ((size_t) n), silent ((size_t) n);
    std::vector<float> midi ((size_t) n, 0.0f);
    for (int i = 0; i < n; ++i)
    {
        const auto& f = frames[(size_t) i];
        silent[(size_t) i] = f.energy <= silenceThreshold;
        voiced[(size_t) i] = f.hz > 0.0f && f.clarity > 0.6f && ! silent[(size_t) i];
        if (voiced[(size_t) i])
            midi[(size_t) i] = music::hzToMidi (f.hz);
    }

    // Median-filter the pitch inside voiced runs to remove octave blips.
    {
        auto filtered = midi;
        for (int i = 0; i < n; ++i)
        {
            if (! voiced[(size_t) i])
                continue;
            std::vector<float> window;
            for (int k = i - 2; k <= i + 2; ++k)
                if (k >= 0 && k < n && voiced[(size_t) k])
                    window.push_back (midi[(size_t) k]);
            filtered[(size_t) i] = median (window);
        }
        midi = filtered;
    }

    std::vector<Segment> segments;
    Segment current;
    bool active = false;
    int gap = 0, silence = 0;

    auto close = [&]
    {
        if (active && ! current.pitches.empty())
            segments.push_back (current);
        active = false;
        current = Segment();
    };

    for (int i = 0; i < n; ++i)
    {
        if (! voiced[(size_t) i])
        {
            // A real silence (even a short one) ends the note; a tracking dropout is bridged.
            silence = silent[(size_t) i] ? silence + 1 : 0;
            if (active && (++gap > 5 || silence >= 2))
                close();
            continue;
        }
        gap = 0;
        silence = 0;

        const float m = midi[(size_t) i];
        const float level = frames[(size_t) i].energy;

        if (! active)
        {
            current = Segment();
            current.first = current.last = i;
            active = true;
            current.pitches.push_back (m);
            current.levels.push_back (level);
            continue;
        }

        const int recent = juce::jmin (8, (int) current.pitches.size());
        const float reference = median (std::vector<float> (current.pitches.end() - recent, current.pitches.end()));

        if (std::abs (m - reference) > 0.75f)
        {
            // A new note only if the new pitch is stable for a few frames (vibrato and scoops aren't).
            bool stable = true;
            for (int k = 1; k <= 2; ++k)
            {
                if (i + k >= n || ! voiced[(size_t) (i + k)])
                {
                    stable = false;
                    break;
                }
                const float mk = midi[(size_t) (i + k)];
                stable = stable && std::abs (mk - reference) > 0.75f && std::abs (mk - m) < 0.6f;
            }

            if (stable)
            {
                close();
                current.first = current.last = i;
                active = true;
                current.pitches.push_back (m);
                current.levels.push_back (level);
            }
            continue;
        }

        // Re-articulation of the same pitch ("da da da"): a dip in level followed by a new attack.
        if (current.levels.size() >= 6)
        {
            const int window = juce::jmin (6, (int) current.levels.size());
            const float recentMin = *std::min_element (current.levels.end() - window, current.levels.end());
            const float typical = median (current.levels);
            if (recentMin < 0.45f * typical && level > 1.8f * recentMin && current.levels.back() < level)
            {
                close();
                current.first = current.last = i;
                current.rearticulated = true;
                active = true;
                current.pitches.push_back (m);
                current.levels.push_back (level);
                continue;
            }
        }

        current.last = i;
        current.pitches.push_back (m);
        current.levels.push_back (level);
    }
    close();

    // Absorb very short fragments (scoops, glitches) into a contiguous neighbour, drop isolated ones.
    for (bool changed = true; changed;)
    {
        changed = false;
        for (size_t s = 0; s < segments.size(); ++s)
        {
            auto& seg = segments[s];
            if (seg.last - seg.first + 1 >= 8)
                continue;

            const bool prevContiguous = s > 0 && seg.first - segments[s - 1].last <= 3;
            const bool nextContiguous = s + 1 < segments.size() && segments[s + 1].first - seg.last <= 3;

            if (! prevContiguous && ! nextContiguous)
            {
                if (seg.last - seg.first + 1 < 5)
                {
                    segments.erase (segments.begin() + (long) s);
                    changed = true;
                    break;
                }
                continue;
            }

            size_t target;
            if (prevContiguous && nextContiguous)
                target = std::abs (segments[s - 1].pitch() - seg.pitch()) <= std::abs (segments[s + 1].pitch() - seg.pitch()) ? s - 1 : s + 1;
            else
                target = prevContiguous ? s - 1 : s + 1;

            auto& t = segments[target];
            t.first = juce::jmin (t.first, seg.first);
            t.last = juce::jmax (t.last, seg.last);
            segments.erase (segments.begin() + (long) s);
            changed = true;
            break;
        }
    }

    const double beatsPerSecond = juce::jmax (1.0, bpm) / 60.0;
    constexpr double grid = 0.25;

    struct Pending { Note note; bool rearticulated; double rawStart, rawEnd; };
    std::vector<Pending> pending;
    for (const auto& seg : segments)
    {
        const double start = seg.first * hop;
        const double end = (seg.last + 1) * hop;

        Note note;
        note.startBeat = std::round (start * beatsPerSecond / grid) * grid;
        note.lengthBeats = std::round (end * beatsPerSecond / grid) * grid - note.startBeat;
        if (note.lengthBeats < grid)
            note.lengthBeats = grid;
        note.midiNote = juce::jlimit (0, 127, (int) std::lround (seg.pitch()));
        note.velocity = juce::jlimit (0.3f, 1.0f, 0.5f + 0.5f * std::sqrt (median (seg.levels) / maxRms));
        pending.push_back ({ note, seg.rearticulated, start, end });
    }

    double previousRawEnd = -1.0;
    for (const auto& p : pending)
    {
        if (! notes.empty())
        {
            auto& prev = notes.back();
            if (std::abs (p.note.startBeat - prev.startBeat) < 1.0e-9)
            {
                prev.lengthBeats = juce::jmax (prev.lengthBeats, p.note.lengthBeats);
                continue;
            }

            if (prev.startBeat + prev.lengthBeats > p.note.startBeat)
                prev.lengthBeats = p.note.startBeat - prev.startBeat;

            // Same pitch, no real gap and no new attack: one long note split by the tracker.
            const bool adjacent = std::abs (prev.startBeat + prev.lengthBeats - p.note.startBeat) < 1.0e-9;
            if (adjacent && prev.midiNote == p.note.midiNote && ! p.rearticulated && p.rawStart - previousRawEnd < 0.02)
            {
                prev.lengthBeats = p.note.startBeat + p.note.lengthBeats - prev.startBeat;
                previousRawEnd = p.rawEnd;
                continue;
            }
        }
        notes.push_back (p.note);
        previousRawEnd = p.rawEnd;
    }

    // Leading silence: keep the phase within the beat but drop whole empty beats (leave one).
    if (! notes.empty())
    {
        const double shift = std::floor (notes.front().startBeat) - 1.0;
        if (shift > 0.0)
            for (auto& note : notes)
                note.startBeat -= shift;
    }

    return notes;
}

} // namespace voxa::sing
