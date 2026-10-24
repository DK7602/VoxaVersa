#pragma once

#include <juce_core/juce_core.h>
#include <array>
#include <atomic>
#include <cmath>

namespace voxa
{

/** Lock-free min/max waveform history written by the audio thread and read by the UI. */
class ScopeBuffer
{
public:
    static constexpr int capacity = 2048;

    void prepare (double sampleRate, double pointsPerSecond = 220.0) noexcept
    {
        samplesPerPoint = juce::jmax (1, (int) std::round (sampleRate / pointsPerSecond));
        counter = 0;
        curMin = 1.0f;
        curMax = -1.0f;
    }

    void push (const float* samples, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            const float s = samples[i];
            curMin = juce::jmin (curMin, s);
            curMax = juce::jmax (curMax, s);

            if (++counter >= samplesPerPoint)
            {
                const int w = writeIndex.load (std::memory_order_relaxed);
                mins[(size_t) w].store (curMin, std::memory_order_relaxed);
                maxs[(size_t) w].store (curMax, std::memory_order_relaxed);
                writeIndex.store ((w + 1) % capacity, std::memory_order_release);
                counter = 0;
                curMin = 1.0f;
                curMax = -1.0f;
            }
        }
    }

    /** Copies the most recent `numPoints` (oldest first). */
    void read (float* destMin, float* destMax, int numPoints) const noexcept
    {
        numPoints = juce::jmin (numPoints, capacity);
        const int w = writeIndex.load (std::memory_order_acquire);
        for (int i = 0; i < numPoints; ++i)
        {
            const int idx = (w - numPoints + i + capacity) % capacity;
            destMin[i] = mins[(size_t) idx].load (std::memory_order_relaxed);
            destMax[i] = maxs[(size_t) idx].load (std::memory_order_relaxed);
        }
    }

private:
    std::array<std::atomic<float>, capacity> mins {}, maxs {};
    std::atomic<int> writeIndex { 0 };
    int samplesPerPoint = 200, counter = 0;
    float curMin = 1.0f, curMax = -1.0f;
};

/** Peak / RMS levels for up to two channels. The UI applies its own ballistics. */
class LevelMeterSource
{
public:
    void update (const float* const* channels, int numChannels, int numSamples) noexcept
    {
        for (int ch = 0; ch < 2; ++ch)
        {
            const float* data = channels[juce::jmin (ch, numChannels - 1)];
            float peak = 0.0f, sum = 0.0f;
            for (int i = 0; i < numSamples; ++i)
            {
                const float a = std::abs (data[i]);
                peak = juce::jmax (peak, a);
                sum += data[i] * data[i];
            }
            const float rms = numSamples > 0 ? std::sqrt (sum / (float) numSamples) : 0.0f;

            // Hold the max until the UI consumes it.
            float prev = peaks[(size_t) ch].load (std::memory_order_relaxed);
            while (peak > prev && ! peaks[(size_t) ch].compare_exchange_weak (prev, peak)) {}
            rmsLevels[(size_t) ch].store (rms, std::memory_order_relaxed);
        }
    }

    /** Returns and resets the held peak (linear). */
    float consumePeak (int channel) noexcept   { return peaks[(size_t) juce::jlimit (0, 1, channel)].exchange (0.0f); }
    float getRms (int channel) const noexcept  { return rmsLevels[(size_t) juce::jlimit (0, 1, channel)].load (std::memory_order_relaxed); }

private:
    std::array<std::atomic<float>, 2> peaks {}, rmsLevels {};
};

} // namespace voxa
