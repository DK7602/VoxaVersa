#pragma once

#include "SingEngine.h"

namespace voxa::sing
{

struct PitchFrame
{
    float hz = 0.0f;          // 0 = unvoiced
    float clarity = 0.0f;     // 1 - YIN aperiodicity (1 = perfectly periodic)
    float rms = 0.0f;         // over the (long) analysis window
    float energy = 0.0f;      // short-time RMS (10 ms) around the frame centre
};

/** YIN fundamental-frequency estimator (de Cheveigne & Kawahara, 2002), with cumulative-mean
    normalisation, absolute threshold and parabolic interpolation. Offline use (allocates). */
class YinPitchTracker
{
public:
    YinPitchTracker (double sampleRate, float minHz = 55.0f, float maxHz = 1200.0f, float threshold = 0.15f);

    /** Number of samples estimate() reads. */
    int getFrameSize() const noexcept   { return windowSize + maxLag + 2; }

    /** Estimates the pitch of the frame starting at `samples` (getFrameSize() samples). */
    PitchFrame estimate (const float* samples);

private:
    double sampleRate;
    int minLag, maxLag, windowSize;
    float threshold;
    std::vector<float> difference;
};

/** Pitch contour of a mono signal at `hopSeconds` intervals (frame i is centred on i * hop).
    Signals above 24 kHz are low-passed and decimated first for speed. */
std::vector<PitchFrame> trackPitch (const float* samples, int numSamples, double sampleRate,
                                    double hopSeconds = 0.01, float minHz = 55.0f, float maxHz = 1200.0f);

/** Hum-to-melody: segments a pitch track into notes and quantises them to a 1/16 grid. */
std::vector<Note> transcribeMelody (const float* samples, int numSamples, double sampleRate, double bpm);

} // namespace voxa::sing
