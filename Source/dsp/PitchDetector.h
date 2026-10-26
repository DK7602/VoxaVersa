#pragma once

#include "Biquad.h"
#include <vector>

namespace voxa::dsp
{

/** Streaming monophonic pitch detector (YIN on a decimated signal, refined at full rate).

    Every `getHopSize()` input samples a new estimate is produced and reported through the callback
    passed to process(). Estimates describe the signal `getAnalysisDelay()` samples before the end of
    the analysed audio. */
class PitchDetector
{
public:
    struct Frame
    {
        float frequencyHz = 0.0f;   // 0 when unvoiced
        float periodSamples = 0.0f; // full-rate samples (0 when unvoiced)
        float confidence = 0.0f;    // 0..1 (1 - YIN aperiodicity)
        float levelDb = -100.0f;    // RMS of the analysis window
        bool voiced = false;
    };

    void prepare (double sampleRate, float minHz = 60.0f, float maxHz = 1200.0f);
    void reset();

    /** Feeds samples. `onFrame (int offsetInBlock, const Frame&)` is called for each new estimate,
        where offsetInBlock is the index of the sample that completed the hop. */
    template <typename Callback>
    void process (const float* input, int numSamples, Callback&& onFrame)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            if (pushSample (input[i]))
                onFrame (i, latest);
        }
    }

    void process (const float* input, int numSamples) { process (input, numSamples, [] (int, const Frame&) {}); }

    const Frame& getLatest() const noexcept { return latest; }
    int getHopSize() const noexcept         { return hopFull; }
    int getAnalysisDelay() const noexcept   { return analysisDelay; }
    float getMinHz() const noexcept         { return minFreq; }
    float getMaxHz() const noexcept         { return maxFreq; }

    /** Voicing sensitivity: frames quieter than this are unvoiced. */
    void setGateDb (float db) noexcept      { gateDb = db; }

    /** Offline analysis of a whole buffer. Frame i describes the signal around sample i * hop. */
    static std::vector<Frame> analyse (const float* input, int numSamples, double sampleRate,
                                       int hopSamples, float minHz = 60.0f, float maxHz = 1200.0f);

private:
    bool pushSample (float x) noexcept;
    void analyseFrame() noexcept;
    float refineAtFullRate (float tauFull) const noexcept;

    double sr = 48000.0;
    float minFreq = 60.0f, maxFreq = 1200.0f, gateDb = -50.0f;
    int decimation = 3;
    double decimatedRate = 16000.0;

    // decimated analysis
    Biquad aa1, aa2;
    std::vector<float> dBuf;     // circular decimated history
    int dWrite = 0, dSize = 0;
    int windowD = 0, maxLagD = 0, minLagD = 0, hopD = 0;
    int decimCounter = 0, hopCounter = 0;
    std::vector<float> frameBuf, diff, cmnd;

    // full-rate history for refinement
    std::vector<float> fBuf;
    int fWrite = 0, fMask = 0, windowFull = 0;

    int hopFull = 192, analysisDelay = 0;
    Frame latest;
    float lastPeriods[3] {};
    int voicedRun = 0, unvoicedRun = 0;

    // The singer's recent register (log2 Hz, slow average of voiced estimates) for the octave check.
    float registerLog2 = 0.0f, registerSeconds = 0.0f;
};

} // namespace voxa::dsp
