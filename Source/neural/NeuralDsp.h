#pragma once

#include <juce_core/juce_core.h>
#include <cstdint>
#include <vector>

/** Signal-processing helpers of the neural voice engine (no ONNX Runtime dependency). */
namespace voxa::neural::dsp
{

// RVC frame geometry. The generator works on 10 ms frames; the content encoder (ContentVec/HuBERT) on
// 20 ms frames at 16 kHz with a 400-sample receptive field.
inline constexpr int encoderSampleRate   = 16000;
inline constexpr int framesPerSecond     = 100;
inline constexpr int samplesPerFrame16k  = encoderSampleRate / framesPerSecond;   // 160
inline constexpr int encoderHop16k       = 320;
inline constexpr int encoderRightPad16k  = 400 - encoderHop16k;                    // 80 -> T = N / 320 exactly

inline constexpr float f0MinHz = 50.0f;
inline constexpr float f0MaxHz = 1100.0f;

/** RVC's coarse pitch: mel-scaled f0 quantised to 1..255 (1 = unvoiced). */
int coarsePitch (float f0Hz) noexcept;

//==============================================================================
/** Stateless, band-limited (Kaiser-windowed sinc, polyphase table) resampler between two integer rates.

    Sample i of a stream at rate R is taken to lie at time i / R, so streams at different rates that start
    at index 0 are time-aligned; the filter is linear-phase and introduces no delay. Because it is
    stateless and works on absolute sample indices, the same object can render a whole clip or any
    window of an endless stream with identical results (needed for overlapping real-time chunks). */
class Resampler
{
public:
    Resampler() = default;
    Resampler (int inputRate, int outputRate)        { prepare (inputRate, outputRate); }

    void prepare (int inputRate, int outputRate);
    bool isPrepared() const noexcept                 { return inRate > 0; }
    int getInputRate() const noexcept                { return inRate; }
    int getOutputRate() const noexcept               { return outRate; }

    /** Number of input samples needed on each side of an output instant. */
    int getHalfWidth() const noexcept                { return halfWidth; }

    /** Renders output samples with absolute indices [outStart, outStart + numOut) from the input samples
        with absolute indices [inStart, inStart + numIn); input outside that range counts as silence. */
    void process (const float* input, int64_t inStart, int numIn,
                  float* output, int64_t outStart, int numOut) const noexcept;

    /** Resamples a whole clip; the result has ceil(numIn * outRate / inRate) samples. */
    std::vector<float> processAll (const float* input, int numIn) const;

    /** Range of input indices [first, last] that affect output index `outIndex`. */
    int64_t firstInputFor (int64_t outIndex) const noexcept;
    int64_t lastInputFor (int64_t outIndex) const noexcept;

private:
    static constexpr int numPhases = 256;

    int inRate = 0, outRate = 0, halfWidth = 0, numTaps = 0;
    std::vector<float> table;   // (numPhases + 1) rows of numTaps coefficients
};

//==============================================================================
/** Interpolates an f0 contour given at `frameRate` frames/s (value k covering [k/rate, (k+1)/rate))
    onto `numFrames` 10 ms frames. Voiced neighbours are interpolated geometrically, voiced/unvoiced
    transitions use the nearest value, frames beyond the contour are unvoiced (0). */
void resampleF0 (const std::vector<float>& f0Hz, double frameRate, float* out, int numFrames);

/** f0 of one frame from per-sample values: mean of the voiced samples when at least half are voiced. */
float frameF0 (const float* perSampleF0, int numSamples) noexcept;

/** Deterministic standard-normal noise indexed by (frame, channel): overlapping chunks of a stream get
    identical noise for the same absolute frame, which keeps their overlaps consistent. */
float gaussianNoise (int64_t frame, int channel, uint64_t seed = 0x5eed) noexcept;

//==============================================================================
/** SOLA: finds the offset in [0, searchLength] at which `candidate` best continues `reference`
    (maximum normalised cross-correlation). `candidate` must hold length + searchLength samples.
    `correlation` receives the normalised correlation (-1..1) at that offset. */
int findBestOffset (const float* reference, const float* candidate, int length, int searchLength,
                    float& correlation) noexcept;

/** Normalised correlation of two equally long signals at zero lag (0 if either is silent). */
float correlation (const float* a, const float* b, int length) noexcept;

/** Crossfades `fadingOut` into `fadingIn` (raised-cosine). The gains are normalised for the signals'
    correlation so that loudness stays constant whether they are aligned (r = 1, amplitude-complementary)
    or unrelated (r = 0, equal-power). Writes `length` samples to `dest` (may alias `fadingIn`). */
void crossfade (const float* fadingOut, const float* fadingIn, float* dest, int length, float correlation) noexcept;

/** The gain crossfade() gives sample i of the fading-in signal when the two are unrelated (r = 0).
    The fading-out signal's gain is crossfadeGain (length - 1 - i, length); the squares of both sum to 1. */
float crossfadeGain (int i, int length) noexcept;

/** True when the block is below -60 dBFS RMS and -40 dBFS peak. */
bool isSilent (const float* samples, int numSamples) noexcept;

} // namespace voxa::neural::dsp
