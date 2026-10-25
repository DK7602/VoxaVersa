#pragma once

#include "../dsp/Biquad.h"
#include <array>

namespace voxa
{

/** What a VoxaVersa watermark says about the audio it is in. */
enum class WatermarkKind
{
    None = 0,
    AiVoice,              // re-sung by a neural voice model
    SynthesizedSinging    // Text to Sing (formant synthesis)
};

/** Inaudible spread-spectrum watermark for AI and synthetic vocals (EU AI Act Article 50(2)).

    A fixed pseudo-random noise pattern (period 2^15 samples at 48 kHz, ~0.68 s), split into eight
    bands between 1 and 8 kHz, is added to the vocal. Each band follows the vocal's own level in that
    band ~30 dB down, so the mark sits under the voice, vanishes in silence and adds no latency. The
    pattern is defined in seconds, so it survives resampling, MP3/AAC encoding, gain changes and light
    noise; it does not survive pitch shifting or time stretching. Each WatermarkKind uses its own pattern.

    process() is real-time safe. prepare() allocates. */
class WatermarkEmbedder
{
public:
    static constexpr float defaultStrength = 0.03f;   // per-band mark level relative to the vocal

    void prepare (double sampleRate);
    /** Restarts the pattern at its first sample and clears the level followers. */
    void reset() noexcept;

    /** Adds the mark of `kind` in place (`right` may be nullptr for mono). With WatermarkKind::None
        the mark fades out over ~50 ms; switching kinds swaps the pattern immediately. */
    void process (float* left, float* right, int numSamples, WatermarkKind kind) noexcept;

    void setStrength (float newStrength) noexcept   { strength = newStrength; }

    static constexpr int numBands = 8;

private:
    struct Channel
    {
        std::array<dsp::Biquad, 2 * numBands> band;   // two cascaded band-passes per band
        std::array<float, numBands> meanSquare {};
    };

    float nextMark (Channel&, float input, const float* const* tables, int index, float frac) noexcept;

    double sr = 48000.0;
    double position = 0.0, increment = 1.0;
    float msCoeff = 0.0f, gain = 0.0f, gainStep = 0.0f, strength = defaultStrength;
    WatermarkKind current = WatermarkKind::AiVoice;
    Channel channels[2];
};

/** Result of searching audio for a VoxaVersa watermark. Scores are detection z-scores: unmarked audio
    stays below ~5, a mark is reported from `threshold` up. A few seconds of marked vocal score 10+. */
struct WatermarkDetection
{
    static constexpr float threshold = 7.0f;

    WatermarkKind kind = WatermarkKind::None;
    float aiVoiceScore = 0.0f;
    float synthesizedSingingScore = 0.0f;
    double offsetSeconds = 0.0;   // pattern phase at the start of the audio

    bool found() const noexcept   { return kind != WatermarkKind::None; }
};

/** Looks for a VoxaVersa watermark in any audio (any sample rate, mono or multichannel). Not real-time. */
WatermarkDetection detectWatermark (const juce::AudioBuffer<float>& audio, double sampleRate);

/** "AI voice", "synthesized singing" or "none". */
juce::String watermarkKindName (WatermarkKind kind);

} // namespace voxa
