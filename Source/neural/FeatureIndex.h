#pragma once

#include <juce_core/juce_core.h>
#include <memory>
#include <vector>

namespace voxa::neural
{

/** The AI singer's own sounds: content features (contentvec) of her training recordings, clustered (k-means), so
    a conversion can pull each frame of the input toward the nearest things she really sang (RVC's "feature
    index"). Without it the model hears a man's vowels in the features and keeps them: she sang with his darker,
    lower vowel shapes, like a man's voice sped up (listening, Female Blues on GTSinger Alto 1: "like the Oompa
    Loompas"; F2 ~1400 Hz where she sings ~1850, 6 dB less 2 - 5 kHz than her own recordings).

    Stored in the pack as a NumPy .npy file: [count, channels], float16 or float32, C order. */
class FeatureIndex
{
public:
    /** nullptr (and `error`) when the file can't be read or its features aren't `channels` wide. */
    static std::shared_ptr<const FeatureIndex> load (const juce::File& file, int channels, juce::String& error);

    int getNumFeatures() const noexcept { return count; }
    int getChannels() const noexcept    { return channels; }

    static constexpr int neighbours = 8;

    /** Replaces `out` (channels floats) with the mean of the `neighbours` nearest features to `query`, weighted
        by 1 / distance^2 (as RVC does). Allocates nothing. */
    void retrieve (const float* query, float* out) const noexcept;

    /** Blends each frame of `features` ([numFrames, channels], time-major) toward what it retrieves:
        rate * retrieved + (1 - rate) * own. Unvoiced frames (consonants, breaths: `voiced[i]` 0) take only
        `protect` of that change, so the words stay clear (RVC's "protect"). Frames equal to the one before
        (the encoder's 20 ms frames, given twice) are retrieved once. `scratch` holds 2 * channels floats. */
    void blend (float* features, int numFrames, const float* voiced, float rate, float protect, float* scratch) const noexcept;

private:
    FeatureIndex() = default;

    int count = 0, channels = 0;
    std::vector<float> data, norms;   // [count, channels], squared norms
};

} // namespace voxa::neural
