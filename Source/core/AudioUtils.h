#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <unordered_map>

namespace voxa
{

/** High-quality offline resampling of every channel (windowed sinc). */
inline juce::AudioBuffer<float> resampleBuffer (const juce::AudioBuffer<float>& source, double fromRate, double toRate)
{
    if (source.getNumSamples() == 0 || fromRate <= 0.0 || toRate <= 0.0 || std::abs (fromRate - toRate) < 1.0e-6)
        return source;

    const double ratio = fromRate / toRate;
    const int outLength = (int) std::ceil (source.getNumSamples() / ratio);
    juce::AudioBuffer<float> result (source.getNumChannels(), outLength);

    // Pad the input so the interpolator can read past the end.
    juce::AudioBuffer<float> padded (source.getNumChannels(), source.getNumSamples() + 256);
    padded.clear();
    for (int ch = 0; ch < source.getNumChannels(); ++ch)
    {
        padded.copyFrom (ch, 0, source, ch, 0, source.getNumSamples());
        juce::WindowedSincInterpolator interpolator;
        interpolator.process (ratio, padded.getReadPointer (ch), result.getWritePointer (ch), outLength);
    }
    return result;
}

/** Mono mix-down of a buffer. */
inline juce::AudioBuffer<float> toMono (const juce::AudioBuffer<float>& source)
{
    juce::AudioBuffer<float> mono (1, source.getNumSamples());
    mono.clear();
    if (source.getNumChannels() == 0)
        return mono;
    const float g = 1.0f / (float) source.getNumChannels();
    for (int ch = 0; ch < source.getNumChannels(); ++ch)
        mono.addFrom (0, 0, source, ch, 0, source.getNumSamples(), g);
    return mono;
}

/** Writes a buffer as a 24-bit WAV, optionally with metadata (see AiDisclosure.h). */
inline bool writeWav (const juce::File& file, const juce::AudioBuffer<float>& buffer, double sampleRate,
                      const std::unordered_map<juce::String, juce::String>& metadata = {})
{
    file.getParentDirectory().createDirectory();
    file.deleteFile();

    std::unique_ptr<juce::OutputStream> stream (file.createOutputStream());
    if (stream == nullptr)
        return false;

    juce::WavAudioFormat wav;
    const auto options = juce::AudioFormatWriterOptions{}.withSampleRate (sampleRate)
                                                         .withNumChannels (buffer.getNumChannels())
                                                         .withBitsPerSample (24)
                                                         .withMetadataValues (metadata);
    auto writer = wav.createWriterFor (stream, options);
    if (writer == nullptr)
        return false;
    return writer->writeFromAudioSampleBuffer (buffer, 0, buffer.getNumSamples());
}

/** Creates a WAV writer for streaming (e.g. recording). Returns nullptr on failure. */
inline std::unique_ptr<juce::AudioFormatWriter> createWavWriter (const juce::File& file, double sampleRate, int numChannels,
                                                                const std::unordered_map<juce::String, juce::String>& metadata = {})
{
    file.getParentDirectory().createDirectory();
    file.deleteFile();
    std::unique_ptr<juce::OutputStream> stream (file.createOutputStream());
    if (stream == nullptr)
        return nullptr;
    juce::WavAudioFormat wav;
    return wav.createWriterFor (stream, juce::AudioFormatWriterOptions{}.withSampleRate (sampleRate)
                                                                         .withNumChannels (numChannels)
                                                                         .withBitsPerSample (24)
                                                                         .withMetadataValues (metadata));
}

} // namespace voxa
