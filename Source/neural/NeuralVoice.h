#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <memory>
#include <vector>

#include "../voices/VoiceProfile.h"

namespace voxa::neural
{

/** Dynamically loads the ONNX Runtime shared library at run time (no link-time dependency).
    If the library cannot be found, neural voices are simply unavailable and VoxaVersa falls back to DSP. */
class OnnxRuntime
{
public:
    static OnnxRuntime& getInstance();

    /** Attempts to load the runtime from the default search locations (idempotent). */
    bool ensureLoaded();
    bool isAvailable() const noexcept;
    juce::String getStatusMessage() const;
    juce::File getLoadedLibraryFile() const;

    /** Locations searched, in order: next to the plugin binary, <user data>/Runtime, system paths. */
    static juce::Array<juce::File> getSearchLocations();

    struct Impl;
    Impl& getImpl() const noexcept { return *impl; }

private:
    OnnxRuntime();
    ~OnnxRuntime();
    std::unique_ptr<Impl> impl;
};

/** An RVC-compatible voice conversion model (content encoder + generator), loaded from a voice pack.
    Immutable after loading and safe to share between threads. */
class NeuralVoiceModel
{
public:
    ~NeuralVoiceModel();

    /** Loads the model referenced by a voice pack profile. Returns nullptr and sets `error` on failure.
        Loading takes seconds (it times the model); `cancel` (optional) stops it between steps, so whoever waits
        for it (closing the plug-in) never has to kill the thread. */
    static std::shared_ptr<NeuralVoiceModel> load (const VoiceProfile& profile, juce::String& error,
                                                   const std::atomic<bool>* cancel = nullptr);

    int getModelSampleRate() const noexcept;
    juce::String getDescription() const;

    /** Offline conversion of a complete mono clip.
        `f0Hz` holds the *target* pitch contour at `f0FrameRate` frames/second (0 = unvoiced).
        With `pitchShiftSemis` (same frame rate; NaN where `f0Hz` must be used as it is) and the neural pitch
        tracker installed, the voice sings the input's own pitch as that tracker hears it, shifted by these
        semitones (see NeuralPitchTracker); `f0Hz` is then only the fallback.
        `southernVowels`: the country voices' Southern "ah" (SouthernVowels), as setSouthernVowels() live.
        The result is written to `output` at `inputSampleRate`. */
    bool convert (const juce::AudioBuffer<float>& monoInput, double inputSampleRate,
                  const std::vector<float>& f0Hz, double f0FrameRate,
                  juce::AudioBuffer<float>& output,
                  const std::atomic<bool>* cancel, juce::String& error,
                  const std::vector<float>* pitchShiftSemis = nullptr, bool southernVowels = false) const;

    /** True when the neural pitch tracker (<Encoders>/fcpe.onnx) was installed when this model loaded. */
    bool hasNeuralPitch() const noexcept;

    struct Impl;
    Impl& getImpl() const noexcept { return *impl; }

private:
    NeuralVoiceModel();
    std::unique_ptr<Impl> impl;
};

/** Real-time streaming wrapper around a NeuralVoiceModel.
    Inference runs on an internal worker thread in overlapping chunks; the audio thread only touches
    lock-free FIFOs. The output is delayed by a constant getLatencySamples(). */
class NeuralVoiceStream
{
public:
    NeuralVoiceStream();
    ~NeuralVoiceStream();

    /** Message thread (or before playback). */
    void prepare (double hostSampleRate, int maxBlockSize);
    /** Message thread. nullptr disables the stream. Safe to call while audio is running. */
    void setModel (std::shared_ptr<NeuralVoiceModel> model);

    bool hasModel() const noexcept;
    /** Fixed latency in host-rate samples, valid after prepare(). */
    int getLatencySamples() const noexcept;
    /** True when the worker keeps up with real time (false while warming up or overloaded). */
    bool isRunningInRealTime() const noexcept;
    /** Chunks live playback could not convert in time (so far). Offline renders never miss any. */
    uint32_t getNumMissedChunks() const noexcept;

    /** Any thread. The host renders offline (export, bounce, render in place): process() then waits for the
        worker instead of skipping what it could not convert in time, and the worker uses the full context. */
    void setNonRealtime (bool isNonRealtime) noexcept;

    /** Audio thread. `targetF0Hz` is the desired output pitch per input sample (0 = unvoiced).
        Writes numSamples of converted audio (silence until the first chunk is ready). `coverage` (optional)
        receives how much of each output sample is converted voice (0..1): 0 where the stream has nothing
        (warming up, after a reset, a chunk that was not converted in time), in between during its crossfades.
        Mixing another rendering of the same voice in with gain 1 - coverage fills those gaps seamlessly.
        `pitchShiftSemis` (optional, per input sample, NaN = use `targetF0Hz`): with the neural pitch tracker
        installed, the voice sings the input's pitch as that tracker hears it, shifted by these semitones. */
    void process (const float* input, const float* targetF0Hz, float* output, int numSamples,
                  float* coverage = nullptr, const float* pitchShiftSemis = nullptr) noexcept;

    /** Any thread. The Southern "ah" of the country voices (SouthernVowels): an "I" that glides to "ee" keeps its
        open vowel. Off by default. */
    void setSouthernVowels (bool shouldBeOn) noexcept;

    /** Audio thread. Clears all buffered audio (e.g. on transport jumps). */
    void reset() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace voxa::neural
