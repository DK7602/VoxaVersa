#pragma once

#include "NeuralRuntime.h"
#include "NeuralDsp.h"
#include "NeuralPitch.h"

#include <functional>

namespace voxa::neural
{

/** Fixed geometry of the real-time stream (NeuralVoiceStream). The resulting latency depends only on these
    constants, the host sample rate and the host block size - never on the model. */
namespace streaming
{
    inline constexpr int blockFrames = 40;                // new audio per chunk: 0.4 s
    inline constexpr int tailFrames = 9;                  // crossfade + SOLA search + resampler margin
    inline constexpr int generatorContextFrames = 20;     // left context the generator gets (the encoder gets more)
    inline constexpr double crossfadeSeconds = 0.06;
    inline constexpr double searchSeconds = 0.0125;       // SOLA search (>= one period of an 80 Hz voice)
    inline constexpr double headroomSeconds = 0.35;       // inference time budget per chunk (a new chunk starts
                                                          // every 0.4 s, so up to ~0.35 s keeps up; the host
                                                          // compensates the latency it adds during playback)
    inline constexpr double pollSeconds = 0.002;          // worker polling interval

    /** Live playback uses the longest left context whose chunk inference takes at most this fraction of the
        budget (the rest is margin for the host's own load); offline renders always use the configured one. */
    inline constexpr double liveLoadTarget = 0.5;
    inline constexpr int liveContextCandidatesMs[] = { 600, 300, 150 };
}

/** Neural settings of a voice pack. The required fields come from VoiceProfile (parsed by VoiceLibrary);
    the optional ones are read from the "model" object of the pack's voice.json:

        "model": {
            "file": "model.onnx",            // RVC generator (required)
            "encoder": "contentvec-768",     // content encoder id -> <Encoders>/<id>.onnx
            "sampleRate": 40000,             // generator output rate
            "encoderFile": "encoder.onnx",   // optional: encoder shipped inside the pack (overrides "encoder")
            "hopSize": 400,                  // optional: generator samples per 10 ms frame (default sampleRate / 100)
            "speakerId": 0,                  // optional: speaker index for multi-speaker models
            "noiseScale": 0.66666,           // optional: scale of the generator's "rnd" noise input
            "streamContextMs": 1000          // optional: left context of each real-time chunk (100..3000)
        }
*/
struct ModelConfig
{
    juce::File packDirectory, generatorFile, encoderFile;
    juce::String encoderId;
    int sampleRate = 0;
    int hopSize = 0;
    int speakerId = 0;
    float noiseScale = 0.66666f;
    int streamContextMs = 1000;   // 0.3 s measurably weakened the timbre; 1 s is close to offline quality
    bool allowAccelerator = true; // false: CPU only, whatever the runtime's accelerator

    static ModelConfig fromProfile (const VoiceProfile&);

    /** Encoder files to try, in order. `featureChannels` (from the generator) picks a default id. */
    juce::Array<juce::File> getEncoderCandidates (int featureChannels) const;
};

//==============================================================================
struct NeuralVoiceModel::Impl
{
    /** Scratch buffers for one caller of runFrames() (models are shared, workspaces are not). */
    struct Workspace
    {
        std::vector<float> phone, pitchf, noise, audio;
        std::vector<int64_t> pitch;
        std::vector<int32_t> pitch32;
        std::vector<uint16_t> encoderHalf, phoneHalf, pitchfHalf, noiseHalf;
        std::vector<uint8_t> mask;
        OutputTensor encoderOut, generatorOut;
        NeuralPitchTracker::Workspace tracker;
        std::vector<float> tracked;
    };

    bool initialise (const ModelConfig&, juce::String& error);

    /** The core of the pipeline. `audio16k` holds numFrames * 160 + dsp::encoderRightPad16k samples at
        16 kHz, `f0Frames` the target pitch of each of the numFrames 10 ms frames. The content encoder sees
        all frames, the generator only frames [generatorSkip, numFrames). On success ws.audio holds
        (numFrames - generatorSkip) * hopSize samples at the model rate, starting at frame generatorSkip.
        `firstFrame` is the absolute index of the first frame (keeps the noise consistent across
        overlapping segments); `offline` releases ONNX Runtime's arena memory after big runs. */
    bool runFrames (const float* audio16k, int numFrames, const float* f0Frames, int64_t firstFrame,
                    Workspace& ws, bool offline, juce::String& error, int generatorSkip = 0) const;

    bool convert (const juce::AudioBuffer<float>& input, double inputSampleRate,
                  const std::vector<float>& f0Hz, double f0FrameRate,
                  juce::AudioBuffer<float>& output, const std::atomic<bool>* cancel, juce::String& error,
                  const std::vector<float>* pitchShiftSemis = nullptr) const;

    /** With the neural pitch tracker: the target pitch of frames [firstFrame, numFrames) becomes the pitch the
        tracker hears in `audio16k` (numFrames * 160 samples), shifted by `shiftSemis` (one per frame; NaN keeps
        the frame's target). Frames before firstFrame only serve as context. */
    bool applyNeuralPitch (const float* audio16k, int numFrames, int firstFrame, const float* shiftSemis,
                           float* f0Frames, Workspace& ws, juce::String& error) const;

    /** Left context (in 10 ms frames) of each real-time chunk, as configured (the best quality; used while
        the host renders offline). */
    int getStreamContextFrames() const noexcept;
    /** Frames processed per real-time chunk (context + tail + new audio, even). */
    int getStreamChunkFrames() const noexcept     { return chunkFramesFor (getStreamContextFrames()); }
    /** Leading frames of each real-time chunk that only the content encoder sees. */
    int getStreamGeneratorSkip() const noexcept   { return generatorSkipFor (getStreamContextFrames()); }

    static int chunkFramesFor (int contextFrames) noexcept;
    static int generatorSkipFor (int contextFrames) noexcept;

    /** Left contexts live playback may use, longest first, all primed at load time: [0] is the longest one
        this computer converts comfortably in real time, the shorter ones are fallbacks if the host gets busier. */
    const std::vector<int>& getLiveContextFrames() const noexcept   { return liveContexts; }

    /** Index of the first (longest) candidate whose measured chunk time fits `budgetSeconds`, else the last
        one. Measures lazily, in order, and stops at the first fit. */
    static size_t chooseLiveContext (size_t numCandidates, const std::function<double (size_t)>& measureSeconds,
                                     double budgetSeconds);

    //==============================================================================
    ModelConfig config;
    std::shared_ptr<OrtModelSession> encoder;     // shared between voices using the same encoder file
    std::unique_ptr<OrtModelSession> generator;
    std::shared_ptr<NeuralPitchTracker> pitchTracker;   // optional (<Encoders>/fcpe.onnx)

    int sampleRate = 0, hopSize = 0, featureChannels = 0, noiseChannels = 192;
    bool usesF0 = false;
    double chunkInferenceSeconds = 0.0;   // inference time of one live chunk (liveContexts[0]), measured at load time
    std::vector<int> liveContexts;
    juce::String description;

    /** Inference time of a live chunk relative to the stream's time budget (streaming::headroomSeconds).
        Above ~0.8 this computer can't run the voice live (offline renders and conversion still work). */
    double getRealTimeLoad() const noexcept;
    /** True when the generator or the encoder runs on an accelerator (GPU). */
    bool usesAccelerator() const noexcept;

    struct EncoderIO
    {
        enum class Layout { unknown, timeMajor, channelMajor };
        std::string input, mask, output;
        ONNXTensorElementDataType inputType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        ONNXTensorElementDataType maskType = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        int inputRank = 3;
        Layout layout = Layout::unknown;
    } enc;

    struct GeneratorIO
    {
        std::string phone, lengths, pitch, pitchf, speaker, noise, output;
        ONNXTensorElementDataType phoneType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                  pitchfType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                  noiseType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                  pitchType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                  lengthsType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                  speakerType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } gen;

private:
    bool discoverGenerator (juce::String& error);
    bool discoverEncoder (juce::String& error);
    bool warmUp (juce::String& error);
};

} // namespace voxa::neural
