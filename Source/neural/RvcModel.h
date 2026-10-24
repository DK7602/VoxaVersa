#pragma once

#include "NeuralRuntime.h"
#include "NeuralDsp.h"

namespace voxa::neural
{

/** Fixed geometry of the real-time stream (NeuralVoiceStream). The resulting latency depends only on these
    constants, the host sample rate and the host block size - never on the model. */
namespace streaming
{
    inline constexpr int blockFrames = 40;                // new audio per chunk: 0.4 s
    inline constexpr int tailFrames = 9;                  // crossfade + SOLA search + resampler margin
    inline constexpr double crossfadeSeconds = 0.06;
    inline constexpr double searchSeconds = 0.0125;       // SOLA search (>= one period of an 80 Hz voice)
    inline constexpr double headroomSeconds = 0.2;        // inference time budget per chunk
    inline constexpr double pollSeconds = 0.002;          // worker polling interval
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
            "streamContextMs": 300           // optional: left context of each real-time chunk (100..3000)
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
    int streamContextMs = 300;

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
    };

    bool initialise (const ModelConfig&, juce::String& error);

    /** The core of the pipeline. `audio16k` holds numFrames * 160 + dsp::encoderRightPad16k samples at
        16 kHz, `f0Frames` the target pitch of each 10 ms frame. On success ws.audio holds
        numFrames * hopSize samples at the model rate, time-aligned with the input.
        `firstFrame` is the absolute index of the first frame (keeps the noise consistent across
        overlapping segments); `offline` releases ONNX Runtime's arena memory after big runs. */
    bool runFrames (const float* audio16k, int numFrames, const float* f0Frames, int64_t firstFrame,
                    Workspace& ws, bool offline, juce::String& error) const;

    bool convert (const juce::AudioBuffer<float>& input, double inputSampleRate,
                  const std::vector<float>& f0Hz, double f0FrameRate,
                  juce::AudioBuffer<float>& output, const std::atomic<bool>* cancel, juce::String& error) const;

    /** Left context (in 10 ms frames) of each real-time chunk. */
    int getStreamContextFrames() const noexcept;
    /** Frames processed per real-time chunk (context + tail + new audio, even). */
    int getStreamChunkFrames() const noexcept;

    //==============================================================================
    ModelConfig config;
    std::shared_ptr<OrtModelSession> encoder;     // shared between voices using the same encoder file
    std::unique_ptr<OrtModelSession> generator;

    int sampleRate = 0, hopSize = 0, featureChannels = 0, noiseChannels = 192;
    bool usesF0 = false;
    double chunkInferenceSeconds = 0.0;   // inference time of one real-time chunk, measured at load time
    juce::String description;

    /** Inference time of a real-time chunk relative to the stream's time budget (streaming::headroomSeconds).
        Above ~0.8 this computer can't run the voice live (offline conversion still works). */
    double getRealTimeLoad() const noexcept;

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
