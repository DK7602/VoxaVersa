#pragma once

#include "NeuralVoice.h"
#include "OrtCApi.h"

#include <array>
#include <map>
#include <mutex>
#include <string>

namespace voxa::neural
{

/** Execution provider preference. CPU is the default; the others are tried when the loaded runtime build
    supports them and fall back to CPU silently. */
enum class Accelerator { cpu, automatic, cuda, directML, coreML };

juce::String getAcceleratorName (Accelerator);
Accelerator parseAccelerator (const juce::String&, Accelerator fallback);

//==============================================================================
/** Name, element type and (possibly dynamic, i.e. <= 0) dimensions of a model input or output. */
struct TensorInfo
{
    std::string name;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64_t> dims;

    int rank() const noexcept                 { return (int) dims.size(); }
    /** Static size of a dimension, or -1 when it is dynamic / out of range. */
    int64_t dim (int index) const noexcept;
    bool isFloat() const noexcept;            // float32 / float16 / double
    juce::String describe() const;            // e.g. "phone float[1,?,768]"
};

/** An input tensor for OrtModelSession::run(). The data is borrowed, not copied. */
struct InputTensor
{
    const char* name = nullptr;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    const void* data = nullptr;
    size_t numBytes = 0;
    std::array<int64_t, 4> shape {};
    size_t rank = 0;
};

/** An output tensor of OrtModelSession::run(), converted to float. */
struct OutputTensor
{
    std::vector<float> data;
    std::vector<int64_t> shape;
};

//==============================================================================
/** One loaded ONNX model (an OrtSession) with its discovered inputs/outputs. Immutable after creation;
    run() may be called from several threads at once. */
class OrtModelSession
{
public:
    ~OrtModelSession();

    /** Loads a model file. Requires OnnxRuntime::getInstance().isAvailable(). With `allowAccelerator` the
        runtime's accelerator (GPU) is tried first, falling back to the CPU if it can't load the model. */
    static std::unique_ptr<OrtModelSession> create (const juce::File& modelFile, juce::String& error,
                                                    bool allowAccelerator = true);

    const std::vector<TensorInfo>& getInputs() const noexcept   { return inputs; }
    const std::vector<TensorInfo>& getOutputs() const noexcept  { return outputs; }
    const TensorInfo* findInput (std::initializer_list<const char*> candidateNames) const noexcept;
    const juce::File& getFile() const noexcept                  { return file; }
    /** "CPU", "CoreML", "DirectML" or "CUDA". */
    const juce::String& getExecutionProvider() const noexcept   { return provider; }

    /** Value of a custom metadata entry stored in the model, or an empty string. */
    juce::String getCustomMetadata (const char* key) const;

    /** Runs the model. Only float32/float16/double outputs are supported (converted to float).
        `shrinkArena` returns the memory used by this run to the system afterwards (for big offline runs). */
    bool run (const InputTensor* inputTensors, size_t numInputs,
              const char* const* outputNames, OutputTensor* outputTensors, size_t numOutputs,
              bool shrinkArena, juce::String& error) const;

private:
    OrtModelSession() = default;
    bool discoverIO (juce::String& error);

    const OrtApi* api = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memoryInfo = nullptr;
    std::vector<TensorInfo> inputs, outputs;
    juce::String provider { "CPU" };
    // GPU providers (DirectML) don't allow concurrent Run() calls on one session; shared sessions (content
    // encoders) are used by the stream worker and by a model loading or rendering at the same time.
    mutable std::mutex runLock;
    juce::File file;

    JUCE_DECLARE_NON_COPYABLE (OrtModelSession)
};

//==============================================================================
/** The dynamically loaded ONNX Runtime. Owned by the OnnxRuntime singleton. */
struct OnnxRuntime::Impl
{
    Impl();
    ~Impl();

    bool ensureLoaded();
    /** Loads the runtime from an explicit library file (used by the tests and by the
        VOXAVERSA_ONNXRUNTIME_PATH environment variable). No-op if a runtime is already loaded. */
    bool loadFromFile (const juce::File& libraryFile);

    bool isAvailable() const noexcept                { return available.load (std::memory_order_acquire); }
    /** Valid (and constant) once isAvailable() returned true. */
    const OrtApi* getApi() const noexcept            { return api; }
    OrtEnv* getEnv() const noexcept                  { return env; }

    juce::String getStatusMessage() const;
    juce::File getLibraryFile() const;
    juce::String getVersion() const;
    int getNumThreads() const noexcept               { return numThreads; }

    /** Execution provider for sessions created from now on (models already loaded keep theirs).
        Can also be set with the VOXAVERSA_AI_ACCELERATOR environment variable or the "accelerator"
        attribute of <NeuralEngine> in VoxaVersa's settings.xml (cpu / auto / cuda / directml / coreml). */
    void setAccelerator (Accelerator);
    Accelerator getAccelerator() const;
    /** setAccelerator() and remembers the choice in settings.xml for the next sessions. */
    void setAndSaveAccelerator (Accelerator);

    /** Session for a model file that may be shared between voices (content encoders are large). */
    std::shared_ptr<OrtModelSession> getSharedSession (const juce::File& modelFile, juce::String& error,
                                                       bool allowAccelerator = true);

    /** Creates session options. `acceleratorUsed` receives the provider actually appended ("CPU" if none). */
    OrtSessionOptions* createSessionOptions (bool tryAccelerator, bool useGlobalThreadPool,
                                             juce::String& acceleratorUsed, juce::String& error) const;
    bool hasGlobalThreadPool() const noexcept        { return globalThreadPool; }
    bool isProviderAvailable (const char* providerName) const;

private:
    bool tryLoad (const juce::String& pathOrName, juce::String& error);   // caller holds `lock`
    void readSettings();
    juce::String describeReady() const;
    bool appendAccelerator (OrtSessionOptions*, Accelerator, juce::String& name) const;

    mutable std::mutex lock;
    std::atomic<bool> available { false };
    bool settingsRead = false;

    void* library = nullptr;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    bool globalThreadPool = false;
    int numThreads = 2;
    // The CPU unless chosen otherwise: through DirectML the RVC generator's output had broadband noise
    // ("static") on a user's machine, so GPUs stay an explicit, experimental choice.
    Accelerator accelerator = Accelerator::cpu;
    juce::StringArray availableProviders;
    juce::String version, status;
    juce::File libraryFile;

    std::mutex sharedSessionLock;
    std::map<juce::String, std::weak_ptr<OrtModelSession>> sharedSessions;
};

//==============================================================================
/** IEEE 754 half <-> float conversion (some RVC exports use float16 tensors). */
uint16_t floatToHalf (float) noexcept;
float halfToFloat (uint16_t) noexcept;

} // namespace voxa::neural
