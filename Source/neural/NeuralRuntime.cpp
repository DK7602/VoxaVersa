#include "NeuralRuntime.h"
#include "NeuralPlatform.h"
#include "../voices/VoiceLibrary.h"

#include <cstring>

namespace voxa::neural
{

namespace
{
    using GetApiBaseFn = const OrtApiBase* (ORT_API_CALL*) ();

    // Exported by DirectML / CoreML builds of ONNX Runtime (not part of the OrtApi table).
    using AppendDmlFn    = OrtStatus* (ORT_API_CALL*) (OrtSessionOptions*, int deviceId);
    using AppendCoreMLFn = OrtStatus* (ORT_API_CALL*) (OrtSessionOptions*, uint32_t flags);

    /** Consumes a status: returns true on success, otherwise stores the message in `error`. */
    bool check (const OrtApi& api, OrtStatus* status, juce::String& error)
    {
        if (status == nullptr)
            return true;

        error = juce::String::fromUTF8 (api.GetErrorMessage (status));
        api.ReleaseStatus (status);
        return false;
    }

    bool succeeded (const OrtApi& api, OrtStatus* status)
    {
        juce::String ignored;
        return check (api, status, ignored);
    }

    const char* typeName (ONNXTensorElementDataType type) noexcept
    {
        switch (type)
        {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "float";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return "double";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "int64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "int32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return "bool";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "uint8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return "int8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
            default:                                    return "unsupported";
        }
    }

    juce::String emDash()
    {
        return juce::String (juce::CharPointer_UTF8 (" \xe2\x80\x94 "));
    }

    juce::String getInstallHint()
    {
        return "install ONNX Runtime " + juce::String (minimumOrtVersionString) + " or newer by copying "
                 + platform::getRuntimeLibraryNames()[0] + " into "
                 + VoiceLibrary::getUserDataDirectory().getChildFile ("Runtime").getFullPathName();
    }
} // namespace

//==============================================================================
uint16_t floatToHalf (float value) noexcept
{
    uint32_t x = 0;
    std::memcpy (&x, &value, sizeof (x));

    const auto sign = (uint32_t) ((x >> 16) & 0x8000u);
    auto mantissa = x & 0x007fffffu;
    const auto exponent = (int) ((x >> 23) & 0xffu);

    if (exponent == 0xff)
        return (uint16_t) (sign | 0x7c00u | (mantissa != 0 ? 0x200u : 0u));

    const int e = exponent - 127 + 15;

    if (e >= 0x1f)
        return (uint16_t) (sign | 0x7c00u);

    if (e <= 0)
    {
        if (e < -10)
            return (uint16_t) sign;

        mantissa |= 0x00800000u;
        const auto shift = (uint32_t) (14 - e);
        auto half = mantissa >> shift;
        const auto remainder = mantissa & ((1u << shift) - 1u);
        const auto halfway = 1u << (shift - 1u);

        if (remainder > halfway || (remainder == halfway && (half & 1u) != 0))
            ++half;

        return (uint16_t) (sign | half);
    }

    auto half = sign | ((uint32_t) e << 10) | (mantissa >> 13);
    const auto remainder = mantissa & 0x1fffu;

    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u) != 0))
        ++half;   // a carry into the exponent is the correct rounding

    return (uint16_t) half;
}

float halfToFloat (uint16_t h) noexcept
{
    const auto sign = (uint32_t) (h & 0x8000u) << 16;
    auto exponent = (uint32_t) ((h >> 10) & 0x1fu);
    auto mantissa = (uint32_t) (h & 0x3ffu);
    uint32_t bits = 0;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            exponent = 127 - 15 + 1;

            while ((mantissa & 0x400u) == 0)
            {
                mantissa <<= 1;
                --exponent;
            }

            bits = sign | (exponent << 23) | ((mantissa & 0x3ffu) << 13);
        }
    }
    else if (exponent == 0x1f)
    {
        bits = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy (&result, &bits, sizeof (result));
    return result;
}

//==============================================================================
juce::String getAcceleratorName (Accelerator a)
{
    switch (a)
    {
        case Accelerator::automatic: return "auto";
        case Accelerator::cuda:      return "CUDA";
        case Accelerator::directML:  return "DirectML";
        case Accelerator::coreML:    return "CoreML";
        case Accelerator::cpu:
        default:                     return "CPU";
    }
}

Accelerator parseAccelerator (const juce::String& text, Accelerator fallback)
{
    const auto t = text.trim().toLowerCase();

    if (t == "cpu")                           return Accelerator::cpu;
    if (t == "auto" || t == "gpu")            return Accelerator::automatic;
    if (t == "cuda")                          return Accelerator::cuda;
    if (t == "directml" || t == "dml")        return Accelerator::directML;
    if (t == "coreml")                        return Accelerator::coreML;
    return fallback;
}

//==============================================================================
int64_t TensorInfo::dim (int index) const noexcept
{
    if (index < 0 || index >= rank())
        return -1;

    const auto d = dims[(size_t) index];
    return d > 0 ? d : -1;
}

bool TensorInfo::isFloat() const noexcept
{
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

juce::String TensorInfo::describe() const
{
    juce::StringArray d;
    for (auto v : dims)
        d.add (v > 0 ? juce::String (v) : juce::String ("?"));

    return juce::String (name) + " " + typeName (type) + "[" + d.joinIntoString (",") + "]";
}

//==============================================================================
OrtModelSession::~OrtModelSession()
{
    if (api != nullptr)
    {
        if (session != nullptr)
            api->ReleaseSession (session);

        if (memoryInfo != nullptr)
            api->ReleaseMemoryInfo (memoryInfo);
    }
}

std::unique_ptr<OrtModelSession> OrtModelSession::create (const juce::File& modelFile, juce::String& error,
                                                          bool allowAccelerator)
{
    auto& runtime = OnnxRuntime::getInstance().getImpl();

    if (! runtime.isAvailable())
    {
        error = runtime.getStatusMessage();
        return nullptr;
    }

    if (! modelFile.existsAsFile())
    {
        error = "Model file not found: " + modelFile.getFullPathName();
        return nullptr;
    }

    const auto& api = *runtime.getApi();
    std::unique_ptr<OrtModelSession> result (new OrtModelSession());
    result->api = &api;
    result->file = modelFile;

    const auto path = modelFile.getFullPathName();
   #if JUCE_WINDOWS
    const auto* ortPath = path.toWideCharPointer();
   #else
    const auto* ortPath = path.toRawUTF8();
   #endif

    // Preferred configuration first; then without the accelerator, then with per-session thread pools
    // (needed if another module in this process created the OrtEnv without global thread pools).
    struct Attempt { bool accelerator, globalPool; };
    const bool wantAccelerator = allowAccelerator && runtime.getAccelerator() != Accelerator::cpu;
    std::vector<Attempt> attempts;

    if (wantAccelerator)
        attempts.push_back ({ true, runtime.hasGlobalThreadPool() });

    attempts.push_back ({ false, runtime.hasGlobalThreadPool() });

    if (runtime.hasGlobalThreadPool())
        attempts.push_back ({ false, false });

    juce::String firstError;

    for (const auto& attempt : attempts)
    {
        juce::String acceleratorUsed, attemptError;
        auto* options = runtime.createSessionOptions (attempt.accelerator, attempt.globalPool, acceleratorUsed, attemptError);

        if (options == nullptr)
        {
            if (firstError.isEmpty())
                firstError = attemptError;
            continue;
        }

        OrtSession* session = nullptr;
        const bool ok = check (api, api.CreateSession (runtime.getEnv(), ortPath, options, &session), attemptError);
        api.ReleaseSessionOptions (options);

        if (ok && session != nullptr)
        {
            result->session = session;
            result->provider = acceleratorUsed;
            break;
        }

        if (firstError.isEmpty())
            firstError = attemptError;

        if (attempt.accelerator && acceleratorUsed == "CPU")
            continue;   // nothing was appended, so the next attempt would fail the same way - still try
    }

    if (result->session == nullptr)
    {
        error = "Could not load " + modelFile.getFileName() + ": " + firstError;
        return nullptr;
    }

    if (! check (api, api.CreateCpuMemoryInfo (OrtArenaAllocator, OrtMemTypeDefault, &result->memoryInfo), error)
        || ! result->discoverIO (error))
    {
        error = modelFile.getFileName() + ": " + error;
        return nullptr;
    }

    return result;
}

bool OrtModelSession::discoverIO (juce::String& error)
{
    OrtAllocator* allocator = nullptr;
    if (! check (*api, api->GetAllocatorWithDefaultOptions (&allocator), error))
        return false;

    auto readTensorInfo = [this, &error] (OrtTypeInfo* typeInfo, TensorInfo& info)
    {
        ONNXType onnxType = ONNX_TYPE_UNKNOWN;
        bool ok = check (*api, api->GetOnnxTypeFromTypeInfo (typeInfo, &onnxType), error);

        if (ok && onnxType == ONNX_TYPE_TENSOR)
        {
            const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
            size_t count = 0;
            ok = check (*api, api->CastTypeInfoToTensorInfo (typeInfo, &tensorInfo), error)
                 && check (*api, api->GetTensorElementType (tensorInfo, &info.type), error)
                 && check (*api, api->GetDimensionsCount (tensorInfo, &count), error);

            if (ok)
            {
                info.dims.resize (count);
                ok = count == 0 || check (*api, api->GetDimensions (tensorInfo, info.dims.data(), count), error);
            }
        }

        api->ReleaseTypeInfo (typeInfo);
        return ok;
    };

    for (int pass = 0; pass < 2; ++pass)
    {
        const bool isInput = pass == 0;
        size_t count = 0;

        if (! check (*api, isInput ? api->SessionGetInputCount (session, &count)
                                   : api->SessionGetOutputCount (session, &count), error))
            return false;

        auto& list = isInput ? inputs : outputs;

        for (size_t i = 0; i < count; ++i)
        {
            TensorInfo info;
            char* name = nullptr;

            if (! check (*api, isInput ? api->SessionGetInputName (session, i, allocator, &name)
                                       : api->SessionGetOutputName (session, i, allocator, &name), error))
                return false;

            info.name = name != nullptr ? name : "";
            juce::ignoreUnused (succeeded (*api, api->AllocatorFree (allocator, name)));

            OrtTypeInfo* typeInfo = nullptr;
            if (! check (*api, isInput ? api->SessionGetInputTypeInfo (session, i, &typeInfo)
                                       : api->SessionGetOutputTypeInfo (session, i, &typeInfo), error)
                || ! readTensorInfo (typeInfo, info))
                return false;

            list.push_back (std::move (info));
        }
    }

    if (inputs.empty() || outputs.empty())
    {
        error = "the model has no inputs or outputs";
        return false;
    }

    return true;
}

const TensorInfo* OrtModelSession::findInput (std::initializer_list<const char*> candidateNames) const noexcept
{
    for (auto* candidate : candidateNames)
        for (auto& info : inputs)
            if (juce::String (info.name).equalsIgnoreCase (candidate))
                return &info;

    return nullptr;
}

juce::String OrtModelSession::getCustomMetadata (const char* key) const
{
    juce::String result;
    OrtModelMetadata* metadata = nullptr;
    OrtAllocator* allocator = nullptr;

    if (succeeded (*api, api->GetAllocatorWithDefaultOptions (&allocator))
        && succeeded (*api, api->SessionGetModelMetadata (session, &metadata)))
    {
        char* value = nullptr;

        if (succeeded (*api, api->ModelMetadataLookupCustomMetadataMap (metadata, allocator, key, &value)) && value != nullptr)
        {
            result = juce::String::fromUTF8 (value);
            juce::ignoreUnused (succeeded (*api, api->AllocatorFree (allocator, value)));
        }

        api->ReleaseModelMetadata (metadata);
    }

    return result;
}

bool OrtModelSession::run (const InputTensor* inputTensors, size_t numInputs,
                           const char* const* outputNames, OutputTensor* outputTensors, size_t numOutputs,
                           bool shrinkArena, juce::String& error) const
{
    std::unique_lock<std::mutex> serialised (runLock, std::defer_lock);
    if (provider != "CPU")
        serialised.lock();

    constexpr size_t maxTensors = 16;

    if (numInputs > maxTensors || numOutputs > maxTensors)
    {
        error = "too many tensors";
        return false;
    }

    std::array<OrtValue*, maxTensors> inputValues {};
    std::array<const char*, maxTensors> inputNames {};
    std::array<OrtValue*, maxTensors> outputValues {};
    bool ok = true;

    for (size_t i = 0; i < numInputs && ok; ++i)
    {
        const auto& t = inputTensors[i];
        inputNames[i] = t.name;
        ok = check (*api, api->CreateTensorWithDataAsOrtValue (memoryInfo, const_cast<void*> (t.data), t.numBytes,
                                                               t.shape.data(), t.rank, t.type, &inputValues[i]), error);
    }

    OrtRunOptions* runOptions = nullptr;

    if (ok && shrinkArena && succeeded (*api, api->CreateRunOptions (&runOptions)))
        juce::ignoreUnused (succeeded (*api, api->AddRunConfigEntry (runOptions, "memory.enable_memory_arena_shrinkage", "cpu:0")));

    if (ok)
        ok = check (*api, api->Run (session, runOptions, inputNames.data(), inputValues.data(), numInputs,
                                    outputNames, numOutputs, outputValues.data()), error);

    if (runOptions != nullptr)
        api->ReleaseRunOptions (runOptions);

    for (size_t i = 0; i < numInputs; ++i)
        if (inputValues[i] != nullptr)
            api->ReleaseValue (inputValues[i]);

    for (size_t i = 0; i < numOutputs; ++i)
    {
        auto* value = outputValues[i];

        if (ok && value != nullptr)
        {
            OrtTensorTypeAndShapeInfo* info = nullptr;
            ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            size_t rank = 0, count = 0;
            void* data = nullptr;

            ok = check (*api, api->GetTensorTypeAndShape (value, &info), error);

            if (ok)
            {
                ok = check (*api, api->GetTensorElementType (info, &type), error)
                     && check (*api, api->GetDimensionsCount (info, &rank), error)
                     && check (*api, api->GetTensorShapeElementCount (info, &count), error);

                if (ok)
                {
                    outputTensors[i].shape.resize (rank);
                    ok = rank == 0 || check (*api, api->GetDimensions (info, outputTensors[i].shape.data(), rank), error);
                }

                api->ReleaseTensorTypeAndShapeInfo (info);
            }

            if (ok)
                ok = check (*api, api->GetTensorMutableData (value, &data), error);

            if (ok)
            {
                auto& out = outputTensors[i].data;
                out.resize (count);

                if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                {
                    if (count > 0)
                        std::memcpy (out.data(), data, count * sizeof (float));
                }
                else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                {
                    const auto* h = static_cast<const uint16_t*> (data);
                    for (size_t j = 0; j < count; ++j)
                        out[j] = halfToFloat (h[j]);
                }
                else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
                {
                    const auto* d = static_cast<const double*> (data);
                    for (size_t j = 0; j < count; ++j)
                        out[j] = (float) d[j];
                }
                else
                {
                    error = juce::String ("unsupported output type ") + typeName (type);
                    ok = false;
                }
            }
        }

        if (value != nullptr)
            api->ReleaseValue (value);
    }

    return ok;
}

//==============================================================================
OnnxRuntime::Impl::Impl()
{
    numThreads = juce::jlimit (1, 4, juce::SystemStats::getNumPhysicalCpus() / 2);
    status = "AI engine not initialised";
}

OnnxRuntime::Impl::~Impl()
{
    // Deliberately leaks the OrtEnv and never unloads the library: at process exit ONNX Runtime's own
    // static objects may already have been destroyed, and releasing the environment then can crash.
}

void OnnxRuntime::Impl::readSettings()
{
    if (settingsRead)
        return;

    settingsRead = true;

    if (auto xml = juce::parseXML (VoiceLibrary::getUserDataDirectory().getChildFile ("settings.xml")))
    {
        if (auto* e = xml->getChildByName ("NeuralEngine"))
        {
            accelerator = parseAccelerator (e->getStringAttribute ("accelerator"), accelerator);

            if (e->hasAttribute ("threads"))
                numThreads = juce::jlimit (1, 16, e->getIntAttribute ("threads", numThreads));
        }
    }

    const auto envAccelerator = juce::SystemStats::getEnvironmentVariable ("VOXAVERSA_AI_ACCELERATOR", {});
    accelerator = parseAccelerator (envAccelerator, accelerator);

    const auto envThreads = juce::SystemStats::getEnvironmentVariable ("VOXAVERSA_AI_THREADS", {});
    if (envThreads.getIntValue() > 0)
        numThreads = juce::jlimit (1, 16, envThreads.getIntValue());
}

bool OnnxRuntime::Impl::ensureLoaded()
{
    if (isAvailable())
        return true;

    const std::lock_guard<std::mutex> sl (lock);

    if (isAvailable())
        return true;

    readSettings();

    juce::String firstError;
    auto attempt = [this, &firstError] (const juce::String& pathOrName)
    {
        juce::String error;

        if (tryLoad (pathOrName, error))
            return true;

        if (firstError.isEmpty() && error.isNotEmpty())
            firstError = error;

        return false;
    };

    // 1. Explicit override
    const auto overridePath = juce::SystemStats::getEnvironmentVariable ("VOXAVERSA_ONNXRUNTIME_PATH", {});

    if (overridePath.isNotEmpty())
    {
        const juce::File f (overridePath);

        if (f.existsAsFile() && attempt (f.getFullPathName()))
            return true;

        if (f.isDirectory())
            for (auto& name : platform::getRuntimeLibraryNames())
                if (f.getChildFile (name).existsAsFile() && attempt (f.getChildFile (name).getFullPathName()))
                    return true;
    }

    // 2. Known directories (next to the binary, user data, common install prefixes)
    for (auto& dir : getSearchLocations())
    {
        if (! dir.isDirectory())
            continue;

        juce::Array<juce::File> candidates;

        for (auto& name : platform::getRuntimeLibraryNames())
            candidates.add (dir.getChildFile (name));

       #if JUCE_MAC
        candidates.addArray (dir.findChildFiles (juce::File::findFiles, false, "libonnxruntime.*.dylib"));
       #elif JUCE_LINUX || JUCE_BSD
        candidates.addArray (dir.findChildFiles (juce::File::findFiles, false, "libonnxruntime.so.*"));
       #endif

        for (auto& f : candidates)
            if (f.existsAsFile() && attempt (f.getFullPathName()))
                return true;
    }

    // 3. Whatever the system loader finds (PATH / LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / ld.so.cache)
    juce::String systemError;
    for (auto& name : platform::getRuntimeLibraryNames())
    {
        if (tryLoad (name, systemError))
            return true;

        // a library that was found but rejected (e.g. too old) is worth reporting
        if (firstError.isEmpty() && systemError.contains ("too old"))
            firstError = systemError;
    }

    status = (firstError.isNotEmpty() ? firstError : juce::String ("ONNX Runtime not found")) + emDash() + getInstallHint();
    return false;
}

bool OnnxRuntime::Impl::loadFromFile (const juce::File& file)
{
    if (isAvailable())
        return true;

    const std::lock_guard<std::mutex> sl (lock);

    if (isAvailable())
        return true;

    readSettings();

    juce::String error;
    if (tryLoad (file.getFullPathName(), error))
        return true;

    status = error;
    return false;
}

bool OnnxRuntime::Impl::tryLoad (const juce::String& pathOrName, juce::String& error)
{
    auto* handle = platform::openLibrary (pathOrName, error);

    if (handle == nullptr)
    {
        error = "Could not load ONNX Runtime from " + pathOrName + ": " + error;
        return false;
    }

    auto getApiBase = reinterpret_cast<GetApiBaseFn> (platform::getSymbol (handle, "OrtGetApiBase"));

    if (getApiBase == nullptr)
    {
        error = pathOrName + " is not an ONNX Runtime library";
        return false;
    }

    const auto* base = getApiBase();
    const auto libFile = platform::getLibraryFile (handle, reinterpret_cast<void*> (getApiBase));
    const auto versionString = base != nullptr && base->GetVersionString != nullptr
                                 ? juce::String::fromUTF8 (base->GetVersionString()) : juce::String ("?");
    const auto* ortApi = base != nullptr ? base->GetApi (requiredOrtApiVersion) : nullptr;

    if (ortApi == nullptr)
    {
        error = "ONNX Runtime " + versionString + " at " + (libFile != juce::File() ? libFile.getFullPathName() : pathOrName)
                  + " is too old (VoxaVersa needs " + minimumOrtVersionString + " or newer)";
        return false;
    }

    // Environment with one global (non-spinning) thread pool shared by every session in the process, so
    // several plugin instances don't multiply the number of busy threads.
    OrtEnv* newEnv = nullptr;
    bool global = false;
    OrtThreadingOptions* threading = nullptr;

    if (succeeded (*ortApi, ortApi->CreateThreadingOptions (&threading)))
    {
        juce::ignoreUnused (succeeded (*ortApi, ortApi->SetGlobalIntraOpNumThreads (threading, numThreads)),
                            succeeded (*ortApi, ortApi->SetGlobalInterOpNumThreads (threading, 1)),
                            succeeded (*ortApi, ortApi->SetGlobalSpinControl (threading, 0)),
                            succeeded (*ortApi, ortApi->SetGlobalDenormalAsZero (threading)));

        global = succeeded (*ortApi, ortApi->CreateEnvWithGlobalThreadPools (ORT_LOGGING_LEVEL_ERROR, "VoxaVersa", threading, &newEnv));
        ortApi->ReleaseThreadingOptions (threading);
    }

    if (newEnv == nullptr)
    {
        global = false;

        if (! check (*ortApi, ortApi->CreateEnv (ORT_LOGGING_LEVEL_ERROR, "VoxaVersa", &newEnv), error))
        {
            error = "ONNX Runtime " + versionString + ": " + error;
            return false;
        }
    }

    juce::ignoreUnused (succeeded (*ortApi, ortApi->DisableTelemetryEvents (newEnv)));

    char** providers = nullptr;
    int numProviders = 0;
    availableProviders.clear();

    if (succeeded (*ortApi, ortApi->GetAvailableProviders (&providers, &numProviders)))
    {
        for (int i = 0; i < numProviders; ++i)
            availableProviders.add (juce::String::fromUTF8 (providers[i]));

        juce::ignoreUnused (succeeded (*ortApi, ortApi->ReleaseAvailableProviders (providers, numProviders)));
    }

    library = handle;
    api = ortApi;
    env = newEnv;
    globalThreadPool = global;
    version = versionString;
    libraryFile = libFile != juce::File() ? libFile : juce::File (pathOrName);
    status = describeReady();
    available.store (true, std::memory_order_release);
    return true;
}

juce::String OnnxRuntime::Impl::describeReady() const
{
    juce::String device = "CPU";

    auto has = [this] (const char* name) { return availableProviders.contains (name); };

    switch (accelerator)
    {
        case Accelerator::cuda:     if (has ("CUDAExecutionProvider"))   device = "CUDA"; break;
        case Accelerator::directML: if (has ("DmlExecutionProvider"))    device = "DirectML"; break;
        case Accelerator::coreML:   if (has ("CoreMLExecutionProvider")) device = "CoreML"; break;
        case Accelerator::automatic:
            if (has ("CoreMLExecutionProvider"))     device = "CoreML";
            else if (has ("DmlExecutionProvider"))   device = "DirectML";
            else if (has ("CUDAExecutionProvider"))  device = "CUDA";
            break;
        case Accelerator::cpu:
        default:
            break;
    }

    return "AI engine ready (ONNX Runtime " + version + ", " + device + ")";
}

juce::String OnnxRuntime::Impl::getStatusMessage() const
{
    const std::lock_guard<std::mutex> sl (lock);
    return status;
}

juce::File OnnxRuntime::Impl::getLibraryFile() const
{
    const std::lock_guard<std::mutex> sl (lock);
    return libraryFile;
}

juce::String OnnxRuntime::Impl::getVersion() const
{
    const std::lock_guard<std::mutex> sl (lock);
    return version;
}

void OnnxRuntime::Impl::setAccelerator (Accelerator a)
{
    const std::lock_guard<std::mutex> sl (lock);
    settingsRead = true;
    accelerator = a;

    if (isAvailable())
        status = describeReady();
}

void OnnxRuntime::Impl::setAndSaveAccelerator (Accelerator a)
{
    setAccelerator (a);

    const auto file = VoiceLibrary::getUserDataDirectory().getChildFile ("settings.xml");
    std::unique_ptr<juce::XmlElement> xml (juce::parseXML (file));
    if (xml == nullptr || ! xml->hasTagName ("VoxaVersaSettings"))
        xml = std::make_unique<juce::XmlElement> ("VoxaVersaSettings");

    auto* e = xml->getChildByName ("NeuralEngine");
    if (e == nullptr)
        e = xml->createNewChildElement ("NeuralEngine");

    e->setAttribute ("accelerator", a == Accelerator::automatic ? juce::String ("auto") : getAcceleratorName (a).toLowerCase());
    file.getParentDirectory().createDirectory();
    xml->writeTo (file);
}

Accelerator OnnxRuntime::Impl::getAccelerator() const
{
    const std::lock_guard<std::mutex> sl (lock);
    return accelerator;
}

bool OnnxRuntime::Impl::isProviderAvailable (const char* providerName) const
{
    const std::lock_guard<std::mutex> sl (lock);
    return availableProviders.contains (providerName);
}

bool OnnxRuntime::Impl::appendAccelerator (OrtSessionOptions* options, Accelerator which, juce::String& name) const
{
    auto has = [this] (const char* provider) { return availableProviders.contains (provider); };

    switch (which)
    {
        case Accelerator::cuda:
        {
            if (! has ("CUDAExecutionProvider"))
                return false;

            OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
            if (! succeeded (*api, api->CreateCUDAProviderOptions (&cudaOptions)))
                return false;

            const bool ok = succeeded (*api, api->SessionOptionsAppendExecutionProvider_CUDA_V2 (options, cudaOptions));
            api->ReleaseCUDAProviderOptions (cudaOptions);

            if (ok)
                name = "CUDA";

            return ok;
        }

        case Accelerator::directML:
        {
            auto fn = reinterpret_cast<AppendDmlFn> (platform::getSymbol (library, "OrtSessionOptionsAppendExecutionProvider_DML"));

            if (fn == nullptr || ! has ("DmlExecutionProvider"))
                return false;

            // DirectML requires sequential execution without memory patterns.
            juce::ignoreUnused (succeeded (*api, api->DisableMemPattern (options)));

            if (! succeeded (*api, fn (options, 0)))
                return false;

            name = "DirectML";
            return true;
        }

        case Accelerator::coreML:
        {
            if (! has ("CoreMLExecutionProvider"))
                return false;

            if (succeeded (*api, api->SessionOptionsAppendExecutionProvider (options, "CoreML", nullptr, nullptr, 0)))
            {
                name = "CoreML";
                return true;
            }

            auto fn = reinterpret_cast<AppendCoreMLFn> (platform::getSymbol (library, "OrtSessionOptionsAppendExecutionProvider_CoreML"));

            if (fn != nullptr && succeeded (*api, fn (options, 0)))
            {
                name = "CoreML";
                return true;
            }

            return false;
        }

        case Accelerator::automatic:
           #if JUCE_MAC
            return appendAccelerator (options, Accelerator::coreML, name);
           #elif JUCE_WINDOWS
            return appendAccelerator (options, Accelerator::directML, name)
                || appendAccelerator (options, Accelerator::cuda, name);
           #else
            return appendAccelerator (options, Accelerator::cuda, name);
           #endif

        case Accelerator::cpu:
        default:
            return false;
    }
}

OrtSessionOptions* OnnxRuntime::Impl::createSessionOptions (bool tryAccelerator, bool useGlobalThreadPool,
                                                           juce::String& acceleratorUsed, juce::String& error) const
{
    acceleratorUsed = "CPU";

    if (! isAvailable())
    {
        error = "ONNX Runtime is not loaded";
        return nullptr;
    }

    OrtSessionOptions* options = nullptr;
    if (! check (*api, api->CreateSessionOptions (&options), error))
        return nullptr;

    juce::ignoreUnused (succeeded (*api, api->SetSessionGraphOptimizationLevel (options, ORT_ENABLE_ALL)),
                        succeeded (*api, api->SetSessionExecutionMode (options, ORT_SEQUENTIAL)),
                        succeeded (*api, api->AddSessionConfigEntry (options, "session.set_denormal_as_zero", "1")));

    if (useGlobalThreadPool)
    {
        juce::ignoreUnused (succeeded (*api, api->DisablePerSessionThreads (options)));
    }
    else
    {
        juce::ignoreUnused (succeeded (*api, api->SetIntraOpNumThreads (options, numThreads)),
                            succeeded (*api, api->SetInterOpNumThreads (options, 1)),
                            succeeded (*api, api->AddSessionConfigEntry (options, "session.intra_op.allow_spinning", "0")),
                            succeeded (*api, api->AddSessionConfigEntry (options, "session.inter_op.allow_spinning", "0")));
    }

    if (tryAccelerator)
    {
        Accelerator which = Accelerator::cpu;
        {
            const std::lock_guard<std::mutex> sl (lock);
            which = accelerator;
        }

        appendAccelerator (options, which, acceleratorUsed);
    }

    return options;
}

std::shared_ptr<OrtModelSession> OnnxRuntime::Impl::getSharedSession (const juce::File& modelFile, juce::String& error,
                                                                      bool allowAccelerator)
{
    const auto key = modelFile.getFullPathName() + "|" + juce::String (modelFile.getSize())
                       + "|" + juce::String (modelFile.getLastModificationTime().toMilliseconds())
                       + "|" + getAcceleratorName (allowAccelerator ? getAccelerator() : Accelerator::cpu);

    const std::lock_guard<std::mutex> sl (sharedSessionLock);

    if (auto existing = sharedSessions[key].lock())
        return existing;

    std::shared_ptr<OrtModelSession> session (OrtModelSession::create (modelFile, error, allowAccelerator));

    if (session != nullptr)
        sharedSessions[key] = session;

    for (auto it = sharedSessions.begin(); it != sharedSessions.end();)
        it = it->second.expired() ? sharedSessions.erase (it) : std::next (it);

    return session;
}

//==============================================================================
OnnxRuntime::OnnxRuntime() : impl (std::make_unique<Impl>()) {}
OnnxRuntime::~OnnxRuntime() = default;

OnnxRuntime& OnnxRuntime::getInstance()
{
    static OnnxRuntime instance;
    return instance;
}

bool OnnxRuntime::ensureLoaded()                    { return impl->ensureLoaded(); }
bool OnnxRuntime::isAvailable() const noexcept      { return impl->isAvailable(); }
juce::String OnnxRuntime::getStatusMessage() const  { return impl->getStatusMessage(); }
juce::File OnnxRuntime::getLoadedLibraryFile() const { return impl->getLibraryFile(); }

juce::Array<juce::File> OnnxRuntime::getSearchLocations()
{
    juce::Array<juce::File> dirs;
    const auto binaryDir = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory();
    dirs.add (binaryDir);

   #if JUCE_MAC
    // .../Foo.vst3/Contents/MacOS/Foo -> Contents/Frameworks, Contents/Resources
    if (binaryDir.getFileName() == "MacOS")
    {
        const auto contents = binaryDir.getParentDirectory();
        dirs.add (contents.getChildFile ("Frameworks"));
        dirs.add (contents.getChildFile ("Resources"));
    }
   #endif

    const auto userRuntime = VoiceLibrary::getUserDataDirectory().getChildFile ("Runtime");
    dirs.add (userRuntime);
    dirs.add (userRuntime.getChildFile ("lib"));   // an extracted release archive's layout

   #if JUCE_MAC
    dirs.add (juce::File ("/opt/homebrew/lib"));
    dirs.add (juce::File ("/usr/local/lib"));
   #elif JUCE_LINUX || JUCE_BSD
    dirs.add (juce::File ("/usr/local/lib"));
    dirs.add (juce::File ("/opt/onnxruntime/lib"));
   #endif

    return dirs;
}

} // namespace voxa::neural
