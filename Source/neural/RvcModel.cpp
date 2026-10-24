#include "RvcModel.h"
#include "../voices/VoiceLibrary.h"

#include <cmath>
#include <limits>

namespace voxa::neural
{

namespace
{
    void setShape (InputTensor& t, std::initializer_list<int64_t> shape)
    {
        t.rank = 0;
        for (auto d : shape)
            t.shape[t.rank++] = d;
    }

    void setFloatTensor (InputTensor& t, const std::string& name, ONNXTensorElementDataType type,
                         const float* data, size_t count, std::vector<uint16_t>& halfStorage,
                         std::initializer_list<int64_t> shape)
    {
        t.name = name.c_str();
        t.type = type;

        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            halfStorage.resize (count);
            for (size_t i = 0; i < count; ++i)
                halfStorage[i] = floatToHalf (data[i]);

            t.data = halfStorage.data();
            t.numBytes = count * sizeof (uint16_t);
        }
        else
        {
            t.data = data;
            t.numBytes = count * sizeof (float);
        }

        setShape (t, shape);
    }

    void setIntTensor (InputTensor& t, const std::string& name, ONNXTensorElementDataType type,
                       const int64_t* data64, const int32_t* data32, size_t count,
                       std::initializer_list<int64_t> shape)
    {
        t.name = name.c_str();
        t.type = type;

        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
        {
            t.data = data32;
            t.numBytes = count * sizeof (int32_t);
        }
        else
        {
            t.data = data64;
            t.numBytes = count * sizeof (int64_t);
        }

        setShape (t, shape);
    }

    bool isIntType (ONNXTensorElementDataType t) noexcept
    {
        return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }

    bool isFloatInputType (ONNXTensorElementDataType t) noexcept
    {
        return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }

    juce::String describeInputs (const OrtModelSession& s)
    {
        juce::StringArray list;
        for (auto& i : s.getInputs())
            list.add (i.describe());
        return list.joinIntoString (", ");
    }

    int parseSampleRate (const juce::var& v)
    {
        if (auto* obj = v.getDynamicObject())
            for (auto* key : { "samplingRate", "sampling_rate", "sample_rate", "sampleRate", "sr" })
                if (obj->hasProperty (key))
                    return (int) obj->getProperty (key);

        return 0;
    }

    int readSampleRateFromMetadata (const OrtModelSession& s)
    {
        for (auto* key : { "samplingRate", "sampling_rate", "sample_rate", "sr" })
            if (auto value = s.getCustomMetadata (key).getIntValue(); value > 0)
                return value;

        // w-okada's voice changer stores a JSON blob under "metadata"
        if (auto json = s.getCustomMetadata ("metadata"); json.isNotEmpty())
            return parseSampleRate (juce::JSON::parse (json));

        return 0;
    }

    int roundUpToEven (int x) noexcept     { return (x + 1) & ~1; }

    /** Known file names of the common content encoders, so a pack saying "contentvec-768" also finds the
        community export "vec-768-layer-12.onnx" and vice versa. */
    juce::StringArray encoderAliases (const juce::String& id)
    {
        static const std::vector<juce::StringArray> groups {
            { "contentvec-768", "vec-768-layer-12", "contentvec768", "contentvec_768" },
            { "contentvec-256", "vec-256-layer-9", "contentvec256", "contentvec_256" },
            { "hubert-base", "hubert_base", "hubert" }
        };

        juce::StringArray names { id };

        for (auto& group : groups)
            if (group.contains (id, true))
                for (auto& alias : group)
                    names.addIfNotAlreadyThere (alias, true);

        return names;
    }
} // namespace

//==============================================================================
ModelConfig ModelConfig::fromProfile (const VoiceProfile& profile)
{
    ModelConfig c;
    c.packDirectory = profile.packDirectory;
    c.generatorFile = profile.neuralModelFile;
    c.encoderId = profile.neuralEncoder.trim();
    c.sampleRate = profile.neuralSampleRate;

    const auto manifest = profile.packDirectory.getChildFile ("voice.json");

    if (profile.packDirectory != juce::File() && manifest.existsAsFile())
    {
        const auto json = juce::JSON::parse (manifest);

        if (auto* model = json.getProperty ("model", {}).getDynamicObject())
        {
            auto get = [model] (const char* key) { return model->getProperty (key); };

            if (auto encoderFile = get ("encoderFile").toString(); encoderFile.isNotEmpty())
                c.encoderFile = profile.packDirectory.getChildFile (encoderFile);

            if (model->hasProperty ("hopSize"))         c.hopSize = juce::jmax (0, (int) get ("hopSize"));
            if (model->hasProperty ("speakerId"))       c.speakerId = juce::jmax (0, (int) get ("speakerId"));
            if (model->hasProperty ("noiseScale"))      c.noiseScale = juce::jlimit (0.0f, 2.0f, (float) (double) get ("noiseScale"));
            if (model->hasProperty ("streamContextMs")) c.streamContextMs = juce::jlimit (100, 3000, (int) get ("streamContextMs"));

            if (c.sampleRate <= 0)
                c.sampleRate = (int) get ("sampleRate");
        }
    }

    return c;
}

juce::Array<juce::File> ModelConfig::getEncoderCandidates (int featureChannels) const
{
    if (encoderFile != juce::File())
        return { encoderFile };

    auto id = encoderId;
    if (id.isEmpty())
        id = featureChannels == 256 ? "contentvec-256" : "contentvec-768";

    juce::Array<juce::File> dirs;
    if (packDirectory != juce::File())
        dirs.add (packDirectory);
    dirs.add (VoiceLibrary::getEncodersDirectory());

    juce::Array<juce::File> result;
    for (auto& dir : dirs)
        for (auto& name : encoderAliases (id))
            result.addIfNotAlreadyThere (dir.getChildFile (name + ".onnx"));

    return result;
}

//==============================================================================
int NeuralVoiceModel::Impl::getStreamContextFrames() const noexcept
{
    return juce::jlimit (10, 300, (config.streamContextMs + 5) / 10);
}

int NeuralVoiceModel::Impl::getStreamChunkFrames() const noexcept
{
    return streaming::blockFrames + roundUpToEven (streaming::tailFrames + getStreamContextFrames());
}

bool NeuralVoiceModel::Impl::initialise (const ModelConfig& c, juce::String& error)
{
    config = c;

    if (! config.generatorFile.existsAsFile())
    {
        error = "Voice model file not found: " + config.generatorFile.getFullPathName();
        return false;
    }

    generator = OrtModelSession::create (config.generatorFile, error);

    if (generator == nullptr)
        return false;

    if (! discoverGenerator (error))
    {
        error = config.generatorFile.getFileName() + ": " + error;
        return false;
    }

    const auto candidates = config.getEncoderCandidates (featureChannels);
    juce::String encoderError;

    for (auto& f : candidates)
    {
        if (f.existsAsFile())
        {
            encoder = OnnxRuntime::getInstance().getImpl().getSharedSession (f, encoderError);
            break;
        }
    }

    if (encoder == nullptr)
    {
        if (encoderError.isEmpty())
        {
            if (config.encoderFile != juce::File())
            {
                encoderError = "Content encoder not found: " + config.encoderFile.getFullPathName();
            }
            else
            {
                const auto id = config.encoderId.isNotEmpty() ? config.encoderId : candidates.getFirst().getFileNameWithoutExtension();
                encoderError = "Content encoder '" + id + "' not found - put " + id + ".onnx into "
                                 + VoiceLibrary::getEncodersDirectory().getFullPathName();
            }
        }

        error = encoderError;
        return false;
    }

    if (! discoverEncoder (error))
    {
        error = encoder->getFile().getFileName() + ": " + error;
        return false;
    }

    if (config.sampleRate > 0)
        sampleRate = config.sampleRate;
    else
        sampleRate = readSampleRateFromMetadata (*generator);

    if (! warmUp (error))
        return false;

    juce::StringArray parts;
    parts.add (juce::String (featureChannels) + "-d features");
    parts.add (usesF0 ? "pitch-guided" : "no pitch input");
    parts.add (juce::String (sampleRate / 1000.0, sampleRate % 1000 == 0 ? 0 : 1) + " kHz");
    parts.add ("encoder " + encoder->getFile().getFileName());
    parts.add (generator->getExecutionProvider());
    parts.add (getRealTimeLoad() < 0.8 ? "real-time load " + juce::String (juce::roundToInt (getRealTimeLoad() * 100.0)) + "%"
                                       : juce::String ("too slow for real time on this computer"));
    description = juce::String (featureChannels == 256 ? "RVC v1" : "RVC v2") + " voice model ("
                    + parts.joinIntoString (", ") + ")";
    return true;
}

bool NeuralVoiceModel::Impl::discoverGenerator (juce::String& error)
{
    const auto* phone   = generator->findInput ({ "phone", "feats", "hubert", "features" });
    const auto* lengths = generator->findInput ({ "phone_lengths", "lengths", "feats_length", "hubert_length", "x_lengths" });
    const auto* pitch   = generator->findInput ({ "pitch", "f0_coarse" });
    const auto* pitchf  = generator->findInput ({ "pitchf", "nsff0", "f0" });
    const auto* speaker = generator->findInput ({ "ds", "sid", "spk", "speaker_id" });
    const auto* noise   = generator->findInput ({ "rnd", "noise" });

    if (phone == nullptr || phone->rank() != 3 || ! isFloatInputType (phone->type))
    {
        error = "not an RVC voice model (expected inputs phone, phone_lengths, pitch, pitchf, ds, rnd; found "
                  + describeInputs (*generator) + ")";
        return false;
    }

    for (auto& input : generator->getInputs())
    {
        const auto* p = &input;

        if (p != phone && p != lengths && p != pitch && p != pitchf && p != speaker && p != noise)
        {
            error = "unsupported model input " + input.describe();
            return false;
        }
    }

    if ((pitch == nullptr) != (pitchf == nullptr))
    {
        error = "the model needs both 'pitch' and 'pitchf' inputs (or neither)";
        return false;
    }

    if ((lengths != nullptr && ! isIntType (lengths->type))
        || (pitch != nullptr && ! isIntType (pitch->type))
        || (speaker != nullptr && ! isIntType (speaker->type))
        || (pitchf != nullptr && ! isFloatInputType (pitchf->type))
        || (noise != nullptr && (! isFloatInputType (noise->type) || noise->rank() != 3)))
    {
        error = "unexpected input types (" + describeInputs (*generator) + ")";
        return false;
    }

    const TensorInfo* output = nullptr;
    for (auto& o : generator->getOutputs())
        if (juce::String (o.name).equalsIgnoreCase ("audio"))
            output = &o;

    if (output == nullptr)
        output = &generator->getOutputs().front();

    if (! output->isFloat())
    {
        error = "the model's audio output must be a float tensor";
        return false;
    }

    gen = {};
    gen.phone = phone->name;
    gen.phoneType = phone->type;
    gen.output = output->name;

    if (lengths != nullptr)  { gen.lengths = lengths->name;  gen.lengthsType = lengths->type; }
    if (pitch != nullptr)    { gen.pitch = pitch->name;      gen.pitchType = pitch->type; }
    if (pitchf != nullptr)   { gen.pitchf = pitchf->name;    gen.pitchfType = pitchf->type; }
    if (speaker != nullptr)  { gen.speaker = speaker->name;  gen.speakerType = speaker->type; }
    if (noise != nullptr)    { gen.noise = noise->name;      gen.noiseType = noise->type; }

    usesF0 = pitch != nullptr;
    featureChannels = (int) juce::jmax ((int64_t) 0, phone->dim (2));
    noiseChannels = noise != nullptr ? (int) (noise->dim (1) > 0 ? noise->dim (1) : 192) : 0;
    return true;
}

bool NeuralVoiceModel::Impl::discoverEncoder (juce::String& error)
{
    const auto& inputs = encoder->getInputs();
    const auto* wave = encoder->findInput ({ "source", "audio", "wav", "waveform", "input_values", "input", "x" });

    for (auto& i : inputs)
    {
        if (wave != nullptr)
            break;

        if (i.isFloat())
            wave = &i;
    }

    if (wave == nullptr || (wave->rank() != 2 && wave->rank() != 3) || ! isFloatInputType (wave->type))
    {
        error = "the content encoder must take a float waveform [1,1,N] or [1,N] (found " + describeInputs (*encoder) + ")";
        return false;
    }

    enc = {};
    enc.input = wave->name;
    enc.inputType = wave->type;
    enc.inputRank = wave->rank();

    for (auto& i : inputs)
    {
        if (&i == wave)
            continue;

        if (juce::String (i.name).containsIgnoreCase ("mask") && i.rank() == 2
            && (i.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL || i.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                || isIntType (i.type)))
        {
            enc.mask = i.name;   // padding mask: all zeros (= nothing is padding)
            enc.maskType = i.type;
            continue;
        }

        error = "unsupported content encoder input " + i.describe();
        return false;
    }

    const auto& out = encoder->getOutputs().front();
    enc.output = out.name;

    if (out.rank() == 2)
    {
        enc.layout = EncoderIO::Layout::timeMajor;   // [T, C]
    }
    else if (out.rank() == 3)
    {
        const auto d1 = out.dim (1), d2 = out.dim (2);
        const auto expected = (int64_t) featureChannels;

        if (d2 > 0 && (expected <= 0 || d2 == expected))
            enc.layout = EncoderIO::Layout::timeMajor;
        else if (d1 > 0 && (expected <= 0 || d1 == expected))
            enc.layout = EncoderIO::Layout::channelMajor;
        else if (expected > 0 && (d1 > 0 || d2 > 0))
        {
            error = "the content encoder produces " + juce::String (d2 > 0 ? d2 : d1) + "-dimensional features but the voice "
                      "model expects " + juce::String (expected) + " (use the matching encoder, e.g. contentvec-"
                      + juce::String (expected) + ")";
            return false;
        }
    }
    else
    {
        error = "unexpected content encoder output " + out.describe();
        return false;
    }

    return true;
}

bool NeuralVoiceModel::Impl::warmUp (juce::String& error)
{
    Workspace ws;

    auto makeInput = [] (int frames, std::vector<float>& audio, std::vector<float>& f0)
    {
        audio.resize ((size_t) (frames * dsp::samplesPerFrame16k + dsp::encoderRightPad16k));
        for (size_t i = 0; i < audio.size(); ++i)
            audio[i] = 0.1f * std::sin (juce::MathConstants<float>::twoPi * 220.0f * (float) i / (float) dsp::encoderSampleRate);

        f0.assign ((size_t) frames, 220.0f);
    };

    // 1. Validates the whole pipeline and measures the generator's output samples per frame.
    std::vector<float> audio, f0;
    constexpr int testFrames = 32;
    makeInput (testFrames, audio, f0);

    const int configuredHop = config.hopSize;
    hopSize = 0;   // runFrames() leaves the raw output length

    if (! runFrames (audio.data(), testFrames, f0.data(), 0, ws, false, error))
    {
        error = "test run failed: " + error;
        return false;
    }

    const auto rawLength = ws.audio.size();

    if (rawLength == 0 || rawLength % (size_t) testFrames != 0)
    {
        error = "unexpected output length (" + juce::String ((int64_t) rawLength) + " samples for "
                  + juce::String (testFrames) + " frames)";
        return false;
    }

    const auto measuredHop = (int) (rawLength / (size_t) testFrames);

    if (configuredHop > 0 && configuredHop != measuredHop)
    {
        error = "voice.json says hopSize " + juce::String (configuredHop) + " but the model produces "
                  + juce::String (measuredHop) + " samples per frame";
        return false;
    }

    hopSize = measuredHop;

    // RVC generators always produce one 10 ms frame per hop, so the hop size defines the true output rate.
    if (sampleRate != hopSize * dsp::framesPerSecond)
    {
        if (sampleRate > 0)
        {
            DBG ("VoxaVersa: " << config.generatorFile.getFileName() << " declares " << sampleRate << " Hz but produces "
                               << hopSize << " samples per 10 ms frame - using " << hopSize * dsp::framesPerSecond << " Hz");
        }

        sampleRate = hopSize * dsp::framesPerSecond;
    }

    for (auto v : ws.audio)
    {
        if (! std::isfinite (v))
        {
            error = "the model produced invalid samples (NaN/Inf)";
            return false;
        }
    }

    // 2. Two runs with the real-time chunk shape: the first primes ONNX Runtime's memory plan for it,
    //    the second measures the inference time of one real-time chunk.
    const int chunkFrames = getStreamChunkFrames();
    makeInput (chunkFrames, audio, f0);
    double elapsedSeconds = 0.0;

    for (int run = 0; run < 2; ++run)
    {
        const auto start = juce::Time::getMillisecondCounterHiRes();

        if (! runFrames (audio.data(), chunkFrames, f0.data(), 0, ws, false, error))
        {
            error = "test run failed: " + error;
            return false;
        }

        elapsedSeconds = (juce::Time::getMillisecondCounterHiRes() - start) * 0.001;
    }

    chunkInferenceSeconds = elapsedSeconds;
    return true;
}

double NeuralVoiceModel::Impl::getRealTimeLoad() const noexcept
{
    return chunkInferenceSeconds / streaming::headroomSeconds;
}

//==============================================================================
bool NeuralVoiceModel::Impl::runFrames (const float* audio16k, int numFrames, const float* f0Frames, int64_t firstFrame,
                                        Workspace& ws, bool offline, juce::String& error) const
{
    jassert (numFrames > 0 && numFrames % 2 == 0);
    const auto numSamples = (size_t) (numFrames * dsp::samplesPerFrame16k + dsp::encoderRightPad16k);
    const auto n = (int64_t) numSamples;

    //=== Content encoder ======================================================
    {
        std::array<InputTensor, 2> inputs {};
        size_t numInputs = 1;

        if (enc.inputRank == 3)
            setFloatTensor (inputs[0], enc.input, enc.inputType, audio16k, numSamples, ws.encoderHalf, { 1, 1, n });
        else
            setFloatTensor (inputs[0], enc.input, enc.inputType, audio16k, numSamples, ws.encoderHalf, { 1, n });

        if (! enc.mask.empty())
        {
            const size_t bytesPerElement = enc.maskType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8
                                         : enc.maskType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? 4 : 1;
            ws.mask.assign (numSamples * bytesPerElement, 0);

            auto& m = inputs[numInputs++];
            m.name = enc.mask.c_str();
            m.type = enc.maskType;
            m.data = ws.mask.data();
            m.numBytes = ws.mask.size();
            setShape (m, { 1, n });
        }

        const char* outputNames[] = { enc.output.c_str() };

        if (! encoder->run (inputs.data(), numInputs, outputNames, &ws.encoderOut, 1, offline, error))
        {
            error = "content encoder: " + error;
            return false;
        }
    }

    //=== Features: [1,T,C] or [1,C,T] -> [1, numFrames, C] (each 20 ms frame repeated twice) ==========
    const auto& shape = ws.encoderOut.shape;
    int64_t numEncoderFrames = 0, channels = 0;
    bool channelMajor = false;

    if (shape.size() == 2)
    {
        numEncoderFrames = shape[0];
        channels = shape[1];
    }
    else if (shape.size() == 3)
    {
        auto layout = enc.layout;

        if (layout == EncoderIO::Layout::unknown)
        {
            const auto expectedFrames = (int64_t) numFrames / 2;

            if (featureChannels > 0)
                layout = shape[2] == featureChannels ? EncoderIO::Layout::timeMajor : EncoderIO::Layout::channelMajor;
            else
                layout = std::abs (shape[1] - expectedFrames) <= std::abs (shape[2] - expectedFrames)
                           ? EncoderIO::Layout::timeMajor : EncoderIO::Layout::channelMajor;
        }

        channelMajor = layout == EncoderIO::Layout::channelMajor;
        numEncoderFrames = channelMajor ? shape[2] : shape[1];
        channels = channelMajor ? shape[1] : shape[2];
    }

    if (numEncoderFrames <= 0 || channels <= 0 || (size_t) (numEncoderFrames * channels) != ws.encoderOut.data.size())
    {
        error = "unexpected content encoder output shape";
        return false;
    }

    if (featureChannels > 0 && channels != featureChannels)
    {
        error = "the content encoder produces " + juce::String (channels) + "-dimensional features but the voice model expects "
                  + juce::String (featureChannels);
        return false;
    }

    const auto c = (size_t) channels;
    const auto frames = (size_t) numFrames;
    ws.phone.resize (frames * c);

    for (size_t i = 0; i < frames; ++i)
    {
        const auto t = (size_t) juce::jmin ((int64_t) (i / 2), numEncoderFrames - 1);
        auto* dest = ws.phone.data() + i * c;

        if (channelMajor)
        {
            for (size_t ch = 0; ch < c; ++ch)
                dest[ch] = ws.encoderOut.data[ch * (size_t) numEncoderFrames + t];
        }
        else
        {
            std::copy_n (ws.encoderOut.data.data() + t * c, c, dest);
        }
    }

    //=== Generator ============================================================
    std::array<InputTensor, 6> inputs {};
    size_t numInputs = 0;
    const auto frameCount = (int64_t) numFrames;
    const int64_t lengthValue64 = frameCount, speaker64 = config.speakerId;
    const int32_t lengthValue32 = numFrames, speaker32 = config.speakerId;

    setFloatTensor (inputs[numInputs++], gen.phone, gen.phoneType, ws.phone.data(), ws.phone.size(), ws.phoneHalf,
                    { 1, frameCount, channels });

    if (! gen.lengths.empty())
        setIntTensor (inputs[numInputs++], gen.lengths, gen.lengthsType, &lengthValue64, &lengthValue32, 1, { 1 });

    if (usesF0)
    {
        ws.pitchf.resize (frames);
        ws.pitch.resize (frames);
        ws.pitch32.resize (frames);

        for (size_t i = 0; i < frames; ++i)
        {
            const float f = f0Frames != nullptr && f0Frames[i] > 0.0f && std::isfinite (f0Frames[i]) ? f0Frames[i] : 0.0f;
            ws.pitchf[i] = f;
            ws.pitch[i] = dsp::coarsePitch (f);
            ws.pitch32[i] = (int32_t) ws.pitch[i];
        }

        setIntTensor (inputs[numInputs++], gen.pitch, gen.pitchType, ws.pitch.data(), ws.pitch32.data(), frames, { 1, frameCount });
        setFloatTensor (inputs[numInputs++], gen.pitchf, gen.pitchfType, ws.pitchf.data(), frames, ws.pitchfHalf, { 1, frameCount });
    }

    if (! gen.speaker.empty())
        setIntTensor (inputs[numInputs++], gen.speaker, gen.speakerType, &speaker64, &speaker32, 1, { 1 });

    if (! gen.noise.empty())
    {
        const auto noiseCh = (size_t) noiseChannels;
        ws.noise.resize (noiseCh * frames);

        for (size_t ch = 0; ch < noiseCh; ++ch)
            for (size_t i = 0; i < frames; ++i)
                ws.noise[ch * frames + i] = config.noiseScale * dsp::gaussianNoise (firstFrame + (int64_t) i, (int) ch);

        setFloatTensor (inputs[numInputs++], gen.noise, gen.noiseType, ws.noise.data(), ws.noise.size(), ws.noiseHalf,
                        { 1, (int64_t) noiseCh, frameCount });
    }

    const char* outputNames[] = { gen.output.c_str() };

    if (! generator->run (inputs.data(), numInputs, outputNames, &ws.generatorOut, 1, offline, error))
    {
        error = "voice model: " + error;
        return false;
    }

    std::swap (ws.audio, ws.generatorOut.data);

    if (hopSize > 0)
        ws.audio.resize (frames * (size_t) hopSize, 0.0f);

    return true;
}

//==============================================================================
bool NeuralVoiceModel::Impl::convert (const juce::AudioBuffer<float>& input, double inputSampleRate,
                                      const std::vector<float>& f0Hz, double f0FrameRate,
                                      juce::AudioBuffer<float>& output, const std::atomic<bool>* cancel,
                                      juce::String& error) const
{
    const int inputRate = juce::roundToInt (inputSampleRate);
    const int numInputSamples = input.getNumSamples();

    if (inputRate < 4000 || inputRate > 768000)
    {
        error = "unsupported sample rate " + juce::String (inputSampleRate);
        return false;
    }

    if (usesF0 && f0Hz.empty())
    {
        error = "this voice needs a target pitch contour";
        return false;
    }

    if (numInputSamples == 0 || input.getNumChannels() == 0)
    {
        output.setSize (1, 0);
        return true;
    }

    auto isCancelled = [cancel] { return cancel != nullptr && cancel->load(); };

    // Mono mix
    std::vector<float> mono ((size_t) numInputSamples, 0.0f);
    const float channelGain = 1.0f / (float) input.getNumChannels();

    for (int ch = 0; ch < input.getNumChannels(); ++ch)
        juce::FloatVectorOperations::addWithMultiply (mono.data(), input.getReadPointer (ch), channelGain, numInputSamples);

    // Band-limited conversion to the encoder rate, padded with silence on both sides
    const auto signal16k = dsp::Resampler (inputRate, dsp::encoderSampleRate).processAll (mono.data(), numInputSamples);

    constexpr int edgePadFrames = 20;
    constexpr int contextFrames = 100;          // 1 s of real audio on both sides of every segment
    constexpr int targetSegmentFrames = 1000;   // ~10 s per inference
    constexpr int maxSegmentFrames = 1300;
    constexpr int cutSearchFrames = 150;

    const int clipFrames = roundUpToEven (juce::jmax (2, (int) ((signal16k.size() + dsp::samplesPerFrame16k - 1) / dsp::samplesPerFrame16k)));
    const int totalFrames = clipFrames + 2 * edgePadFrames;

    std::vector<float> padded ((size_t) (totalFrames * dsp::samplesPerFrame16k + dsp::encoderRightPad16k), 0.0f);
    std::copy (signal16k.begin(), signal16k.end(), padded.begin() + edgePadFrames * dsp::samplesPerFrame16k);

    std::vector<float> f0Frames ((size_t) totalFrames, 0.0f);
    if (usesF0)
        dsp::resampleF0 (f0Hz, f0FrameRate, f0Frames.data() + edgePadFrames, clipFrames);

    // Segment boundaries at the quietest 20 ms-aligned points near every ~10 s
    auto energyAround = [&padded] (int frame)
    {
        double e = 0.0;
        const auto start = (size_t) ((frame - 2) * dsp::samplesPerFrame16k);
        for (size_t i = start; i < start + 4 * (size_t) dsp::samplesPerFrame16k; ++i)
            e += (double) padded[i] * padded[i];
        return e;
    };

    std::vector<int> cuts { edgePadFrames };
    const int clipEnd = edgePadFrames + clipFrames;

    while (clipEnd - cuts.back() > maxSegmentFrames)
    {
        const int nominal = cuts.back() + targetSegmentFrames;
        int best = nominal;
        double bestEnergy = std::numeric_limits<double>::max();

        for (int frame = nominal - cutSearchFrames; frame <= nominal + cutSearchFrames; frame += 2)
        {
            const auto e = energyAround (frame);

            if (e < bestEnergy)
            {
                bestEnergy = e;
                best = frame;
            }
        }

        cuts.push_back (best);
    }

    cuts.push_back (clipEnd);

    // Inference per segment, assembled at the model rate. Every inference restarts the generator's
    // excitation phase, so each segment is placed with a small shift (|shift| <= search / 2, relative to
    // the fixed grid, so it never accumulates) that best continues the previous segment (SOLA), and the
    // join is crossfaded.
    const auto hop = (int64_t) hopSize;
    const auto clipStart = (int64_t) edgePadFrames * hop;
    std::vector<float> assembled ((size_t) ((int64_t) clipFrames * hop), 0.0f);
    const int halfFade = juce::jmax (1, juce::roundToInt (0.025 * sampleRate));
    const int fadeLength = 2 * halfFade;
    const int search = juce::jmax (2, juce::roundToInt (streaming::searchSeconds * sampleRate)) & ~1;
    std::vector<float> previousTail ((size_t) fadeLength, 0.0f), head ((size_t) fadeLength);
    Workspace ws;

    // Copies segment samples to assembled indices [from, to) (padded model-rate coordinates) with a shift.
    auto writeRange = [&] (const float* segmentAudio, int64_t segmentStart, int64_t shift, int64_t from, int64_t to)
    {
        for (auto q = from; q < to; ++q)
        {
            const auto index = q - clipStart;
            if (index >= 0 && index < (int64_t) assembled.size())
                assembled[(size_t) index] = segmentAudio[q + shift - segmentStart];
        }
    };

    for (size_t s = 0; s + 1 < cuts.size(); ++s)
    {
        if (isCancelled())
        {
            error = "Cancelled";
            return false;
        }

        const int a = cuts[s], b = cuts[s + 1];
        const int runStart = juce::jmax (0, a - contextFrames);
        const int runEnd = juce::jmin (totalFrames, b + contextFrames);
        const bool first = s == 0, last = s + 2 == cuts.size();

        if (! runFrames (padded.data() + (size_t) runStart * dsp::samplesPerFrame16k, runEnd - runStart,
                         f0Frames.data() + runStart, runStart, ws, true, error))
            return false;

        const auto* segment = ws.audio.data();
        const auto segmentStart = (int64_t) runStart * hop;
        const auto boundaryA = (int64_t) a * hop, boundaryB = (int64_t) b * hop;
        int64_t shift = 0;

        if (! first)
        {
            float r = 0.0f;
            const auto* candidate = segment + (boundaryA - halfFade - search / 2 - segmentStart);
            shift = dsp::findBestOffset (previousTail.data(), candidate, fadeLength, search, r) - search / 2;

            for (int i = 0; i < fadeLength; ++i)
                head[(size_t) i] = segment[boundaryA - halfFade + i + shift - segmentStart];

            dsp::crossfade (previousTail.data(), head.data(), head.data(), fadeLength, r);
            writeRange (head.data(), boundaryA - halfFade, 0, boundaryA - halfFade, boundaryA + halfFade);
        }

        writeRange (segment, segmentStart, shift, first ? boundaryA : boundaryA + halfFade, last ? boundaryB : boundaryB - halfFade);

        if (! last)
            for (int i = 0; i < fadeLength; ++i)
                previousTail[(size_t) i] = segment[boundaryB - halfFade + i + shift - segmentStart];
    }

    if (isCancelled())
    {
        error = "Cancelled";
        return false;
    }

    // Back to the caller's sample rate
    output.setSize (1, numInputSamples, false, false, true);
    dsp::Resampler (sampleRate, inputRate).process (assembled.data(), 0, (int) assembled.size(),
                                                    output.getWritePointer (0), 0, numInputSamples);
    return true;
}

} // namespace voxa::neural
