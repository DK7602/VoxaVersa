#include "neural/NeuralVoice.h"
#include "neural/NeuralRuntime.h"
#include "neural/NeuralDsp.h"
#include "neural/RvcModel.h"
#include "voices/VoiceLibrary.h"

#include <cmath>
#include <ctime>
#include <thread>

/*  Neural engine tests (category "Neural").

    The DSP tests always run. The ONNX Runtime tests need a runtime and the dummy models built by
    tools/neural/make_dummy_models.py:

        VOXAVERSA_TEST_ORT_LIB=/path/to/libonnxruntime.so      (optional - else the normal search is used)
        VOXAVERSA_TEST_MODELS_DIR=/path/to/dummy/models

    Without them those tests log "skipped" and pass, so CI machines without ONNX Runtime stay green. */

namespace voxa::neural
{

namespace
{
    constexpr double pi = juce::MathConstants<double>::pi;

    std::vector<float> makeSine (double freq, double rate, int numSamples, float amplitude, double phase = 0.0)
    {
        std::vector<float> s ((size_t) numSamples);
        for (int i = 0; i < numSamples; ++i)
            s[(size_t) i] = amplitude * (float) std::sin (2.0 * pi * freq * i / rate + phase);
        return s;
    }

    /** Frequency of a (nearly) pure tone from its positive-going zero crossings; 0 if fewer than 3. */
    double estimateFrequency (const float* x, int n, double rate)
    {
        double first = -1.0, last = -1.0;
        int crossings = 0;

        for (int i = 1; i < n; ++i)
        {
            if (x[i - 1] < 0.0f && x[i] >= 0.0f)
            {
                const double t = (i - 1) + (double) -x[i - 1] / (double) (x[i] - x[i - 1]);
                if (first < 0.0)
                    first = t;
                last = t;
                ++crossings;
            }
        }

        return crossings >= 3 ? (crossings - 1) * rate / (last - first) : 0.0;
    }

    double rms (const float* x, int n)
    {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += (double) x[i] * x[i];
        return n > 0 ? std::sqrt (s / n) : 0.0;
    }

    std::vector<double> envelope (const std::vector<float>& x, double rate, double hopSeconds = 0.005)
    {
        const int hop = juce::roundToInt (hopSeconds * rate), window = 2 * hop;
        std::vector<double> e;
        for (int start = 0; start + window <= (int) x.size(); start += hop)
            e.push_back (rms (x.data() + start, window));
        return e;
    }

    /** Lag (in envelope hops) maximising the correlation of b against a, and that correlation. */
    int bestLag (const std::vector<double>& a, const std::vector<double>& b, int maxLag, double& bestCorrelation)
    {
        int best = 0;
        bestCorrelation = -2.0;

        for (int lag = -maxLag; lag <= maxLag; ++lag)
        {
            double ab = 0, aa = 0, bb = 0;

            for (int i = 0; i < (int) a.size(); ++i)
            {
                const int j = i + lag;
                if (j < 0 || j >= (int) b.size())
                    continue;
                ab += a[(size_t) i] * b[(size_t) j];
                aa += a[(size_t) i] * a[(size_t) i];
                bb += b[(size_t) j] * b[(size_t) j];
            }

            const double c = ab / std::sqrt (aa * bb + 1.0e-12);
            if (c > bestCorrelation)
            {
                bestCorrelation = c;
                best = lag;
            }
        }

        return best;
    }

    bool allFinite (const std::vector<float>& x, float limit)
    {
        for (auto v : x)
            if (! std::isfinite (v) || std::abs (v) > limit)
                return false;
        return true;
    }

    /** Test input: tone bursts with different levels, so onsets give a clear alignment reference. */
    std::vector<float> makeBurstInput (double rate, double seconds)
    {
        const int n = juce::roundToInt (rate * seconds);
        std::vector<float> x ((size_t) n, 0.0f);

        struct Burst { double start, end; float level; };
        const Burst bursts[] = { { 0.30, 1.00, 0.5f }, { 1.25, 1.60, 0.25f }, { 1.80, 2.70, 0.6f }, { 2.95, 3.40, 0.4f } };

        for (auto& b : bursts)
        {
            for (int i = juce::roundToInt (b.start * rate); i < juce::jmin (n, juce::roundToInt (b.end * rate)); ++i)
            {
                const double t = i / rate;
                const auto fade = (float) juce::jmin (1.0, (t - b.start) / 0.005, (b.end - t) / 0.005);
                x[(size_t) i] = b.level * fade * (float) std::sin (2.0 * pi * 180.0 * t);
            }
        }

        return x;
    }

    float targetPitchAt (double seconds)   { return seconds < 1.7 ? 220.0f : 330.0f; }

    juce::File writeTempPack (const juce::String& json)
    {
        auto dir = juce::File::getSpecialLocation (juce::File::tempDirectory).getNonexistentChildFile ("voxaversa-neural-test", "");
        dir.createDirectory();
        dir.getChildFile ("voice.json").replaceWithText (json);
        return dir;
    }

    juce::String packJson (const juce::File& generator, const juce::File& encoder, int sampleRate,
                           const juce::String& encoderId = "dummy", int streamContextMs = 0)
    {
        auto* model = new juce::DynamicObject();
        model->setProperty ("file", generator.getFullPathName());
        model->setProperty ("encoder", encoderId);
        if (encoder != juce::File())
            model->setProperty ("encoderFile", encoder.getFullPathName());
        model->setProperty ("sampleRate", sampleRate);
        if (streamContextMs > 0)
            model->setProperty ("streamContextMs", streamContextMs);

        auto* root = new juce::DynamicObject();
        root->setProperty ("name", "Test voice");
        root->setProperty ("model", juce::var (model));
        return juce::JSON::toString (juce::var (root));
    }
} // namespace

//==============================================================================
class NeuralDspTests : public juce::UnitTest
{
public:
    NeuralDspTests() : juce::UnitTest ("Neural DSP", "Neural") {}

    void runTest() override
    {
        beginTest ("Coarse pitch");
        {
            expectEquals (dsp::coarsePitch (0.0f), 1);
            expectEquals (dsp::coarsePitch (-5.0f), 1);
            expectEquals (dsp::coarsePitch (std::nanf ("")), 1);
            expectEquals (dsp::coarsePitch (40.0f), 1);
            expectEquals (dsp::coarsePitch (50.0f), 1);
            expectEquals (dsp::coarsePitch (1100.0f), 255);
            expectEquals (dsp::coarsePitch (5000.0f), 255);

            const double melMin = 1127.0 * std::log (1.0 + 50.0 / 700.0), melMax = 1127.0 * std::log (1.0 + 1100.0 / 700.0);
            for (float f : { 80.0f, 220.0f, 440.0f, 880.0f })
            {
                const double mel = 1127.0 * std::log (1.0 + f / 700.0);
                const auto expected = (int) std::lrint ((mel - melMin) * 254.0 / (melMax - melMin) + 1.0);
                expectEquals (dsp::coarsePitch (f), expected);
            }

            expect (dsp::coarsePitch (220.0f) < dsp::coarsePitch (221.0f) + 1 && dsp::coarsePitch (300.0f) > dsp::coarsePitch (220.0f));
        }

        beginTest ("Band-limited resampler");
        {
            const dsp::Resampler down (48000, 16000), up (16000, 48000);
            const auto tone = makeSine (1000.0, 48000.0, 48000, 0.8f);
            const auto tone16 = down.processAll (tone.data(), (int) tone.size());
            expectEquals ((int) tone16.size(), 16000);
            expectWithinAbsoluteError (rms (tone16.data() + 1000, 14000), 0.8 / std::sqrt (2.0), 0.005);
            expectWithinAbsoluteError (estimateFrequency (tone16.data() + 1000, 14000, 16000.0), 1000.0, 0.5);

            // time alignment: a resampled sine must keep its phase
            const auto reference16 = makeSine (1000.0, 16000.0, 16000, 0.8f);
            double maxError = 0.0;
            for (int i = 500; i < 15500; ++i)
                maxError = juce::jmax (maxError, (double) std::abs (tone16[(size_t) i] - reference16[(size_t) i]));
            expect (maxError < 0.01, "resampled tone is not phase aligned: " + juce::String (maxError));

            // a tone above the new Nyquist must be removed (no aliasing)
            const auto high = makeSine (10000.0, 48000.0, 48000, 0.8f);
            const auto high16 = down.processAll (high.data(), (int) high.size());
            expect (rms (high16.data() + 1000, 14000) < 0.8 * 0.001, "aliasing: " + juce::String (rms (high16.data() + 1000, 14000)));

            const auto back = up.processAll (tone16.data(), (int) tone16.size());
            expectEquals ((int) back.size(), 48000);
            maxError = 0.0;
            for (int i = 3000; i < 45000; ++i)
                maxError = juce::jmax (maxError, (double) std::abs (back[(size_t) i] - tone[(size_t) i]));
            expect (maxError < 0.01, "round trip error " + juce::String (maxError));

            // stateless windows of a stream are identical to processing the whole signal
            const dsp::Resampler odd (44100, 40000);
            const auto src = makeSine (440.0, 44100.0, 44100, 0.5f);
            const auto whole = odd.processAll (src.data(), (int) src.size());
            std::vector<float> part (5000);
            odd.process (src.data(), 0, (int) src.size(), part.data(), 12345, (int) part.size());
            double windowError = 0.0;
            for (size_t i = 0; i < part.size(); ++i)
                windowError = juce::jmax (windowError, (double) std::abs (part[i] - whole[12345 + i]));
            expect (windowError < 1.0e-6);

            // input given as a window with an absolute offset
            const auto firstIn = odd.firstInputFor (20000), lastIn = odd.lastInputFor (20999);
            std::vector<float> window (src.begin() + firstIn, src.begin() + lastIn + 1);
            odd.process (window.data(), firstIn, (int) window.size(), part.data(), 20000, 1000);
            windowError = 0.0;
            for (size_t i = 0; i < 1000; ++i)
                windowError = juce::jmax (windowError, (double) std::abs (part[i] - whole[20000 + i]));
            expect (windowError < 1.0e-6);
        }

        beginTest ("Pitch contour interpolation");
        {
            std::vector<float> f0 { 100.0f, 200.0f, 0.0f, 0.0f, 400.0f };
            std::vector<float> out (8);
            dsp::resampleF0 (f0, 100.0, out.data(), (int) out.size());
            expectWithinAbsoluteError (out[0], 100.0f, 0.01f);
            expectWithinAbsoluteError (out[1], 200.0f, 0.01f);
            expectEquals (out[2], 0.0f);
            expectWithinAbsoluteError (out[4], 400.0f, 0.01f);
            expectEquals (out[5], 0.0f);   // beyond the contour
            expectEquals (out[7], 0.0f);

            std::vector<float> slow { 100.0f, 400.0f };   // 50 frames/s
            std::vector<float> fast (4);
            dsp::resampleF0 (slow, 50.0, fast.data(), 4);
            expectWithinAbsoluteError (fast[0], 100.0f, 0.01f);
            expect (fast[1] > 100.0f && fast[1] < 200.0f);           // geometric interpolation
            expect (fast[2] > 200.0f && fast[2] < 400.0f);
            expectWithinAbsoluteError (fast[3], 400.0f, 0.01f);

            std::vector<float> samples (480, 220.0f);
            expectWithinAbsoluteError (dsp::frameF0 (samples.data(), 480), 220.0f, 0.001f);
            std::fill (samples.begin(), samples.begin() + 300, 0.0f);
            expectEquals (dsp::frameF0 (samples.data(), 480), 0.0f);
        }

        beginTest ("SOLA alignment and crossfades");
        {
            const int length = 2400, search = 500;
            const auto reference = makeSine (150.0, 40000.0, length, 0.7f);
            const auto candidate = makeSine (150.0, 40000.0, length + search, 0.7f, -2.0 * pi * 150.0 * 123.0 / 40000.0);
            float r = 0.0f;
            const int offset = dsp::findBestOffset (reference.data(), candidate.data(), length, search, r);
            expectEquals (offset, 123);
            expect (r > 0.999f);

            std::vector<float> faded (2000);
            const auto a = makeSine (300.0, 48000.0, 2000, 0.5f);
            dsp::crossfade (a.data(), a.data(), faded.data(), 2000, 1.0f);
            double err = 0.0;
            for (int i = 0; i < 2000; ++i)
                err = juce::jmax (err, (double) std::abs (faded[(size_t) i] - a[(size_t) i]));
            expect (err < 1.0e-5, "aligned crossfade must be transparent");

            juce::Random rng (42);
            std::vector<float> n1 (20000), n2 (20000), mixed (20000);
            for (size_t i = 0; i < n1.size(); ++i)
            {
                n1[i] = rng.nextFloat() * 2.0f - 1.0f;
                n2[i] = rng.nextFloat() * 2.0f - 1.0f;
            }
            dsp::crossfade (n1.data(), n2.data(), mixed.data(), 20000, dsp::correlation (n1.data(), n2.data(), 20000));
            const double middle = rms (mixed.data() + 9000, 2000), edges = rms (n1.data(), 20000);
            expectWithinAbsoluteError (middle / edges, 1.0, 0.08);   // equal power for unrelated signals

            expect (dsp::isSilent (std::vector<float> (1000, 0.0f).data(), 1000));
            expect (! dsp::isSilent (a.data(), 2000));
        }

        beginTest ("Noise and float16");
        {
            double sum = 0.0, sumSq = 0.0;
            const int n = 192 * 400;
            for (int i = 0; i < n; ++i)
            {
                const double v = dsp::gaussianNoise (i / 192, i % 192);
                sum += v;
                sumSq += v * v;
            }
            expectWithinAbsoluteError (sum / n, 0.0, 0.02);
            expectWithinAbsoluteError (std::sqrt (sumSq / n), 1.0, 0.02);
            expectEquals (dsp::gaussianNoise (12345, 7), dsp::gaussianNoise (12345, 7));
            expect (std::abs (dsp::gaussianNoise (12345, 7) - dsp::gaussianNoise (12346, 7)) > 0.0f);

            for (float v : { 0.0f, 1.0f, -2.5f, 0.333f, 65504.0f, 1.0e-6f, -1.0e-7f })
                expectWithinAbsoluteError (halfToFloat (floatToHalf (v)), v, std::abs (v) * 0.001f + 1.0e-7f);
            expect (std::isinf (halfToFloat (floatToHalf (1.0e6f))));
        }
    }
};

static NeuralDspTests neuralDspTests;

//==============================================================================
class NeuralEngineTests : public juce::UnitTest
{
public:
    NeuralEngineTests() : juce::UnitTest ("Neural engine", "Neural") {}

    void runTest() override
    {
        beginTest ("ONNX Runtime discovery");

        auto& runtime = OnnxRuntime::getInstance();
        const auto searchLocations = OnnxRuntime::getSearchLocations();
        expect (! searchLocations.isEmpty());
        expect (searchLocations.contains (VoiceLibrary::getUserDataDirectory().getChildFile ("Runtime")));

        const auto testLibrary = juce::SystemStats::getEnvironmentVariable ("VOXAVERSA_TEST_ORT_LIB", {});

        if (testLibrary.isNotEmpty())
            expect (runtime.getImpl().loadFromFile (juce::File (testLibrary)), runtime.getStatusMessage());
        else
            runtime.ensureLoaded();

        logMessage ("  " + runtime.getStatusMessage());

        if (! runtime.isAvailable())
        {
            testWithoutRuntime();
            logMessage ("  ONNX Runtime not available - skipped the inference tests");
            return;
        }

        expect (runtime.ensureLoaded());   // idempotent
        expect (runtime.getStatusMessage().startsWith ("AI engine ready (ONNX Runtime "), runtime.getStatusMessage());
        expect (runtime.getLoadedLibraryFile().existsAsFile());
        logMessage ("  loaded " + runtime.getLoadedLibraryFile().getFullPathName());

        modelsDir = juce::File (juce::SystemStats::getEnvironmentVariable ("VOXAVERSA_TEST_MODELS_DIR", {}));

        if (modelsDir == juce::File() || ! modelsDir.getChildFile ("voice.json").existsAsFile())
        {
            logMessage ("  VOXAVERSA_TEST_MODELS_DIR not set - skipped the model tests (see tools/neural/make_dummy_models.py)");
            return;
        }

        testModelLoading();
        testIoVariants();
        testOfflineConversion();
        testStreaming();
        testHotSwap();
        testOfflineRenderDetection();

        for (auto& dir : tempDirs)
            dir.deleteRecursively();
    }

private:
    juce::File modelsDir;
    juce::Array<juce::File> tempDirs;
    std::shared_ptr<NeuralVoiceModel> mainModel;

    std::shared_ptr<NeuralVoiceModel> loadPack (const juce::File& dir, juce::String& error)
    {
        VoiceProfile profile;
        if (! VoiceLibrary::loadVoicePack (dir, profile, error))
            return nullptr;
        return NeuralVoiceModel::load (profile, error);
    }

    std::shared_ptr<NeuralVoiceModel> loadTempPack (const juce::String& json, juce::String& error)
    {
        auto dir = writeTempPack (json);
        tempDirs.add (dir);
        return loadPack (dir, error);
    }

    //==============================================================================
    void testWithoutRuntime()
    {
        beginTest ("Graceful failure without a runtime");

        expect (runtime().getStatusMessage().isNotEmpty());

        VoiceProfile profile;
        profile.name = "Missing";
        profile.neuralModelFile = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("missing-model.onnx");
        juce::String error;
        expect (NeuralVoiceModel::load (profile, error) == nullptr);
        expect (error.isNotEmpty());

        NeuralVoiceStream stream;
        stream.prepare (48000.0, 512);
        expect (! stream.hasModel());
        expect (! stream.isRunningInRealTime());
        expect (stream.getLatencySamples() > 0);

        std::vector<float> in ((size_t) 512, 0.5f), f0 ((size_t) 512, 220.0f), out ((size_t) 512, 1.0f);
        stream.process (in.data(), f0.data(), out.data(), 512);
        expectEquals (rms (out.data(), 512), 0.0);
        stream.reset();
    }

    static OnnxRuntime& runtime()   { return OnnxRuntime::getInstance(); }

    //==============================================================================
    void testModelLoading()
    {
        beginTest ("Model loading");

        juce::String error;
        const auto start = juce::Time::getMillisecondCounterHiRes();
        mainModel = loadPack (modelsDir, error);
        expect (mainModel != nullptr, error);

        if (mainModel == nullptr)
            return;

        logMessage ("  " + mainModel->getDescription() + " - loaded in "
                    + juce::String (juce::Time::getMillisecondCounterHiRes() - start, 0) + " ms");

        auto& impl = mainModel->getImpl();
        expectEquals (mainModel->getModelSampleRate(), 40000);
        expectEquals (impl.hopSize, 400);
        expectEquals (impl.featureChannels, 768);
        expectEquals (impl.noiseChannels, 192);
        expect (impl.usesF0);
        expect (impl.enc.layout == NeuralVoiceModel::Impl::EncoderIO::Layout::timeMajor);
        expectEquals (impl.enc.inputRank, 3);
        expect (impl.gen.phone == "phone" && impl.gen.pitch == "pitch" && impl.gen.pitchf == "pitchf"
                && impl.gen.speaker == "ds" && impl.gen.noise == "rnd" && impl.gen.lengths == "phone_lengths");
        expect (mainModel->getDescription().contains ("pitch-guided"));
        logMessage ("  one real-time chunk (" + juce::String (impl.getStreamChunkFrames() * 10) + " ms of audio) takes "
                    + juce::String (impl.chunkInferenceSeconds * 1000.0, 2) + " ms = " + juce::String (impl.getRealTimeLoad() * 100.0, 1)
                    + "% of the real-time budget");
        expect (impl.chunkInferenceSeconds > 0.0);
    }

    void testIoVariants()
    {
        beginTest ("Model I/O discovery and errors");

        juce::String error;

        // channels-first [1,C,T] encoder with a rank-2 [1,N] input + a 48 kHz model without pitch inputs
        auto nof0 = loadTempPack (packJson (modelsDir.getChildFile ("generator_48k_nof0.onnx"),
                                            modelsDir.getChildFile ("encoder_768_cf_rank2.onnx"), 48000), error);
        expect (nof0 != nullptr, error);

        if (nof0 != nullptr)
        {
            expectEquals (nof0->getModelSampleRate(), 48000);
            expect (! nof0->getImpl().usesF0);
            expectEquals (nof0->getImpl().enc.inputRank, 2);
            expect (nof0->getImpl().enc.layout == NeuralVoiceModel::Impl::EncoderIO::Layout::channelMajor);

            juce::AudioBuffer<float> in (1, 44100), out;
            const auto tone = makeSine (200.0, 44100.0, 44100, 0.5f);
            in.copyFrom (0, 0, tone.data(), 44100);
            expect (nof0->convert (in, 44100.0, {}, 100.0, out, nullptr, error), error);
            expectEquals (out.getNumSamples(), 44100);
            expectWithinAbsoluteError (estimateFrequency (out.getReadPointer (0) + 8820, 26460, 44100.0), 220.0, 1.0);
        }

        // float16 model; sample rate taken from the model (voice.json says 0)
        auto half = loadTempPack (packJson (modelsDir.getChildFile ("generator_32k_fp16.onnx"),
                                            modelsDir.getChildFile ("encoder_768.onnx"), 0), error);
        expect (half != nullptr, error);

        if (half != nullptr)
        {
            expectEquals (half->getModelSampleRate(), 32000);
            expect (half->getImpl().gen.phoneType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        }

        // 256-d encoder with a 768-d model
        auto mismatch = loadTempPack (packJson (modelsDir.getChildFile ("generator_40k.onnx"),
                                                modelsDir.getChildFile ("encoder_256.onnx"), 40000), error);
        expect (mismatch == nullptr);
        expect (error.contains ("256") && error.contains ("768"), error);
        logMessage ("  mismatch: " + error);

        // encoder id that does not exist anywhere
        auto missingEncoder = loadTempPack (packJson (modelsDir.getChildFile ("generator_40k.onnx"), {}, 40000,
                                                      "no-such-encoder"), error);
        expect (missingEncoder == nullptr);
        expect (error.contains ("no-such-encoder") && error.contains ("not found"), error);
        logMessage ("  missing encoder: " + error);

        // corrupt model file
        auto corruptDir = writeTempPack ("{}");
        tempDirs.add (corruptDir);
        corruptDir.getChildFile ("broken.onnx").replaceWithText ("this is not an ONNX model");
        corruptDir.getChildFile ("voice.json").replaceWithText (packJson (corruptDir.getChildFile ("broken.onnx"),
                                                                          modelsDir.getChildFile ("encoder_768.onnx"), 40000));
        expect (loadPack (corruptDir, error) == nullptr);
        expect (error.contains ("broken.onnx"), error);
        logMessage ("  corrupt model: " + error);

        // a model that is not an RVC generator (the encoder) used as voice model
        auto wrongModel = loadTempPack (packJson (modelsDir.getChildFile ("encoder_768.onnx"),
                                                  modelsDir.getChildFile ("encoder_768.onnx"), 40000), error);
        expect (wrongModel == nullptr);
        expect (error.contains ("not an RVC voice model"), error);

        // missing file (profile built by hand, bypassing VoiceLibrary's own check)
        VoiceProfile missing;
        missing.neuralModelFile = modelsDir.getChildFile ("does-not-exist.onnx");
        expect (NeuralVoiceModel::load (missing, error) == nullptr);
        expect (error.contains ("not found"), error);
    }

    //==============================================================================
    void testOfflineConversion()
    {
        beginTest ("Offline conversion");

        if (mainModel == nullptr)
            return;

        const double rate = 44100.0;
        const int n = (int) (rate * 3.0);
        const auto tone = makeSine (330.0, rate, n, 0.5f);
        juce::AudioBuffer<float> in (1, n), out;
        in.copyFrom (0, 0, tone.data(), n);

        // target: 220 Hz, unvoiced 1.0-1.2 s, then 330 Hz
        auto contour = [] (double fps)
        {
            std::vector<float> f0 ((size_t) (3.0 * fps));
            for (size_t i = 0; i < f0.size(); ++i)
            {
                const double t = ((double) i + 0.5) / fps;
                f0[i] = t < 1.0 ? 220.0f : (t < 1.2 ? 0.0f : 330.0f);
            }
            return f0;
        };

        for (double fps : { 100.0, 200.0, 86.13 })
        {
            juce::String error;
            const auto start = juce::Time::getMillisecondCounterHiRes();
            expect (mainModel->convert (in, rate, contour (fps), fps, out, nullptr, error), error);
            const auto elapsed = (juce::Time::getMillisecondCounterHiRes() - start) * 0.001;

            expectEquals (out.getNumChannels(), 1);
            expectEquals (out.getNumSamples(), n);

            const auto* y = out.getReadPointer (0);
            const auto at = [rate] (double t) { return juce::roundToInt (t * rate); };
            expectWithinAbsoluteError (estimateFrequency (y + at (0.2), at (0.6), rate), 220.0, 1.0);
            expectWithinAbsoluteError (estimateFrequency (y + at (1.4), at (1.4), rate), 330.0, 1.0);
            expect (rms (y + at (1.05), at (0.1)) < 0.01, "unvoiced frames must be silent");
            expectWithinAbsoluteError (rms (y + at (1.4), at (1.4)), 0.5 / std::sqrt (2.0), 0.05);

            if (juce::approximatelyEqual (fps, 100.0))
                logMessage ("  offline: 3 s clip converted in " + juce::String (elapsed * 1000.0, 1) + " ms (real-time factor "
                            + juce::String (elapsed / 3.0, 4) + ")");
        }

        beginTest ("Offline conversion of a long clip (segmented)");
        {
            const double rate48 = 48000.0;
            const int longN = (int) (rate48 * 25.0);
            auto longTone = makeSine (250.0, rate48, longN, 0.4f);
            for (int i = juce::roundToInt (11.0 * rate48); i < juce::roundToInt (11.3 * rate48); ++i)
                longTone[(size_t) i] = 0.0f;   // a quiet spot where a segment boundary should go

            juce::AudioBuffer<float> longIn (1, longN), longOut;
            longIn.copyFrom (0, 0, longTone.data(), longN);
            std::vector<float> f0 (2500, 200.0f);

            juce::String error;
            const auto start = juce::Time::getMillisecondCounterHiRes();
            expect (mainModel->convert (longIn, rate48, f0, 100.0, longOut, nullptr, error), error);
            const auto elapsed = (juce::Time::getMillisecondCounterHiRes() - start) * 0.001;
            logMessage ("  offline: 25 s clip converted in " + juce::String (elapsed, 2) + " s (real-time factor "
                        + juce::String (elapsed / 25.0, 4) + ")");

            expectEquals (longOut.getNumSamples(), longN);
            const auto* y = longOut.getReadPointer (0);

            // no dropouts or level jumps at the segment joins (except the quiet gap itself)
            int badWindows = 0;
            const int window = (int) (0.02 * rate48);
            for (int start2 = (int) (0.2 * rate48); start2 + window < (int) (24.8 * rate48); start2 += window)
            {
                const double t = start2 / rate48;
                if (t > 10.9 && t < 11.45)
                    continue;
                const auto level = rms (y + start2, window);
                if (level < 0.4 / std::sqrt (2.0) * 0.8 || level > 0.4 / std::sqrt (2.0) * 1.2)
                    ++badWindows;
            }
            expectEquals (badWindows, 0);
            expectWithinAbsoluteError (estimateFrequency (y + (int) (20.0 * rate48), (int) rate48, rate48), 200.0, 1.0);

            std::atomic<bool> cancel { true };
            expect (! mainModel->convert (longIn, rate48, f0, 100.0, longOut, &cancel, error));
            expectEquals (error, juce::String ("Cancelled"));
        }

        beginTest ("Offline conversion errors");
        {
            juce::String error;
            expect (! mainModel->convert (in, rate, {}, 100.0, out, nullptr, error));   // pitch-guided model needs f0
            expect (error.isNotEmpty());
            expect (! mainModel->convert (in, 0.0, contour (100.0), 100.0, out, nullptr, error));

            juce::AudioBuffer<float> empty (1, 0);
            expect (mainModel->convert (empty, rate, contour (100.0), 100.0, out, nullptr, error));
            expectEquals (out.getNumSamples(), 0);
        }
    }

    //==============================================================================
    /** Feeds `input` through a stream in blocks, paced to real time (or as fast as possible). */
    static std::vector<float> runStream (NeuralVoiceStream& stream, const std::vector<float>& input, double rate,
                                         int blockSize, bool realTimePacing, double* cpuSecondsOut = nullptr)
    {
        const int total = (int) input.size() + stream.getLatencySamples() + blockSize;
        std::vector<float> in ((size_t) total, 0.0f), f0 ((size_t) total), out ((size_t) total, 0.0f);
        std::copy (input.begin(), input.end(), in.begin());

        for (int i = 0; i < total; ++i)
            f0[(size_t) i] = targetPitchAt (i / rate);

        const auto startMs = juce::Time::getMillisecondCounterHiRes();
        const auto cpuStart = std::clock();

        for (int pos = 0; pos < total; pos += blockSize)
        {
            const int n = juce::jmin (blockSize, total - pos);
            stream.process (in.data() + pos, f0.data() + pos, out.data() + pos, n);

            if (realTimePacing)
                while (juce::Time::getMillisecondCounterHiRes() - startMs < 1000.0 * (pos + n) / rate)
                    juce::Thread::sleep (1);
        }

        if (cpuSecondsOut != nullptr)
            *cpuSecondsOut = (double) (std::clock() - cpuStart) / CLOCKS_PER_SEC;

        return out;
    }

    void testStreaming()
    {
        beginTest ("Streaming latency");

        if (mainModel == nullptr)
            return;

        for (double rate : { 44100.0, 48000.0 })
        {
            for (int block : { 128, 512, 1024 })
            {
                NeuralVoiceStream stream;
                stream.prepare (rate, block);
                const int latency = stream.getLatencySamples();
                stream.setModel (mainModel);
                expectEquals (stream.getLatencySamples(), latency);
                stream.reset();
                expectEquals (stream.getLatencySamples(), latency);
                stream.setModel (nullptr);
                expectEquals (stream.getLatencySamples(), latency);
                logMessage ("  latency at " + juce::String (rate, 0) + " Hz / " + juce::String (block) + " samples: "
                            + juce::String (latency) + " samples (" + juce::String (1000.0 * latency / rate, 1) + " ms)");
            }
        }

        // default context (300 ms) and a long one (1 s, most of it seen by the content encoder only)
        juce::String error;
        auto longContext = loadTempPack (packJson (modelsDir.getChildFile ("generator_40k.onnx"),
                                                   modelsDir.getChildFile ("encoder_768.onnx"), 40000, "dummy", 1000), error);
        expect (longContext != nullptr, error);

        if (longContext != nullptr)
            expect (longContext->getImpl().getStreamGeneratorSkip() > 0);

        for (auto& model : { mainModel, longContext })
        {
            if (model == nullptr)
                continue;

            beginTest ("Streaming output matches the offline conversion after the latency (context "
                       + juce::String (model->getImpl().getStreamContextFrames() * 10) + " ms)");

            const double rate = 48000.0;
            const int block = 512;
            const auto input = makeBurstInput (rate, 3.6);

            NeuralVoiceStream stream;
            stream.prepare (rate, block);
            stream.setModel (model);
            const int latency = stream.getLatencySamples();

            double cpuSeconds = 0.0;
            auto out = runStream (stream, input, rate, block, true, &cpuSeconds);
            expect (stream.isRunningInRealTime(), "the stream should keep up with real time");
            expect (allFinite (out, 4.0f));
            logMessage ("  streaming CPU (whole process, all threads): " + juce::String (100.0 * cpuSeconds / ((double) out.size() / rate), 1)
                        + "% of one core");

            // offline reference with the same target pitch
            juce::AudioBuffer<float> in (1, (int) input.size()), reference;
            in.copyFrom (0, 0, input.data(), (int) input.size());
            std::vector<float> f0 ((size_t) (3.6 * 100));
            for (size_t i = 0; i < f0.size(); ++i)
                f0[i] = targetPitchAt (((double) i + 0.5) / 100.0);

            expect (model->convert (in, rate, f0, 100.0, reference, nullptr, error), error);

            std::vector<float> aligned (input.size());
            std::copy_n (out.begin() + latency, aligned.size(), aligned.begin());
            const std::vector<float> ref (reference.getReadPointer (0), reference.getReadPointer (0) + reference.getNumSamples());

            double correlation = 0.0;
            const auto envRef = envelope (ref, rate), envOut = envelope (aligned, rate);
            const int lag = bestLag (envRef, envOut, 20, correlation);
            logMessage ("  envelope correlation " + juce::String (correlation, 4) + " at lag " + juce::String (lag * 5) + " ms");
            expect (correlation > 0.95, "envelope correlation " + juce::String (correlation));
            expect (std::abs (lag) <= 2, "streaming output is misaligned by " + juce::String (lag * 5) + " ms");

            // silence before the first burst, then the target pitch
            expect (rms (aligned.data() + juce::roundToInt (0.05 * rate), juce::roundToInt (0.2 * rate)) < 0.01);
            const auto at = [rate] (double t) { return juce::roundToInt (t * rate); };
            const auto f1 = estimateFrequency (aligned.data() + at (0.45), at (0.4), rate);
            const auto f2 = estimateFrequency (aligned.data() + at (2.0), at (0.5), rate);
            expectWithinAbsoluteError (f1, 220.0, 3.0);
            expectWithinAbsoluteError (f2, 330.0, 3.0);
            expectWithinAbsoluteError (rms (aligned.data() + at (2.0), at (0.5)), rms (ref.data() + at (2.0), at (0.5)), 0.05);
        }
    }

    void testHotSwap()
    {
        beginTest ("Model hot-swap and resets while streaming");

        if (mainModel == nullptr)
            return;

        juce::String error;
        auto other = loadTempPack (packJson (modelsDir.getChildFile ("generator_48k_nof0.onnx"),
                                             modelsDir.getChildFile ("encoder_768_cf_rank2.onnx"), 48000), error);
        auto half = loadTempPack (packJson (modelsDir.getChildFile ("generator_32k_fp16.onnx"),
                                            modelsDir.getChildFile ("encoder_768.onnx"), 32000), error);
        expect (other != nullptr && half != nullptr, error);

        const double rate = 44100.0;
        const int block = 256;
        NeuralVoiceStream stream;
        stream.prepare (rate, block);
        stream.setModel (mainModel);
        const int latency = stream.getLatencySamples();

        std::atomic<bool> stop { false };
        std::thread swapper ([&]
        {
            std::vector<std::shared_ptr<NeuralVoiceModel>> models { mainModel, other, half, nullptr };
            juce::Random rng (7);
            while (! stop.load())
            {
                stream.setModel (models[(size_t) rng.nextInt ((int) models.size())]);
                std::this_thread::sleep_for (std::chrono::milliseconds (20 + rng.nextInt (40)));
            }
        });

        const auto input = makeBurstInput (rate, 2.0);
        std::vector<float> f0 ((size_t) block, 250.0f), out ((size_t) block);
        std::vector<float> all;
        juce::Random rng (3);
        const auto startMs = juce::Time::getMillisecondCounterHiRes();

        for (int pos = 0; pos + block <= (int) input.size(); pos += block)
        {
            stream.process (input.data() + pos, f0.data(), out.data(), block);
            all.insert (all.end(), out.begin(), out.end());

            if (rng.nextInt (60) == 0)
                stream.reset();

            while (juce::Time::getMillisecondCounterHiRes() - startMs < 1000.0 * (pos + block) / rate)
                juce::Thread::sleep (1);
        }

        stop.store (true);
        swapper.join();

        expect (allFinite (all, 4.0f), "hot-swapping produced invalid samples");
        expectEquals (stream.getLatencySamples(), latency);

        // the models stay usable after being dropped by the stream
        stream.setModel (nullptr);
        juce::AudioBuffer<float> in (1, 4410), converted;
        std::vector<float> contour (10, 200.0f);
        expect (mainModel->convert (in, rate, contour, 100.0, converted, nullptr, error), error);
    }

    void testOfflineRenderDetection()
    {
        beginTest ("Offline bounce (blocks faster than real time)");

        if (mainModel == nullptr)
            return;

        const double rate = 48000.0;
        const int block = 512;
        const auto input = makeBurstInput (rate, 3.6);

        NeuralVoiceStream stream;
        stream.prepare (rate, block);
        stream.setModel (mainModel);
        const int latency = stream.getLatencySamples();

        const auto start = juce::Time::getMillisecondCounterHiRes();
        auto out = runStream (stream, input, rate, block, false);
        logMessage ("  rendered " + juce::String ((double) out.size() / rate, 2) + " s in "
                    + juce::String (juce::Time::getMillisecondCounterHiRes() - start, 0) + " ms");

        // every burst must be present (the stream waited for the worker instead of dropping chunks)
        const auto at = [rate, latency] (double t) { return latency + juce::roundToInt (t * rate); };
        for (double t : { 0.5, 1.4, 2.2, 3.1 })
            expect (rms (out.data() + at (t), juce::roundToInt (0.1 * rate)) > 0.1, "missing output at " + juce::String (t) + " s");
    }
};

static NeuralEngineTests neuralEngineTests;

} // namespace voxa::neural
