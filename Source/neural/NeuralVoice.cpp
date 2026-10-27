#include "NeuralVoice.h"
#include "RvcModel.h"

#include <limits>
#include <mutex>

namespace voxa::neural
{

//==============================================================================
NeuralVoiceModel::NeuralVoiceModel() : impl (std::make_unique<Impl>()) {}
NeuralVoiceModel::~NeuralVoiceModel() = default;

std::shared_ptr<NeuralVoiceModel> NeuralVoiceModel::load (const VoiceProfile& profile, juce::String& error,
                                                          const std::atomic<bool>* cancel)
{
    error.clear();

    if (profile.neuralModelFile == juce::File())
    {
        error = (profile.name.isNotEmpty() ? profile.name : juce::String ("This voice")) + " has no neural voice model";
        return nullptr;
    }

    auto& runtime = OnnxRuntime::getInstance();

    if (! runtime.ensureLoaded())
    {
        error = runtime.getStatusMessage();
        return nullptr;
    }

    auto config = ModelConfig::fromProfile (profile);
    std::shared_ptr<NeuralVoiceModel> model (new NeuralVoiceModel());

    if (model->impl->initialise (config, error, cancel))
    {
        // Some (integrated) GPUs run these small chunks slower than the CPU: if this one is too slow for live
        // playback, keep whichever of the two is faster.
        if (! model->impl->usesAccelerator() || model->impl->getRealTimeLoad() < 0.8)
            return model;

        config.allowAccelerator = false;
        std::shared_ptr<NeuralVoiceModel> cpu (new NeuralVoiceModel());
        juce::String cpuError;

        if (cpu->impl->initialise (config, cpuError, cancel) && cpu->impl->getRealTimeLoad() < model->impl->getRealTimeLoad())
            return cpu;

        return model;
    }

    if (! config.allowAccelerator || runtime.getImpl().getAccelerator() == Accelerator::cpu
        || (cancel != nullptr && cancel->load()))
        return nullptr;

    // The GPU could not run this model (a driver or operator problem): the CPU can.
    const auto gpuError = error;
    config.allowAccelerator = false;
    model.reset (new NeuralVoiceModel());

    if (! model->impl->initialise (config, error, cancel))
    {
        error << " (GPU: " << gpuError << ")";
        return nullptr;
    }

    DBG ("VoxaVersa AI voice: GPU failed (" << gpuError << "), using the CPU");
    return model;
}

int NeuralVoiceModel::getModelSampleRate() const noexcept   { return impl->sampleRate; }
juce::String NeuralVoiceModel::getDescription() const       { return impl->description; }
bool NeuralVoiceModel::hasNeuralPitch() const noexcept      { return impl->pitchTracker != nullptr; }

bool NeuralVoiceModel::convert (const juce::AudioBuffer<float>& monoInput, double inputSampleRate,
                                const std::vector<float>& f0Hz, double f0FrameRate,
                                juce::AudioBuffer<float>& output,
                                const std::atomic<bool>* cancel, juce::String& error,
                                const std::vector<float>* pitchShiftSemis) const
{
    error.clear();
    return impl->convert (monoInput, inputSampleRate, f0Hz, f0FrameRate, output, cancel, error, pitchShiftSemis);
}

//==============================================================================
/*  Real-time streaming
    -------------------
    Every stream lives on one absolute timeline of host samples (0 = prepare()). The audio thread writes
    input sample n (audio, target f0, pitch shift) into ring slot n and publishes `inputWritten`; it reads output
    position n from the output ring if the worker has already published it (`outputWritten`), otherwise
    it outputs silence with coverage 0 (the caller fills it with its DSP voice). Nothing on the audio thread
    allocates, locks or waits (except when the host renders offline, see updatePace()).

    The worker converts chunk k = frames [40k, 40k + 40) (0.4 s) once its input has arrived:
      - context + tail + chunk is band-limited to 16 kHz and fed through the content encoder & generator
        (one fixed-size inference, so ONNX Runtime can reuse its memory plan),
      - the generator output is resampled to the host rate for [e0 - X - S, e1) (X = crossfade,
        S = SOLA search window),
      - SOLA picks the offset in [0, S] that best continues the previous chunk's tail, which is then
        crossfaded in (correlation-aware gains), and exactly e1 - e0 samples are emitted, so the stream
        never drifts,
      - those samples go to ring positions [e0 + outputOffset, e1 + outputOffset).

    The latency L is therefore constant: chunk + crossfade + S/2 + resampler lookahead + one host block +
    inference headroom + polling interval. A chunk whose output would arrive too late is skipped (coverage
    0), which keeps the stream aligned; isRunningInRealTime() then reports false.

    The left context of each chunk is the configured one while the host renders offline (best quality, the
    audio thread waits for it). Live, it is the longest one this computer converts comfortably in real time
    (measured when the model loads, NeuralVoiceModel::Impl::getLiveContextFrames()), and the next shorter
    one whenever chunks keep missing their deadline (a busier host). */
namespace
{
    int64_t floorDiv (int64_t a, int64_t b) noexcept
    {
        auto q = a / b;

        if ((a % b != 0) && ((a < 0) != (b < 0)))
            --q;

        return q;
    }

    size_t nextPowerOfTwo (int64_t x) noexcept
    {
        size_t p = 1;
        while ((int64_t) p < x)
            p <<= 1;
        return p;
    }
} // namespace

struct NeuralVoiceStream::Impl : private juce::Thread
{
    Impl() : juce::Thread ("VoxaVersa AI voice") {}

    ~Impl() override
    {
        stopWorker();
    }

    //==============================================================================
    void prepare (double sampleRate, int maxBlockSize)
    {
        stopWorker();
        prepared.store (false, std::memory_order_release);

        hostRate = juce::jlimit (8000, 768000, juce::roundToInt (sampleRate > 0.0 ? sampleRate : 44100.0));
        maxBlock = juce::jlimit (1, 1 << 16, maxBlockSize);
        crossfadeLength = juce::jmax (8, juce::roundToInt (streaming::crossfadeSeconds * hostRate));
        searchLength = juce::jmax (8, juce::roundToInt (streaming::searchSeconds * hostRate));
        toEncoder.prepare (hostRate, dsp::encoderSampleRate);

        const int lookahead = toEncoder.getHalfWidth() + 2;
        const auto blockLength = (int) (((int64_t) streaming::blockFrames * hostRate + dsp::framesPerSecond - 1) / dsp::framesPerSecond);
        const int headroom = juce::roundToInt (streaming::headroomSeconds * hostRate);
        const int poll = (int) std::ceil (streaming::pollSeconds * hostRate);

        const int newLatency = blockLength + crossfadeLength + searchLength / 2 + lookahead + maxBlock + headroom + poll;
        latency.store (newLatency);
        outputOffset = newLatency - crossfadeLength - searchLength / 2;

        const auto inputSize = nextPowerOfTwo (juce::jmax ((int64_t) hostRate * 4, (int64_t) newLatency * 2 + 8 * (int64_t) maxBlock));
        const auto outputSize = nextPowerOfTwo ((int64_t) newLatency * 2 + 4 * (int64_t) maxBlock + hostRate);
        inputRing.assign (inputSize, 0.0f);
        f0Ring.assign (inputSize, 0.0f);
        shiftRing.assign (inputSize, std::numeric_limits<float>::quiet_NaN());
        outputRing.assign (outputSize, 0.0f);
        coverageRing.assign (outputSize, 0.0f);
        coverageScratch.assign ((size_t) maxBlock, 0.0f);
        inputMask = (int64_t) inputSize - 1;
        outputMask = (int64_t) outputSize - 1;

        // shared counters
        inputWritten.store (0);
        audioReadPosition.store (0);
        outputWritten.store (outputOffset);
        resetPosition.store (0);
        underruns.store (0);
        missedChunks.store (0);
        realTime.store (false);
        offlineMode.store (false);

        // audio-thread state
        audioPosition = 0;
        validFrom = newLatency;
        pace = {};
        coverageGain = 0.0f;
        coverageStep = 1.0f / juce::jmax (1.0f, 0.005f * (float) hostRate);

        // worker state
        nextChunk = 0;
        seenGeneration = resetGeneration.load();
        seenUnderruns = 0;
        historyValidFrom = 0;
        onTimeStreak = 0;
        hasSola = false;
        liveLevel = 0;
        missHistory = 0;
        solaBuffer.assign ((size_t) crossfadeLength, 0.0f);
        chunkOutput.assign ((size_t) (blockLength + 1 + crossfadeLength + searchLength), 0.0f);
        chunkCoverage.assign ((size_t) blockLength + 1, 0.0f);

        prepared.store (true, std::memory_order_release);
        startThread (juce::Thread::Priority::high);
    }

    void setModel (std::shared_ptr<NeuralVoiceModel> newModel)
    {
        std::shared_ptr<NeuralVoiceModel> replaced;   // released outside the lock
        modelSet.store (newModel != nullptr);

        {
            const std::lock_guard<std::mutex> sl (modelLock);
            replaced = std::move (pendingModel);
            pendingModel = std::move (newModel);
            hasPendingModel = true;
        }

        notify();
    }

    bool hasModel() const noexcept               { return modelSet.load(); }
    int getLatencySamples() const noexcept       { return latency.load(); }
    bool isRunningInRealTime() const noexcept    { return modelSet.load() && realTime.load(); }
    uint32_t getNumMissedChunks() const noexcept { return missedChunks.load (std::memory_order_relaxed); }
    void setNonRealtime (bool b) noexcept        { hostOffline.store (b, std::memory_order_relaxed); }

    //==============================================================================
    // Audio thread

    void process (const float* input, const float* targetF0, float* output, int numSamples, float* coverage,
                  const float* shiftSemis) noexcept
    {
        if (! prepared.load (std::memory_order_acquire))
        {
            juce::FloatVectorOperations::clear (output, numSamples);
            if (coverage != nullptr)
                juce::FloatVectorOperations::clear (coverage, numSamples);
            return;
        }

        for (int done = 0; done < numSamples;)
        {
            const int n = juce::jmin (maxBlock, numSamples - done);
            processBlock (input != nullptr ? input + done : nullptr,
                          targetF0 != nullptr ? targetF0 + done : nullptr,
                          shiftSemis != nullptr ? shiftSemis + done : nullptr,
                          output + done, coverage != nullptr ? coverage + done : coverageScratch.data(), n);
            done += n;
        }
    }

    void reset() noexcept
    {
        if (! prepared.load (std::memory_order_acquire))
            return;

        resetPosition.store (audioPosition, std::memory_order_relaxed);
        resetGeneration.fetch_add (1, std::memory_order_release);
        validFrom = audioPosition + latency.load();
    }

private:
    //==============================================================================
    void processBlock (const float* input, const float* targetF0, const float* shiftSemis, float* output, float* coverage,
                       int n) noexcept
    {
        const auto position = audioPosition;

        // input first: `output` may alias `input`
        writeRing (inputRing, inputMask, position, input, n);
        writeRing (f0Ring, inputMask, position, targetF0, n);
        writeRing (shiftRing, inputMask, position, shiftSemis, n, std::numeric_limits<float>::quiet_NaN());
        inputWritten.store (position + n, std::memory_order_release);

        const bool active = modelSet.load (std::memory_order_relaxed);

        if (active)
            updatePace (position + n, n);

        const auto written = outputWritten.load (std::memory_order_acquire);
        const auto from = juce::jmax (position, validFrom);
        const auto to = juce::jmin (position + n, written);

        juce::FloatVectorOperations::clear (output, n);
        juce::FloatVectorOperations::clear (coverage, n);

        if (to > from)
        {
            readRing (outputRing, outputMask, from, output + (from - position), (int) (to - from));
            readRing (coverageRing, outputMask, from, coverage + (from - position), (int) (to - from));
        }

        smoothCoverage (output, coverage, n);

        if (active && position + n > juce::jmax (validFrom, written))
            underruns.fetch_add (1, std::memory_order_relaxed);

        audioPosition = position + n;
        audioReadPosition.store (audioPosition, std::memory_order_release);
    }

    /** The worker's crossfades are already in `coverage`. What is left are hard edges: a chunk that arrived
        partly too late starts abruptly, and output can stop where the next chunk is missing. Both are turned
        into 5 ms ramps (the converted voice fades in; the caller's fill fades in where it stops). */
    void smoothCoverage (float* output, float* coverage, int n) noexcept
    {
        for (int i = 0; i < n; ++i)
        {
            const float c = coverage[i];

            if (c > coverageGain)
            {
                coverageGain = juce::jmin (c, coverageGain + coverageStep);
                output[i] *= coverageGain / c;
            }
            else
            {
                coverageGain = juce::jmax (c, coverageGain - coverageStep);
            }

            coverage[i] = coverageGain;
        }
    }

    /** The host renders offline (it says so, or blocks arrive much faster than real time): wait for the
        worker, so bounces/exports are complete. In real time this never blocks. */
    void updatePace (int64_t endPosition, int n) noexcept
    {
        const auto now = juce::Time::getMillisecondCounterHiRes();

        if (pace.windowStart < 0.0)
            pace.windowStart = now;

        pace.samples += n;
        ++pace.blocks;

        if (pace.blocks >= 4)
        {
            const auto busyMs = juce::jmax (0.001, now - pace.windowStart - pace.waitedMs);
            const auto speed = (1000.0 * (double) pace.samples / hostRate) / busyMs;
            pace.offline = pace.offline ? speed > 1.5 : speed > 3.0;

            if (now - pace.windowStart > 250.0)
                pace = { now, 0, 0, 0.0, pace.offline };
        }

        const bool hostSaysOffline = hostOffline.load (std::memory_order_relaxed);
        const bool offline = hostSaysOffline || pace.offline;
        offlineMode.store (offline, std::memory_order_relaxed);

        if (offline && endPosition > validFrom && isThreadRunning())
        {
            // generous when the host says it renders offline: a slow computer converts slower than real time
            const double timeoutMs = hostSaysOffline ? 10000.0 : 1000.0;

            while (outputWritten.load (std::memory_order_acquire) < endPosition && modelSet.load()
                   && juce::Time::getMillisecondCounterHiRes() - now < timeoutMs)
                juce::Thread::sleep (1);

            pace.waitedMs += juce::Time::getMillisecondCounterHiRes() - now;
        }
    }

    static void writeRing (std::vector<float>& ring, int64_t mask, int64_t position, const float* source, int n,
                           float fill = 0.0f) noexcept
    {
        const auto start = (size_t) (position & mask);
        const auto first = juce::jmin ((size_t) n, ring.size() - start);

        if (source != nullptr)
        {
            std::copy_n (source, first, ring.data() + start);
            std::copy_n (source + first, (size_t) n - first, ring.data());
        }
        else
        {
            std::fill_n (ring.data() + start, first, fill);
            std::fill_n (ring.data(), (size_t) n - first, fill);
        }
    }

    static void readRing (const std::vector<float>& ring, int64_t mask, int64_t position, float* dest, int n) noexcept
    {
        const auto start = (size_t) (position & mask);
        const auto first = juce::jmin ((size_t) n, ring.size() - start);
        std::copy_n (ring.data() + start, first, dest);
        std::copy_n (ring.data(), (size_t) n - first, dest + first);
    }

    //==============================================================================
    // Worker thread

    void run() override
    {
        while (! threadShouldExit())
        {
            adoptPendingModel();

            // A reset() that happened before these samples were pushed is visible after this acquire.
            const auto available = inputWritten.load (std::memory_order_acquire);
            checkForReset();

            while (! threadShouldExit() && isChunkReady (available))
                processNextChunk();

            wait (streaming::pollSeconds * 1000.0);
        }
    }

    void stopWorker()
    {
        signalThreadShouldExit();
        notify();
        stopThread (10000);
    }

    void adoptPendingModel()
    {
        std::shared_ptr<NeuralVoiceModel> incoming;

        {
            const std::lock_guard<std::mutex> sl (modelLock);

            if (! hasPendingModel)
                return;

            incoming = std::move (pendingModel);
            hasPendingModel = false;
        }

        std::swap (model, incoming);   // the previous model is released here, on the worker thread
        incoming.reset();
        liveLevel = 0;
        missHistory = 0;
        onTimeStreak = 0;
        reportedError = false;
        realTime.store (false);
    }

    void checkForReset()
    {
        const auto generation = resetGeneration.load (std::memory_order_acquire);

        if (generation != seenGeneration)
        {
            seenGeneration = generation;
            historyValidFrom = resetPosition.load (std::memory_order_relaxed);
            std::fill (solaBuffer.begin(), solaBuffer.end(), 0.0f);
            hasSola = false;
        }
    }

    int64_t hostIndexForFrame (int64_t frame) const noexcept
    {
        return floorDiv (frame * hostRate, dsp::framesPerSecond);
    }

    bool isChunkReady (int64_t available) const noexcept
    {
        const auto endFrame = (nextChunk + 1) * streaming::blockFrames;
        return available > toEncoder.lastInputFor (endFrame * dsp::samplesPerFrame16k - 1);
    }

    void readInput (const std::vector<float>& ring, int64_t from, int count, std::vector<float>& dest,
                    float outside = 0.0f) const
    {
        dest.resize ((size_t) count);

        for (int i = 0; i < count; ++i)
        {
            const auto p = from + i;
            dest[(size_t) i] = p >= historyValidFrom && p >= 0 ? ring[(size_t) (p & inputMask)] : outside;
        }
    }

    void processNextChunk()
    {
        const auto chunk = nextChunk++;
        const auto startFrame = chunk * streaming::blockFrames;
        const auto endFrame = startFrame + streaming::blockFrames;
        const auto e0 = hostIndexForFrame (startFrame), e1 = hostIndexForFrame (endFrame);
        const auto emitLength = (int) (e1 - e0);
        const auto ringStart = e0 + outputOffset;
        const auto regionStart = e0 - crossfadeLength - searchLength;
        const auto regionLength = (int) (e1 - regionStart);

        const bool late = audioReadPosition.load (std::memory_order_acquire) >= ringStart + emitLength;
        const auto* m = model != nullptr ? &model->getImpl() : nullptr;
        const bool live = ! offlineMode.load (std::memory_order_relaxed);
        bool produced = false;

        if (m != nullptr && ! late)
        {
            readInput (inputRing, regionStart, regionLength, hostScratch);

            if (! dsp::isSilent (hostScratch.data(), regionLength))
            {
                const auto& liveContexts = m->getLiveContextFrames();
                const int context = live && ! liveContexts.empty() ? liveContexts[juce::jmin (liveLevel, liveContexts.size() - 1)]
                                                                   : m->getStreamContextFrames();
                produced = infer (*m, context, endFrame, regionStart, regionLength);
            }
        }

        if (! produced)
            std::fill_n (chunkOutput.begin(), regionLength, 0.0f);

        int offset = searchLength / 2;
        float r = 0.0f;

        if (produced && hasSola)
            offset = dsp::findBestOffset (solaBuffer.data(), chunkOutput.data(), crossfadeLength, searchLength, r);

        auto* emitted = chunkOutput.data() + offset;
        dsp::crossfade (solaBuffer.data(), emitted, emitted, crossfadeLength, hasSola ? r : 0.0f);

        // Coverage: 1 while converted voice continues, 0 where there is none, and where it fades in from (or out
        // to) silence the complement of the gain the caller's fill needs to crossfade with it.
        std::fill_n (chunkCoverage.begin(), emitLength, produced ? 1.0f : 0.0f);

        if (produced != hasSola)
            for (int i = 0; i < crossfadeLength; ++i)
                chunkCoverage[(size_t) i] = 1.0f - dsp::crossfadeGain (produced ? crossfadeLength - 1 - i : i, crossfadeLength);

        if (produced)
            std::copy_n (emitted + emitLength, crossfadeLength, solaBuffer.data());
        else
            std::fill (solaBuffer.begin(), solaBuffer.end(), 0.0f);

        hasSola = produced;

        writeRing (outputRing, outputMask, ringStart, emitted, emitLength);
        writeRing (coverageRing, outputMask, ringStart, chunkCoverage.data(), emitLength);
        outputWritten.store (ringStart + emitLength, std::memory_order_release);

        const bool missedDeadline = late || audioReadPosition.load (std::memory_order_acquire) > ringStart;
        const auto currentUnderruns = underruns.load (std::memory_order_relaxed);

        if (m == nullptr || missedDeadline || currentUnderruns != seenUnderruns)
        {
            onTimeStreak = 0;
            realTime.store (false);
        }
        else if (++onTimeStreak >= 2)
        {
            realTime.store (true);
        }

        seenUnderruns = currentUnderruns;

        if (m != nullptr && live)
            trackDeadline (*m, missedDeadline);
    }

    /** Live: after 3 missed deadlines within 20 chunks (8 s), continue with the next shorter context. */
    void trackDeadline (const NeuralVoiceModel::Impl& m, bool missed) noexcept
    {
        missHistory = (missHistory << 1) | (missed ? 1u : 0u);

        if (! missed)
            return;

        missedChunks.fetch_add (1, std::memory_order_relaxed);

        if (juce::countNumberOfBits (missHistory & 0xfffffu) >= 3 && liveLevel + 1 < m.getLiveContextFrames().size())
        {
            ++liveLevel;
            missHistory = 0;
            DBG ("VoxaVersa AI voice: live context " << m.getLiveContextFrames()[liveLevel] * 10 << " ms");
        }
    }

    bool infer (const NeuralVoiceModel::Impl& m, int contextFrames, int64_t endFrame, int64_t regionStart, int regionLength)
    {
        const int chunkFrames = NeuralVoiceModel::Impl::chunkFramesFor (contextFrames);
        const auto segmentStart = endFrame - chunkFrames;
        const auto first16k = segmentStart * dsp::samplesPerFrame16k;
        const int num16k = chunkFrames * dsp::samplesPerFrame16k;

        // Input around the chunk, band-limited to 16 kHz. Samples after the chunk are not known yet.
        const auto hostFrom = toEncoder.firstInputFor (first16k);
        const auto hostTo = toEncoder.lastInputFor (first16k + num16k - 1) + 1;
        readInput (inputRing, hostFrom, (int) (hostTo - hostFrom), hostScratch);

        audio16k.resize ((size_t) (num16k + dsp::encoderRightPad16k));
        toEncoder.process (hostScratch.data(), hostFrom, (int) (hostTo - hostFrom), audio16k.data(), first16k, num16k);
        std::fill (audio16k.begin() + num16k, audio16k.end(), 0.0f);

        // Target pitch of each 10 ms frame
        const auto f0From = hostIndexForFrame (segmentStart);
        readInput (f0Ring, f0From, (int) (hostIndexForFrame (endFrame) - f0From), f0Scratch);
        f0Frames.resize ((size_t) chunkFrames);

        for (int i = 0; i < chunkFrames; ++i)
        {
            const auto a = hostIndexForFrame (segmentStart + i) - f0From;
            const auto b = hostIndexForFrame (segmentStart + i + 1) - f0From;
            f0Frames[(size_t) i] = dsp::frameF0 (f0Scratch.data() + a, (int) (b - a));
        }

        juce::String error;

        const int generatorSkip = NeuralVoiceModel::Impl::generatorSkipFor (contextFrames);

        // The input's own pitch from the neural tracker (with the caller's shift), for the frames the generator sings.
        if (m.pitchTracker != nullptr)
        {
            readInput (shiftRing, f0From, (int) (hostIndexForFrame (endFrame) - f0From), f0Scratch,
                       std::numeric_limits<float>::quiet_NaN());
            shiftFrames.resize ((size_t) chunkFrames);
            bool any = false;

            for (int i = 0; i < chunkFrames; ++i)
            {
                const auto a = hostIndexForFrame (segmentStart + i) - f0From;
                const auto b = hostIndexForFrame (segmentStart + i + 1) - f0From;
                shiftFrames[(size_t) i] = dsp::frameShift (f0Scratch.data() + a, (int) (b - a));
                any = any || (i >= generatorSkip && ! std::isnan (shiftFrames[(size_t) i]));
            }

            if (any && ! m.applyNeuralPitch (audio16k.data(), chunkFrames, generatorSkip, shiftFrames.data(), f0Frames.data(),
                                             workspace, error))
            {
                DBG ("VoxaVersa AI voice: pitch tracker: " << error);   // the target pitch stays
                error.clear();
            }
        }

        if (! m.runFrames (audio16k.data(), chunkFrames, f0Frames.data(), segmentStart, workspace, false, error, generatorSkip))
        {
            if (! reportedError)
            {
                DBG ("VoxaVersa AI voice: " << error);
                reportedError = true;
            }

            return false;
        }

        if (! toHost.isPrepared() || toHost.getInputRate() != m.sampleRate)
            toHost.prepare (m.sampleRate, hostRate);

        toHost.process (workspace.audio.data(), (segmentStart + generatorSkip) * m.hopSize, (int) workspace.audio.size(),
                        chunkOutput.data(), regionStart, regionLength);
        return true;
    }

    //==============================================================================
    // Configuration (written by prepare() while the worker is stopped)
    std::atomic<bool> prepared { false };
    std::atomic<int> latency { 0 };
    int hostRate = 44100, maxBlock = 512, crossfadeLength = 0, searchLength = 0;
    int64_t outputOffset = 0;
    dsp::Resampler toEncoder;

    // Lock-free timeline rings (audio thread -> worker: input & f0, worker -> audio thread: output)
    std::vector<float> inputRing, f0Ring, shiftRing, outputRing, coverageRing;
    int64_t inputMask = 0, outputMask = 0;
    std::atomic<int64_t> inputWritten { 0 }, audioReadPosition { 0 }, outputWritten { 0 }, resetPosition { 0 };
    std::atomic<uint32_t> resetGeneration { 0 }, underruns { 0 }, missedChunks { 0 };
    std::atomic<bool> realTime { false }, modelSet { false }, hostOffline { false }, offlineMode { false };

    // Audio thread only
    struct Pace
    {
        double windowStart = -1.0;
        int64_t samples = 0;
        int blocks = 0;
        double waitedMs = 0.0;
        bool offline = false;
    };

    int64_t audioPosition = 0, validFrom = 0;
    Pace pace;
    float coverageGain = 0.0f, coverageStep = 0.01f;
    std::vector<float> coverageScratch;

    // Message thread -> worker
    std::mutex modelLock;
    std::shared_ptr<NeuralVoiceModel> pendingModel;
    bool hasPendingModel = false;

    // Worker only
    std::shared_ptr<NeuralVoiceModel> model;
    int64_t nextChunk = 0, historyValidFrom = 0;
    uint32_t seenGeneration = 0, seenUnderruns = 0, missHistory = 0;
    int onTimeStreak = 0;
    size_t liveLevel = 0;
    bool hasSola = false, reportedError = false;
    dsp::Resampler toHost;
    NeuralVoiceModel::Impl::Workspace workspace;
    std::vector<float> solaBuffer, chunkOutput, chunkCoverage, hostScratch, f0Scratch, audio16k, f0Frames, shiftFrames;

    JUCE_DECLARE_NON_COPYABLE (Impl)
};

//==============================================================================
NeuralVoiceStream::NeuralVoiceStream() : impl (std::make_unique<Impl>()) {}
NeuralVoiceStream::~NeuralVoiceStream() = default;

void NeuralVoiceStream::prepare (double hostSampleRate, int maxBlockSize)        { impl->prepare (hostSampleRate, maxBlockSize); }
void NeuralVoiceStream::setModel (std::shared_ptr<NeuralVoiceModel> model)       { impl->setModel (std::move (model)); }
bool NeuralVoiceStream::hasModel() const noexcept                                 { return impl->hasModel(); }
int NeuralVoiceStream::getLatencySamples() const noexcept                         { return impl->getLatencySamples(); }
bool NeuralVoiceStream::isRunningInRealTime() const noexcept                      { return impl->isRunningInRealTime(); }
uint32_t NeuralVoiceStream::getNumMissedChunks() const noexcept                   { return impl->getNumMissedChunks(); }
void NeuralVoiceStream::setNonRealtime (bool isNonRealtime) noexcept              { impl->setNonRealtime (isNonRealtime); }
void NeuralVoiceStream::reset() noexcept                                          { impl->reset(); }

void NeuralVoiceStream::process (const float* input, const float* targetF0Hz, float* output, int numSamples,
                                 float* coverage, const float* pitchShiftSemis) noexcept
{
    impl->process (input, targetF0Hz, output, numSamples, coverage, pitchShiftSemis);
}

} // namespace voxa::neural
