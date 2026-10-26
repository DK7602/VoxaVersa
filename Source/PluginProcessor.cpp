#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "core/AiDisclosure.h"
#include "core/AudioUtils.h"
#include "core/Watermark.h"
#include "dsp/VoiceChain.h"
#include "dsp/VoiceGenderEstimator.h"
#include "neural/NeuralVoice.h"
#include "neural/NeuralRuntime.h"

namespace voxa
{

namespace
{
    /** A mono audio clip kept at its original rate plus a copy resampled to the session rate. */
    struct AudioClip
    {
        juce::AudioBuffer<float> original;
        double originalRate = 48000.0;
        juce::AudioBuffer<float> playback;   // mono, at the processor sample rate
        juce::String name;
        juce::File file;
        int gender = 0;   // 0 unknown, 1 male, 2 female (analysed once on load)

        void resampleTo (double rate)
        {
            playback = resampleBuffer (original, originalRate, rate);
        }
    };

    struct SingTake
    {
        AudioClip clip;
        std::vector<sing::SungNote> notes;
        double bpm = 120.0;
        bool neuralRendered = false;
        sing::SingResult result;   // guide render (for MIDI export)
    };

    struct CapturedMidiEvent
    {
        double beat = 0.0;
        int note = 60;
        float velocity = 0.0f;   // 0 = note off
    };

    /** Raw parameter pointers for lock-free reads on the audio thread. */
    struct ParamCache
    {
        explicit ParamCache (juce::AudioProcessorValueTreeState& s)
        {
            auto get = [&s] (const char* id)
            {
                auto* p = s.getRawParameterValue (id);
                jassert (p != nullptr);
                return p;
            };
            namespace P = ParamID;
            power = get (P::power); mode = get (P::mode); inputGain = get (P::inputGain); outputGain = get (P::outputGain);
            mix = get (P::mix); autoLevel = get (P::autoLevel); quality = get (P::quality); aiEngine = get (P::aiEngine);
            sourceVoice = get (P::sourceVoice);
            pitch = get (P::pitch); fineTune = get (P::fineTune); gender = get (P::gender); age = get (P::age);
            tone = get (P::tone); style = get (P::style); voiceBlend = get (P::voiceBlend);
            pitchPreserve = get (P::pitchPreserve); formantShift = get (P::formantShift); expression = get (P::expression);
            vibe = get (P::vibe); intensity = get (P::intensity);
            vibratoDepth = get (P::vibratoDepth); vibratoRate = get (P::vibratoRate); breath = get (P::breath); rasp = get (P::rasp);
            tuneOn = get (P::tuneOn); key = get (P::key); scale = get (P::scale); retuneSpeed = get (P::retuneSpeed);
            humanize = get (P::humanize); midiControl = get (P::midiControl);
            harmonyOn = get (P::harmonyOn);
            harmInterval[0] = get (P::harm1Interval); harmLevel[0] = get (P::harm1Level); harmPan[0] = get (P::harm1Pan);
            harmInterval[1] = get (P::harm2Interval); harmLevel[1] = get (P::harm2Level); harmPan[1] = get (P::harm2Pan);
            harmInterval[2] = get (P::harm3Interval); harmLevel[2] = get (P::harm3Level); harmPan[2] = get (P::harm3Pan);
            doublerOn = get (P::doublerOn); doublerAmount = get (P::doublerAmount); doublerWidth = get (P::doublerWidth);
            lowCut = get (P::lowCut); deEsser = get (P::deEsser); compressor = get (P::compressor); presence = get (P::presence);
            air = get (P::air); saturation = get (P::saturation); delayMix = get (P::delayMix); delayTime = get (P::delayTime);
            delayFeedback = get (P::delayFeedback); reverbMix = get (P::reverbMix); reverbSize = get (P::reverbSize);
            singTempo = get (P::singTempo); singSyncHost = get (P::singSyncHost); melodySource = get (P::melodySource);
        }

        std::atomic<float>* power, *mode, *inputGain, *outputGain, *mix, *autoLevel, *quality, *aiEngine, *sourceVoice,
                          *pitch, *fineTune, *gender, *age, *tone, *style, *voiceBlend,
                          *pitchPreserve, *formantShift, *expression, *vibe, *intensity,
                          *vibratoDepth, *vibratoRate, *breath, *rasp,
                          *tuneOn, *key, *scale, *retuneSpeed, *humanize, *midiControl,
                          *harmonyOn, *harmInterval[3], *harmLevel[3], *harmPan[3],
                          *doublerOn, *doublerAmount, *doublerWidth,
                          *lowCut, *deEsser, *compressor, *presence, *air, *saturation,
                          *delayMix, *delayTime, *delayFeedback, *reverbMix, *reverbSize,
                          *singTempo, *singSyncHost, *melodySource;

        static bool on (const std::atomic<float>* p) noexcept  { return p->load (std::memory_order_relaxed) > 0.5f; }
        static float v (const std::atomic<float>* p) noexcept  { return p->load (std::memory_order_relaxed); }
        static int idx (const std::atomic<float>* p) noexcept  { return juce::roundToInt (p->load (std::memory_order_relaxed)); }
    };

    dsp::ChainSettings buildChainSettings (const ParamCache& p, const VoiceTraits& traits, double bpm, int midiNote, bool singMode)
    {
        using PC = ParamCache;
        dsp::ChainSettings s;

        auto& m = s.pitch;
        m.pitchSemis = PC::v (p.pitch);
        m.fineCents = PC::v (p.fineTune);
        m.tuneOn = PC::on (p.tuneOn) && ! singMode;
        m.keyRoot = PC::idx (p.key) - 1;
        m.scale = (ScaleType) juce::jlimit (0, (int) ScaleType::NumScales - 1, PC::idx (p.scale));
        m.retuneMs = PC::v (p.retuneSpeed);
        m.humanize = PC::v (p.humanize) * 0.01f;
        m.midiNote = PC::on (p.midiControl) && ! singMode ? midiNote : -1;
        m.pitchPreserve = PC::on (p.pitchPreserve);
        m.expression = PC::on (p.expression);
        m.styleScale = 1.0f + PC::v (p.style) * 0.01f;
        m.age = PC::v (p.age) * 0.01f;
        m.userVibratoCents = PC::v (p.vibratoDepth);
        m.userVibratoRate = PC::v (p.vibratoRate);
        m.sourceGender = PC::idx (p.sourceVoice);
        m.sourceIsSynth = singMode;
        m.traits = traits;

        s.formantShift = PC::on (p.formantShift);
        s.intensity = PC::v (p.intensity) * 0.01f;
        s.gender = PC::v (p.gender) * 0.01f;
        s.age = PC::v (p.age) * 0.01f;
        s.tone = PC::v (p.tone) * 0.01f;
        s.style = PC::v (p.style) * 0.01f;
        s.vibe = PC::on (p.vibe);
        s.breath = PC::v (p.breath) * 0.01f;
        s.rasp = PC::v (p.rasp) * 0.01f;

        s.harmonyOn = PC::on (p.harmonyOn);
        for (int k = 0; k < 3; ++k)
        {
            s.harmonyInterval[k] = (HarmonyInterval) juce::jlimit (0, (int) HarmonyInterval::NumIntervals - 1, PC::idx (p.harmInterval[k]));
            s.harmonyLevel[k] = PC::v (p.harmLevel[k]) * 0.01f;
            s.harmonyPan[k] = PC::v (p.harmPan[k]) * 0.01f;
        }

        s.doublerOn = PC::on (p.doublerOn);
        s.doublerAmount = PC::v (p.doublerAmount) * 0.01f;
        s.doublerWidth = PC::v (p.doublerWidth) * 0.01f;

        s.lowCutHz = PC::v (p.lowCut);
        s.deEsser = PC::v (p.deEsser) * 0.01f;
        s.compressor = PC::v (p.compressor) * 0.01f;
        s.presenceDb = PC::v (p.presence);
        s.airDb = PC::v (p.air);
        s.saturation = PC::v (p.saturation) * 0.01f;
        s.delayMix = PC::v (p.delayMix) * 0.01f;
        s.delayDivision = PC::idx (p.delayTime);
        s.delayFeedback = PC::v (p.delayFeedback) * 0.01f;
        s.reverbMix = PC::v (p.reverbMix) * 0.01f;
        s.reverbSize = PC::v (p.reverbSize) * 0.01f;
        s.bpm = bpm;
        s.useNeural = PC::on (p.aiEngine);
        return s;
    }

    juce::String sanitiseForFileName (const juce::String& s)
    {
        return juce::File::createLegalFileName (s).replaceCharacter (' ', '_');
    }
}

//==============================================================================
struct VoxaVersaProcessor::Impl : private juce::Timer
{
    explicit Impl (VoxaVersaProcessor& p) : owner (p), params (p.apvts)
    {
        formatManager.registerBasicFormats();
        startTimerHz (20);
    }

    ~Impl() override
    {
        *alive = false;
        stopTimer();
        cancelJobs.store (true);
        cancelRender.store (true);
        cancelSing.store (true);
        pool.removeAllJobs (true, 10000);
        stopRecordingInternal();
        chain.setNeuralStream (nullptr);
    }

    VoxaVersaProcessor& owner;
    ParamCache params;
    std::shared_ptr<bool> alive = std::make_shared<bool> (true);
    bool lastAiEngine = true;
    int lastQuality = -1;
    juce::String lastLyrics;
    bool stateJustLoaded = false;
    juce::AudioFormatManager formatManager;

    // DSP
    dsp::VoiceChain chain;
    neural::NeuralVoiceStream neuralStream;
    sing::LiveSinger liveSinger;
    int maxBlock = 512;
    bool prepared = false;

    // Voice traits (message thread writes, audio thread try-reads)
    juce::SpinLock traitsLock;
    VoiceTraits traitsA, traitsB;
    bool hasBlend = false;
    VoiceTraits audioTraitsA, audioTraitsB;
    bool audioHasBlend = false;

    // Neural model
    std::shared_ptr<neural::NeuralVoiceModel> currentModel;
    juce::String currentModelVoiceId, neuralMessage;
    std::atomic<bool> neuralAttached { false };
    int modelGeneration = 0;

    // Clips
    juce::SpinLock clipLock;
    std::unique_ptr<AudioClip> sample;
    std::unique_ptr<SingTake> take;
    std::atomic<double> samplePos { 0.0 }, previewPos { 0.0 };

    // MIDI
    std::array<int, 128> heldVelocity {};
    std::array<int, 16> heldStack {};
    int heldCount = 0;
    std::array<CapturedMidiEvent, 8192> captured {};
    std::atomic<int> capturedCount { 0 };
    double captureClockBeats = 0.0;

    // Hum recording
    juce::AudioBuffer<float> humBuffer;
    std::atomic<int> humSamples { 0 };

    // Mix / gains
    dsp::AutoLevel autoLevel;
    WatermarkEmbedder watermark;
    dsp::Smoothed inGain, outGain, mixAmount, powerFade;
    juce::AudioBuffer<float> work;      // [0]=mono source [1]=wetL [2]=wetR [3]=dry
    std::vector<float> dryDelay;
    int dryWrite = 0, dryDelaySamples = 0;

    // Host state
    bool wasPlaying = false;

    // Offline jobs
    juce::ThreadPool pool { 1 };
    std::atomic<bool> cancelJobs { false }, cancelRender { false }, cancelSing { false };
    juce::CriticalSection resultLock;
    juce::File lastRenderedFile, lastRecording;
    juce::String lastRenderError, singError;
    juce::uint32 singSeed = 1;

    // Recording
    juce::TimeSliceThread writerThread { "VoxaVersa recorder" };
    std::unique_ptr<juce::AudioFormatWriter::ThreadedWriter> threadedWriter;
    juce::CriticalSection writerLock;
    juce::AudioFormatWriter::ThreadedWriter* activeWriter = nullptr;
    std::atomic<juce::int64> recordedSamples { 0 };
    juce::File recordingFile;

    // Pending work flagged by the audio thread, handled on the message thread timer.
    std::atomic<bool> qualityChangePending { false }, previewFinished { false }, sampleFinished { false };

    //==========================================================================
    void timerCallback() override
    {
        if (qualityChangePending.exchange (false))
            owner.updateLatency();

        const bool ai = ParamCache::on (params.aiEngine);
        if (ai != lastAiEngine)
        {
            lastAiEngine = ai;
            owner.updateNeuralModel();
        }
    }

    void prepare (double sampleRate, int blockSize)
    {
        maxBlock = juce::jmax (64, blockSize);
        neuralStream.prepare (sampleRate, maxBlock);
        chain.prepare (sampleRate, maxBlock, currentQuality());
        chain.setNeuralStream (neuralAttached.load() ? &neuralStream : nullptr);
        liveSinger.prepare (sampleRate, maxBlock);
        autoLevel.prepare (sampleRate);
        watermark.prepare (sampleRate);

        const double blockRate = sampleRate;
        for (auto* s : { &inGain, &outGain, &mixAmount, &powerFade })
            s->prepare (blockRate, 20.0);
        inGain.reset (juce::Decibels::decibelsToGain (ParamCache::v (params.inputGain)));
        outGain.reset (juce::Decibels::decibelsToGain (ParamCache::v (params.outputGain)));
        mixAmount.reset (ParamCache::v (params.mix) * 0.01f);
        powerFade.reset (ParamCache::on (params.power) ? 1.0f : 0.0f);

        work.setSize (4, maxBlock, false, true, false);
        dryDelay.assign ((size_t) (sampleRate * 4.0) + 8, 0.0f);
        dryWrite = 0;
        dryDelaySamples = chain.getLatencySamples();

        humBuffer.setSize (1, (int) (sampleRate * 30.0));
        humSamples = 0;

        {
            const juce::SpinLock::ScopedLockType lock (clipLock);
            if (sample != nullptr) sample->resampleTo (sampleRate);
            if (take != nullptr) take->clip.resampleTo (sampleRate);
        }
        prepared = true;
    }

    Quality currentQuality() const
    {
        return (Quality) juce::jlimit (0, 2, ParamCache::idx (params.quality));
    }

    //==========================================================================
    void handleMidi (const juce::MidiMessage& m, bool singMode, bool useMidiSource, double beat)
    {
        if (m.isNoteOn())
        {
            const int note = m.getNoteNumber();
            heldVelocity[(size_t) note] = juce::jmax (1, (int) m.getVelocity());
            if (heldCount < (int) heldStack.size())
                heldStack[(size_t) heldCount++] = note;

            if (singMode && useMidiSource)
                liveSinger.noteOn (note, m.getFloatVelocity());

            capture (beat, note, m.getFloatVelocity());
        }
        else if (m.isNoteOff())
        {
            const int note = m.getNoteNumber();
            heldVelocity[(size_t) note] = 0;
            for (int i = 0; i < heldCount; ++i)
            {
                if (heldStack[(size_t) i] == note)
                {
                    for (int j = i; j < heldCount - 1; ++j)
                        heldStack[(size_t) j] = heldStack[(size_t) j + 1];
                    --heldCount;
                    break;
                }
            }
            liveSinger.noteOff (note);
            capture (beat, note, 0.0f);
        }
        else if (m.isAllNotesOff() || m.isAllSoundOff())
        {
            heldVelocity.fill (0);
            heldCount = 0;
            liveSinger.allNotesOff();
        }
    }

    void capture (double beat, int note, float velocity)
    {
        if (! owner.midiCapture.load())
            return;
        const int n = capturedCount.load();
        if (n < (int) captured.size())
        {
            captured[(size_t) n] = { beat, note, velocity };
            capturedCount.store (n + 1);
        }
    }

    int currentMidiNote() const noexcept { return heldCount > 0 ? heldStack[(size_t) heldCount - 1] : -1; }

    //==========================================================================
    void stopRecordingInternal()
    {
        {
            const juce::ScopedLock sl (writerLock);
            activeWriter = nullptr;
        }
        threadedWriter.reset();
        writerThread.stopThread (2000);
    }
};

//==============================================================================
VoxaVersaProcessor::VoxaVersaProcessor()
    : AudioProcessor (isInstrumentBuild()
                        ? BusesProperties()
                              .withInput ("Sidechain", juce::AudioChannelSet::stereo(), false)
                              .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                        : BusesProperties()
                              .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                              .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, &undoManager, "VoxaVersa", createParameterLayout()),
      presetManager (apvts)
{
    apvts.state.setProperty (StateID::voiceId, voiceLibrary.getDefaultVoice().id, nullptr);
    apvts.state.setProperty (StateID::presetName, "Init Preset", nullptr);
    if (isInstrumentBuild())
        apvts.getParameter (ParamID::mode)->setValueNotifyingHost (1.0f);

    impl = std::make_unique<Impl> (*this);
    apvts.state.addListener (this);
    updateVoiceTraits();
    updateNeuralModel();
}

VoxaVersaProcessor::~VoxaVersaProcessor()
{
    cancelPendingUpdate();
    apvts.state.removeListener (this);
    impl.reset();
}

bool VoxaVersaProcessor::isInstrumentBuild() noexcept
{
   #if defined (JucePlugin_IsSynth) && JucePlugin_IsSynth
    return true;
   #else
    return false;
   #endif
}

const juce::String VoxaVersaProcessor::getName() const
{
    return JucePlugin_Name;
}

//==============================================================================
void VoxaVersaProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    inputScope.prepare (sampleRate);
    outputScope.prepare (sampleRate);
    impl->prepare (sampleRate, samplesPerBlock);
    setLatencySamples (impl->chain.getLatencySamples());
}

void VoxaVersaProcessor::releaseResources()
{
    impl->neuralStream.reset();
}

bool VoxaVersaProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::stereo() && out != juce::AudioChannelSet::mono())
        return false;

    const auto in = layouts.getMainInputChannelSet();
    if (isInstrumentBuild())
        return in.isDisabled() || in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo();

    return in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo();
}

//==============================================================================
void VoxaVersaProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    auto& d = *impl;
    const auto& p = d.params;

    const int numSamples = buffer.getNumSamples();
    const int numIn = getTotalNumInputChannels();
    const int numOut = getTotalNumOutputChannels();

    if (! d.prepared || numSamples == 0)
        return;

    // Exports / render in place: the AI voice waits for every chunk instead of skipping late ones.
    d.neuralStream.setNonRealtime (isNonRealtime());

    // Quality changes need a re-prepare (different latency): defer to the message thread.
    if (d.currentQuality() != d.chain.getQuality())
        d.qualityChangePending.store (true);

    //--------------------------------------------------------------------------
    // Host transport
    double bpm = 0.0, ppq = 0.0;
    juce::int64 hostTime = -1;
    bool playing = false;
    if (auto* ph = getPlayHead())
    {
        if (auto pos = ph->getPosition())
        {
            playing = pos->getIsPlaying();
            if (auto b = pos->getBpm()) bpm = *b;
            if (auto q = pos->getPpqPosition()) ppq = *q;
            if (auto t = pos->getTimeInSamples()) hostTime = *t;
        }
    }
    hostBpm.store (bpm);
    hostIsPlaying.store (playing);

    const bool singMode = ParamCache::idx (p.mode) == (int) Mode::TextToSing;
    const auto melodySrc = (MelodySource) ParamCache::idx (p.melodySource);
    const bool useMidiSource = melodySrc == MelodySource::UseMidi;
    const double singBpm = (ParamCache::on (p.singSyncHost) && bpm > 0.0) ? bpm : (double) ParamCache::v (p.singTempo);
    const double effectiveBpm = bpm > 0.0 ? bpm : singBpm;

    if (playing && ! d.wasPlaying)
        d.liveSinger.restartLyrics();
    d.wasPlaying = playing;

    // Voice traits (keep the previous copy if the message thread is updating).
    {
        const juce::SpinLock::ScopedTryLockType lock (d.traitsLock);
        if (lock.isLocked())
        {
            d.audioTraitsA = d.traitsA;
            d.audioTraitsB = d.traitsB;
            d.audioHasBlend = d.hasBlend;
        }
    }
    const VoiceTraits traits = d.audioHasBlend
        ? VoiceTraits::blend (d.audioTraitsA, d.audioTraitsB, ParamCache::v (p.voiceBlend) * 0.01f)
        : d.audioTraitsA;
    d.liveSinger.setVoice (traits);

    const bool powerOn = ParamCache::on (p.power);
    d.powerFade.setTarget (powerOn ? 1.0f : 0.0f);
    d.inGain.setTarget (juce::Decibels::decibelsToGain (ParamCache::v (p.inputGain)));
    d.outGain.setTarget (juce::Decibels::decibelsToGain (ParamCache::v (p.outputGain)));
    d.mixAmount.setTarget (ParamCache::v (p.mix) * 0.01f);
    const bool autoLevelOn = ParamCache::on (p.autoLevel);

    const double samplesPerBeat = currentSampleRate * 60.0 / juce::jmax (1.0, effectiveBpm);

    //--------------------------------------------------------------------------
    // Process in chunks split at MIDI events (sample-accurate live singing).
    juce::SpinLock::ScopedTryLockType clipGuard (d.clipLock);
    AudioClip* sampleClip = clipGuard.isLocked() ? d.sample.get() : nullptr;
    SingTake* take = clipGuard.isLocked() ? d.take.get() : nullptr;

    auto midiIt = midi.cbegin();
    int pos = 0;

    while (pos < numSamples)
    {
        int chunkEnd = juce::jmin (numSamples, pos + d.maxBlock);

        // Apply MIDI events due at `pos`, find the next event boundary.
        while (midiIt != midi.cend() && (*midiIt).samplePosition <= pos)
        {
            const auto meta = *midiIt;
            const double beat = playing ? ppq + (double) meta.samplePosition / samplesPerBeat
                                        : d.captureClockBeats + (double) meta.samplePosition / samplesPerBeat;
            d.handleMidi (meta.getMessage(), singMode, useMidiSource, beat);
            ++midiIt;
        }
        if (midiIt != midi.cend())
            chunkEnd = juce::jmin (chunkEnd, juce::jmax (pos + 1, (*midiIt).samplePosition));

        const int n = chunkEnd - pos;
        float* mono = d.work.getWritePointer (0);
        float* wetL = d.work.getWritePointer (1);
        float* wetR = d.work.getWritePointer (2);
        float* dry = d.work.getWritePointer (3);

        // 1. Live input (mono) with gain + auto level.
        if (numIn > 0)
        {
            juce::FloatVectorOperations::copy (mono, buffer.getReadPointer (0, pos), n);
            for (int ch = 1; ch < numIn; ++ch)
                juce::FloatVectorOperations::add (mono, buffer.getReadPointer (ch, pos), n);
            if (numIn > 1)
                juce::FloatVectorOperations::multiply (mono, 1.0f / (float) numIn, n);
        }
        else
        {
            juce::FloatVectorOperations::clear (mono, n);
        }

        for (int i = 0; i < n; ++i)
            mono[i] *= d.inGain.next();
        if (autoLevelOn)
            d.autoLevel.process (mono, n);

        // Hum capture uses the live input.
        if (humRecording.load())
        {
            const int have = d.humSamples.load();
            const int room = d.humBuffer.getNumSamples() - have;
            const int take2 = juce::jmin (room, n);
            if (take2 > 0)
            {
                d.humBuffer.copyFrom (0, have, mono, take2);
                d.humSamples.store (have + take2);
            }
            if (take2 < n)
                humRecording.store (false);
        }

        // 2. Source selection
        bool usingSample = false;
        if (singMode)
        {
            juce::FloatVectorOperations::clear (mono, n);

            if (take != nullptr && take->clip.playback.getNumSamples() > 0)
            {
                const auto& tb = take->clip.playback;
                const float* src = tb.getReadPointer (0);
                const int len = tb.getNumSamples();

                if (playing && hostTime >= 0)
                {
                    const juce::int64 start = hostTime + pos;
                    for (int i = 0; i < n; ++i)
                    {
                        const auto idx = start + i;
                        if (idx >= 0 && idx < len)
                            mono[i] = src[idx];
                    }
                }
                else if (singPreviewPlaying.load())
                {
                    auto idx = (juce::int64) d.previewPos.load();
                    for (int i = 0; i < n; ++i, ++idx)
                        if (idx < len)
                            mono[i] = src[idx];
                    d.previewPos.store ((double) idx);
                    if (idx >= len)
                    {
                        singPreviewPlaying.store (false);
                        d.previewPos.store (0.0);
                    }
                }
            }

            if (useMidiSource)
            {
                d.liveSinger.render (wetL, n);   // borrow wetL as scratch
                juce::FloatVectorOperations::add (mono, wetL, n);
            }
        }
        else if (samplePlaying.load() && sampleClip != nullptr && sampleClip->playback.getNumSamples() > 0)
        {
            const float* src = sampleClip->playback.getReadPointer (0);
            const int len = sampleClip->playback.getNumSamples();
            auto idx = (juce::int64) d.samplePos.load();
            usingSample = true;
            for (int i = 0; i < n; ++i, ++idx)
                mono[i] = idx < len ? src[idx] : 0.0f;
            d.samplePos.store ((double) idx);
            if (idx >= len)
            {
                samplePlaying.store (false);
                d.samplePos.store (0.0);
            }
        }

        juce::FloatVectorOperations::copy (dry, mono, n);

        // 3. Voice chain
        const bool takeIsNeural = take != nullptr && take->neuralRendered;
        auto settings = buildChainSettings (p, traits, effectiveBpm, d.currentMidiNote(), singMode);
        if (singMode && takeIsNeural && ! useMidiSource)
            settings.useNeural = false;
        if (usingSample && settings.pitch.sourceGender == 0)
            settings.pitch.sourceGender = sampleClip->gender;
        d.chain.process (mono, wetL, wetR, n, settings);

        // 4. Dry alignment, mix, output gain, power
        const int dlySize = (int) d.dryDelay.size();
        for (int i = 0; i < n; ++i)
        {
            d.dryDelay[(size_t) d.dryWrite] = dry[i];
            int r = d.dryWrite - d.dryDelaySamples;
            if (r < 0) r += dlySize;
            const float dryAligned = d.dryDelay[(size_t) r];
            d.dryWrite = (d.dryWrite + 1) % dlySize;

            const float m = d.mixAmount.next();
            const float g = d.outGain.next();
            const float pw = d.powerFade.next();
            const float l = (dryAligned * (1.0f - m) + wetL[i] * m) * g;
            const float r2 = (dryAligned * (1.0f - m) + wetR[i] * m) * g;
            wetL[i] = l * pw + dryAligned * (1.0f - pw);
            wetR[i] = r2 * pw + dryAligned * (1.0f - pw);
            dry[i] = dryAligned;
        }

        // Inaudible watermark whenever the output carries an AI or synthesized vocal (EU AI Act Art. 50).
        {
            auto mark = WatermarkKind::None;
            if (powerOn && ParamCache::v (p.mix) > 1.0f)
            {
                if (d.chain.isNeuralLeadActive() || (singMode && takeIsNeural))
                    mark = WatermarkKind::AiVoice;
                else if (singMode)
                    mark = WatermarkKind::SynthesizedSinging;
            }
            d.watermark.process (wetL, wetR, n, mark);
        }

        // 5. Visual feeds
        inputScope.push (mono, n);
        {
            const float* inPtrs[1] = { mono };
            inputMeter.update (inPtrs, 1, n);
        }
        outputScope.push (wetL, n);
        {
            const float* outPtrs[2] = { wetL, wetR };
            outputMeter.update (outPtrs, 2, n);
        }

        // 6. Output
        if (numOut >= 2)
        {
            buffer.copyFrom (0, pos, wetL, n);
            buffer.copyFrom (1, pos, wetR, n);
            for (int ch = 2; ch < numOut; ++ch)
                buffer.clear (ch, pos, n);
        }
        else if (numOut == 1)
        {
            float* out = buffer.getWritePointer (0, pos);
            for (int i = 0; i < n; ++i)
                out[i] = 0.5f * (wetL[i] + wetR[i]);
        }

        // 7. Recording
        if (recording.load())
        {
            const juce::ScopedTryLock tl (d.writerLock);
            if (tl.isLocked() && d.activeWriter != nullptr)
            {
                const float* recPtrs[2] = { wetL, wetR };
                if (d.activeWriter->write (recPtrs, n))
                    d.recordedSamples.fetch_add (n);
            }
        }

        pos = chunkEnd;
    }

    // Remaining MIDI events at the very end of the block.
    for (; midiIt != midi.cend(); ++midiIt)
        d.handleMidi ((*midiIt).getMessage(), singMode, useMidiSource, playing ? ppq : d.captureClockBeats);

    if (! playing)
        d.captureClockBeats += (double) numSamples / samplesPerBeat;

    // Telemetry
    detectedPitchHz.store (d.chain.getDetectedHz());
    targetPitchHz.store (d.chain.getTargetHz());
    detectedKeyRoot.store (d.chain.getMapper().getDetectedKeyRoot());
    detectedScale.store ((int) d.chain.getMapper().getDetectedScale());
    sourceGenderEstimate.store (d.chain.getMapper().getSourceGenderEstimate());
    autoLevelGainDb.store (d.autoLevel.getGainDb());

    // The plugin does not output MIDI.
    midi.clear();
}

//==============================================================================
juce::AudioProcessorEditor* VoxaVersaProcessor::createEditor()
{
    return new VoxaVersaEditor (*this);
}

int VoxaVersaProcessor::getNumPrograms()
{
    return juce::jmax (1, (int) presetManager.getPresets().size());
}

int VoxaVersaProcessor::getCurrentProgram()
{
    return presetManager.getCurrentIndex();
}

void VoxaVersaProcessor::setCurrentProgram (int index)
{
    if (juce::MessageManager::existsAndIsCurrentThread())
        presetManager.loadPreset (index);
}

const juce::String VoxaVersaProcessor::getProgramName (int index)
{
    const auto& presets = presetManager.getPresets();
    return juce::isPositiveAndBelow (index, (int) presets.size()) ? presets[(size_t) index].name : juce::String();
}

void VoxaVersaProcessor::getStateInformation (juce::MemoryBlock& dest)
{
    auto state = apvts.copyState();
    state.setProperty ("version", VOXAVERSA_VERSION_STRING, nullptr);
    if (auto xml = state.createXml())
        copyXmlToBinary (*xml, dest);
}

void VoxaVersaProcessor::setStateInformation (const void* data, int size)
{
    if (auto xml = getXmlFromBinary (data, size))
    {
        auto tree = juce::ValueTree::fromXml (*xml);
        if (tree.hasType (apvts.state.getType()))
        {
            apvts.replaceState (tree);
            impl->stateJustLoaded = true;
            triggerAsyncUpdate();
        }
    }
}

//==============================================================================
// Voices
juce::String VoxaVersaProcessor::getSelectedVoiceId() const
{
    return apvts.state.getProperty (StateID::voiceId).toString();
}

const VoiceProfile& VoxaVersaProcessor::getSelectedVoice() const
{
    if (auto* v = voiceLibrary.findById (getSelectedVoiceId()))
        return *v;
    return voiceLibrary.getDefaultVoice();
}

void VoxaVersaProcessor::selectVoice (const juce::String& voiceId)
{
    if (voiceLibrary.findById (voiceId) == nullptr)
        return;
    undoManager.beginNewTransaction ("Select voice");
    apvts.state.setProperty (StateID::voiceId, voiceId, &undoManager);
}

juce::String VoxaVersaProcessor::getBlendVoiceId() const
{
    return apvts.state.getProperty (StateID::blendVoiceId).toString();
}

void VoxaVersaProcessor::setBlendVoice (const juce::String& voiceIdOrEmpty)
{
    undoManager.beginNewTransaction ("Blend voice");
    apvts.state.setProperty (StateID::blendVoiceId, voiceIdOrEmpty, &undoManager);
    if (voiceIdOrEmpty.isNotEmpty() && ParamCache::v (impl->params.voiceBlend) < 1.0f)
        if (auto* p = apvts.getParameter (ParamID::voiceBlend))
            p->setValueNotifyingHost (p->convertTo0to1 (50.0f));
}

void VoxaVersaProcessor::updateVoiceTraits()
{
    const auto& a = getSelectedVoice();
    const auto* b = voiceLibrary.findById (getBlendVoiceId());

    {
        const juce::SpinLock::ScopedLockType lock (impl->traitsLock);
        impl->traitsA = a.traits;
        impl->hasBlend = b != nullptr && b->id != a.id;
        if (b != nullptr)
            impl->traitsB = b->traits;
    }
}

juce::String VoxaVersaProcessor::getNeuralStatusMessage() const
{
    switch (neuralStatus.load())
    {
        case NeuralStatus::Active:          return impl->neuralMessage.isNotEmpty() ? impl->neuralMessage : "AI voice model active";
        case NeuralStatus::Loading:         return "Loading AI voice model...";
        case NeuralStatus::NoModelForVoice: return "DSP voice engine (this voice has no AI model)";
        case NeuralStatus::Error:           return "AI model error: " + impl->neuralMessage;
        case NeuralStatus::Unavailable:
        default:
            return "DSP voice engine - " + (impl->neuralMessage.isNotEmpty() ? impl->neuralMessage
                                                                              : neural::OnnxRuntime::getInstance().getStatusMessage());
    }
}

uint32_t VoxaVersaProcessor::getNeuralMissedChunks() const noexcept
{
    return impl->neuralStream.getNumMissedChunks();
}

bool VoxaVersaProcessor::isNeuralCpuOnly() const
{
    auto& runtime = neural::OnnxRuntime::getInstance();
    return ! runtime.ensureLoaded() || runtime.getImpl().getAccelerator() == neural::Accelerator::cpu;
}

void VoxaVersaProcessor::setNeuralCpuOnly (bool cpuOnly)
{
    auto& runtime = neural::OnnxRuntime::getInstance();
    if (! runtime.ensureLoaded())
        return;

    runtime.getImpl().setAndSaveAccelerator (cpuOnly ? neural::Accelerator::cpu : neural::Accelerator::automatic);
    impl->currentModelVoiceId.clear();   // reload the model on the chosen device
    updateNeuralModel();
}

void VoxaVersaProcessor::updateNeuralModel()
{
    const auto& voice = getSelectedVoice();
    const bool aiOn = ParamCache::on (impl->params.aiEngine);
    const bool voiceHasModel = voice.hasNeuralModel();

    if (! aiOn || ! voiceHasModel)
    {
        ++impl->modelGeneration;   // invalidates any load in flight
        if (impl->currentModel != nullptr || impl->neuralAttached.load())
        {
            impl->currentModel.reset();
            impl->currentModelVoiceId.clear();
            impl->neuralStream.setModel (nullptr);
            impl->neuralAttached.store (false);
            updateLatency();
        }
        impl->neuralMessage = aiOn ? juce::String() : "AI engine switched off";
        neuralStatus.store (voiceHasModel ? NeuralStatus::Unavailable : NeuralStatus::NoModelForVoice);
        return;
    }

    if (impl->currentModelVoiceId == voice.id && impl->currentModel != nullptr)
        return;

    if (! neural::OnnxRuntime::getInstance().ensureLoaded())
    {
        impl->neuralMessage.clear();
        neuralStatus.store (NeuralStatus::Unavailable);
        return;
    }

    neuralStatus.store (NeuralStatus::Loading);
    const int generation = ++impl->modelGeneration;
    const VoiceProfile profile = voice;
    auto alive = impl->alive;
    auto* self = this;

    impl->pool.addJob ([self, alive, profile, generation]
    {
        juce::String error;
        auto model = neural::NeuralVoiceModel::load (profile, error);

        juce::MessageManager::callAsync ([self, alive, model, error, generation, id = profile.id]
        {
            if (! *alive || generation != self->impl->modelGeneration)
                return;

            if (model == nullptr)
            {
                self->impl->neuralMessage = error;
                self->neuralStatus.store (NeuralStatus::Error);
                return;
            }

            self->impl->currentModel = model;
            self->impl->currentModelVoiceId = id;
            self->impl->neuralMessage = model->getDescription();
            self->impl->neuralStream.setModel (model);
            self->impl->neuralAttached.store (true);
            self->neuralStatus.store (NeuralStatus::Active);
            self->updateLatency();
        });
    });
}

void VoxaVersaProcessor::updateLatency()
{
    if (! impl->prepared)
        return;

    suspendProcessing (true);
    if (impl->currentQuality() != impl->chain.getQuality())
        impl->chain.prepare (currentSampleRate, impl->maxBlock, impl->currentQuality());
    impl->chain.setNeuralStream (impl->neuralAttached.load() ? &impl->neuralStream : nullptr);
    impl->dryDelaySamples = juce::jmin ((int) impl->dryDelay.size() - 1, impl->chain.getLatencySamples());
    std::fill (impl->dryDelay.begin(), impl->dryDelay.end(), 0.0f);
    suspendProcessing (false);

    setLatencySamples (impl->chain.getLatencySamples());
    updateHostDisplay (ChangeDetails().withLatencyChanged (true));
}

//==============================================================================
void VoxaVersaProcessor::valueTreePropertyChanged (juce::ValueTree&, const juce::Identifier& property)
{
    if (property.toString() == StateID::voiceId || property.toString() == StateID::blendVoiceId
        || property.toString() == StateID::lyrics)
        triggerAsyncUpdate();
}

void VoxaVersaProcessor::valueTreeRedirected (juce::ValueTree&)
{
    impl->stateJustLoaded = true;
    triggerAsyncUpdate();
}

void VoxaVersaProcessor::handleAsyncUpdate()
{
    updateVoiceTraits();
    updateNeuralModel();

    const auto lyrics = getLyrics();
    if (lyrics != impl->lastLyrics)
    {
        impl->lastLyrics = lyrics;
        impl->liveSinger.setLyrics (lyrics);
    }

    if (impl->stateJustLoaded)
    {
        impl->stateJustLoaded = false;
        presetManager.syncWithState();

        const juce::File samplePath (apvts.state.getProperty (StateID::samplePath).toString());
        if (samplePath.existsAsFile() && (impl->sample == nullptr || impl->sample->file != samplePath))
        {
            juce::String error;
            loadSample (samplePath, error);
        }
    }

    if (impl->currentQuality() != impl->chain.getQuality())
        updateLatency();
}

//==============================================================================
// Samples
bool VoxaVersaProcessor::loadSample (const juce::File& file, juce::String& error)
{
    std::unique_ptr<juce::AudioFormatReader> reader (impl->formatManager.createReaderFor (file));
    if (reader == nullptr)
    {
        error = "Unsupported or unreadable audio file: " + file.getFileName();
        return false;
    }

    const double maxSeconds = 600.0;
    if ((double) reader->lengthInSamples / reader->sampleRate > maxSeconds)
    {
        error = "Samples longer than 10 minutes are not supported";
        return false;
    }

    juce::AudioBuffer<float> raw ((int) reader->numChannels, (int) reader->lengthInSamples);
    reader->read (&raw, 0, (int) reader->lengthInSamples, 0, true, true);

    auto clip = std::make_unique<AudioClip>();
    clip->original = toMono (raw);
    clip->originalRate = reader->sampleRate;
    clip->name = file.getFileNameWithoutExtension();
    clip->file = file;
    clip->gender = (int) dsp::VoiceGenderEstimator::analyse (clip->original.getReadPointer (0), clip->original.getNumSamples(),
                                                             clip->originalRate);
    clip->resampleTo (currentSampleRate);

    samplePlaying.store (false);
    {
        const juce::SpinLock::ScopedLockType lock (impl->clipLock);
        std::swap (impl->sample, clip);
        impl->samplePos.store (0.0);
    }
    apvts.state.setProperty (StateID::samplePath, file.getFullPathName(), nullptr);
    return true;
}

void VoxaVersaProcessor::clearSample()
{
    samplePlaying.store (false);
    std::unique_ptr<AudioClip> old;
    {
        const juce::SpinLock::ScopedLockType lock (impl->clipLock);
        std::swap (impl->sample, old);
    }
    apvts.state.setProperty (StateID::samplePath, "", nullptr);
}

bool VoxaVersaProcessor::hasSample() const noexcept
{
    return impl->sample != nullptr;
}

juce::String VoxaVersaProcessor::getSampleName() const
{
    return impl->sample != nullptr ? impl->sample->name : juce::String();
}

double VoxaVersaProcessor::getSampleLengthSeconds() const noexcept
{
    return impl->sample != nullptr ? impl->sample->original.getNumSamples() / impl->sample->originalRate : 0.0;
}

void VoxaVersaProcessor::setSamplePlaying (bool shouldPlay)
{
    if (shouldPlay && ! hasSample())
        return;
    if (shouldPlay)
        impl->samplePos.store (0.0);
    samplePlaying.store (shouldPlay);
}

double VoxaVersaProcessor::getSamplePlayPosition() const noexcept
{
    if (impl->sample == nullptr || impl->sample->playback.getNumSamples() == 0)
        return 0.0;
    return impl->samplePos.load() / impl->sample->playback.getNumSamples();
}

//==============================================================================
// Offline rendering
juce::File VoxaVersaProcessor::getExportDirectory()
{
    return juce::File::getSpecialLocation (juce::File::userDocumentsDirectory).getChildFile ("VoxaVersa").getChildFile ("Exports");
}

void VoxaVersaProcessor::cancelOfflineRender()
{
    impl->cancelRender.store (true);
}

juce::File VoxaVersaProcessor::getLastRenderedFile() const
{
    const juce::ScopedLock sl (impl->resultLock);
    return impl->lastRenderedFile;
}

juce::String VoxaVersaProcessor::getLastRenderError() const
{
    const juce::ScopedLock sl (impl->resultLock);
    return impl->lastRenderError;
}

void VoxaVersaProcessor::startOfflineRender (RenderTarget target)
{
    if (offlineRendering.load())
        return;

    // Snapshot everything on the message thread.
    juce::AudioBuffer<float> source;
    juce::String name;
    bool sourceIsSynth = false, sourceIsNeural = false;
    int sourceGender = 0;
    {
        const juce::SpinLock::ScopedLockType lock (impl->clipLock);
        if (target == RenderTarget::Sample && impl->sample != nullptr)
        {
            source = impl->sample->playback;
            name = impl->sample->name;
            sourceGender = impl->sample->gender;
        }
        else if (target == RenderTarget::SingTake && impl->take != nullptr)
        {
            source = impl->take->clip.playback;
            name = "Sung";
            sourceIsSynth = true;
            sourceIsNeural = impl->take->neuralRendered;
        }
    }

    if (source.getNumSamples() == 0)
    {
        const juce::ScopedLock sl (impl->resultLock);
        impl->lastRenderError = target == RenderTarget::Sample ? "Load a voice sample first" : "Generate a vocal first";
        return;
    }

    VoiceTraits traits;
    {
        const juce::SpinLock::ScopedLockType lock (impl->traitsLock);
        traits = impl->hasBlend ? VoiceTraits::blend (impl->traitsA, impl->traitsB, ParamCache::v (impl->params.voiceBlend) * 0.01f)
                                : impl->traitsA;
    }

    const double bpm = hostBpm.load() > 0.0 ? hostBpm.load() : (double) ParamCache::v (impl->params.singTempo);
    auto settings = buildChainSettings (impl->params, traits, bpm, -1, sourceIsSynth);
    if (settings.pitch.sourceGender == 0)
        settings.pitch.sourceGender = sourceGender;   // whole-file analysis beats the live estimate
    const auto model = (! sourceIsNeural && settings.useNeural) ? impl->currentModel : nullptr;
    settings.useNeural = false;
    settings.pitch.naturalVibrato = model != nullptr;   // the AI lead keeps the singer's own vibrato

    const float mix = ParamCache::v (impl->params.mix) * 0.01f;
    const float outGainLin = juce::Decibels::decibelsToGain (ParamCache::v (impl->params.outputGain));
    const double sr = currentSampleRate;
    const auto quality = Quality::HighQuality;   // latency is irrelevant offline
    const auto voiceName = getSelectedVoice().name;
    const auto origin = sourceIsSynth ? ((sourceIsNeural || model != nullptr) ? VocalOrigin::SynthesizedSingingAiVoice
                                                                               : VocalOrigin::SynthesizedSinging)
                                      : (model != nullptr ? VocalOrigin::AiVoiceModel : VocalOrigin::Transformed);
    const auto metadata = disclosureMetadata (origin, voiceName);
    const auto mark = origin == VocalOrigin::Transformed ? WatermarkKind::None
                    : (origin == VocalOrigin::SynthesizedSinging ? WatermarkKind::SynthesizedSinging : WatermarkKind::AiVoice);
    const auto outFile = getExportDirectory().getNonexistentChildFile (
        sanitiseForFileName ("VoxaVersa " + voiceName + " " + name), ".wav");

    impl->cancelRender.store (false);
    offlineProgress.store (0.0f);
    offlineRendering.store (true);

    auto* self = this;
    impl->pool.addJob ([self, source, settings, model, mix, outGainLin, sr, quality, outFile, metadata, mark]
    {
        auto& cancel = self->impl->cancelRender;
        juce::String error;

        dsp::VoiceChain chain;
        chain.prepare (sr, 1024, quality);
        const int latency = chain.getLatencySamples();
        const int tail = (int) (sr * 3.0);
        const int total = source.getNumSamples() + latency + tail;

        juce::AudioBuffer<float> padded (1, total);
        padded.clear();
        padded.copyFrom (0, 0, source, 0, 0, source.getNumSamples());

        // Optional neural lead: pass 1 captures the target pitch contour, then the model converts offline.
        juce::AudioBuffer<float> neuralLead;
        if (model != nullptr)
        {
            std::vector<float> f0;
            f0.reserve ((size_t) (total / sr * 100.0) + 16);
            chain.setF0Capture (&f0, 100.0);
            juce::AudioBuffer<float> scratch (2, 1024);
            for (int pos = 0; pos < total && ! cancel.load(); pos += 1024)
            {
                const int n = juce::jmin (1024, total - pos);
                chain.process (padded.getReadPointer (0, pos), scratch.getWritePointer (0), scratch.getWritePointer (1), n, settings);
                self->offlineProgress.store (0.3f * (float) pos / (float) total);
            }
            chain.setF0Capture (nullptr, 100.0);

            juce::AudioBuffer<float> converted;
            if (! cancel.load() && model->convert (source, sr, f0, 100.0, converted, &cancel, error))
            {
                // Align to the chain's output timeline (DSP latency).
                neuralLead.setSize (1, total);
                neuralLead.clear();
                const int copyLen = juce::jmin (converted.getNumSamples(), total - latency);
                if (copyLen > 0)
                    neuralLead.copyFrom (0, latency, converted, 0, 0, copyLen);
            }
            chain.reset();
        }

        juce::AudioBuffer<float> wet (2, total);
        for (int pos = 0; pos < total && ! cancel.load(); pos += 1024)
        {
            const int n = juce::jmin (1024, total - pos);
            chain.process (padded.getReadPointer (0, pos), wet.getWritePointer (0, pos), wet.getWritePointer (1, pos), n, settings,
                           neuralLead.getNumSamples() > 0 ? neuralLead.getReadPointer (0, pos) : nullptr);
            self->offlineProgress.store ((model != nullptr ? 0.5f : 0.0f) + (model != nullptr ? 0.45f : 0.95f) * (float) pos / (float) total);
        }

        bool ok = false;
        if (! cancel.load())
        {
            const int outLen = source.getNumSamples() + tail;
            juce::AudioBuffer<float> out (2, outLen);
            for (int ch = 0; ch < 2; ++ch)
            {
                float* dst = out.getWritePointer (ch);
                const float* w = wet.getReadPointer (ch, latency);
                for (int i = 0; i < outLen; ++i)
                {
                    const float dryS = i < source.getNumSamples() ? source.getSample (0, i) : 0.0f;
                    dst[i] = (dryS * (1.0f - mix) + w[i] * mix) * outGainLin;
                }
            }

            // Trim silent tail.
            int last = outLen - 1;
            while (last > source.getNumSamples() && out.getMagnitude (last, 1) < 1.0e-4f)
                --last;
            out.setSize (2, last + 1, true);

            if (mark != WatermarkKind::None)
            {
                WatermarkEmbedder embedder;
                embedder.prepare (sr);
                embedder.process (out.getWritePointer (0), out.getWritePointer (1), out.getNumSamples(), mark);
            }

            const float peak = out.getMagnitude (0, out.getNumSamples());
            if (peak > 0.99f)
                out.applyGain (0.99f / peak);

            ok = writeWav (outFile, out, sr, metadata);
            if (! ok)
                error = "Could not write " + outFile.getFullPathName();
        }
        else
        {
            error = "Render cancelled";
        }

        {
            const juce::ScopedLock sl (self->impl->resultLock);
            self->impl->lastRenderError = ok ? juce::String() : error;
            if (ok)
                self->impl->lastRenderedFile = outFile;
        }
        self->offlineProgress.store (1.0f);
        self->offlineRendering.store (false);
    });
}

//==============================================================================
// Text to Sing
void VoxaVersaProcessor::setLyrics (const juce::String& lyrics)
{
    apvts.state.setProperty (StateID::lyrics, lyrics.substring (0, maxLyricsLength), nullptr);
}

juce::String VoxaVersaProcessor::getLyrics() const
{
    return apvts.state.getProperty (StateID::lyrics).toString();
}

bool VoxaVersaProcessor::loadMelodyMidiFile (const juce::File& file, juce::String& error)
{
    double bpm = 120.0;
    auto notes = sing::SingEngine::melodyFromMidiFile (file, bpm, error);
    if (notes.empty())
    {
        if (error.isEmpty())
            error = "No notes found in " + file.getFileName();
        return false;
    }

    apvts.state.setProperty (StateID::melodyFile, file.getFullPathName(), nullptr);
    if (auto* p = apvts.getParameter (ParamID::melodySource))
        p->setValueNotifyingHost (p->convertTo0to1 ((float) MelodySource::MidiFile));
    if (auto* t = apvts.getParameter (ParamID::singTempo); t != nullptr && bpm > 0.0)
        t->setValueNotifyingHost (t->convertTo0to1 ((float) bpm));
    return true;
}

juce::String VoxaVersaProcessor::getMelodyFileName() const
{
    return juce::File (apvts.state.getProperty (StateID::melodyFile).toString()).getFileName();
}

void VoxaVersaProcessor::setMidiCaptureEnabled (bool shouldCapture)
{
    if (shouldCapture && ! midiCapture.load())
        impl->capturedCount.store (0);
    midiCapture.store (shouldCapture);
}

int VoxaVersaProcessor::getNumCapturedNotes() const
{
    const int n = impl->capturedCount.load();
    int notes = 0;
    for (int i = 0; i < n; ++i)
        if (impl->captured[(size_t) i].velocity > 0.0f)
            ++notes;
    return notes;
}

void VoxaVersaProcessor::clearCapturedMidi()
{
    impl->capturedCount.store (0);
}

void VoxaVersaProcessor::setHumRecording (bool shouldRecord)
{
    if (shouldRecord)
    {
        impl->humSamples.store (0);
        if (auto* p = apvts.getParameter (ParamID::melodySource))
            p->setValueNotifyingHost (p->convertTo0to1 ((float) MelodySource::HumMelody));
    }
    humRecording.store (shouldRecord);
}

double VoxaVersaProcessor::getHumRecordedSeconds() const noexcept
{
    return impl->humSamples.load() / juce::jmax (1.0, currentSampleRate);
}

juce::String VoxaVersaProcessor::getSingError() const
{
    const juce::ScopedLock sl (impl->resultLock);
    return impl->singError;
}

std::vector<sing::SungNote> VoxaVersaProcessor::getSungNotes() const
{
    const juce::SpinLock::ScopedLockType lock (impl->clipLock);
    return impl->take != nullptr ? impl->take->notes : std::vector<sing::SungNote>();
}

double VoxaVersaProcessor::getSingTakeLengthSeconds() const
{
    const juce::SpinLock::ScopedLockType lock (impl->clipLock);
    return impl->take != nullptr ? impl->take->clip.playback.getNumSamples() / juce::jmax (1.0, currentSampleRate) : 0.0;
}

void VoxaVersaProcessor::cancelSinging()
{
    impl->cancelSing.store (true);
}

void VoxaVersaProcessor::generateSinging()
{
    if (singState.load() == SingState::Rendering)
        return;

    const auto lyrics = getLyrics().trim();
    if (lyrics.isEmpty())
    {
        const juce::ScopedLock sl (impl->resultLock);
        impl->singError = "Type some lyrics first";
        singState.store (SingState::Failed);
        return;
    }

    const auto& p = impl->params;
    sing::SingRequest request;
    request.lyrics = lyrics;
    request.keyRoot = ParamCache::idx (p.key) - 1;
    request.scale = (ScaleType) juce::jlimit (0, (int) ScaleType::NumScales - 1, ParamCache::idx (p.scale));
    request.bpm = (ParamCache::on (p.singSyncHost) && hostBpm.load() > 0.0) ? hostBpm.load() : (double) ParamCache::v (p.singTempo);
    request.voice = getSelectedVoice();
    request.seed = impl->singSeed++;
    request.sampleRate = currentSampleRate;

    const auto source = (MelodySource) ParamCache::idx (p.melodySource);
    juce::String melodyError;

    switch (source)
    {
        case MelodySource::AutoMelody:
            request.melodyMode = sing::MelodyMode::AutoGenerate;
            break;

        case MelodySource::MidiFile:
        {
            const juce::File file (apvts.state.getProperty (StateID::melodyFile).toString());
            double fileBpm = request.bpm;
            request.melody = sing::SingEngine::melodyFromMidiFile (file, fileBpm, melodyError);
            request.melodyMode = sing::MelodyMode::Provided;
            if (request.melody.empty() && melodyError.isEmpty())
                melodyError = "Load a MIDI file with a melody first";
            break;
        }

        case MelodySource::UseMidi:
        {
            juce::MidiMessageSequence seq;
            const int n = impl->capturedCount.load();
            double first = -1.0;
            for (int i = 0; i < n; ++i)
            {
                const auto& e = impl->captured[(size_t) i];
                if (first < 0.0 && e.velocity > 0.0f)
                    first = e.beat;
                if (first < 0.0)
                    continue;
                auto m = e.velocity > 0.0f ? juce::MidiMessage::noteOn (1, e.note, e.velocity)
                                           : juce::MidiMessage::noteOff (1, e.note);
                m.setTimeStamp (e.beat - first);
                seq.addEvent (m);
            }
            seq.updateMatchedPairs();
            request.melody = sing::SingEngine::melodyFromSequence (seq);
            request.melodyMode = sing::MelodyMode::Provided;
            if (request.melody.empty())
                melodyError = "No MIDI captured yet: enable capture and play your melody into VoxaVersa "
                              "(or sing it live with MIDI, or choose Auto Melody)";
            break;
        }

        case MelodySource::HumMelody:
        {
            const int n = impl->humSamples.load();
            if (n < (int) (currentSampleRate * 0.5))
            {
                melodyError = "Record your hummed melody first (at least half a second)";
                break;
            }
            juce::AudioBuffer<float> hum (1, n);
            hum.copyFrom (0, 0, impl->humBuffer, 0, 0, n);
            request.melody = sing::SingEngine::melodyFromAudio (hum, currentSampleRate, request.bpm);
            request.melodyMode = sing::MelodyMode::Provided;
            if (request.melody.empty())
                melodyError = "Could not detect a melody in the recording";
            break;
        }
    }

    if (melodyError.isNotEmpty())
    {
        const juce::ScopedLock sl (impl->resultLock);
        impl->singError = melodyError;
        singState.store (SingState::Failed);
        return;
    }

    const auto model = ParamCache::on (p.aiEngine) ? impl->currentModel : nullptr;

    impl->cancelSing.store (false);
    singProgress.store (0.0f);
    singState.store (SingState::Rendering);
    singPreviewPlaying.store (false);

    auto* self = this;
    impl->pool.addJob ([self, request, model]
    {
        auto& cancel = self->impl->cancelSing;
        sing::SingEngine engine;
        auto result = engine.render (request, &cancel, [self, hasModel = model != nullptr] (float progress)
        {
            self->singProgress.store (progress * (hasModel ? 0.6f : 0.98f));
        });

        if (cancel.load() || ! result.ok())
        {
            const juce::ScopedLock sl (self->impl->resultLock);
            self->impl->singError = cancel.load() ? "Cancelled" : result.errorMessage;
            self->singState.store (SingState::Failed);
            return;
        }

        auto newTake = std::make_unique<SingTake>();
        newTake->notes = result.notes;
        newTake->bpm = request.bpm;
        newTake->clip.original = result.audio;
        newTake->clip.originalRate = result.sampleRate;
        newTake->clip.name = "Sung vocal";

        if (model != nullptr)
        {
            juce::AudioBuffer<float> converted;
            juce::String error;
            if (model->convert (result.audio, result.sampleRate, result.f0Hz, result.f0FrameRate, converted, &cancel, error))
            {
                newTake->clip.original = converted;
                newTake->neuralRendered = true;
            }
        }

        newTake->clip.resampleTo (self->currentSampleRate);
        newTake->result = std::move (result);

        {
            const juce::SpinLock::ScopedLockType lock (self->impl->clipLock);
            std::swap (self->impl->take, newTake);
            self->impl->previewPos.store (0.0);
        }
        {
            const juce::ScopedLock sl (self->impl->resultLock);
            self->impl->singError.clear();
        }
        self->singProgress.store (1.0f);
        self->singState.store (SingState::Ready);
    });
}

void VoxaVersaProcessor::setSingPreviewPlaying (bool shouldPlay)
{
    if (shouldPlay)
        impl->previewPos.store (0.0);
    singPreviewPlaying.store (shouldPlay && getSingTakeLengthSeconds() > 0.0);
}

double VoxaVersaProcessor::getSingPlayPosition() const noexcept
{
    const double len = getSingTakeLengthSeconds() * currentSampleRate;
    return len > 0.0 ? impl->previewPos.load() / len : 0.0;
}

juce::File VoxaVersaProcessor::exportSingMidi()
{
    juce::MidiFile midiFile;
    double bpm = 120.0;
    {
        const juce::SpinLock::ScopedLockType lock (impl->clipLock);
        if (impl->take == nullptr)
            return {};
        bpm = impl->take->bpm;
        midiFile = sing::SingEngine::toMidiFile (impl->take->result, bpm);
    }

    auto file = getExportDirectory().getNonexistentChildFile (sanitiseForFileName ("VoxaVersa " + getSelectedVoice().name + " melody"), ".mid");
    file.getParentDirectory().createDirectory();
    juce::FileOutputStream out (file);
    if (! out.openedOk() || ! midiFile.writeTo (out))
        return {};
    return file;
}

int VoxaVersaProcessor::getLiveSyllableIndex() const noexcept
{
    return impl->liveSinger.getCurrentSyllableIndex();
}

//==============================================================================
// Recording
void VoxaVersaProcessor::startRecording()
{
    if (recording.load())
        return;

    auto file = juce::File::getSpecialLocation (juce::File::userDocumentsDirectory)
                    .getChildFile ("VoxaVersa").getChildFile ("Recordings")
                    .getNonexistentChildFile ("VoxaVersa Recording " + juce::Time::getCurrentTime().formatted ("%Y-%m-%d %H-%M-%S"), ".wav");

    auto writer = createWavWriter (file, currentSampleRate, 2, disclosureMetadata (VocalOrigin::Recording, getSelectedVoice().name));
    if (writer == nullptr)
        return;

    impl->writerThread.startThread();
    impl->threadedWriter = std::make_unique<juce::AudioFormatWriter::ThreadedWriter> (writer.release(), impl->writerThread, 32768);
    impl->recordingFile = file;
    impl->recordedSamples.store (0);
    {
        const juce::ScopedLock sl (impl->writerLock);
        impl->activeWriter = impl->threadedWriter.get();
    }
    recording.store (true);
}

void VoxaVersaProcessor::stopRecording()
{
    if (! recording.exchange (false))
        return;
    impl->stopRecordingInternal();
    const juce::ScopedLock sl (impl->resultLock);
    impl->lastRecording = impl->recordingFile;
}

double VoxaVersaProcessor::getRecordingSeconds() const noexcept
{
    return impl->recordedSamples.load() / juce::jmax (1.0, currentSampleRate);
}

juce::File VoxaVersaProcessor::getLastRecording() const
{
    const juce::ScopedLock sl (impl->resultLock);
    return impl->lastRecording;
}

} // namespace voxa

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new voxa::VoxaVersaProcessor();
}
