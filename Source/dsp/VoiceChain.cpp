#include "VoiceChain.h"
#include "../neural/NeuralVoice.h"

namespace voxa::dsp
{

namespace
{
    inline float semitonesToRatio (float st) noexcept { return std::exp2 (st / 12.0f); }

    inline float lfo (double pos, double sampleRate, double rateHz, double phase) noexcept
    {
        return (float) std::sin (juce::MathConstants<double>::twoPi * rateHz * pos / sampleRate + phase);
    }

    inline void panGains (float pan, float& l, float& r) noexcept
    {
        pan = juce::jlimit (-1.0f, 1.0f, pan);
        l = juce::jmin (1.0f, 1.0f - pan);
        r = juce::jmin (1.0f, 1.0f + pan);
    }

    constexpr float harmonyDelayMs[3]  { 7.0f, 12.0f, 17.0f };
    constexpr float harmonyDetune[3]   { 3.0f, -4.0f, 2.0f };
    constexpr float harmonyFormant[3]  { 0.10f, -0.12f, 0.18f };
}

VoiceChain::VoiceChain() = default;
VoiceChain::~VoiceChain() = default;

void VoiceChain::prepare (double sampleRate, int maxBlockSize, Quality q)
{
    sr = sampleRate;
    maxBlock = juce::jmax (16, maxBlockSize);
    quality = q;

    const float minHz = q == Quality::LowLatency ? 80.0f : (q == Quality::Balanced ? 60.0f : 52.0f);
    constexpr float maxHz = 1300.0f;

    detector.prepare (sr, minHz, maxHz);
    genderEstimator.prepare (sr);
    mapper.prepare (sr, detector.getHopSize());
    analyzer.prepare (sr, minHz, maxHz);

    const float tMax = analyzer.getMaxPeriod();
    maxHalfGrain = 1.25f * tMax + 2.0f;

    const bool lowLatency = q == Quality::LowLatency;
    timeOffset = lowLatency ? 0 : detector.getAnalysisDelay();
    placementLag = lowLatency ? 0 : detector.getAnalysisDelay() + 2 * detector.getHopSize();
    dspLatency = (int) std::ceil (juce::jmax (2.75f * tMax, (float) placementLag + 2.0f * tMax)) + 16;

    lead.prepare (dspLatency);
    for (auto& h : harmony) h.prepare (dspLatency);
    for (auto& d : doubler) d.prepare (dspLatency);

    character.prepare (sr);
    deEsser.prepare (sr);
    compressor.prepare (sr);
    saturator.prepare (sr);
    delay.prepare (sr);
    reverb.prepare (sr);

    for (auto* b : { &leadBuf, &voiceBuf, &f0Buf, &neuralIn, &neuralOut, &neuralCoverage, &busL, &busR })
        b->assign ((size_t) maxBlock, 0.0f);

    const int maxAlign = (int) (sr * 3.0);
    alignL.prepare (maxAlign);
    alignR.prepare (maxAlign);
    alignNeural.prepare (maxAlign);
    alignLead.prepare (maxAlign);
    alignCoverage.prepare (maxAlign);

    setNeuralStream (neural);
    reset();
}

void VoiceChain::setNeuralStream (neural::NeuralVoiceStream* stream)
{
    neural = stream;

    if (neural != nullptr && neural->hasModel())
        neuralLatency = detector.getAnalysisDelay() + 2 * detector.getHopSize() + neural->getLatencySamples();
    else
        neuralLatency = 0;

    totalLatency = juce::jmax (dspLatency, neuralLatency);
    alignL.delay = alignR.delay = alignLead.delay = totalLatency - dspLatency;
    alignNeural.delay = alignCoverage.delay = neuralLatency > 0 ? totalLatency - neuralLatency : 0;
    alignL.clear();
    alignR.clear();
    alignNeural.clear();
    alignLead.clear();
    alignCoverage.clear();
}

void VoiceChain::reset()
{
    detector.reset();
    genderEstimator.reset();
    mapper.reset();
    timeline.reset();
    analyzer.reset();
    lead.reset (0.0);
    for (auto& h : harmony) h.reset (0.0);
    for (auto& d : doubler) d.reset (0.0);
    for (auto& a : harmonyActive) a = false;
    for (auto& a : doublerActive) a = false;
    character.reset();
    deEsser.reset();
    compressor.reset();
    saturator.reset();
    delay.reset();
    reverb.reset();
    alignL.clear();
    alignR.clear();
    alignNeural.clear();
    alignLead.clear();
    alignCoverage.clear();
    blockStart = 0;
    formantSmoothed = 0.0f;
    detectedHz = targetHz = 0.0f;
}

void VoiceChain::setF0Capture (std::vector<float>* destination, double frameRate)
{
    f0Capture = destination;
    f0CaptureRate = frameRate;
}

float VoiceChain::computeFormantSemis (const ChainSettings& s) const noexcept
{
    float semis = 0.0f;
    if (s.formantShift && ! s.pitch.sourceIsSynth)
        semis += (s.pitch.traits.formantSemis - mapper.getSourceFormantBaseline (s.pitch.sourceGender)) * s.intensity;

    semis += s.gender * 3.0f;
    semis += s.age < 0.0f ? -s.age * 1.0f : -s.age * 0.8f;
    return juce::jlimit (-7.0f, 7.0f, semis);
}

void VoiceChain::process (const float* in, float* outL, float* outR, int numSamples,
                          const ChainSettings& s, const float* leadOverride) noexcept
{
    // Split oversized host blocks.
    if (numSamples > maxBlock)
    {
        for (int offset = 0; offset < numSamples; offset += maxBlock)
        {
            const int n = juce::jmin (maxBlock, numSamples - offset);
            process (in + offset, outL + offset, outR + offset, n, s,
                     leadOverride != nullptr ? leadOverride + offset : nullptr);
        }
        return;
    }

    const int n = numSamples;
    const auto& traits = s.pitch.traits;
    const float styleScale = 1.0f + s.style;

    const bool neuralActive = s.useNeural && neural != nullptr && neuralLatency > 0 && neural->hasModel();
    neuralLeadActive = neuralActive;

    // An AI lead re-sings the performance with its own vibrato: don't stack the profile's on top.
    auto pitchSettings = s.pitch;
    pitchSettings.naturalVibrato = pitchSettings.naturalVibrato || neuralActive;

    //==========================================================================
    // 1. Analysis
    const juce::int64 start = blockStart;
    if (! s.pitch.sourceIsSynth)
    {
        genderEstimator.process (in, n);
        mapper.setSourceGenderEstimate ((int) genderEstimator.getGender());
    }
    detector.process (in, n, [&] (int i, const PitchDetector::Frame& f)
    {
        if (f.voiced)
            genderEstimator.addPitch (f.frequencyHz);
        const juce::int64 t = start + i + 1 - timeOffset;
        timeline.push (mapper.process (f, t, pitchSettings));
    });
    analyzer.push (in, n, timeline, placementLag);
    blockStart += n;

    const juce::int64 writePos = analyzer.getWritePosition();
    const juce::int64 tEnd = writePos - dspLatency;
    const juce::int64 tStart = tEnd - n;
    const double untilPos = (double) tEnd + (double) maxHalfGrain + 2.0;

    if (f0Capture != nullptr && ! timeline.isEmpty())
    {
        const double step = sr / f0CaptureRate;
        const double newest = (double) timeline.newest().time;
        while ((double) f0Capture->size() * step <= newest)
        {
            PitchFrame pf;
            timeline.lookup ((double) f0Capture->size() * step, pf);
            f0Capture->push_back (pf.voiced ? music::midiToHz (pf.srcMidi + pf.mainShift) : 0.0f);
        }
    }

    if (! timeline.isEmpty())
    {
        const auto& newest = timeline.newest();
        detectedHz = newest.voiced ? music::midiToHz (newest.srcMidi) : 0.0f;
        targetHz = newest.voiced ? music::midiToHz (newest.srcMidi + newest.mainShift) : 0.0f;
    }

    //==========================================================================
    // 2. Block-rate parameters
    formantSmoothed += (computeFormantSemis (s) - formantSmoothed) * 0.25f;
    const float formantRatio = semitonesToRatio (formantSmoothed);
    const int keyRoot = mapper.getEffectiveKeyRoot (s.pitch);
    const ScaleType scale = mapper.getEffectiveScale (s.pitch);

    std::fill (busL.begin(), busL.begin() + n, 0.0f);
    std::fill (busR.begin(), busR.begin() + n, 0.0f);

    //==========================================================================
    // 3. Lead voice (PSOLA)
    lead.synthesise (analyzer, untilPos, [&] (double pos, const PsolaAnalyzer::Mark&)
    {
        PitchFrame pf;
        timeline.lookup (pos, pf);
        PsolaVoice::GrainParams g;
        g.pitchRatio = semitonesToRatio (pf.mainShift);
        g.formantRatio = formantRatio;
        return g;
    });
    lead.read (tStart, leadBuf.data(), n);

    if (leadOverride != nullptr)
        std::copy (leadOverride, leadOverride + n, leadBuf.begin());

    if (! neuralActive)
    {
        for (int i = 0; i < n; ++i)
        {
            busL[(size_t) i] += leadBuf[(size_t) i];
            busR[(size_t) i] += leadBuf[(size_t) i];
        }
    }

    //==========================================================================
    // 4. Harmony voices
    for (int k = 0; k < 3; ++k)
    {
        const auto interval = s.harmonyInterval[k];
        const bool active = s.harmonyOn && interval != HarmonyInterval::Off && s.harmonyLevel[k] > 0.001f;

        if (active && ! harmonyActive[k])
            harmony[k].reset (untilPos);
        harmonyActive[k] = active;
        if (! active)
            continue;

        const float delaySamples = harmonyDelayMs[k] * 0.001f * (float) sr;
        const float sign = (int) interval < (int) HarmonyInterval::Unison ? -1.0f : 1.0f;
        const float fRatio = formantRatio * semitonesToRatio (harmonyFormant[k] + 0.15f * sign);

        harmony[k].synthesise (analyzer, untilPos, [&] (double pos, const PsolaAnalyzer::Mark&)
        {
            PitchFrame pf;
            timeline.lookup (pos - delaySamples, pf);
            PsolaVoice::GrainParams g;
            g.pitchRatio = semitonesToRatio (PitchMapper::harmonyShift (pf, interval, keyRoot, scale) + harmonyDetune[k] * 0.01f);
            g.formantRatio = fRatio;
            g.readDelay = delaySamples;
            return g;
        });
        harmony[k].read (tStart, voiceBuf.data(), n);

        float gl, gr;
        panGains (s.harmonyPan[k], gl, gr);
        const float level = s.harmonyLevel[k] * 0.8f;
        for (int i = 0; i < n; ++i)
        {
            busL[(size_t) i] += voiceBuf[(size_t) i] * level * gl;
            busR[(size_t) i] += voiceBuf[(size_t) i] * level * gr;
        }
    }

    //==========================================================================
    // 5. Doubler voices
    const bool vibeDoubler = s.vibe && traits.vibeDoubler > 0.05f;
    const bool doublerOn = s.doublerOn || vibeDoubler;
    const float doublerAmount = s.doublerOn ? s.doublerAmount : traits.vibeDoubler;
    const float doublerWidth = s.doublerOn ? s.doublerWidth : 0.8f;

    for (int k = 0; k < 2; ++k)
    {
        const bool active = doublerOn && doublerAmount > 0.001f;
        if (active && ! doublerActive[k])
            doubler[k].reset (untilPos);
        doublerActive[k] = active;
        if (! active)
            continue;

        const float side = k == 0 ? -1.0f : 1.0f;
        const float baseDetune = side * (4.0f + 6.0f * doublerAmount);
        const double baseDelay = (k == 0 ? 0.014 : 0.022) * sr;
        const double delayDepth = (k == 0 ? 0.005 : 0.006) * sr;
        const float fRatio = formantRatio * semitonesToRatio (0.12f * side);

        doubler[k].synthesise (analyzer, untilPos, [&] (double pos, const PsolaAnalyzer::Mark&)
        {
            const float readDelay = (float) (baseDelay + delayDepth * lfo (pos, sr, k == 0 ? 0.21 : 0.27, k * 1.3));
            PitchFrame pf;
            timeline.lookup (pos - readDelay, pf);
            PsolaVoice::GrainParams g;
            g.pitchRatio = semitonesToRatio (pf.mainShift + 0.01f * (baseDetune + 2.5f * lfo (pos, sr, k == 0 ? 0.13 : 0.17, k * 2.0)));
            g.formantRatio = fRatio;
            g.readDelay = readDelay;
            return g;
        });
        doubler[k].read (tStart, voiceBuf.data(), n);

        float gl, gr;
        panGains (side * doublerWidth, gl, gr);
        const float level = doublerAmount * 0.7f;
        for (int i = 0; i < n; ++i)
        {
            busL[(size_t) i] += voiceBuf[(size_t) i] * level * gl;
            busR[(size_t) i] += voiceBuf[(size_t) i] * level * gr;
        }
    }

    //==========================================================================
    // 6. Latency alignment + neural lead
    if (alignL.delay > 0)
    {
        for (int i = 0; i < n; ++i)
        {
            busL[(size_t) i] = alignL.process (busL[(size_t) i]);
            busR[(size_t) i] = alignR.process (busR[(size_t) i]);
        }
    }

    if (neural != nullptr && neuralLatency > 0)
    {
        // Feed the model input delayed far enough that its target pitch is known.
        const int f0Lag = detector.getAnalysisDelay() + 2 * detector.getHopSize();
        const juce::int64 feedStart = writePos - f0Lag - n;
        for (int i = 0; i < n; ++i)
        {
            const auto t = feedStart + i;
            neuralIn[(size_t) i] = t >= 0 ? analyzer.interpolate ((double) t) : 0.0f;
            PitchFrame pf;
            timeline.lookup ((double) t, pf);
            f0Buf[(size_t) i] = pf.voiced ? music::midiToHz (pf.srcMidi + pf.mainShift) : 0.0f;
        }
        neural->process (neuralIn.data(), f0Buf.data(), neuralOut.data(), n, neuralCoverage.data());

        // Where the neural lead has nothing (warming up, after a reset, a chunk this computer could not
        // convert in time) the DSP lead fills in, crossfaded by the stream's coverage: never a gap.
        for (int i = 0; i < n; ++i)
        {
            const float v = alignNeural.process (neuralOut[(size_t) i]);
            const float c = alignCoverage.process (neuralCoverage[(size_t) i]);
            const float fill = alignLead.process (leadBuf[(size_t) i]);

            if (neuralActive)
            {
                busL[(size_t) i] += v + fill * (1.0f - c);
                busR[(size_t) i] += v + fill * (1.0f - c);
            }
        }
    }

    //==========================================================================
    // 7. Character
    const bool synth = s.pitch.sourceIsSynth;
    const float I = s.intensity;
    CharacterProcessor::Settings cs;
    cs.lowCutHz = juce::jmax (s.lowCutHz, s.vibe ? traits.vibeLowCutHz : 20.0f);
    cs.tiltDb = traits.tiltDb * I - s.tone * 6.0f - s.age * 1.5f;
    cs.warmthDb = traits.warmthDb * I + juce::jmax (0.0f, s.age) * 1.0f;
    cs.nasalDb = traits.nasalDb * I;
    cs.ringHz = traits.ringHz;
    cs.ringDb = traits.ringDb * I;
    cs.bodyHz = traits.bodyHz;
    cs.bodyDb = traits.bodyDb * I;
    cs.presenceDb = s.presenceDb + traits.presenceDb * I;
    cs.airDb = s.airDb + traits.airDb * I;
    cs.breath = juce::jlimit (0.0f, 1.0f, s.breath + (synth ? 0.0f : traits.breathiness * I) + juce::jmax (0.0f, -s.age) * 0.08f);
    cs.rasp = juce::jlimit (0.0f, 1.0f, s.rasp + (synth ? 0.0f : traits.rasp * I * styleScale) + juce::jmax (0.0f, s.age) * 0.12f);
    character.process (busL.data(), busR.data(), n, cs);

    //==========================================================================
    // 8. Production FX (user settings combined with the voice's "vibe")
    const float deEss = juce::jmax (s.deEsser, s.vibe ? traits.vibeDeEsser : 0.0f);
    const float comp = juce::jmax (s.compressor, s.vibe ? traits.vibeCompressor : 0.0f);
    const float sat = juce::jlimit (0.0f, 1.0f, s.saturation + (s.vibe ? traits.vibeSaturation * styleScale : 0.0f));

    const bool vibeDelay = s.vibe && traits.vibeDelayMix > s.delayMix;
    const float delayMix = vibeDelay ? traits.vibeDelayMix : s.delayMix;
    const int division = vibeDelay ? traits.vibeDelayDivision : s.delayDivision;
    const float delaySeconds = (float) (60.0 / juce::jlimit (30.0, 300.0, s.bpm) * delayDivisionInBeats (division));

    const bool vibeReverb = s.vibe && traits.vibeReverbMix > s.reverbMix;
    const float reverbMix = vibeReverb ? traits.vibeReverbMix : s.reverbMix;
    const float reverbSize = vibeReverb ? traits.vibeReverbSize : s.reverbSize;

    deEsser.process (busL.data(), busR.data(), n, deEss);
    compressor.process (busL.data(), busR.data(), n, comp);
    saturator.process (busL.data(), busR.data(), n, sat);
    delay.process (busL.data(), busR.data(), n, delayMix, delaySeconds, s.delayFeedback);
    reverb.process (busL.data(), busR.data(), n, reverbMix, reverbSize);

    std::copy (busL.begin(), busL.begin() + n, outL);
    std::copy (busR.begin(), busR.begin() + n, outR);
}

} // namespace voxa::dsp
