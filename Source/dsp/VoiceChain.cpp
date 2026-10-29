#include "VoiceChain.h"
#include "../neural/NeuralVoice.h"

#include <limits>

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

    // The choir section: who sings (even: the lead's singer, odd: the other section's), how late (ms), how far
    // the delay drifts (ms: the detune), pan, and each copy's two resonances (Hz, dB).
    struct SectionVoice { float delayMs, driftMs, rateA, rateB, pan, hzA, dbA, hzB, dbB; };
    constexpr SectionVoice sectionVoices[] {
        { 11.0f, 4.0f, 0.13f, 0.47f, -0.85f,  750.0f,  3.0f, 2600.0f, -3.0f },
        { 16.0f, 5.0f, 0.17f, 0.53f,  0.85f,  900.0f, -3.0f, 3200.0f,  2.5f },
        { 23.0f, 4.5f, 0.11f, 0.61f, -0.5f,   550.0f,  2.0f, 2200.0f,  2.0f },
        { 29.0f, 5.5f, 0.19f, 0.43f,  0.5f,   650.0f, -2.0f, 3000.0f, -3.0f },
        { 37.0f, 4.0f, 0.23f, 0.71f, -0.2f,  1100.0f,  3.0f, 2400.0f, -2.0f },
        { 44.0f, 6.0f, 0.07f, 0.37f,  0.2f,   800.0f, -2.5f, 2800.0f,  3.0f },
    };
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
    rLookAhead = juce::jmin (juce::roundToInt (RKeeper::lookAheadSeconds * sr), dspLatency);

    lead.prepare (dspLatency);
    for (auto& h : harmony) h.prepare (dspLatency);
    for (auto& d : doubler) d.prepare (dspLatency);

    consonants.prepare (sr);
    rKeeper.prepare (sr);
    vowels.prepare (sr);
    grit.prepare (sr);
    whisperizer.prepare (sr);
    breaths.prepare (sr);
    character.prepare (sr);
    deEsser.prepare (sr);
    compressor.prepare (sr);
    saturator.prepare (sr);
    delay.prepare (sr);
    reverb.prepare (sr);

    for (auto* b : { &leadBuf, &voiceBuf, &dryBuf, &aheadBuf, &voicedBuf, &voicedNowBuf, &f0Buf, &shiftBuf, &neuralIn, &neuralOut, &neuralCoverage, &busL, &busR, &dryHzBuf, &leadHzBuf, &whisperIn, &whisperHz })
        b->assign ((size_t) maxBlock, 0.0f);

    const int maxAlign = (int) (sr * 3.0);
    alignL.prepare (maxAlign);
    alignR.prepare (maxAlign);
    alignNeural.prepare (maxAlign);
    alignLead.prepare (maxAlign);
    alignCoverage.prepare (maxAlign);
    alignDry.prepare (maxAlign);
    alignAhead.prepare (maxAlign);
    for (auto& d : aiDoubler)
        d.prepare ((int) std::ceil (sr * 0.04));
    alignPartner.prepare (maxAlign);
    for (int k = 0; k < sectionSize; ++k)
    {
        const auto& v = sectionVoices[k];
        section[k].delay.prepare ((int) std::ceil (sr * (v.delayMs + v.driftMs + 2.0f) * 0.001));
        section[k].tractA.c = BiquadCoeffs::peak (sr, v.hzA, 1.5, v.dbA);
        section[k].tractB.c = BiquadCoeffs::peak (sr, v.hzB, 1.5, v.dbB);
    }
    for (auto* b : { &partnerF0, &partnerShift, &partnerOut, &partnerCoverage, &partnerBuf })
        b->assign ((size_t) maxBlock, 0.0f);

    setNeuralStream (neural);
    reset();
}

bool VoiceChain::setNeuralStream (neural::NeuralVoiceStream* stream)
{
    neural = stream;

    const int newNeural = neural != nullptr ? detector.getAnalysisDelay() + 2 * detector.getHopSize() + neural->getLatencySamples() : 0;
    const int newTotal = juce::jmax (dspLatency, newNeural);

    if (newNeural == neuralLatency && newTotal == totalLatency && alignL.delay == newTotal - dspLatency)
        return false;

    neuralLatency = newNeural;
    totalLatency = newTotal;
    alignL.delay = alignR.delay = alignLead.delay = alignDry.delay = alignAhead.delay = totalLatency - dspLatency;
    alignNeural.delay = alignCoverage.delay = alignPartner.delay = neuralLatency > 0 ? totalLatency - neuralLatency : 0;
    alignL.clear();
    alignR.clear();
    alignNeural.clear();
    alignLead.clear();
    alignCoverage.clear();
    alignDry.clear();
    alignAhead.clear();
    alignPartner.clear();
    return true;
}

void VoiceChain::clearTails()
{
    delay.reset();
    reverb.reset();
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
    consonants.reset();
    rKeeper.reset();
    vowels.reset();
    grit.reset();
    whisperizer.reset();
    breaths.reset();
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
    alignDry.clear();
    alignAhead.clear();
    for (auto& d : aiDoubler)
        d.clear();
    aiDoublerActive = false;
    alignPartner.clear();
    for (auto& v : section)
    {
        v.delay.clear();
        v.tractA.reset();
        v.tractB.reset();
    }
    sectionActive = partnerActive = false;
    blockStart = 0;
    formantSmoothed = 0.0f;
    detectedHz = targetHz = 0.0f;
}

void VoiceChain::setF0Capture (std::vector<float>* destination, double frameRate, std::vector<float>* shiftDestination)
{
    f0Capture = destination;
    f0CaptureRate = frameRate;
    shiftCapture = destination != nullptr ? shiftDestination : nullptr;
}

float VoiceChain::neuralPitchShift (const PitchFrame& pf, const PitchMapper::Settings& s) noexcept
{
    // MIDI notes, the Text to Sing guide and a hard-tune voice: exactly the target pitch, nothing of the input's
    // (a hard-tune voice where the live detector heard a note: elsewhere, e.g. a fast slide it lost, the neural
    // tracker's pitch keeps the AI singing instead of breaking up).
    if (s.midiNote >= 0 || s.sourceIsSynth || (s.traits.hardTune && pf.voiced))
        return std::numeric_limits<float>::quiet_NaN();

    // Correction moves a note by less than a semitone (and the profile's vibrato and scoops by less still);
    // anything bigger is the live detector reading a different note than the one the input sings.
    const float moves = pf.voiced ? juce::jlimit (-1.5f, 1.5f, pf.mainShift - pf.globalOffset) : 0.0f;
    return pf.globalOffset + moves;
}

void VoiceChain::keepVowelsAndAddGrit (float* lead, juce::int64 from, int n, const ChainSettings& s) noexcept
{
    // both voices' pitches on the lead's timeline (`from`: the input time of its first sample)
    for (int i = 0; i < n; ++i)
    {
        PitchFrame pf;
        const bool voiced = timeline.lookup ((double) (from + i), pf) && pf.voiced;
        dryHzBuf[(size_t) i] = voiced ? music::midiToHz (pf.srcMidi) : 0.0f;
        leadHzBuf[(size_t) i] = voiced ? music::midiToHz (pf.srcMidi + pf.mainShift) : 0.0f;
    }
    // the AI singer's formants over the input's: a man's voice into a woman's ~1.17 (a Text to Sing guide is
    // already the target voice's)
    const auto& traits = s.pitch.traits;
    const int source = s.pitch.sourceIsSynth ? 0 : (s.pitch.sourceGender != 0 ? s.pitch.sourceGender : (int) genderEstimator.getGender());
    const float tract = source == 1 && traits.gender == VoiceGender::Female ? 1.17f
                      : (source == 2 && traits.gender == VoiceGender::Male ? 1.0f / 1.17f : 1.0f);
    vowels.setTopNote (traits.aiHighMidi > 0 ? (float) traits.aiHighMidi - 1.0f : 0.0f);
    vowels.process (lead, dryBuf.data(), dryHzBuf.data(), leadHzBuf.data(), n, tract);
    if (traits.aiGrit > 0.0f)   // an AI singer's rasp (Male Rap, Male Blues)
        grit.process (lead, leadHzBuf.data(), n, juce::jlimit (0.0f, 1.0f, traits.aiGrit * (1.0f + s.style)));
}

void VoiceChain::fillInputVoicing (juce::int64 from, int n, std::vector<float>& dest) noexcept
{
    PitchFrame pf;
    for (int i = 0; i < n; ++i)
        dest[(size_t) i] = timeline.lookup ((double) (from + i), pf) && pf.voiced ? 1.0f : 0.0f;
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
    const bool aiLead = neuralActive || s.aiLead;

    // An AI lead re-sings the performance with its own vibrato: don't stack the profile's on top.
    auto pitchSettings = s.pitch;
    pitchSettings.naturalVibrato = pitchSettings.naturalVibrato || aiLead;
    pitchSettings.aiRegister = aiLead;

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
            if (shiftCapture != nullptr)
                shiftCapture->push_back (neuralPitchShift (pf, pitchSettings));
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

    // The dry voice on the lead's timeline (an AI lead keeps its unvoiced consonants).
    for (int i = 0; i < n; ++i)
        dryBuf[(size_t) i] = analyzer.interpolate ((double) (tStart + i));

    // A whispering voice: the input itself, its pitch removed (read early by the whisperizer's latency, which
    // the DSP voice's own delay covers).
    if (traits.whisper && ! s.pitch.sourceIsSynth)
    {
        const int early = juce::jmin (whisperizer.getLatency(), dspLatency);
        for (int i = 0; i < n; ++i)
        {
            PitchFrame pf;
            const auto t = tStart + early + i;
            whisperIn[(size_t) i] = analyzer.interpolate ((double) t);
            whisperHz[(size_t) i] = timeline.lookup ((double) t, pf) && pf.voiced ? music::midiToHz (pf.srcMidi) : 0.0f;
        }
        whisperizer.process (whisperIn.data(), whisperHz.data(), leadBuf.data(), n);
    }

    // ... and a little earlier for the R keeper (the input is ahead of the lead by dspLatency).
    const auto aheadStart = tStart + rLookAhead;
    for (int i = 0; i < n; ++i)
        aheadBuf[(size_t) i] = analyzer.interpolate ((double) (aheadStart + i));
    {
        const int source = s.pitch.sourceGender != 0 ? s.pitch.sourceGender : (int) genderEstimator.getGender();
        rKeeper.setCeilings (source == 2 ? RKeeper::femaleCeilingHz : RKeeper::maleCeilingHz,
                             s.pitch.traits.gender == VoiceGender::Male ? RKeeper::maleCeilingHz : RKeeper::femaleCeilingHz);
    }

    if (leadOverride != nullptr)
    {
        std::copy (leadOverride, leadOverride + n, leadBuf.begin());
        if (s.aiLead)
        {
            consonants.process (leadBuf.data(), dryBuf.data(), n);
            fillInputVoicing (aheadStart, n, voicedBuf);
            rKeeper.process (leadBuf.data(), aheadBuf.data(), voicedBuf.data(), n);
            keepVowelsAndAddGrit (leadBuf.data(), tStart, n, s);
        }
    }

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

    // An AI lead is doubled after the model (section 6), with the AI voice itself; a choir's section replaces it.
    const bool aiDoublerOn = aiLead && doublerOn && doublerAmount > 0.001f && ! traits.choir;
    for (int k = 0; k < 2; ++k)
    {
        const bool active = doublerOn && doublerAmount > 0.001f && ! aiLead;
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
            shiftBuf[(size_t) i] = neuralPitchShift (pf, pitchSettings);
        }
        neural->process (neuralIn.data(), f0Buf.data(), neuralOut.data(), n, neuralCoverage.data(), shiftBuf.data());

        // A choir's second singer sings the same input an octave away (silent until its stream has audio).
        const bool partnerOn = neuralActive && traits.choir && s.choirPartnerOctave != 0 && partner != nullptr && partner->hasModel();
        if (partnerOn && ! partnerActive)
        {
            partner->reset();
            alignPartner.clear();
        }
        partnerActive = partnerOn;
        if (partnerOn)
        {
            const float semis = 12.0f * (float) s.choirPartnerOctave;
            const float ratio = semitonesToRatio (semis);
            for (int i = 0; i < n; ++i)
            {
                partnerF0[(size_t) i] = f0Buf[(size_t) i] * ratio;
                partnerShift[(size_t) i] = shiftBuf[(size_t) i] + semis;   // NaN stays NaN
            }
            partner->process (neuralIn.data(), partnerF0.data(), partnerOut.data(), n, partnerCoverage.data(), partnerShift.data());
            for (int i = 0; i < n; ++i)
                partnerBuf[(size_t) i] = alignPartner.process (partnerOut[(size_t) i] * partnerCoverage[(size_t) i]);
        }

        // Where the neural lead has nothing (warming up, after a reset, a chunk this computer could not
        // convert in time) the DSP lead fills in, crossfaded by the stream's coverage: never a gap.
        for (int i = 0; i < n; ++i)
        {
            const float v = alignNeural.process (neuralOut[(size_t) i]);
            const float c = alignCoverage.process (neuralCoverage[(size_t) i]);
            const float fill = alignLead.process (leadBuf[(size_t) i]);
            dryBuf[(size_t) i] = alignDry.process (dryBuf[(size_t) i]);   // now aligned with the neural lead
            aheadBuf[(size_t) i] = alignAhead.process (aheadBuf[(size_t) i]);
            voiceBuf[(size_t) i] = v + fill * (1.0f - c);
        }

        if (neuralActive)
        {
            consonants.process (voiceBuf.data(), dryBuf.data(), n);
            fillInputVoicing (aheadStart - alignAhead.delay, n, voicedBuf);
            rKeeper.process (voiceBuf.data(), aheadBuf.data(), voicedBuf.data(), n);
            keepVowelsAndAddGrit (voiceBuf.data(), tStart - alignDry.delay, n, s);   // dryBuf's input time
            for (int i = 0; i < n; ++i)
            {
                busL[(size_t) i] += voiceBuf[(size_t) i];
                busR[(size_t) i] += voiceBuf[(size_t) i];
            }
        }
    }

    // Doubles of an AI lead: copies of the AI voice itself, slightly late and slowly drifting in pitch (same
    // delays, drift rates and pan as the DSP doubler; the drift depth gives 2 to 6 cents of detune).
    if (aiDoublerOn && ! aiDoublerActive)
        for (auto& d : aiDoubler)
            d.clear();
    aiDoublerActive = aiDoublerOn;

    if (aiDoublerOn)
    {
        const float* aiLeadSignal = neuralActive ? voiceBuf.data() : leadBuf.data();
        float gl[2], gr[2];
        for (int k = 0; k < 2; ++k)
            panGains ((k == 0 ? -1.0f : 1.0f) * doublerWidth, gl[k], gr[k]);
        const float level = doublerAmount * 0.7f;
        const double driftDepth = (0.001 + 0.002 * doublerAmount) * sr;

        for (int i = 0; i < n; ++i)
        {
            const double pos = (double) (tStart + i);
            for (int k = 0; k < 2; ++k)
            {
                const double base = (k == 0 ? 0.014 : 0.022) * sr;
                const double drift = driftDepth * (0.75 * lfo (pos, sr, k == 0 ? 0.21 : 0.27, k * 1.3)
                                                   + 0.25 * lfo (pos, sr, k == 0 ? 0.53 : 0.61, k * 2.0));
                const float y = aiDoubler[k].process (aiLeadSignal[(size_t) i], (float) (base + drift));
                busL[(size_t) i] += y * level * gl[k];
                busR[(size_t) i] += y * level * gr[k];
            }
        }
    }

    // A choir with an AI singer: the other section's singer and six copies of the two around the lead.
    const bool sectionOn = aiLead && traits.choir;
    if (sectionOn && ! sectionActive)
        for (auto& v : section)
        {
            v.delay.clear();
            v.tractA.reset();
            v.tractB.reset();
        }
    sectionActive = sectionOn;

    if (sectionOn)
    {
        const float* leadSignal = neuralActive ? voiceBuf.data() : leadBuf.data();
        const bool withPartner = partnerActive && neuralActive;
        const float amount = s.doublerOn ? juce::jmax (0.2f, s.doublerAmount) : 1.0f;
        const float width = s.doublerOn ? s.doublerWidth : 1.0f;
        float gl[sectionSize], gr[sectionSize];
        for (int k = 0; k < sectionSize; ++k)
            panGains (sectionVoices[k].pan * width, gl[k], gr[k]);
        float pl, pr;
        panGains (0.4f * width, pl, pr);
        const float copyLevel = 0.4f * amount, partnerLevel = 0.62f;

        for (int i = 0; i < n; ++i)
        {
            const double pos = (double) (tStart + i);
            const float leadVoice = leadSignal[(size_t) i];
            const float otherVoice = withPartner ? partnerBuf[(size_t) i] : leadVoice;
            if (withPartner)
            {
                busL[(size_t) i] += otherVoice * partnerLevel * pl;
                busR[(size_t) i] += otherVoice * partnerLevel * pr;
            }
            for (int k = 0; k < sectionSize; ++k)
            {
                const auto& v = sectionVoices[k];
                const double drift = v.driftMs * (0.65 * lfo (pos, sr, v.rateA, k * 1.7) + 0.35 * lfo (pos, sr, v.rateB, k * 0.9));
                const float d = (float) ((v.delayMs + drift) * 0.001 * sr);
                float y = section[k].delay.process (k % 2 == 0 ? leadVoice : otherVoice, d);
                y = section[k].tractB.process (section[k].tractA.process (y));
                busL[(size_t) i] += y * copyLevel * gl[k];
                busR[(size_t) i] += y * copyLevel * gr[k];
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
    // An AI singer already breathes and has its own grit: the profile's synthetic breath noise and rasp on top
    // of it only sound like static. The Breath, Rasp and Age controls still add them when turned up.
    const bool profileTexture = ! synth && ! aiLead;
    cs.breath = juce::jlimit (0.0f, 1.0f, s.breath + (profileTexture ? traits.breathiness * I : 0.0f) + juce::jmax (0.0f, -s.age) * 0.08f);
    cs.rasp = juce::jlimit (0.0f, 1.0f, s.rasp + (profileTexture ? traits.rasp * I * styleScale : 0.0f) + juce::jmax (0.0f, s.age) * 0.12f);
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

    // Breaths at the singer's own level (after the EQ and compressor, which lift them too; `dryBuf` is aligned
    // with the bus here). Not for the Text to Sing guide, whose breaths are synthesized.
    if (! synth)
    {
        fillInputVoicing (tStart - alignDry.delay, n, voicedNowBuf);
        breaths.process (busL.data(), busR.data(), dryBuf.data(), voicedNowBuf.data(), n);
    }
    delay.process (busL.data(), busR.data(), n, delayMix, delaySeconds, s.delayFeedback);
    reverb.process (busL.data(), busR.data(), n, reverbMix, reverbSize);

    std::copy (busL.begin(), busL.begin() + n, outL);
    std::copy (busR.begin(), busR.begin() + n, outR);
}

} // namespace voxa::dsp
