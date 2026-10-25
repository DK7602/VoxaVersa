#include "PitchMapper.h"

namespace voxa::dsp
{

void PitchMapper::prepare (double sampleRate, int hopSamples)
{
    sr = sampleRate;
    hopSec = (float) (hopSamples / sampleRate);
    reset();
}

void PitchMapper::reset()
{
    corrState = 0.0f;
    currentNote = -1;
    wasVoiced = false;
    unvoicedTime = 1.0f;
    noteTime = 0.0f;
    scoopOffset = 0.0f;
    vibPhase = 0.0f;
    vibRateJitter = 0.0f;
    vibDepthRamp = 0.0f;
    jitterState = 0.0f;
    sourceMedian = 0.0f;
    sourceWeight = 0.0f;
    genderEstimate = 0;
    registerOctaves = 0;
    registerLocked = false;
    lockedCentre = -1000;
    lockedSource = -1;
    keyDetector.reset();
    keyKnown = false;
    keyUpdateCounter = 0;
}

int PitchMapper::getEffectiveKeyRoot (const Settings& s) const noexcept
{
    if (s.keyRoot >= 0)
        return s.keyRoot;
    return keyKnown ? keyDetector.getKeyRoot() : 0;
}

ScaleType PitchMapper::getEffectiveScale (const Settings& s) const noexcept
{
    if (s.keyRoot >= 0)
        return s.scale;
    // Auto key: use the detected mode once known; chromatic until then.
    return keyKnown ? keyDetector.getScale() : ScaleType::Chromatic;
}

float PitchMapper::getSourceFormantBaseline (int setting) const noexcept
{
    const int g = setting != 0 ? setting : genderEstimate;
    return g == 1 ? -1.0f : (g == 2 ? 1.0f : 0.0f);
}

float PitchMapper::harmonyShift (const PitchFrame& f, HarmonyInterval interval, int keyRoot, ScaleType scale) noexcept
{
    if (! f.voiced || interval == HarmonyInterval::Off)
        return f.mainShift;

    const int base = (int) std::lround (f.melodyMidi);
    const int note = music::harmonyNote (base, interval, keyRoot, scale);
    const float target = (float) note + (f.melodyMidi - (float) base) + f.expression + f.globalOffset;
    return juce::jlimit (-36.0f, 36.0f, target - f.srcMidi);
}

void PitchMapper::updateRegister (const Settings& s) noexcept
{
    const int centre = s.traits.centreMidi();
    const int source = s.sourceGender != 0 ? s.sourceGender : genderEstimate;

    // Once the input voice is known the octave is locked, so phrases never jump between registers.
    // It is re-evaluated only for a new target voice or a different source gender.
    if (registerLocked && centre == lockedCentre && source == lockedSource)
        return;

    // Male -> female moves up an octave, female -> male down; same-gender pairs stay put.
    int k = 0;
    if (source == 1 && s.traits.gender == VoiceGender::Female) k = 1;
    if (source == 2 && s.traits.gender == VoiceGender::Male)   k = -1;

    // Keep the performance inside the target voice's register (with a little headroom).
    if (sourceWeight > 0.5f)
    {
        while (k > -2 && sourceMedian + 12.0f * (float) k > (float) s.traits.rangeHighMidi + 3.0f) --k;
        while (k < 2 && sourceMedian + 12.0f * (float) k < (float) s.traits.rangeLowMidi - 3.0f) ++k;
    }
    registerOctaves = k;

    if (sourceWeight > 1.5f && (source != 0 || s.traits.gender == VoiceGender::Neutral))
    {
        registerLocked = true;
        lockedCentre = centre;
        lockedSource = source;
    }
}

PitchFrame PitchMapper::process (const PitchDetector::Frame& frame, juce::int64 time, const Settings& s) noexcept
{
    PitchFrame out;
    out.time = time;
    out.voiced = frame.voiced && frame.frequencyHz > 0.0f;
    out.periodSamples = out.voiced ? frame.periodSamples : 0.0f;

    // Global offset: transposition + smart register.
    const float global = s.pitchSemis + s.fineCents * 0.01f + (s.pitchPreserve || s.sourceIsSynth ? 0.0f : 12.0f * (float) registerOctaves);
    out.globalOffset = global;

    if (! out.voiced)
    {
        unvoicedTime += hopSec;
        if (unvoicedTime > 0.08f)
        {
            wasVoiced = false;
            currentNote = -1;
        }
        out.srcMidi = 0.0f;
        out.mainShift = global;
        out.melodyMidi = corrState;
        return out;
    }

    const float src = music::hzToMidi (frame.frequencyHz);
    out.srcMidi = src;

    //--------------------------------------------------------------------------
    // Source analysis: running median-ish pitch, gender guess, key estimate.
    {
        const float w = juce::jlimit (0.0f, 1.0f, frame.confidence);
        const float tau = sourceWeight < 2.0f ? 0.25f : 3.0f;                 // fast start, then slow
        const float a = 1.0f - std::exp (-hopSec * w / tau);
        sourceMedian = sourceWeight <= 0.0f ? src : sourceMedian + (src - sourceMedian) * a;
        sourceWeight += hopSec * w;

        keyDetector.addPitch (src, w * hopSec);
        if (++keyUpdateCounter >= 64)
        {
            keyUpdateCounter = 0;
            keyDetector.decay (0.985f);
            keyDetector.update();
            keyKnown = keyDetector.getTotalWeight() > 2.0f && keyDetector.getConfidence() > 0.15f;
        }
    }

    //--------------------------------------------------------------------------
    // Phrase start: re-evaluate the register and trigger a scoop.
    const bool onset = ! wasVoiced;
    if (onset)
    {
        if (! s.pitchPreserve && ! s.sourceIsSynth && sourceWeight > 0.3f)
            updateRegister (s);
        noteTime = 0.0f;
        vibDepthRamp = 0.0f;
    }
    wasVoiced = true;
    unvoicedTime = 0.0f;

    //--------------------------------------------------------------------------
    // Correction / MIDI control.
    const int keyRoot = getEffectiveKeyRoot (s);
    const ScaleType scale = getEffectiveScale (s);
    const bool midiControlled = s.midiNote >= 0;
    const bool correcting = (s.tuneOn || midiControlled) && ! s.sourceIsSynth;

    float corrected = src;

    if (correcting)
    {
        int target;
        if (midiControlled)
        {
            target = s.midiNote;
        }
        else
        {
            const int candidate = music::nearestScaleNote (src, keyRoot, scale);
            // Hysteresis: only leave the current note when clearly closer to another.
            if (currentNote >= 0 && music::isInScale (currentNote, keyRoot, scale)
                && std::abs (src - (float) currentNote) < std::abs (src - (float) candidate) + 0.18f)
                target = currentNote;
            else
                target = candidate;
        }

        if (onset || currentNote < 0)
        {
            corrState = (float) target;
        }
        else
        {
            const float retune = midiControlled ? juce::jmax (10.0f, s.retuneMs) : s.retuneMs;
            if (retune < 1.0f)
            {
                corrState = (float) target;
            }
            else
            {
                const float tau = retune * 0.001f / 2.3f;
                corrState += ((float) target - corrState) * (1.0f - std::exp (-hopSec / tau));
            }
        }

        if (target != currentNote && currentNote >= 0 && std::abs (target - currentNote) >= 1)
        {
            noteTime = 0.0f;
            vibDepthRamp = 0.0f;
            if (s.expression && ! midiControlled)
                scoopOffset = juce::jmin (scoopOffset, -0.5f * s.traits.scoopCents * 0.01f * s.styleScale);
        }
        currentNote = target;

        const float deviation = src - (float) target;
        corrected = corrState + deviation * s.humanize * 0.7f;
    }
    else
    {
        corrState = src;
        const int n = (int) std::lround (src);
        if (currentNote >= 0 && std::abs (n - currentNote) >= 2)
            noteTime = 0.0f;
        currentNote = n;
    }

    out.melodyMidi = corrected;
    noteTime += hopSec;

    //--------------------------------------------------------------------------
    // Expression: scoops, vibrato, jitter.
    float expression = 0.0f;
    const bool profileExpression = s.expression && ! s.sourceIsSynth;

    if (profileExpression && onset)
        scoopOffset = -s.traits.scoopCents * 0.01f * s.styleScale;

    if (scoopOffset < 0.0f)
    {
        const float tau = juce::jmax (0.01f, s.traits.scoopMs * 0.001f / 3.0f);
        scoopOffset *= std::exp (-hopSec / tau);
        if (scoopOffset > -0.002f)
            scoopOffset = 0.0f;
        expression += scoopOffset;
    }

    // Profile vibrato is strongest when correction has flattened the natural vibrato.
    const float ageVibrato = 1.0f + 0.35f * juce::jmax (0.0f, s.age);
    float profileDepth = profileExpression
                       ? s.traits.vibratoDepthCents * s.styleScale * ageVibrato * (correcting ? 1.0f : (s.naturalVibrato ? 0.0f : 0.3f))
                       : 0.0f;
    const float depthCents = profileDepth + s.userVibratoCents;

    if (depthCents > 0.1f)
    {
        const float onsetSec = s.traits.vibratoOnsetMs * 0.001f;
        if (noteTime > onsetSec)
            vibDepthRamp = juce::jmin (1.0f, vibDepthRamp + hopSec / 0.25f);

        vibRateJitter += (noise.next() * 0.02f - vibRateJitter * 0.01f);
        const float rate = (s.userVibratoCents > profileDepth ? s.userVibratoRate : s.traits.vibratoRateHz)
                           * (1.0f - 0.12f * juce::jmax (0.0f, s.age)) * (1.0f + juce::jlimit (-0.06f, 0.06f, vibRateJitter));
        vibPhase += juce::MathConstants<float>::twoPi * rate * hopSec;
        if (vibPhase > juce::MathConstants<float>::twoPi)
            vibPhase -= juce::MathConstants<float>::twoPi;

        expression += depthCents * 0.01f * vibDepthRamp * std::sin (vibPhase);
    }

    if (profileExpression && correcting)
    {
        const float jitterCents = s.traits.jitterCents * (1.0f + 0.8f * juce::jmax (0.0f, s.age)) * juce::jmin (1.5f, s.styleScale);
        jitterState += noise.next() * 0.25f - jitterState * 0.15f;
        expression += jitterState * jitterCents * 0.01f;
    }

    out.expression = expression;
    const float target = corrected + expression + global;
    out.mainShift = juce::jlimit (-36.0f, 36.0f, target - src);
    return out;
}

} // namespace voxa::dsp
