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
    sourceNotes.fill (0.0f);
    sourceNotesTotal = 0.0f;
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
    // Chromatic needs no key, so it applies with Auto key too.
    if (s.keyRoot >= 0 || s.scale == ScaleType::Chromatic)
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

float PitchMapper::sourcePercentile (float fraction) const noexcept
{
    if (sourceNotesTotal <= 0.0f)
        return sourceMedian;
    const float wanted = fraction * sourceNotesTotal;
    float sum = 0.0f;
    for (size_t n = 0; n < sourceNotes.size(); ++n)
        if ((sum += sourceNotes[n]) >= wanted)
            return (float) n;
    return sourceMedian;
}

void PitchMapper::updateRegister (const Settings& s) noexcept
{
    const bool ai = s.aiRegister && s.traits.aiCentreMidi > 0.0f;
    const int centre = ai ? 1000 + juce::roundToInt (s.traits.aiCentreMidi) : s.traits.centreMidi();
    const int source = s.sourceGender != 0 ? s.sourceGender : genderEstimate;

    // Once the input voice is known the octave is locked, so phrases never jump between registers.
    // It is re-evaluated only for a new target voice or a different source gender (an AI voice: see below).
    const bool sameTarget = registerLocked && centre == lockedCentre && source == lockedSource;
    if (sameTarget && ! ai)
        return;

    // Male -> female moves up an octave, female -> male down; same-gender pairs stay put.
    int k = 0;
    if (source == 1 && s.traits.gender == VoiceGender::Female) k = 1;
    if (source == 2 && s.traits.gender == VoiceGender::Male)   k = -1;

    // Keep the performance inside the target voice's register (with a little headroom).
    const bool heard = sourceWeight > 0.5f;
    if (ai && (heard || source != 0))
    {
        // An AI model sounds right only near the notes its singer sang (far above them it turns to noise):
        // the octave closest to the middle of that range, the gender's usual direction on a near tie. The song's
        // high notes count too: above the singer's top a voice turns thin and "chipmunk", which is worse than
        // dipping below the bottom.
        const int preferred = k;
        // Until the input has been heard for a moment: a typical singer of its gender.
        const float aim = s.traits.aiCentreMidi + s.traits.registerBiasSemis;
        const float median = heard ? sourceMedian : (source == 2 ? 64.0f : 55.0f);
        const float low = heard ? sourcePercentile (0.05f) : median - 4.0f;
        const float high = heard ? sourcePercentile (0.95f) : median + 4.0f;
        auto cost = [&] (int o)
        {
            const float shift = 12.0f * (float) o;
            float c = std::abs (median + shift - aim) + (o == preferred ? 0.0f : 2.0f);
            if (s.traits.aiHighMidi > 0)
                c += 1.25f * juce::jmax (0.0f, high + shift - (float) s.traits.aiHighMidi)
                   + 0.75f * juce::jmax (0.0f, (float) s.traits.aiLowMidi - (low + shift));
            return c;
        };
        for (int o = -2; o <= 2; ++o)
            if (cost (o) < cost (k))
                k = o;

        // The first phrases can be unrepresentative of the whole song, so the choice stays open (checked at
        // every phrase start) but only changes when another octave is clearly closer: no back and forth.
        if (sameTarget && cost (registerOctaves) <= cost (k) + 3.0f)
            k = registerOctaves;
    }
    else if (heard)
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

        sourceNotes[(size_t) juce::jlimit (0, 127, (int) std::lround (src))] += w * hopSec;
        if ((sourceNotesTotal += w * hopSec) > 240.0f)   // a few minutes: an old song fades out
        {
            for (auto& n : sourceNotes) n *= 0.5f;
            sourceNotesTotal *= 0.5f;
        }

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
        // From the very first phrase: until the input has been heard for a moment the source gender's usual
        // octave (a male input into a female voice goes up), so the first phrase is never an octave off the rest.
        if (! s.pitchPreserve && ! s.sourceIsSynth)
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
    const bool hardTune = s.traits.hardTune && ! midiControlled;   // a hard-tune voice: instant, exact notes
    const bool correcting = (s.tuneOn || midiControlled || hardTune) && ! s.sourceIsSynth;

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
            const float retune = hardTune ? 0.0f : (midiControlled ? juce::jmax (10.0f, s.retuneMs) : s.retuneMs);
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
        corrected = corrState + deviation * (hardTune ? 0.0f : s.humanize) * 0.7f;
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

    // An AI lead keeps the singer's own attacks too, unless correction has flattened them.
    if (profileExpression && onset && (correcting || ! s.naturalVibrato))
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
