#include "Parameters.h"
#include "core/MusicTheory.h"

namespace voxa
{

namespace Choices
{
    const juce::StringArray& mode()
    {
        static const juce::StringArray c { "Voice Transform", "Text to Sing" };
        return c;
    }

    const juce::StringArray& quality()
    {
        static const juce::StringArray c { "Low Latency", "Balanced", "High Quality" };
        return c;
    }

    const juce::StringArray& sourceVoice()
    {
        static const juce::StringArray c { "Auto", "Male", "Female" };
        return c;
    }

    const juce::StringArray& key()
    {
        static const juce::StringArray c = []
        {
            juce::StringArray k { "Auto" };
            k.addArray (music::getKeyNames());
            return k;
        }();
        return c;
    }

    const juce::StringArray& scale()            { return music::getScaleNames(); }
    const juce::StringArray& harmonyInterval()  { return music::getHarmonyIntervalNames(); }

    const juce::StringArray& delayTime()
    {
        static const juce::StringArray c { "1/4", "1/8", "1/8 Dotted", "1/16", "1/4 Dotted", "1/2" };
        return c;
    }

    const juce::StringArray& melodySource()
    {
        static const juce::StringArray c { "Use MIDI", "Auto Melody", "MIDI File", "Hum Melody" };
        return c;
    }
}

double delayDivisionInBeats (int choiceIndex) noexcept
{
    static constexpr double beats[] = { 1.0, 0.5, 0.75, 0.25, 1.5, 2.0 };
    return beats[juce::jlimit (0, 5, choiceIndex)];
}

//==============================================================================
namespace
{
    using Layout = juce::AudioProcessorValueTreeState::ParameterLayout;
    using APF = juce::AudioParameterFloat;
    using APB = juce::AudioParameterBool;
    using APC = juce::AudioParameterChoice;
    using Attr = juce::AudioParameterFloatAttributes;

    juce::ParameterID pid (const char* id) { return { id, 1 }; }

    juce::String dbString (float v, int)       { return (v > 0 ? "+" : "") + juce::String (v, 1) + " dB"; }
    juce::String pctString (float v, int)      { return juce::String (juce::roundToInt (v)) + "%"; }
    juce::String bipolarString (float v, int)  { return (v > 0.5f ? "+" : "") + juce::String (juce::roundToInt (v)); }
    juce::String semiString (float v, int)     { const int s = juce::roundToInt (v); return (s > 0 ? "+" : "") + juce::String (s) + " st"; }
    juce::String centString (float v, int)     { const int c = juce::roundToInt (v); return (c > 0 ? "+" : "") + juce::String (c) + " ct"; }
    juce::String hzString (float v, int)       { return v >= 1000.0f ? juce::String (v / 1000.0f, 2) + " kHz" : juce::String (juce::roundToInt (v)) + " Hz"; }
    juce::String msString (float v, int)       { return juce::String (juce::roundToInt (v)) + " ms"; }
    juce::String rateString (float v, int)     { return juce::String (v, 1) + " Hz"; }
    juce::String bpmString (float v, int)      { return juce::String (juce::roundToInt (v)) + " BPM"; }
    juce::String panString (float v, int)
    {
        const int p = juce::roundToInt (v);
        if (p == 0) return "C";
        return p < 0 ? juce::String (-p) + "L" : juce::String (p) + "R";
    }

    void addFloat (Layout& layout, const char* id, const juce::String& name, juce::NormalisableRange<float> range,
                   float def, juce::String (*toText) (float, int), const juce::String& label = {})
    {
        layout.add (std::make_unique<APF> (pid (id), name, range, def,
                                           Attr().withLabel (label).withStringFromValueFunction (toText)));
    }

    void addBool (Layout& layout, const char* id, const juce::String& name, bool def)
    {
        layout.add (std::make_unique<APB> (pid (id), name, def));
    }

    void addChoice (Layout& layout, const char* id, const juce::String& name, const juce::StringArray& choices, int def)
    {
        layout.add (std::make_unique<APC> (pid (id), name, choices, def));
    }

    juce::NormalisableRange<float> range (float lo, float hi, float step = 0.0f)
    {
        return { lo, hi, step };
    }

    juce::NormalisableRange<float> skewedRange (float lo, float hi, float centre)
    {
        juce::NormalisableRange<float> r { lo, hi };
        r.setSkewForCentre (centre);
        return r;
    }
}

juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
{
    Layout layout;
    namespace P = ParamID;

    // Global / IO
    addBool   (layout, P::bypass,     "Bypass", false);
    addChoice (layout, P::mode,       "Mode", Choices::mode(), 0);
    addFloat  (layout, P::inputGain,  "Input Gain",  range (-24.0f, 24.0f, 0.1f), 0.0f, dbString, "dB");
    addFloat  (layout, P::outputGain, "Output Gain", range (-24.0f, 24.0f, 0.1f), 0.0f, dbString, "dB");
    addFloat  (layout, P::mix,        "Mix",         range (0.0f, 100.0f, 1.0f), 100.0f, pctString, "%");
    addBool   (layout, P::autoLevel,  "Auto Level", false);
    addChoice (layout, P::quality,    "Quality", Choices::quality(), 1);
    addBool   (layout, P::aiEngine,   "AI Voice Engine", true);
    addChoice (layout, P::sourceVoice, "Source Voice", Choices::sourceVoice(), 0);

    // Voice character
    addFloat (layout, P::pitch,      "Pitch",      range (-24.0f, 24.0f, 1.0f), 0.0f, semiString, "st");
    addFloat (layout, P::fineTune,   "Fine Tune",  range (-100.0f, 100.0f, 1.0f), 0.0f, centString, "ct");
    addFloat (layout, P::gender,     "Gender",     range (-100.0f, 100.0f, 1.0f), 0.0f, bipolarString);
    addFloat (layout, P::age,        "Age",        range (-100.0f, 100.0f, 1.0f), 0.0f, bipolarString);
    addFloat (layout, P::tone,       "Tone",       range (-100.0f, 100.0f, 1.0f), 0.0f, bipolarString);
    addFloat (layout, P::style,      "Style",      range (-100.0f, 100.0f, 1.0f), 0.0f, bipolarString);
    addFloat (layout, P::voiceBlend, "Voice Blend", range (0.0f, 100.0f, 1.0f), 0.0f, pctString, "%");

    // Advanced
    addBool  (layout, P::pitchPreserve, "Pitch Preserve", true);
    addBool  (layout, P::formantShift,  "Formant Shift", true);
    addBool  (layout, P::expression,    "Expression", true);
    addBool  (layout, P::vibe,          "Vibe", true);
    addFloat (layout, P::intensity,     "Intensity", range (0.0f, 100.0f, 1.0f), 70.0f, pctString, "%");

    // Expression
    addFloat (layout, P::vibratoDepth, "Vibrato Depth", range (0.0f, 100.0f, 1.0f), 0.0f, centString, "ct");
    addFloat (layout, P::vibratoRate,  "Vibrato Rate",  range (3.0f, 9.0f, 0.1f), 5.5f, rateString, "Hz");
    addFloat (layout, P::breath,       "Breath",        range (0.0f, 100.0f, 1.0f), 0.0f, pctString, "%");
    addFloat (layout, P::rasp,         "Rasp",          range (0.0f, 100.0f, 1.0f), 0.0f, pctString, "%");

    // Pitch correction
    addBool   (layout, P::tuneOn,      "Pitch Correction", false);
    addChoice (layout, P::key,         "Key", Choices::key(), 0);
    addChoice (layout, P::scale,       "Scale", Choices::scale(), 0);
    addFloat  (layout, P::retuneSpeed, "Retune Speed", skewedRange (0.0f, 400.0f, 60.0f), 40.0f, msString, "ms");
    addFloat  (layout, P::humanize,    "Humanize", range (0.0f, 100.0f, 1.0f), 30.0f, pctString, "%");
    addBool   (layout, P::midiControl, "MIDI Pitch Control", false);

    // Harmony
    addBool   (layout, P::harmonyOn,     "Harmony", false);
    addChoice (layout, P::harm1Interval, "Harmony 1 Interval", Choices::harmonyInterval(), 7);   // +3rd
    addFloat  (layout, P::harm1Level,    "Harmony 1 Level", range (0.0f, 100.0f, 1.0f), 70.0f, pctString, "%");
    addFloat  (layout, P::harm1Pan,      "Harmony 1 Pan", range (-100.0f, 100.0f, 1.0f), -50.0f, panString);
    addChoice (layout, P::harm2Interval, "Harmony 2 Interval", Choices::harmonyInterval(), 9);   // +5th
    addFloat  (layout, P::harm2Level,    "Harmony 2 Level", range (0.0f, 100.0f, 1.0f), 60.0f, pctString, "%");
    addFloat  (layout, P::harm2Pan,      "Harmony 2 Pan", range (-100.0f, 100.0f, 1.0f), 50.0f, panString);
    addChoice (layout, P::harm3Interval, "Harmony 3 Interval", Choices::harmonyInterval(), 0);   // Off
    addFloat  (layout, P::harm3Level,    "Harmony 3 Level", range (0.0f, 100.0f, 1.0f), 50.0f, pctString, "%");
    addFloat  (layout, P::harm3Pan,      "Harmony 3 Pan", range (-100.0f, 100.0f, 1.0f), 0.0f, panString);

    // Doubler
    addBool  (layout, P::doublerOn,     "Doubler", false);
    addFloat (layout, P::doublerAmount, "Doubler Amount", range (0.0f, 100.0f, 1.0f), 50.0f, pctString, "%");
    addFloat (layout, P::doublerWidth,  "Doubler Width",  range (0.0f, 100.0f, 1.0f), 80.0f, pctString, "%");

    // FX
    addFloat  (layout, P::lowCut,        "Low Cut",     skewedRange (20.0f, 500.0f, 100.0f), 70.0f, hzString, "Hz");
    addFloat  (layout, P::deEsser,       "De-Esser",    range (0.0f, 100.0f, 1.0f), 25.0f, pctString, "%");
    addFloat  (layout, P::compressor,    "Compressor",  range (0.0f, 100.0f, 1.0f), 25.0f, pctString, "%");
    addFloat  (layout, P::presence,      "Presence",    range (-12.0f, 12.0f, 0.1f), 0.0f, dbString, "dB");
    addFloat  (layout, P::air,           "Air",         range (-12.0f, 12.0f, 0.1f), 0.0f, dbString, "dB");
    addFloat  (layout, P::saturation,    "Saturation",  range (0.0f, 100.0f, 1.0f), 0.0f, pctString, "%");
    addFloat  (layout, P::delayMix,      "Delay Mix",   range (0.0f, 100.0f, 1.0f), 0.0f, pctString, "%");
    addChoice (layout, P::delayTime,     "Delay Time",  Choices::delayTime(), 1);
    addFloat  (layout, P::delayFeedback, "Delay Feedback", range (0.0f, 90.0f, 1.0f), 30.0f, pctString, "%");
    addFloat  (layout, P::reverbMix,     "Reverb Mix",  range (0.0f, 100.0f, 1.0f), 12.0f, pctString, "%");
    addFloat  (layout, P::reverbSize,    "Reverb Size", range (0.0f, 100.0f, 1.0f), 50.0f, pctString, "%");

    // Text to Sing
    addFloat  (layout, P::singTempo,    "Sing Tempo", range (40.0f, 240.0f, 1.0f), 120.0f, bpmString, "BPM");
    addBool   (layout, P::singSyncHost, "Sync to Host Tempo", true);
    addChoice (layout, P::melodySource, "Melody Source", Choices::melodySource(), 0);

    return layout;
}

} // namespace voxa
