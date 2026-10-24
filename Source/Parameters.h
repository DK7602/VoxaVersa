#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

namespace voxa
{

/** Every automatable parameter of VoxaVersa. IDs are part of the saved-state format: never rename. */
namespace ParamID
{
    // Global / IO
    inline constexpr const char* power        = "power";
    inline constexpr const char* mode         = "mode";          // Voice Transform | Text to Sing
    inline constexpr const char* inputGain    = "inputGain";     // dB
    inline constexpr const char* outputGain   = "outputGain";    // dB
    inline constexpr const char* mix          = "mix";           // %
    inline constexpr const char* autoLevel    = "autoLevel";
    inline constexpr const char* quality      = "quality";       // Low Latency | Balanced | High Quality
    inline constexpr const char* aiEngine     = "aiEngine";      // use neural voice model when available
    inline constexpr const char* sourceVoice  = "sourceVoice";   // Auto | Male | Female (your input voice)

    // Voice character
    inline constexpr const char* pitch        = "pitch";         // semitones, higher/deeper
    inline constexpr const char* fineTune     = "fineTune";      // cents
    inline constexpr const char* gender       = "gender";        // -100 Male .. +100 Female
    inline constexpr const char* age          = "age";           // -100 Young .. +100 Mature
    inline constexpr const char* tone         = "tone";          // -100 Bright .. +100 Dark
    inline constexpr const char* style        = "style";         // -100 Natural .. +100 Creative
    inline constexpr const char* voiceBlend   = "voiceBlend";    // % morph toward the blend voice

    // Advanced
    inline constexpr const char* pitchPreserve = "pitchPreserve";
    inline constexpr const char* formantShift  = "formantShift";
    inline constexpr const char* expression    = "expression";
    inline constexpr const char* vibe          = "vibe";
    inline constexpr const char* intensity     = "intensity";    // %

    // Expression
    inline constexpr const char* vibratoDepth = "vibratoDepth";  // cents
    inline constexpr const char* vibratoRate  = "vibratoRate";   // Hz
    inline constexpr const char* breath       = "breath";        // %
    inline constexpr const char* rasp         = "rasp";          // %

    // Pitch correction
    inline constexpr const char* tuneOn       = "tuneOn";
    inline constexpr const char* key          = "key";           // Auto | C .. B
    inline constexpr const char* scale        = "scale";
    inline constexpr const char* retuneSpeed  = "retuneSpeed";   // ms
    inline constexpr const char* humanize     = "humanize";      // %
    inline constexpr const char* midiControl  = "midiControl";   // MIDI notes set the target pitch

    // Harmony
    inline constexpr const char* harmonyOn    = "harmonyOn";
    inline constexpr const char* harm1Interval = "harm1Interval";
    inline constexpr const char* harm1Level    = "harm1Level";
    inline constexpr const char* harm1Pan      = "harm1Pan";
    inline constexpr const char* harm2Interval = "harm2Interval";
    inline constexpr const char* harm2Level    = "harm2Level";
    inline constexpr const char* harm2Pan      = "harm2Pan";
    inline constexpr const char* harm3Interval = "harm3Interval";
    inline constexpr const char* harm3Level    = "harm3Level";
    inline constexpr const char* harm3Pan      = "harm3Pan";

    // Doubler
    inline constexpr const char* doublerOn     = "doublerOn";
    inline constexpr const char* doublerAmount = "doublerAmount"; // %
    inline constexpr const char* doublerWidth  = "doublerWidth";  // %

    // FX
    inline constexpr const char* lowCut        = "lowCut";        // Hz
    inline constexpr const char* deEsser       = "deEsser";       // %
    inline constexpr const char* compressor    = "compressor";    // %
    inline constexpr const char* presence      = "presence";      // dB
    inline constexpr const char* air           = "air";           // dB
    inline constexpr const char* saturation    = "saturation";    // %
    inline constexpr const char* delayMix      = "delayMix";      // %
    inline constexpr const char* delayTime     = "delayTime";     // note division
    inline constexpr const char* delayFeedback = "delayFeedback"; // %
    inline constexpr const char* reverbMix     = "reverbMix";     // %
    inline constexpr const char* reverbSize    = "reverbSize";    // %

    // Text to Sing
    inline constexpr const char* singTempo     = "singTempo";     // BPM
    inline constexpr const char* singSyncHost  = "singSyncHost";  // follow host tempo
    inline constexpr const char* melodySource  = "melodySource";  // Use MIDI | Auto Melody | MIDI File | Hum Melody
}

/** Non-parameter properties stored on the APVTS state tree (saved with sessions & presets). */
namespace StateID
{
    inline constexpr const char* voiceId      = "voiceId";
    inline constexpr const char* blendVoiceId = "blendVoiceId";
    inline constexpr const char* lyrics       = "lyrics";
    inline constexpr const char* presetName   = "presetName";
    inline constexpr const char* melodyFile   = "melodyFile";
    inline constexpr const char* samplePath   = "samplePath";
}

namespace Choices
{
    const juce::StringArray& mode();          // "Voice Transform", "Text to Sing"
    const juce::StringArray& quality();       // "Low Latency", "Balanced", "High Quality"
    const juce::StringArray& sourceVoice();   // "Auto", "Male", "Female"
    const juce::StringArray& key();           // "Auto", "C", ... "B"
    const juce::StringArray& scale();         // see music::getScaleNames()
    const juce::StringArray& harmonyInterval();
    const juce::StringArray& delayTime();     // "1/4", "1/8", "1/8 Dotted", "1/16", "1/4 Dotted", "1/2"
    const juce::StringArray& melodySource();  // "Use MIDI", "Auto Melody", "MIDI File", "Hum Melody"
}

enum class Mode { VoiceTransform = 0, TextToSing };
enum class Quality { LowLatency = 0, Balanced, HighQuality };
enum class SourceVoice { Auto = 0, Male, Female };
enum class MelodySource { UseMidi = 0, AutoMelody, MidiFile, HumMelody };

/** Delay time in quarter-note beats for a delayTime choice index. */
double delayDivisionInBeats (int choiceIndex) noexcept;

juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

} // namespace voxa
