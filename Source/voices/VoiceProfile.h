#pragma once

#include <juce_core/juce_core.h>
#include <juce_graphics/juce_graphics.h>

namespace voxa
{

enum class VoiceGender { Male = 0, Female, Neutral };

/** Plain-old-data DSP description of a voice. Safe to copy on the audio thread.

    Formant units: semitones of spectral-envelope scaling relative to a neutral adult reference.
    An average adult male speaking voice sits around -1, an average female around +1. */
struct VoiceTraits
{
    VoiceGender gender = VoiceGender::Neutral;

    // Register (MIDI notes) - used for smart octave placement and by Text-to-Sing melodies.
    int rangeLowMidi  = 48;
    int rangeHighMidi = 72;

    // Timbre
    float formantSemis = 0.0f;   // spectral envelope shift
    float tiltDb       = 0.0f;   // + brighter / - darker (applied as a tilt around 1 kHz)
    float warmthDb     = 0.0f;   // low-mid body around 250 Hz
    float presenceDb   = 0.0f;   // 3-5 kHz
    float airDb        = 0.0f;   // 12 kHz shelf
    float nasalDb      = 0.0f;   // narrow 1.2-1.8 kHz resonance (twang)
    float ringHz       = 3000.0f;  // the voice's own upper resonance ("ring" / singer's formant region)
    float ringDb       = 0.0f;
    float bodyHz       = 250.0f;   // the voice's own lower resonance (chest / body)
    float bodyDb       = 0.0f;
    float breathiness  = 0.0f;   // 0..1 aspiration noise
    float rasp         = 0.0f;   // 0..1 grit / vocal fry

    // Expression
    float vibratoRateHz     = 5.5f;
    float vibratoDepthCents = 20.0f;
    float vibratoOnsetMs    = 250.0f;  // time a note must be held before vibrato fades in
    float scoopCents        = 0.0f;    // pitch approach from below at note onsets
    float scoopMs           = 80.0f;
    float jitterCents       = 3.0f;    // natural micro pitch variation

    // "Vibe" production chain suggested for the genre (all 0..1 unless noted)
    float vibeCompressor = 0.3f;
    float vibeDeEsser    = 0.3f;
    float vibeSaturation = 0.0f;
    float vibeReverbMix  = 0.12f;
    float vibeReverbSize = 0.5f;
    float vibeDelayMix   = 0.0f;
    int   vibeDelayDivision = 1;       // index into Choices::delayTime()
    float vibeDoubler    = 0.0f;
    float vibeLowCutHz   = 80.0f;

    // Text-to-Sing performance hints
    float singLegato      = 0.6f;      // 0 = detached, 1 = fully legato
    float singRhythmBusy  = 0.5f;      // 0 = long notes, 1 = dense rhythmic phrasing (rap)
    float singPortamentoMs = 60.0f;

    /** Linear interpolation between two trait sets (used for voice blending). */
    static VoiceTraits blend (const VoiceTraits& a, const VoiceTraits& b, float t) noexcept;

    int centreMidi() const noexcept { return (rangeLowMidi + rangeHighMidi) / 2; }
};

/** A voice in the library: DSP traits plus presentation metadata and an optional neural model. */
struct VoiceProfile
{
    juce::String id;            // stable identifier, e.g. "female-pop"
    juce::String name;          // "Female Pop"
    juce::String tagline;       // short description
    juce::String genre;         // "Pop"
    juce::StringArray styles;   // descriptive tags, e.g. "Breathy", "Powerful"
    VoiceTraits traits;

    juce::Colour accentA { 0xff3a7bff };
    juce::Colour accentB { 0xff9ad0ff };

    bool isFactory = true;

    // User voice packs (see docs/NEURAL_ENGINE.md)
    juce::File packDirectory;
    juce::File avatarImage;
    juce::File neuralModelFile;         // RVC-compatible ONNX generator
    juce::String neuralEncoder;         // content encoder id, e.g. "contentvec-768"
    int neuralSampleRate = 0;           // model output sample rate
    juce::String author, license;

    // AI models for built-in voices: a pack may list factory voice ids ("factoryVoices" in voice.json);
    // each listed built-in voice then uses that pack's model (packDirectory/neural* are copied over).
    juce::StringArray factoryVoices;    // on a pack: the built-in voices it gives an AI model
    juce::String aiPackName;            // on a built-in voice: the pack its AI model comes from
    bool listed = true;                 // false: the pack only powers built-in voices and is not shown itself

    VoiceGender getGender() const noexcept { return traits.gender; }
    bool hasNeuralModel() const          { return neuralModelFile.existsAsFile(); }
    juce::String getGenderName() const;
};

} // namespace voxa
