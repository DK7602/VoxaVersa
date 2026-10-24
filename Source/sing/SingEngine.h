#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "../core/MusicTheory.h"
#include "../voices/VoiceProfile.h"

namespace voxa::sing
{

/** A melody note expressed in quarter-note beats. */
struct Note
{
    double startBeat = 0.0;
    double lengthBeats = 1.0;
    int midiNote = 60;
    float velocity = 0.8f;
};

/** A rendered, lyric-aligned note (for UI display and MIDI export). */
struct SungNote
{
    double startSeconds = 0.0;
    double endSeconds = 0.0;
    int midiNote = 60;
    juce::String syllable;
    int wordIndex = 0;
};

enum class MelodyMode
{
    Provided,       // use SingRequest::melody (from MIDI input capture, MIDI file or hum extraction)
    AutoGenerate    // compose a melody for the lyrics in the requested key/scale/tempo
};

struct SingRequest
{
    juce::String lyrics;
    int keyRoot = -1;                  // 0 = C .. 11 = B, -1 = choose automatically
    ScaleType scale = ScaleType::Major;
    double bpm = 120.0;
    MelodyMode melodyMode = MelodyMode::AutoGenerate;
    std::vector<Note> melody;          // used when melodyMode == Provided
    VoiceProfile voice;                // timbre, register and performance style
    juce::uint32 seed = 1;             // auto-melody variation
    double sampleRate = 48000.0;
};

struct SingResult
{
    juce::AudioBuffer<float> audio;    // mono guide vocal at request.sampleRate
    double sampleRate = 48000.0;
    std::vector<SungNote> notes;
    std::vector<float> f0Hz;           // pitch contour, 0 = unvoiced
    double f0FrameRate = 100.0;        // frames per second of f0Hz
    int keyRoot = 0;                   // key actually used
    ScaleType scale = ScaleType::Major;
    juce::String errorMessage;

    bool ok() const noexcept { return errorMessage.isEmpty() && audio.getNumSamples() > 0; }
};

/** Offline text-to-sing renderer: lyrics -> phonemes -> notes -> sung guide vocal. Thread-safe per instance. */
class SingEngine
{
public:
    SingEngine();
    ~SingEngine();

    /** Renders the whole request. Call from a background thread. `cancel` may be null. */
    SingResult render (const SingRequest& request,
                       const std::atomic<bool>* cancel = nullptr,
                       std::function<void (float progress01)> progress = {});

    /** Number of sung syllables in the lyrics (after parsing hyphens, melisma marks, etc). */
    static int countSyllables (const juce::String& lyrics);

    /** Monophonic melody from the first track with notes in a standard MIDI file. */
    static std::vector<Note> melodyFromMidiFile (const juce::File& file, double& bpmOut, juce::String& errorMessage);

    /** Monophonic melody from a sequence whose timestamps are in quarter-note beats. */
    static std::vector<Note> melodyFromSequence (const juce::MidiMessageSequence& sequenceInBeats);

    /** Hum-to-melody: pitch-tracks a monophonic recording and quantises it to notes. */
    static std::vector<Note> melodyFromAudio (const juce::AudioBuffer<float>& mono, double sampleRate, double bpm);

    /** The rendered melody as a MIDI file (one track, lyric meta events on each note). */
    static juce::MidiFile toMidiFile (const SingResult& result, double bpm);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

/** Real-time, MIDI-driven singer: every note-on sings the next syllable of the lyrics.
    All methods except setLyrics() are real-time safe and must be called from the audio thread. */
class LiveSinger
{
public:
    LiveSinger();
    ~LiveSinger();

    void prepare (double sampleRate, int maxBlockSize);
    void reset();

    /** Message thread. Parses the lyrics off the audio thread and swaps them in lock-free. */
    void setLyrics (const juce::String& lyrics);

    /** Audio thread. Voice timbre/register used for upcoming notes. */
    void setVoice (const VoiceTraits& traits);

    void noteOn (int midiNote, float velocity);
    void noteOff (int midiNote);
    void allNotesOff();

    /** Jump back to the first syllable (called when the host transport starts). */
    void restartLyrics();

    /** Renders (replaces) numSamples of mono output. */
    void render (float* output, int numSamples);

    int getCurrentSyllableIndex() const noexcept;
    int getNumSyllables() const noexcept;
    bool isSounding() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace voxa::sing
