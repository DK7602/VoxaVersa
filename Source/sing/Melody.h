#pragma once

#include "Lyrics.h"
#include "SingEngine.h"

namespace voxa::sing
{

/** A melody note bound to the lyric syllable it carries. Times are in quarter-note beats. */
struct ScoreNote
{
    double startBeat = 0.0;
    double lengthBeats = 1.0;
    int midiNote = 60;
    float velocity = 0.8f;
    int syllable = -1;            // index into LyricScript::syllables
    bool continuation = false;    // melisma: the syllable continues from the previous note
    bool breathBefore = false;    // a phrase starts here: breathe before it

    double endBeat() const noexcept   { return startBeat + lengthBeats; }
};

struct Score
{
    std::vector<ScoreNote> notes;  // sorted by start, non-overlapping
    int keyRoot = 0;
    ScaleType scale = ScaleType::Major;
};

namespace melody
{
    /** A key whose tonic sits comfortably in the voice's register. */
    int chooseKeyRoot (const VoiceTraits&);

    /** Composes a melody for the lyrics (deterministic for a given seed). keyRoot -1 = automatic. */
    Score compose (const LyricScript&, const VoiceTraits&, int keyRoot, ScaleType, juce::uint32 seed);

    /** Binds a user melody to the lyrics: syllables in order, "_" holds, the melody repeats when the
        lyrics are longer and spare notes extend the last syllable. Octave-shifted to suit the voice. */
    Score fitProvided (const LyricScript&, const std::vector<Note>& melody, const VoiceTraits&,
                       int keyRoot, ScaleType, double bpm);

    /** Sorts, removes chords (keeps the top note) and overlaps (last-note priority). */
    std::vector<Note> makeMonophonic (std::vector<Note>);

    /** Octave shift (multiple of 12) that best fits the melody into the voice's range. */
    int bestOctaveShift (const std::vector<Note>&, const VoiceTraits&);

    /** Key estimate (Krumhansl-Schmuckler) of a melody, weighted by note length. */
    void detectKey (const std::vector<Note>&, int& keyRoot, ScaleType& scale);
}

} // namespace voxa::sing
