#include "sing/Melody.h"
#include "sing/SingEngine.h"
#include "voices/VoiceLibrary.h"

#include <cmath>
#include <map>

namespace voxa::sing
{

/** Melody composition, provided-melody fitting, MIDI import/export and hum-to-melody. */
class SingMelodyTests : public juce::UnitTest
{
public:
    SingMelodyTests() : juce::UnitTest ("Text to Sing: melody", "Sing") {}

    void runTest() override
    {
        const auto voices = createFactoryVoices();
        const auto lyrics = parseLyrics ("Hello world, I love you so\n"
                                         "The sun is shining bright today\n"
                                         "Beau-ti-ful _ night, stay with me\n"
                                         "Don't let go of my heart");

        beginTest ("Auto melody stays in key and inside every voice's register");
        {
            const ScaleType scales[] = { ScaleType::Major, ScaleType::NaturalMinor, ScaleType::MajorPentatonic,
                                         ScaleType::Blues, ScaleType::Dorian, ScaleType::HarmonicMinor };
            int checked = 0;
            for (const auto& voice : voices)
            {
                for (auto scale : scales)
                {
                    for (int key : { -1, 0, 7 })
                    {
                        const auto score = melody::compose (lyrics, voice.traits, key, scale, 42);
                        expect (! score.notes.empty());
                        if (key >= 0)
                            expectEquals (score.keyRoot, key);

                        for (const auto& n : score.notes)
                        {
                            if (! music::isInScale (n.midiNote, score.keyRoot, scale))
                                expect (false, voice.id + " note out of key: " + music::noteName (n.midiNote));
                            if (n.midiNote < voice.traits.rangeLowMidi || n.midiNote > voice.traits.rangeHighMidi)
                                expect (false, voice.id + " note out of range: " + music::noteName (n.midiNote));
                        }
                        expectWellFormed (score, lyrics);
                        ++checked;
                    }
                }
            }
            logMessage ("checked " + juce::String (checked) + " melodies");
        }

        beginTest ("Auto melody is deterministic per seed and varies between seeds");
        {
            const auto& traits = voices.front().traits;
            const auto a = melody::compose (lyrics, traits, 0, ScaleType::Major, 7);
            const auto b = melody::compose (lyrics, traits, 0, ScaleType::Major, 7);
            const auto c = melody::compose (lyrics, traits, 0, ScaleType::Major, 8);
            expect (sameNotes (a, b));
            expect (! sameNotes (a, c));
        }

        beginTest ("Phrases end on stable degrees with long notes; the song ends on the tonic");
        {
            for (juce::uint32 seed = 1; seed <= 10; ++seed)
            {
                const auto score = melody::compose (lyrics, voices.front().traits, 2, ScaleType::Major, seed);
                std::vector<double> lengths;
                for (const auto& n : score.notes)
                    lengths.push_back (n.lengthBeats);
                std::sort (lengths.begin(), lengths.end());
                const double median = lengths[lengths.size() / 2];

                for (size_t i = 0; i < score.notes.size(); ++i)
                {
                    const bool phraseEnd = i + 1 == score.notes.size() || score.notes[i + 1].breathBefore;
                    if (! phraseEnd)
                        continue;
                    const auto& n = score.notes[i];
                    const int degree = ((n.midiNote - score.keyRoot) % 12 + 12) % 12;
                    expect (degree == 0 || degree == 4 || degree == 7, "phrase must end on 1/3/5");
                    expect (n.lengthBeats >= median, "phrase-final notes are long");
                }
                expectEquals (((score.notes.back().midiNote - score.keyRoot) % 12 + 12) % 12, 0);
            }
        }

        beginTest ("Melodic motion is mostly stepwise with occasional leaps");
        {
            int steps = 0, leaps = 0, bigLeaps = 0;
            for (juce::uint32 seed = 1; seed <= 20; ++seed)
            {
                const auto score = melody::compose (lyrics, voices.front().traits, 0, ScaleType::Major, seed);
                for (size_t i = 1; i < score.notes.size(); ++i)
                {
                    if (score.notes[i].breathBefore)
                        continue;
                    const int interval = std::abs (score.notes[i].midiNote - score.notes[i - 1].midiNote);
                    if (interval <= 2) ++steps;
                    else ++leaps;
                    if (interval > 9) ++bigLeaps;
                }
            }
            logMessage ("steps " + juce::String (steps) + ", leaps " + juce::String (leaps) + ", leaps > 9: " + juce::String (bigLeaps));
            expect (steps > leaps);
            expect (leaps > 0);
            expect (bigLeaps * 20 < steps + leaps);
        }

        beginTest ("Rhythm density follows singRhythmBusy");
        {
            auto rap = voices.front().traits;
            rap.singRhythmBusy = 0.95f;
            auto ballad = rap;
            ballad.singRhythmBusy = 0.25f;

            const auto fast = melody::compose (lyrics, rap, 0, ScaleType::Major, 3);
            const auto slow = melody::compose (lyrics, ballad, 0, ScaleType::Major, 3);
            const double fastSpan = fast.notes.back().endBeat() - fast.notes.front().startBeat;
            const double slowSpan = slow.notes.back().endBeat() - slow.notes.front().startBeat;
            logMessage ("song length: busy " + juce::String (fastSpan) + " beats, ballad " + juce::String (slowSpan) + " beats");
            expect (fastSpan < slowSpan * 0.75);
            expect (repeatedPitchFraction (fast) > repeatedPitchFraction (slow));
        }

        beginTest ("Stressed syllables land on strong beats");
        {
            const auto words = parseLyrics ("Remember the morning, forget the hollow evening\n"
                                            "Believe in tomorrow, surrender the yesterday\n"
                                            "Amazing and beautiful, forever and ever");
            int stressedOnBeat = 0, stressed = 0, weakOnBeat = 0, weak = 0;
            for (juce::uint32 seed = 1; seed <= 10; ++seed)
            {
                const auto score = melody::compose (words, voices.front().traits, 0, ScaleType::Major, seed);
                for (const auto& n : score.notes)
                {
                    if (n.continuation)
                        continue;
                    const bool onBeat = std::abs (n.startBeat - std::round (n.startBeat)) < 1.0e-6;
                    if (words.syllables[(size_t) n.syllable].stress == 1) { ++stressed; stressedOnBeat += onBeat ? 1 : 0; }
                    else                                                  { ++weak; weakOnBeat += onBeat ? 1 : 0; }
                }
            }
            const double s = (double) stressedOnBeat / stressed, w = (double) weakOnBeat / weak;
            logMessage ("on-beat: stressed " + juce::String (s, 2) + ", unstressed " + juce::String (w, 2));
            expect (s > 0.8);
            expect (s > w + 0.2);
        }

        beginTest ("Automatic key suits the voice");
        {
            for (const auto& voice : voices)
            {
                const int key = melody::chooseKeyRoot (voice.traits);
                expect (key >= 0 && key < 12);
                const auto score = melody::compose (lyrics, voice.traits, -1, ScaleType::Major, 1);
                expectEquals (score.keyRoot, key);
            }
        }

        beginTest ("Provided melody: order, melisma, repetition and spare notes");
        {
            VoiceTraits traits;
            traits.rangeLowMidi = 64;
            traits.rangeHighMidi = 86;
            const std::vector<Note> melody { { 0.0, 1.0, 60, 0.8f }, { 1.0, 1.0, 62, 0.8f }, { 2.0, 1.0, 64, 0.8f }, { 3.0, 1.0, 65, 0.8f } };

            const auto plain = melody::fitProvided (parseLyrics ("one two three four"), melody, traits, 0, ScaleType::Major, 120.0);
            expectEquals ((int) plain.notes.size(), 4);
            for (int i = 0; i < 4; ++i)
            {
                expectEquals (plain.notes[(size_t) i].syllable, i);
                expectEquals (plain.notes[(size_t) i].midiNote, melody[(size_t) i].midiNote + 12);   // octave fit
            }

            const auto held = melody::fitProvided (parseLyrics ("one two _ three"), melody, traits, 0, ScaleType::Major, 120.0);
            expectEquals ((int) held.notes.size(), 4);
            expect (held.notes[2].continuation && held.notes[2].syllable == 1);
            expectEquals (held.notes[3].syllable, 2);

            const auto longer = melody::fitProvided (parseLyrics ("one two three four five six"), melody, traits, 0, ScaleType::Major, 120.0);
            expect (longer.notes.size() >= 6 && longer.notes.size() <= 8);
            expectWithinAbsoluteError (longer.notes[4].startBeat, 4.0, 1.0e-9);      // the pattern repeats a bar later
            expectEquals (longer.notes[5].midiNote, longer.notes[1].midiNote);
            for (size_t i = 6; i < longer.notes.size(); ++i)                       // spare notes finish the figure
                expect (longer.notes[i].continuation && longer.notes[i].syllable == 5);

            const auto shorter = melody::fitProvided (parseLyrics ("one two"), melody, traits, 0, ScaleType::Major, 120.0);
            expectEquals ((int) shorter.notes.size(), 4);
            expect (shorter.notes[2].continuation && shorter.notes[3].continuation && shorter.notes[3].syllable == 1);

            const auto phrased = melody::fitProvided (parseLyrics ("one two\nthree four"), melody, traits, 0, ScaleType::Major, 120.0);
            expect (phrased.notes[2].breathBefore);
            expect (phrased.notes[1].endBeat() < phrased.notes[2].startBeat - 0.1);   // room to breathe

            VoiceTraits low;
            low.rangeLowMidi = 40;
            low.rangeHighMidi = 60;
            const auto bass = melody::fitProvided (parseLyrics ("one two three four"), melody, low, -1, ScaleType::Major, 120.0);
            for (const auto& n : bass.notes)
            {
                expect (n.midiNote >= 40 && n.midiNote <= 60);
                expect (music::isInScale (n.midiNote, bass.keyRoot, bass.scale));   // detected key fits
            }
        }

        beginTest ("melodyFromSequence is monophonic (top note of chords, last-note priority)");
        {
            juce::MidiMessageSequence seq;
            auto add = [&seq] (int note, double on, double off)
            {
                seq.addEvent (juce::MidiMessage::noteOn (1, note, (juce::uint8) 100), on);
                seq.addEvent (juce::MidiMessage::noteOff (1, note), off);
            };
            add (60, 0.0, 1.0);
            add (64, 0.0, 1.0);      // chord with 60: keep 64
            add (67, 0.5, 2.0);      // overlaps 64: 64 is cut at 0.5
            add (65, 3.0, 3.5);
            seq.updateMatchedPairs();

            const auto notes = SingEngine::melodyFromSequence (seq);
            expectEquals ((int) notes.size(), 3);
            expectEquals (notes[0].midiNote, 64);
            expectWithinAbsoluteError (notes[0].lengthBeats, 0.5, 1.0e-9);
            expectEquals (notes[1].midiNote, 67);
            expectWithinAbsoluteError (notes[1].lengthBeats, 1.5, 1.0e-9);
            expectEquals (notes[2].midiNote, 65);
            expectWithinAbsoluteError (notes[2].startBeat, 3.0, 1.0e-9);
        }

        beginTest ("MIDI file round trip with lyric events");
        {
            SingResult take;
            const double bpm = 96.0;
            const double spb = 60.0 / bpm;
            const char* syllables[] = { "Hel", "lo", "world", "_" };
            const int pitches[] = { 62, 64, 67, 65 };
            const double beats[][2] = { { 1.0, 0.5 }, { 1.5, 1.0 }, { 2.5, 1.5 }, { 4.0, 1.0 } };
            for (int i = 0; i < 4; ++i)
            {
                SungNote n;
                n.startSeconds = beats[i][0] * spb;
                n.endSeconds = (beats[i][0] + beats[i][1]) * spb;
                n.midiNote = pitches[i];
                n.syllable = syllables[i];
                n.wordIndex = i < 2 ? 0 : 1;
                take.notes.push_back (n);
            }

            const auto midi = SingEngine::toMidiFile (take, bpm);
            expectEquals (midi.getNumTracks(), 1);

            juce::StringArray lyricEvents;
            const auto* track = midi.getTrack (0);
            for (int i = 0; i < track->getNumEvents(); ++i)
            {
                const auto& m = track->getEventPointer (i)->message;
                if (m.isTextMetaEvent() && m.getMetaEventType() == 0x05)
                    lyricEvents.add (m.getTextFromTextMetaEvent());
            }
            expectEquals (lyricEvents.joinIntoString ("|"), juce::String ("Hel-|lo|world"));

            const auto file = juce::File::createTempFile (".mid");
            {
                juce::FileOutputStream out (file);
                expect (out.openedOk());
                expect (midi.writeTo (out));
            }

            double bpmOut = 0.0;
            juce::String error;
            const auto notes = SingEngine::melodyFromMidiFile (file, bpmOut, error);
            file.deleteFile();

            expect (error.isEmpty(), error);
            expectWithinAbsoluteError (bpmOut, bpm, 0.01);
            expectEquals ((int) notes.size(), 4);
            for (int i = 0; i < juce::jmin (4, (int) notes.size()); ++i)
            {
                expectEquals (notes[(size_t) i].midiNote, pitches[i]);
                expectWithinAbsoluteError (notes[(size_t) i].startBeat, beats[i][0], 1.0e-3);
                expectWithinAbsoluteError (notes[(size_t) i].lengthBeats, beats[i][1], 1.0e-3);
            }

            double unusedBpm = 0.0;
            juce::String missingError;
            expect (SingEngine::melodyFromMidiFile (juce::File ("/nonexistent/x.mid"), unusedBpm, missingError).empty());
            expect (missingError.isNotEmpty());
        }

        beginTest ("Hum-to-melody recovers a sine melody");
        {
            const double sr = 44100.0, bpm = 120.0;
            const std::vector<Note> truth { { 1.0, 1.0, 60, 0.8f }, { 2.0, 1.0, 64, 0.8f }, { 3.0, 2.0, 67, 0.8f },
                                            { 5.0, 0.5, 65, 0.8f }, { 5.5, 0.5, 64, 0.8f }, { 6.0, 1.0, 62, 0.8f },
                                            { 7.0, 1.0, 62, 0.8f } };
            const auto audio = hum (truth, sr, bpm, 2.5);
            const auto notes = SingEngine::melodyFromAudio (audio, sr, bpm);
            expectMatches (notes, truth);
        }

        beginTest ("Hum-to-melody recovers the melody of a sung take");
        {
            SingEngine engine;
            SingRequest request;
            request.lyrics = "la la la la la";
            request.melodyMode = MelodyMode::Provided;
            request.melody = { { 1.0, 1.0, 57, 0.8f }, { 2.0, 1.0, 60, 0.8f }, { 3.0, 1.0, 62, 0.8f },
                               { 4.0, 1.0, 64, 0.8f }, { 5.0, 2.0, 60, 0.8f } };
            request.voice = voices.front();     // male pop: 48..69, the melody fits as given
            request.bpm = 100.0;
            request.sampleRate = 44100.0;
            const auto take = engine.render (request);
            expect (take.ok(), take.errorMessage);
            const auto notes = SingEngine::melodyFromAudio (take.audio, take.sampleRate, request.bpm);
            expectMatches (notes, request.melody);
        }
    }

private:
    static bool sameNotes (const Score& a, const Score& b)
    {
        if (a.notes.size() != b.notes.size())
            return false;
        for (size_t i = 0; i < a.notes.size(); ++i)
            if (a.notes[i].midiNote != b.notes[i].midiNote || std::abs (a.notes[i].startBeat - b.notes[i].startBeat) > 1.0e-9)
                return false;
        return true;
    }

    static double repeatedPitchFraction (const Score& s)
    {
        int repeats = 0;
        for (size_t i = 1; i < s.notes.size(); ++i)
            repeats += s.notes[i].midiNote == s.notes[i - 1].midiNote ? 1 : 0;
        return (double) repeats / (double) juce::jmax ((size_t) 1, s.notes.size() - 1);
    }

    void expectWellFormed (const Score& score, const LyricScript& script)
    {
        int nextSyllable = 0;
        for (size_t i = 0; i < score.notes.size(); ++i)
        {
            const auto& n = score.notes[i];
            expect (n.lengthBeats > 0.0);
            if (i > 0)
                expect (n.startBeat >= score.notes[i - 1].endBeat() - 1.0e-9, "notes must not overlap");
            if (! n.continuation)
                expectEquals (n.syllable, nextSyllable++);
        }
        expectEquals (nextSyllable, script.getNumSyllables());
    }

    /** A hummed rendition: sine with vibrato, soft attacks and small gaps between notes. */
    static juce::AudioBuffer<float> hum (const std::vector<Note>& notes, double sr, double bpm, double leadSeconds)
    {
        const double spb = 60.0 / bpm;
        const double total = leadSeconds + (notes.back().startBeat + notes.back().lengthBeats) * spb + 0.5;
        juce::AudioBuffer<float> audio (1, (int) (total * sr));
        audio.clear();
        auto* d = audio.getWritePointer (0);
        double phase = 0.0;
        for (const auto& n : notes)
        {
            const int a = (int) ((leadSeconds + n.startBeat * spb) * sr);
            const int b = (int) ((leadSeconds + (n.startBeat + n.lengthBeats) * spb - 0.04) * sr);
            const double hz = 440.0 * std::pow (2.0, (n.midiNote - 69) / 12.0);
            for (int i = a; i < b; ++i)
            {
                const double t = (i - a) / sr;
                const double vib = 1.0 + 0.004 * std::sin (2.0 * juce::MathConstants<double>::pi * 5.5 * t);
                phase += 2.0 * juce::MathConstants<double>::pi * hz * vib / sr;
                const double env = juce::jmin (1.0, t / 0.02, (b - i) / sr / 0.02);
                d[i] = (float) (0.3 * env * (std::sin (phase) + 0.3 * std::sin (2.0 * phase)));
            }
        }
        return audio;
    }

    /** Same pitches, and the same rhythm up to a whole-beat shift of the start. */
    void expectMatches (const std::vector<Note>& found, const std::vector<Note>& truth)
    {
        expectEquals ((int) found.size(), (int) truth.size());
        if (found.empty() || found.size() != truth.size())
        {
            for (const auto& n : found)
                logMessage ("  got " + music::noteName (n.midiNote) + " @" + juce::String (n.startBeat) + " /" + juce::String (n.lengthBeats));
            return;
        }

        const double offset = found.front().startBeat - truth.front().startBeat;
        expectWithinAbsoluteError (offset - std::round (offset), 0.0, 0.26);
        for (size_t i = 0; i < truth.size(); ++i)
        {
            expectEquals (found[i].midiNote, truth[i].midiNote);
            expectWithinAbsoluteError (found[i].startBeat - offset, truth[i].startBeat, 0.26);
            expectWithinAbsoluteError (found[i].lengthBeats, truth[i].lengthBeats, 0.26);
        }
    }
};

static SingMelodyTests singMelodyTests;

} // namespace voxa::sing
