#include "sing/PitchTracker.h"
#include "sing/SingEngine.h"
#include "voices/VoiceLibrary.h"

#include <thread>

namespace voxa::sing
{

/** Offline rendering, pitch accuracy, f0 contour, cancellation and the real-time LiveSinger. */
class SingRenderTests : public juce::UnitTest
{
public:
    SingRenderTests() : juce::UnitTest ("Text to Sing: synthesis", "Sing") {}

    void runTest() override
    {
        voices = createFactoryVoices();

        beginTest ("Render: duration, level, sanity of the audio");
        {
            auto request = makeRequest ("female-pop", "la la la la", 120.0,
                                        { { 1.0, 1.0, 64, 0.8f }, { 2.0, 1.0, 67, 0.8f }, { 3.0, 1.0, 69, 0.8f }, { 4.0, 2.0, 67, 0.8f } });
            SingEngine engine;
            const auto result = engine.render (request);
            expect (result.ok(), result.errorMessage);

            const double seconds = result.audio.getNumSamples() / result.sampleRate;
            const double lastNoteEnd = 6.0 * 0.5;
            expect (seconds >= lastNoteEnd && seconds <= lastNoteEnd + 1.0, "duration " + juce::String (seconds));
            expectEquals (result.audio.getNumChannels(), 1);
            expect (allFinite (result.audio));

            const float peakDb = juce::Decibels::gainToDecibels (result.audio.getMagnitude (0, 0, result.audio.getNumSamples()));
            expectWithinAbsoluteError (peakDb, -12.0f, 0.5f);
            expect (result.audio.getRMSLevel (0, 0, result.audio.getNumSamples()) > 0.01f);

            // Silent before the first phrase, quiet again at the end.
            expect (rmsBetween (result, 0.0, 0.1) < 1.0e-4f);
            expect (rmsBetween (result, seconds - 0.05, seconds) < 1.0e-3f);

            expectEquals ((int) result.notes.size(), 4);
            expectEquals (result.notes[0].syllable, juce::String ("la"));
            expectWithinAbsoluteError (result.notes[0].startSeconds, 0.5, 1.0e-9);
            expectWithinAbsoluteError (result.notes[3].endSeconds, 3.0, 1.0e-9);
            expectEquals (result.notes[2].midiNote, 69);
        }

        beginTest ("Sung pitch matches the notes within 25 cents (YIN on the output)");
        {
            struct Case { const char* voice; std::vector<Note> melody; };
            const Case cases[] = {
                { "male-pop",       { { 1.0, 2.0, 55, 0.8f }, { 3.0, 2.0, 60, 0.8f }, { 5.0, 2.0, 64, 0.8f }, { 7.0, 2.0, 59, 0.8f } } },
                { "female-pop",     { { 1.0, 2.0, 64, 0.8f }, { 3.0, 2.0, 69, 0.8f }, { 5.0, 2.0, 72, 0.8f }, { 7.0, 2.0, 67, 0.8f } } },
                { "female-opera",   { { 1.0, 2.0, 72, 0.8f }, { 3.0, 2.0, 76, 0.8f }, { 5.0, 2.0, 79, 0.8f }, { 7.0, 2.0, 81, 0.8f } } },
                { "male-deep-bass", { { 1.0, 2.0, 38, 0.8f }, { 3.0, 2.0, 43, 0.8f }, { 5.0, 2.0, 45, 0.8f }, { 7.0, 2.0, 50, 0.8f } } },
                { "male-rnb",       { { 1.0, 2.0, 57, 0.8f }, { 3.0, 2.0, 62, 0.8f }, { 5.0, 2.0, 66, 0.8f }, { 7.0, 2.0, 64, 0.8f } } },
            };

            for (const auto& c : cases)
            {
                auto request = makeRequest (c.voice, "ah oh ee ay", 100.0, c.melody);
                SingEngine engine;
                const auto result = engine.render (request);
                expect (result.ok(), result.errorMessage);

                const auto track = trackPitch (result.audio.getReadPointer (0), result.audio.getNumSamples(), result.sampleRate, 0.01);
                for (const auto& note : result.notes)
                {
                    const float measured = medianMidi (track, 0.01, note.startSeconds, note.endSeconds);
                    const float cents = 100.0f * (measured - (float) note.midiNote);
                    expect (std::abs (cents) <= 25.0f, juce::String (c.voice) + " " + music::noteName (note.midiNote)
                                                       + ": measured " + juce::String (cents, 1) + " cents");
                }
            }
        }

        beginTest ("f0 contour matches the notes and is 0 when unvoiced");
        {
            auto request = makeRequest ("female-pop", "hel-lo my friend", 110.0,
                                        { { 2.0, 1.0, 64, 0.8f }, { 3.0, 1.0, 66, 0.8f }, { 4.0, 1.0, 68, 0.8f }, { 5.0, 3.0, 64, 0.8f } });
            SingEngine engine;
            const auto result = engine.render (request);
            expect (result.ok(), result.errorMessage);
            expectEquals (result.f0FrameRate, 100.0);

            const double seconds = result.audio.getNumSamples() / result.sampleRate;
            expect (std::abs ((double) result.f0Hz.size() - seconds * 100.0) <= 2.0);

            const auto first = result.notes.front().startSeconds;
            for (int i = 0; i < (int) (first * 100.0) - 30; ++i)
                expectEquals (result.f0Hz[(size_t) i], 0.0f);

            const auto track = trackPitch (result.audio.getReadPointer (0), result.audio.getNumSamples(), result.sampleRate, 0.01);
            for (const auto& note : result.notes)
            {
                const int a = (int) std::ceil ((note.startSeconds + 0.25 * (note.endSeconds - note.startSeconds)) * 100.0);
                const int b = (int) std::floor ((note.startSeconds + 0.75 * (note.endSeconds - note.startSeconds)) * 100.0);
                int voiced = 0, total = 0;
                for (int i = a; i < b && i < (int) result.f0Hz.size(); ++i)
                {
                    ++total;
                    const float hz = result.f0Hz[(size_t) i];
                    if (hz <= 0.0f)
                        continue;
                    ++voiced;
                    expect (std::abs (music::hzToMidi (hz) - (float) note.midiNote) < 0.5f,
                            "f0Hz " + juce::String (hz) + " vs note " + juce::String (note.midiNote));

                    // ...and it is the pitch actually sung.
                    if (i < (int) track.size() && track[(size_t) i].hz > 0.0f)
                        expect (std::abs (music::hzToMidi (track[(size_t) i].hz) - music::hzToMidi (hz)) < 0.3f,
                                "measured " + juce::String (track[(size_t) i].hz) + " Hz vs f0Hz " + juce::String (hz));
                }
                expect (voiced >= total * 0.9, "the middle of each note is voiced");
            }
        }

        beginTest ("Automatic melody render");
        {
            SingEngine engine;
            SingRequest request;
            request.lyrics = "Hello world, I love you so\nThe sun is shining bright today";
            request.voice = findVoice ("male-rock");
            request.bpm = 110.0;
            request.seed = 5;
            request.sampleRate = 48000.0;

            float lastProgress = -1.0f;
            bool monotonic = true;
            const auto result = engine.render (request, nullptr, [&] (float p)
            {
                monotonic = monotonic && p >= lastProgress;
                lastProgress = p;
            });

            expect (result.ok(), result.errorMessage);
            expect (monotonic);
            expectWithinAbsoluteError (lastProgress, 1.0f, 1.0e-6f);
            expect (allFinite (result.audio));

            int sung = 0;
            for (const auto& n : result.notes)
                sung += n.syllable != "_" ? 1 : 0;
            expectEquals (sung, SingEngine::countSyllables (request.lyrics));
            expect (result.keyRoot >= 0 && result.keyRoot < 12);

            const double seconds = result.audio.getNumSamples() / result.sampleRate;
            expect (seconds > result.notes.back().endSeconds && seconds < result.notes.back().endSeconds + 1.0);

            // Same request, same take.
            const auto again = engine.render (request);
            expect (again.audio.getNumSamples() == result.audio.getNumSamples());
            bool identical = again.audio.getNumSamples() == result.audio.getNumSamples();
            if (identical)
                identical = std::memcmp (again.audio.getReadPointer (0), result.audio.getReadPointer (0),
                                         sizeof (float) * (size_t) result.audio.getNumSamples()) == 0;
            expect (identical, "rendering must be deterministic");
        }

        beginTest ("Every factory voice renders");
        {
            SingEngine engine;
            for (const auto& voice : voices)
            {
                SingRequest request;
                request.lyrics = "Shine on, sweet light";
                request.voice = voice;
                request.sampleRate = 44100.0;
                const auto result = engine.render (request);
                expect (result.ok() && allFinite (result.audio), voice.id + ": " + result.errorMessage);
                const float peakDb = juce::Decibels::gainToDecibels (result.audio.getMagnitude (0, 0, result.audio.getNumSamples()));
                expectWithinAbsoluteError (peakDb, -12.0f, 0.5f);
            }
        }

        beginTest ("Cancellation and errors");
        {
            SingEngine engine;
            SingRequest request;
            request.lyrics = "This is a long song that would take a while to render if nobody stopped it\n"
                             "and it goes on and on with more and more words in every single line";
            request.voice = findVoice ("female-pop");

            std::atomic<bool> cancel { true };
            const auto start = juce::Time::getMillisecondCounterHiRes();
            auto result = engine.render (request, &cancel);
            expect (! result.ok());
            expectEquals (result.errorMessage, juce::String ("Cancelled"));
            expect (juce::Time::getMillisecondCounterHiRes() - start < 50.0);

            cancel = false;
            float maxProgress = 0.0f;
            result = engine.render (request, &cancel, [&] (float p)
            {
                maxProgress = juce::jmax (maxProgress, p);
                if (p > 0.3f)
                    cancel = true;
            });
            expect (! result.ok());
            expect (result.audio.getNumSamples() == 0);
            expect (maxProgress < 0.9f, "render stops soon after the flag is raised");

            request.lyrics = "  ";
            expect (engine.render (request).errorMessage.isNotEmpty());

            request.lyrics = "hello";
            request.melodyMode = MelodyMode::Provided;
            request.melody.clear();
            expect (engine.render (request).errorMessage.isNotEmpty());

            request.melodyMode = MelodyMode::AutoGenerate;
            request.sampleRate = 0.0;
            expect (engine.render (request).errorMessage.isNotEmpty());
        }

        beginTest ("Robustness: long lyrics, extreme tempi, junk text");
        {
            juce::String longLyrics;
            while (longLyrics.length() < 1950)
                longLyrics << "I keep on running through the night, chasing every falling light\n";

            struct Case { juce::String lyrics; double bpm; const char* voice; };
            const Case cases[] = {
                { longLyrics, 120.0, "female-pop" },
                { longLyrics.substring (0, 400), 300.0, "male-rap" },
                { "so slow and low", 30.0, "male-jazz" },
                { juce::String::fromUTF8 ("\xe2\x80\x9cWow!!\xe2\x80\x9d \xe2\x80\x94 caf\xc3\xa9 na\xc3\xafve 123456789 $$$ %%% ___ --- [br][br] \xe6\x97\xa5\xe6\x9c\xac"), 120.0, "female-rock" },
            };

            SingEngine engine;
            for (const auto& c : cases)
            {
                SingRequest request;
                request.lyrics = c.lyrics;
                request.bpm = c.bpm;
                request.voice = findVoice (c.voice);
                const auto result = engine.render (request);
                expect (result.ok() && allFinite (result.audio), result.errorMessage);

                int sung = 0;
                for (const auto& n : result.notes)
                    sung += n.syllable != "_" ? 1 : 0;
                expectEquals (sung, SingEngine::countSyllables (c.lyrics));
            }

            SingRequest onlyMarks;
            onlyMarks.lyrics = "_ _ [br] -- !!";
            expect (engine.render (onlyMarks).errorMessage.isNotEmpty());
        }

        beginTest ("Other sample rates");
        {
            for (double sr : { 22050.0, 44100.0, 96000.0 })
            {
                auto request = makeRequest ("male-pop", "la la", 120.0, { { 1.0, 1.0, 57, 0.8f }, { 2.0, 1.0, 60, 0.8f } });
                request.sampleRate = sr;
                SingEngine engine;
                const auto result = engine.render (request);
                expect (result.ok() && allFinite (result.audio), "sample rate " + juce::String (sr));
                expectWithinAbsoluteError (result.sampleRate, sr, 1.0e-9);
                const auto track = trackPitch (result.audio.getReadPointer (0), result.audio.getNumSamples(), sr, 0.01);
                expect (std::abs (medianMidi (track, 0.01, 0.5, 1.0) - 57.0f) < 0.25f);
            }
        }

        testLiveSinger();
    }

private:
    std::vector<VoiceProfile> voices;

    const VoiceProfile& findVoice (const juce::String& id) const
    {
        for (const auto& v : voices)
            if (v.id == id)
                return v;
        return voices.front();
    }

    SingRequest makeRequest (const char* voiceId, const juce::String& lyrics, double bpm, std::vector<Note> melody)
    {
        SingRequest r;
        r.lyrics = lyrics;
        r.voice = findVoice (voiceId);
        r.bpm = bpm;
        r.melodyMode = MelodyMode::Provided;
        r.melody = std::move (melody);
        r.sampleRate = 48000.0;
        return r;
    }

    static bool allFinite (const juce::AudioBuffer<float>& b)
    {
        for (int c = 0; c < b.getNumChannels(); ++c)
            for (int i = 0; i < b.getNumSamples(); ++i)
                if (! std::isfinite (b.getSample (c, i)))
                    return false;
        return true;
    }

    static float rmsBetween (const SingResult& r, double from, double to)
    {
        const int a = juce::jlimit (0, r.audio.getNumSamples(), (int) (from * r.sampleRate));
        const int b = juce::jlimit (0, r.audio.getNumSamples(), (int) (to * r.sampleRate));
        return b > a ? r.audio.getRMSLevel (0, a, b - a) : 0.0f;
    }

    /** Median measured pitch over the middle half of [start, end]. */
    static float medianMidi (const std::vector<PitchFrame>& track, double hop, double start, double end)
    {
        std::vector<float> values;
        const double a = start + 0.25 * (end - start), b = start + 0.75 * (end - start);
        for (int i = (int) std::ceil (a / hop); i <= (int) std::floor (b / hop) && i < (int) track.size(); ++i)
            if (track[(size_t) i].hz > 0.0f)
                values.push_back (music::hzToMidi (track[(size_t) i].hz));
        if (values.empty())
            return 0.0f;
        std::sort (values.begin(), values.end());
        return values[values.size() / 2];
    }

    static float renderLive (LiveSinger& singer, juce::AudioBuffer<float>& out, int& pos, double seconds, double sr)
    {
        const int end = juce::jmin (out.getNumSamples(), pos + (int) (seconds * sr));
        const int start = pos;
        while (pos < end)
        {
            const int n = juce::jmin (256, end - pos);
            singer.render (out.getWritePointer (0) + pos, n);
            pos += n;
        }
        return end > start ? out.getRMSLevel (0, start, end - start) : 0.0f;
    }

    void testLiveSinger()
    {
        constexpr double sr = 48000.0;

        beginTest ("LiveSinger: sound on note-on, decay to silence after note-off");
        {
            LiveSinger singer;
            singer.prepare (sr, 256);
            singer.setVoice (findVoice ("female-pop").traits);
            singer.setLyrics ("la la la");

            juce::AudioBuffer<float> out (1, (int) (sr * 4));
            out.clear();
            int pos = 0;

            expectEquals (renderLive (singer, out, pos, 0.1, sr), 0.0f);
            expect (! singer.isSounding());

            singer.noteOn (64, 0.8f);
            const int noteStart = pos;
            const float sounding = renderLive (singer, out, pos, 0.6, sr);
            expect (sounding > 0.02f, "rms " + juce::String (sounding));
            expect (singer.isSounding());
            expect (allFinite (out));

            const float peakDb = juce::Decibels::gainToDecibels (out.getMagnitude (0, noteStart, pos - noteStart));
            expect (peakDb > -16.0f && peakDb < -6.0f, "live peak " + juce::String (peakDb, 1) + " dBFS");

            const auto track = trackPitch (out.getReadPointer (0), pos, sr, 0.01);
            expect (std::abs (medianMidi (track, 0.01, 0.35, 0.7) - 64.0f) < 0.25f);

            singer.noteOff (64);
            renderLive (singer, out, pos, 0.7, sr);
            expect (renderLive (singer, out, pos, 0.3, sr) < 1.0e-4f);
            expect (! singer.isSounding());
        }

        beginTest ("LiveSinger: syllables advance, wrap and restart");
        {
            LiveSinger singer;
            singer.prepare (sr, 256);
            singer.setLyrics ("one two _ three");
            expectEquals (singer.getNumSyllables(), 3);
            expectEquals (singer.getNumSyllables(), SingEngine::countSyllables ("one two _ three"));

            juce::AudioBuffer<float> out (1, (int) (sr * 3));
            int pos = 0;
            const int expected[] = { 1, 1, 2, 0, 1 };   // syllable the *next* note sings; "_" holds "two"
            renderLive (singer, out, pos, 0.01, sr);
            expectEquals (singer.getCurrentSyllableIndex(), 0);
            for (int i = 0; i < 5; ++i)
            {
                singer.noteOn (60 + i, 0.8f);
                renderLive (singer, out, pos, 0.1, sr);
                singer.noteOff (60 + i);
                renderLive (singer, out, pos, 0.05, sr);
                expectEquals (singer.getCurrentSyllableIndex(), expected[i]);
            }

            singer.restartLyrics();
            expectEquals (singer.getCurrentSyllableIndex(), 0);
        }

        beginTest ("LiveSinger: legato glides, last-note priority");
        {
            LiveSinger singer;
            singer.prepare (sr, 256);
            singer.setVoice (findVoice ("male-pop").traits);
            singer.setLyrics ("oh oh oh");

            juce::AudioBuffer<float> out (1, (int) (sr * 3));
            out.clear();
            int pos = 0;
            singer.noteOn (55, 0.8f);
            renderLive (singer, out, pos, 0.6, sr);
            singer.noteOn (62, 0.8f);                     // legato: 55 still held
            renderLive (singer, out, pos, 0.6, sr);
            singer.noteOff (62);                          // back to the held 55
            renderLive (singer, out, pos, 0.6, sr);
            expect (singer.isSounding());

            const auto track = trackPitch (out.getReadPointer (0), pos, sr, 0.01);
            expect (std::abs (medianMidi (track, 0.01, 0.3, 0.6) - 55.0f) < 0.3f);
            expect (std::abs (medianMidi (track, 0.01, 0.9, 1.2) - 62.0f) < 0.3f);
            expect (std::abs (medianMidi (track, 0.01, 1.5, 1.8) - 55.0f) < 0.3f);

            singer.allNotesOff();
            renderLive (singer, out, pos, 0.8, sr);
            expect (! singer.isSounding());
        }

        beginTest ("LiveSinger: empty lyrics sing a neutral vowel; lyrics can be swapped while playing");
        {
            LiveSinger singer;
            singer.prepare (sr, 256);
            juce::AudioBuffer<float> out (1, (int) (sr * 2));
            int pos = 0;
            singer.noteOn (60, 0.8f);
            expect (renderLive (singer, out, pos, 0.4, sr) > 0.01f);
            expectEquals (singer.getNumSyllables(), 0);

            singer.setLyrics ("new words here");
            renderLive (singer, out, pos, 0.1, sr);
            singer.noteOff (60);
            singer.noteOn (62, 0.8f);
            expect (renderLive (singer, out, pos, 0.3, sr) > 0.01f);
            expectEquals (singer.getNumSyllables(), 3);
            expectEquals (singer.getCurrentSyllableIndex(), 1);
        }

        beginTest ("LiveSinger: concurrent lyric updates from another thread");
        {
            LiveSinger singer;
            singer.prepare (sr, 128);
            std::atomic<bool> done { false };
            std::thread messageThread ([&]
            {
                for (int i = 0; i < 200; ++i)
                    singer.setLyrics (i % 2 == 0 ? "one two three four five" : "la la");
                done = true;
            });

            std::vector<float> block (128);
            int note = 0, blocks = 0;
            while (! done.load() || blocks < 200)
            {
                if (blocks % 20 == 0)
                {
                    singer.noteOff (48 + note);
                    note = (note + 1) % 12;
                    singer.noteOn (48 + note, 0.7f);
                }
                singer.render (block.data(), (int) block.size());
                ++blocks;
            }
            messageThread.join();
            singer.allNotesOff();
            expect (singer.getNumSyllables() == 2);
        }

        beginTest ("LiveSinger: CPU cost");
        {
            LiveSinger singer;
            singer.prepare (sr, 256);
            singer.setVoice (findVoice ("male-rock").traits);
            singer.setLyrics ("Hello world, I love you so, the sun is shining bright today");

            const int total = (int) (sr * 10.0);
            std::vector<float> block (256);
            const auto start = juce::Time::getMillisecondCounterHiRes();
            for (int pos = 0, n = 0; pos < total; pos += 256, ++n)
            {
                if (n % 50 == 0)
                {
                    singer.noteOff (50 + (n / 50 - 1) % 7);
                    singer.noteOn (50 + (n / 50) % 7, 0.8f);
                }
                singer.render (block.data(), 256);
            }
            const double elapsedMs = juce::Time::getMillisecondCounterHiRes() - start;
            const double percent = elapsedMs / 10000.0 * 100.0;
            logMessage ("LiveSinger CPU: " + juce::String (percent, 2) + "% of one core at 48 kHz");
            expect (percent < 5.0);
        }
    }
};

static SingRenderTests singRenderTests;

} // namespace voxa::sing
