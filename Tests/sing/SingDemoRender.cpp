// Development aid: renders demo takes to WAV files and prints per-phoneme levels.
// Does nothing unless the environment variable VOXA_SING_RENDER_DIR names an output directory:
//   VOXA_SING_RENDER_DIR=/tmp/sing ./VoxaVersaTests SingDemo

#include "sing/SingEngine.h"
#include "sing/Lyrics.h"
#include "sing/Melody.h"
#include "sing/OfflineRenderer.h"
#include "voices/VoiceLibrary.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <map>

namespace voxa::sing
{

class SingDemoRender : public juce::UnitTest
{
public:
    SingDemoRender() : juce::UnitTest ("Sing demo renders", "SingDemo") {}

    void runTest() override
    {
        beginTest ("Demo renders");

        const auto dirName = juce::SystemStats::getEnvironmentVariable ("VOXA_SING_RENDER_DIR", {});
        if (dirName.isEmpty())
        {
            expect (true);
            return;
        }

        outputDir = juce::File (dirName);
        outputDir.createDirectory();

        const auto voices = createFactoryVoices();
        auto voice = [&voices] (const juce::String& id)
        {
            for (const auto& v : voices)
                if (v.id == id)
                    return v;
            return voices.front();
        };

        const juce::String song = "Hello world, I love you so\n"
                                  "The sun is shining bright today\n"
                                  "Beau-ti-ful _ night, stay with me\n"
                                  "Don't let go of my heart";

        renderAuto ("pop_female", voice ("female-pop"), song, 1, 100.0);
        renderAuto ("pop_male", voice ("male-pop"), song, 1, 100.0);
        renderAuto ("rap_male", voice ("male-rap"), song, 3, 92.0);
        renderAuto ("opera_female", voice ("female-opera"), song, 2, 80.0);
        renderAuto ("rock_male", voice ("male-rock"), song, 4, 120.0);
        renderAuto ("whisper", voice ("neutral-whisper"), song, 5, 100.0);
        renderAuto ("deep_bass", voice ("male-deep-bass"), song, 6, 90.0);
        renderAuto ("metal_male", voice ("male-metal"), song, 7, 120.0);

        // Phoneme inventories on a monotone, for calibration and listening.
        renderMonotone ("vowels", voice ("female-pop"), "ah eh ee ih oh oo uh ur ay i ow oy", 62);
        renderMonotone ("fricatives", voice ("male-pop"), "sa sha fa tha ha za va zha tha", 52);
        renderMonotone ("stops", voice ("male-pop"), "pa ta ka ba da ga cha ja", 52);
        renderMonotone ("sonorants", voice ("male-pop"), "ma na nga la ra wa ya", 52);
        renderMonotone ("sentence", voice ("female-pop"), "she sells sea shells by the sea shore", 64);

        renderLive (voice ("female-pop"));

        // Breathiness / rasp ladders on a steady vowel (no vibrato or drift) for HNR measurements.
        for (float b : { 0.0f, 0.2f, 0.4f, 0.6f, 0.9f })
        {
            auto v = voice ("female-pop");
            v.traits.breathiness = b;
            v.traits.rasp = 0.0f;
            v.traits.vibratoDepthCents = 0.0f;
            v.traits.jitterCents = 0.0f;
            renderMonotone ("breath_" + juce::String ((int) (b * 10)), v, "ah ah", 62);
        }
        for (float r : { 0.0f, 0.3f, 0.7f })
        {
            auto v = voice ("male-rock");
            v.traits.rasp = r;
            v.traits.vibratoDepthCents = 0.0f;
            v.traits.jitterCents = 0.0f;
            renderMonotone ("rasp_" + juce::String ((int) (r * 10)), v, "ah ah", 52);
        }
    }

private:
    juce::File outputDir;

    void writeWav (const juce::String& takeName, const juce::AudioBuffer<float>& audio, double sampleRate)
    {
        auto file = outputDir.getChildFile (takeName + ".wav");
        file.deleteFile();
        std::unique_ptr<juce::OutputStream> stream (file.createOutputStream().release());
        juce::WavAudioFormat wav;
        auto writer = wav.createWriterFor (stream, juce::AudioFormatWriterOptions().withSampleRate (sampleRate)
                                                                                  .withNumChannels (1)
                                                                                  .withBitsPerSample (24));
        if (writer != nullptr)
            writer->writeFromAudioSampleBuffer (audio, 0, audio.getNumSamples());
    }

    void renderAuto (const juce::String& takeName, const VoiceProfile& voice, const juce::String& lyrics, juce::uint32 seed, double bpm)
    {
        SingEngine engine;
        SingRequest request;
        request.lyrics = lyrics;
        request.voice = voice;
        request.seed = seed;
        request.bpm = bpm;
        request.sampleRate = 48000.0;

        const auto start = juce::Time::getMillisecondCounterHiRes();
        auto result = engine.render (request);
        const auto elapsed = juce::Time::getMillisecondCounterHiRes() - start;

        if (! result.ok())
        {
            logMessage (takeName + ": " + result.errorMessage);
            return;
        }

        writeWav (takeName, result.audio, result.sampleRate);
        logMessage (takeName + ": " + juce::String (result.audio.getNumSamples() / result.sampleRate, 2) + " s in "
                    + juce::String (elapsed, 0) + " ms, key " + music::getKeyNames()[result.keyRoot]);

        juce::String line;
        const double beatsPerSecond = bpm / 60.0;
        for (const auto& n : result.notes)
            line << n.syllable << "(" << music::noteName (n.midiNote) << "@"
                 << juce::String (n.startSeconds * beatsPerSecond, 2) << "/"
                 << juce::String ((n.endSeconds - n.startSeconds) * beatsPerSecond, 2) << ") ";
        logMessage ("   " + line);
    }

    void renderMonotone (const juce::String& takeName, const VoiceProfile& voice, const juce::String& lyrics, int midi)
    {
        const auto script = parseLyrics (lyrics);
        Score score;
        double beat = 1.0;
        for (int i = 0; i < script.getNumSyllables(); ++i)
        {
            ScoreNote n;
            n.startBeat = beat;
            n.lengthBeats = 1.5;
            n.midiNote = midi;
            n.velocity = 0.8f;
            n.syllable = i;
            n.breathBefore = true;
            score.notes.push_back (n);
            beat += 2.0;
        }

        OfflineRenderer renderer (script, score, voice.traits, 120.0, 48000.0, 1);
        if (! renderer.render (nullptr, {}))
            return;

        writeWav (takeName, renderer.audio, 48000.0);

        // Per-phoneme RMS (dB relative to the loudest vowel segment).
        std::map<juce::String, std::pair<double, int>> energy;
        const auto* data = renderer.audio.getReadPointer (0);
        const int total = renderer.audio.getNumSamples();
        double vowelMax = 1.0e-12;
        for (const auto& seg : renderer.getSegments())
        {
            const int a = juce::jlimit (0, total, (int) (seg.start * 48000.0));
            const int b = juce::jlimit (0, total, (int) (seg.end * 48000.0));
            if (b - a < 48)
                continue;
            double sum = 0.0;
            for (int i = a; i < b; ++i)
                sum += (double) data[i] * data[i];
            const double rms = sum / (b - a);
            auto& e = energy[phonemeSymbol (seg.event.ph)];
            e.first += rms;
            e.second += 1;
            if (seg.nucleus)
                vowelMax = juce::jmax (vowelMax, rms);
        }

        juce::String line = takeName + " levels (dB re loudest vowel): ";
        for (const auto& [ph, e] : energy)
            line << ph << " " << juce::String (10.0 * std::log10 (e.first / e.second / vowelMax + 1.0e-12), 1) << "  ";
        logMessage (line);
    }

    void renderLive (const VoiceProfile& voice)
    {
        LiveSinger singer;
        singer.prepare (48000.0, 512);
        singer.setVoice (voice.traits);
        singer.setLyrics ("Hel-lo my friend, how are you");

        juce::AudioBuffer<float> out (1, 48000 * 8);
        out.clear();
        const int block = 256;
        int pos = 0;
        auto run = [&] (double seconds)
        {
            const int end = juce::jmin (out.getNumSamples(), pos + (int) (seconds * 48000.0));
            while (pos < end)
            {
                const int n = juce::jmin (block, end - pos);
                singer.render (out.getWritePointer (0) + pos, n);
                pos += n;
            }
        };

        const int melody[] = { 64, 67, 69, 67, 64, 62, 60 };
        run (0.3);
        for (int i = 0; i < 7; ++i)
        {
            singer.noteOn (melody[i], 0.8f);
            if (i > 0)
                singer.noteOff (melody[i - 1]);   // legato
            run (i == 2 ? 0.9 : 0.45);
            if (i == 3)
            {
                singer.noteOff (melody[i]);
                run (0.3);
            }
        }
        singer.noteOff (melody[6]);
        run (1.0);
        writeWav ("live", out, 48000.0);
        logMessage ("live peak: " + juce::String (juce::Decibels::gainToDecibels (out.getMagnitude (0, out.getNumSamples())), 1) + " dBFS");
    }
};

static SingDemoRender singDemoRender;

} // namespace voxa::sing
