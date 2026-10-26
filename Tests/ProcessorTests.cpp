#include "PluginProcessor.h"
#include "core/Watermark.h"
#include "core/AudioUtils.h"
#include "dsp/PitchDetector.h"
#include "TestSignals.h"

namespace voxa
{

class ProcessorTests : public juce::UnitTest
{
public:
    ProcessorTests() : juce::UnitTest ("Processor", "Processor") {}

    static constexpr double sr = 48000.0;
    static constexpr int block = 512;

    /** Runs a mono signal through the processor (as stereo input) and returns the left output. */
    std::vector<float> run (VoxaVersaProcessor& p, const std::vector<float>& in, juce::MidiBuffer* firstBlockMidi = nullptr)
    {
        std::vector<float> out (in.size(), 0.0f);
        juce::AudioBuffer<float> buffer (2, block);
        juce::MidiBuffer midi;
        for (size_t pos = 0; pos < in.size(); pos += block)
        {
            const int n = (int) juce::jmin ((size_t) block, in.size() - pos);
            buffer.setSize (2, n, false, false, true);
            for (int ch = 0; ch < 2; ++ch)
                buffer.copyFrom (ch, 0, in.data() + pos, n);
            midi.clear();
            if (pos == 0 && firstBlockMidi != nullptr)
                midi = *firstBlockMidi;
            p.processBlock (buffer, midi);
            std::copy (buffer.getReadPointer (0), buffer.getReadPointer (0) + n, out.begin() + (long) pos);
        }
        return out;
    }

    static void setParam (VoxaVersaProcessor& p, const char* id, float value)
    {
        auto* param = p.getState().getParameter (id);
        param->setValueNotifyingHost (param->convertTo0to1 (value));
    }

    static float medianPitch (const std::vector<float>& x, int from)
    {
        auto frames = dsp::PitchDetector::analyse (x.data() + from, (int) x.size() - from, sr, 256);
        std::vector<float> f;
        for (auto& fr : frames)
            if (fr.voiced) f.push_back (fr.frequencyHz);
        if (f.empty()) return 0.0f;
        std::sort (f.begin(), f.end());
        return f[f.size() / 2];
    }

    void runTest() override
    {
        beginTest ("Prepare, latency and default processing");
        {
            VoxaVersaProcessor p;
            p.setPlayConfigDetails (2, 2, sr, block);
            p.prepareToPlay (sr, block);
            expect (p.getLatencySamples() > 0, "latency reported");

            auto in = test::makeVoice (sr, (int) (sr * 2.0), [] (int) { return 196.0f; });
            auto out = run (p, in);
            float peak = 0.0f;
            bool finite = true;
            for (auto v : out) { finite = finite && std::isfinite (v); peak = juce::jmax (peak, std::abs (v)); }
            expect (finite);
            expect (peak > 0.02f && peak < 2.0f, "peak " + juce::String (peak));
            expect (p.detectedPitchHz.load() > 150.0f, "telemetry pitch " + juce::String (p.detectedPitchHz.load()));
        }

        beginTest ("Pitch knob transposes the output");
        {
            VoxaVersaProcessor p;
            p.prepareToPlay (sr, block);
            setParam (p, ParamID::pitch, 5.0f);
            setParam (p, ParamID::reverbMix, 0.0f);
            setParam (p, ParamID::vibe, 0.0f);
            setParam (p, ParamID::expression, 0.0f);
            auto in = test::makeVoice (sr, (int) (sr * 2.0), [] (int) { return 196.0f; });
            const float measured = medianPitch (run (p, in), (int) (sr * 0.5));
            const float expected = 196.0f * std::exp2 (5.0f / 12.0f);
            expect (std::abs (1200.0f * std::log2 (measured / expected)) < 15.0f, "measured " + juce::String (measured));
        }

        beginTest ("MIDI pitch control");
        {
            VoxaVersaProcessor p;
            p.prepareToPlay (sr, block);
            setParam (p, ParamID::midiControl, 1.0f);
            setParam (p, ParamID::retuneSpeed, 0.0f);
            setParam (p, ParamID::humanize, 0.0f);
            setParam (p, ParamID::expression, 0.0f);
            setParam (p, ParamID::vibe, 0.0f);
            setParam (p, ParamID::reverbMix, 0.0f);
            juce::MidiBuffer midi;
            midi.addEvent (juce::MidiMessage::noteOn (1, 67, (juce::uint8) 100), 0);   // G4
            auto in = test::makeVoice (sr, (int) (sr * 2.0), [] (int) { return 220.0f; });
            const float measured = medianPitch (run (p, in, &midi), (int) (sr * 0.5));
            expect (std::abs (1200.0f * std::log2 (measured / 392.0f)) < 15.0f, "measured " + juce::String (measured));
        }

        beginTest ("Power off passes the (latency-aligned) input through");
        {
            VoxaVersaProcessor p;
            p.prepareToPlay (sr, block);
            setParam (p, ParamID::power, 0.0f);
            auto in = test::makeVoice (sr, (int) sr, [] (int) { return 200.0f; });
            auto out = run (p, in);
            const int latency = p.getLatencySamples();
            double err = 0.0, energy = 0.0;
            for (int i = (int) (sr * 0.2); i < (int) sr - latency; ++i)
            {
                const double e = out[(size_t) (i + latency)] - in[(size_t) i];
                err += e * e;
                energy += (double) in[(size_t) i] * in[(size_t) i];
            }
            expect (err < energy * 1.0e-4, "bypass error ratio " + juce::String (err / energy));
        }

        beginTest ("State save / restore round trip");
        {
            juce::MemoryBlock state;
            {
                VoxaVersaProcessor p;
                p.selectVoice ("male-country");
                setParam (p, ParamID::pitch, -7.0f);
                setParam (p, ParamID::harmonyOn, 1.0f);
                p.setLyrics ("Hello world, this is VoxaVersa");
                p.getStateInformation (state);
            }
            VoxaVersaProcessor q;
            q.setStateInformation (state.getData(), (int) state.getSize());
            test::waitFor ([] { return false; }, 50);
            expectEquals (q.getSelectedVoiceId(), juce::String ("male-country"));
            expectEquals (q.getLyrics(), juce::String ("Hello world, this is VoxaVersa"));
            expectWithinAbsoluteError (q.getState().getRawParameterValue (ParamID::pitch)->load(), -7.0f, 0.01f);
            expect (q.getState().getRawParameterValue (ParamID::harmonyOn)->load() > 0.5f);
        }

        beginTest ("Every factory preset loads and processes");
        {
            VoxaVersaProcessor p;
            p.prepareToPlay (sr, block);
            auto in = test::makeVoice (sr, (int) (sr * 0.5), [] (int i) { return 180.0f + 40.0f * std::sin ((float) i * 0.0002f); });
            auto& pm = p.getPresetManager();
            for (int i = 0; i < (int) pm.getPresets().size(); ++i)
            {
                if (! pm.getPresets()[(size_t) i].isFactory)
                    continue;
                pm.loadPreset (i);
                test::waitFor ([] { return false; }, 30);   // let quality changes apply
                p.prepareToPlay (sr, block);
                auto out = run (p, in);
                bool finite = true;
                float peak = 0.0f;
                for (auto v : out) { finite = finite && std::isfinite (v); peak = juce::jmax (peak, std::abs (v)); }
                expect (finite && peak < 4.0f, pm.getPresets()[(size_t) i].name + " peak " + juce::String (peak));
                expectEquals (p.getPresetManager().getCurrentName(), pm.getPresets()[(size_t) i].name);
            }
        }

        beginTest ("Factory presets keep the session's key, mode and levels");
        {
            VoxaVersaProcessor p;
            auto& state = p.getState();
            auto set = [&state] (const char* id, float value)
            {
                auto* param = state.getParameter (id);
                param->setValueNotifyingHost (param->convertTo0to1 (value));
            };
            auto value = [&state] (const char* id) { return state.getRawParameterValue (id)->load(); };
            set (ParamID::key, 10.0f);          // A
            set (ParamID::scale, 1.0f);         // Minor
            set (ParamID::mode, 1.0f);          // Text to Sing
            set (ParamID::outputGain, -6.0f);
            set (ParamID::reverbMix, 80.0f);

            auto& pm = p.getPresetManager();
            auto load = [&pm] (const juce::String& presetName)
            {
                for (int i = 0; i < (int) pm.getPresets().size(); ++i)
                    if (pm.getPresets()[(size_t) i].name == presetName)
                        pm.loadPreset (i);
            };
            load ("Robot Hard Tune");
            expectWithinAbsoluteError (value (ParamID::key), 10.0f, 0.01f);
            expectWithinAbsoluteError (value (ParamID::scale), 1.0f, 0.01f);
            expectWithinAbsoluteError (value (ParamID::mode), 1.0f, 0.01f);
            expectWithinAbsoluteError (value (ParamID::outputGain), -6.0f, 0.01f);
            expectWithinAbsoluteError (value (ParamID::reverbMix), 12.0f, 0.01f);   // sound settings: defaults

            load ("Chipmunk");
            expect (value (ParamID::aiEngine) < 0.5f, "Chipmunk uses the DSP engine");
            load ("Radio Pop Star");
            expect (value (ParamID::aiEngine) > 0.5f, "AI back on");
            expect (value (ParamID::pitchPreserve) < 0.5f, "smart register for a female voice");
        }

        beginTest ("Voice sample offline render to WAV");
        {
            auto tempDir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("voxaversa-tests");
            tempDir.createDirectory();
            auto wavFile = tempDir.getChildFile ("sample.wav");
            const double fileRate = 44100.0;
            auto voice = test::makeVoice (fileRate, (int) (fileRate * 1.5), [] (int) { return 150.0f; });
            juce::AudioBuffer<float> buf (1, (int) voice.size());
            buf.copyFrom (0, 0, voice.data(), (int) voice.size());
            expect (writeWav (wavFile, buf, fileRate));

            VoxaVersaProcessor p;
            // the DSP transform: AI voices installed on this computer would otherwise take over the default voice
            if (auto* ai = p.getState().getParameter (ParamID::aiEngine))
                ai->setValueNotifyingHost (0.0f);
            p.prepareToPlay (sr, block);
            juce::String error;
            expect (p.loadSample (wavFile, error), error);
            expect (p.hasSample());
            expectWithinAbsoluteError (p.getSampleLengthSeconds(), 1.5, 0.01);

            p.startOfflineRender (VoxaVersaProcessor::RenderTarget::Sample);
            expect (test::waitFor ([&p] { return ! p.isOfflineRendering(); }, 60000), "render finished");
            const auto rendered = p.getLastRenderedFile();
            expect (rendered.existsAsFile(), "rendered file: " + p.getLastRenderError());

            juce::AudioFormatManager fm;
            fm.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor (rendered));
            expect (reader != nullptr);
            if (reader != nullptr)
            {
                const double seconds = (double) reader->lengthInSamples / reader->sampleRate;
                expect (seconds >= 1.45 && seconds < 5.0, "rendered length " + juce::String (seconds));
                expectEquals ((int) reader->numChannels, 2);
                // Every export carries a machine-readable disclosure of how the vocal was made.
                expect (reader->metadataValues[juce::WavAudioFormat::riffInfoComment2].contains ("VoxaVersa"),
                        "disclosure comment: " + reader->metadataValues[juce::WavAudioFormat::riffInfoComment2]);
                expect (reader->metadataValues[juce::WavAudioFormat::riffInfoSoftware].startsWith ("VoxaVersa"));

                // A DSP transform of the user's own voice is not AI output: no watermark.
                juce::AudioBuffer<float> audio ((int) reader->numChannels, (int) reader->lengthInSamples);
                reader->read (&audio, 0, (int) reader->lengthInSamples, 0, true, true);
                expect (! detectWatermark (audio, reader->sampleRate).found(), "transformed export carries an AI watermark");
            }
            rendered.deleteFile();

            // Sample playback through the live path.
            p.setSamplePlaying (true);
            std::vector<float> silence ((size_t) (sr * 0.5), 0.0f);
            auto out = run (p, silence);
            float peak = 0.0f;
            for (auto v : out) peak = juce::jmax (peak, std::abs (v));
            expect (peak > 0.01f, "sample playback audible");
            tempDir.deleteRecursively();
        }

        beginTest ("Text to Sing generation completes");
        {
            VoxaVersaProcessor p;
            p.prepareToPlay (sr, block);
            setParam (p, ParamID::mode, 1.0f);
            setParam (p, ParamID::melodySource, (float) MelodySource::AutoMelody);
            p.setLyrics ("Shine a little light tonight\nwe can sing until the morning");
            p.generateSinging();
            expect (test::waitFor ([&p] { return p.getSingState() != VoxaVersaProcessor::SingState::Rendering; }, 120000));

            if (p.getSingState() == VoxaVersaProcessor::SingState::Ready)
            {
                expect (p.getSingTakeLengthSeconds() > 1.0);
                expect (! p.getSungNotes().empty());
                p.setSingPreviewPlaying (true);
                // Auto melodies can start after a lead-in, so listen until past the first note.
                const double firstNote = p.getSungNotes().front().startSeconds;
                std::vector<float> silence ((size_t) (sr * (firstNote + 1.5)), 0.0f);
                auto out = run (p, silence);
                float peak = 0.0f;
                for (auto v : out) peak = juce::jmax (peak, std::abs (v));
                expect (peak > 0.01f, "sung preview audible");
                auto midiFile = p.exportSingMidi();
                expect (midiFile.existsAsFile());
                midiFile.deleteFile();

                // The exported take carries the synthetic-singing watermark.
                p.startOfflineRender (VoxaVersaProcessor::RenderTarget::SingTake);
                expect (test::waitFor ([&p] { return ! p.isOfflineRendering(); }, 120000), "sing render finished");
                const auto rendered = p.getLastRenderedFile();
                juce::AudioFormatManager fm;
                fm.registerBasicFormats();
                std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor (rendered));
                expect (reader != nullptr, "sung export: " + p.getLastRenderError());
                if (reader != nullptr)
                {
                    juce::AudioBuffer<float> audio ((int) reader->numChannels, (int) reader->lengthInSamples);
                    reader->read (&audio, 0, (int) reader->lengthInSamples, 0, true, true);
                    const auto mark = detectWatermark (audio, reader->sampleRate);
                    expect (mark.kind == WatermarkKind::SynthesizedSinging,
                            "sung export watermark: " + watermarkKindName (mark.kind) + " score "
                                + juce::String (mark.synthesizedSingingScore));
                }
                reader.reset();
                rendered.deleteFile();
            }
            else
            {
                logMessage ("Text to Sing not available: " + p.getSingError());
            }
        }
    }
};

static ProcessorTests processorTests;

} // namespace voxa
