// VoxaVersaRender: renders an audio file through the complete VoxaVersa processor offline.
//
//   VoxaVersaRender <in.wav> <out.wav> [--voice <id>] [--preset "<name>"] [--quality 0|1|2]
//                   [--set <paramId>=<value> ...] [--list-voices] [--list-params]
//                   [--realtime | --host-offline]
//   VoxaVersaRender <file> --analyse          source gender analysis
//   VoxaVersaRender <file> --detect           look for the VoxaVersa AI / synthetic-singing watermark
//   VoxaVersaRender --sing <lyrics.txt> <guide.wav> [--voice <id>] [--bpm 120] [--seed 1]
//                   [--segments <plan.tsv>] [--chain <voiced.wav> [--ai]] [--melody <file.mid>]
//                                             Text to Sing with an automatic melody: the plain guide vocal,
//                                             its phoneme plan (one row per segment) and, with --chain, the
//                                             take rendered through the voice chain exactly as Generate +
//                                             export do in the plug-in (DSP path, AI engine off).
//   VoxaVersaRender --rkeeper|--breaths <dry.wav> <ai.wav> <out.wav> [ai-lag-seconds] [f3.csv]
//                                             developer check of the R keeper / breath keeper on its own
//
// Values for --set are real parameter values (e.g. pitch=5, gender=60, tuneOn=1, key=1 for C).
// --realtime feeds the blocks at the pace of live playback (what a DAW does while playing);
// --host-offline flags the render as non-real-time, like a DAW export or Render in Place.
// Prints source/target pitch statistics so renders can be compared objectively.

#include "PluginProcessor.h"
#include "Parameters.h"
#include "core/AiDisclosure.h"
#include "core/AudioUtils.h"
#include "core/Watermark.h"
#include "dsp/PitchDetector.h"
#include "dsp/VoiceGenderEstimator.h"
#include "dsp/VocalFx.h"
#include "sing/Lyrics.h"
#include "sing/Melody.h"
#include "sing/OfflineRenderer.h"
#include "sing/SingEngine.h"
#include <map>

namespace
{
    int fail (const juce::String& message);

    juce::String optionValue (const juce::StringArray& args, const char* name, const juce::String& fallback)
    {
        const int i = args.indexOf (name);
        return i >= 0 && i + 1 < args.size() ? args[i + 1] : fallback;
    }

    /** --sing: the same pipeline as SingEngine::render (seed 1 matches the plug-in's first Generate). */
    int renderSing (voxa::VoxaVersaProcessor& processor, const juce::StringArray& args)
    {
        using namespace voxa::sing;
        if (args.size() < 3)
            return fail ("usage: VoxaVersaRender --sing <lyrics.txt> <guide.wav> [--voice id] [--bpm N] [--seed N] "
                         "[--segments plan.tsv] [--chain voiced.wav]");

        const auto cwd = juce::File::getCurrentWorkingDirectory();
        const auto lyrics = cwd.getChildFile (args[1]).loadFileAsString();
        const auto guideFile = cwd.getChildFile (args[2]);
        const double bpm = optionValue (args, "--bpm", "120").getDoubleValue();
        const auto seed = (juce::uint32) optionValue (args, "--seed", "1").getIntValue();
        const double sr = 48000.0;

        if (const auto id = optionValue (args, "--voice", {}); id.isNotEmpty())
        {
            if (processor.getVoiceLibrary().findById (id) == nullptr)
                return fail ("unknown voice " + id + " (see --list-voices)");
            processor.selectVoice (id);
        }
        juce::MessageManager::getInstance()->runDispatchLoopUntil (20);
        auto voice = processor.getSelectedVoice();
        if (args.contains ("--ai") && voice.hasNeuralModel())
            voice.traits = voice.traits.withAiSingingRange();   // as the plug-in composes for its AI singer

        // --trait name=value overrides a voice trait of the guide (for experiments; not used by --chain).
        for (int i = 3; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--trait")
                continue;
            const auto name = args[i + 1].upToFirstOccurrenceOf ("=", false, false);
            const auto value = args[i + 1].fromFirstOccurrenceOf ("=", false, false).getFloatValue();
            auto& t = voice.traits;
            std::map<juce::String, float*> traits { { "formantSemis", &t.formantSemis }, { "breathiness", &t.breathiness },
                                                    { "rasp", &t.rasp }, { "tiltDb", &t.tiltDb },
                                                    { "vibratoDepthCents", &t.vibratoDepthCents },
                                                    { "vibratoRateHz", &t.vibratoRateHz }, { "scoopCents", &t.scoopCents },
                                                    { "jitterCents", &t.jitterCents }, { "singLegato", &t.singLegato },
                                                    { "singRhythmBusy", &t.singRhythmBusy },
                                                    { "singPortamentoMs", &t.singPortamentoMs } };
            if (traits.count (name) == 0)
                return fail ("unknown trait " + name);
            *traits[name] = value;
        }

        const auto script = parseLyrics (lyrics);
        if (script.isEmpty())
            return fail ("no lyrics in " + args[1]);
        // --melody <file.mid>: sing a given melody (as the plug-in's MIDI File melody source) instead of composing one
        const auto melodyFile = optionValue (args, "--melody", {});
        std::vector<voxa::sing::Note> provided;
        if (melodyFile.isNotEmpty())
        {
            double fileBpm = bpm;
            juce::String error;
            provided = voxa::sing::SingEngine::melodyFromMidiFile (cwd.getChildFile (melodyFile), fileBpm, error);
            if (provided.empty())
                return fail ("melody: " + error);
        }
        const auto score = provided.empty() ? melody::compose (script, voice.traits, -1, voxa::ScaleType::Major, seed)
                                            : melody::fitProvided (script, provided, voice.traits, -1, voxa::ScaleType::Major, bpm);
        OfflineRenderer renderer (script, score, voice.traits, bpm, sr, seed);
        if (! renderer.render (nullptr, {}))
            return fail ("render failed");
        if (! voxa::writeWav (guideFile, renderer.audio, sr, voxa::disclosureMetadata (voxa::VocalOrigin::SynthesizedSinging, voice.name)))
            return fail ("cannot write " + guideFile.getFullPathName());
        std::printf ("%s: %s, %.1f s, %d syllables, key %s %s\n", guideFile.getFileName().toRawUTF8(), voice.name.toRawUTF8(),
                     renderer.audio.getNumSamples() / sr, script.getNumSyllables(),
                     voxa::music::getKeyNames()[score.keyRoot % 12].toRawUTF8(), voxa::music::getScaleNames()[(int) score.scale].toRawUTF8());

        if (const auto plan = optionValue (args, "--segments", {}); plan.isNotEmpty())
        {
            // One row per planned segment, with the syllable/word it belongs to.
            juce::String tsv ("start\tend\tph\tnucleus\tsyllable\tstress\tfunction\tword\ttext\tmidi\tnoteStart\tnoteEnd\n");
            const double secondsPerBeat = 60.0 / bpm;
            for (const auto& s : renderer.getSegments())
            {
                int syllable = -1;
                const ScoreNote* note = nullptr;
                if (s.note >= 0 && s.note < (int) score.notes.size())
                {
                    note = &score.notes[(size_t) s.note];
                    syllable = note->syllable;
                }
                const auto* syl = syllable >= 0 && syllable < script.getNumSyllables() ? &script.syllables[(size_t) syllable] : nullptr;
                tsv << juce::String (s.start, 4) << "\t" << juce::String (s.end, 4) << "\t" << phonemeSymbol (s.event.ph) << "\t"
                    << (s.nucleus ? 1 : 0) << "\t" << syllable << "\t" << (syl != nullptr ? syl->stress : -1) << "\t"
                    << (syl != nullptr && syl->functionWord ? 1 : 0) << "\t"
                    << (syl != nullptr ? script.words[syl->wordIndex] : juce::String ("-")) << "\t"
                    << (syl != nullptr ? syl->text : juce::String ("-")) << "\t" << (note != nullptr ? note->midiNote : -1) << "\t"
                    << juce::String (note != nullptr ? note->startBeat * secondsPerBeat : 0.0, 4) << "\t"
                    << juce::String (note != nullptr ? note->endBeat() * secondsPerBeat : 0.0, 4) << "\n";
            }
            cwd.getChildFile (plan).replaceWithText (tsv);
        }

        const auto chainName = optionValue (args, "--chain", {});
        if (chainName.isEmpty())
            return 0;

        // Through the plug-in: Generate (auto melody, this tempo) then export the take through the voice chain.
        auto& state = processor.getState();
        auto set = [&state] (const char* id, float value)
        {
            if (auto* p = state.getParameter (id))
                p->setValueNotifyingHost (p->convertTo0to1 (value));
        };
        set (voxa::ParamID::mode, (float) (int) voxa::Mode::TextToSing);
        set (voxa::ParamID::melodySource, melodyFile.isNotEmpty() ? 2.0f : 1.0f);   // MIDI File or Auto Melody
        if (melodyFile.isNotEmpty())
            state.state.setProperty (voxa::StateID::melodyFile, cwd.getChildFile (melodyFile).getFullPathName(), nullptr);
        set (voxa::ParamID::singTempo, (float) bpm);
        set (voxa::ParamID::key, 0.0f);            // automatic
        set (voxa::ParamID::aiEngine, args.contains ("--ai") ? 1.0f : 0.0f);   // --ai: the AI singer sings the take
        processor.setPlayConfigDetails (2, 2, sr, 512);
        for (int i = 0; i < 1200 && processor.getNeuralStatus() == voxa::VoxaVersaProcessor::NeuralStatus::Loading; ++i)
            juce::MessageManager::getInstance()->runDispatchLoopUntil (50);
        processor.prepareToPlay (sr, 512);
        juce::MessageManager::getInstance()->runDispatchLoopUntil (50);

        processor.setLyrics (lyrics);
        for (juce::uint32 i = 1; i < seed; ++i)   // the plug-in advances its seed on every Generate
        {
            processor.generateSinging();
            while (processor.getSingState() == voxa::VoxaVersaProcessor::SingState::Rendering)
                juce::MessageManager::getInstance()->runDispatchLoopUntil (10);
        }
        processor.generateSinging();
        while (processor.getSingState() == voxa::VoxaVersaProcessor::SingState::Rendering)
            juce::MessageManager::getInstance()->runDispatchLoopUntil (10);
        if (processor.getSingState() != voxa::VoxaVersaProcessor::SingState::Ready)
            return fail ("Generate failed: " + processor.getSingError());

        processor.startOfflineRender (voxa::VoxaVersaProcessor::RenderTarget::SingTake);
        while (processor.isOfflineRendering())
            juce::MessageManager::getInstance()->runDispatchLoopUntil (10);
        const auto rendered = processor.getLastRenderedFile();
        if (! rendered.existsAsFile())
            return fail ("voice chain render failed: " + processor.getLastRenderError());

        const auto chainFile = cwd.getChildFile (chainName);
        chainFile.deleteFile();
        if (! rendered.moveFileTo (chainFile))
            return fail ("cannot write " + chainFile.getFullPathName());
        std::printf ("%s: through the %s voice chain\n", chainFile.getFileName().toRawUTF8(), voice.name.toRawUTF8());
        return 0;
    }

    float medianVoicedPitch (const float* x, int n, double sr)
    {
        auto frames = voxa::dsp::PitchDetector::analyse (x, n, sr, 480);
        std::vector<float> f;
        for (auto& fr : frames)
            if (fr.voiced)
                f.push_back (fr.frequencyHz);
        if (f.empty())
            return 0.0f;
        std::sort (f.begin(), f.end());
        return f[f.size() / 2];
    }

    int fail (const juce::String& message)
    {
        std::fprintf (stderr, "%s\n", message.toRawUTF8());
        return 1;
    }
}

/** Developer check of the R keeper on files: --rkeeper dry.wav ai.wav out.wav [ai-lag-seconds] [f3.csv]
    (the AI file may be a DAW bounce that starts `lag` seconds late). Voicing from the live pitch detector. */
static int rKeeperTool (const juce::StringArray& args)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    auto load = [&] (const juce::String& path, double& rate)
    {
        std::unique_ptr<juce::AudioFormatReader> r (formats.createReaderFor (juce::File::getCurrentWorkingDirectory().getChildFile (path)));
        if (r == nullptr) return juce::AudioBuffer<float>();
        rate = r->sampleRate;
        juce::AudioBuffer<float> b ((int) r->numChannels, (int) r->lengthInSamples);
        r->read (&b, 0, (int) r->lengthInSamples, 0, true, true);
        return voxa::toMono (b);
    };
    double sr = 0.0, sr2 = 0.0;
    auto dry = load (args[1], sr), ai = load (args[2], sr2);
    if (dry.getNumSamples() == 0 || ai.getNumSamples() == 0 || sr != sr2)
        return fail ("cannot read the two files (same sample rate needed)");
    const int lag = args.size() > 4 ? juce::roundToInt (args[4].getDoubleValue() * sr) : 0;
    const int n = juce::jmin (dry.getNumSamples(), ai.getNumSamples() - lag);

    const int hop = 240;
    auto frames = voxa::dsp::PitchDetector::analyse (dry.getReadPointer (0), n, sr, hop);
    std::vector<float> voiced ((size_t) n), lead ((size_t) n), dryv ((size_t) n);
    for (int i = 0; i < n; ++i)
    {
        voiced[(size_t) i] = frames[(size_t) juce::jmin ((int) frames.size() - 1, i / hop)].voiced ? 1.0f : 0.0f;
        lead[(size_t) i] = ai.getSample (0, i + lag);
        dryv[(size_t) i] = dry.getSample (0, i);
    }

    using voxa::dsp::RKeeper;
    using voxa::dsp::VoiceGenderEstimator;
    auto ceilingOf = [&] (const float* x)
    {
        return VoiceGenderEstimator::analyse (x, n, sr) == VoiceGenderEstimator::Gender::Female ? RKeeper::femaleCeilingHz
                                                                                                 : RKeeper::maleCeilingHz;
    };
    const double dryCeiling = ceilingOf (dryv.data()), aiCeiling = ceilingOf (lead.data());
    RKeeper keeper;
    keeper.prepare (sr);
    keeper.setCeilings (dryCeiling, aiCeiling);
    RKeeper::FormantTracker td, ta;
    td.prepare (sr, dryCeiling);
    ta.prepare (sr, aiCeiling);
    juce::String csv ("time,dry_f3,ai_f3,dry_db,voiced\n");
    for (int i = 0; i < n; ++i)
    {
        ta.push (lead[(size_t) i]);
        td.push (dryv[(size_t) i]);
        if ((i + 1) % 240 == 0)
        {
            td.analyse();
            ta.analyse();
            csv << juce::String (i / sr, 3) << "," << td.f3 << "," << ta.f3 << "," << td.levelDb << "," << voiced[(size_t) i] << "\n";
        }
    }
    auto out = lead;
    voxa::dsp::BreathKeeper breaths;
    breaths.prepare (sr);
    const bool breathsOnly = args[0] == "--breaths";
    for (int pos = 0; pos < n; pos += 512)
    {
        const int m = juce::jmin (512, n - pos);
        if (breathsOnly)
            breaths.process (out.data() + pos, nullptr, dryv.data() + pos, voiced.data() + pos, m);
        else
            keeper.process (out.data() + pos, dryv.data() + pos, voiced.data() + pos, m);
    }

    juce::AudioBuffer<float> buf (1, n);
    buf.copyFrom (0, 0, out.data(), n);
    if (! voxa::writeWav (juce::File::getCurrentWorkingDirectory().getChildFile (args[3]), buf, sr, {}))
        return fail ("cannot write " + args[3]);
    if (args.size() > 5)
        juce::File::getCurrentWorkingDirectory().getChildFile (args[5]).replaceWithText (csv);
    const auto& st = keeper.getStats();
    std::printf ("ceilings: input %.0f Hz, AI %.0f Hz\n", dryCeiling, aiCeiling);
    std::printf ("R keeper: %d sung frames, r applied in %d (%.1f%%); F3 baselines dry %.0f Hz, AI %.0f Hz\n",
                 st.sungFrames, st.rFrames, 100.0 * st.rFrames / juce::jmax (1, st.sungFrames), st.dryBase, st.aiBase);
    return 0;
}

int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juceInit;
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String::fromUTF8 (argv[i]));

    voxa::VoxaVersaProcessor processor;

    if (args.contains ("--list-voices"))
    {
        for (auto& v : processor.getVoiceLibrary().getVoices())
            std::printf ("%-24s %-20s %s\n", v.id.toRawUTF8(), v.name.toRawUTF8(), v.tagline.toRawUTF8());
        return 0;
    }

    if (args.contains ("--list-params"))
    {
        for (auto* p : processor.getParameters())
            if (auto* rp = dynamic_cast<juce::RangedAudioParameter*> (p))
                std::printf ("%-16s %s\n", rp->paramID.toRawUTF8(), rp->getName (64).toRawUTF8());
        return 0;
    }

    if (args.size() > 0 && args[0] == "--sing")
        return renderSing (processor, args);

    if (args.size() >= 4 && (args[0] == "--rkeeper" || args[0] == "--breaths"))
        return rKeeperTool (args);

    if (args.size() < 2 && ! args.contains ("--analyse") && ! args.contains ("--detect"))
        return fail ("usage: VoxaVersaRender <in.wav> <out.wav> [--voice id] [--preset name] [--quality 0|1|2] [--set id=value ...]");

    const juce::File inFile = juce::File::getCurrentWorkingDirectory().getChildFile (args[0]);
    const juce::File outFile = juce::File::getCurrentWorkingDirectory().getChildFile (args[1]);

    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (inFile));
    if (reader == nullptr)
        return fail ("cannot read " + inFile.getFullPathName());

    const double sr = reader->sampleRate;
    juce::AudioBuffer<float> raw ((int) reader->numChannels, (int) reader->lengthInSamples);
    reader->read (&raw, 0, (int) reader->lengthInSamples, 0, true, true);
    auto input = voxa::toMono (raw);

    if (args.contains ("--detect"))
    {
        const auto r = voxa::detectWatermark (raw, sr);
        std::printf ("%-62s watermark: %s (AI voice score %.1f, synthesized singing score %.1f, threshold %.1f)\n",
                     inFile.getFileName().toRawUTF8(), voxa::watermarkKindName (r.kind).toRawUTF8(),
                     r.aiVoiceScore, r.synthesizedSingingScore, voxa::WatermarkDetection::threshold);
        return r.found() ? 0 : 2;
    }

    if (args.contains ("--analyse"))
    {
        voxa::dsp::VoiceGenderEstimator est;
        est.prepare (sr);
        est.process (input.getReadPointer (0), input.getNumSamples());
        for (const auto& f : voxa::dsp::PitchDetector::analyse (input.getReadPointer (0), input.getNumSamples(), sr, (int) (sr * 0.01)))
            if (f.voiced)
                est.addPitch (f.frequencyHz);
        est.finalise();
        const auto g = est.getGender();
        std::printf ("%-62s %s score %+.2f f0 %.1f F3+F4 %.0f frames %d\n", inFile.getFileName().toRawUTF8(),
                     g == voxa::dsp::VoiceGenderEstimator::Gender::Female ? "FEMALE" : (g == voxa::dsp::VoiceGenderEstimator::Gender::Male ? "MALE  " : "?     "),
                     est.getScore(), est.getMedianF0(), est.getMedianF3F4(), est.getNumVoicedFrames());
        return 0;
    }

    auto& state = processor.getState();
    auto setParam = [&state] (const juce::String& id, float value)
    {
        if (auto* p = state.getParameter (id))
        {
            p->setValueNotifyingHost (p->convertTo0to1 (value));
            return true;
        }
        return false;
    };

    for (int i = 2; i < args.size(); ++i)
    {
        if (args[i] == "--voice" && i + 1 < args.size())
        {
            const auto id = args[++i];
            auto* voice = processor.getVoiceLibrary().findByIdForTools (id);
            if (voice == nullptr)
                return fail ("unknown voice " + id + " (see --list-voices)");
            // --trait name=value (after --voice): try a voice setting without rebuilding (experiments)
            for (int k = 2; k + 1 < args.size(); ++k)
            {
                if (args[k] != "--trait")
                    continue;
                const auto name = args[k + 1].upToFirstOccurrenceOf ("=", false, false);
                const auto value = args[k + 1].fromFirstOccurrenceOf ("=", false, false).getFloatValue();
                auto& t = voice->traits;
                std::map<juce::String, float*> traits { { "aiGrit", &t.aiGrit }, { "tiltDb", &t.tiltDb }, { "warmthDb", &t.warmthDb },
                                                        { "presenceDb", &t.presenceDb }, { "airDb", &t.airDb }, { "ringHz", &t.ringHz },
                                                        { "ringDb", &t.ringDb }, { "bodyHz", &t.bodyHz }, { "bodyDb", &t.bodyDb },
                                                        { "breathiness", &t.breathiness }, { "vibeSaturation", &t.vibeSaturation },
                                                        { "vibeCompressor", &t.vibeCompressor }, { "vibeReverbMix", &t.vibeReverbMix },
                                                        { "registerBiasSemis", &t.registerBiasSemis } };
                if (traits.count (name) == 0)
                    return fail ("unknown trait " + name);
                *traits[name] = value;
            }
            processor.selectVoice (id);
        }
        else if (args[i] == "--trait")
        {
            ++i;   // applied with --voice
        }
        else if (args[i] == "--preset" && i + 1 < args.size())
        {
            const auto name = args[++i];
            auto& pm = processor.getPresetManager();
            bool found = false;
            for (int k = 0; k < (int) pm.getPresets().size(); ++k)
                if (pm.getPresets()[(size_t) k].name.equalsIgnoreCase (name))
                {
                    pm.loadPreset (k);
                    found = true;
                }
            if (! found)
                return fail ("unknown preset " + name);
        }
        else if (args[i] == "--quality" && i + 1 < args.size())
        {
            setParam (voxa::ParamID::quality, args[++i].getFloatValue());
        }
        else if (args[i] == "--set" && i + 1 < args.size())
        {
            const auto kv = args[++i];
            if (! setParam (kv.upToFirstOccurrenceOf ("=", false, false), kv.fromFirstOccurrenceOf ("=", false, false).getFloatValue()))
                return fail ("unknown parameter in " + kv + " (see --list-params)");
        }
    }

    // Like a loaded voice sample, analyse the whole file's voice up front unless --set sourceVoice=... was given.
    if (state.getRawParameterValue (voxa::ParamID::sourceVoice)->load() < 0.5f && ! args.contains ("--live"))
        setParam (voxa::ParamID::sourceVoice, (float) (int) voxa::dsp::VoiceGenderEstimator::analyse (
                                                  input.getReadPointer (0), input.getNumSamples(), sr));

    // Let the processor apply voice / quality changes (message-thread work).
    juce::MessageManager::getInstance()->runDispatchLoopUntil (50);

    constexpr int block = 512;
    const bool realTimePacing = args.contains ("--realtime");
    processor.setNonRealtime (args.contains ("--host-offline"));
    processor.setPlayConfigDetails (2, 2, sr, block);
    processor.prepareToPlay (sr, block);
    juce::MessageManager::getInstance()->runDispatchLoopUntil (50);

    // An AI voice model loads on a background thread and is attached on the message thread: wait for it,
    // otherwise the render silently uses the DSP engine.
    for (int i = 0; i < 1200 && (processor.getNeuralStatus() == voxa::VoxaVersaProcessor::NeuralStatus::Loading
                                 || processor.isChoirPartnerLoading()); ++i)
        juce::MessageManager::getInstance()->runDispatchLoopUntil (50);
    if (processor.getNeuralStatus() != voxa::VoxaVersaProcessor::NeuralStatus::NoModelForVoice)
        std::printf ("  %s\n", processor.getNeuralStatusMessage().toRawUTF8());

    const int latency = processor.getLatencySamples();
    const int tail = (int) (sr * 2.0);
    const int total = input.getNumSamples() + latency + tail;

    juce::AudioBuffer<float> output (2, total);
    juce::AudioBuffer<float> buffer (2, block);
    juce::MidiBuffer midi;
    const auto start = juce::Time::getMillisecondCounterHiRes();

    for (int pos = 0; pos < total; pos += block)
    {
        const int n = juce::jmin (block, total - pos);
        buffer.setSize (2, n, false, false, true);
        buffer.clear();
        const int avail = juce::jmax (0, juce::jmin (n, input.getNumSamples() - pos));
        for (int ch = 0; ch < 2 && avail > 0; ++ch)
            buffer.copyFrom (ch, 0, input, 0, pos, avail);
        midi.clear();
        processor.processBlock (buffer, midi);
        for (int ch = 0; ch < 2; ++ch)
            output.copyFrom (ch, pos, buffer, ch, 0, n);

        if (realTimePacing)
            while (juce::Time::getMillisecondCounterHiRes() - start < 1000.0 * (pos + n) / sr)
                juce::Thread::sleep (1);
    }

    const double ms = juce::Time::getMillisecondCounterHiRes() - start;

    // Remove the plug-in latency so the render lines up with the input.
    juce::AudioBuffer<float> aligned (2, input.getNumSamples() + tail);
    for (int ch = 0; ch < 2; ++ch)
        aligned.copyFrom (ch, 0, output, ch, latency, aligned.getNumSamples());

    const float peak = aligned.getMagnitude (0, aligned.getNumSamples());
    if (peak > 0.99f)
        aligned.applyGain (0.99f / peak);

    const bool aiVoice = processor.getNeuralStatus() == voxa::VoxaVersaProcessor::NeuralStatus::Active
                         && state.getRawParameterValue (voxa::ParamID::aiEngine)->load() > 0.5f;
    if (! voxa::writeWav (outFile, aligned, sr,
                          voxa::disclosureMetadata (aiVoice ? voxa::VocalOrigin::AiVoiceModel : voxa::VocalOrigin::Transformed,
                                                    processor.getSelectedVoice().name)))
        return fail ("cannot write " + outFile.getFullPathName());

    const float inPitch = medianVoicedPitch (input.getReadPointer (0), input.getNumSamples(), sr);
    const float outPitch = medianVoicedPitch (aligned.getReadPointer (0), input.getNumSamples(), sr);
    const int key = processor.detectedKeyRoot.load();
    const int gender = processor.sourceGenderEstimate.load();

    std::printf ("%s -> %s\n", inFile.getFileName().toRawUTF8(), outFile.getFileName().toRawUTF8());
    std::printf ("  voice %s | latency %d | %.1f s audio in %.0f ms (%.1fx real time)\n",
                 processor.getSelectedVoice().name.toRawUTF8(), latency, input.getNumSamples() / sr, ms,
                 (input.getNumSamples() / sr * 1000.0) / juce::jmax (1.0, ms));
    std::printf ("  median pitch in %.1f Hz (%s) -> out %.1f Hz (%s), shift %+.2f st | source %s | key %s\n",
                 inPitch, inPitch > 0 ? voxa::music::noteName ((int) std::lround (voxa::music::hzToMidi (inPitch))).toRawUTF8() : "-",
                 outPitch, outPitch > 0 ? voxa::music::noteName ((int) std::lround (voxa::music::hzToMidi (outPitch))).toRawUTF8() : "-",
                 inPitch > 0 && outPitch > 0 ? 12.0 * std::log2 (outPitch / inPitch) : 0.0,
                 gender == 1 ? "male" : (gender == 2 ? "female" : "unknown"),
                 key >= 0 ? voxa::music::getKeyNames()[key].toRawUTF8() : "?");
    if (peak > 0.99f)
        std::printf ("  (normalised from peak %.2f)\n", peak);

    // Per-frame shift histogram (frames voiced in both input and output).
    {
        const int hop = 480;
        auto fin = voxa::dsp::PitchDetector::analyse (input.getReadPointer (0), input.getNumSamples(), sr, hop);
        auto fout = voxa::dsp::PitchDetector::analyse (aligned.getReadPointer (0), input.getNumSamples(), sr, hop);
        std::map<int, int> bins;
        int both = 0, inVoiced = 0, outVoiced = 0;
        for (size_t k = 0; k < juce::jmin (fin.size(), fout.size()); ++k)
        {
            inVoiced += fin[k].voiced ? 1 : 0;
            outVoiced += fout[k].voiced ? 1 : 0;
            if (fin[k].voiced && fout[k].voiced)
            {
                ++both;
                ++bins[(int) std::lround (12.0 * std::log2 (fout[k].frequencyHz / fin[k].frequencyHz))];
            }
        }
        if (auto dumpIndex = args.indexOf ("--dump"); dumpIndex >= 0 && dumpIndex + 1 < args.size())
        {
            juce::String csv ("time,in_hz,out_hz\n");
            for (size_t k = 0; k < juce::jmin (fin.size(), fout.size()); ++k)
                csv << juce::String ((double) k * hop / sr, 3) << "," << fin[k].frequencyHz << "," << fout[k].frequencyHz << "\n";
            juce::File::getCurrentWorkingDirectory().getChildFile (args[dumpIndex + 1]).replaceWithText (csv);
        }
        std::printf ("  voiced frames in %d out %d both %d | shift histogram:", inVoiced, outVoiced, both);
        for (auto& [st, count] : bins)
            if (count * 100 >= both * 2)
                std::printf (" %+d st %d%%", st, count * 100 / juce::jmax (1, both));
        std::printf ("\n");
    }
    return 0;
}
