// VoxaVersaRender: renders an audio file through the complete VoxaVersa processor offline.
//
//   VoxaVersaRender <in.wav> <out.wav> [--voice <id>] [--preset "<name>"] [--quality 0|1|2]
//                   [--set <paramId>=<value> ...] [--list-voices] [--list-params]
//                   [--realtime | --host-offline]
//   VoxaVersaRender <file> --analyse          source gender analysis
//   VoxaVersaRender <file> --detect           look for the VoxaVersa AI / synthetic-singing watermark
//
// Values for --set are real parameter values (e.g. pitch=5, gender=60, tuneOn=1, key=1 for C).
// --realtime feeds the blocks at the pace of live playback (what a DAW does while playing);
// --host-offline flags the render as non-real-time, like a DAW export or Render in Place.
// Prints source/target pitch statistics so renders can be compared objectively.

#include "PluginProcessor.h"
#include "core/AiDisclosure.h"
#include "core/AudioUtils.h"
#include "core/Watermark.h"
#include "dsp/PitchDetector.h"
#include "dsp/VoiceGenderEstimator.h"
#include <map>

namespace
{
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
            if (processor.getVoiceLibrary().findById (id) == nullptr)
                return fail ("unknown voice " + id + " (see --list-voices)");
            processor.selectVoice (id);
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
    for (int i = 0; i < 1200 && processor.getNeuralStatus() == voxa::VoxaVersaProcessor::NeuralStatus::Loading; ++i)
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
