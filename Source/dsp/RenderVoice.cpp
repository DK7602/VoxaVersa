#include "RenderVoice.h"
#include "../neural/RvcModel.h"

namespace voxa::dsp
{

bool renderVoice (const juce::AudioBuffer<float>& source, const VoiceRenderJob& job, const std::atomic<bool>& cancel,
                  const std::function<void (float)>& progress, juce::AudioBuffer<float>& wet, juce::String& error)
{
    constexpr int block = 1024;
    const double sr = job.sampleRate;
    const auto report = [&progress] (float p) { if (progress) progress (juce::jlimit (0.0f, 1.0f, p)); };

    VoiceChain chain;
    chain.prepare (sr, block, Quality::HighQuality);   // latency is irrelevant offline
    const int latency = chain.getLatencySamples();
    const int length = source.getNumSamples() + (int) (sr * job.tailSeconds);
    const int total = length + latency;

    juce::AudioBuffer<float> padded (1, total);
    padded.clear();
    padded.copyFrom (0, 0, source, 0, 0, source.getNumSamples());

    // AI lead: pass 1 records the target pitch contour, then the model converts the whole clip.
    juce::AudioBuffer<float> neuralLead;
    std::vector<float> neuralAccent;   // per sample of neuralLead: how much it was given the accent's vowel
    const auto& model = job.model;
    auto settings = job.settings;
    settings.useNeural = false;              // offline the model converts the whole clip, not the live stream
    settings.aiLead = model != nullptr;      // pass 1: the AI lead keeps the singer's own vibrato, breath and register
    if (model != nullptr)
    {
        std::vector<float> f0, shift;
        f0.reserve ((size_t) (total / sr * 100.0) + 16);
        shift.reserve (f0.capacity());
        chain.setF0Capture (&f0, 100.0, &shift);
        juce::AudioBuffer<float> scratch (2, block);
        for (int pos = 0; pos < total && ! cancel.load(); pos += block)
        {
            const int n = juce::jmin (block, total - pos);
            chain.process (padded.getReadPointer (0, pos), scratch.getWritePointer (0), scratch.getWritePointer (1), n, settings);
            report (0.3f * (float) pos / (float) total);
        }
        chain.setF0Capture (nullptr, 100.0);

        juce::AudioBuffer<float> converted;
        std::vector<float> accentFrames;
        if (! cancel.load() && model->convert (source, sr, f0, 100.0, converted, &cancel, error, &shift, job.accent,
                                               job.accent != 0 ? &accentFrames : nullptr))
        {
            // Aligned to the chain's output timeline (its DSP latency).
            neuralLead.setSize (1, total);
            neuralLead.clear();
            const int copyLen = juce::jmin (converted.getNumSamples(), total - latency);
            if (copyLen > 0)
                neuralLead.copyFrom (0, latency, converted, 0, 0, copyLen);
            if (! accentFrames.empty())
            {
                neuralAccent.assign ((size_t) total, 0.0f);
                for (int i = 0; i < copyLen; ++i)
                    neuralAccent[(size_t) (latency + i)] = accentFrames[(size_t) juce::jmin ((int) accentFrames.size() - 1,
                                                                                             (int) ((double) i * 100.0 / sr))];
            }
        }
        chain.reset();
        report (0.55f);
    }

    settings.aiLead = neuralLead.getNumSamples() > 0;   // the DSP voice if the conversion failed
    const float start = model != nullptr ? 0.55f : 0.0f;
    juce::AudioBuffer<float> out (2, total);
    for (int pos = 0; pos < total && ! cancel.load(); pos += block)
    {
        const int n = juce::jmin (block, total - pos);
        chain.process (padded.getReadPointer (0, pos), out.getWritePointer (0, pos), out.getWritePointer (1, pos), n, settings,
                       settings.aiLead ? neuralLead.getReadPointer (0, pos) : nullptr,
                       settings.aiLead && ! neuralAccent.empty() ? neuralAccent.data() + pos : nullptr);
        report (start + (1.0f - start) * (float) pos / (float) total);
    }
    if (cancel.load())
        return false;

    wet.setSize (2, length);
    for (int ch = 0; ch < 2; ++ch)
        wet.copyFrom (ch, 0, out, ch, latency, length);
    report (1.0f);
    return true;
}

} // namespace voxa::dsp
