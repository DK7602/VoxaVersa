#pragma once

#include "VoiceChain.h"
#include <functional>
#include <memory>

namespace voxa::neural { class NeuralVoiceModel; }

namespace voxa::dsp
{

/** What one offline pass of the voice needs (a snapshot of the settings, taken on the message thread). */
struct VoiceRenderJob
{
    ChainSettings settings;                              // as live (renderVoice sets useNeural and aiLead)
    std::shared_ptr<neural::NeuralVoiceModel> model;     // the AI singer, or nullptr for the DSP voice
    int accent = 0;                                      // SouthernVowels::Accent of an AI lead
    double sampleRate = 48000.0;
    double tailSeconds = 3.0;                            // reverb / delay ring-out after the source
};

/** Runs `source` (mono) through a fresh voice chain at the best quality: the DSP at High Quality and, with a model,
    the AI singer converting the whole clip at once with context on both sides (NeuralVoiceModel::convert) instead
    of the live stream's short chunks. `wet` gets two channels, the source's length plus the tail, on the source's
    own timeline (the chain's latency is removed). `progress` (0..1) is called now and then; `cancel` is checked
    between steps. If the AI conversion fails the DSP voice is rendered and `error` says why. Returns false only
    when cancelled. Any thread except the audio thread. */
bool renderVoice (const juce::AudioBuffer<float>& source, const VoiceRenderJob& job, const std::atomic<bool>& cancel,
                  const std::function<void (float)>& progress, juce::AudioBuffer<float>& wet, juce::String& error);

} // namespace voxa::dsp
