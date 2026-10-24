// TEMPORARY STUB - replaced by the full text-to-sing implementation.
#include "SingEngine.h"

namespace voxa::sing
{

struct SingEngine::Impl {};
SingEngine::SingEngine() : impl (std::make_unique<Impl>()) {}
SingEngine::~SingEngine() = default;

SingResult SingEngine::render (const SingRequest& request, const std::atomic<bool>*, std::function<void (float)>)
{
    SingResult r;
    r.sampleRate = request.sampleRate;
    r.errorMessage = "Text-to-Sing engine not built yet";
    return r;
}

int SingEngine::countSyllables (const juce::String& lyrics)
{
    return juce::StringArray::fromTokens (lyrics, " \n\r\t-", "").size();
}

std::vector<Note> SingEngine::melodyFromMidiFile (const juce::File&, double& bpmOut, juce::String& error)
{
    bpmOut = 120.0;
    error = "Not implemented";
    return {};
}

std::vector<Note> SingEngine::melodyFromSequence (const juce::MidiMessageSequence&) { return {}; }
std::vector<Note> SingEngine::melodyFromAudio (const juce::AudioBuffer<float>&, double, double) { return {}; }
juce::MidiFile SingEngine::toMidiFile (const SingResult&, double) { return {}; }

struct LiveSinger::Impl {};
LiveSinger::LiveSinger() : impl (std::make_unique<Impl>()) {}
LiveSinger::~LiveSinger() = default;
void LiveSinger::prepare (double, int) {}
void LiveSinger::reset() {}
void LiveSinger::setLyrics (const juce::String&) {}
void LiveSinger::setVoice (const VoiceTraits&) {}
void LiveSinger::noteOn (int, float) {}
void LiveSinger::noteOff (int) {}
void LiveSinger::allNotesOff() {}
void LiveSinger::restartLyrics() {}
void LiveSinger::render (float* output, int numSamples) { juce::FloatVectorOperations::clear (output, numSamples); }
int LiveSinger::getCurrentSyllableIndex() const noexcept { return 0; }
int LiveSinger::getNumSyllables() const noexcept { return 0; }
bool LiveSinger::isSounding() const noexcept { return false; }

} // namespace voxa::sing
