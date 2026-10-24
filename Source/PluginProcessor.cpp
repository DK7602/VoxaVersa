// TEMPORARY SKELETON - the full processor implementation replaces this file.
#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace voxa
{

struct VoxaVersaProcessor::Impl {};

VoxaVersaProcessor::VoxaVersaProcessor()
    : AudioProcessor (BusesProperties()
                        .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                        .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      impl (std::make_unique<Impl>()),
      apvts (*this, &undoManager, "VoxaVersa", createParameterLayout()),
      presetManager (apvts)
{
    apvts.state.setProperty (StateID::voiceId, "female-pop", nullptr);
}

VoxaVersaProcessor::~VoxaVersaProcessor() = default;

bool VoxaVersaProcessor::isInstrumentBuild() noexcept { return JucePlugin_IsSynth != 0; }
const juce::String VoxaVersaProcessor::getName() const { return JucePlugin_Name; }

void VoxaVersaProcessor::prepareToPlay (double sr, int)
{
    currentSampleRate = sr;
    inputScope.prepare (sr);
    outputScope.prepare (sr);
}

void VoxaVersaProcessor::releaseResources() {}

bool VoxaVersaProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::stereo() || out == juce::AudioChannelSet::mono();
}

void VoxaVersaProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    inputScope.push (buffer.getReadPointer (0), buffer.getNumSamples());
    inputMeter.update (buffer.getArrayOfReadPointers(), buffer.getNumChannels(), buffer.getNumSamples());
    outputScope.push (buffer.getReadPointer (0), buffer.getNumSamples());
    outputMeter.update (buffer.getArrayOfReadPointers(), buffer.getNumChannels(), buffer.getNumSamples());
}

juce::AudioProcessorEditor* VoxaVersaProcessor::createEditor() { return new VoxaVersaEditor (*this); }

int VoxaVersaProcessor::getNumPrograms() { return 1; }
int VoxaVersaProcessor::getCurrentProgram() { return 0; }
void VoxaVersaProcessor::setCurrentProgram (int) {}
const juce::String VoxaVersaProcessor::getProgramName (int) { return {}; }

void VoxaVersaProcessor::getStateInformation (juce::MemoryBlock& dest)
{
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary (*xml, dest);
}

void VoxaVersaProcessor::setStateInformation (const void* data, int size)
{
    if (auto xml = getXmlFromBinary (data, size))
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::String VoxaVersaProcessor::getSelectedVoiceId() const { return apvts.state.getProperty (StateID::voiceId).toString(); }
const VoiceProfile& VoxaVersaProcessor::getSelectedVoice() const
{
    if (auto* v = voiceLibrary.findById (getSelectedVoiceId())) return *v;
    return voiceLibrary.getDefaultVoice();
}
void VoxaVersaProcessor::selectVoice (const juce::String& id) { apvts.state.setProperty (StateID::voiceId, id, &undoManager); }
juce::String VoxaVersaProcessor::getBlendVoiceId() const { return apvts.state.getProperty (StateID::blendVoiceId).toString(); }
void VoxaVersaProcessor::setBlendVoice (const juce::String& id) { apvts.state.setProperty (StateID::blendVoiceId, id, &undoManager); }
juce::String VoxaVersaProcessor::getNeuralStatusMessage() const { return "DSP engine"; }

bool VoxaVersaProcessor::loadSample (const juce::File&, juce::String& e) { e = "Not implemented"; return false; }
void VoxaVersaProcessor::clearSample() {}
bool VoxaVersaProcessor::hasSample() const noexcept { return false; }
juce::String VoxaVersaProcessor::getSampleName() const { return {}; }
double VoxaVersaProcessor::getSampleLengthSeconds() const noexcept { return 0.0; }
void VoxaVersaProcessor::setSamplePlaying (bool) {}
double VoxaVersaProcessor::getSamplePlayPosition() const noexcept { return 0.0; }

void VoxaVersaProcessor::startOfflineRender (RenderTarget) {}
void VoxaVersaProcessor::cancelOfflineRender() {}
juce::File VoxaVersaProcessor::getLastRenderedFile() const { return {}; }
juce::String VoxaVersaProcessor::getLastRenderError() const { return {}; }
juce::File VoxaVersaProcessor::getExportDirectory() { return juce::File::getSpecialLocation (juce::File::tempDirectory); }

void VoxaVersaProcessor::setLyrics (const juce::String& l) { apvts.state.setProperty (StateID::lyrics, l.substring (0, maxLyricsLength), nullptr); }
juce::String VoxaVersaProcessor::getLyrics() const { return apvts.state.getProperty (StateID::lyrics).toString(); }
bool VoxaVersaProcessor::loadMelodyMidiFile (const juce::File&, juce::String& e) { e = "Not implemented"; return false; }
juce::String VoxaVersaProcessor::getMelodyFileName() const { return {}; }
void VoxaVersaProcessor::setMidiCaptureEnabled (bool b) { midiCapture = b; }
int VoxaVersaProcessor::getNumCapturedNotes() const { return 0; }
void VoxaVersaProcessor::clearCapturedMidi() {}
void VoxaVersaProcessor::setHumRecording (bool b) { humRecording = b; }
double VoxaVersaProcessor::getHumRecordedSeconds() const noexcept { return 0.0; }
void VoxaVersaProcessor::generateSinging() {}
void VoxaVersaProcessor::cancelSinging() {}
juce::String VoxaVersaProcessor::getSingError() const { return {}; }
std::vector<sing::SungNote> VoxaVersaProcessor::getSungNotes() const { return {}; }
double VoxaVersaProcessor::getSingTakeLengthSeconds() const { return 0.0; }
void VoxaVersaProcessor::setSingPreviewPlaying (bool b) { singPreviewPlaying = b; }
double VoxaVersaProcessor::getSingPlayPosition() const noexcept { return 0.0; }
juce::File VoxaVersaProcessor::exportSingMidi() { return {}; }
int VoxaVersaProcessor::getLiveSyllableIndex() const noexcept { return 0; }

void VoxaVersaProcessor::startRecording() {}
void VoxaVersaProcessor::stopRecording() {}
double VoxaVersaProcessor::getRecordingSeconds() const noexcept { return 0.0; }
juce::File VoxaVersaProcessor::getLastRecording() const { return {}; }

void VoxaVersaProcessor::valueTreePropertyChanged (juce::ValueTree&, const juce::Identifier&) {}
void VoxaVersaProcessor::valueTreeRedirected (juce::ValueTree&) {}
void VoxaVersaProcessor::handleAsyncUpdate() {}
void VoxaVersaProcessor::updateVoiceTraits() {}
void VoxaVersaProcessor::updateNeuralModel() {}
void VoxaVersaProcessor::updateLatency() {}

} // namespace voxa

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new voxa::VoxaVersaProcessor();
}
