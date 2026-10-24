// TEMPORARY STUB - replaced by the ONNX Runtime implementation.
#include "NeuralVoice.h"

namespace voxa::neural
{

struct OnnxRuntime::Impl {};
OnnxRuntime::OnnxRuntime() : impl (std::make_unique<Impl>()) {}
OnnxRuntime::~OnnxRuntime() = default;
OnnxRuntime& OnnxRuntime::getInstance() { static OnnxRuntime instance; return instance; }
bool OnnxRuntime::ensureLoaded() { return false; }
bool OnnxRuntime::isAvailable() const noexcept { return false; }
juce::String OnnxRuntime::getStatusMessage() const { return "Neural engine not built"; }
juce::File OnnxRuntime::getLoadedLibraryFile() const { return {}; }
juce::Array<juce::File> OnnxRuntime::getSearchLocations() { return {}; }

struct NeuralVoiceModel::Impl {};
NeuralVoiceModel::NeuralVoiceModel() : impl (std::make_unique<Impl>()) {}
NeuralVoiceModel::~NeuralVoiceModel() = default;
std::shared_ptr<NeuralVoiceModel> NeuralVoiceModel::load (const VoiceProfile&, juce::String& error)
{
    error = "Neural engine not built";
    return nullptr;
}
int NeuralVoiceModel::getModelSampleRate() const noexcept { return 0; }
juce::String NeuralVoiceModel::getDescription() const { return {}; }
bool NeuralVoiceModel::convert (const juce::AudioBuffer<float>&, double, const std::vector<float>&, double,
                                juce::AudioBuffer<float>&, const std::atomic<bool>*, juce::String& error) const
{
    error = "Neural engine not built";
    return false;
}

struct NeuralVoiceStream::Impl {};
NeuralVoiceStream::NeuralVoiceStream() : impl (std::make_unique<Impl>()) {}
NeuralVoiceStream::~NeuralVoiceStream() = default;
void NeuralVoiceStream::prepare (double, int) {}
void NeuralVoiceStream::setModel (std::shared_ptr<NeuralVoiceModel>) {}
bool NeuralVoiceStream::hasModel() const noexcept { return false; }
int NeuralVoiceStream::getLatencySamples() const noexcept { return 0; }
bool NeuralVoiceStream::isRunningInRealTime() const noexcept { return false; }
void NeuralVoiceStream::process (const float*, const float*, float* output, int numSamples) noexcept
{
    juce::FloatVectorOperations::clear (output, numSamples);
}
void NeuralVoiceStream::reset() noexcept {}

} // namespace voxa::neural
