#include "NeuralPitch.h"
#include "NeuralDsp.h"
#include "../dsp/Biquad.h"
#include "../voices/VoiceLibrary.h"

#include <cmath>

namespace voxa::neural
{

namespace
{
    constexpr float voicedConfidence = 0.006f;   // FCPE's own threshold
    constexpr float hissRatioDb = 6.0f;          // hiss (3.5-8 kHz) this far above the body (< 1.5 kHz): unvoiced
    constexpr int blockFrames = 1000;            // long clips: ~10 s per inference
}

juce::File NeuralPitchTracker::getDefaultFile()
{
    return VoiceLibrary::getEncodersDirectory().getChildFile ("fcpe.onnx");
}

std::shared_ptr<NeuralPitchTracker> NeuralPitchTracker::load (juce::String& error, const juce::File& file)
{
    if (! file.existsAsFile() || ! OnnxRuntime::getInstance().isAvailable())
        return nullptr;

    // Small and fast: always on the CPU (GPU providers serialise runs and add transfer overhead).
    auto session = OnnxRuntime::getInstance().getImpl().getSharedSession (file, error, false);

    if (session == nullptr)
        return nullptr;

    const auto& inputs = session->getInputs();
    const auto& outputs = session->getOutputs();

    if (inputs.size() != 1 || outputs.size() != 2 || inputs[0].rank() != 2)
    {
        error = file.getFileName() + ": not a VoxaVersa pitch tracker (export it with tools/training/export_pitch.py)";
        return nullptr;
    }

    auto tracker = std::make_shared<NeuralPitchTracker>();
    tracker->session = std::move (session);
    tracker->inputName = inputs[0].name;

    for (const auto& o : outputs)
        (o.name == "confidence" ? tracker->confidenceName : tracker->f0Name) = o.name;

    if (tracker->f0Name.empty() || tracker->confidenceName.empty())
    {
        error = file.getFileName() + ": expected outputs f0 and confidence";
        return nullptr;
    }

    return tracker;
}

bool NeuralPitchTracker::runBlock (const float* audio16k, int numFrames, Workspace& ws, juce::String& error) const
{
    InputTensor input;
    input.name = inputName.c_str();
    input.data = audio16k;
    input.numBytes = (size_t) numFrames * dsp::samplesPerFrame16k * sizeof (float);
    input.shape = { 1, (int64_t) numFrames * dsp::samplesPerFrame16k };
    input.rank = 2;

    const char* names[] = { f0Name.c_str(), confidenceName.c_str() };
    OutputTensor outputs[2];
    outputs[0].data.swap (ws.f0.data);
    outputs[1].data.swap (ws.confidence.data);

    const bool ok = session->run (&input, 1, names, outputs, 2, false, error);
    ws.f0.data.swap (outputs[0].data);
    ws.confidence.data.swap (outputs[1].data);

    if (ok && (ws.f0.data.size() < (size_t) numFrames || ws.confidence.data.size() < (size_t) numFrames))
    {
        error = "pitch tracker returned " + juce::String ((int64_t) ws.f0.data.size()) + " frames for " + juce::String (numFrames);
        return false;
    }

    return ok;
}

bool NeuralPitchTracker::track (const float* audio16k, int numFrames, float* f0, Workspace& ws, juce::String& error) const
{
    if (numFrames <= 0)
        return true;

    // Blocks of up to ~10 s with contextFrames of real audio either side (the network is convolution-only).
    for (int start = 0; start < numFrames; start += blockFrames)
    {
        const int end = juce::jmin (numFrames, start + blockFrames);
        const int from = juce::jmax (0, start - contextFrames);
        const int to = juce::jmin (numFrames, end + contextFrames);

        if (to - from < 2)
        {
            std::fill (f0 + start, f0 + end, 0.0f);
            continue;
        }

        if (! runBlock (audio16k + (size_t) from * dsp::samplesPerFrame16k, to - from, ws, error))
            return false;

        for (int k = start; k < end; ++k)
        {
            const float hz = ws.f0.data[(size_t) (k - from)];
            f0[k] = ws.confidence.data[(size_t) (k - from)] > voicedConfidence && std::isfinite (hz)
                      ? juce::jlimit (dsp::f0MinHz, dsp::f0MaxHz, hz) : 0.0f;
        }
    }

    // Hiss gate: energy above 3.5 kHz against the body below 1.5 kHz, per frame.
    voxa::dsp::Biquad hiss[2], body;
    hiss[0].c = hiss[1].c = voxa::dsp::BiquadCoeffs::highPass (dsp::encoderSampleRate, 3500.0);
    body.c = voxa::dsp::BiquadCoeffs::lowPass (dsp::encoderSampleRate, 1500.0);

    for (int k = 0; k < numFrames; ++k)
    {
        double eHiss = 1.0e-12, eBody = 1.0e-12;

        for (int i = 0; i < dsp::samplesPerFrame16k; ++i)
        {
            const float x = audio16k[(size_t) k * dsp::samplesPerFrame16k + (size_t) i];
            const float h = hiss[1].process (hiss[0].process (x));
            const float b = body.process (x);
            eHiss += (double) h * h;
            eBody += (double) b * b;
        }

        if (10.0 * std::log10 (eHiss / eBody) > hissRatioDb)
            f0[k] = 0.0f;
    }

    return true;
}

void applyTrackedPitch (const float* tracked, const float* shiftSemis, float* target, int numFrames) noexcept
{
    for (int k = 0; k < numFrames; ++k)
        if (! std::isnan (shiftSemis[k]))
            target[k] = tracked[k] > 0.0f ? juce::jlimit (dsp::f0MinHz, dsp::f0MaxHz, tracked[k] * std::exp2 (shiftSemis[k] / 12.0f))
                                          : 0.0f;
}

} // namespace voxa::neural
