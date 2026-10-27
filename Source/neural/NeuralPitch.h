#pragma once

#include "NeuralRuntime.h"

#include <memory>
#include <vector>

namespace voxa::neural
{

/** The AI voices' neural pitch tracker: FCPE (CNChTu/FCPE, MIT; the network RVC's real-time mode uses),
    exported with its spectrogram by tools/training/export_pitch.py and installed as <Encoders>/fcpe.onnx.

    VoxaVersa's live pitch detector (YIN) drives the DSP voices and the pitch mapping, but in runs and fast
    slides its periodicity measure fails: measured on R&B vocals it went silent on 12-35% of the frames of
    runs (0-3% on held notes), so the AI singer was told "no note" in the middle of runs. FCPE stays voiced
    there (1-6%) and follows the notes; AI voices sing the pitch it tracks, shifted by VoxaVersa's mapping.
    Without the file, everything works as before. */
class NeuralPitchTracker
{
public:
    /** <Encoders>/fcpe.onnx. */
    static juce::File getDefaultFile();

    /** Loads the installed tracker (its session is shared by every voice). nullptr when it is not installed or
        fails to load (`error` then says why; empty when the file is simply not there). */
    static std::shared_ptr<NeuralPitchTracker> load (juce::String& error, const juce::File& file = getDefaultFile());

    struct Workspace
    {
        OutputTensor f0, confidence;
    };

    /** Pitch (Hz, 0 = unvoiced) of each 10 ms frame of `audio16k` (numFrames * 160 samples at 16 kHz; frame k is
        samples [160k, 160k + 160)). Long clips are tracked in overlapping blocks. Frames dominated by hiss
        ("s", "sh", "f": the network calls ~20% of them voiced) are unvoiced, so the AI singer doesn't buzz them. */
    bool track (const float* audio16k, int numFrames, float* f0, Workspace&, juce::String& error) const;

    /** Frames with less than 50 ms of audio either side match a whole-clip run less well; the stream's newest
        frames have none after them (measured: 2.6-4% of the frames of runs then differ by > 50 cents, 0.6% with
        100 ms; still far better than YIN). */
    static constexpr int contextFrames = 50;

private:
    bool runBlock (const float* audio16k, int numFrames, Workspace&, juce::String& error) const;

    std::shared_ptr<OrtModelSession> session;
    std::string inputName, f0Name, confidenceName;
};

/** The target pitch of each frame: the tracked pitch shifted by `shiftSemis`. Frames whose shift is NaN keep
    `target` as it is (the caller decides their pitch, e.g. MIDI pitch control). */
void applyTrackedPitch (const float* tracked, const float* shiftSemis, float* target, int numFrames) noexcept;

} // namespace voxa::neural
