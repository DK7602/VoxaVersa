#pragma once

#include "PitchDetector.h"
#include "PitchMapper.h"
#include "VoiceGenderEstimator.h"
#include "Psola.h"
#include "VocalFx.h"
#include "../Parameters.h"

namespace voxa::neural { class NeuralVoiceStream; }

namespace voxa::dsp
{

/** Everything the voice chain needs for one block (built from parameters + the selected voice). */
struct ChainSettings
{
    PitchMapper::Settings pitch;

    bool formantShift = true;
    float intensity = 0.7f;      // 0..1
    float gender = 0.0f;         // -1 male .. +1 female
    float age = 0.0f;            // -1 young .. +1 mature
    float tone = 0.0f;           // -1 bright .. +1 dark
    float style = 0.0f;          // -1 natural .. +1 creative
    bool vibe = true;
    float breath = 0.0f;         // 0..1 (user)
    float rasp = 0.0f;           // 0..1 (user)

    bool harmonyOn = false;
    HarmonyInterval harmonyInterval[3] { HarmonyInterval::ThirdUp, HarmonyInterval::FifthUp, HarmonyInterval::Off };
    float harmonyLevel[3] { 0.7f, 0.6f, 0.5f };
    float harmonyPan[3] { -0.5f, 0.5f, 0.0f };

    bool doublerOn = false;
    float doublerAmount = 0.5f;
    float doublerWidth = 0.8f;

    float lowCutHz = 70.0f;
    float deEsser = 0.25f;
    float compressor = 0.25f;
    float presenceDb = 0.0f;
    float airDb = 0.0f;
    float saturation = 0.0f;
    float delayMix = 0.0f;
    int delayDivision = 1;
    float delayFeedback = 0.3f;
    float reverbMix = 0.12f;
    float reverbSize = 0.5f;
    double bpm = 120.0;

    bool useNeural = false;      // lead voice comes from the neural stream when it has a model
    bool aiLead = false;         // the lead is an AI voice rendered elsewhere (offline: process's leadOverride)
};

/** The complete mono-in / stereo-out voice transformation chain. */
class VoiceChain
{
public:
    VoiceChain();
    ~VoiceChain();

    /** Allocates everything. Not real-time safe. */
    void prepare (double sampleRate, int maxBlockSize, Quality quality);
    void reset();
    /** Clears the reverb and delay tails (the transport stopped; learned pitch / key / gender state stays). */
    void clearTails();

    /** Optional neural lead voice (owned elsewhere, must outlive the chain or be cleared first). Its latency
        counts from now on, whether or not it has a model yet: hosts may ignore a latency that changes later
        (when the model has loaded), so it is reserved up front. Returns true when the chain's latency changed
        (the alignment buffers are then cleared; otherwise playback continues untouched). */
    bool setNeuralStream (neural::NeuralVoiceStream* stream);

    /** Latency of the DSP path alone, in samples. */
    int getDspLatency() const noexcept { return dspLatency; }
    /** Latency of the neural path (input-to-output) when a stream is attached, else 0. */
    int getNeuralLatency() const noexcept { return neuralLatency; }
    /** Total latency of the chain (constant until the next prepare / setNeuralStream). */
    int getLatencySamples() const noexcept { return totalLatency; }
    Quality getQuality() const noexcept { return quality; }
    /** True when the last processed block's lead voice came from the neural stream. */
    bool isNeuralLeadActive() const noexcept { return neuralLeadActive; }

    /** Processes one block. `leadOverride` (optional) replaces the lead voice with externally rendered
        audio already aligned to the chain's output timeline (used for offline neural rendering). */
    void process (const float* monoInput, float* outLeft, float* outRight, int numSamples,
                  const ChainSettings& settings, const float* leadOverride = nullptr) noexcept;

    /** Records the lead target pitch contour (Hz, 0 = unvoiced) at `frameRate`, aligned to input time, and
        optionally the pitch shift an AI voice applies to the input's own pitch (see neuralPitchShift()). */
    void setF0Capture (std::vector<float>* destination, double frameRate, std::vector<float>* shiftDestination = nullptr);

    /** The shift (semitones) an AI voice applies to the pitch its neural tracker hears in the input: the
        register and Pitch knob, plus pitch correction's small moves (an octave-wrong reading of the live
        detector can't leak in). NaN where the target pitch is not the input's (MIDI pitch control, synth guide). */
    static float neuralPitchShift (const PitchFrame&, const PitchMapper::Settings&) noexcept;

    // Telemetry (audio thread writes, UI reads through the processor)
    float getDetectedHz() const noexcept   { return detectedHz; }
    float getTargetHz() const noexcept     { return targetHz; }
    const PitchMapper& getMapper() const noexcept { return mapper; }

private:
    float computeFormantSemis (const ChainSettings&) const noexcept;
    /** dest[i] = 1 where the pitch detector heard the input voiced at input time `from + i`. */
    void fillInputVoicing (juce::int64 from, int n, std::vector<float>& dest) noexcept;

    double sr = 48000.0;
    int maxBlock = 512;
    Quality quality = Quality::Balanced;

    PitchDetector detector;
    VoiceGenderEstimator genderEstimator;
    PitchMapper mapper;
    PitchTimeline timeline;
    PsolaAnalyzer analyzer;
    PsolaVoice lead, harmony[3], doubler[2];
    bool harmonyActive[3] {}, doublerActive[2] {};

    int timeOffset = 0, placementLag = 0, dspLatency = 0, neuralLatency = 0, totalLatency = 0;
    bool neuralLeadActive = false;
    float maxHalfGrain = 1000.0f;
    juce::int64 blockStart = 0;

    ConsonantKeeper consonants;
    RKeeper rKeeper;
    BreathKeeper breaths;
    CharacterProcessor character;
    DeEsser deEsser;
    VocalCompressor compressor;
    Saturator saturator;
    StereoDelay delay;
    FdnReverb reverb;

    std::vector<float> leadBuf, voiceBuf, dryBuf, aheadBuf, voicedBuf, voicedNowBuf, f0Buf, shiftBuf, neuralIn, neuralOut, neuralCoverage, busL, busR;

    // Alignment delays towards totalLatency: DSP bus (stereo), neural lead (mono) and, while the neural lead
    // is active, the DSP lead that fills its gaps, the neural coverage that crossfades the two and the dry input
    // whose consonants the AI lead keeps.
    struct DelayLine
    {
        std::vector<float> buffer;
        int write = 0, delay = 0;
        void prepare (int maxDelay) { buffer.assign ((size_t) maxDelay + 1, 0.0f); write = 0; }
        void clear() { std::fill (buffer.begin(), buffer.end(), 0.0f); write = 0; }
        inline float process (float x) noexcept
        {
            if (delay <= 0 || buffer.empty()) return x;
            const int size = (int) buffer.size();
            buffer[(size_t) write] = x;
            int r = write - delay;
            if (r < 0) r += size;
            write = (write + 1) % size;
            return buffer[(size_t) r];
        }
    };
    DelayLine alignL, alignR, alignNeural, alignLead, alignCoverage, alignDry, alignAhead;
    int rLookAhead = 0;   // samples: the dry voice for the R keeper, RKeeper::lookAheadSeconds early

    // Doubler voices of an AI lead: copies of the AI singer itself, slightly late and slowly drifting in pitch
    // (a modulated delay), so the double has the lead's own timbre rather than the DSP voice's.
    struct ModulatedDelay
    {
        std::vector<float> buffer;
        int write = 0;
        void prepare (int maxDelay) { buffer.assign ((size_t) maxDelay + 2, 0.0f); write = 0; }
        void clear() { std::fill (buffer.begin(), buffer.end(), 0.0f); write = 0; }
        inline float process (float x, float delay) noexcept
        {
            const int size = (int) buffer.size();
            buffer[(size_t) write] = x;
            float r = (float) write - delay;
            if (r < 0.0f) r += (float) size;
            const int i0 = juce::jmin (size - 1, (int) r);
            const int i1 = (i0 + 1) % size;
            const float frac = r - (float) i0;
            write = (write + 1) % size;
            return buffer[(size_t) i0] + frac * (buffer[(size_t) i1] - buffer[(size_t) i0]);
        }
    };
    ModulatedDelay aiDoubler[2];
    bool aiDoublerActive = false;

    neural::NeuralVoiceStream* neural = nullptr;

    std::vector<float>* f0Capture = nullptr;
    std::vector<float>* shiftCapture = nullptr;
    double f0CaptureRate = 100.0;

    float formantSmoothed = 0.0f;
    float detectedHz = 0.0f, targetHz = 0.0f;
};

} // namespace voxa::dsp
