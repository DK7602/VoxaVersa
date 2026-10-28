#pragma once

#include "Biquad.h"
#include <vector>
#include <memory>
#include <juce_dsp/juce_dsp.h>

namespace voxa::dsp
{

/** Voice-character processing on the stereo voice bus: low cut, tonal shaping, breath and rasp. */
class CharacterProcessor
{
public:
    struct Settings
    {
        float lowCutHz = 70.0f;
        float tiltDb = 0.0f;       // + brighter
        float warmthDb = 0.0f;
        float nasalDb = 0.0f;
        float presenceDb = 0.0f;
        float airDb = 0.0f;
        float ringHz = 3000.0f, ringDb = 0.0f;   // voice colour: upper resonance
        float bodyHz = 250.0f, bodyDb = 0.0f;    // voice colour: lower resonance
        float breath = 0.0f;       // 0..1
        float rasp = 0.0f;         // 0..1
    };

    void prepare (double sampleRate);
    void reset();
    void process (float* left, float* right, int numSamples, const Settings&) noexcept;

private:
    void updateFilters (const Settings&) noexcept;

    double sr = 48000.0;
    Settings current;
    bool filtersValid = false;

    struct Channel
    {
        Biquad hp1, hp2, lowShelf, highShelf, warmth, nasal, presence, air, ring, body;
        Biquad breathHp, breathLp;
        Noise noise;
    } ch[2];

    float envelope = 0.0f, envAttack = 0.0f, envRelease = 0.0f;
    Biquad raspLp;
    Noise raspNoise;
};

/** Split-band de-esser (stereo linked). */
class DeEsser
{
public:
    void prepare (double sampleRate);
    void reset();
    void process (float* left, float* right, int numSamples, float amount) noexcept;
    float getReductionDb() const noexcept { return reductionDb; }

private:
    double sr = 48000.0;
    Biquad hp[2];
    float env = 0.0f, gain = 1.0f, attack = 0.0f, release = 0.0f, gainSmooth = 0.0f;
    float reductionDb = 0.0f;
};

/** Keeps the singer's own unvoiced consonants ("s", "sh", "t", "k", "f") in an AI lead voice.
    Voice conversion re-synthesises them from noise and smears them down toward "sh" and "th": a lisp. They carry
    almost no voice identity, so where the input hisses (or the AI hisses over a sound the input did not voice)
    the AI voice above ~2.5 kHz is replaced by the input's, level-matched to the AI voice. */
class ConsonantKeeper
{
public:
    void prepare (double sampleRate);
    void reset();
    /** `lead` (the AI voice, mono) is processed in place; `input` is the dry voice, aligned with it. */
    void process (float* lead, const float* input, int numSamples) noexcept;

private:
    double sr = 48000.0;
    // Linkwitz-Riley split (low + high sum to an all-pass, so the AI voice is untouched where nothing is kept)
    Biquad leadLow[2], leadHigh[2], inputHigh[2];
    // hiss detectors: energy above 3.5 kHz against below 1.5 kHz
    Biquad inputHiss[2], inputBody, leadHiss[2], leadBody;
    float inHiss = 0.0f, inBody = 0.0f, aiHiss = 0.0f, aiBody = 0.0f, envCoeff = 0.0f;
    float weight = 0.0f, attack = 0.0f, release = 0.0f;
    // voiced levels of both voices, to match the kept consonants to the AI voice's loudness
    float leadLevel = 0.0f, inputLevel = 0.0f, levelCoeff = 0.0f, gain = 1.0f;
};

/** Keeps an AI lead voice's breaths (and other quiet, unvoiced sounds) as loud as the singer's own. RVC renders a
    breath the way its training singer breathed: measured on a user's song through Female Pop, breaths came out up
    to 13 dB louder against the singing than the singer's own, and brighter. Where the input is unvoiced and more
    than 15 dB below its sung level, the AI voice is turned down to the input's level relative to the singing
    (+1.5 dB of leeway); it is never turned up, and sung notes are untouched. It runs after the voice's EQ and the
    compressor, which lift breaths too. */
class BreathKeeper
{
public:
    void prepare (double sampleRate);
    void reset();
    /** The voice (`left`, and `right` unless it is nullptr) is processed in place; `input` is the dry voice
        aligned with it and `inputVoiced` (> 0.5) where the pitch detector hears that input voiced, per sample. */
    void process (float* left, float* right, const float* input, const float* inputVoiced, int numSamples) noexcept;
    float getGain() const noexcept { return gain; }

private:
    float inEnv = 0.0f, outEnv = 0.0f, inSung = 0.0f, outSung = 0.0f;   // mean squares: now, and while sung
    float envCoeff = 0.0f, sungCoeff = 0.0f, gain = 1.0f, attack = 0.0f, release = 0.0f;
};

/** Gives an AI lead voice the input's American "r". An r is sung with the third formant (F3) dipping ~20 - 25 %
    below the vowels around it; the AI singers barely follow (measured on a user's song: the dry R at ~1.98 kHz
    against 2.56 kHz on the other words, the AI voice 2.50 against 2.95 kHz), so "friend" drifts toward "fwend".
    F3 of the dry voice and of the AI voice are tracked (LPC, 25 ms frames every 5 ms); where the dry voice's F3
    dips, the AI voice gets a resonance at its own F3 lowered by the same proportion, and its F3 is cut.
    Which LPC pole is "the third formant" depends on the analysis ceiling (Praat's maximum formant: 5000 Hz for
    men, 5500 Hz for women); the wrong one confuses F3 with F2 or F4 on a third of the frames, so each voice is
    analysed with its own (setCeilings). */
class RKeeper
{
public:
    void prepare (double sampleRate);
    void reset();
    /** `lead` (the AI voice, mono) is processed in place; `input` is the dry voice lookAheadSeconds ahead of it
        and `inputVoiced` (> 0.5) where the pitch detector hears that input voiced, per sample. */
    void process (float* lead, const float* input, const float* inputVoiced, int numSamples) noexcept;

    /** How early the dry voice should arrive: the analysis looks back half its 25 ms window and the 3-frame
        median 5 ms more, so an r lands on time (measured: the r's F3 then ~140 Hz lower than without). */
    static constexpr double lookAheadSeconds = 0.0175;

    static constexpr double maleCeilingHz = 5000.0, femaleCeilingHz = 5500.0;
    /** The analysis ceilings of the input and the AI voice (maleCeilingHz or femaleCeilingHz); a change restarts
        that voice's analysis. */
    void setCeilings (double inputHz, double leadHz) noexcept;

    /** Third-formant tracker, as Praat measures it: the audio resampled to twice the ceiling, LPC (order 10, five
        formants) over 25 ms, the poles' frequencies. */
    struct FormantTracker
    {
        static constexpr double maxCeilingHz = 6000.0;
        void prepare (double sampleRate, double ceilingHz);
        /** A new ceiling (up to maxCeilingHz) without allocating. */
        void setCeiling (double ceilingHz) noexcept;
        double getCeiling() const noexcept { return ceiling; }
        void reset();
        /** One full-rate sample. */
        void push (float x) noexcept;
        /** F3 (Hz, 0 when none was found) and level of the latest 25 ms. */
        void analyse() noexcept;
        float f3 = 0.0f, levelDb = -100.0f;

    private:
        double sr = 48000.0, ceiling = maleCeilingHz, rate = 10000.0, step = 0.2, phase = 0.0;
        float previousInput = 0.0f;
        int window = 250, write = 0;
        Biquad aa[4];
        std::vector<float> ring, frame, hamming;
    };

    static constexpr float maxGainDb = 15.0f;
    static constexpr float maxDepth = 0.35f;   // deeper dips (a mis-tracked frame) don't push the resonance lower

    /** Frames analysed while the input was sung, and those with an r applied (weight > 0.1), since reset(). */
    struct Stats { int sungFrames = 0, rFrames = 0; float dryBase = 0.0f, aiBase = 0.0f; };
    const Stats& getStats() const noexcept { return stats; }

private:
    float baseline (std::vector<float>& history, int count, std::vector<float>& scratch) const;

    float steady (float* recent, float latest) const noexcept;

    double sr = 48000.0;
    FormantTracker dry, ai;
    float dryRecent[3] {}, aiRecent[3] {};   // the last frames' F3: a median of 3 ignores one-frame slips
    Biquad boost, cut;
    // the filters glide toward the frame's targets, their coefficients updated every `updateEvery` samples
    float wantedWeight = 0.0f, boostTarget = 2000.0f, cutTarget = 2500.0f, logBoost = 7.6f, logCut = 7.8f;
    float glide = 0.0f, rise = 0.0f, fall = 0.0f;
    int updateEvery = 16, untilUpdate = 0;
    std::vector<float> dryHistory, aiHistory, scratch;   // F3 of recent voiced frames (~1.5 s)
    int dryCount = 0, aiCount = 0, frames = 0;
    float dryBase = 0.0f, aiBase = 0.0f, peakDb = -100.0f, weight = 0.0f;
    int voicedSamples = 0, frameSamples = 0, hop = 240, sinceFrame = 0;
    Stats stats;
};

/** Keeps the AI singer's vowels on the input's: an AI model renders some vowels "cleaner" than they were sung
    (the lax i of "quits" turned toward the ee of "quates", F2 ~240 Hz high and F1 low on every singer). Each 5 ms
    it compares the spectral envelopes of the dry voice and the AI voice (the strongest harmonic within half a
    fundamental of each band, the input's bands scaled by the vocal-tract ratio) and moves the AI voice's envelope
    toward the input's frame-to-frame changes, each measured from its own long-term average: the singer keeps
    their own timbre and follows the input's vowels. A bank of 1/6-octave peaking filters (250 Hz - 4.5 kHz) does
    it with no latency; the analysis looks at the last 21 ms of both voices, which is why the input must be on
    the AI voice's timeline. */
class VowelKeeper
{
public:
    void prepare (double sampleRate);
    void reset();
    /** `lead` (the AI voice, mono) is processed in place; `input` is the dry voice on the same timeline. `inputHz`
        and `leadHz` are both voices' pitches per sample (0 where unvoiced). `tractRatio`: the AI singer's
        formants over the input's (1 for the same gender, ~1.17 for a man's input into a woman's voice). */
    void process (float* lead, const float* input, const float* inputHz, const float* leadHz, int numSamples,
                  float tractRatio) noexcept;

    static constexpr int numBands = 25;
    static constexpr float strength = 1.0f, maxDb = 10.0f;

    /** The current correction per band in dB (for tests and tools). */
    const float* getBandGainsDb() const noexcept { return gainDb; }
    float getBandHz (int band) const noexcept { return centreHz[band]; }

private:
    void analyse (float inputHz, float leadHz, float tractRatio) noexcept;
    void envelope (const std::vector<float>& ring, float f0, float scale, float* out) noexcept;
    void updateFilters (float fraction) noexcept;

    static constexpr int fftOrder = 10, fftSize = 1 << fftOrder, hop = 256;
    double sr = 48000.0;
    std::unique_ptr<juce::dsp::FFT> fft;
    std::vector<float> dryRing, leadRing, dryHzRing, leadHzRing, window, fftData;
    int write = 0, sinceHop = 0, filled = 0;
    float centreHz[numBands] {}, bandWeight[numBands] {}, cosW[numBands] {}, alpha[numBands] {};
    std::vector<float> inverse;   // numBands x numBands: band gains that give the wanted response at the centres
    float dryBase[numBands] {}, leadBase[numBands] {}, dryEnv[numBands] {}, leadEnv[numBands] {};
    float wanted[numBands] {}, smoothed[numBands] {}, gainDb[numBands] {}, previousDb[numBands] {};
    bool haveBase = false;
    float baseCoeff = 0.0f, smoothCoeff = 0.0f;
    Biquad bank[numBands];
};

/** One-knob vocal compressor (feed-forward, soft knee, stereo linked, auto make-up). */
class VocalCompressor
{
public:
    void prepare (double sampleRate);
    void reset();
    void process (float* left, float* right, int numSamples, float amount) noexcept;
    float getReductionDb() const noexcept { return reductionDb; }

private:
    double sr = 48000.0;
    float meanSquare = 0.0f, msCoeff = 0.0f, gainDb = 0.0f, attack = 0.0f, release = 0.0f;
    float reductionDb = 0.0f;
};

/** Anti-aliased (ADAA) tanh saturation with DC blocking. */
class Saturator
{
public:
    void prepare (double sampleRate);
    void reset();
    void process (float* left, float* right, int numSamples, float amount) noexcept;

private:
    float prev[2] {}, prevF[2] {};
    Biquad dcBlock[2];
};

/** Tempo-synced ping-pong delay with filtered feedback and ducking (send-style, adds to input). */
class StereoDelay
{
public:
    void prepare (double sampleRate);
    void reset();
    void process (float* left, float* right, int numSamples, float mix, float delaySeconds, float feedback) noexcept;

private:
    double sr = 48000.0;
    std::vector<float> bufL, bufR;
    int writeIndex = 0;
    Smoothed delaySamples;
    Biquad fbHp[2], fbLp[2];
    float duckEnv = 0.0f, duckAttack = 0.0f, duckRelease = 0.0f;
};

/** 8-line feedback-delay-network reverb with early reflections, damping and modulation. */
class FdnReverb
{
public:
    void prepare (double sampleRate);
    void reset();
    void process (float* left, float* right, int numSamples, float mix, float size) noexcept;

private:
    void updateSize (float size) noexcept;

    static constexpr int numLines = 8;
    double sr = 48000.0;
    std::vector<float> lines[numLines];
    int lineLength[numLines] {}, writePos[numLines] {};
    float baseLength[numLines] {}, feedbackGain[numLines] {}, dampState[numLines] {};
    float dampCoeff = 0.3f, currentSize = -1.0f;

    std::vector<float> preDelay;
    int preWrite = 0, preDelaySamples = 0;
    Biquad inHp, inLp;
    float modPhase = 0.0f, modInc = 0.0f;
};

/** Slow automatic gain control for the input (target ≈ -18 dBFS RMS). */
class AutoLevel
{
public:
    void prepare (double sampleRate);
    void reset();
    void process (float* data, int numSamples) noexcept;
    float getGainDb() const noexcept { return gainDb; }

private:
    float meanSquare = 0.0f, msCoeff = 0.0f, gainDb = 0.0f, upCoeff = 0.0f, downCoeff = 0.0f;
};

} // namespace voxa::dsp
