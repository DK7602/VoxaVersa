#pragma once

#include "Biquad.h"
#include <vector>

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
