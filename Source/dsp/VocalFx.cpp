#include "VocalFx.h"
#include <algorithm>

namespace voxa::dsp
{

static float coeffForMs (double sr, double ms) noexcept
{
    return ms <= 0.0 ? 0.0f : (float) std::exp (-1.0 / (sr * ms * 0.001));
}

static bool differs (float a, float b, float tolerance) noexcept { return std::abs (a - b) > tolerance; }

//==============================================================================
void CharacterProcessor::prepare (double sampleRate)
{
    sr = sampleRate;
    envAttack = coeffForMs (sr, 4.0);
    envRelease = coeffForMs (sr, 70.0);
    raspLp.c = BiquadCoeffs::lowPass (sr, 70.0, 0.7071);
    ch[1].noise.state = 0x9e3779b9u;
    filtersValid = false;
    reset();
}

void CharacterProcessor::reset()
{
    for (auto& c : ch)
        for (auto* b : { &c.hp1, &c.hp2, &c.lowShelf, &c.highShelf, &c.warmth, &c.nasal, &c.presence, &c.air,
                         &c.breathHp, &c.breathLp })
            b->reset();
    raspLp.reset();
    envelope = 0.0f;
}

void CharacterProcessor::updateFilters (const Settings& s) noexcept
{
    const bool changed = ! filtersValid
        || differs (s.lowCutHz, current.lowCutHz, 0.5f) || differs (s.tiltDb, current.tiltDb, 0.05f)
        || differs (s.warmthDb, current.warmthDb, 0.05f) || differs (s.nasalDb, current.nasalDb, 0.05f)
        || differs (s.presenceDb, current.presenceDb, 0.05f) || differs (s.airDb, current.airDb, 0.05f);

    if (! changed)
        return;

    const auto hp1 = BiquadCoeffs::highPass (sr, s.lowCutHz, 0.5412);
    const auto hp2 = BiquadCoeffs::highPass (sr, s.lowCutHz, 1.3066);
    const auto ls = BiquadCoeffs::lowShelf (sr, 280.0, -0.5 * s.tiltDb);
    const auto hs = BiquadCoeffs::highShelf (sr, 2600.0, 0.5 * s.tiltDb);
    const auto warm = BiquadCoeffs::peak (sr, 240.0, 0.8, s.warmthDb);
    const auto nasal = BiquadCoeffs::peak (sr, 1500.0, 2.5, s.nasalDb);
    const auto pres = BiquadCoeffs::peak (sr, 4200.0, 0.9, s.presenceDb);
    const auto air = BiquadCoeffs::highShelf (sr, 11000.0, s.airDb);
    const auto bhp = BiquadCoeffs::highPass (sr, 1800.0, 0.7071);
    const auto blp = BiquadCoeffs::lowPass (sr, juce::jmin (9500.0, sr * 0.45), 0.7071);

    for (auto& c : ch)
    {
        c.hp1.c = hp1; c.hp2.c = hp2; c.lowShelf.c = ls; c.highShelf.c = hs;
        c.warmth.c = warm; c.nasal.c = nasal; c.presence.c = pres; c.air.c = air;
        c.breathHp.c = bhp; c.breathLp.c = blp;
    }

    current = s;
    filtersValid = true;
}

void CharacterProcessor::process (float* left, float* right, int n, const Settings& s) noexcept
{
    updateFilters (s);

    const bool doBreath = s.breath > 0.001f;
    const bool doRasp = s.rasp > 0.001f;
    const bool doTone = std::abs (s.tiltDb) > 0.01f;
    const bool doWarm = std::abs (s.warmthDb) > 0.01f;
    const bool doNasal = std::abs (s.nasalDb) > 0.01f;
    const bool doPresence = std::abs (s.presenceDb) > 0.01f;
    const bool doAir = std::abs (s.airDb) > 0.01f;
    const float breathGain = 0.9f * s.breath * s.breath + 0.25f * s.breath;
    const float raspDepth = 0.75f * s.rasp;
    const float raspDrive = 1.0f + 3.0f * s.rasp;
    const float raspNorm = 1.0f / std::tanh (raspDrive);

    float* data[2] = { left, right };

    for (int i = 0; i < n; ++i)
    {
        const float mono = 0.5f * (left[i] + right[i]);
        const float a = std::abs (mono);
        envelope = a > envelope ? a + envAttack * (envelope - a) : a + envRelease * (envelope - a);

        float raspMod = 0.0f;
        if (doRasp)
            raspMod = juce::jlimit (-1.0f, 1.0f, raspLp.process (raspNoise.next()) * 6.0f);

        for (int c = 0; c < 2; ++c)
        {
            auto& st = ch[c];
            float x = data[c][i];
            x = st.hp2.process (st.hp1.process (x));

            if (doTone)     x = st.highShelf.process (st.lowShelf.process (x));
            if (doWarm)     x = st.warmth.process (x);
            if (doNasal)    x = st.nasal.process (x);
            if (doPresence) x = st.presence.process (x);
            if (doAir)      x = st.air.process (x);

            if (doRasp)
            {
                x *= 1.0f + raspDepth * raspMod;
                x = x + s.rasp * 0.5f * (std::tanh (raspDrive * x) * raspNorm - x);
            }

            if (doBreath)
            {
                const float noise = st.breathLp.process (st.breathHp.process (st.noise.next()));
                x = x * (1.0f - 0.25f * s.breath) + noise * envelope * breathGain * 2.2f;
            }

            data[c][i] = x;
        }
    }

    for (auto& c : ch)
        for (auto* b : { &c.hp1, &c.hp2, &c.lowShelf, &c.highShelf, &c.warmth, &c.nasal, &c.presence, &c.air,
                         &c.breathHp, &c.breathLp })
            b->sanitise();
}

//==============================================================================
void DeEsser::prepare (double sampleRate)
{
    sr = sampleRate;
    for (auto& h : hp)
        h.c = BiquadCoeffs::highPass (sr, 5500.0, 0.7071);
    attack = coeffForMs (sr, 0.5);
    release = coeffForMs (sr, 45.0);
    gainSmooth = coeffForMs (sr, 2.0);
    reset();
}

void DeEsser::reset()
{
    for (auto& h : hp) h.reset();
    env = 0.0f;
    gain = 1.0f;
    reductionDb = 0.0f;
}

void DeEsser::process (float* left, float* right, int n, float amount) noexcept
{
    if (amount < 0.005f)
    {
        reductionDb = 0.0f;
        return;
    }

    const float thresholdDb = -12.0f - 24.0f * amount;
    const float maxReduction = 3.0f + 13.0f * amount;
    float maxGr = 0.0f;

    for (int i = 0; i < n; ++i)
    {
        const float hl = hp[0].process (left[i]);
        const float hr = hp[1].process (right[i]);
        const float level = juce::jmax (std::abs (hl), std::abs (hr));
        env = level > env ? level + attack * (env - level) : level + release * (env - level);

        const float over = juce::Decibels::gainToDecibels (env, -100.0f) - thresholdDb;
        const float grDb = over > 0.0f ? juce::jmin (over * 0.75f, maxReduction) : 0.0f;
        const float target = juce::Decibels::decibelsToGain (-grDb);
        gain = target + gainSmooth * (gain - target);
        maxGr = juce::jmax (maxGr, grDb);

        left[i] = (left[i] - hl) + hl * gain;
        right[i] = (right[i] - hr) + hr * gain;
    }

    hp[0].sanitise();
    hp[1].sanitise();
    reductionDb = maxGr;
}

//==============================================================================
void VocalCompressor::prepare (double sampleRate)
{
    sr = sampleRate;
    msCoeff = coeffForMs (sr, 5.0);
    attack = coeffForMs (sr, 4.0);
    release = coeffForMs (sr, 110.0);
    reset();
}

void VocalCompressor::reset()
{
    meanSquare = 0.0f;
    gainDb = 0.0f;
    reductionDb = 0.0f;
}

void VocalCompressor::process (float* left, float* right, int n, float amount) noexcept
{
    if (amount < 0.005f)
    {
        reductionDb = 0.0f;
        return;
    }

    const float threshold = -8.0f - 24.0f * amount;
    const float ratio = 1.5f + 4.5f * amount;
    const float knee = 6.0f;
    const float makeup = -threshold * (1.0f - 1.0f / ratio) * 0.45f;
    float maxGr = 0.0f;

    for (int i = 0; i < n; ++i)
    {
        const float peak = juce::jmax (std::abs (left[i]), std::abs (right[i]));
        meanSquare = peak * peak + msCoeff * (meanSquare - peak * peak);
        const float levelDb = 10.0f * std::log10 (meanSquare + 1.0e-12f);

        float over = levelDb - threshold;
        float grDb;
        if (over <= -knee * 0.5f)      grDb = 0.0f;
        else if (over >= knee * 0.5f)  grDb = over * (1.0f - 1.0f / ratio);
        else
        {
            const float k = over + knee * 0.5f;
            grDb = (1.0f - 1.0f / ratio) * k * k / (2.0f * knee);
        }

        const float target = -grDb;
        gainDb = target < gainDb ? target + attack * (gainDb - target) : target + release * (gainDb - target);
        maxGr = juce::jmax (maxGr, -gainDb);

        const float g = juce::Decibels::decibelsToGain (gainDb + makeup);
        left[i] *= g;
        right[i] *= g;
    }

    reductionDb = maxGr;
}

//==============================================================================
static inline float logCosh (float x) noexcept
{
    const float ax = std::abs (x);
    return ax + std::log1p (std::exp (-2.0f * ax)) - 0.69314718f;
}

void Saturator::prepare (double sampleRate)
{
    for (auto& d : dcBlock)
        d.c = BiquadCoeffs::highPass (sampleRate, 12.0, 0.7071);
    reset();
}

void Saturator::reset()
{
    prev[0] = prev[1] = 0.0f;
    prevF[0] = prevF[1] = 0.0f;
    for (auto& d : dcBlock) d.reset();
}

void Saturator::process (float* left, float* right, int n, float amount) noexcept
{
    if (amount < 0.005f)
        return;

    const float drive = 1.0f + 7.0f * amount;
    const float makeup = 1.0f / juce::jmin (drive, 1.0f + 2.0f * amount * drive);
    const float wet = juce::jmin (1.0f, 0.4f + 0.6f * amount);
    float* data[2] = { left, right };

    for (int c = 0; c < 2; ++c)
    {
        for (int i = 0; i < n; ++i)
        {
            const float dry = data[c][i];
            const float x = dry * drive + 0.08f * amount;   // slight asymmetry = even harmonics
            const float fx = logCosh (x);
            const float diff = x - prev[c];
            const float y = std::abs (diff) > 1.0e-4f ? (fx - prevF[c]) / diff : std::tanh (0.5f * (x + prev[c]));
            prev[c] = x;
            prevF[c] = fx;
            const float sat = dcBlock[c].process (y) * makeup * 1.6f;
            data[c][i] = dry + wet * (sat - dry);
        }
        dcBlock[c].sanitise();
    }
}

//==============================================================================
void StereoDelay::prepare (double sampleRate)
{
    sr = sampleRate;
    const int size = (int) (sr * 2.6) + 4;
    bufL.assign ((size_t) size, 0.0f);
    bufR.assign ((size_t) size, 0.0f);
    delaySamples.prepare (sr, 120.0);
    for (auto& f : fbHp) f.c = BiquadCoeffs::highPass (sr, 220.0, 0.7071);
    for (auto& f : fbLp) f.c = BiquadCoeffs::lowPass (sr, 5200.0, 0.7071);
    duckAttack = coeffForMs (sr, 10.0);
    duckRelease = coeffForMs (sr, 280.0);
    reset();
}

void StereoDelay::reset()
{
    std::fill (bufL.begin(), bufL.end(), 0.0f);
    std::fill (bufR.begin(), bufR.end(), 0.0f);
    writeIndex = 0;
    delaySamples.reset ((float) (sr * 0.25));
    for (auto& f : fbHp) f.reset();
    for (auto& f : fbLp) f.reset();
    duckEnv = 0.0f;
}

void StereoDelay::process (float* left, float* right, int n, float mix, float delaySeconds, float feedback) noexcept
{
    if (mix < 0.001f)
    {
        // Keep the line flowing silently so re-enabling does not replay stale audio.
        return;
    }

    const int size = (int) bufL.size();
    delaySamples.setTarget (juce::jlimit (16.0f, (float) size - 8.0f, (float) (delaySeconds * sr)));
    feedback = juce::jlimit (0.0f, 0.92f, feedback);

    for (int i = 0; i < n; ++i)
    {
        const float d = delaySamples.next();
        float readPos = (float) writeIndex - d;
        if (readPos < 0.0f) readPos += (float) size;
        const int i0 = (int) readPos;
        const int i1 = (i0 + 1) % size;
        const float frac = readPos - (float) i0;
        const float outL = bufL[(size_t) i0] + frac * (bufL[(size_t) i1] - bufL[(size_t) i0]);
        const float outR = bufR[(size_t) i0] + frac * (bufR[(size_t) i1] - bufR[(size_t) i0]);

        const float in = 0.5f * (left[i] + right[i]);
        const float level = std::abs (in);
        duckEnv = level > duckEnv ? level + duckAttack * (duckEnv - level) : level + duckRelease * (duckEnv - level);
        const float duck = 1.0f / (1.0f + 6.0f * duckEnv);

        // Ping-pong: input enters the left line, lines cross-feed.
        const float fbL = fbLp[0].process (fbHp[0].process (outR * feedback));
        const float fbR = fbLp[1].process (fbHp[1].process (outL * feedback));
        bufL[(size_t) writeIndex] = in + fbL;
        bufR[(size_t) writeIndex] = fbR;
        writeIndex = (writeIndex + 1) % size;

        left[i] += outL * mix * duck;
        right[i] += outR * mix * duck;
    }

    for (auto& f : fbHp) f.sanitise();
    for (auto& f : fbLp) f.sanitise();
}

//==============================================================================
void FdnReverb::prepare (double sampleRate)
{
    sr = sampleRate;
    static constexpr float lengthsMs[numLines] = { 29.7f, 37.1f, 41.1f, 43.7f, 53.3f, 59.3f, 67.1f, 73.9f };
    for (int i = 0; i < numLines; ++i)
    {
        baseLength[i] = (float) (lengthsMs[i] * 0.001 * sr);
        lines[i].assign ((size_t) (baseLength[i] * 1.8f) + 64, 0.0f);
    }

    preDelaySamples = (int) (sr * 0.018);
    preDelay.assign ((size_t) preDelaySamples + 1, 0.0f);
    inHp.c = BiquadCoeffs::highPass (sr, 200.0, 0.7071);
    inLp.c = BiquadCoeffs::lowPass (sr, 8500.0, 0.7071);
    modInc = (float) (juce::MathConstants<double>::twoPi * 0.45 / sr);
    currentSize = -1.0f;
    reset();
}

void FdnReverb::reset()
{
    for (int i = 0; i < numLines; ++i)
    {
        std::fill (lines[i].begin(), lines[i].end(), 0.0f);
        writePos[i] = 0;
        dampState[i] = 0.0f;
    }
    std::fill (preDelay.begin(), preDelay.end(), 0.0f);
    preWrite = 0;
    inHp.reset();
    inLp.reset();
    modPhase = 0.0f;
}

void FdnReverb::updateSize (float size) noexcept
{
    if (std::abs (size - currentSize) < 0.005f)
        return;
    currentSize = size;

    const float scale = 0.6f + 1.0f * size;
    const float rt60 = 0.45f + 4.2f * size * size + 0.6f * size;
    for (int i = 0; i < numLines; ++i)
    {
        lineLength[i] = juce::jlimit (16, (int) lines[i].size() - 32, (int) (baseLength[i] * scale));
        feedbackGain[i] = std::pow (10.0f, -3.0f * (float) lineLength[i] / (rt60 * (float) sr));
    }
    dampCoeff = 0.15f + 0.45f * size;
}

void FdnReverb::process (float* left, float* right, int n, float mix, float size) noexcept
{
    if (mix < 0.001f)
        return;

    updateSize (juce::jlimit (0.0f, 1.0f, size));
    const float dryGain = 1.0f - 0.3f * mix;
    const float wetGain = mix * 0.55f;

    for (int i = 0; i < n; ++i)
    {
        const float in = inLp.process (inHp.process (0.5f * (left[i] + right[i])));

        const float delayed = preDelay[(size_t) preWrite];
        preDelay[(size_t) preWrite] = in;
        preWrite = (preWrite + 1) % (int) preDelay.size();

        modPhase += modInc;
        if (modPhase > juce::MathConstants<float>::twoPi)
            modPhase -= juce::MathConstants<float>::twoPi;
        const float mod = std::sin (modPhase) * 6.0f;

        float outs[numLines];
        for (int l = 0; l < numLines; ++l)
        {
            const int len = (int) lines[l].size();
            float readPos = (float) writePos[l] - (float) lineLength[l] - (l == 1 ? mod : (l == 6 ? -mod : 0.0f));
            while (readPos < 0.0f) readPos += (float) len;
            const int i0 = (int) readPos % len;
            const int i1 = (i0 + 1) % len;
            const float frac = readPos - std::floor (readPos);
            float v = lines[l][(size_t) i0] + frac * (lines[l][(size_t) i1] - lines[l][(size_t) i0]);
            dampState[l] = v + dampCoeff * (dampState[l] - v);
            outs[l] = dampState[l] * feedbackGain[l];
        }

        // Fast Walsh-Hadamard transform (orthonormal mixing).
        float h[numLines];
        std::copy (outs, outs + numLines, h);
        for (int len = 1; len < numLines; len <<= 1)
            for (int a = 0; a < numLines; a += len << 1)
                for (int b = a; b < a + len; ++b)
                {
                    const float x = h[b], y = h[b + len];
                    h[b] = x + y;
                    h[b + len] = x - y;
                }
        const float norm = 0.35355339f;   // 1/sqrt(8)

        for (int l = 0; l < numLines; ++l)
        {
            const float input = (l % 2 == 0 ? delayed : -delayed) * 0.5f;
            lines[l][(size_t) writePos[l]] = h[l] * norm + input;
            writePos[l] = (writePos[l] + 1) % (int) lines[l].size();
        }

        const float wetL = outs[0] - outs[2] + outs[4] - outs[6] + 0.5f * (outs[1] + outs[5]);
        const float wetR = outs[1] - outs[3] + outs[5] - outs[7] + 0.5f * (outs[2] + outs[6]);

        left[i] = left[i] * dryGain + wetL * wetGain;
        right[i] = right[i] * dryGain + wetR * wetGain;
    }

    inHp.sanitise();
    inLp.sanitise();
    for (auto& d : dampState)
        if (std::abs (d) < 1.0e-20f) d = 0.0f;
}

//==============================================================================
void AutoLevel::prepare (double sampleRate)
{
    msCoeff = coeffForMs (sampleRate, 300.0);
    upCoeff = coeffForMs (sampleRate, 900.0);
    downCoeff = coeffForMs (sampleRate, 60.0);
    reset();
}

void AutoLevel::reset()
{
    meanSquare = 0.0f;
    gainDb = 0.0f;
}

void AutoLevel::process (float* data, int n) noexcept
{
    constexpr float targetDb = -18.0f;
    for (int i = 0; i < n; ++i)
    {
        const float x = data[i];
        meanSquare = x * x + msCoeff * (meanSquare - x * x);
        const float levelDb = 10.0f * std::log10 (meanSquare + 1.0e-12f);

        if (levelDb > -55.0f)    // gate: hold gain during silence
        {
            const float wanted = juce::jlimit (-12.0f, 18.0f, targetDb - (levelDb + gainDb) + gainDb);
            const float coeff = wanted > gainDb ? upCoeff : downCoeff;
            gainDb = wanted + coeff * (gainDb - wanted);
        }

        data[i] = x * juce::Decibels::decibelsToGain (gainDb);
    }
}

} // namespace voxa::dsp
