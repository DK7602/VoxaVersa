#include "VocalFx.h"
#include <algorithm>
#include <complex>

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
                         &c.ring, &c.body, &c.breathHp, &c.breathLp })
            b->reset();
    raspLp.reset();
    envelope = 0.0f;
}

void CharacterProcessor::updateFilters (const Settings& s) noexcept
{
    const bool changed = ! filtersValid
        || differs (s.lowCutHz, current.lowCutHz, 0.5f) || differs (s.tiltDb, current.tiltDb, 0.05f)
        || differs (s.warmthDb, current.warmthDb, 0.05f) || differs (s.nasalDb, current.nasalDb, 0.05f)
        || differs (s.presenceDb, current.presenceDb, 0.05f) || differs (s.airDb, current.airDb, 0.05f)
        || differs (s.ringHz, current.ringHz, 1.0f) || differs (s.ringDb, current.ringDb, 0.05f)
        || differs (s.bodyHz, current.bodyHz, 1.0f) || differs (s.bodyDb, current.bodyDb, 0.05f);

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
    const auto ring = BiquadCoeffs::peak (sr, s.ringHz, 2.0, s.ringDb);
    const auto body = BiquadCoeffs::peak (sr, s.bodyHz, 1.0, s.bodyDb);
    const auto bhp = BiquadCoeffs::highPass (sr, 1800.0, 0.7071);
    const auto blp = BiquadCoeffs::lowPass (sr, juce::jmin (9500.0, sr * 0.45), 0.7071);

    for (auto& c : ch)
    {
        c.hp1.c = hp1; c.hp2.c = hp2; c.lowShelf.c = ls; c.highShelf.c = hs;
        c.warmth.c = warm; c.nasal.c = nasal; c.presence.c = pres; c.air.c = air;
        c.ring.c = ring; c.body.c = body;
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
    const bool doRing = std::abs (s.ringDb) > 0.01f;
    const bool doBody = std::abs (s.bodyDb) > 0.01f;
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
            if (doRing)     x = st.ring.process (x);
            if (doBody)     x = st.body.process (x);

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
                         &c.ring, &c.body, &c.breathHp, &c.breathLp })
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
void ConsonantKeeper::prepare (double sampleRate)
{
    sr = sampleRate;
    constexpr double split = 2500.0;
    for (int k = 0; k < 2; ++k)
    {
        leadLow[k].c = BiquadCoeffs::lowPass (sr, split);
        leadHigh[k].c = inputHigh[k].c = BiquadCoeffs::highPass (sr, split);
        inputHiss[k].c = leadHiss[k].c = BiquadCoeffs::highPass (sr, 3500.0);
    }
    inputBody.c = leadBody.c = BiquadCoeffs::lowPass (sr, 1500.0);
    envCoeff = coeffForMs (sr, 4.0);
    attack = coeffForMs (sr, 2.0);
    release = coeffForMs (sr, 12.0);
    levelCoeff = coeffForMs (sr, 3000.0);
    reset();
}

void ConsonantKeeper::reset()
{
    for (auto* b : { &leadLow[0], &leadLow[1], &leadHigh[0], &leadHigh[1], &inputHigh[0], &inputHigh[1],
                     &inputHiss[0], &inputHiss[1], &inputBody, &leadHiss[0], &leadHiss[1], &leadBody })
        b->reset();
    inHiss = inBody = aiHiss = aiBody = 0.0f;
    weight = 0.0f;
    leadLevel = inputLevel = 0.0f;
    gain = 1.0f;
}

void ConsonantKeeper::process (float* lead, const float* input, int n) noexcept
{
    constexpr float floor = 1.0e-10f;   // -100 dB
    constexpr float gate = 1.0e-7f;     // -70 dB: quieter than this is silence, not a consonant

    for (int i = 0; i < n; ++i)
    {
        const float x = input[i], y = lead[i];

        const float hissIn = inputHiss[1].process (inputHiss[0].process (x));
        const float bodyIn = inputBody.process (x);
        const float hissAi = leadHiss[1].process (leadHiss[0].process (y));
        const float bodyAi = leadBody.process (y);
        inHiss = hissIn * hissIn + envCoeff * (inHiss - hissIn * hissIn);
        inBody = bodyIn * bodyIn + envCoeff * (inBody - bodyIn * bodyIn);
        aiHiss = hissAi * hissAi + envCoeff * (aiHiss - hissAi * hissAi);
        aiBody = bodyAi * bodyAi + envCoeff * (aiBody - bodyAi * bodyAi);

        const float inRatio = 10.0f * std::log10 ((inHiss + floor) / (inBody + floor));
        const float aiRatio = 10.0f * std::log10 ((aiHiss + floor) / (aiBody + floor));
        const bool audible = inHiss + inBody > gate;

        // The input hisses (2 dB above its body, fully at 10 dB), or the AI hisses where the input has no voiced
        // sound (breaths, consonant tails): both are the input's own consonant.
        float target = audible ? juce::jlimit (0.0f, 1.0f, (inRatio - 2.0f) * 0.125f) : 0.0f;
        target = juce::jmax (target, juce::jlimit (0.0f, 1.0f, (aiRatio - 2.0f) * 0.125f)
                                     * juce::jlimit (0.0f, 1.0f, (inRatio + 8.0f) * 0.25f));
        weight = target + (target > weight ? attack : release) * (weight - target);

        // Level match on clearly voiced sound (vowels) of both.
        if (audible && inRatio < -10.0f)
        {
            leadLevel = y * y + levelCoeff * (leadLevel - y * y);
            inputLevel = x * x + levelCoeff * (inputLevel - x * x);
        }

        const float low = leadLow[1].process (leadLow[0].process (y));
        const float high = leadHigh[1].process (leadHigh[0].process (y));
        const float kept = inputHigh[1].process (inputHigh[0].process (x));
        lead[i] = low + high + weight * (gain * kept - high);
    }

    if (inputLevel > gate)
        gain = juce::jlimit (0.5f, 2.0f, std::sqrt (leadLevel / inputLevel));

    for (auto* b : { &leadLow[0], &leadLow[1], &leadHigh[0], &leadHigh[1], &inputHigh[0], &inputHigh[1],
                     &inputHiss[0], &inputHiss[1], &inputBody, &leadHiss[0], &leadHiss[1], &leadBody })
        b->sanitise();
}

//==============================================================================
namespace
{
    /** Roots of z^p + a[1] z^(p-1) + ... + a[p] (Durand-Kerner; p <= 16). */
    void polynomialRoots (const double* a, int p, std::complex<double>* roots) noexcept
    {
        for (int k = 0; k < p; ++k)
            roots[k] = std::polar (0.9, juce::MathConstants<double>::twoPi * k / p + 0.4);

        for (int iteration = 0; iteration < 80; ++iteration)
        {
            double change = 0.0;
            for (int k = 0; k < p; ++k)
            {
                const auto z = roots[k];
                std::complex<double> value = 1.0, denominator = 1.0;
                for (int j = 1; j <= p; ++j)
                    value = value * z + a[j];
                for (int j = 0; j < p; ++j)
                    if (j != k)
                        denominator *= z - roots[j];
                if (std::abs (denominator) < 1.0e-300)
                    continue;
                const auto delta = value / denominator;
                roots[k] = z - delta;
                change = juce::jmax (change, std::abs (delta));
            }
            if (change < 1.0e-9)
                break;
        }
    }
}

//==============================================================================
void BreathKeeper::prepare (double sampleRate)
{
    envCoeff = (float) std::exp (-1.0 / (0.010 * sampleRate));
    sungCoeff = (float) std::exp (-1.0 / (0.400 * sampleRate));
    attack = (float) std::exp (-1.0 / (0.003 * sampleRate));
    release = (float) std::exp (-1.0 / (0.005 * sampleRate));   // the next note starts at full level
    reset();
}

void BreathKeeper::reset()
{
    inEnv = outEnv = inSung = outSung = 0.0f;
    gain = 1.0f;
}

void BreathKeeper::process (float* left, float* right, const float* input, const float* inputVoiced, int n) noexcept
{
    constexpr float quiet = 0.0316f;    // -15 dB below the singing: breaths, not consonants or notes
    constexpr float leeway = 1.41f;     // +1.5 dB
    constexpr float floorGain = 0.03f;  // -30 dB at most
    constexpr float silence = 1.0e-9f;  // -90 dB

    for (int i = 0; i < n; ++i)
    {
        const float x = input[i], y = right != nullptr ? 0.5f * (left[i] + right[i]) : left[i];
        inEnv = x * x + envCoeff * (inEnv - x * x);
        outEnv = y * y + envCoeff * (outEnv - y * y);
        const bool voiced = inputVoiced[i] > 0.5f;

        if (voiced)
        {
            inSung = inEnv + sungCoeff * (inSung - inEnv);
            outSung = outEnv + sungCoeff * (outSung - outEnv);
        }

        float target = 1.0f;
        if (! voiced && inSung > silence && outSung > silence && inEnv < inSung * quiet)
        {
            const float allowed = inEnv / inSung * outSung * leeway;
            if (outEnv > allowed)
                target = juce::jmax (floorGain, std::sqrt (allowed / outEnv));
        }
        gain = target + (target < gain ? attack : release) * (gain - target);
        left[i] *= gain;
        if (right != nullptr)
            right[i] *= gain;
    }
}

void RKeeper::FormantTracker::prepare (double sampleRate, double ceilingHz)
{
    sr = sampleRate;
    const auto longest = (size_t) std::ceil (0.025 * 2.0 * maxCeilingHz) + 1;
    ring.assign (longest, 0.0f);
    frame.assign (longest, 0.0f);
    hamming.assign (longest, 0.0f);
    ceiling = 0.0;
    setCeiling (ceilingHz);
}

void RKeeper::FormantTracker::setCeiling (double ceilingHz) noexcept
{
    ceilingHz = juce::jlimit (3000.0, maxCeilingHz, ceilingHz);
    if (juce::exactlyEqual (ceilingHz, ceiling))
        return;
    ceiling = ceilingHz;
    rate = juce::jmin (sr, 2.0 * ceiling);
    step = rate / sr;
    // 8th-order Butterworth anti-alias filter just below the new Nyquist
    const double cutoff = 0.46 * rate;
    const double q[] = { 0.5098, 0.6013, 0.9000, 2.5629 };
    for (int k = 0; k < 4; ++k)
        aa[k].c = BiquadCoeffs::lowPass (sr, cutoff, q[k]);
    window = juce::jmin ((int) ring.size(), juce::roundToInt (0.025 * rate));
    for (int i = 0; i < window; ++i)
        hamming[(size_t) i] = 0.54f - 0.46f * std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) (window - 1));
    reset();
}

void RKeeper::FormantTracker::reset()
{
    for (auto& f : aa)
        f.reset();
    std::fill (ring.begin(), ring.end(), 0.0f);
    write = 0;
    phase = 0.0;
    previousInput = 0.0f;
    f3 = 0.0f;
    levelDb = -100.0f;
}

void RKeeper::FormantTracker::push (float x) noexcept
{
    float y = x;
    for (auto& f : aa)
        y = f.process (y);

    // resampled to `rate` by linear interpolation (the filtered audio is ~10x oversampled)
    phase += step;
    if (phase >= 1.0)
    {
        phase -= 1.0;
        const float back = (float) (phase / step);   // how far before this sample the output instant lies
        ring[(size_t) write] = y - back * (y - previousInput);
        write = (write + 1) % window;
    }
    previousInput = y;
}

void RKeeper::FormantTracker::analyse() noexcept
{
    constexpr int order = 10;
    double energy = 0.0;
    float previous = ring[(size_t) write];
    for (int i = 0; i < window; ++i)
    {
        const float x = ring[(size_t) ((write + i) % window)];
        energy += (double) x * x;
        frame[(size_t) i] = (x - 0.97f * previous) * hamming[(size_t) i];   // pre-emphasis, window
        previous = x;
    }
    levelDb = (float) (10.0 * std::log10 (energy / window + 1.0e-20));
    f3 = 0.0f;

    double r[order + 1];
    for (int lag = 0; lag <= order; ++lag)
    {
        double acc = 0.0;
        for (int i = lag; i < window; ++i)
            acc += (double) frame[(size_t) i] * frame[(size_t) (i - lag)];
        r[lag] = acc;
    }
    if (r[0] < 1.0e-12)
        return;
    r[0] *= 1.0 + 1.0e-9;

    // Levinson-Durbin: A(z) = 1 + a1 z^-1 + ... + a12 z^-12
    double a[order + 1] = { 1.0 }, previousA[order + 1];
    double error = r[0];
    for (int i = 1; i <= order; ++i)
    {
        double acc = r[i];
        for (int j = 1; j < i; ++j)
            acc += a[j] * r[i - j];
        const double k = -acc / error;
        std::copy (a, a + order + 1, previousA);
        for (int j = 1; j < i; ++j)
            a[j] = previousA[j] + k * previousA[i - j];
        a[i] = k;
        error *= 1.0 - k * k;
        if (error <= 0.0)
            return;
    }

    std::complex<double> roots[order];
    polynomialRoots (a, order, roots);

    float formants[order];
    int count = 0;
    for (const auto& z : roots)
    {
        if (z.imag() <= 0.0)
            continue;
        const double hz = std::atan2 (z.imag(), z.real()) * rate / juce::MathConstants<double>::twoPi;
        if (hz > 50.0 && hz < 0.5 * rate - 50.0)   // every pole counts, like Praat: a bandwidth limit skips formants
            formants[count++] = (float) hz;
    }
    std::sort (formants, formants + count);
    if (count >= 3)
        f3 = formants[2];
}

void RKeeper::prepare (double sampleRate)
{
    sr = sampleRate;
    hop = juce::jmax (1, juce::roundToInt (0.005 * sr));
    updateEvery = juce::jmax (1, juce::roundToInt (sr / 3000.0));
    auto per = [this] (double ms) { return (float) (1.0 - std::exp (-updateEvery / (0.001 * ms * sr))); };
    glide = per (15.0);
    rise = per (8.0);
    fall = per (15.0);
    dry.prepare (sr, maleCeilingHz);
    ai.prepare (sr, femaleCeilingHz);
    constexpr int history = 300;   // 1.5 s of 5 ms frames
    dryHistory.assign (history, 0.0f);
    aiHistory.assign (history, 0.0f);
    scratch.assign (history, 0.0f);
    reset();
}

void RKeeper::reset()
{
    dry.reset();
    ai.reset();
    boost.c = cut.c = BiquadCoeffs();
    wantedWeight = 0.0f;
    untilUpdate = 0;
    boost.reset();
    cut.reset();
    dryCount = aiCount = frames = sinceFrame = 0;
    std::fill (std::begin (dryRecent), std::end (dryRecent), 0.0f);
    std::fill (std::begin (aiRecent), std::end (aiRecent), 0.0f);
    dryBase = aiBase = 0.0f;
    peakDb = -100.0f;
    weight = 0.0f;
    voicedSamples = frameSamples = 0;
    stats = {};
}

void RKeeper::setCeilings (double inputHz, double leadHz) noexcept
{
    if (! juce::exactlyEqual (juce::jlimit (3000.0, FormantTracker::maxCeilingHz, inputHz), dry.getCeiling()))
    {
        dry.setCeiling (inputHz);
        dryCount = 0;
        dryBase = 0.0f;
    }
    if (! juce::exactlyEqual (juce::jlimit (3000.0, FormantTracker::maxCeilingHz, leadHz), ai.getCeiling()))
    {
        ai.setCeiling (leadHz);
        aiCount = 0;
        aiBase = 0.0f;
    }
}

float RKeeper::steady (float* recent, float latest) const noexcept
{
    recent[0] = recent[1];
    recent[1] = recent[2];
    recent[2] = latest;
    return juce::jmax (juce::jmin (recent[0], recent[1]), juce::jmin (juce::jmax (recent[0], recent[1]), recent[2]));
}

float RKeeper::baseline (std::vector<float>& history, int count, std::vector<float>& work) const
{
    const int n = juce::jmin (count, (int) history.size());
    if (n < 20)
        return 0.0f;
    std::copy (history.begin(), history.begin() + n, work.begin());
    auto middle = work.begin() + (n * 7) / 10;   // the 70th percentile: the vowels, not the r's dips
    std::nth_element (work.begin(), middle, work.begin() + n);
    return *middle;
}

void RKeeper::process (float* lead, const float* input, const float* inputVoiced, int n) noexcept
{
    for (int i = 0; i < n; ++i)
    {
        const float y = lead[i];
        voicedSamples += inputVoiced[i] > 0.5f ? 1 : 0;
        ++frameSamples;
        ai.push (y);   // the AI voice as it came, before this
        dry.push (input[i]);

        if (++sinceFrame >= hop)   // a frame every 5 ms, the same moment in both voices
        {
            sinceFrame = 0;
            dry.analyse();
            ai.analyse();
            const float dryF3 = steady (dryRecent, dry.f3), aiF3 = steady (aiRecent, ai.f3);
            const bool voiced = voicedSamples * 2 > frameSamples;
            voicedSamples = frameSamples = 0;
            peakDb = juce::jmax (dry.levelDb, peakDb - 0.01f);   // forgets ~2 dB a second
            const bool sung = voiced && dry.levelDb > peakDb - 25.0f && dryF3 > 0.0f;

            if (sung)
            {
                dryHistory[(size_t) (dryCount++ % (int) dryHistory.size())] = dryF3;
                if (aiF3 > 0.0f)
                    aiHistory[(size_t) (aiCount++ % (int) aiHistory.size())] = aiF3;
            }
            if (++frames % 10 == 0)
            {
                dryBase = baseline (dryHistory, dryCount, scratch);
                aiBase = baseline (aiHistory, aiCount, scratch);
            }

            // An r: the dry F3 8 % below its vowels starts it, 20 % is a full one.
            wantedWeight = 0.0f;
            if (sung && aiF3 > 0.0f && dryBase > 0.0f && aiBase > 0.0f)
            {
                const float depth = (dryBase - dryF3) / dryBase;
                const float w = juce::jlimit (0.0f, 1.0f, (depth - 0.08f) / 0.12f);
                const float target = aiBase * (1.0f - juce::jmin (depth, maxDepth));
                if (w > 0.0f && target < 0.97f * aiF3)
                {
                    wantedWeight = w;
                    boostTarget = target;
                    cutTarget = aiF3;
                }
            }
            stats.sungFrames += sung ? 1 : 0;
            stats.rFrames += weight > 0.1f ? 1 : 0;
            stats.dryBase = dryBase;
            stats.aiBase = aiBase;
        }

        // The filters glide (frequencies over ~15 ms, strength ~8 ms up and ~15 ms down): set afresh every 5 ms,
        // they jumped by hundreds of Hz at up to 15 dB, which spiked and rang on deep r's ("friend", heard as a
        // strange tone).
        if (--untilUpdate <= 0)
        {
            untilUpdate = updateEvery;
            const bool idle = weight < 0.02f;
            logBoost = idle ? std::log (boostTarget) : logBoost + glide * (std::log (boostTarget) - logBoost);
            logCut = idle ? std::log (cutTarget) : logCut + glide * (std::log (cutTarget) - logCut);
            weight += (wantedWeight - weight) * (wantedWeight > weight ? rise : fall);
            const float gainDb = maxGainDb * weight;
            boost.c = BiquadCoeffs::peak (sr, std::exp (logBoost), 5.0, gainDb);
            cut.c = BiquadCoeffs::peak (sr, std::exp (logCut), 5.0, -gainDb);
        }

        lead[i] = boost.process (cut.process (y));
    }

    boost.sanitise();
    cut.sanitise();
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

//==============================================================================
void VowelKeeper::prepare (double sampleRate)
{
    sr = sampleRate;
    fft = std::make_unique<juce::dsp::FFT> (fftOrder);
    dryRing.assign ((size_t) fftSize, 0.0f);
    leadRing.assign ((size_t) fftSize, 0.0f);
    dryHzRing.assign ((size_t) fftSize, 0.0f);
    leadHzRing.assign ((size_t) fftSize, 0.0f);
    fftData.assign ((size_t) fftSize * 2, 0.0f);
    window.resize ((size_t) fftSize);
    for (int i = 0; i < fftSize; ++i)
        window[(size_t) i] = 0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) fftSize);

    constexpr double q = 4.3;   // a 1/3-octave bell every 1/6 octave
    for (int j = 0; j < numBands; ++j)
    {
        centreHz[j] = 250.0f * std::pow (2.0f, (float) j / 6.0f);
        bandWeight[j] = juce::jlimit (0.0f, 1.0f, juce::jmin ((centreHz[j] - 150.0f) / 150.0f, (5000.0f - centreHz[j]) / 1000.0f));
        const double w = juce::MathConstants<double>::twoPi * centreHz[j] / sr;
        cosW[j] = (float) std::cos (w);
        alpha[j] = (float) (std::sin (w) / (2.0 * q));
    }

    // Neighbouring bells overlap: solve once for the band gains whose summed response hits the wanted dB at the
    // centres (regularised least squares, B = response in dB per dB of each band at each centre).
    constexpr int nb = numBands;
    std::vector<double> b ((size_t) nb * nb), m ((size_t) nb * nb * 2, 0.0);
    for (int j = 0; j < nb; ++j)
    {
        const auto c = BiquadCoeffs::peak (sr, centreHz[j], q, 6.0);
        for (int i = 0; i < nb; ++i)
        {
            const auto z = std::polar (1.0, -juce::MathConstants<double>::twoPi * centreHz[i] / sr);
            const auto h = ((double) c.b0 + (double) c.b1 * z + (double) c.b2 * z * z) / (1.0 + (double) c.a1 * z + (double) c.a2 * z * z);
            b[(size_t) (i * nb + j)] = 20.0 * std::log10 (std::abs (h)) / 6.0;
        }
    }
    // m = [B'B + 0.05 I | I], Gauss-Jordan, then inverse = (B'B + 0.05 I)^-1 B'
    for (int r = 0; r < nb; ++r)
    {
        for (int c = 0; c < nb; ++c)
        {
            double sum = r == c ? 0.05 : 0.0;
            for (int i = 0; i < nb; ++i)
                sum += b[(size_t) (i * nb + r)] * b[(size_t) (i * nb + c)];
            m[(size_t) (r * 2 * nb + c)] = sum;
        }
        m[(size_t) (r * 2 * nb + nb + r)] = 1.0;
    }
    for (int col = 0; col < nb; ++col)
    {
        int pivot = col;
        for (int r = col + 1; r < nb; ++r)
            if (std::abs (m[(size_t) (r * 2 * nb + col)]) > std::abs (m[(size_t) (pivot * 2 * nb + col)]))
                pivot = r;
        for (int c = 0; c < 2 * nb; ++c)
            std::swap (m[(size_t) (col * 2 * nb + c)], m[(size_t) (pivot * 2 * nb + c)]);
        const double d = m[(size_t) (col * 2 * nb + col)];
        for (int c = 0; c < 2 * nb; ++c)
            m[(size_t) (col * 2 * nb + c)] /= d;
        for (int r = 0; r < nb; ++r)
        {
            if (r == col)
                continue;
            const double f = m[(size_t) (r * 2 * nb + col)];
            for (int c = 0; c < 2 * nb; ++c)
                m[(size_t) (r * 2 * nb + c)] -= f * m[(size_t) (col * 2 * nb + c)];
        }
    }
    inverse.assign ((size_t) nb * nb, 0.0f);
    for (int r = 0; r < nb; ++r)
        for (int c = 0; c < nb; ++c)
        {
            double sum = 0.0;
            for (int k = 0; k < nb; ++k)
                sum += m[(size_t) (r * 2 * nb + nb + k)] * b[(size_t) (c * nb + k)];
            inverse[(size_t) (r * nb + c)] = (float) sum;
        }

    baseCoeff = (float) (1.0 - std::exp (-hop / sr / 3.0));      // the voices' own timbre: ~3 s averages
    smoothCoeff = (float) (1.0 - std::exp (-hop / sr / 0.02));   // the correction glides over ~20 ms
    reset();
}

void VowelKeeper::reset()
{
    std::fill (dryRing.begin(), dryRing.end(), 0.0f);
    std::fill (leadRing.begin(), leadRing.end(), 0.0f);
    std::fill (dryHzRing.begin(), dryHzRing.end(), 0.0f);
    std::fill (leadHzRing.begin(), leadHzRing.end(), 0.0f);
    write = sinceHop = filled = 0;
    haveBase = false;
    for (int j = 0; j < numBands; ++j)
    {
        dryBase[j] = leadBase[j] = wanted[j] = smoothed[j] = gainDb[j] = previousDb[j] = 0.0f;
        bank[j].reset();
    }
    updateFilters (1.0f);
}

void VowelKeeper::envelope (const std::vector<float>& ring, float f0, float scale, float* out) noexcept
{
    for (int i = 0; i < fftSize; ++i)
        fftData[(size_t) i] = ring[(size_t) ((write + i) % fftSize)] * window[(size_t) i];
    std::fill (fftData.begin() + fftSize, fftData.end(), 0.0f);
    fft->performFrequencyOnlyForwardTransform (fftData.data());

    // The strongest harmonic within half a fundamental of each band: the envelope wherever the harmonics fall.
    const float binHz = (float) (sr / fftSize);
    for (int j = 0; j < numBands; ++j)
    {
        const float c = centreHz[j] * scale;
        const int lo = juce::jlimit (1, fftSize / 2, (int) ((c - 0.5f * f0) / binHz));
        const int hi = juce::jlimit (lo, fftSize / 2, (int) std::ceil ((c + 0.5f * f0) / binHz));
        float peak = 0.0f;
        for (int k = lo; k <= hi; ++k)
            peak = juce::jmax (peak, fftData[(size_t) k]);
        out[j] = 20.0f * std::log10 (peak + 1.0e-9f);
    }
}

void VowelKeeper::analyse (float inputHz, float leadHz, float tractRatio) noexcept
{
    bool sung = filled >= fftSize && inputHz > 0.0f && leadHz > 0.0f;
    if (sung)
    {
        float energy = 0.0f;
        for (auto x : dryRing)
            energy += x * x;
        sung = energy > (float) fftSize * 1.0e-6f;   // above -60 dBFS
    }

    if (sung)
    {
        envelope (dryRing, inputHz, 1.0f / juce::jlimit (0.7f, 1.4f, tractRatio), dryEnv);
        envelope (leadRing, leadHz, 1.0f, leadEnv);
        if (! haveBase)
        {
            std::copy (dryEnv, dryEnv + numBands, dryBase);
            std::copy (leadEnv, leadEnv + numBands, leadBase);
            haveBase = true;
        }
        // Above the singer's top note: the envelope the AI voice would have with its resonances lower (bands are
        // 1/6 octave apart, so a shift of k semitones reads k/2 bands higher).
        float topShift = 0.0f;
        if (topMidi > 0.0f)
            topShift = juce::jmin (2.5f, 0.4f * juce::jmax (0.0f, 69.0f + 12.0f * std::log2 (leadHz / 440.0f) - topMidi));
        float mean = 0.0f, weights = 0.0f;
        for (int j = 0; j < numBands; ++j)
        {
            dryBase[j] += (dryEnv[j] - dryBase[j]) * baseCoeff;
            leadBase[j] += (leadEnv[j] - leadBase[j]) * baseCoeff;
            wanted[j] = (dryEnv[j] - dryBase[j]) - (leadEnv[j] - leadBase[j]);
            if (topShift > 0.0f)
            {
                const float at = juce::jmin ((float) (numBands - 1), (float) j + 0.5f * topShift);
                const int k = (int) at;
                const float frac = at - (float) k;
                const float lowered = leadEnv[k] + (k + 1 < numBands ? (leadEnv[k + 1] - leadEnv[k]) * frac : 0.0f);
                wanted[j] += lowered - leadEnv[j];
            }
            mean += wanted[j] * bandWeight[j];
            weights += bandWeight[j];
        }
        for (int j = 0; j < numBands; ++j)   // the shape only: the level is the AI singer's own
            wanted[j] = juce::jlimit (-maxDb, maxDb, strength * (wanted[j] - mean / weights)) * bandWeight[j];
    }
    else
    {
        std::fill (wanted, wanted + numBands, 0.0f);
    }

    for (int j = 0; j < numBands; ++j)
    {
        smoothed[j] += (wanted[j] - smoothed[j]) * smoothCoeff;
        previousDb[j] = gainDb[j];
    }
    for (int r = 0; r < numBands; ++r)
    {
        float g = 0.0f;
        for (int c = 0; c < numBands; ++c)
            g += inverse[(size_t) (r * numBands + c)] * smoothed[c];
        gainDb[r] = juce::jlimit (-1.5f * maxDb, 1.5f * maxDb, g);
    }
}

void VowelKeeper::updateFilters (float fraction) noexcept
{
    for (int j = 0; j < numBands; ++j)
    {
        const float db = previousDb[j] + (gainDb[j] - previousDb[j]) * fraction;
        const float A = std::pow (10.0f, db / 40.0f);
        const float a0 = 1.0f + alpha[j] / A;
        auto& c = bank[j].c;
        c.b0 = (1.0f + alpha[j] * A) / a0;
        c.b1 = -2.0f * cosW[j] / a0;
        c.b2 = (1.0f - alpha[j] * A) / a0;
        c.a1 = c.b1;
        c.a2 = (1.0f - alpha[j] / A) / a0;
    }
}

void VowelKeeper::process (float* lead, const float* input, const float* inputHz, const float* leadHz, int n,
                           float tractRatio) noexcept
{
    for (int i = 0; i < n; ++i)
    {
        dryRing[(size_t) write] = input[i];
        leadRing[(size_t) write] = lead[i];
        dryHzRing[(size_t) write] = inputHz[i];
        leadHzRing[(size_t) write] = leadHz[i];
        write = (write + 1) % fftSize;
        filled = juce::jmin (filled + 1, fftSize);

        if (++sinceHop >= hop)
        {
            sinceHop = 0;
            const auto centre = (size_t) ((write + fftSize / 2) % fftSize);   // the pitch in the window's middle
            analyse (dryHzRing[centre], leadHzRing[centre], tractRatio);
        }
        if ((sinceHop & 31) == 0)
            updateFilters ((float) sinceHop / (float) hop);

        float y = lead[i];
        for (auto& f : bank)
            y = f.process (y);
        lead[i] = y;
    }
}

//==============================================================================
void Grit::prepare (double sampleRate)
{
    sr = sampleRate;
    for (int k = 0; k < 2; ++k)
    {
        growlBand[k].c = BiquadCoeffs::bandPass (sr, 65.0, 0.7);
        noiseBand[k].c = BiquadCoeffs::bandPass (sr, 2700.0, 0.6);
        voiceHigh[k].c = BiquadCoeffs::highPass (sr, 1200.0);
        voiceLow[k].c = BiquadCoeffs::lowPass (sr, 6000.0);
    }
    attack = (float) (1.0 - std::exp (-1.0 / (0.0005 * sr)));
    release = (float) (1.0 - std::exp (-1.0 / (0.004 * sr)));
    powerCoeff = (float) (1.0 - std::exp (-1.0 / (0.5 * sr)));
    reset();
}

void Grit::reset()
{
    for (int k = 0; k < 2; ++k)
    {
        growlBand[k].reset();
        noiseBand[k].reset();
        voiceHigh[k].reset();
        voiceLow[k].reset();
        env[k] = 0.0f;
    }
    growlPower = noisePower = 0.02f;
}

void Grit::process (float* left, float* right, int n, float amount) noexcept
{
    if (amount <= 0.001f)
        return;
    for (int i = 0; i < n; ++i)
    {
        // one growl and one noise for both channels (the voice sits in the middle)
        const float g = growlBand[1].process (growlBand[0].process (2.0f * random.nextFloat() - 1.0f));
        growlPower += (g * g - growlPower) * powerCoeff;
        const float growl = 1.0f + 0.45f * amount * juce::jlimit (-2.5f, 2.5f, g / std::sqrt (growlPower + 1.0e-12f));
        const float w = noiseBand[1].process (noiseBand[0].process (2.0f * random.nextFloat() - 1.0f));
        noisePower += (w * w - noisePower) * powerCoeff;
        const float noise = w / std::sqrt (noisePower + 1.0e-12f);

        for (int c = 0; c < 2; ++c)
        {
            float* x = c == 0 ? left : right;
            if (x == nullptr)
                continue;
            const float band = std::abs (voiceLow[c].process (voiceHigh[c].process (x[i])));
            env[c] += (band - env[c]) * (band > env[c] ? attack : release);
            x[i] = x[i] * growl + 0.5f * amount * env[c] * noise;
        }
    }
}

//==============================================================================
void Whisperizer::prepare (double sampleRate)
{
    size = juce::nextPowerOfTwo (juce::roundToInt (sampleRate * 0.021));   // ~21 ms
    hop = size / 4;
    fft = std::make_unique<juce::dsp::FFT> ((int) std::round (std::log2 ((double) size)));
    ring.assign ((size_t) size, 0.0f);
    accum.assign ((size_t) size * 2, 0.0f);
    data.assign ((size_t) size * 2, 0.0f);
    magnitude.assign ((size_t) size / 2 + 1, 0.0f);
    smoothed.assign ((size_t) size / 2 + 1, 0.0f);
    window.resize ((size_t) size);
    for (int i = 0; i < size; ++i)   // square-root Hann at both ends
        window[(size_t) i] = std::sqrt (0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) size));
    smoothBins = juce::jmax (1, juce::roundToInt (180.0 / (sampleRate / size)));
    reset();
}

void Whisperizer::reset()
{
    std::fill (ring.begin(), ring.end(), 0.0f);
    std::fill (accum.begin(), accum.end(), 0.0f);
    write = read = sinceHop = 0;
}

void Whisperizer::frame() noexcept
{
    for (int i = 0; i < size; ++i)
        data[(size_t) i] = ring[(size_t) ((write + i) % size)] * window[(size_t) i];
    std::fill (data.begin() + size, data.end(), 0.0f);
    fft->performRealOnlyForwardTransform (data.data(), true);

    const int bins = size / 2 + 1;
    double energy = 0.0, smoothEnergy = 0.0;
    for (int k = 0; k < bins; ++k)
    {
        magnitude[(size_t) k] = std::hypot (data[(size_t) (2 * k)], data[(size_t) (2 * k + 1)]);
        energy += (double) magnitude[(size_t) k] * magnitude[(size_t) k];
    }
    for (int k = 0; k < bins; ++k)
    {
        float sum = 0.0f;
        int count = 0;
        for (int j = juce::jmax (0, k - smoothBins); j <= juce::jmin (bins - 1, k + smoothBins); ++j, ++count)
            sum += magnitude[(size_t) j];
        smoothed[(size_t) k] = sum / (float) count;
        smoothEnergy += (double) smoothed[(size_t) k] * smoothed[(size_t) k];
    }
    const float gain = smoothEnergy > 0.0 ? (float) std::sqrt (energy / smoothEnergy) : 0.0f;
    for (int k = 0; k < bins; ++k)
    {
        const float phase = juce::MathConstants<float>::twoPi * random.nextFloat();
        data[(size_t) (2 * k)] = smoothed[(size_t) k] * gain * std::cos (phase);
        data[(size_t) (2 * k + 1)] = k == 0 || k == size / 2 ? 0.0f : smoothed[(size_t) k] * gain * std::sin (phase);
    }
    for (int k = 1; k < size / 2; ++k)   // the mirrored half, so the inverse transform is the whole real signal
    {
        data[(size_t) (2 * (size - k))] = data[(size_t) (2 * k)];
        data[(size_t) (2 * (size - k) + 1)] = -data[(size_t) (2 * k + 1)];
    }
    fft->performRealOnlyInverseTransform (data.data());

    // frame sample i belongs to output time read + i (the output runs `size` samples behind the input). With
    // random phases the overlapping frames add up in power, not in amplitude: no 1/2 for the Hann overlap.
    const int total = (int) accum.size();
    for (int i = 0; i < size; ++i)
        accum[(size_t) ((read + i) % total)] += data[(size_t) i] * window[(size_t) i];
}

void Whisperizer::process (const float* input, float* output, int n) noexcept
{
    const int total = (int) accum.size();
    for (int i = 0; i < n; ++i)
    {
        ring[(size_t) write] = input[i];
        write = (write + 1) % size;
        output[i] = accum[(size_t) read];
        accum[(size_t) read] = 0.0f;
        read = (read + 1) % total;
        if (++sinceHop >= hop)
        {
            sinceHop = 0;
            frame();
        }
    }
}

} // namespace voxa::dsp
