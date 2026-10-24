#include "VocalSynth.h"

#include <complex>
#include <juce_audio_basics/juce_audio_basics.h>

namespace voxa::sing
{

namespace
{
    using Complex = std::complex<double>;

    /** In-place iterative radix-2 FFT. The inverse is scaled by 1/n. Only used to build tables. */
    void fft (std::vector<Complex>& a, bool inverse)
    {
        const size_t n = a.size();
        for (size_t i = 1, j = 0; i < n; ++i)
        {
            size_t bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                std::swap (a[i], a[j]);
        }

        for (size_t len = 2; len <= n; len <<= 1)
        {
            const double angle = juce::MathConstants<double>::twoPi / (double) len * (inverse ? 1.0 : -1.0);
            const Complex wl (std::cos (angle), std::sin (angle));
            for (size_t i = 0; i < n; i += len)
            {
                Complex w (1.0, 0.0);
                for (size_t j = 0; j < len / 2; ++j)
                {
                    const auto u = a[i + j];
                    const auto v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wl;
                }
            }
        }

        if (inverse)
            for (auto& x : a)
                x /= (double) n;
    }

    /** One period of the LF glottal flow derivative (Fant, Liljencrants & Lin 1985) parameterised
        by Rd (Fant 1995), with the excitation peak Ee normalised to 1. */
    std::vector<double> lfPeriod (double rd, int n, double& closingInstant)
    {
        rd = juce::jlimit (0.3, 2.7, rd);
        const double ra = (-1.0 + 4.8 * rd) / 100.0;
        const double rk = (22.4 + 11.8 * rd) / 100.0;
        const double rg = rk / (4.0 * (0.11 * rd / (0.5 + 1.2 * rk) - ra));
        const double tp = 1.0 / (2.0 * rg);
        const double te = juce::jlimit (tp + 1.0e-3, 0.97, tp * (1.0 + rk));
        const double ta = juce::jmax (1.0e-4, ra);
        const double tb = 1.0 - te;

        // Return phase: eps * ta = 1 - exp (-eps * tb)
        double eps = 1.0 / ta;
        for (int it = 0; it < 60; ++it)
        {
            const double e = std::exp (-eps * tb);
            const double f = eps * ta - 1.0 + e;
            const double df = ta - tb * e;
            const double step = f / df;
            eps = juce::jmax (1.0e-3, eps - step);
            if (std::abs (step) < 1.0e-12)
                break;
        }

        const double wg = juce::MathConstants<double>::pi / tp;
        const double s = std::sin (wg * te), c = std::cos (wg * te);

        // Net flow over the period must be zero: solve for the growth factor alpha.
        auto openArea = [&] (double alpha)
        {
            return (-1.0 / s) * ((alpha * s - wg * c) + wg * std::exp (-alpha * te)) / (alpha * alpha + wg * wg);
        };
        const double returnArea = -(1.0 / (eps * ta)) * ((1.0 - std::exp (-eps * tb)) / eps - tb * std::exp (-eps * tb));
        auto g = [&] (double alpha) { return openArea (alpha) + returnArea; };

        double lo = 0.0, hi = 1.0;
        for (int i = 0; i < 40 && g (lo) < 0.0; ++i)
            lo -= 2.0 * (i + 1);
        for (int i = 0; i < 40 && g (hi) > 0.0; ++i)
            hi *= 2.0;
        for (int it = 0; it < 200; ++it)
        {
            const double mid = 0.5 * (lo + hi);
            if (g (mid) > 0.0)
                lo = mid;
            else
                hi = mid;
        }
        const double alpha = 0.5 * (lo + hi);

        std::vector<double> y ((size_t) n);
        double mean = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double t = (double) i / n;
            double v;
            if (t <= te)
                v = -std::exp (alpha * (t - te)) * std::sin (wg * t) / s;
            else
                v = -(1.0 / (eps * ta)) * (std::exp (-eps * (t - te)) - std::exp (-eps * tb));
            y[(size_t) i] = v;
            mean += v;
        }

        mean /= n;
        for (auto& v : y)
            v -= mean;

        closingInstant = te;
        return y;
    }
}

//==============================================================================
int GlottalTables::harmonicsAtLevel (int level) noexcept
{
    return juce::jmax (1, (int) std::floor (1023.0 * std::exp2 (-0.5 * level)));
}

int GlottalTables::levelForMaxHarmonic (float maxHarmonic) noexcept
{
    for (int level = 0; level < numLevels; ++level)
        if ((float) harmonicsAtLevel (level) <= maxHarmonic)
            return level;
    return numLevels - 1;
}

const GlottalTables& GlottalTables::get()
{
    static const GlottalTables instance;
    return instance;
}

GlottalTables::GlottalTables()
{
    constexpr int sourceSize = 8192;
    data.resize ((size_t) numRd * numLevels * (tableSize + 1));

    std::vector<Complex> spectrum ((size_t) sourceSize), table ((size_t) tableSize);

    for (int r = 0; r < numRd; ++r)
    {
        double te = 0.6;
        const auto period = lfPeriod (rdMin + rdStep * r, sourceSize, te);
        closing[(size_t) r] = (float) te;

        for (int i = 0; i < sourceSize; ++i)
            spectrum[(size_t) i] = Complex (period[(size_t) i], 0.0);
        fft (spectrum, false);

        // Harmonic k has complex amplitude 2 X[k] / N. Normalise the energy of the first 12 harmonics
        // so that changing Rd changes the spectral balance rather than the loudness.
        double lowEnergy = 0.0;
        for (int k = 1; k <= 12; ++k)
            lowEnergy += std::norm (2.0 * spectrum[(size_t) k] / (double) sourceSize) * 0.5;
        const double normalise = 0.25 / std::sqrt (juce::jmax (1.0e-12, lowEnergy));

        for (int level = 0; level < numLevels; ++level)
        {
            const int harmonics = juce::jmin (tableSize / 2 - 1, harmonicsAtLevel (level));
            std::fill (table.begin(), table.end(), Complex());
            for (int k = 1; k <= harmonics; ++k)
            {
                const auto amplitude = 2.0 * spectrum[(size_t) k] / (double) sourceSize * normalise;
                table[(size_t) k] = amplitude * (0.5 * tableSize);
                table[(size_t) (tableSize - k)] = std::conj (table[(size_t) k]);
            }
            fft (table, true);

            auto* dest = data.data() + ((size_t) r * numLevels + (size_t) level) * (size_t) (tableSize + 1);
            for (int i = 0; i < tableSize; ++i)
                dest[i] = (float) table[(size_t) i].real();
            dest[tableSize] = dest[0];
        }
    }
}

//==============================================================================
void VocalSynth::prepare (double newSampleRate, juce::uint32 seed)
{
    sampleRate = newSampleRate > 0.0 ? newSampleRate : 48000.0;
    sr = (float) sampleRate;
    invSr = 1.0f / sr;
    tables = &GlottalTables::get();

    noiseA.seed (seed * 2654435761u + 1u);
    noiseB.seed (seed * 2246822519u + 7u);
    noiseC.seed (seed * 3266489917u + 13u);

    aspHpCoeff = std::exp (-juce::MathConstants<float>::twoPi * 500.0f / sr);

    // First-order high shelf (+10 dB above ~5 kHz) on the tract output: the LF closure and a finite
    // formant cascade leave the top octaves darker than a real voice.
    shelfCoeff = std::exp (-juce::MathConstants<float>::twoPi * 3000.0f / sr);
    shelfGain = juce::Decibels::decibelsToGain (10.0f) - 1.0f;
    dcCoeff = std::exp (-juce::MathConstants<float>::twoPi * 20.0f / sr);
    reset();
}

void VocalSynth::reset() noexcept
{
    for (auto& f : cascade)
        f.reset();
    nasal.reset();
    fric1.reset();
    fric2.reset();
    fricHp.reset();
    phase = 0.0f;
    aspHpState = 0.0f;
    shelfState = 0.0f;
    dcX = dcY = 0.0f;
    lastAv = lastAh = lastAf = 0.0f;
    lastF0 = 120.0f;
    lastRd = 1.0f;
    if (tables == nullptr)
        tables = &GlottalTables::get();
    startPeriod (lastF0, lastRd);
}

void VocalSynth::setSourceCharacter (const SourceCharacter& c) noexcept
{
    character = c;
}

void VocalSynth::startPeriod (float f0, float rd) noexcept
{
    oddPeriod = ! oddPeriod;
    const float rasp = character.rasp;

    float freq = 1.0f + character.jitter * noiseC.gaussian();
    float amp = 1.0f + character.shimmer * noiseC.gaussian();

    if (rasp > 0.0f)
    {
        // Period doubling (alternating long/short, strong/weak pulses) gives the subharmonic growl;
        // occasional weak pulses add fry-like irregularity.
        freq *= 1.0f + 0.05f * rasp * (oddPeriod ? 1.0f : -1.0f);
        amp *= oddPeriod ? 1.0f : 1.0f - 0.45f * rasp;
        if (0.5f * (noiseC.next() + 1.0f) < 0.12f * rasp)
            amp *= 0.35f;
    }

    periodFreqFactor = juce::jlimit (0.8f, 1.25f, freq);
    periodAmp = juce::jmax (0.0f, amp);

    const float rdPos = juce::jlimit (0.0f, (float) GlottalTables::numRd - 1.001f, (rd - GlottalTables::rdMin) / GlottalTables::rdStep);
    const int index = (int) rdPos;
    rdFrac = rdPos - (float) index;

    const float maxFrequency = juce::jmin (0.45f * sr, 20000.0f);
    const int level = GlottalTables::levelForMaxHarmonic (maxFrequency / juce::jmax (20.0f, f0 * periodFreqFactor));
    tableA = tables->table (index, level);
    tableB = tables->table (index + 1, level);
    closing = tables->closingInstant (index) + rdFrac * (tables->closingInstant (index + 1) - tables->closingInstant (index));
}

void VocalSynth::process (const SynthFrame& f, float* output, int numSamples) noexcept
{
    if (numSamples <= 0)
        return;

    numActiveFormants = 0;
    for (int i = 0; i < SynthFrame::numFormants; ++i)
    {
        if (f.formants[(size_t) i] >= 0.45f * sr)
            break;
        cascade[(size_t) i].set (f.formants[(size_t) i], f.bandwidths[(size_t) i], sr);
        numActiveFormants = i + 1;
    }

    nasal.set (f.nasalZeroHz, juce::jmax (80.0f, 0.12f * f.nasalZeroHz), sr);
    fric1.set (f.fricFreq[0], f.fricBandwidth[0], sr);
    fric2.set (f.fricFreq[1], f.fricBandwidth[1], sr);
    fricHp.set (f.fricHighpass, 1.4f * f.fricHighpass, sr);

    const float nasality = juce::jlimit (0.0f, 1.0f, f.nasality);
    const float g1 = f.fricGain[0], g2 = f.fricGain[1], gb = f.fricBypass;
    const float breath = character.breathiness;

    // Calibrated so that a sustained vowel at av = 1 peaks around 0.5 and the noise branches sit at
    // natural levels relative to it (see Tests/sing).
    constexpr float voiceGain = 0.35f;
    constexpr float aspirationGain = 0.07f;
    constexpr float fricationGain = 0.45f;

    const float step = 1.0f / (float) numSamples;
    float t = 0.0f;

    for (int s = 0; s < numSamples; ++s)
    {
        t += step;
        const float f0 = lastF0 + (f.f0 - lastF0) * t;
        const float av = lastAv + (f.av - lastAv) * t;
        const float ah = lastAh + (f.ah - lastAh) * t;
        const float af = lastAf + (f.af - lastAf) * t;

        phase += juce::jmax (20.0f, f0) * periodFreqFactor * invSr;
        if (phase >= 1.0f)
        {
            phase -= std::floor (phase);
            startPeriod (f0, lastRd + (f.rd - lastRd) * t);
        }

        const float pos = phase * (float) GlottalTables::tableSize;
        const int index = juce::jmin ((int) pos, GlottalTables::tableSize - 1);
        const float frac = pos - (float) index;
        const float a = tableA[index] + frac * (tableA[index + 1] - tableA[index]);
        const float b = tableB[index] + frac * (tableB[index + 1] - tableB[index]);
        const float glottal = (a + rdFrac * (b - a)) * periodAmp;

        // Aspiration: high-passed noise, strongest while the glottis is open.
        const float white = noiseA.next();
        aspHpState = white + aspHpCoeff * (aspHpState - white);
        const float aspiration = white - aspHpState;
        const float gate = phase < closing ? 1.0f : 0.3f;
        const float aspAmp = ah + av * breath * gate;

        float x = voiceGain * av * glottal + aspirationGain * aspAmp * aspiration;

        nasal.lowpass (x);
        x -= nasality * nasal.k * nasal.bp;

        for (int i = 0; i < numActiveFormants; ++i)
            x = cascade[(size_t) i].lowpass (x);

        const float n2 = fricHp.highpass (noiseB.next());
        const float frication = gb * n2 + g1 * fric1.bandpass (n2) + g2 * fric2.bandpass (n2);

        shelfState = x + shelfCoeff * (shelfState - x);
        x += shelfGain * (x - shelfState);

        const float y = x + fricationGain * af * frication;

        dcY = y - dcX + dcCoeff * dcY;
        dcX = y;
        output[s] = dcY;
    }

    lastF0 = f.f0;
    lastAv = f.av;
    lastAh = f.ah;
    lastAf = f.af;
    lastRd = f.rd;
}

} // namespace voxa::sing
