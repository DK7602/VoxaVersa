#include "Articulation.h"
#include "../core/MusicTheory.h"
#include <juce_audio_basics/juce_audio_basics.h>

namespace voxa::sing
{

namespace
{
    using V = VoiceSetup;

    // Sung vowel targets (F1, F2, F3 in Hz). Based on Peterson & Barney (1952) and Hillenbrand et al.
    // (1995) averages, nudged towards the slightly darker, more open vowels of singing.
    const float maleVowels[V::numVowelTargets][3] = {
        { 750, 1220, 2500 },   // AA  father
        { 660, 1700, 2450 },   // AE  cat
        { 640, 1190, 2450 },   // AH  cut
        { 590,  900, 2450 },   // AO  caught
        { 500, 1400, 2450 },   // AX  schwa
        { 550, 1770, 2500 },   // EH  bed
        { 480, 1350, 1700 },   // ER  bird
        { 400, 1950, 2550 },   // IH  bit
        { 300, 2200, 2950 },   // IY  beat
        { 450, 1050, 2300 },   // UH  book
        { 320,  880, 2250 },   // UW  boot
        { 480, 1850, 2500 },   // EY  onset
        { 360, 2050, 2600 },   // EY  off-glide
        { 520,  900, 2400 },   // OW  onset
        { 400,  850, 2300 },   // OW  off-glide
        { 740, 1250, 2500 },   // AY  onset
        { 420, 1900, 2550 },   // AY  off-glide
        { 720, 1200, 2450 },   // AW  onset
        { 440,  950, 2300 },   // AW  off-glide
        { 570,  880, 2450 },   // OY  onset
        { 400, 1950, 2550 },   // OY  off-glide
    };

    const float femaleVowels[V::numVowelTargets][3] = {
        { 900, 1400, 2850 },   // AA
        { 820, 2000, 2900 },   // AE
        { 760, 1400, 2800 },   // AH
        { 650, 1000, 2750 },   // AO
        { 560, 1600, 2800 },   // AX
        { 650, 2100, 2950 },   // EH
        { 520, 1580, 1950 },   // ER
        { 450, 2350, 3000 },   // IH
        { 340, 2650, 3250 },   // IY
        { 490, 1180, 2700 },   // UH
        { 380,  980, 2650 },   // UW
        { 560, 2250, 2950 },   // EY onset
        { 420, 2450, 3050 },   // EY off-glide
        { 580, 1000, 2750 },   // OW onset
        { 450,  950, 2650 },   // OW off-glide
        { 880, 1450, 2850 },   // AY onset
        { 470, 2300, 3000 },   // AY off-glide
        { 860, 1400, 2800 },   // AW onset
        { 480, 1050, 2650 },   // AW off-glide
        { 630,  980, 2750 },   // OY onset
        { 450, 2350, 3000 },   // OY off-glide
    };

    // F4 .. F14. Above F8 these are the neutral-tube resonances that a uniform tract keeps producing;
    // without them a cascade loses ~20-40 dB between 4 and 8 kHz (Klatt's higher-pole correction).
    constexpr int numUpper = SynthFrame::numFormants - 3;
    const float maleUpper[numUpper]   = { 3400, 4300, 5200, 6300, 7400, 8500, 9550, 10600, 11650, 12700, 13750 };
    const float femaleUpper[numUpper] = { 4100, 4900, 5900, 7000, 8200, 9400, 10600, 11800, 13000, 14200, 15400 };
    const float upperBandwidth[numUpper] = { 250, 320, 450, 600, 750, 800, 900, 1000, 1100, 1200, 1300 };

    void vowelTargets (Ph p, int& a, int& b) noexcept
    {
        // Indexed by Ph (the vowels come first in the enum): onset target, off-glide target.
        static constexpr int targets[16][2] = {
            { V::vAA, V::vAA },    // AA
            { V::vAE, V::vAE },    // AE
            { V::vAH, V::vAH },    // AH
            { V::vAO, V::vAO },    // AO
            { V::vAWa, V::vAWb },  // AW
            { V::vAX, V::vAX },    // AX
            { V::vAYa, V::vAYb },  // AY
            { V::vEH, V::vEH },    // EH
            { V::vER, V::vER },    // ER
            { V::vEYa, V::vEYb },  // EY
            { V::vIH, V::vIH },    // IH
            { V::vIY, V::vIY },    // IY
            { V::vOWa, V::vOWb },  // OW
            { V::vOYa, V::vOYb },  // OY
            { V::vUH, V::vUH },    // UH
            { V::vUW, V::vUW },    // UW
        };
        static_assert ((int) Ph::UW == 15, "vowel table must follow the Ph enum");

        if (! isVowel (p))
        {
            a = b = V::vAX;
            return;
        }
        a = targets[(int) p][0];
        b = targets[(int) p][1];
    }

    // Consonant loci (adult male, Hz) - the formant positions that transitions point towards.
    struct Locus { float f1, f2, f3; };

    Locus locusForPlace (Place place) noexcept
    {
        switch (place)
        {
            case Place::Labial:       return { 250,  900, 2200 };
            case Place::LabioDental:  return { 300, 1100, 2300 };
            case Place::Dental:       return { 300, 1450, 2550 };
            case Place::Alveolar:     return { 280, 1700, 2600 };
            case Place::PostAlveolar: return { 280, 1900, 2500 };
            case Place::Palatal:      return { 270, 2150, 2900 };
            case Place::Velar:
            case Place::Glottal:
            case Place::None:
            default:                  return { 280, 1500, 2450 };
        }
    }

    struct FricShape { float f1, b1, g1, f2, b2, g2, bypass, highpass, amplitude; };

    FricShape fricationShape (Ph p) noexcept
    {
        //                        f1    b1    g1    f2    b2    g2     bypass highpass amplitude
        if (p == Ph::S)  return { 5500, 2200, 1.0f, 8000, 3000, 0.6f,  0.03f, 3200, 0.55f };
        if (p == Ph::Z)  return { 5500, 2200, 1.0f, 8000, 3000, 0.6f,  0.03f, 3200, 0.30f };
        if (p == Ph::SH) return { 2800, 1000, 1.0f, 4300, 2000, 0.55f, 0.0f,  1700, 0.80f };
        if (p == Ph::ZH) return { 2800, 1000, 1.0f, 4300, 2000, 0.55f, 0.0f,  1700, 0.40f };
        if (p == Ph::CH) return { 2900, 1100, 1.0f, 4500, 2000, 0.55f, 0.0f,  1700, 0.80f };
        if (p == Ph::JH) return { 2900, 1100, 1.0f, 4500, 2000, 0.55f, 0.0f,  1700, 0.45f };
        if (p == Ph::F)  return { 3500, 5000, 0.4f, 8000, 4000, 0.35f, 0.25f, 1200, 0.16f };
        if (p == Ph::V)  return { 3500, 5000, 0.4f, 8000, 4000, 0.35f, 0.25f, 1200, 0.08f };
        if (p == Ph::TH) return { 4500, 4500, 0.4f, 7500, 3500, 0.35f, 0.2f,  1500, 0.15f };
        if (p == Ph::DH) return { 4500, 4500, 0.4f, 7500, 3500, 0.35f, 0.2f,  1500, 0.07f };
        return { 4000, 3000, 0.5f, 7000, 3000, 0.3f, 0.2f, 1000, 0.2f };
    }

    FricShape burstShape (Ph p, float velarCentre) noexcept
    {
        if (p == Ph::P)  return { 900,  1500, 0.6f, 2500, 3000, 0.4f,  0.4f,  400,  0.55f };
        if (p == Ph::B)  return { 900,  1500, 0.6f, 2500, 3000, 0.4f,  0.4f,  400,  0.30f };
        if (p == Ph::T)  return { 4200, 2000, 1.0f, 6500, 3000, 0.6f,  0.15f, 2200, 0.80f };
        if (p == Ph::D)  return { 4200, 2000, 1.0f, 6500, 3000, 0.6f,  0.15f, 2200, 0.45f };
        if (p == Ph::K)  return { velarCentre, 800, 1.0f, 4000, 2000, 0.35f, 0.05f, 900, 0.75f };
        if (p == Ph::G)  return { velarCentre, 800, 1.0f, 4000, 2000, 0.35f, 0.05f, 900, 0.40f };
        return { 4000, 2000, 1.0f, 6000, 3000, 0.5f, 0.1f, 1000, 0.5f };
    }

    /** Estimated output power (dB) of a voiced sound: harmonics of a -6 dB/octave source through the
        first five formant resonances. */
    float tractPowerDb (const float* formants, const float* bandwidths, float f0) noexcept
    {
        f0 = juce::jmax (40.0f, f0);
        const int harmonics = juce::jlimit (1, 60, (int) (5000.0f / f0));
        double power = 0.0;
        for (int h = 1; h <= harmonics; ++h)
        {
            const double f = h * (double) f0;
            double gain = 1.0;
            for (int i = 0; i < 5; ++i)
            {
                const double x = f / formants[i];
                const double k = bandwidths[i] / formants[i];
                gain /= (1.0 - x * x) * (1.0 - x * x) + (k * x) * (k * x);
            }
            power += gain / ((double) h * h);
        }
        return (float) (10.0 * std::log10 (power + 1.0e-20));
    }

    float smoothstep (float x) noexcept
    {
        x = juce::jlimit (0.0f, 1.0f, x);
        return x * x * (3.0f - 2.0f * x);
    }
}

//==============================================================================
float articSmoothingSeconds (int param) noexcept
{
    switch (param)
    {
        case apF1: case apF2: case apF3:     return 0.014f;
        case apB1: case apB2: case apB3:     return 0.012f;
        case apUpperDamping:                 return 0.008f;
        case apNasalZero:                    return 0.006f;
        case apNasality:                     return 0.004f;
        case apAV:                           return 0.004f;
        case apAH:                           return 0.005f;
        case apAF:                           return 0.0015f;
        case apFricF1: case apFricF2:
        case apFricB1: case apFricB2:
        case apFricHighpass:                 return 0.003f;
        default:                             return 0.002f;
    }
}

bool articParamIsFrequency (int param) noexcept
{
    return param == apF1 || param == apF2 || param == apF3 || param == apNasalZero
        || param == apFricF1 || param == apFricF2 || param == apFricHighpass;
}

float tunedFirstFormant (float f1, float f0) noexcept
{
    return juce::jmax (f1, 1.08f * f0 + 30.0f);
}

//==============================================================================
void VoiceSetup::prepare (const VoiceTraits& t) noexcept
{
    float mix = 0.5f, nominal = 0.0f;
    if (t.gender == VoiceGender::Male)        { mix = 0.0f; nominal = -1.0f; }
    else if (t.gender == VoiceGender::Female) { mix = 1.0f; nominal = 1.0f; }

    const float residual = juce::jlimit (0.7f, 1.4f, std::exp2 ((t.formantSemis - nominal) / 12.0f));

    auto blend = [mix, residual] (float male, float female)
    {
        return std::pow (male, 1.0f - mix) * std::pow (female, mix) * residual;
    };

    for (int v = 0; v < numVowelTargets; ++v)
        for (int k = 0; k < 3; ++k)
            vowels[(size_t) v][(size_t) k] = blend (maleVowels[v][k], femaleVowels[v][k]);

    for (int k = 0; k < numUpper; ++k)
    {
        upperFormants[(size_t) k + 3] = blend (maleUpper[k], femaleUpper[k]);
        upperBandwidths[(size_t) k + 3] = upperBandwidth[k] * (1.0f + 0.1f * mix);
    }

    consonantScale = std::pow (1.15f, mix) * residual;
    bandwidthScale = 1.0f + 0.1f * mix;

    const float breath = juce::jlimit (0.0f, 1.0f, t.breathiness);
    const float rasp = juce::jlimit (0.0f, 1.0f, t.rasp);
    const float jitterCents = juce::jlimit (0.0f, 30.0f, t.jitterCents);

    source.breathiness = 0.02f + 4.5f * breath * breath;
    source.rasp = rasp;
    source.jitter = 0.0015f + 0.0006f * jitterCents + 0.01f * rasp;
    source.shimmer = 0.015f + 0.06f * rasp + 0.02f * breath;

    baseRd = juce::jlimit (0.45f, 2.6f, 0.95f + 1.1f * breath - 0.07f * t.tiltDb - 0.15f * rasp);
    voicingGain = 1.0f - 0.75f * juce::jmax (0.0f, breath - 0.5f) / 0.5f;
    aspirationLevel = source.breathiness;

    vibratoRateHz = juce::jlimit (3.0f, 9.0f, t.vibratoRateHz);
    vibratoDepthSemis = juce::jlimit (0.0f, 1.5f, t.vibratoDepthCents / 100.0f);
    vibratoOnsetSeconds = juce::jlimit (0.0f, 2.0f, t.vibratoOnsetMs / 1000.0f);
    scoopSemis = juce::jlimit (0.0f, 3.0f, t.scoopCents / 100.0f);
    scoopSeconds = juce::jlimit (0.02f, 0.4f, t.scoopMs / 1000.0f);
    driftSemis = jitterCents * 0.006f;
    portamentoSeconds = juce::jlimit (0.0f, 0.5f, t.singPortamentoMs / 1000.0f);
    legato = juce::jlimit (0.0f, 1.0f, t.singLegato);
    busy = juce::jlimit (0.0f, 1.0f, t.singRhythmBusy);
    fallSemis = 0.3f * juce::jlimit (0.0f, 1.0f, jitterCents / 3.0f);

    loudnessReferenceDb = vowelPowerDb (Ph::AA, music::midiToHz ((float) t.centreMidi()));
}

float VoiceSetup::vowelLoudnessCompensation (Ph vowel, float f0) const noexcept
{
    if (! isVowel (vowel))
        return 1.0f;

    const float db = juce::jlimit (-6.0f, 9.0f, 0.7f * (loudnessReferenceDb - vowelPowerDb (vowel, f0)));
    return juce::Decibels::decibelsToGain (db);
}

float VoiceSetup::vowelPowerDb (Ph vowel, float f0) const noexcept
{
    float f[5], b[5];
    vowelFormants (vowel, 0.0f, f);
    f[0] = tunedFirstFormant (f[0], f0);
    f[1] = juce::jmax (f[1], f[0] + 200.0f);
    f[2] = juce::jmax (f[2], f[1] + 250.0f);
    f[3] = juce::jmax (upperFormants[3], f[2] + 300.0f);
    f[4] = juce::jmax (upperFormants[4], f[3] + 300.0f);
    b[0] = juce::jmax ((0.07f * f[0] + 35.0f) * bandwidthScale, 0.12f * f0);
    b[1] = (0.05f * f[1] + 50.0f) * bandwidthScale;
    b[2] = (0.04f * f[2] + 60.0f) * bandwidthScale;
    b[3] = upperBandwidths[3];
    b[4] = upperBandwidths[4];
    return tractPowerDb (f, b, f0);
}

void VoiceSetup::vowelFormants (Ph vowel, float glide, float* f) const noexcept
{
    int a = vAX, b = vAX;
    vowelTargets (vowel, a, b);
    const auto& va = vowels[(size_t) a];
    const auto& vb = vowels[(size_t) b];

    if (a == b || glide <= 0.0f)
    {
        f[0] = va[0]; f[1] = va[1]; f[2] = va[2];
        return;
    }

    for (int k = 0; k < 3; ++k)
        f[k] = va[(size_t) k] * std::pow (vb[(size_t) k] / va[(size_t) k], juce::jlimit (0.0f, 1.0f, glide));
}

//==============================================================================
void articulate (const PhoneEvent& e, float t, const VoiceSetup& vs, ArticFrame& out) noexcept
{
    auto& v = out.v;
    out.formantsValid = true;

    const float cs = vs.consonantScale;
    const float bs = vs.bandwidthScale;

    v.fill (0.0f);
    v[apUpperDamping] = 1.0f;
    v[apNasalZero] = 1500.0f * cs;
    v[apFricF1] = 5000.0f * cs;
    v[apFricB1] = 2000.0f * cs;
    v[apFricF2] = 8000.0f * cs;
    v[apFricB2] = 3000.0f * cs;
    v[apFricHighpass] = 1000.0f * cs;

    auto setFormants = [&] (float f1, float f2, float f3)
    {
        v[apF1] = f1;
        v[apF2] = f2;
        v[apF3] = f3;
        v[apB1] = (0.07f * f1 + 35.0f) * bs;
        v[apB2] = (0.05f * f2 + 50.0f) * bs;
        v[apB3] = (0.04f * f3 + 60.0f) * bs;
    };

    auto setFrication = [&] (const FricShape& s)
    {
        v[apFricF1] = s.f1 * cs;
        v[apFricB1] = s.b1 * cs;
        v[apFricG1] = s.g1;
        v[apFricF2] = s.f2 * cs;
        v[apFricB2] = s.b2 * cs;
        v[apFricG2] = s.g2;
        v[apFricBypass] = s.bypass;
        v[apFricHighpass] = s.highpass * cs;
    };

    float context[3];
    vs.vowelFormants (e.contextVowel, 0.0f, context);

    const auto& info = phonemeInfo (e.ph);

    // Velar consonants take their place from the neighbouring vowel ("velar pinch").
    const float velarF2 = juce::jlimit (1100.0f * cs, 2450.0f * cs, context[1] + 150.0f * cs);
    const float velarF3 = juce::jmax (velarF2 + 300.0f * cs, context[2] - 200.0f * cs);

    auto setLocus = [&] (Place place)
    {
        if (place == Place::Velar)
        {
            setFormants (260.0f * cs, velarF2, velarF3);
            return;
        }
        const auto l = locusForPlace (place);
        setFormants (l.f1 * cs, l.f2 * cs, l.f3 * cs);
    };

    switch (info.manner)
    {
        case Manner::Vowel:
        case Manner::Diphthong:
        {
            float glide = 0.0f;
            if (info.manner == Manner::Diphthong && t > e.glideStart)
                glide = smoothstep ((t - e.glideStart) / juce::jmax (0.01f, e.glideTime));

            float f[3];
            vs.vowelFormants (e.ph, glide, f);
            setFormants (f[0], f[1], f[2]);
            v[apAV] = 1.0f;
            break;
        }

        case Manner::Silence:
        {
            out.formantsValid = false;
            setFormants (context[0], context[1], context[2]);
            break;
        }

        case Manner::Breath:
        {
            const float x = juce::jlimit (0.0f, 1.0f, t / juce::jmax (0.05f, e.duration));
            const float env = std::pow (std::sin (juce::MathConstants<float>::pi * x), 1.5f);
            setFormants (600.0f * cs, 1300.0f * cs, 2500.0f * cs);
            v[apB1] = 300.0f;
            v[apB2] = 350.0f;
            v[apB3] = 450.0f;
            v[apUpperDamping] = 2.0f;
            v[apAH] = env;
            break;
        }

        case Manner::Aspirate:
        {
            setFormants (context[0], context[1], context[2]);
            v[apB1] *= 2.2f;
            v[apB2] *= 1.3f;
            v[apAH] = juce::jmin (1.0f, t / 0.012f);
            break;
        }

        case Manner::Fricative:
        {
            setLocus (info.place);
            const auto shape = fricationShape (e.ph);
            setFrication (shape);
            v[apAF] = shape.amplitude;
            if (info.voiced)
            {
                v[apAV] = 0.2f;
                v[apUpperDamping] = 1.5f;
            }
            break;
        }

        case Manner::Stop:
        {
            const bool voiced = info.voiced;
            const float duration = juce::jmax (0.02f, e.duration);
            const float burst = juce::jmin (voiced ? 0.006f : 0.010f, 0.2f * duration);
            float vot;
            if (voiced)
                vot = juce::jmin (0.012f, 0.2f * duration);
            else
                vot = e.aspirated ? juce::jlimit (0.02f, 0.06f, 0.35f * duration) : juce::jmin (0.015f, 0.2f * duration);

            const float closure = e.released ? juce::jmax (0.0f, duration - burst - vot) : duration;

            setLocus (info.place);
            const float velarCentre = 0.5f * (velarF2 + velarF3) / cs;

            if (t < closure)
            {
                v[apF1] = 200.0f * cs;
                v[apB2] *= 1.5f;
                v[apB3] *= 1.5f;
                v[apUpperDamping] = 3.0f;
                v[apAV] = voiced ? 0.06f : 0.0f;
            }
            else if (t < closure + burst)
            {
                const auto shape = burstShape (e.ph, velarCentre);
                setFrication (shape);
                v[apAF] = shape.amplitude;
                v[apAV] = voiced ? 0.15f : 0.0f;
            }
            else if (voiced)
            {
                v[apAV] = 0.3f;
            }
            else
            {
                // Aspiration: the formants move from the locus towards the vowel while the noise
                // excites them (a strong place cue).
                const float x = juce::jlimit (0.0f, 1.0f, (t - closure - burst) / juce::jmax (0.005f, vot));
                const auto shape = burstShape (e.ph, velarCentre);
                setFrication (shape);
                v[apAF] = 0.15f * shape.amplitude * (1.0f - x);
                const float move = 0.85f * x;
                setFormants (v[apF1] * std::pow (context[0] / v[apF1], move),
                             v[apF2] * std::pow (context[1] / v[apF2], move),
                             v[apF3] * std::pow (context[2] / v[apF3], move));
                v[apB1] *= 2.0f;
                v[apAH] = 0.75f * (1.0f - 0.4f * juce::jlimit (0.0f, 1.0f, x));
            }
            break;
        }

        case Manner::Affricate:
        {
            const bool voiced = info.voiced;
            const float duration = juce::jmax (0.03f, e.duration);
            const float closure = juce::jmax (0.02f, 0.35f * duration);
            setLocus (Place::PostAlveolar);

            if (t < closure)
            {
                v[apF1] = 200.0f * cs;
                v[apUpperDamping] = 3.0f;
                v[apAV] = voiced ? 0.1f : 0.0f;
            }
            else
            {
                const auto shape = fricationShape (e.ph);
                setFrication (shape);
                const float x = (t - closure) / juce::jmax (0.01f, duration - closure);
                v[apAF] = shape.amplitude * (x < 0.15f ? 1.25f : 1.0f);
                v[apAV] = voiced ? 0.4f : 0.0f;
            }
            break;
        }

        case Manner::Nasal:
        {
            setLocus (info.place);
            v[apF1] = 270.0f * cs;
            v[apB1] = 45.0f * bs;
            v[apB2] = 350.0f * bs;
            v[apB3] = 450.0f * bs;
            v[apUpperDamping] = 2.5f;
            v[apNasality] = 1.0f;
            v[apNasalZero] = (e.ph == Ph::M ? 1100.0f : (e.ph == Ph::N ? 1700.0f : 2700.0f)) * cs;
            v[apAV] = e.syllabic ? 0.75f : 0.45f;
            break;
        }

        case Manner::Liquid:
        {
            if (e.ph == Ph::L)
                setFormants (360.0f * cs, 1150.0f * cs, 2650.0f * cs);
            else
                setFormants (420.0f * cs, 1100.0f * cs, 1550.0f * cs);
            v[apAV] = e.syllabic ? 0.9f : 0.6f;
            break;
        }

        case Manner::Glide:
        {
            if (e.ph == Ph::W)
                setFormants (300.0f * cs, 700.0f * cs, 2200.0f * cs);
            else
                setFormants (270.0f * cs, 2150.0f * cs, 2900.0f * cs);
            v[apAV] = e.ph == Ph::W ? 0.45f : 0.55f;
            break;
        }
    }
}

void makeSynthFrame (const ArticFrame& a, const VoiceSetup& vs, float f0, float rd, SynthFrame& out) noexcept
{
    const auto& v = a.v;
    out.f0 = f0;
    out.rd = rd;
    out.av = juce::jmax (0.0f, v[apAV]) * vs.voicingGain;
    out.ah = juce::jmax (0.0f, v[apAH]);
    out.af = juce::jmax (0.0f, v[apAF]);

    const float f1 = tunedFirstFormant (v[apF1], f0);
    const float f2 = juce::jmax (v[apF2], f1 + 200.0f);
    const float f3 = juce::jmax (v[apF3], f2 + 250.0f);
    out.formants[0] = f1;
    out.formants[1] = f2;
    out.formants[2] = f3;
    out.bandwidths[0] = juce::jmax (v[apB1], 0.12f * f0);
    out.bandwidths[1] = v[apB2];
    out.bandwidths[2] = v[apB3];

    float previous = f3;
    for (int k = 3; k < SynthFrame::numFormants; ++k)
    {
        previous = juce::jmax (vs.upperFormants[(size_t) k], previous + 300.0f);
        out.formants[(size_t) k] = previous;
        out.bandwidths[(size_t) k] = vs.upperBandwidths[(size_t) k] * juce::jmax (1.0f, v[apUpperDamping]);
    }

    out.nasalZeroHz = v[apNasalZero];
    out.nasality = juce::jlimit (0.0f, 1.0f, v[apNasality]);
    out.fricFreq = { v[apFricF1], v[apFricF2] };
    out.fricBandwidth = { juce::jmax (100.0f, v[apFricB1]), juce::jmax (100.0f, v[apFricB2]) };
    out.fricGain = { v[apFricG1], v[apFricG2] };
    out.fricBypass = v[apFricBypass];
    out.fricHighpass = v[apFricHighpass];
}

} // namespace voxa::sing
