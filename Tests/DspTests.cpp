#include <juce_dsp/juce_dsp.h>
#include <functional>
#include "dsp/PitchDetector.h"
#include "dsp/PitchMapper.h"
#include "dsp/VoiceChain.h"
#include "dsp/VoiceGenderEstimator.h"
#include "dsp/VocalFx.h"
#include "voices/VoiceLibrary.h"
#include "TestSignals.h"

namespace voxa
{

namespace
{
    using test::makeVoice;

    float medianPitch (const std::vector<float>& x, double sr, int from, int to)
    {
        auto frames = dsp::PitchDetector::analyse (x.data() + from, to - from, sr, 256, 50.0f, 1500.0f);
        std::vector<float> f;
        for (auto& fr : frames)
            if (fr.voiced)
                f.push_back (fr.frequencyHz);
        if (f.empty())
            return 0.0f;
        std::sort (f.begin(), f.end());
        return f[f.size() / 2];
    }

    float centsBetween (float a, float b) { return 1200.0f * std::log2 (a / b); }

    dsp::ChainSettings neutralSettings (const VoiceProfile& voice)
    {
        dsp::ChainSettings s;
        s.pitch.traits = voice.traits;
        s.pitch.expression = false;
        s.formantShift = false;
        s.intensity = 0.0f;
        s.vibe = false;
        s.deEsser = 0.0f;
        s.compressor = 0.0f;
        s.reverbMix = 0.0f;
        s.lowCutHz = 20.0f;
        return s;
    }
}

class PitchDetectorTests : public juce::UnitTest
{
public:
    PitchDetectorTests() : juce::UnitTest ("Pitch detector", "DSP") {}

    void runTest() override
    {
        for (double sr : { 44100.0, 48000.0, 96000.0 })
        {
            beginTest ("Accuracy at " + juce::String (sr) + " Hz");
            for (float f0 : { 82.4f, 110.0f, 196.0f, 261.6f, 440.0f, 659.3f, 987.8f })
            {
                auto x = makeVoice (sr, (int) sr, [f0] (int) { return f0; });
                const float measured = medianPitch (x, sr, (int) (sr * 0.2), (int) sr);
                expect (std::abs (centsBetween (measured, f0)) < 5.0f,
                        "f0 " + juce::String (f0) + " measured " + juce::String (measured));
            }
        }

        beginTest ("Silence and noise are unvoiced");
        {
            const double sr = 48000.0;
            std::vector<float> silence ((size_t) sr, 0.0f);
            expectEquals (medianPitch (silence, sr, 0, (int) sr), 0.0f);

            dsp::Noise noise;
            std::vector<float> n ((size_t) sr);
            for (auto& v : n) v = noise.next() * 0.3f;
            auto frames = dsp::PitchDetector::analyse (n.data(), (int) n.size(), sr, 256);
            int voiced = 0;
            for (auto& f : frames) voiced += f.voiced ? 1 : 0;
            expect (voiced < (int) frames.size() / 10, "noise voiced frames: " + juce::String (voiced));
        }

        beginTest ("A weak fundamental is not read an octave high");
        {
            // 2 s of an ordinary 150 Hz voice, then the same pitch with a weak fundamental and strong even
            // harmonics: half the period looks periodic enough for plain YIN, which then reads 300 Hz.
            const double sr = 48000.0;
            std::vector<float> x ((size_t) (sr * 4));
            double phase = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                const double t = (double) i / sr;
                phase += juce::MathConstants<double>::twoPi * 150.0 * (1.0 + 0.004 * std::sin (juce::MathConstants<double>::twoPi * 5.5 * t)) / sr;
                double v = 0.0;
                if (t < 2.0)
                    for (int k = 1; k < 12; ++k)
                        v += 0.6 / k * std::sin (k * phase);
                else
                    v = 0.08 * std::sin (phase) + std::sin (2 * phase) + 0.06 * std::sin (3 * phase) + 0.5 * std::sin (4 * phase)
                      + 0.05 * std::sin (5 * phase) + 0.3 * std::sin (6 * phase);
                x[i] = (float) (0.3 * v);
            }
            const int hop = 256;
            auto frames = dsp::PitchDetector::analyse (x.data(), (int) x.size(), sr, hop);
            std::vector<float> f;
            for (int k = (int) (sr * 2.2) / hop; k < (int) (sr * 3.9) / hop; ++k)
                if (frames[(size_t) k].voiced)
                    f.push_back (frames[(size_t) k].frequencyHz);
            std::sort (f.begin(), f.end());
            const float measured = f.empty() ? 0.0f : f[f.size() / 2];
            expect (std::abs (centsBetween (measured, 150.0f)) < 20.0f, "measured " + juce::String (measured));
        }
    }
};

class VoiceChainTests : public juce::UnitTest
{
public:
    VoiceChainTests() : juce::UnitTest ("Voice chain", "DSP") {}

    std::vector<float> run (dsp::VoiceChain& chain, const std::vector<float>& input, const dsp::ChainSettings& s,
                            int blockSize, std::vector<float>* right = nullptr)
    {
        std::vector<float> outL (input.size()), outR (input.size());
        for (size_t pos = 0; pos < input.size(); pos += (size_t) blockSize)
        {
            const int n = (int) juce::jmin ((size_t) blockSize, input.size() - pos);
            chain.process (input.data() + pos, outL.data() + pos, outR.data() + pos, n, s);
        }
        if (right != nullptr)
            *right = outR;
        return outL;
    }

    void runTest() override
    {
        VoiceLibrary library;
        const auto& voice = library.getDefaultVoice();
        const double sr = 48000.0;

        for (auto quality : { Quality::LowLatency, Quality::Balanced, Quality::HighQuality })
        {
            beginTest ("Pitch shifting accuracy, quality " + juce::String ((int) quality));
            dsp::VoiceChain chain;
            chain.prepare (sr, 512, quality);
            expect (chain.getLatencySamples() > 0 && chain.getLatencySamples() < (int) (sr * 0.08),
                    "latency " + juce::String (chain.getLatencySamples()));

            for (float semis : { -12.0f, -5.0f, 7.0f, 12.0f })
            {
                chain.reset();
                auto s = neutralSettings (voice);
                s.pitch.pitchSemis = semis;
                auto in = makeVoice (sr, (int) (sr * 1.5), [] (int) { return 165.0f; });
                auto out = run (chain, in, s, 480);
                const float measured = medianPitch (out, sr, (int) (sr * 0.4), (int) out.size());
                const float expected = 165.0f * std::exp2 (semis / 12.0f);
                logMessage (juce::String (semis) + " st error " + juce::String (centsBetween (measured, expected), 2) + " cents");
                expect (std::abs (centsBetween (measured, expected)) < 12.0f,
                        juce::String (semis) + " st: expected " + juce::String (expected) + " got " + juce::String (measured));
            }
        }

        beginTest ("Identity settings preserve the signal (latency aligned)");
        {
            dsp::VoiceChain chain;
            chain.prepare (sr, 256, Quality::Balanced);
            auto s = neutralSettings (voice);
            auto in = makeVoice (sr, (int) sr, [] (int i) { return 180.0f + 20.0f * std::sin ((float) i / 4800.0f); });
            auto out = run (chain, in, s, 256);
            const int latency = chain.getLatencySamples();

            double num = 0.0, ea = 0.0, eb = 0.0;
            for (int i = (int) (sr * 0.3); i < (int) sr - latency; ++i)
            {
                num += (double) in[(size_t) i] * out[(size_t) (i + latency)];
                ea += (double) in[(size_t) i] * in[(size_t) i];
                eb += (double) out[(size_t) (i + latency)] * out[(size_t) (i + latency)];
            }
            const double corr = num / std::sqrt (ea * eb + 1e-12);
            const double gainDb = 10.0 * std::log10 (eb / ea);
            logMessage ("identity correlation " + juce::String (corr, 4) + ", gain " + juce::String (gainDb, 2) + " dB, latency " + juce::String (latency));
            expect (corr > 0.9, "correlation " + juce::String (corr));
            expect (std::abs (gainDb) < 2.0, "gain " + juce::String (gainDb) + " dB");
        }

        beginTest ("Pitch correction snaps to the scale");
        {
            dsp::VoiceChain chain;
            chain.prepare (sr, 512, Quality::Balanced);
            auto s = neutralSettings (voice);
            s.pitch.tuneOn = true;
            s.pitch.keyRoot = 0;
            s.pitch.scale = ScaleType::Major;
            s.pitch.retuneMs = 0.0f;
            s.pitch.humanize = 0.0f;
            const float sung = 440.0f * std::exp2 (35.0f / 1200.0f);          // A4 + 35 cents
            auto in = makeVoice (sr, (int) (sr * 1.5), [sung] (int) { return sung; });
            auto out = run (chain, in, s, 512);
            const float measured = medianPitch (out, sr, (int) (sr * 0.4), (int) out.size());
            expect (std::abs (centsBetween (measured, 440.0f)) < 8.0f, "corrected to " + juce::String (measured));

            // F#4 is not in C major: must land on F4 or G4.
            chain.reset();
            auto in2 = makeVoice (sr, (int) (sr * 1.5), [] (int) { return 369.99f * std::exp2 (20.0f / 1200.0f); });
            auto out2 = run (chain, in2, s, 512);
            const float m2 = medianPitch (out2, sr, (int) (sr * 0.4), (int) out2.size());
            expect (std::abs (centsBetween (m2, 392.0f)) < 8.0f, "F# corrected to " + juce::String (m2));
        }

        beginTest ("MIDI pitch control");
        {
            dsp::VoiceChain chain;
            chain.prepare (sr, 512, Quality::Balanced);
            auto s = neutralSettings (voice);
            s.pitch.midiNote = 64;          // E4
            s.pitch.retuneMs = 0.0f;
            s.pitch.humanize = 0.0f;
            auto in = makeVoice (sr, (int) (sr * 1.5), [] (int) { return 220.0f; });
            auto out = run (chain, in, s, 512);
            const float measured = medianPitch (out, sr, (int) (sr * 0.4), (int) out.size());
            expect (std::abs (centsBetween (measured, 329.63f)) < 10.0f, "MIDI controlled pitch " + juce::String (measured));
        }

        beginTest ("Formant shift moves the spectral centroid");
        {
            auto centroid = [sr] (const std::vector<float>& x)
            {
                juce::dsp::FFT fft (12);
                std::vector<float> acc (2048, 0.0f), buf (8192);
                for (size_t start = (size_t) (sr * 0.4); start + 4096 < x.size(); start += 4096)
                {
                    std::fill (buf.begin(), buf.end(), 0.0f);
                    for (int i = 0; i < 4096; ++i)
                        buf[(size_t) i] = x[start + (size_t) i] * (0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * (float) i / 4096.0f));
                    fft.performFrequencyOnlyForwardTransform (buf.data());
                    for (int b = 0; b < 2048; ++b)
                        acc[(size_t) b] += buf[(size_t) b];
                }
                double num = 0.0, den = 0.0;
                for (int b = 1; b < 2048; ++b)
                {
                    num += acc[(size_t) b] * b;
                    den += acc[(size_t) b];
                }
                return num / den * sr / 4096.0;
            };

            dsp::VoiceChain chain;
            chain.prepare (sr, 512, Quality::Balanced);
            auto s = neutralSettings (voice);
            auto in = makeVoice (sr, (int) (sr * 1.5), [] (int) { return 150.0f; });
            const double base = centroid (run (chain, in, s, 512));

            chain.reset();
            s.gender = 1.0f;                       // +3 semitones of formant shift
            const double up = centroid (run (chain, in, s, 512));

            chain.reset();
            s.gender = -1.0f;
            const double down = centroid (run (chain, in, s, 512));

            expect (up > base * 1.08, "up " + juce::String (up) + " base " + juce::String (base));
            expect (down < base * 0.93, "down " + juce::String (down) + " base " + juce::String (base));
        }

        beginTest ("Harmony and doubler stay finite and add level");
        {
            dsp::VoiceChain chain;
            chain.prepare (sr, 333, Quality::Balanced);
            auto s = neutralSettings (voice);
            s.harmonyOn = true;
            s.harmonyInterval[2] = HarmonyInterval::OctaveDown;
            s.doublerOn = true;
            s.pitch.tuneOn = true;
            s.pitch.keyRoot = 0;
            auto in = makeVoice (sr, (int) (sr * 2.0), [] (int i) { return i < 48000 ? 261.6f : 293.7f; });
            std::vector<float> right;
            auto left = run (chain, in, s, 333, &right);
            bool finite = true;
            double energy = 0.0, diff = 0.0;
            for (size_t i = 0; i < left.size(); ++i)
            {
                finite = finite && std::isfinite (left[i]) && std::isfinite (right[i]);
                energy += left[i] * left[i];
                diff += (left[i] - right[i]) * (left[i] - right[i]);
            }
            expect (finite);
            expect (energy > 1.0);
            expect (diff > energy * 0.01, "stereo spread expected");
        }

        beginTest ("Voiced / unvoiced transitions are click-free");
        {
            // Phrase: vowel, sibilant noise, vowel at another pitch, silence, vowel.
            std::vector<float> in;
            auto append = [&in] (const std::vector<float>& v) { in.insert (in.end(), v.begin(), v.end()); };
            dsp::Noise noise;
            dsp::Biquad hp;
            hp.c = dsp::BiquadCoeffs::highPass (sr, 4000.0, 0.7071);
            std::vector<float> sss ((size_t) (sr * 0.12));
            for (auto& v : sss) v = hp.process (noise.next()) * 0.15f;
            append (makeVoice (sr, (int) (sr * 0.4), [] (int) { return 200.0f; }));
            append (sss);
            append (makeVoice (sr, (int) (sr * 0.4), [] (int) { return 260.0f; }));
            append (std::vector<float> ((size_t) (sr * 0.2), 0.0f));
            append (makeVoice (sr, (int) (sr * 0.4), [] (int) { return 150.0f; }));

            auto maxStep = [] (const std::vector<float>& x, size_t from)
            {
                float rms = 0.0f, step = 0.0f;
                for (size_t i = from + 1; i < x.size(); ++i)
                {
                    rms += x[i] * x[i];
                    step = juce::jmax (step, std::abs (x[i] - x[i - 1]));
                }
                rms = std::sqrt (rms / (float) (x.size() - from));
                return step / juce::jmax (1.0e-6f, rms);
            };

            for (float semis : { -7.0f, 0.0f, 7.0f })
            {
                dsp::VoiceChain chain;
                chain.prepare (sr, 256, Quality::Balanced);
                auto s = neutralSettings (voice);
                s.pitch.pitchSemis = semis;
                s.gender = 0.6f;
                auto out = run (chain, in, s, 256);
                const float inStep = maxStep (in, 0);
                const float outStep = maxStep (out, (size_t) chain.getLatencySamples());
                logMessage (juce::String (semis) + " st: max step/rms in " + juce::String (inStep, 2) + ", out " + juce::String (outStep, 2));
                expect (outStep < inStep * 2.5f, "possible clicks at " + juce::String (semis) + " st");
            }
        }

        beginTest ("CPU budget with every voice and effect enabled");
        {
            dsp::VoiceChain chain;
            chain.prepare (sr, 256, Quality::HighQuality);
            auto s = neutralSettings (voice);
            s.harmonyOn = true;
            s.harmonyInterval[2] = HarmonyInterval::OctaveDown;
            s.doublerOn = true;
            s.pitch.tuneOn = true;
            s.pitch.pitchSemis = 3.0f;
            s.gender = 0.5f;
            s.breath = 0.3f; s.rasp = 0.3f; s.saturation = 0.3f;
            s.deEsser = 0.5f; s.compressor = 0.5f; s.delayMix = 0.3f; s.reverbMix = 0.3f;
            auto in = makeVoice (sr, (int) (sr * 10.0), [] (int i) { return 220.0f + 30.0f * std::sin ((float) i * 0.0001f); });
            const auto t0 = juce::Time::getMillisecondCounterHiRes();
            run (chain, in, s, 256);
            const double ms = juce::Time::getMillisecondCounterHiRes() - t0;
            const double load = ms / 10000.0 * 100.0;
            logMessage ("Full chain CPU: " + juce::String (load, 2) + "% of one core (10 s of audio in " + juce::String (ms, 1) + " ms)");
            expect (load < 50.0, "chain too slow: " + juce::String (load) + "%");
        }

        beginTest ("Every factory voice processes cleanly with full FX");
        {
            dsp::VoiceChain chain;
            chain.prepare (44100.0, 512, Quality::Balanced);
            auto in = makeVoice (44100.0, 44100, [] (int i) { return 200.0f + 30.0f * std::sin ((float) i * 0.0003f); });
            for (auto& v : library.getVoices())
            {
                chain.reset();
                dsp::ChainSettings s;
                s.pitch.traits = v.traits;
                s.pitch.pitchPreserve = false;
                s.pitch.tuneOn = true;
                s.reverbMix = 0.3f;
                s.delayMix = 0.3f;
                s.saturation = 0.4f;
                s.breath = 0.3f;
                s.rasp = 0.3f;
                s.doublerOn = true;
                auto out = run (chain, in, s, 512);
                float peak = 0.0f;
                bool finite = true;
                for (auto x : out)
                {
                    finite = finite && std::isfinite (x);
                    peak = juce::jmax (peak, std::abs (x));
                }
                expect (finite && peak < 4.0f && peak > 0.01f, v.name + " peak " + juce::String (peak));
            }
        }
    }
};

class GenderEstimatorTests : public juce::UnitTest
{
public:
    GenderEstimatorTests() : juce::UnitTest ("Voice gender estimator", "DSP") {}

    /** Glottal pulses through five formants scaled by `tract` (1 = adult male). */
    static std::vector<float> makeTractVoice (double sr, int n, float f0, float tract, bool vibrato)
    {
        std::vector<float> out ((size_t) n);
        // Typical sung-vowel formants of an adult male (F3 + F4 ~ 6800 Hz, as measured on real singers).
        const double formants[5] = { 550.0, 1500.0, 2900.0, 3850.0, 4700.0 };
        const double gains[5] = { 1.0, 0.6, 0.35, 0.25, 0.15 };
        dsp::Biquad f[5];
        for (int k = 0; k < 5; ++k)
            f[k].c = dsp::BiquadCoeffs::bandPass (sr, formants[k] * tract, 8.0 + k * 2.0);
        double phase = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double hz = f0 * (vibrato ? 1.0 + 0.02 * std::sin (i * 2.0 * juce::MathConstants<double>::pi * 5.5 / sr) : 1.0);
            phase += hz / sr;
            if (phase >= 1.0) phase -= 1.0;
            const float g = phase < 0.4 ? (float) (0.5 - 0.5 * std::cos (juce::MathConstants<double>::pi * phase / 0.4))
                                        : (phase < 0.55 ? (float) std::cos (juce::MathConstants<double>::halfPi * (phase - 0.4) / 0.15) : 0.0f);
            float y = 0.0f;
            for (int k = 0; k < 5; ++k)
                y += (float) gains[k] * f[k].process (g - 0.3f);
            out[(size_t) i] = 0.3f * y;
        }
        return out;
    }

    void runTest() override
    {
        const double sr = 48000.0;
        using G = dsp::VoiceGenderEstimator::Gender;

        beginTest ("Male and female vocal tracts at typical pitches");
        {
            auto male = makeTractVoice (sr, (int) (sr * 4), 130.0f, 1.0f, true);
            auto female = makeTractVoice (sr, (int) (sr * 4), 240.0f, 1.1f, true);
            float sm = 0.0f, sf = 0.0f;
            const auto gm = dsp::VoiceGenderEstimator::analyse (male.data(), (int) male.size(), sr, &sm);
            const auto gf = dsp::VoiceGenderEstimator::analyse (female.data(), (int) female.size(), sr, &sf);
            expect (gm == G::Male, "male score " + juce::String (sm));
            expect (gf == G::Female, "female score " + juce::String (sf));
        }

        beginTest ("Vocal-tract length outweighs a misleading pitch");
        {
            // A man singing high (240 Hz) and a woman singing low (200 Hz).
            auto highMale = makeTractVoice (sr, (int) (sr * 4), 240.0f, 0.98f, true);
            auto lowFemale = makeTractVoice (sr, (int) (sr * 4), 200.0f, 1.12f, true);
            float sm = 0.0f, sf = 0.0f;
            const auto gm = dsp::VoiceGenderEstimator::analyse (highMale.data(), (int) highMale.size(), sr, &sm);
            const auto gf = dsp::VoiceGenderEstimator::analyse (lowFemale.data(), (int) lowFemale.size(), sr, &sf);
            expect (gm == G::Male, "high male score " + juce::String (sm));
            expect (gf == G::Female, "low female score " + juce::String (sf));
        }

        beginTest ("Silence stays unknown");
        {
            std::vector<float> silence ((size_t) sr * 2, 0.0f);
            expect (dsp::VoiceGenderEstimator::analyse (silence.data(), (int) silence.size(), sr) == G::Unknown);
        }
    }
};

class PitchMapperTests : public juce::UnitTest
{
public:
    PitchMapperTests() : juce::UnitTest ("Pitch mapper", "DSP") {}

    /** Feeds phrases (1 s sung, 0.2 s rest) cycling through `notes` (MIDI); returns the last voiced frame. */
    static dsp::PitchFrame sing (dsp::PitchMapper& mapper, const dsp::PitchMapper::Settings& s,
                                 std::initializer_list<float> notes, int phrases)
    {
        constexpr double sr = 48000.0;
        constexpr int hop = 256;
        dsp::PitchFrame last;
        juce::int64 t = 0;
        for (int p = 0; p < phrases; ++p)
        {
            const float midi = *(notes.begin() + p % (int) notes.size());
            for (int i = 0; i < (int) (sr / hop); ++i, t += hop)
            {
                dsp::PitchDetector::Frame f;
                f.voiced = true;
                f.frequencyHz = music::midiToHz (midi);
                f.periodSamples = (float) sr / f.frequencyHz;
                f.confidence = 0.95f;
                last = mapper.process (f, t, s);
            }
            for (int i = 0; i < (int) (0.2 * sr / hop); ++i, t += hop)
                mapper.process (dsp::PitchDetector::Frame(), t, s);
        }
        return last;
    }

    void runTest() override
    {
        beginTest ("Chromatic applies with Auto key");
        {
            dsp::PitchMapper mapper;
            mapper.prepare (48000.0, 256);
            dsp::PitchMapper::Settings s;
            sing (mapper, s, { 60.0f, 64.0f, 67.0f, 62.0f, 65.0f, 69.0f, 71.0f }, 14);   // C major melody
            expect (mapper.getDetectedKeyRoot() >= 0, "key detected");
            expect (mapper.getEffectiveScale (s) != ScaleType::Chromatic, "auto key uses the detected scale");
            s.scale = ScaleType::Chromatic;
            expect (mapper.getEffectiveScale (s) == ScaleType::Chromatic, "chosen chromatic scale is kept");
        }

        beginTest ("AI voices sing the tracked pitch with the register and correction's small moves");
        {
            dsp::PitchMapper::Settings s;
            dsp::PitchFrame pf;
            pf.voiced = true;
            pf.globalOffset = 12.0f;
            pf.mainShift = 12.3f;
            expectWithinAbsoluteError (dsp::VoiceChain::neuralPitchShift (pf, s), 12.3f, 0.001f);
            pf.mainShift = 24.0f;   // the live detector read another note: only the register applies
            expectWithinAbsoluteError (dsp::VoiceChain::neuralPitchShift (pf, s), 13.5f, 0.001f);
            pf.voiced = false;
            expectWithinAbsoluteError (dsp::VoiceChain::neuralPitchShift (pf, s), 12.0f, 0.001f);
            s.midiNote = 60;        // MIDI decides the pitch
            expect (std::isnan (dsp::VoiceChain::neuralPitchShift (pf, s)));
            s.midiNote = -1;
            s.sourceIsSynth = true; // the Text to Sing guide's pitch is exact
            expect (std::isnan (dsp::VoiceChain::neuralPitchShift (pf, s)));
        }

        beginTest ("AI register stays near the singer's own notes");
        {
            auto run = [this] (float aiCentre, bool ai)
            {
                dsp::PitchMapper mapper;
                mapper.prepare (48000.0, 256);
                dsp::PitchMapper::Settings s;
                s.pitchPreserve = false;
                s.sourceGender = 1;                               // a man singing around D3
                s.traits.gender = VoiceGender::Female;
                s.traits.rangeLowMidi = 57;
                s.traits.rangeHighMidi = 79;
                s.traits.aiCentreMidi = aiCentre;
                s.aiRegister = ai;
                return juce::roundToInt (sing (mapper, s, { 50.0f, 52.0f, 48.0f }, 6).globalOffset);
            };
            expectEquals (run (0.0f, false), 12, "DSP female voice: an octave up");
            expectEquals (run (63.0f, true), 12, "AI singer centred on D#4: an octave up");
            expectEquals (run (52.0f, true), 0, "AI singer centred on E3: stays put");
        }

        beginTest ("AI register: the first phrase is in the song's octave; the song's high notes count");
        {
            // A man singing around G#3 with phrases up to C#4.
            auto run = [this] (int low, int high, float median, float bias, VoiceGender gender, int phrases)
            {
                dsp::PitchMapper mapper;
                mapper.prepare (48000.0, 256);
                dsp::PitchMapper::Settings s;
                s.pitchPreserve = false;
                s.sourceGender = 1;
                s.traits.gender = gender;
                s.traits.aiCentreMidi = median;
                s.traits.aiLowMidi = low;
                s.traits.aiHighMidi = high;
                s.traits.registerBiasSemis = bias;
                s.aiRegister = true;
                return juce::roundToInt (sing (mapper, s, { 56.0f, 61.0f, 52.0f, 57.0f }, phrases).globalOffset);
            };
            expectEquals (run (50, 75, 62.8f, 0.0f, VoiceGender::Female, 1), 12, "first phrase already up an octave");
            expectEquals (run (50, 75, 62.8f, 0.0f, VoiceGender::Female, 8), 12, "a singer reaching D#5 takes the octave up");
            expectEquals (run (55, 69, 61.8f, 0.0f, VoiceGender::Female, 8), 0, "a singer topping out at A4 stays put");
            expectEquals (run (41, 66, 52.6f, -7.0f, VoiceGender::Male, 1), -12, "a bass voice sings an octave down from the start");
            expectEquals (run (41, 66, 52.6f, -7.0f, VoiceGender::Male, 8), -12, "and stays there");
        }

        beginTest ("A hard-tune voice snaps every note at once, also with tuning switched off");
        {
            dsp::PitchMapper mapper;
            mapper.prepare (48000.0, 256);
            dsp::PitchMapper::Settings s;
            s.traits.hardTune = true;
            s.expression = false;
            s.humanize = 1.0f;
            s.retuneMs = 300.0f;
            s.keyRoot = 0;
            s.scale = ScaleType::Chromatic;
            const auto f = sing (mapper, s, { 60.4f }, 1);
            expectWithinAbsoluteError (f.srcMidi + f.mainShift, 60.0f, 0.01f);
            expect (std::isnan (dsp::VoiceChain::neuralPitchShift (f, s)), "the AI singer gets the exact note");
        }
    }
};

class ConsonantKeeperTests : public juce::UnitTest
{
public:
    ConsonantKeeperTests() : juce::UnitTest ("Consonant keeper", "DSP") {}

    void runTest() override
    {
        constexpr double sr = 48000.0;
        const int seg = (int) (sr * 0.5);

        // A sung vowel, then an "s" (the input hisses above 6.5 kHz, the AI smears it down to ~4 kHz), then the vowel.
        dsp::Noise noise;
        dsp::Biquad sHigh[2], smearLow[2], smearHigh[2];
        for (int k = 0; k < 2; ++k)
        {
            sHigh[k].c = dsp::BiquadCoeffs::highPass (sr, 6500.0);
            smearHigh[k].c = dsp::BiquadCoeffs::highPass (sr, 3500.0);
            smearLow[k].c = dsp::BiquadCoeffs::lowPass (sr, 4500.0);
        }
        std::vector<float> input ((size_t) seg * 3), lead ((size_t) seg * 3);
        for (int i = 0; i < seg * 3; ++i)
        {
            const bool hiss = i >= seg && i < 2 * seg;
            const double t = i / sr;
            float vowelIn = 0.0f, vowelAi = 0.0f;
            for (int h = 1; h <= 8; ++h)
            {
                vowelIn += 0.2f / (float) h * (float) std::sin (juce::MathConstants<double>::twoPi * 150.0 * h * t);
                vowelAi += 0.2f / (float) h * (float) std::sin (juce::MathConstants<double>::twoPi * 200.0 * h * t + h);
            }
            const float s = sHigh[1].process (sHigh[0].process (noise.next()));
            const float smear = smearLow[1].process (smearLow[0].process (smearHigh[1].process (smearHigh[0].process (noise.next()))));
            input[(size_t) i] = hiss ? 0.3f * s : vowelIn;
            lead[(size_t) i] = hiss ? 0.3f * smear : vowelAi;
        }

        auto out = lead;
        dsp::ConsonantKeeper keeper;
        keeper.prepare (sr);
        for (int pos = 0; pos < seg * 3; pos += 512)
            keeper.process (out.data() + pos, input.data() + pos, juce::jmin (512, seg * 3 - pos));

        // Band level from averaged Hann-windowed spectra (sharp band edges: the "s" is far louder overall).
        auto bandDb = [sr] (const std::vector<float>& x, int from, int to, double lo, double hi)
        {
            constexpr int order = 12, size = 1 << order;
            juce::dsp::FFT fft (order);
            std::vector<float> frame ((size_t) size * 2);
            double e = 0.0;
            for (int pos = from; pos + size <= to; pos += size / 2)
            {
                std::fill (frame.begin(), frame.end(), 0.0f);
                for (int i = 0; i < size; ++i)
                    frame[(size_t) i] = x[(size_t) (pos + i)] * 0.5f * (1.0f - std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) size));
                fft.performFrequencyOnlyForwardTransform (frame.data());
                for (int b = (int) std::ceil (lo * size / sr); b <= (int) (hi * size / sr); ++b)
                    e += (double) frame[(size_t) b] * frame[(size_t) b];
            }
            return 10.0 * std::log10 (e + 1.0e-20);
        };

        beginTest ("The input's own \"s\" replaces the AI's smeared one");
        {
            const int from = seg + seg / 5, to = 2 * seg - seg / 5;
            const double keptHigh = bandDb (out, from, to, 7000.0, 16000.0) - bandDb (input, from, to, 7000.0, 16000.0);
            const double smearLeft = bandDb (out, from, to, 3800.0, 4200.0) - bandDb (lead, from, to, 3800.0, 4200.0);
            expect (std::abs (keptHigh) < 3.0, "input hiss kept: " + juce::String (keptHigh, 1) + " dB");
            expect (smearLeft < -8.0, "AI smear removed: " + juce::String (smearLeft, 1) + " dB");
        }

        beginTest ("Sung vowels stay the AI voice");
        {
            // Untouched lead = its Linkwitz-Riley split summed back (an all-pass): compare against that.
            dsp::Biquad low[2], high[2];
            for (int k = 0; k < 2; ++k)
            {
                low[k].c = dsp::BiquadCoeffs::lowPass (sr, 2500.0);
                high[k].c = dsp::BiquadCoeffs::highPass (sr, 2500.0);
            }
            std::vector<float> untouched (lead.size());
            for (size_t i = 0; i < lead.size(); ++i)
                untouched[i] = low[1].process (low[0].process (lead[i])) + high[1].process (high[0].process (lead[i]));

            for (int start : { seg / 5, 2 * seg + seg / 5 })
            {
                double eLead = 0.0, eDiff = 0.0;
                for (int i = start; i < start + seg / 2; ++i)
                {
                    const double d = (double) out[(size_t) i] - untouched[(size_t) i];
                    eLead += (double) untouched[(size_t) i] * untouched[(size_t) i];
                    eDiff += d * d;
                }
                const double changeDb = 10.0 * std::log10 (eDiff / eLead + 1.0e-20);
                expect (changeDb < -40.0, "vowel changed by " + juce::String (changeDb, 1) + " dB");
            }
        }
    }
};

class RKeeperTests : public juce::UnitTest
{
public:
    RKeeperTests() : juce::UnitTest ("R keeper", "DSP") {}

    /** A sung vowel: pulses at `f0` through resonators at F1 600, F2 1200, F3 (`f3` per sample), F4 and F5. */
    static std::vector<float> formantVoice (double sr, int n, double f0, const std::function<double (int)>& f3,
                                            double f4, double f5)
    {
        std::vector<float> out ((size_t) n);
        double phase = 0.0, z[5][2] = {};
        const double bandwidths[] = { 80.0, 90.0, 120.0, 150.0, 200.0 };
        for (int i = 0; i < n; ++i)
        {
            phase += f0 / sr;
            double x = 0.0;
            if (phase >= 1.0) { phase -= 1.0; x = 1.0; }
            const double freqs[] = { 600.0, 1200.0, f3 (i), f4, f5 };
            for (int k = 0; k < 5; ++k)
            {
                const double r = std::exp (-juce::MathConstants<double>::pi * bandwidths[k] / sr);
                const double c = 2.0 * r * std::cos (juce::MathConstants<double>::twoPi * freqs[k] / sr);
                const double y = x * (1.0 - r) + c * z[k][0] - r * r * z[k][1];
                z[k][1] = z[k][0];
                z[k][0] = y;
                x = y;
            }
            out[(size_t) i] = (float) x;
        }
        float peak = 0.0f;
        for (auto v : out) peak = juce::jmax (peak, std::abs (v));
        for (auto& v : out) v *= 0.3f / peak;
        return out;
    }

    void runTest() override
    {
        constexpr double sr = 48000.0;
        const int total = (int) (sr * 3.0), rFrom = (int) (sr * 2.0), rTo = (int) (sr * 2.25);
        // a man says an r from 2.0 to 2.25 s (F3 2600 -> 1750 Hz); the AI voice, a woman, keeps F3 at 2900 Hz
        auto dry = formantVoice (sr, total, 150.0, [=] (int i) { return i >= rFrom && i < rTo ? 1750.0 : 2600.0; }, 3500.0, 4400.0);
        auto lead = formantVoice (sr, total, 220.0, [] (int) { return 2900.0; }, 4000.0, 4900.0);
        std::vector<float> voiced ((size_t) total, 1.0f);

        beginTest ("Formant tracker finds F3 (a man's and a woman's ceiling)");
        {
            auto track = [&] (const std::vector<float>& voice, double ceiling)
            {
                dsp::RKeeper::FormantTracker tracker;
                tracker.prepare (sr, dsp::RKeeper::femaleCeilingHz);
                tracker.setCeiling (ceiling);   // as the plug-in changes it: no allocation, a fresh start
                std::vector<float> vowel, r;
                for (int i = 0; i < total; ++i)
                {
                    tracker.push (voice[(size_t) i]);
                    if ((i + 1) % 240 != 0)
                        continue;
                    tracker.analyse();
                    if (i > (int) (sr * 0.5) && i < rFrom) vowel.push_back (tracker.f3);
                    if (i > rFrom + (int) (sr * 0.05) && i < rTo) r.push_back (tracker.f3);
                }
                auto median = [] (std::vector<float> v) { std::sort (v.begin(), v.end()); return v[v.size() / 2]; };
                return std::make_pair (median (vowel), median (r));
            };
            const auto man = track (dry, dsp::RKeeper::maleCeilingHz);
            expectWithinAbsoluteError (man.first, 2600.0f, 130.0f, "vowel F3");
            expectWithinAbsoluteError (man.second, 1750.0f, 90.0f, "r F3");
            expectWithinAbsoluteError (track (lead, dsp::RKeeper::femaleCeilingHz).first, 2900.0f, 145.0f, "AI voice F3");
        }

        beginTest ("The AI voice gets the r, and the vowels stay untouched");
        {
            auto out = lead;
            dsp::RKeeper keeper;
            keeper.prepare (sr);
            keeper.setCeilings (dsp::RKeeper::maleCeilingHz, dsp::RKeeper::femaleCeilingHz);
            for (int pos = 0; pos < total; pos += 512)
            {
                const int n = juce::jmin (512, total - pos);
                keeper.process (out.data() + pos, dry.data() + pos, voiced.data() + pos, n);
            }

            auto bandDb = [sr] (const std::vector<float>& x, int from, int to, double lo, double hi)
            {
                constexpr int order = 11, size = 1 << order;
                juce::dsp::FFT fft (order);
                std::vector<float> frame ((size_t) size * 2);
                double e = 0.0;
                for (int pos = from; pos + size <= to; pos += size / 2)
                {
                    std::fill (frame.begin(), frame.end(), 0.0f);
                    for (int i = 0; i < size; ++i)
                        frame[(size_t) i] = x[(size_t) (pos + i)] * 0.5f * (1.0f - std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) size));
                    fft.performFrequencyOnlyForwardTransform (frame.data());
                    for (int b = (int) std::ceil (lo * size / sr); b <= (int) (hi * size / sr); ++b)
                        e += (double) frame[(size_t) b] * frame[(size_t) b];
                }
                return 10.0 * std::log10 (e + 1.0e-20);
            };

            // r: the AI F3 (2900 Hz) lowered, a resonance near 2900 * 1750 / 2600 = 1950 Hz
            const int a = rFrom + (int) (sr * 0.06), b = rTo;
            const double raised = bandDb (out, a, b, 1800.0, 2100.0) - bandDb (lead, a, b, 1800.0, 2100.0);
            const double lowered = bandDb (out, a, b, 2750.0, 3050.0) - bandDb (lead, a, b, 2750.0, 3050.0);
            expect (raised > 4.0, "r resonance raised by " + juce::String (raised, 1) + " dB");
            expect (lowered < -4.0, "AI F3 lowered by " + juce::String (lowered, 1) + " dB");

            // vowels before the r: unchanged
            double eLead = 0.0, eDiff = 0.0;
            for (int i = (int) (sr * 0.8); i < (int) (sr * 1.8); ++i)
            {
                const double d = (double) out[(size_t) i] - lead[(size_t) i];
                eLead += (double) lead[(size_t) i] * lead[(size_t) i];
                eDiff += d * d;
            }
            expect (10.0 * std::log10 (eDiff / eLead + 1.0e-20) < -30.0, "vowels changed");
        }
    }
};

class BreathKeeperTests : public juce::UnitTest
{
public:
    BreathKeeperTests() : juce::UnitTest ("Breath keeper", "DSP") {}

    void runTest() override
    {
        constexpr double sr = 48000.0;
        const int seg = (int) sr;   // sung 1 s, breath 1 s, sung 1 s
        std::vector<float> input ((size_t) seg * 3), voiced ((size_t) seg * 3), lead ((size_t) seg * 3);
        juce::Random rng (3);
        for (int i = 0; i < seg * 3; ++i)
        {
            const bool sung = i < seg || i >= 2 * seg;
            const float tone = 0.3f * std::sin ((float) (juce::MathConstants<double>::twoPi * 220.0 * i / sr));
            const float noise = rng.nextFloat() * 2.0f - 1.0f;
            input[(size_t) i] = sung ? tone : 0.01f * noise;   // breath ~33 dB below the singing
            lead[(size_t) i] = sung ? tone : 0.04f * noise;    // the AI singer's breath: 12 dB louder
            voiced[(size_t) i] = sung ? 1.0f : 0.0f;
        }
        auto out = lead;
        dsp::BreathKeeper keeper;
        keeper.prepare (sr);
        for (int pos = 0; pos < seg * 3; pos += 512)
            keeper.process (out.data() + pos, nullptr, input.data() + pos, voiced.data() + pos, juce::jmin (512, seg * 3 - pos));

        auto db = [] (const std::vector<float>& x, int from, int to)
        {
            double e = 0.0;
            for (int i = from; i < to; ++i) e += (double) x[(size_t) i] * x[(size_t) i];
            return 10.0 * std::log10 (e / (to - from) + 1.0e-20);
        };
        beginTest ("An AI breath louder than the singer's is brought to the singer's level");
        {
            const double inRel = db (input, seg + seg / 10, 2 * seg) - db (input, seg / 2, seg);
            const double outRel = db (out, seg + seg / 10, 2 * seg) - db (out, seg / 2, seg);
            expect (outRel - inRel < 2.5 && outRel - inRel > -1.0, "breath vs singing: input " + juce::String (inRel, 1)
                    + " dB, output " + juce::String (outRel, 1) + " dB");
        }
        beginTest ("Sung notes are untouched");
        {
            double err = 0.0, energy = 0.0;
            for (int i = 2 * seg + seg / 10; i < 3 * seg; ++i)
            {
                err += std::pow ((double) out[(size_t) i] - lead[(size_t) i], 2.0);
                energy += (double) lead[(size_t) i] * lead[(size_t) i];
            }
            expect (10.0 * std::log10 (err / energy + 1.0e-20) < -60.0);
        }
    }
};

static PitchDetectorTests pitchDetectorTests;
class VowelKeeperTests : public juce::UnitTest
{
public:
    VowelKeeperTests() : juce::UnitTest ("Vowel keeper", "DSP") {}

    // 200 Hz harmonics shaped by two formants (F1, F2) with 100 Hz wide resonances.
    static float vowel (double t, float f1, float f2)
    {
        float x = 0.0f;
        for (int h = 1; h * 200.0 < 5000.0; ++h)
        {
            const float f = 200.0f * (float) h;
            const float a = 1.0f / (1.0f + juce::square ((f - f1) / 100.0f)) + 0.6f / (1.0f + juce::square ((f - f2) / 100.0f)) + 0.02f;
            x += 0.05f * a * (float) std::sin (juce::MathConstants<double>::twoPi * f * t);
        }
        return x;
    }

    void runTest() override
    {
        beginTest ("The AI voice's vowel follows the input's: a too-high F2 comes down");
        {
            constexpr double sr = 48000.0;
            const int settle = (int) (sr * 3.0), change = (int) (sr * 0.6), total = settle + change;
            std::vector<float> input ((size_t) total), lead ((size_t) total), hz ((size_t) total, 200.0f);
            for (int i = 0; i < total; ++i)
            {
                const double t = i / sr;
                const bool later = i >= settle;   // both sing "ah", then the input "ih" and the AI an "ee"-ish ih
                input[(size_t) i] = later ? vowel (t, 400.0f, 1800.0f) : vowel (t, 600.0f, 1400.0f);
                lead[(size_t) i] = later ? vowel (t, 400.0f, 2200.0f) : vowel (t, 600.0f, 1400.0f);
            }
            auto out = lead;
            dsp::VowelKeeper keeper;
            keeper.prepare (sr);
            for (int pos = 0; pos < total; pos += 480)
            {
                const int n = juce::jmin (480, total - pos);
                keeper.process (out.data() + pos, input.data() + pos, hz.data() + pos, hz.data() + pos, n, 1.0f);
            }

            // the harmonics at 1800 and 2200 Hz over the last 0.3 s
            auto level = [&] (const std::vector<float>& x, double f)
            {
                double re = 0.0, im = 0.0;
                for (int i = total - (int) (0.3 * sr); i < total; ++i)
                {
                    re += x[(size_t) i] * std::cos (juce::MathConstants<double>::twoPi * f * i / sr);
                    im += x[(size_t) i] * std::sin (juce::MathConstants<double>::twoPi * f * i / sr);
                }
                return 10.0 * std::log10 (re * re + im * im + 1e-20);
            };
            const double before = level (lead, 1800.0) - level (lead, 2200.0);
            const double after = level (out, 1800.0) - level (out, 2200.0);
            logMessage ("1800 Hz over 2200 Hz: " + juce::String (before, 1) + " dB -> " + juce::String (after, 1) + " dB");
            expectGreaterThan (after - before, 8.0, "F2 moves toward the input's");

            // where both voices sang the same vowel nothing changes
            double diff = 0.0, ref = 0.0;
            for (int i = (int) (2.0 * sr); i < settle; ++i)
            {
                diff += juce::square ((double) out[(size_t) i] - lead[(size_t) i]);
                ref += juce::square ((double) lead[(size_t) i]);
            }
            expectLessThan (10.0 * std::log10 (diff / ref), -30.0, "the same vowel passes untouched");
        }
    }
};

static BreathKeeperTests breathKeeperTests;
static VowelKeeperTests vowelKeeperTests;
static RKeeperTests rKeeperTests;
static ConsonantKeeperTests consonantKeeperTests;
static GenderEstimatorTests genderEstimatorTests;
static VoiceChainTests voiceChainTests;
static PitchMapperTests pitchMapperTests;

} // namespace voxa
