#include <juce_dsp/juce_dsp.h>
#include "dsp/PitchDetector.h"
#include "dsp/VoiceChain.h"
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

static PitchDetectorTests pitchDetectorTests;
static VoiceChainTests voiceChainTests;

} // namespace voxa
