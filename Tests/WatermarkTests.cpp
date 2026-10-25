#include <juce_events/juce_events.h>
#include "core/AudioUtils.h"
#include "core/Watermark.h"
#include "TestSignals.h"

namespace voxa
{

namespace
{
    /** A few seconds of "singing": the test voice gliding through a melody with vibrato and gaps. */
    juce::AudioBuffer<float> makeSinging (double sr, double seconds)
    {
        const int n = (int) (sr * seconds);
        auto v = test::makeVoice (sr, n, [sr] (int i)
        {
            const double t = i / sr;
            const double notes[] = { 220.0, 247.0, 262.0, 294.0, 330.0, 294.0, 262.0, 247.0 };
            return (float) (notes[(int) (t / 0.5) % 8] * std::pow (2.0, 0.3 / 12.0 * std::sin (2.0 * juce::MathConstants<double>::pi * 5.5 * t)));
        });
        for (int i = 0; i < n; ++i)
        {
            const double t = std::fmod (i / sr, 2.0);
            v[(size_t) i] *= (float) (t < 1.8 ? 1.0 : 0.0);   // short breaths between phrases
        }
        juce::AudioBuffer<float> buf (2, n);
        buf.copyFrom (0, 0, v.data(), n);
        buf.copyFrom (1, 0, v.data(), n);
        return buf;
    }

    juce::AudioBuffer<float> marked (juce::AudioBuffer<float> audio, double sr, WatermarkKind kind)
    {
        WatermarkEmbedder embedder;
        embedder.prepare (sr);
        embedder.process (audio.getWritePointer (0), audio.getWritePointer (1), audio.getNumSamples(), kind);
        return audio;
    }

    double energy (const float* x, int n)
    {
        double e = 0.0;
        for (int i = 0; i < n; ++i)
            e += (double) x[i] * x[i];
        return e;
    }
}

class WatermarkTests : public juce::UnitTest
{
public:
    WatermarkTests() : juce::UnitTest ("Watermark", "Core") {}

    void runTest() override
    {
        const double sr = 48000.0;
        const auto clean = makeSinging (sr, 8.0);

        beginTest ("AI voice mark is found, unmarked audio is not");
        {
            const auto ai = marked (clean, sr, WatermarkKind::AiVoice);
            const auto r = detectWatermark (ai, sr);
            expect (r.kind == WatermarkKind::AiVoice, "kind " + watermarkKindName (r.kind));
            expect (r.aiVoiceScore > 12.0f, "AI score " + juce::String (r.aiVoiceScore));
            expect (r.synthesizedSingingScore < 6.0f, "singing score " + juce::String (r.synthesizedSingingScore));

            const auto none = detectWatermark (clean, sr);
            expect (! none.found(), "clean flagged as " + watermarkKindName (none.kind));
            expect (none.aiVoiceScore < 6.0f && none.synthesizedSingingScore < 6.0f,
                    "clean scores " + juce::String (none.aiVoiceScore) + " / " + juce::String (none.synthesizedSingingScore));

            juce::AudioBuffer<float> noise (1, (int) (sr * 5.0));
            juce::Random rng (7);
            for (int i = 0; i < noise.getNumSamples(); ++i)
                noise.setSample (0, i, 0.1f * (rng.nextFloat() * 2.0f - 1.0f));
            expect (! detectWatermark (noise, sr).found(), "white noise flagged");

            juce::AudioBuffer<float> silence (2, (int) sr);
            silence.clear();
            expect (! detectWatermark (silence, sr).found(), "silence flagged");
        }

        beginTest ("Synthesized singing has its own mark");
        {
            const auto r = detectWatermark (marked (clean, sr, WatermarkKind::SynthesizedSinging), sr);
            expect (r.kind == WatermarkKind::SynthesizedSinging, "kind " + watermarkKindName (r.kind));
            expect (r.aiVoiceScore < 6.0f, "AI score " + juce::String (r.aiVoiceScore));
        }

        beginTest ("Mark sits ~30 dB under the voice and vanishes in silence");
        {
            juce::AudioBuffer<float> withGap (2, clean.getNumSamples() + (int) sr);
            withGap.clear();
            for (int ch = 0; ch < 2; ++ch)
                withGap.copyFrom (ch, 0, clean, ch, 0, clean.getNumSamples());
            const auto ai = marked (withGap, sr, WatermarkKind::AiVoice);

            std::vector<float> diff ((size_t) clean.getNumSamples());
            for (int i = 0; i < clean.getNumSamples(); ++i)
                diff[(size_t) i] = ai.getSample (0, i) - withGap.getSample (0, i);
            const double ratioDb = 10.0 * std::log10 (energy (diff.data(), (int) diff.size())
                                                      / energy (withGap.getReadPointer (0), clean.getNumSamples()));
            expect (ratioDb < -30.0 && ratioDb > -50.0, "mark level " + juce::String (ratioDb, 1) + " dB");

            const int tailStart = clean.getNumSamples() + (int) (sr * 0.3);
            expect (ai.getMagnitude (0, tailStart, ai.getNumSamples() - tailStart) < 1.0e-5f, "mark lingers in silence");
        }

        beginTest ("Survives resampling, gain and noise");
        {
            auto ai = resampleBuffer (marked (clean, sr, WatermarkKind::AiVoice), sr, 44100.0);
            ai.applyGain (0.25f);
            juce::Random rng (3);
            for (int ch = 0; ch < ai.getNumChannels(); ++ch)
                for (int i = 0; i < ai.getNumSamples(); ++i)
                    ai.addSample (ch, i, 0.0005f * (rng.nextFloat() * 2.0f - 1.0f));   // the test voice has little treble
            const auto r = detectWatermark (ai, 44100.0);
            expect (r.kind == WatermarkKind::AiVoice, "after processing: " + watermarkKindName (r.kind)
                                                          + " score " + juce::String (r.aiVoiceScore));

            // Cropped: the detector does not need the start of the file.
            juce::AudioBuffer<float> cropped (1, (int) (sr * 5.0));
            const auto full = marked (clean, sr, WatermarkKind::AiVoice);
            cropped.copyFrom (0, 0, full, 0, (int) (sr * 2.37), cropped.getNumSamples());
            expect (detectWatermark (cropped, sr).kind == WatermarkKind::AiVoice, "cropped excerpt");
        }

        beginTest ("Embedding at 44.1 and 96 kHz");
        for (double rate : { 44100.0, 96000.0 })
        {
            const auto r = detectWatermark (marked (makeSinging (rate, 8.0), rate, WatermarkKind::AiVoice), rate);
            expect (r.kind == WatermarkKind::AiVoice, juce::String (rate) + " Hz score " + juce::String (r.aiVoiceScore));
        }

        beginTest ("Block size does not change the result");
        {
            const auto once = marked (clean, sr, WatermarkKind::AiVoice);
            auto blocks = clean;
            WatermarkEmbedder embedder;
            embedder.prepare (sr);
            juce::Random rng (11);
            for (int pos = 0; pos < blocks.getNumSamples();)
            {
                const int n = juce::jmin (blocks.getNumSamples() - pos, 1 + rng.nextInt (700));
                embedder.process (blocks.getWritePointer (0, pos), blocks.getWritePointer (1, pos), n, WatermarkKind::AiVoice);
                pos += n;
            }
            float maxDiff = 0.0f;
            for (int i = 0; i < once.getNumSamples(); ++i)
                maxDiff = juce::jmax (maxDiff, std::abs (once.getSample (0, i) - blocks.getSample (0, i)));
            expect (maxDiff < 1.0e-6f, "max difference " + juce::String (maxDiff));
        }

        beginTest ("Switching off fades the mark out");
        {
            auto audio = clean;
            WatermarkEmbedder embedder;
            embedder.prepare (sr);
            const int half = audio.getNumSamples() / 2;
            embedder.process (audio.getWritePointer (0), audio.getWritePointer (1), half, WatermarkKind::AiVoice);
            embedder.process (audio.getWritePointer (0, half), audio.getWritePointer (1, half), audio.getNumSamples() - half,
                              WatermarkKind::None);
            const int after = half + (int) (sr * 0.06);
            float maxDiff = 0.0f;
            for (int i = after; i < audio.getNumSamples(); ++i)
                maxDiff = juce::jmax (maxDiff, std::abs (audio.getSample (0, i) - clean.getSample (0, i)));
            expectEquals (maxDiff, 0.0f);
        }
    }
};

static WatermarkTests watermarkTests;

} // namespace voxa
