#include "neural/FeatureIndex.h"

#include <cmath>
#include <cstring>

/*  The AI singer's feature index (category "Neural"): reading the pack's .npy file, the nearest-feature retrieval,
    and the blend toward it. */

namespace voxa::neural
{

namespace
{
    /** A NumPy .npy file of [rows, channels] (float16 when `half`; the values must be exact in float16). */
    juce::File writeNpy (const juce::String& name, const std::vector<std::vector<float>>& rows, bool half)
    {
        const int channels = (int) rows.front().size();
        juce::String header = "{'descr': '" + juce::String (half ? "<f2" : "<f4") + "', 'fortran_order': False, 'shape': ("
                              + juce::String ((int) rows.size()) + ", " + juce::String (channels) + "), }";
        while ((10 + header.length() + 1) % 64 != 0)
            header << " ";
        header << "\n";

        juce::MemoryOutputStream out;
        out.write ("\x93NUMPY\x01\x00", 8);
        out.writeShort ((short) header.length());
        out.write (header.toRawUTF8(), (size_t) header.length());
        for (const auto& row : rows)
        {
            for (float v : row)
            {
                if (half)
                {
                    // exact for the small integers and halves used here
                    uint32_t bits;
                    std::memcpy (&bits, &v, 4);
                    const uint16_t sign = (uint16_t) ((bits >> 16) & 0x8000);
                    const int exponent = (int) ((bits >> 23) & 0xff) - 127 + 15;
                    const uint16_t h = v == 0.0f ? sign : (uint16_t) (sign | (exponent << 10) | ((bits >> 13) & 0x3ff));
                    out.writeShort ((short) h);
                }
                else
                {
                    out.writeFloat (v);
                }
            }
        }

        auto file = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile (name);
        file.replaceWithData (out.getData(), out.getDataSize());
        return file;
    }
}

class FeatureIndexTests : public juce::UnitTest
{
public:
    FeatureIndexTests() : juce::UnitTest ("Feature index", "Neural") {}

    void runTest() override
    {
        // three of her sounds, far apart, in 20 channels (more than one vector lane of the distance loop)
        constexpr int channels = 20;
        std::vector<std::vector<float>> sounds (3, std::vector<float> (channels, 0.0f));
        for (int ch = 0; ch < channels; ++ch)
        {
            sounds[0][(size_t) ch] = 1.0f;
            sounds[1][(size_t) ch] = ch % 2 == 0 ? 4.0f : -4.0f;
            sounds[2][(size_t) ch] = ch < 10 ? 8.5f : 0.0f;
        }

        beginTest ("The pack's .npy features load as float16 or float32; a file for another encoder is refused");
        {
            for (bool half : { true, false })
            {
                juce::String error;
                const auto file = writeNpy ("voxa-index-test.npy", sounds, half);
                auto index = FeatureIndex::load (file, channels, error);
                expect (index != nullptr, error);
                if (index == nullptr)
                    continue;
                expectEquals (index->getNumFeatures(), 3);
                expectEquals (index->getChannels(), channels);

                // a sound she sang comes back as itself
                std::vector<float> out (channels);
                for (const auto& s : sounds)
                {
                    index->retrieve (s.data(), out.data());
                    float worst = 0.0f;
                    for (int ch = 0; ch < channels; ++ch)
                        worst = juce::jmax (worst, std::abs (out[(size_t) ch] - s[(size_t) ch]));
                    expectLessThan (worst, 1.0e-3f);
                }

                expect (FeatureIndex::load (file, 768, error) == nullptr && error.contains ("768"),
                        "features of another width");
                file.deleteFile();
            }

            juce::String error;
            auto notNumpy = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("voxa-index-test.txt");
            notNumpy.replaceWithText ("hello, not an index");
            expect (FeatureIndex::load (notNumpy, channels, error) == nullptr && error.isNotEmpty());
            notNumpy.deleteFile();
        }

        beginTest ("A sound near one of hers is pulled to it; the nearest counts most");
        {
            juce::String error;
            const auto file = writeNpy ("voxa-index-test.npy", sounds, false);
            auto index = FeatureIndex::load (file, channels, error);
            file.deleteFile();
            expect (index != nullptr, error);
            if (index == nullptr)
                return;

            auto query = sounds[1];
            for (auto& v : query)
                v *= 0.9f;   // close to her second sound
            std::vector<float> out (channels);
            index->retrieve (query.data(), out.data());
            float toSecond = 0.0f, toFirst = 0.0f;
            for (int ch = 0; ch < channels; ++ch)
            {
                toSecond += std::pow (out[(size_t) ch] - sounds[1][(size_t) ch], 2.0f);
                toFirst += std::pow (out[(size_t) ch] - sounds[0][(size_t) ch], 2.0f);
            }
            expectLessThan (std::sqrt (toSecond), 0.5f);
            expectGreaterThan (std::sqrt (toFirst), 10.0f);

            beginTest ("Blend: rate of the way in sung frames, `protect` of that in consonants; repeated frames alike");
            {
                // frames: voiced, the same frame again (the encoder's 20 ms frames come twice), unvoiced
                std::vector<float> frames;
                for (int k = 0; k < 3; ++k)
                    frames.insert (frames.end(), query.begin(), query.end());
                const float voiced[] = { 1.0f, 1.0f, 0.0f };
                std::vector<float> scratch (2 * channels);
                const float rate = 0.75f, protect = 0.33f;
                index->blend (frames.data(), 3, voiced, rate, protect, scratch.data());

                for (int ch = 0; ch < channels; ++ch)
                {
                    const float own = query[(size_t) ch], target = out[(size_t) ch];
                    expectWithinAbsoluteError (frames[(size_t) ch], own + rate * (target - own), 1.0e-4f);
                    expectWithinAbsoluteError (frames[(size_t) (channels + ch)], own + rate * (target - own), 1.0e-4f);
                    expectWithinAbsoluteError (frames[(size_t) (2 * channels + ch)], own + rate * protect * (target - own), 1.0e-4f);
                }

                auto untouched = query;
                index->blend (untouched.data(), 1, voiced, 0.0f, protect, scratch.data());
                expect (untouched == query, "rate 0 changes nothing");
            }
        }
    }
};

static FeatureIndexTests featureIndexTests;

} // namespace voxa::neural
