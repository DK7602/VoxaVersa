#include "FeatureIndex.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace voxa::neural
{

namespace
{
    float halfToFloat (uint16_t h) noexcept
    {
        const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
        const uint32_t exponent = (h >> 10) & 0x1f;
        uint32_t mantissa = h & 0x3ff;
        uint32_t bits;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else   // subnormal
            {
                int e = -1;
                do { ++e; mantissa <<= 1; } while ((mantissa & 0x400) == 0);
                bits = sign | ((uint32_t) (127 - 15 - e) << 23) | ((mantissa & 0x3ff) << 13);
            }
        }
        else if (exponent == 31)
        {
            bits = sign | 0x7f800000u | (mantissa << 13);
        }
        else
        {
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }

        float f;
        std::memcpy (&f, &bits, sizeof (f));
        return f;
    }

    /** a . b, in independent lanes the compiler can vectorise without reordering a single sum */
    float dot (const float* a, const float* b, int n) noexcept
    {
        constexpr int lanes = 16;
        float acc[lanes] {};
        int i = 0;
        for (; i + lanes <= n; i += lanes)
            for (int k = 0; k < lanes; ++k)
                acc[k] += a[i + k] * b[i + k];
        float sum = 0.0f;
        for (int k = 0; k < lanes; ++k)
            sum += acc[k];
        for (; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }
}

std::shared_ptr<const FeatureIndex> FeatureIndex::load (const juce::File& file, int expectedChannels, juce::String& error)
{
    juce::MemoryBlock bytes;
    if (! file.loadFileAsData (bytes) || bytes.getSize() < 16)
    {
        error = "can't read " + file.getFileName();
        return nullptr;
    }

    const auto* p = static_cast<const uint8_t*> (bytes.getData());
    if (std::memcmp (p, "\x93NUMPY", 6) != 0)
    {
        error = file.getFileName() + " is not a NumPy .npy file";
        return nullptr;
    }

    const int major = p[6];
    const size_t headerLength = major >= 2 ? (size_t) (p[8] | (p[9] << 8) | (p[10] << 16) | ((uint32_t) p[11] << 24))
                                           : (size_t) (p[8] | (p[9] << 8));
    const size_t headerStart = major >= 2 ? 12 : 10;
    if (headerStart + headerLength > bytes.getSize())
    {
        error = file.getFileName() + ": damaged header";
        return nullptr;
    }

    const juce::String header (reinterpret_cast<const char*> (p + headerStart), headerLength);
    const bool half = header.contains ("'<f2'");
    const bool single = header.contains ("'<f4'");
    if ((! half && ! single) || header.contains ("'fortran_order': True"))
    {
        error = file.getFileName() + ": expected little-endian float16 or float32 in C order";
        return nullptr;
    }

    const auto shape = header.fromFirstOccurrenceOf ("'shape':", false, false)
                             .fromFirstOccurrenceOf ("(", false, false)
                             .upToFirstOccurrenceOf (")", false, false);
    juce::StringArray dims;
    dims.addTokens (shape, ",", {});
    dims.removeEmptyStrings();
    if (dims.size() != 2)
    {
        error = file.getFileName() + ": expected a [count, channels] array";
        return nullptr;
    }

    const int count = dims[0].trim().getIntValue();
    const int channels = dims[1].trim().getIntValue();
    const size_t elementSize = half ? 2 : 4;
    const size_t dataStart = headerStart + headerLength;
    if (count <= 0 || channels != expectedChannels
        || dataStart + (size_t) count * (size_t) channels * elementSize > bytes.getSize())
    {
        error = file.getFileName() + ": " + juce::String (count) + " x " + juce::String (channels)
                  + " features, the voice model needs " + juce::String (expectedChannels) + " channels";
        return nullptr;
    }

    std::shared_ptr<FeatureIndex> index (new FeatureIndex());
    index->count = count;
    index->channels = channels;
    index->data.resize ((size_t) count * (size_t) channels);
    index->norms.resize ((size_t) count);

    const auto* source = p + dataStart;
    for (size_t i = 0; i < index->data.size(); ++i)
    {
        if (half)
        {
            uint16_t h;
            std::memcpy (&h, source + i * 2, 2);
            index->data[i] = halfToFloat (h);
        }
        else
        {
            std::memcpy (&index->data[i], source + i * 4, 4);
        }
    }

    for (int k = 0; k < count; ++k)
    {
        const float* row = index->data.data() + (size_t) k * (size_t) channels;
        index->norms[(size_t) k] = dot (row, row, channels);
    }

    return index;
}

void FeatureIndex::retrieve (const float* query, float* out) const noexcept
{
    // the nearest `neighbours`, kept sorted (squared distance = |q|^2 + |c|^2 - 2 q.c)
    float best[neighbours];
    int which[neighbours];
    std::fill (best, best + neighbours, std::numeric_limits<float>::max());
    std::fill (which, which + neighbours, 0);

    const float q2 = dot (query, query, channels);
    for (int k = 0; k < count; ++k)
    {
        const float d = q2 + norms[(size_t) k] - 2.0f * dot (query, data.data() + (size_t) k * (size_t) channels, channels);
        if (d >= best[neighbours - 1])
            continue;
        int j = neighbours - 1;
        while (j > 0 && best[j - 1] > d)
        {
            best[j] = best[j - 1];
            which[j] = which[j - 1];
            --j;
        }
        best[j] = d;
        which[j] = k;
    }

    const int found = std::min (count, neighbours);
    float weights[neighbours], total = 0.0f;
    for (int j = 0; j < found; ++j)
    {
        const float d = std::max (best[j], 1.0e-6f);
        weights[j] = 1.0f / (d * d);
        total += weights[j];
    }

    std::fill (out, out + channels, 0.0f);
    for (int j = 0; j < found; ++j)
    {
        const float w = weights[j] / total;
        const float* row = data.data() + (size_t) which[j] * (size_t) channels;
        for (int ch = 0; ch < channels; ++ch)
            out[ch] += w * row[ch];
    }
}

void FeatureIndex::blend (float* features, int numFrames, const float* voiced, float rate, float protect,
                          float* scratch) const noexcept
{
    if (rate <= 0.0f || count == 0)
        return;

    float* own = scratch;                 // the frame as it came from the encoder
    float* retrieved = scratch + channels;
    bool haveRetrieved = false;

    for (int i = 0; i < numFrames; ++i)
    {
        float* frame = features + (size_t) i * (size_t) channels;
        // the encoder's 20 ms frames come twice: the second copy equals the previous frame's encoder output
        const bool repeat = haveRetrieved && std::equal (frame, frame + channels, own);
        if (! repeat)
        {
            std::copy (frame, frame + channels, own);
            retrieve (own, retrieved);
            haveRetrieved = true;
        }

        const float amount = rate * (voiced == nullptr || voiced[i] > 0.0f ? 1.0f : protect);
        for (int ch = 0; ch < channels; ++ch)
            frame[ch] = own[ch] + amount * (retrieved[ch] - own[ch]);
    }
}

} // namespace voxa::neural
