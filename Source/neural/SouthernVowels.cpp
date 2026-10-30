#include "SouthernVowels.h"
#include "VoxaAssets.h"

#include <cmath>
#include <cstring>

namespace voxa::neural
{

namespace
{
    float halfToFloat (uint16_t h) noexcept
    {
        const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
        uint32_t exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff, bits;

        if (exponent == 0)
        {
            if (mantissa == 0)
                bits = sign;
            else
            {
                exponent = 127 - 15 + 1;
                while ((mantissa & 0x400) == 0) { mantissa <<= 1; --exponent; }
                bits = sign | (exponent << 23) | ((mantissa & 0x3ff) << 13);
            }
        }
        else if (exponent == 31)
            bits = sign | 0x7f800000 | (mantissa << 13);
        else
            bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);

        float f;
        std::memcpy (&f, &bits, sizeof f);
        return f;
    }
}

//==============================================================================
const SouthernVowels* SouthernVowels::get()
{
    static const SouthernVowels* instance = []() -> const SouthernVowels*
    {
        int size = 0;
        const auto* data = VoxaAssets::getNamedResource ("southern_vowels_gz", size);
        if (data == nullptr || size <= 0)
            return nullptr;

        juce::MemoryInputStream compressed (data, (size_t) size, false);
        juce::GZIPDecompressorInputStream gunzip (&compressed, false, juce::GZIPDecompressorInputStream::gzipFormat);
        juce::MemoryBlock raw;
        gunzip.readIntoMemoryBlock (raw);

        static SouthernVowels loaded;
        return loaded.load (raw.getData(), raw.getSize()) ? &loaded : nullptr;
    }();
    return instance;
}

/** "VXVW", version (1 or 2), channels, context, hidden, classes (uint32 each), the class names (uint8 length +
    text), then float16: w1 [hidden][context * channels], b1 [hidden], w2 [classes][hidden], b2 [classes].
    Version 2 goes on with the shifts (ShiftRule): their number (uint32), then each its name (uint8 length + text:
    raise, lower, drawl) and float16 [channels]. */
bool SouthernVowels::load (const void* data, size_t size)
{
    juce::MemoryInputStream in (data, size, false);
    char magic[4];
    if (in.read (magic, 4) != 4 || std::memcmp (magic, "VXVW", 4) != 0)
        return false;
    const int version = in.readInt();
    if (version != 1 && version != 2)
        return false;

    channels = in.readInt(); context = in.readInt(); hidden = in.readInt(); numClasses = in.readInt();
    if (channels <= 0 || context <= 0 || context % 2 == 0 || hidden <= 0 || numClasses <= 0 || channels > 4096 || hidden > 4096)
        return false;

    for (int c = 0; c < numClasses; ++c)
    {
        const auto label = [&in]
        {
            juce::MemoryBlock name;
            in.readIntoMemoryBlock (name, (int) in.readByte());
            return name.toString();
        }();

        if (label == "AY" || label == "OY")
            iyClasses.push_back (c);
        else if (label == "AY_g" || label == "OY_g")
            glideClasses.push_back (c);
        if (label != "C" && label != "SIL")
            vowelClasses.push_back (c);
        if (label == "EH_N") shiftClasses[ShiftRule::raise] = c;
        if (label == "EY")   shiftClasses[ShiftRule::lower] = c;
        if (label == "AE")   shiftClasses[ShiftRule::drawl] = c;
    }

    auto readName = [&in]
    {
        juce::MemoryBlock name;
        in.readIntoMemoryBlock (name, (int) in.readByte());
        return name.toString();
    };

    auto readHalves = [&in] (std::vector<float>& dest, size_t count)
    {
        dest.resize (count);
        for (auto& v : dest)
            v = halfToFloat ((uint16_t) in.readShort());
        return ! in.isExhausted() || in.getPosition() == in.getTotalLength();
    };

    const auto inputs = (size_t) (context * channels);
    if (! (readHalves (w1, (size_t) hidden * inputs) && readHalves (b1, (size_t) hidden)
           && readHalves (w2, (size_t) numClasses * (size_t) hidden) && readHalves (b2, (size_t) numClasses)
           && ! iyClasses.empty() && ! glideClasses.empty()))
        return false;

    if (version == 2)
    {
        const int numShifts = in.readInt();
        for (int k = 0; k < numShifts; ++k)
        {
            const auto name = readName();
            std::vector<float> delta;
            if (! readHalves (delta, (size_t) channels))
                return false;
            const int which = name == "raise" ? ShiftRule::raise : name == "lower" ? ShiftRule::lower : name == "drawl" ? ShiftRule::drawl : -1;
            if (which >= 0 && shiftClasses[(size_t) which] >= 0)
                deltas[(size_t) which] = std::move (delta);
        }
    }

    return in.getPosition() == in.getTotalLength();
}

//==============================================================================
SouthernVowels::Rule::Event SouthernVowels::Rule::step (const Frame& f, const Frame& next, const Frame& afterNext) noexcept
{
    if (! isVowel (f))
    {
        reset();
        return Event::none;
    }

    if (held)
        return Event::holding;

    const bool glides = isGlide (f) && isGlide (next);
    const bool ends = isVowel (next) && ! isVowel (afterNext);

    float nucleus = 0.0f;
    for (auto p : recent)
        nucleus += p / holdFrames;

    if ((glides || ends) && vowelFrames >= holdFrames && evidence / vowelFrames >= evidenceNeeded && nucleus >= nucleusNeeded)
    {
        held = true;
        return Event::glideStarts;
    }

    // the glide itself is no evidence: a long one after another vowel must not add up to an "I"
    if (! isGlide (f))
    {
        recent[(size_t) (vowelFrames % holdFrames)] = f.iy / f.vowel;
        ++vowelFrames;
        evidence += (double) ((f.iy + f.glide) / f.vowel);
    }

    return Event::none;
}

SouthernVowels::ShiftRule::Decision SouthernVowels::ShiftRule::step (const Frame& f, const Frame& next,
                                                                    const Frame& afterNext) noexcept
{
    if (! Rule::isVowel (f))
    {
        reset();
        return {};
    }

    ++index;
    for (size_t k = 0; k < numShifts; ++k)
        sums[k] += (double) (f.shift[k] / f.vowel);

    auto lowerShape = [] (int i) { return juce::jmin (1.0f, (float) (i + 1) / 3.0f) * juce::jmax (0.0f, 1.0f - (float) i / (float) lowerFrames); };
    auto bump = [] (int k) { return k < 0 ? 0.0f : (k < 4 ? (float) (k + 1) / 4.0f : juce::jmax (0.0f, 1.0f - (float) (k - 3) / 4.0f)); };

    Decision d;
    if (shift < 0 && index >= 2)
    {
        int best = -1;
        double margin = -1.0e9;
        for (size_t k = 0; k < numShifts; ++k)
            if (const auto m = sums[k] / (index + 1) - (double) needed[k]; m > margin)
            {
                margin = m;
                best = (int) k;
            }

        if (margin >= 0.0)
        {
            shift = best;
            drawlStart = juce::jmax (index, drawlDelay);
            for (int j = 0; j < 2; ++j)   // the two frames before: raise and lower start there
            {
                const int at = index - 2 + j;
                if (at >= 0 && shift == raise)
                    d.before[(size_t) j] = (float) (j + 1) / 3.0f;
                else if (at >= 0 && shift == lower)
                    d.before[(size_t) j] = lowerShape (at);
            }
        }
    }

    if (shift >= 0)
    {
        // in and out with the vowel's end, known 2 frames ahead
        const float out = ! Rule::isVowel (next) ? 1.0f / 3.0f : (! Rule::isVowel (afterNext) ? 2.0f / 3.0f : 1.0f);
        d.shift = shift;
        d.weight = out * (shift == raise ? 1.0f : shift == lower ? lowerShape (index) : bump (index - drawlStart));
    }
    return d;
}

void SouthernVowels::State::reset() noexcept
{
    rule.reset();
    shiftRule.reset();
    shiftOf.fill (-1);
    shiftWeight.fill (0.0f);
    recognised = -1;
    frameOf.fill (-1);
    weight.fill (0.0f);
    holdIndex.fill (0);
    holds[0].clear();
    holds[1].clear();
    currentHold = 0;
}

//==============================================================================
SouthernVowels::Frame SouthernVowels::recognise (const float* features, int numEncoderFrames, int frame,
                                                 std::vector<float>& scratch) const
{
    const auto inputs = (size_t) (context * channels);
    scratch.resize (inputs + (size_t) hidden + (size_t) numClasses);
    float* x = scratch.data();
    float* h = x + inputs;
    float* logits = h + hidden;

    for (int o = 0; o < context; ++o)
    {
        const int e = juce::jlimit (0, numEncoderFrames - 1, frame + o - context / 2);
        std::copy_n (features + (size_t) e * (size_t) channels, channels, x + (size_t) o * (size_t) channels);
    }

    for (int u = 0; u < hidden; ++u)
    {
        const float* w = w1.data() + (size_t) u * inputs;
        float sum = b1[(size_t) u];
        for (size_t i = 0; i < inputs; ++i)
            sum += w[i] * x[i];
        h[u] = juce::jmax (0.0f, sum);
    }

    float top = -1.0e30f;
    for (int c = 0; c < numClasses; ++c)
    {
        const float* w = w2.data() + (size_t) c * (size_t) hidden;
        float sum = b2[(size_t) c];
        for (int u = 0; u < hidden; ++u)
            sum += w[u] * h[u];
        logits[c] = sum;
        top = juce::jmax (top, sum);
    }

    float total = 0.0f;
    for (int c = 0; c < numClasses; ++c)
        total += (logits[c] = std::exp (logits[c] - top));

    Frame f;
    for (auto c : vowelClasses) f.vowel += logits[c] / total;
    for (auto c : iyClasses)    f.iy += logits[c] / total;
    for (auto c : glideClasses) f.glide += logits[c] / total;
    for (size_t k = 0; k < ShiftRule::numShifts; ++k)
        if (shiftClasses[k] >= 0)
            f.shift[k] = logits[shiftClasses[k]] / total;
    return f;
}

void SouthernVowels::apply (const float* features, int numEncoderFrames, int ch, const float* f0Frames, int numFrames,
                            int skip, int64_t firstFrame, float* phone, State& state, std::vector<float>& scratch) const
{
    if (ch != channels || numEncoderFrames < context || numFrames <= skip)
        return;

    const int64_t first = firstFrame >= 0 ? firstFrame / 2 : (firstFrame - 1) / 2;   // the chunk's first encoder frame
    if (state.recognised >= 0 && (state.recognised < first - 1 || state.recognised >= first + numEncoderFrames))
        state.reset();                                                                // a jump (transport, reset)

    auto slot = [] (int64_t a) { return (size_t) (a % State::history + State::history) % State::history; };
    auto voicedAt = [&] (int e)
    {
        const int i = 2 * e;
        return (i < numFrames && f0Frames[i] > 0.0f) || (i + 1 < numFrames && f0Frames[i + 1] > 0.0f);
    };

    // recognise every frame that has its full context in this chunk, deciding the one `lookahead` frames before
    const int64_t lastRecognisable = first + numEncoderFrames - 1 - context / 2;
    for (int64_t a = juce::jmax (state.recognised + 1, first); a <= lastRecognisable; ++a)
    {
        const int e = (int) (a - first);
        auto f = recognise (features, numEncoderFrames, e, scratch);
        f.voiced = voicedAt (e);

        const auto s = slot (a);
        state.frameOf[s] = a;
        state.frames[s] = f;
        state.weight[s] = 0.0f;
        state.shiftOf[s] = -1;
        state.shiftWeight[s] = 0.0f;
        state.recognised = a;

        const int64_t d = a - Rule::lookahead;
        const auto sd = slot (d);
        if (state.frameOf[sd] != d)
            continue;   // just after a reset

        const auto event = state.rule.step (state.frames[sd], state.frames[slot (d + 1)], f);

        if (event == Rule::Event::glideStarts)
        {
            // the 60 ms of this vowel before the glide that sound most like its "ah" (the latest of equals)
            int64_t from = -1;
            float best = -1.0f;
            for (int64_t w = d - Rule::holdFrames; w >= juce::jmax (first, d - Rule::searchFrames); --w)
            {
                float score = 0.0f;
                bool vowel = true, glide = false;
                for (int k = 0; k < Rule::holdFrames; ++k)
                {
                    const auto t = slot (w + k);
                    const auto& x = state.frames[t];
                    vowel = vowel && state.frameOf[t] == w + k && Rule::isVowel (x);
                    glide = glide || Rule::isGlide (x);
                    score += vowel ? x.iy / x.vowel : 0.0f;
                }
                if (! vowel)
                    break;   // the vowel's start
                if (! glide && score > best)
                {
                    best = score;
                    from = w;
                }
            }

            if (from >= 0)
            {
                state.currentHold ^= 1;
                auto& hold = state.holds[state.currentHold];
                hold.assign ((size_t) channels, 0.0f);
                for (int k = 0; k < Rule::holdFrames; ++k)
                    for (int c = 0; c < channels; ++c)
                        hold[(size_t) c] += features[(size_t) (from - first + k) * (size_t) channels + (size_t) c] / Rule::holdFrames;

                // from 40 ms before the glide: half, then all of it
                for (int64_t b = d - 2; b <= d; ++b)
                {
                    const auto t = slot (b);
                    if (state.frameOf[t] == b)
                    {
                        state.weight[t] = b == d - 2 ? 0.5f : 1.0f;
                        state.holdIndex[t] = state.currentHold;
                    }
                }
            }
            else
            {
                state.rule.reset();   // none of the vowel before the glide is in this chunk: leave it alone
            }
        }
        else if (event == Rule::Event::holding)
        {
            state.weight[sd] = 1.0f;
            state.holdIndex[sd] = state.currentHold;
        }

        if (hasShifts())
        {
            const auto decision = state.shiftRule.step (state.frames[sd], state.frames[slot (d + 1)], f);
            state.shiftOf[sd] = (int8_t) decision.shift;
            state.shiftWeight[sd] = decision.weight;
            for (int j = 0; j < 2; ++j)
            {
                const int64_t b = d - 2 + j;
                const auto t = slot (b);
                if (decision.before[(size_t) j] >= 0.0f && state.frameOf[t] == b)
                {
                    state.shiftOf[t] = (int8_t) decision.shift;
                    state.shiftWeight[t] = decision.before[(size_t) j];
                }
            }
        }
    }

    const int64_t decided = state.recognised - Rule::lookahead;

    // the generator's frames: decided ones as decided; the last few (no context yet) go on as the last decided
    // frame did while voiced
    const auto lastDecided = slot (decided);
    const bool lastKnown = decided >= 0 && state.frameOf[lastDecided] == decided;

    for (int i = skip; i < numFrames; ++i)
    {
        const int e = i / 2;
        const int64_t a = first + e;
        float w = 0.0f, shiftWeight = 0.0f;
        uint8_t which = state.currentHold;
        int shift = -1;

        if (a <= decided)
        {
            const auto s = slot (a);
            if (state.frameOf[s] == a)
            {
                w = state.weight[s];
                which = state.holdIndex[s];
                shift = state.shiftOf[s];
                shiftWeight = state.shiftWeight[s];
            }
        }
        else if (voicedAt (e))
        {
            w = state.rule.held ? 1.0f : 0.0f;
            if (lastKnown)
            {
                shift = state.shiftOf[lastDecided];
                shiftWeight = state.shiftWeight[lastDecided];
            }
        }

        float* dest = phone + (size_t) (i - skip) * (size_t) channels;

        if (const auto& hold = state.holds[which]; w > 0.0f && hold.size() == (size_t) channels)
            for (int c = 0; c < channels; ++c)
                dest[c] += w * (hold[(size_t) c] - dest[c]);

        if (shift >= 0 && shiftWeight > 0.0f)
        {
            const auto& delta = deltas[(size_t) shift];
            for (int c = 0; c < channels; ++c)
                dest[c] += shiftWeight * delta[(size_t) c];
        }
    }
}

} // namespace voxa::neural
