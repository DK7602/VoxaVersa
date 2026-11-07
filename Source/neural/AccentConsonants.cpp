#include "AccentConsonants.h"
#include "VoxaAssets.h"

#include <algorithm>
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
const AccentConsonants* AccentConsonants::get()
{
    static const AccentConsonants* instance = []() -> const AccentConsonants*
    {
        int size = 0;
        const auto* data = VoxaAssets::getNamedResource ("accent_consonants_gz", size);
        if (data == nullptr || size <= 0)
            return nullptr;

        juce::MemoryInputStream compressed (data, (size_t) size, false);
        juce::GZIPDecompressorInputStream gunzip (&compressed, false, juce::GZIPDecompressorInputStream::gzipFormat);
        juce::MemoryBlock raw;
        gunzip.readIntoMemoryBlock (raw);

        static AccentConsonants loaded;
        return loaded.load (raw.getData(), raw.getSize()) ? &loaded : nullptr;
    }();
    return instance;
}

std::unique_ptr<AccentConsonants> AccentConsonants::fromData (const void* data, size_t size)
{
    std::unique_ptr<AccentConsonants> c (new AccentConsonants());
    return c->load (data, size) ? std::move (c) : nullptr;
}

/** "VXVC", version 1, channels, context, hidden, classes (uint32 each), the class names (uint8 length + text), then
    float16: w1 [hidden][context * channels], b1 [hidden], w2 [classes][hidden], b2 [classes]; then the templates:
    their number (uint32), each its name (uint8 length + text: t, d_a, d_e, d_i, silence), its frames (uint32) and
    float16 [frames][channels] (tools/research/accents/export_consonants.py). */
bool AccentConsonants::load (const void* data, size_t size)
{
    juce::MemoryInputStream in (data, size, false);
    char magic[4];
    if (in.read (magic, 4) != 4 || std::memcmp (magic, "VXVC", 4) != 0 || in.readInt() != 1)
        return false;

    channels = in.readInt(); context = in.readInt(); hidden = in.readInt(); numClasses = in.readInt();
    if (channels <= 0 || context <= 0 || context % 2 == 0 || hidden <= 0 || numClasses <= 0 || channels > 4096 || hidden > 4096
        || numClasses > 256)
        return false;

    auto readName = [&in]
    {
        juce::MemoryBlock name;
        in.readIntoMemoryBlock (name, (int) in.readByte());
        return name.toString();
    };

    for (int c = 0; c < numClasses; ++c)
    {
        const auto label = readName();
        for (auto [name, index] : { std::pair<const char*, int*> { "TH", &thClass }, { "DH", &dhClass }, { "D", &dClass }, { "T", &tClass }, { "Y", &yClass },
                                    { "N", &nClass }, { "R", &rClass }, { "VOW", &vowelClass } })
            if (label == name)
                *index = c;
    }

    auto readHalves = [&in] (std::vector<float>& dest, size_t count)
    {
        dest.resize (count);
        for (auto& v : dest)
            v = halfToFloat ((uint16_t) in.readShort());
        return ! in.isExhausted() || in.getPosition() == in.getTotalLength();
    };

    const auto inputs = (size_t) (context * channels);
    if (! (readHalves (w1, (size_t) hidden * inputs) && readHalves (b1, (size_t) hidden)
           && readHalves (w2, (size_t) numClasses * (size_t) hidden) && readHalves (b2, (size_t) numClasses)))
        return false;
    if (thClass < 0 || dhClass < 0 || dClass < 0 || tClass < 0 || nClass < 0 || rClass < 0 || yClass < 0 || vowelClass < 0)
        return false;

    const int numTemplates = in.readInt();
    for (int k = 0; k < numTemplates; ++k)
    {
        const auto name = readName();
        const int frames = in.readInt();
        if (frames <= 0 || frames > 64)
            return false;
        std::vector<float> values;
        if (! readHalves (values, (size_t) frames * (size_t) channels))
            return false;
        const int which = name == "t" ? Rule::t : name == "d_a" ? Rule::dA : name == "d_e" ? Rule::dE : name == "d_i" ? Rule::dI
                        : name == "roll" ? Rule::roll : name == "jh" ? Rule::jh : -1;
        if (which >= 0)
            templates[(size_t) which] = std::move (values);
        else if (name == "silence")
            silence = std::move (values);
    }

    for (int t = Rule::t; t <= Rule::dI; ++t)
        if (templates[(size_t) t].size() != 4 * (size_t) channels)
            return false;
    return silence.size() == (size_t) channels;
}

//==============================================================================
AccentConsonants::Frame AccentConsonants::recognise (const float* features, int numEncoderFrames, int frame,
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
    f.th = logits[thClass] / total;
    f.dh = logits[dhClass] / total;
    f.d = logits[dClass] / total;
    f.t = logits[tClass] / total;
    f.n = logits[nClass] / total;
    f.r = logits[rClass] / total;
    f.y = logits[yClass] / total;
    f.vowel = logits[vowelClass] / total;
    return f;
}

//==============================================================================
AccentConsonants::Rule::Decision AccentConsonants::Rule::step (const Frame& f, const Frame& next, const Frame& afterNext) noexcept
{
    Decision d;

    // "th" -> "t" / "d": the template's last two frames at the last two frames of the "th", closure before them
    if (isTh (f))
    {
        if (thRun == 0)
        {
            legato = prevVoiced;
            thSum = dhSum = 0.0f;
        }
        ++thRun;
        thSum += f.th;
        dhSum += f.dh;
        d.voicedStop = dhSum >= thSum;
        if (! isTh (next))
        {
            d.kind = Decision::stop;
            d.frame = 3;
        }
        else if (! isTh (afterNext))
        {
            d.kind = Decision::stop;
            d.frame = 2;
        }
        else
        {
            d.kind = d.voicedStop ? Decision::voiceBar : Decision::closure;
        }
    }
    else
    {
        thRun = 0;
    }

    // an' you: a "d" (or "t") right after an "n" sung as the "n"
    const bool stopAfterN = f.d + f.t >= dNeeded;
    if (d.kind == Decision::none && stopAfterN && (sinceN <= nWithin || droppedD > 0) && droppedD < maxDroppedD)
    {
        ++droppedD;
        d.kind = Decision::holdN;
        d.weight = droppedD == 1 ? 0.5f : 1.0f;
    }
    else if (! stopAfterN)
    {
        droppedD = 0;
    }

    // fo: an "r" after a vowel, long enough not to be one between vowels (very), sung as the vowel
    const bool r = f.voiced && f.r >= rNeeded;
    if (r)
    {
        ++rRun;
        if (d.kind == Decision::none)
        {
            if (holdingVowel)
            {
                d.kind = Decision::holdVowel;
                d.weight = 1.0f;
                // Latin: the vowel held, then a rolled "rr" on the r's last two frames (fo-rr)
                if (spanish && next.r < rNeeded)
                {
                    d.kind = Decision::roll;
                    d.frame = 1;
                }
                else if (spanish && afterNext.r < rNeeded)
                {
                    d.kind = Decision::roll;
                    d.frame = 0;
                }
            }
            else if (rRun == 2 && vowelRun >= vowelBeforeR && next.r >= rNeeded && afterNext.r >= rNeeded)
            {
                holdingVowel = true;
                d.kind = Decision::holdVowel;
                d.weight = 0.5f;
            }
        }
    }
    else
    {
        rRun = 0;
        holdingVowel = false;
    }

    // Latin: "you" -> "jou": a "y" sung as Kokoro's "j" and its move into the vowel
    if (spanish && f.voiced && f.y >= yNeeded)
    {
        if (d.kind == Decision::none && yRun < 5)
        {
            d.kind = Decision::jh;
            d.frame = (int8_t) yRun;
        }
        ++yRun;
    }
    else
    {
        yRun = 0;
    }

    vowelRun = isVowel (f) && ! r ? vowelRun + 1 : (r ? vowelRun : 0);
    sinceN = f.n >= nNeeded ? 1 : juce::jmin (99, sinceN + 1);
    prevVoiced = f.voiced;
    return d;
}

void AccentConsonants::State::reset() noexcept
{
    rule.reset();
    recognised = -1;
    frameOf.fill (-1);
    decisions.fill ({});
    stopTemplate.fill (-1);
    stopPitch.fill (0.0f);
    holdIndex.fill (0);
    holds[0].clear();
    holds[1].clear();
    currentHold = 0;
    currentTemplate = Rule::dA;
    lastPitch = 0.0f;
}

//==============================================================================
void AccentConsonants::apply (const float* features, int numEncoderFrames, int ch, float* f0Frames, int numFrames,
                              int skip, int64_t firstFrame, float* phone, State& state, std::vector<float>& scratch,
                              float* changed, int lookAheadFrames) const
{
    if (ch != channels || numEncoderFrames < context || numFrames <= skip)
        return;

    const int64_t first = firstFrame >= 0 ? firstFrame / 2 : (firstFrame - 1) / 2;   // the chunk's first encoder frame
    if (state.recognised >= 0 && (state.recognised < first - 1 || state.recognised >= first + numEncoderFrames))
        state.reset();                                                                // a jump (transport, reset)
    state.rule.spanish = state.latin && hasLatin();

    auto slot = [] (int64_t a) { return (size_t) (a % State::history + State::history) % State::history; };
    auto pitchAt = [&] (int e)   // the encoder frame's pitch (0: unvoiced)
    {
        const int i = 2 * e;
        return i < numFrames && f0Frames[i] > 0.0f ? f0Frames[i] : (i + 1 < numFrames ? juce::jmax (0.0f, f0Frames[i + 1]) : 0.0f);
    };
    auto featuresOf = [&] (int64_t a) { return features + (size_t) (a - first) * (size_t) channels; };
    auto similarity = [&] (const float* a, const float* b)
    {
        double ab = 0.0, aa = 0.0, bb = 0.0;
        for (int c = 0; c < channels; ++c)
        {
            ab += (double) a[c] * b[c];
            aa += (double) a[c] * a[c];
            bb += (double) b[c] * b[c];
        }
        return ab / std::sqrt (aa * bb + 1.0e-12);
    };

    constexpr int lookahead = 2;
    const int64_t lastRecognisable = juce::jmin (first + numEncoderFrames - 1 - context / 2,
                                                 first + (numFrames - lookAheadFrames) / 2 - 1 + lookahead);
    for (int64_t a = juce::jmax (state.recognised + 1, first); a <= lastRecognisable; ++a)
    {
        const int e = (int) (a - first);
        auto f = recognise (features, numEncoderFrames, e, scratch);
        f.voiced = pitchAt (e) > 0.0f;

        const auto s = slot (a);
        state.frameOf[s] = a;
        state.frames[s] = f;
        state.decisions[s] = {};
        state.stopTemplate[s] = -1;
        state.stopPitch[s] = 0.0f;
        state.recognised = a;

        const int64_t d = a - lookahead;
        const auto sd = slot (d);
        if (state.frameOf[sd] != d)
            continue;   // just after a reset

        auto decision = state.rule.step (state.frames[sd], state.frames[slot (d + 1)], f);
        using D = Rule::Decision;

        if (decision.kind == D::stop && decision.frame == 2 && decision.voicedStop)
        {
            // the "d" for the vowel after it: the template whose burst frame sounds most like that vowel's start
            double best = -2.0;
            for (int t = Rule::dA; t <= Rule::dI; ++t)
            {
                const double sim = similarity (templates[(size_t) t].data() + 3 * (size_t) channels, featuresOf (a));
                if (sim > best)
                {
                    best = sim;
                    state.currentTemplate = (int8_t) t;
                }
            }
        }
        if (decision.kind == D::stop || decision.kind == D::voiceBar)
        {
            state.stopTemplate[sd] = decision.voicedStop ? state.currentTemplate : (int8_t) Rule::t;
            if (decision.voicedStop)
            {
                // a "d" is voiced at the note after it once that is in view; before, at the note before it when the
                // voice ran into it, else not at all (after a pause the pause stays; round 25 A)
                float next = 0.0f;
                for (int64_t b = d + 1; b <= a && next <= 0.0f; ++b)
                    next = pitchAt ((int) (b - first));
                state.stopPitch[sd] = next > 0.0f ? next : (state.rule.legato ? state.lastPitch : 0.0f);
            }
        }
        if ((decision.kind == D::holdN || decision.kind == D::holdVowel) && decision.weight < 1.0f)
        {
            // what the hold sings: the "n" before the "d"; the vowel's last 2 frames before the "r" (the hold
            // starts on the r's 2nd frame)
            int64_t from = d - 3, to = d - 1;
            std::array<int64_t, 2> clean { -1, -1 };
            if (decision.kind == D::holdVowel)
            {
                // the vowel turns r-coloured before the "r" is heard: its 2 frames nearest the "r" with almost none
                // (round 25 A held the vowel's clean start; the frames just before the "r" sang one in male voices)
                int found = 0;
                for (int64_t b = d - 2; b >= d - 8 && b >= first && found < 2; --b)
                    if (const auto t = slot (b); state.frameOf[t] == b && state.frames[t].r < Rule::cleanVowelR
                                                  && state.frames[t].vowel >= Rule::vowelNeeded)
                        clean[(size_t) found++] = b;
                if (found == 2)
                {
                    from = clean[1];
                    to = clean[0] + 1;
                }
            }
            if (decision.kind == D::holdN)
            {
                // the most "n"-like of the 3 frames before
                float best = -1.0f;
                for (int64_t b = d - 1; b >= d - 3 && b >= first; --b)
                    if (const auto t = slot (b); state.frameOf[t] == b && state.frames[t].n > best)
                    {
                        best = state.frames[t].n;
                        from = b;
                    }
                to = from + 1;
            }
            if (from < first)
            {
                decision = {};   // not in this chunk: leave it alone
                state.rule.holdingVowel = false;
                state.rule.droppedD = state.rule.maxDroppedD;
            }
            else
            {
                state.currentHold ^= 1;
                auto& hold = state.holds[state.currentHold];
                hold.assign ((size_t) channels, 0.0f);
                const bool pair = clean[1] >= 0;   // the two clean vowel frames (not always next to each other)
                const int count = pair ? 2 : (int) (to - from);
                for (int64_t b = from; b < to; ++b)
                    if (! pair || b == clean[0] || b == clean[1])
                        for (int c = 0; c < channels; ++c)
                            hold[(size_t) c] += featuresOf (b)[c] / (float) count;
            }
        }
        state.holdIndex[sd] = state.currentHold;
        state.decisions[sd] = decision;
        if (const float p = d >= first ? pitchAt ((int) (d - first)) : 0.0f; p > 0.0f)
            state.lastPitch = p;
    }

    // the generator's frames: decided ones as decided; the last few go on as the last decided frame did
    const int64_t decided = state.recognised - lookahead;
    const auto lastDecided = slot (decided);
    const bool lastKnown = decided >= 0 && state.frameOf[lastDecided] == decided;

    for (int i = skip; i < numFrames; ++i)
    {
        const int e = i / 2;
        const int64_t a = first + e;
        Rule::Decision dec;
        int8_t tpl = -1;
        float pitch = 0.0f;
        uint8_t which = state.currentHold;

        if (a <= decided)
        {
            const auto s = slot (a);
            if (state.frameOf[s] != a)
                continue;
            dec = state.decisions[s];
            tpl = state.stopTemplate[s];
            pitch = state.stopPitch[s];
            which = state.holdIndex[s];
        }
        else if (lastKnown)
        {
            const auto& last = state.decisions[lastDecided];
            using D = Rule::Decision;
            if (last.kind == D::closure || last.kind == D::voiceBar || last.kind == D::holdN || last.kind == D::holdVowel)
            {
                dec = last;
                dec.weight = 1.0f;
                tpl = state.stopTemplate[lastDecided];
                pitch = state.stopPitch[lastDecided];
                which = state.holdIndex[lastDecided];
            }
        }

        using D = Rule::Decision;
        if (dec.kind == D::none)
            continue;

        float* dest = phone + (size_t) (i - skip) * (size_t) channels;
        switch (dec.kind)
        {
            case D::closure:
                std::copy (silence.begin(), silence.end(), dest);
                f0Frames[i] = 0.0f;
                break;
            case D::voiceBar:
                std::copy_n (templates[(size_t) Rule::dA].data(), channels, dest);
                f0Frames[i] = pitch;
                break;
            case D::stop:
            {
                const auto& t = templates[(size_t) (tpl >= 0 ? tpl : (dec.voicedStop ? Rule::dA : Rule::t))];
                std::copy_n (t.data() + (size_t) dec.frame * (size_t) channels, channels, dest);
                f0Frames[i] = dec.voicedStop ? (pitch > 0.0f ? pitch : f0Frames[i]) : 0.0f;
                break;
            }
            case D::holdN:
            case D::holdVowel:
            {
                const auto& hold = state.holds[which];
                if (hold.size() != (size_t) channels)
                    continue;
                for (int c = 0; c < channels; ++c)
                    dest[c] += dec.weight * (hold[(size_t) c] - dest[c]);
                if (dec.kind == D::holdN && f0Frames[i] <= 0.0f)
                    f0Frames[i] = state.lastPitch;   // the "n" runs on into the next word
                break;
            }
            case D::roll:
            case D::jh:
            {
                const auto which = dec.kind == D::roll ? Rule::roll : Rule::jh;
                if (dec.frame < getTemplateFrames (which))
                    std::copy_n (templates[(size_t) which].data() + (size_t) dec.frame * (size_t) channels, channels, dest);
                break;
            }
            case D::none:
                break;
        }
        if (changed != nullptr)
            changed[i - skip] = juce::jmax (changed[i - skip], dec.kind == D::holdN || dec.kind == D::holdVowel ? dec.weight : 1.0f);
    }
}

} // namespace voxa::neural
