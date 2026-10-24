#include "MusicTheory.h"

namespace voxa
{
namespace music
{

const juce::StringArray& getKeyNames()
{
    static const juce::StringArray names { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    return names;
}

const juce::StringArray& getScaleNames()
{
    static const juce::StringArray names { "Major", "Minor", "Chromatic", "Major Pentatonic", "Minor Pentatonic",
                                           "Blues", "Dorian", "Mixolydian", "Harmonic Minor", "Phrygian", "Lydian" };
    return names;
}

const juce::StringArray& getHarmonyIntervalNames()
{
    static const juce::StringArray names { "Off", "-Octave", "-6th", "-5th", "-4th", "-3rd", "Unison",
                                           "+3rd", "+4th", "+5th", "+6th", "+Octave" };
    return names;
}

const std::vector<int>& scaleSteps (ScaleType s)
{
    static const std::vector<int> table[] = {
        { 0, 2, 4, 5, 7, 9, 11 },                    // Major
        { 0, 2, 3, 5, 7, 8, 10 },                    // Natural minor
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 },    // Chromatic
        { 0, 2, 4, 7, 9 },                           // Major pentatonic
        { 0, 3, 5, 7, 10 },                          // Minor pentatonic
        { 0, 3, 5, 6, 7, 10 },                       // Blues
        { 0, 2, 3, 5, 7, 9, 10 },                    // Dorian
        { 0, 2, 4, 5, 7, 9, 10 },                    // Mixolydian
        { 0, 2, 3, 5, 7, 8, 11 },                    // Harmonic minor
        { 0, 1, 3, 5, 7, 8, 10 },                    // Phrygian
        { 0, 2, 4, 6, 7, 9, 11 },                    // Lydian
    };
    const auto index = juce::jlimit (0, (int) ScaleType::NumScales - 1, (int) s);
    return table[index];
}

static int pitchClass (int midiNote, int keyRoot) noexcept
{
    return ((midiNote - keyRoot) % 12 + 12) % 12;
}

bool isInScale (int midiNote, int keyRoot, ScaleType s)
{
    const auto pc = pitchClass (midiNote, keyRoot);
    for (auto step : scaleSteps (s))
        if (step == pc)
            return true;
    return false;
}

int nearestScaleNote (float midi, int keyRoot, ScaleType s)
{
    const int base = (int) std::floor (midi);
    int best = base;
    float bestDist = 1.0e9f;

    for (int candidate = base - 2; candidate <= base + 3; ++candidate)
    {
        if (! isInScale (candidate, keyRoot, s))
            continue;

        const float d = std::abs ((float) candidate - midi);
        if (d < bestDist - 1.0e-4f)
        {
            bestDist = d;
            best = candidate;
        }
    }
    return best;
}

int diatonicTranspose (int midiNote, int steps, int keyRoot, ScaleType s)
{
    const auto& sc = scaleSteps (s);
    const int n = (int) sc.size();
    const int snapped = nearestScaleNote ((float) midiNote, keyRoot, s);

    // Locate the degree of the snapped note.
    const int rel = snapped - keyRoot;
    const int octave = (int) std::floor ((float) rel / 12.0f);
    const int pc = rel - octave * 12;
    int degree = 0;
    for (int i = 0; i < n; ++i)
        if (sc[(size_t) i] == pc)
            degree = i;

    const int target = degree + steps;
    const int targetOctave = (int) std::floor ((float) target / (float) n);
    const int targetDegree = target - targetOctave * n;
    return keyRoot + (octave + targetOctave) * 12 + sc[(size_t) targetDegree];
}

int harmonyNote (int melodyNote, HarmonyInterval interval, int keyRoot, ScaleType s)
{
    if (interval == HarmonyInterval::Off)
        return -1;

    // Nominal sizes (semitones) and diatonic steps for 7-note scales.
    struct Info { float semis; int steps; };
    static const Info info[] = {
        { 0.0f, 0 },     // Off (unused)
        { -12.0f, -7 },  // -Octave
        { -8.5f, -5 },   // -6th
        { -7.0f, -4 },   // -5th
        { -5.0f, -3 },   // -4th
        { -3.5f, -2 },   // -3rd
        { 0.0f, 0 },     // Unison
        { 3.5f, 2 },     // +3rd
        { 5.0f, 3 },     // +4th
        { 7.0f, 4 },     // +5th
        { 8.5f, 5 },     // +6th
        { 12.0f, 7 },    // +Octave
    };

    const auto& in = info[(int) interval];

    if (interval == HarmonyInterval::OctaveDown || interval == HarmonyInterval::OctaveUp)
        return melodyNote + (int) in.semis;

    if (interval == HarmonyInterval::Unison)
        return melodyNote;

    const auto& sc = scaleSteps (s);

    if (sc.size() == 7)
        return diatonicTranspose (melodyNote, in.steps, keyRoot, s);

    if (s == ScaleType::Chromatic)
    {
        // Treat chromatic as major-flavoured fixed intervals.
        static const int fixedSemis[] = { 0, -12, -9, -7, -5, -4, 0, 4, 5, 7, 9, 12 };
        return melodyNote + fixedSemis[(int) interval];
    }

    // Pentatonic / blues: nearest in-scale note to the nominal interval, never landing on the melody note.
    const int snapped = nearestScaleNote ((float) melodyNote, keyRoot, s);
    int result = nearestScaleNote ((float) snapped + in.semis, keyRoot, s);
    if (result == snapped)
        result = diatonicTranspose (snapped, in.semis > 0 ? 1 : -1, keyRoot, s);
    return result;
}

juce::String noteName (int midiNote)
{
    const int octave = midiNote / 12 - 1;
    return getKeyNames()[((midiNote % 12) + 12) % 12] + juce::String (octave);
}

} // namespace music

//==============================================================================
void KeyDetector::reset() noexcept
{
    histogram.fill (0.0f);
    keyRoot = 0;
    scale = ScaleType::Major;
    confidence = 0.0f;
}

void KeyDetector::addPitch (float midi, float weight) noexcept
{
    if (weight <= 0.0f || ! std::isfinite (midi))
        return;

    // Spread the weight between the two nearest pitch classes.
    const float m = midi - 12.0f * std::floor (midi / 12.0f);
    const int lo = (int) std::floor (m) % 12;
    const int hi = (lo + 1) % 12;
    const float frac = m - std::floor (m);
    histogram[(size_t) lo] += weight * (1.0f - frac);
    histogram[(size_t) hi] += weight * frac;
}

void KeyDetector::decay (float factor) noexcept
{
    for (auto& h : histogram)
        h *= factor;
}

float KeyDetector::getTotalWeight() const noexcept
{
    float sum = 0.0f;
    for (auto h : histogram)
        sum += h;
    return sum;
}

void KeyDetector::update() noexcept
{
    static constexpr float majorProfile[12] = { 6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f, 2.52f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f };
    static constexpr float minorProfile[12] = { 6.33f, 2.68f, 3.52f, 5.38f, 2.60f, 3.53f, 2.54f, 4.75f, 3.98f, 2.69f, 3.34f, 3.17f };

    auto correlate = [this] (const float* profile, int root)
    {
        float meanH = 0.0f, meanP = 0.0f;
        for (int i = 0; i < 12; ++i) { meanH += histogram[(size_t) i]; meanP += profile[i]; }
        meanH /= 12.0f; meanP /= 12.0f;

        float num = 0.0f, dh = 0.0f, dp = 0.0f;
        for (int i = 0; i < 12; ++i)
        {
            const float h = histogram[(size_t) ((i + root) % 12)] - meanH;
            const float p = profile[i] - meanP;
            num += h * p; dh += h * h; dp += p * p;
        }
        return dh > 0.0f ? num / std::sqrt (dh * dp) : 0.0f;
    };

    float best = -2.0f, second = -2.0f;
    for (int root = 0; root < 12; ++root)
    {
        for (int mode = 0; mode < 2; ++mode)
        {
            const float c = correlate (mode == 0 ? majorProfile : minorProfile, root);
            if (c > best)
            {
                second = best;
                best = c;
                keyRoot = root;
                scale = mode == 0 ? ScaleType::Major : ScaleType::NaturalMinor;
            }
            else if (c > second)
            {
                second = c;
            }
        }
    }
    confidence = juce::jlimit (0.0f, 1.0f, (best - second) * 4.0f + juce::jmax (0.0f, best) * 0.5f);
}

} // namespace voxa
