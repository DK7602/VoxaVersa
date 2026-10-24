#include "Melody.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace voxa::sing::melody
{

namespace
{
    constexpr double gridUnit = 0.25;     // 16th note
    constexpr double leadInBeats = 4.0;   // the first phrase starts in bar 2 (room for a breath)
    constexpr double inf = std::numeric_limits<double>::infinity();

    struct Slot
    {
        int syllable = -1;
        bool melisma = false;
        int stress = 0;          // 2 strong (primary stress), 1 secondary, 0 weak
        bool wordEnd = true;
        bool pauseAfter = false;
    };

    using Phrase = std::vector<Slot>;

    /** Deterministic hash noise in [-1, 1]. */
    double noise (juce::uint32 seed, int a, int b, int c) noexcept
    {
        auto h = (juce::uint32) (seed * 0x9E3779B1u) ^ ((juce::uint32) a * 0x85EBCA77u)
               ^ ((juce::uint32) b * 0xC2B2AE3Du) ^ ((juce::uint32) c * 0x27D4EB2Fu);
        h ^= h >> 15; h *= 0x2C1B3C6Du;
        h ^= h >> 12; h *= 0x297A2D39u;
        h ^= h >> 15;
        return (double) (h & 0xFFFFFFu) / (double) 0x7FFFFF - 1.0;
    }

    double unitNoise (juce::uint32 seed, int a, int b, int c) noexcept
    {
        return 0.5 * (noise (seed, a, b, c) + 1.0);
    }

    double metricStrength (double beat) noexcept
    {
        double pos = std::fmod (beat, 4.0);
        if (pos < 0.0)
            pos += 4.0;

        auto at = [pos] (double x) { return std::abs (pos - x) < 1.0e-6; };
        if (at (0.0) || at (4.0)) return 1.0;
        if (at (2.0))             return 0.8;
        if (at (1.0) || at (3.0)) return 0.6;
        const double frac = pos - std::floor (pos);
        if (std::abs (frac - 0.5) < 1.0e-6) return 0.35;
        return 0.1;
    }

    bool isSixteenthPosition (double beat) noexcept
    {
        const double eighths = beat * 2.0;
        return std::abs (eighths - std::round (eighths)) > 1.0e-6;
    }

    std::vector<Phrase> extractPhrases (const LyricScript& script)
    {
        std::vector<Phrase> phrases;
        Phrase current;

        for (const auto& e : script.events)
        {
            switch (e.type)
            {
                case LyricEvent::Type::Syllable:
                {
                    const auto& syl = script.syllables[(size_t) e.syllable];
                    Slot s;
                    s.syllable = e.syllable;
                    s.stress = syl.stress == 1 ? 2 : (syl.stress == 2 ? 1 : 0);
                    s.wordEnd = syl.wordEnd;
                    s.pauseAfter = syl.pauseAfter;
                    current.push_back (s);
                    break;
                }
                case LyricEvent::Type::Melisma:
                {
                    if (current.empty())
                        break;

                    Slot s;
                    s.syllable = e.syllable;
                    s.melisma = true;
                    s.wordEnd = current.back().wordEnd;
                    s.pauseAfter = current.back().pauseAfter;
                    current.back().wordEnd = false;
                    current.back().pauseAfter = false;
                    current.push_back (s);
                    break;
                }
                case LyricEvent::Type::Breath:
                case LyricEvent::Type::PhraseEnd:
                    if (! current.empty())
                        phrases.push_back (std::move (current));
                    current.clear();
                    break;
            }
        }

        if (! current.empty())
            phrases.push_back (std::move (current));

        return phrases;
    }

    /** Splits lines that are too long for one musical phrase at word boundaries (commas preferred). */
    std::vector<Phrase> splitLongPhrases (const std::vector<Phrase>& phrases, int maxSlots)
    {
        std::vector<Phrase> out;
        for (const auto& phrase : phrases)
        {
            std::vector<Phrase> pending { phrase };
            while (! pending.empty())
            {
                auto p = std::move (pending.back());
                pending.pop_back();

                const int n = (int) p.size();
                if (n <= maxSlots)
                {
                    out.push_back (std::move (p));
                    continue;
                }

                int best = n / 2;
                double bestScore = -inf;
                for (int j = 1; j < n; ++j)
                {
                    if (p[(size_t) j].melisma || ! p[(size_t) j - 1].wordEnd)
                        continue;

                    const double balance = std::abs ((double) j - n * 0.5) / n;
                    const double score = (p[(size_t) j - 1].pauseAfter ? 2.0 : 0.0) - balance * 6.0;
                    if (score > bestScore)
                    {
                        bestScore = score;
                        best = j;
                    }
                }

                Phrase first (p.begin(), p.begin() + best), second (p.begin() + best, p.end());
                pending.push_back (std::move (second));   // processed after `first` (LIFO)
                pending.push_back (std::move (first));
            }
        }
        return out;
    }

    //==============================================================================
    struct RhythmParams
    {
        double busy = 0.5;
        double density = 1.4;       // syllables per beat
        double restMin = 1.0;       // breath at the end of each phrase, beats
        double finalMin = 1.0;      // shortest phrase-final note
        double finalMax = 3.0;      // longest phrase-final note
        double minIOI = 0.25;
        double maxIOI = 2.0;
    };

    RhythmParams makeRhythmParams (const VoiceTraits& traits)
    {
        RhythmParams r;
        r.busy = juce::jlimit (0.0, 1.0, (double) traits.singRhythmBusy);
        r.density = 0.7 * std::pow (4.0, r.busy);
        r.restMin = r.busy < 0.5 ? 1.0 : 0.5;
        r.finalMin = juce::jmap (r.busy, 1.5, 0.75);
        r.finalMax = juce::jmap (r.busy, 4.0, 2.0);
        r.minIOI = r.busy < 0.35 ? 0.5 : 0.25;
        r.maxIOI = juce::jmap (r.busy, 3.0, 1.0);
        return r;
    }

    double beatsNeeded (int numSlots, const RhythmParams& r)
    {
        return (numSlots - 1) / r.density + r.finalMin + r.restMin;
    }

    double beatsMinimum (int numSlots, const RhythmParams& r)
    {
        return (numSlots - 1) * r.minIOI + r.finalMin * 0.5 + r.restMin;
    }

    struct RhythmContext
    {
        const Phrase* phrase = nullptr;
        const RhythmParams* params = nullptr;
        double blockStart = 0.0;
        double sungEnd = 0.0;
        juce::uint32 seed = 1;
        int phraseIndex = 0;
    };

    double emissionScore (const RhythmContext& ctx, int i, double pos)
    {
        const auto& slot = (*ctx.phrase)[(size_t) i];
        const auto& r = *ctx.params;
        const double str = metricStrength (pos);

        double e = 0.0;
        if (slot.stress == 2)
            e += 2.0 * str;
        else if (slot.stress == 1)
            e += 0.8 * str;
        else
            e -= 1.2 * std::max (0.0, str - 0.35);

        if (isSixteenthPosition (pos))
            e -= (1.0 - r.busy) * 1.6 + 0.15;

        if (i == 0)
        {
            const double rel = pos - ctx.blockStart;
            if (rel < 0.0)
                e += slot.stress == 2 ? -2.5 : 0.3;           // pickups only for weak syllables
            else
                e -= 0.7 * std::max (0.0, rel - 0.5);         // don't start the phrase late
        }

        e += 0.55 * noise (ctx.seed, ctx.phraseIndex * 131 + i, (int) std::lround (pos * 4.0), 17);
        return e;
    }

    double transitionScore (const RhythmContext& ctx, int i, double ioi)
    {
        const auto& prev = (*ctx.phrase)[(size_t) i - 1];
        const auto& r = *ctx.params;

        double target = 1.0 / r.density;
        if (prev.stress == 2)       target *= 1.35;
        else if (prev.stress == 0)  target *= 0.8;
        if (! prev.wordEnd)         target *= 0.9;
        if (prev.pauseAfter)        target *= 1.9;

        const double maxIOI = r.maxIOI + (prev.pauseAfter ? 1.0 : 0.0);
        if (ioi > maxIOI + 1.0e-9 || ioi < r.minIOI - 1.0e-9)
            return -inf;

        const double l = std::log2 (ioi / target);
        return -1.2 * l * l;
    }

    double finalScore (const RhythmContext& ctx, double pos)
    {
        const auto& r = *ctx.params;
        const double duration = ctx.sungEnd - pos;
        double s = 1.6 * metricStrength (pos);
        s -= 0.5 * std::max (0.0, duration - r.finalMax);
        if (duration >= 1.5)
            s += 0.4;
        return s;
    }

    double evaluateRhythm (const RhythmContext& ctx, const std::vector<double>& positions)
    {
        double total = 0.0;
        for (size_t i = 0; i < positions.size(); ++i)
        {
            total += emissionScore (ctx, (int) i, positions[i]);
            if (i > 0)
                total += transitionScore (ctx, (int) i, positions[i] - positions[i - 1]);
        }
        if (! positions.empty())
            total += finalScore (ctx, positions.back());
        return total;
    }

    /** Places the phrase's syllables on a 16th grid: stressed syllables on strong beats, note lengths
        around the genre's density, a long phrase-final note. Dynamic programming over grid positions. */
    std::vector<double> composeRhythm (const RhythmContext& ctx, double earliestStart, double& bestScoreOut)
    {
        const auto& phrase = *ctx.phrase;
        const auto& r = *ctx.params;
        const int n = (int) phrase.size();

        const double firstPos = std::ceil (std::max (ctx.blockStart - 1.0, earliestStart) / gridUnit - 1.0e-9) * gridUnit;
        const double lastOnset = ctx.sungEnd - r.finalMin * 0.5;
        const int numPositions = (int) std::floor ((lastOnset - firstPos) / gridUnit + 1.0e-9) + 1;

        std::vector<double> positions;
        if (numPositions < n || n == 0)
        {
            // Not enough room (shouldn't happen - the block is sized for the phrase): spread evenly.
            const double span = std::max (gridUnit * n, ctx.sungEnd - ctx.blockStart - r.finalMin);
            for (int i = 0; i < n; ++i)
                positions.push_back (ctx.blockStart + std::floor (span * i / n / gridUnit) * gridUnit);
            bestScoreOut = -inf;
            return positions;
        }

        auto posAt = [&] (int g) { return firstPos + g * gridUnit; };
        const int maxStep = (int) std::ceil ((r.maxIOI + 1.0) / gridUnit) + 1;

        std::vector<double> score ((size_t) (n * numPositions), -inf);
        std::vector<int> back ((size_t) (n * numPositions), -1);
        auto at = [numPositions] (int i, int g) { return (size_t) (i * numPositions + g); };

        for (int g = 0; g < numPositions; ++g)
            score[at (0, g)] = emissionScore (ctx, 0, posAt (g));

        for (int i = 1; i < n; ++i)
        {
            for (int g = i; g < numPositions; ++g)
            {
                const double pos = posAt (g);
                double best = -inf;
                int bestPrev = -1;
                for (int gp = std::max (0, g - maxStep); gp < g; ++gp)
                {
                    const double prevScore = score[at (i - 1, gp)];
                    if (std::isinf (prevScore))
                        continue;

                    const double t = transitionScore (ctx, i, pos - posAt (gp));
                    if (std::isinf (t))
                        continue;

                    if (prevScore + t > best)
                    {
                        best = prevScore + t;
                        bestPrev = gp;
                    }
                }

                if (bestPrev >= 0)
                {
                    score[at (i, g)] = best + emissionScore (ctx, i, pos);
                    back[at (i, g)] = bestPrev;
                }
            }
        }

        int bestEnd = -1;
        double bestTotal = -inf;
        for (int g = 0; g < numPositions; ++g)
        {
            const double s = score[at (n - 1, g)];
            if (std::isinf (s))
                continue;

            const double total = s + finalScore (ctx, posAt (g));
            if (total > bestTotal)
            {
                bestTotal = total;
                bestEnd = g;
            }
        }

        if (bestEnd < 0)
        {
            const double span = ctx.sungEnd - r.finalMin - std::max (firstPos, ctx.blockStart);
            for (int i = 0; i < n; ++i)
                positions.push_back (std::max (firstPos, ctx.blockStart)
                                     + std::floor (span * i / std::max (1, n) / gridUnit) * gridUnit);
            bestScoreOut = -inf;
            return positions;
        }

        positions.resize ((size_t) n);
        for (int i = n - 1, g = bestEnd; i >= 0; --i)
        {
            positions[(size_t) i] = posAt (g);
            g = back[at (i, g)];
        }

        bestScoreOut = bestTotal;
        return positions;
    }

    //==============================================================================
    struct PitchSpace
    {
        int keyRoot = 0;
        std::vector<int> steps;      // scale steps used for composition
        int tonicMidi = 60;
        int lowMidi = 48, highMidi = 72;
        std::vector<int> degrees;    // allowed scale degrees (relative to tonicMidi), ascending
        int thirdIndex = 2, fifthIndex = 4;

        int numSteps() const noexcept   { return (int) steps.size(); }

        int degreeClass (int degree) const noexcept
        {
            const int n = numSteps();
            return ((degree % n) + n) % n;
        }

        int midiOf (int degree) const noexcept
        {
            const int n = numSteps();
            const int octave = (int) std::floor ((double) degree / n);
            return tonicMidi + octave * 12 + steps[(size_t) (degree - octave * n)];
        }

        bool isStable (int degree) const noexcept
        {
            const int c = degreeClass (degree);
            return c == 0 || c == thirdIndex || c == fifthIndex;
        }
    };

    PitchSpace makePitchSpace (const VoiceTraits& traits, int keyRoot, ScaleType scale)
    {
        PitchSpace ps;
        ps.keyRoot = keyRoot;

        // A chromatic "scale" has no tonal centre - compose in major (every note is in key anyway).
        const auto compositionScale = scale == ScaleType::Chromatic ? ScaleType::Major : scale;
        ps.steps = music::scaleSteps (compositionScale);

        int low = traits.rangeLowMidi, high = traits.rangeHighMidi;
        if (high < low)
            std::swap (low, high);
        if (high - low < 9)
        {
            const int centre = (low + high) / 2;
            low = centre - 6;
            high = centre + 6;
        }
        // Keep a small safety margin at the extremes of wide ranges.
        if (high - low >= 16)
        {
            ++low;
            --high;
        }
        ps.lowMidi = low;
        ps.highMidi = high;

        const int centre = (low + high) / 2;
        int bestTonic = keyRoot;
        int bestDistance = 1 << 30;
        for (int t = keyRoot; t < 128; t += 12)
        {
            const int d = std::abs (t + 4 - centre);
            if (d < bestDistance)
            {
                bestDistance = d;
                bestTonic = t;
            }
        }
        ps.tonicMidi = bestTonic;

        const int n = ps.numSteps();
        for (int i = 0; i < n; ++i)
        {
            if (ps.steps[(size_t) i] == 3 || ps.steps[(size_t) i] == 4)
                ps.thirdIndex = i;
            if (ps.steps[(size_t) i] == 7)
                ps.fifthIndex = i;
        }

        for (int d = -4 * n; d <= 4 * n; ++d)
        {
            const int m = ps.midiOf (d);
            if (m >= low && m <= high)
                ps.degrees.push_back (d);
        }
        return ps;
    }

    /** A four-bar chord loop (scale-degree roots) implied under the melody; empty for scales that
        are not heptatonic (pentatonic / blues melodies just lean on the tonic triad). */
    std::vector<int> chooseProgression (const PitchSpace& ps, juce::uint32 seed)
    {
        if (ps.numSteps() != 7)
            return {};

        const bool minor = ps.steps[2] == 3;
        static const int majorLoops[][4] = { { 0, 4, 5, 3 }, { 0, 5, 3, 4 }, { 0, 3, 0, 4 }, { 5, 3, 0, 4 }, { 0, 3, 5, 4 } };
        static const int minorLoops[][4] = { { 0, 5, 2, 6 }, { 0, 3, 4, 0 }, { 0, 6, 5, 6 }, { 0, 5, 3, 4 }, { 0, 3, 6, 2 } };

        const auto& loops = minor ? minorLoops : majorLoops;
        const int choice = juce::jlimit (0, 4, (int) (unitNoise (seed, 3, 7, 11) * 5.0));
        return { loops[choice][0], loops[choice][1], loops[choice][2], loops[choice][3] };
    }

    bool isChordTone (int degreeClass, int chordRoot) noexcept
    {
        return degreeClass == chordRoot || degreeClass == (chordRoot + 2) % 7 || degreeClass == (chordRoot + 4) % 7;
    }

    struct PhrasePlan
    {
        double startDegree = 0.0;
        double peakDegree = 4.0;
        double peakPosition = 0.5;
        int endDegree = 0;           // exact cadence degree
        bool flat = false;           // rap: reciting tone
        bool descending = false;

        double contour (double x) const noexcept
        {
            if (flat)
            {
                if (x < 0.75)
                    return startDegree;
                const double t = (x - 0.75) / 0.25;
                return startDegree + (endDegree - startDegree) * t * t;
            }

            if (descending)
                return peakDegree + (endDegree - peakDegree) * x;

            if (x < peakPosition)
            {
                const double t = x / std::max (0.05, peakPosition);
                return startDegree + (peakDegree - startDegree) * std::sin (t * juce::MathConstants<double>::halfPi);
            }

            const double t = (x - peakPosition) / std::max (0.05, 1.0 - peakPosition);
            const double s = t * t * (3.0 - 2.0 * t);
            return peakDegree + (endDegree - peakDegree) * s;
        }
    };

    int nearestDegreeWithClass (const PitchSpace& ps, int targetClass, double near)
    {
        int best = ps.degrees.empty() ? 0 : ps.degrees.front();
        double bestDistance = inf;
        for (auto d : ps.degrees)
        {
            if (ps.degreeClass (d) != targetClass)
                continue;
            const double dist = std::abs (d - near);
            if (dist < bestDistance)
            {
                bestDistance = dist;
                best = d;
            }
        }
        return best;
    }

    double semitoneCost (int semis, double busy, bool melismaSlot)
    {
        const int a = std::abs (semis);
        if (a == 0)
            return melismaSlot ? 6.0 : juce::jmap (busy * busy, 1.2, -0.6);

        double c;
        if (a <= 2)       c = 0.0;
        else if (a <= 4)  c = 0.7;
        else if (a == 5)  c = 1.4;
        else if (a == 6)  c = 4.0;     // tritone
        else if (a == 7)  c = 2.0;
        else if (a <= 9)  c = 3.0;
        else if (a == 12) c = 3.5;
        else if (a <= 11) c = 5.0;
        else              c = 14.0;

        if (melismaSlot && a > 2)
            c += 1.5;      // runs move stepwise
        return c;
    }

    /** Second-order Viterbi over scale degrees: follows the phrase contour, prefers steps, recovers
        from leaps, puts chord tones on strong beats and ends on the planned cadence degree. */
    std::vector<int> composePitches (const Phrase& phrase, const std::vector<double>& positions, double sungEnd,
                                     const PitchSpace& ps, const PhrasePlan& plan, double busy,
                                     int previousEndMidi, const std::vector<int>* motif,
                                     const std::vector<int>& chordRoots, juce::uint32 seed, int phraseIndex)
    {
        const int n = (int) phrase.size();
        const int numDegrees = (int) ps.degrees.size();
        std::vector<int> result ((size_t) n, ps.degrees.empty() ? 0 : ps.degrees[(size_t) numDegrees / 2]);
        if (n == 0 || numDegrees == 0)
            return result;

        const double start = positions.front();
        juce::ignoreUnused (sungEnd);
        const double span = std::max (0.25, positions.back() - start);   // the last note sits at x = 1
        const int endClass = ps.degreeClass (plan.endDegree);

        std::vector<int> midi ((size_t) numDegrees);
        for (int k = 0; k < numDegrees; ++k)
            midi[(size_t) k] = ps.midiOf (ps.degrees[(size_t) k]);

        auto emission = [&] (int i, int k)
        {
            const int degree = ps.degrees[(size_t) k];
            const double x = (positions[(size_t) i] - start) / span;
            const double target = plan.contour (x);
            const double w = (plan.flat ? 0.6 : 0.22) * (x > 0.7 ? 2.0 : 1.0);
            double e = w * (degree - target) * (degree - target);

            // Implied harmony: chord tones on the strong beats.
            const double str = metricStrength (positions[(size_t) i]);
            const int chordRoot = i < (int) chordRoots.size() ? chordRoots[(size_t) i] : -1;
            if (chordRoot >= 0)
            {
                const bool chordTone = isChordTone (ps.degreeClass (degree), chordRoot);
                if (str >= 0.6)
                    e += chordTone ? -0.3 * str : 1.1 * str;
                else if (str >= 0.35 && ! chordTone)
                    e += 0.35;
            }
            else if (str >= 0.6 && ! ps.isStable (degree))
            {
                e += 0.9 * str;
            }

            if (i == n - 1)
            {
                if (ps.degreeClass (degree) != endClass)
                    e += 50.0;
                e += 0.6 * std::abs (degree - plan.endDegree);
            }

            if (i == 0 && previousEndMidi > 0)
                e += 0.12 * std::abs (midi[(size_t) k] - previousEndMidi);

            if (motif != nullptr && i < (int) motif->size() && i < (int) std::ceil (n * 0.6) && (*motif)[(size_t) i] == degree)
                e -= 0.9;

            e += 0.5 * noise (seed, phraseIndex * 977 + i, degree, 91);
            return e;
        };

        auto transition = [&] (int i, int kPrevPrev, int kPrev, int k)
        {
            const bool melismaSlot = phrase[(size_t) i].melisma;
            const int step = midi[(size_t) k] - midi[(size_t) kPrev];
            double c = semitoneCost (step, busy, melismaSlot);

            // A melodic leap onto a stressed syllable sounds intentional; endless scale runs don't.
            if (phrase[(size_t) i].stress == 2 && std::abs (step) >= 3 && std::abs (step) <= 5)
                c -= 0.35;

            if (kPrevPrev >= 0)
            {
                const int prevStep = midi[(size_t) kPrev] - midi[(size_t) kPrevPrev];
                if (step != 0 && prevStep != 0 && std::abs (step) <= 2 && std::abs (prevStep) <= 2 && (step > 0) == (prevStep > 0))
                    c += 0.25;
                if (std::abs (prevStep) >= 5)
                {
                    if (step != 0 && (step > 0) == (prevStep > 0))
                        c += 1.5;                                  // keep going after a leap: avoid
                    else if (step != 0 && std::abs (step) <= 4)
                        c -= 0.4;                                  // step back: leap recovery
                }
                if (step == 0 && prevStep == 0)
                    c += (1.0 - busy) * 1.5;
            }

            if (i == n - 1 && std::abs (step) > 5)
                c += 1.0;
            return c;
        };

        // V[i][prev * D + cur]
        const int numStates = numDegrees * numDegrees;
        std::vector<double> cost ((size_t) numStates, inf), next ((size_t) numStates, inf);
        std::vector<std::vector<int>> backPointers ((size_t) n);

        // Slot 0: the "previous" index is a dummy equal to the current one.
        std::vector<double> first ((size_t) numDegrees);
        for (int k = 0; k < numDegrees; ++k)
            first[(size_t) k] = emission (0, k);

        if (n == 1)
        {
            int best = 0;
            for (int k = 1; k < numDegrees; ++k)
                if (first[(size_t) k] < first[(size_t) best])
                    best = k;
            result[0] = ps.degrees[(size_t) best];
            return result;
        }

        for (int a = 0; a < numDegrees; ++a)
            for (int b = 0; b < numDegrees; ++b)
                cost[(size_t) (a * numDegrees + b)] = first[(size_t) a] + transition (1, -1, a, b) + emission (1, b);

        for (int i = 2; i < n; ++i)
        {
            std::fill (next.begin(), next.end(), inf);
            auto& bp = backPointers[(size_t) i];
            bp.assign ((size_t) numStates, -1);

            for (int b = 0; b < numDegrees; ++b)
            {
                for (int c = 0; c < numDegrees; ++c)
                {
                    if (std::abs (midi[(size_t) c] - midi[(size_t) b]) > 12)
                        continue;

                    double best = inf;
                    int bestA = -1;
                    for (int a = 0; a < numDegrees; ++a)
                    {
                        const double prev = cost[(size_t) (a * numDegrees + b)];
                        if (std::isinf (prev))
                            continue;
                        const double v = prev + transition (i, a, b, c);
                        if (v < best)
                        {
                            best = v;
                            bestA = a;
                        }
                    }

                    if (bestA >= 0)
                    {
                        next[(size_t) (b * numDegrees + c)] = best + emission (i, c);
                        bp[(size_t) (b * numDegrees + c)] = bestA;
                    }
                }
            }
            std::swap (cost, next);
        }

        int bestState = 0;
        for (int s = 1; s < numStates; ++s)
            if (cost[(size_t) s] < cost[(size_t) bestState])
                bestState = s;

        std::vector<int> indices ((size_t) n);
        int b = bestState / numDegrees, c = bestState % numDegrees;
        indices[(size_t) n - 1] = c;
        indices[(size_t) n - 2] = b;
        for (int i = n - 1; i >= 2; --i)
        {
            const int a = backPointers[(size_t) i][(size_t) (b * numDegrees + c)];
            indices[(size_t) i - 2] = a < 0 ? b : a;
            c = b;
            b = a < 0 ? b : a;
        }

        for (int i = 0; i < n; ++i)
            result[(size_t) i] = ps.degrees[(size_t) indices[(size_t) i]];
        return result;
    }
}

//==============================================================================
int chooseKeyRoot (const VoiceTraits& traits)
{
    const int target = traits.centreMidi() - 4;
    static const int friendly[] = { 0, 2, 3, 4, 5, 7, 9, 10 };   // C D Eb E F G A Bb

    int bestRoot = 0, bestDistance = 1 << 30;
    for (auto pc : friendly)
    {
        // distance from the target to the nearest note with this pitch class
        int d = ((target - pc) % 12 + 12) % 12;
        d = std::min (d, 12 - d);
        if (d < bestDistance)
        {
            bestDistance = d;
            bestRoot = pc;
        }
    }
    return bestRoot;
}

Score compose (const LyricScript& script, const VoiceTraits& traits, int keyRoot, ScaleType scale, juce::uint32 seed)
{
    Score score;
    score.keyRoot = keyRoot >= 0 ? keyRoot % 12 : chooseKeyRoot (traits);
    score.scale = scale;

    const auto rp = makeRhythmParams (traits);
    const int maxSlots = std::max (4, (int) std::floor (rp.density * 13.0));
    const auto phrases = splitLongPhrases (extractPhrases (script), maxSlots);
    if (phrases.empty())
        return score;

    const auto ps = makePitchSpace (traits, score.keyRoot, scale);
    const auto progression = chooseProgression (ps, seed);
    double homeDegree = 0.0;

    // A common phrase length (2 or 4 bars) keeps the song's structure regular.
    std::vector<double> needs;
    for (const auto& p : phrases)
        needs.push_back (beatsNeeded ((int) p.size(), rp));
    auto sortedNeeds = needs;
    std::sort (sortedNeeds.begin(), sortedNeeds.end());
    const double medianNeed = sortedNeeds[sortedNeeds.size() / 2];
    const int standardBars = medianNeed <= 8.0 ? 2 : 4;

    const int numPhrases = (int) phrases.size();
    double blockStart = leadInBeats;
    double previousEnd = leadInBeats - 1.0;
    int previousEndMidi = 0;
    int previousEndDegree = 0;

    struct Memory { int numSlots; double blockBeats; std::vector<double> relative; std::vector<int> degrees; };
    std::vector<Memory> memory;

    for (int k = 0; k < numPhrases; ++k)
    {
        const auto& phrase = phrases[(size_t) k];
        const int n = (int) phrase.size();

        int bars = standardBars;
        while (bars < 64 && (needs[(size_t) k] > bars * 4.0 + 0.5 || beatsMinimum (n, rp) > bars * 4.0))
            bars *= 2;
        const double blockBeats = bars * 4.0;

        RhythmContext ctx;
        ctx.phrase = &phrase;
        ctx.params = &rp;
        ctx.blockStart = blockStart;
        ctx.sungEnd = blockStart + blockBeats - rp.restMin;
        ctx.seed = seed;
        ctx.phraseIndex = k;

        const double earliest = previousEnd + (k == 0 ? 0.0 : 0.5);
        double dpScore = -inf;
        auto positions = composeRhythm (ctx, earliest, dpScore);

        // Re-use the rhythm (and opening melody) of an earlier phrase of the same shape: songs repeat.
        const Memory* motifSource = nullptr;
        for (auto it = memory.rbegin(); it != memory.rend(); ++it)
        {
            if (it->numSlots != n || std::abs (it->blockBeats - blockBeats) > 1.0e-6)
                continue;

            std::vector<double> candidate;
            bool valid = true;
            for (auto rel : it->relative)
            {
                const double pos = blockStart + rel;
                valid = valid && pos >= earliest - 1.0e-9;
                candidate.push_back (pos);
            }
            if (! valid)
                continue;

            const double s = evaluateRhythm (ctx, candidate);
            if (s > dpScore - 2.0 && unitNoise (seed, k, 5, 5) < 0.75)
            {
                positions = candidate;
                motifSource = &*it;
            }
            break;
        }

        // Phrase plan: cadence, contour and range.
        const bool lastPhrase = k == numPhrases - 1;
        int endClass;
        if (lastPhrase)
            endClass = 0;
        else if (k % 2 == 1)
            endClass = unitNoise (seed, k, 1, 1) < 0.65 ? 0 : ps.thirdIndex;
        else
            endClass = unitNoise (seed, k, 1, 2) < 0.6 ? ps.fifthIndex : ps.thirdIndex;

        PhrasePlan plan;
        plan.flat = rp.busy >= 0.75;
        const int n7 = ps.numSteps();

        // Range of degrees around the middle of the voice.
        const int lowDegree = ps.degrees.empty() ? 0 : ps.degrees.front();

        if (k == 0)
        {
            const double r = unitNoise (seed, k, 2, 0);
            const int startClass = r < 0.4 ? 0 : (r < 0.7 ? ps.thirdIndex : ps.fifthIndex);
            plan.startDegree = nearestDegreeWithClass (ps, startClass, (double) (startClass == ps.fifthIndex ? -n7 + ps.fifthIndex : startClass));
            if (plan.startDegree < lowDegree + 1)
                plan.startDegree = nearestDegreeWithClass (ps, startClass, lowDegree + n7 * 0.5);
        }
        else
        {
            // Continue near the previous phrase but keep coming home to the first phrase's register.
            plan.startDegree = std::round (0.5 * (previousEndDegree + homeDegree) + noise (seed, k, 2, 1) * 1.2 + 0.3);
        }

        double span = juce::jmap (rp.busy, 4.5, 1.5) + noise (seed, k, 3, 0);
        span = std::max (1.0, span);

        // Keep the tessitura in the middle of the voice; only the climax phrase reaches higher.
        const bool climax = numPhrases >= 3 && k == (numPhrases * 2) / 3;
        const double centreMidi = 0.5 * (ps.lowMidi + ps.highMidi);
        const double peakLimitMidi = centreMidi + (climax ? 0.85 : 0.6) * (ps.highMidi - centreMidi);
        double peakLimit = lowDegree;
        for (auto d : ps.degrees)
            if (ps.midiOf (d) <= peakLimitMidi)
                peakLimit = d;

        plan.startDegree = juce::jlimit ((double) lowDegree, juce::jmax ((double) lowDegree, peakLimit - 1.0), plan.startDegree);
        plan.peakDegree = juce::jlimit ((double) lowDegree, peakLimit, plan.startDegree + std::round (span));
        plan.peakPosition = 0.35 + 0.35 * unitNoise (seed, k, 4, 0);
        plan.descending = ! plan.flat && unitNoise (seed, k, 4, 1) < 0.2;
        plan.endDegree = nearestDegreeWithClass (ps, endClass, plan.startDegree - (plan.flat ? 0.0 : 1.0));
        if (lastPhrase && ! plan.flat)
        {
            // The song ends by coming down to the tonic, not by jumping up to it.
            int best = plan.endDegree;
            for (auto d : ps.degrees)
                if (ps.degreeClass (d) == 0 && d <= plan.startDegree + 1.0 && d >= lowDegree)
                    best = d;
            plan.endDegree = best;
        }

        if (k == 0)
            homeDegree = plan.startDegree;

        // Chord per bar from the loop; the phrase's last bar carries its cadence (I, or V for a half cadence).
        std::vector<int> chordRoots ((size_t) n, -1);
        if (! progression.empty())
        {
            const int lastBar = (int) std::floor ((positions.back() - leadInBeats) / 4.0 + 1.0e-9);
            const int cadenceChord = (endClass == ps.fifthIndex && ! lastPhrase) ? 4 : 0;
            for (int i = 0; i < n; ++i)
            {
                const int bar = (int) std::floor ((positions[(size_t) i] - leadInBeats) / 4.0 + 1.0e-9);
                chordRoots[(size_t) i] = bar == lastBar ? cadenceChord : progression[(size_t) (((bar % 4) + 4) % 4)];
            }
        }

        const auto degrees = composePitches (phrase, positions, ctx.sungEnd, ps, plan, rp.busy, previousEndMidi,
                                             motifSource != nullptr ? &motifSource->degrees : nullptr, chordRoots, seed, k);

        // Emit notes.
        for (int i = 0; i < n; ++i)
        {
            const auto& slot = phrase[(size_t) i];
            const double pos = positions[(size_t) i];

            ScoreNote note;
            note.startBeat = pos;
            if (i + 1 < n)
            {
                const double ioi = positions[(size_t) i + 1] - pos;
                note.lengthBeats = (slot.pauseAfter && ioi >= 1.0) ? ioi - 0.5 : ioi;
            }
            else
            {
                note.lengthBeats = std::max (gridUnit, std::min (ctx.sungEnd - pos, rp.finalMax));
            }

            note.midiNote = ps.midiOf (degrees[(size_t) i]);
            const double x = (pos - positions.front()) / std::max (0.25, ctx.sungEnd - positions.front());
            note.velocity = (float) juce::jlimit (0.45, 1.0, 0.68 + (slot.stress == 2 ? 0.12 : (slot.stress == 1 ? 0.06 : 0.0))
                                                            + 0.08 * metricStrength (pos)
                                                            + 0.06 * std::sin (juce::MathConstants<double>::pi * x));
            note.syllable = slot.syllable;
            note.continuation = slot.melisma;
            note.breathBefore = (i == 0);
            score.notes.push_back (note);
        }

        Memory m;
        m.numSlots = n;
        m.blockBeats = blockBeats;
        for (auto p : positions)
            m.relative.push_back (p - blockStart);
        m.degrees = degrees;
        memory.push_back (std::move (m));

        previousEnd = score.notes.back().endBeat();
        previousEndMidi = score.notes.back().midiNote;
        previousEndDegree = degrees.back();
        blockStart += blockBeats;
    }

    return score;
}

//==============================================================================
std::vector<Note> makeMonophonic (std::vector<Note> notes)
{
    notes.erase (std::remove_if (notes.begin(), notes.end(), [] (const Note& n)
                 {
                     return ! std::isfinite (n.startBeat) || ! std::isfinite (n.lengthBeats) || n.lengthBeats <= 1.0e-6;
                 }), notes.end());

    std::stable_sort (notes.begin(), notes.end(), [] (const Note& a, const Note& b)
    {
        if (std::abs (a.startBeat - b.startBeat) > 1.0 / 32.0)
            return a.startBeat < b.startBeat;
        return a.midiNote > b.midiNote;   // chords: the top note first
    });

    std::vector<Note> result;
    for (const auto& n : notes)
    {
        if (! result.empty())
        {
            auto& prev = result.back();
            if (std::abs (n.startBeat - prev.startBeat) <= 1.0 / 32.0)
                continue;   // a lower chord tone

            if (prev.startBeat + prev.lengthBeats > n.startBeat)
                prev.lengthBeats = n.startBeat - prev.startBeat;
        }
        result.push_back (n);
    }

    result.erase (std::remove_if (result.begin(), result.end(), [] (const Note& n) { return n.lengthBeats < 1.0 / 64.0; }),
                  result.end());
    for (auto& n : result)
    {
        n.midiNote = juce::jlimit (0, 127, n.midiNote);
        n.velocity = juce::jlimit (0.05f, 1.0f, n.velocity);
    }
    return result;
}

int bestOctaveShift (const std::vector<Note>& notes, const VoiceTraits& traits)
{
    if (notes.empty())
        return 0;

    const int low = std::min (traits.rangeLowMidi, traits.rangeHighMidi);
    const int high = std::max (traits.rangeLowMidi, traits.rangeHighMidi);
    const double centre = 0.5 * (low + high);

    double totalLength = 0.0, weightedSum = 0.0;
    for (const auto& n : notes)
    {
        totalLength += n.lengthBeats;
        weightedSum += n.lengthBeats * n.midiNote;
    }
    const double mean = weightedSum / std::max (1.0e-9, totalLength);

    int bestShift = 0;
    double bestCost = inf;
    for (int octave = -5; octave <= 5; ++octave)
    {
        const int shift = octave * 12;
        double outside = 0.0;
        for (const auto& n : notes)
        {
            const int m = n.midiNote + shift;
            if (m < 0 || m > 127)
                outside += 100.0 * n.lengthBeats;
            outside += n.lengthBeats * (std::max (0, low - m) + std::max (0, m - high));
        }
        const double cost = outside / std::max (1.0e-9, totalLength) + 0.05 * std::abs (mean + shift - centre)
                          + 0.001 * std::abs (octave);
        if (cost < bestCost)
        {
            bestCost = cost;
            bestShift = shift;
        }
    }
    return bestShift;
}

void detectKey (const std::vector<Note>& notes, int& keyRoot, ScaleType& scale)
{
    KeyDetector detector;
    detector.reset();
    for (const auto& n : notes)
        detector.addPitch ((float) n.midiNote, (float) std::max (0.05, n.lengthBeats));
    detector.update();
    keyRoot = detector.getKeyRoot();
    scale = detector.getScale();
}

Score fitProvided (const LyricScript& script, const std::vector<Note>& melodyIn, const VoiceTraits& traits,
                   int keyRoot, ScaleType scale, double bpm)
{
    Score score;
    auto mel = makeMonophonic (melodyIn);
    if (mel.empty())
        return score;

    const int shift = bestOctaveShift (mel, traits);
    for (auto& n : mel)
        n.midiNote = juce::jlimit (0, 127, n.midiNote + shift);

    if (keyRoot >= 0)
    {
        score.keyRoot = keyRoot % 12;
        score.scale = scale;
    }
    else
    {
        detectKey (mel, score.keyRoot, score.scale);
    }

    int slotsNeeded = 0;
    {
        bool haveSyllable = false;
        for (const auto& e : script.events)
        {
            if (e.type == LyricEvent::Type::Syllable)       { ++slotsNeeded; haveSyllable = true; }
            else if (e.type == LyricEvent::Type::Melisma)   { if (haveSyllable) ++slotsNeeded; }
            else                                            haveSyllable = false;
        }
    }

    // More syllables than notes: repeat the melody (whole bars) until everything can be sung.
    if (slotsNeeded > (int) mel.size())
    {
        const double barStart = std::floor (mel.front().startBeat / 4.0) * 4.0;
        double lastEnd = 0.0;
        for (const auto& n : mel)
            lastEnd = std::max (lastEnd, n.startBeat + n.lengthBeats);

        const double period = std::max (4.0, std::ceil ((lastEnd - barStart) / 4.0 - 1.0e-9) * 4.0);
        const auto pattern = mel;
        for (int rep = 1; (int) mel.size() < slotsNeeded && rep < 10000; ++rep)
        {
            for (auto n : pattern)
            {
                n.startBeat += period * rep;
                mel.push_back (n);
            }
        }
    }

    const double beatsPerSecond = std::max (1.0, bpm) / 60.0;
    size_t next = 0;
    bool pendingBreath = false;
    int lastSyllable = -1;

    for (const auto& e : script.events)
    {
        if (next >= mel.size())
            break;

        switch (e.type)
        {
            case LyricEvent::Type::Breath:
            case LyricEvent::Type::PhraseEnd:
                pendingBreath = true;
                lastSyllable = -1;
                break;

            case LyricEvent::Type::Syllable:
            {
                const auto& n = mel[next++];
                ScoreNote sn;
                sn.startBeat = n.startBeat;
                sn.lengthBeats = n.lengthBeats;
                sn.midiNote = n.midiNote;
                sn.velocity = n.velocity;
                sn.syllable = e.syllable;
                sn.breathBefore = pendingBreath || score.notes.empty();
                pendingBreath = false;
                lastSyllable = e.syllable;
                score.notes.push_back (sn);
                break;
            }

            case LyricEvent::Type::Melisma:
            {
                if (lastSyllable < 0)
                    break;

                const auto& n = mel[next++];
                ScoreNote sn;
                sn.startBeat = n.startBeat;
                sn.lengthBeats = n.lengthBeats;
                sn.midiNote = n.midiNote;
                sn.velocity = n.velocity;
                sn.syllable = lastSyllable;
                sn.continuation = true;
                score.notes.push_back (sn);
                break;
            }
        }
    }

    // Spare notes: the last syllable carries on through the rest of its phrase (as a melisma);
    // anything after the next rest is dropped.
    if (! score.notes.empty())
    {
        const int held = score.notes.back().syllable;
        int extended = 0;
        while (next < mel.size() && extended < 8)
        {
            const auto& n = mel[next];
            const double gap = n.startBeat - score.notes.back().endBeat();
            if (gap > 0.26)
                break;

            ScoreNote sn;
            sn.startBeat = n.startBeat;
            sn.lengthBeats = n.lengthBeats;
            sn.midiNote = n.midiNote;
            sn.velocity = n.velocity;
            sn.syllable = held;
            sn.continuation = true;
            score.notes.push_back (sn);
            ++next;
            ++extended;
        }
    }

    // Phrase boundaries need a moment to breathe: shorten the previous note if the melody is legato.
    for (size_t i = 1; i < score.notes.size(); ++i)
    {
        auto& note = score.notes[i];
        if (! note.breathBefore)
            continue;

        auto& prev = score.notes[i - 1];
        const double wantedGap = std::min (0.3 * beatsPerSecond, 0.4 * prev.lengthBeats);
        const double gap = note.startBeat - prev.endBeat();
        if (gap < wantedGap)
            prev.lengthBeats = std::max (prev.lengthBeats * 0.5, note.startBeat - wantedGap - prev.startBeat);
    }

    return score;
}

} // namespace voxa::sing::melody
