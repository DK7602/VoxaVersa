#include "OfflineRenderer.h"

#include <algorithm>
#include <numeric>

namespace voxa::sing
{

namespace
{
    double smoothstep (double x) noexcept
    {
        x = juce::jlimit (0.0, 1.0, x);
        return x * x * (3.0 - 2.0 * x);
    }

    double total (const std::vector<float>& v) noexcept
    {
        return std::accumulate (v.begin(), v.end(), 0.0);
    }

    /** Squeezes consonant durations into `window` seconds: proportional compression, respecting each
        phoneme's minimum first and only then going below it. */
    void fitDurations (std::vector<float>& durations, const std::vector<float>& minimums, double window)
    {
        const double sum = total (durations);
        if (sum <= window || sum <= 0.0)
            return;

        window = juce::jmax (0.0, window);
        const double scale = window / sum;
        double newSum = 0.0;
        for (size_t k = 0; k < durations.size(); ++k)
        {
            durations[k] = juce::jmax (minimums[k], (float) (durations[k] * scale));
            newSum += durations[k];
        }

        if (newSum > window)
        {
            const double scale2 = window / newSum;
            for (auto& d : durations)
                d = juce::jmax (0.008f, (float) (d * scale2));
        }
    }

    /** Zero-phase Gaussian smoothing with clamped edges. */
    void gaussianSmooth (std::vector<float>& track, double sigmaFrames, std::vector<float>& scratch)
    {
        if (sigmaFrames < 0.3 || track.size() < 2)
            return;

        const int radius = juce::jmax (1, (int) std::ceil (3.0 * sigmaFrames));
        std::vector<float> kernel ((size_t) (2 * radius + 1));
        double norm = 0.0;
        for (int k = -radius; k <= radius; ++k)
        {
            const double w = std::exp (-0.5 * (k * k) / (sigmaFrames * sigmaFrames));
            kernel[(size_t) (k + radius)] = (float) w;
            norm += w;
        }
        for (auto& w : kernel)
            w = (float) (w / norm);

        scratch = track;
        const int n = (int) track.size();
        for (int i = 0; i < n; ++i)
        {
            float acc = 0.0f;
            if (i >= radius && i + radius < n)
            {
                const float* src = scratch.data() + i - radius;
                for (int k = 0; k <= 2 * radius; ++k)
                    acc += kernel[(size_t) k] * src[k];
            }
            else
            {
                for (int k = -radius; k <= radius; ++k)
                    acc += kernel[(size_t) (k + radius)] * scratch[(size_t) juce::jlimit (0, n - 1, i + k)];
            }
            track[(size_t) i] = acc;
        }
    }

    float velocityGain (float velocity) noexcept
    {
        const float v = juce::jlimit (0.0f, 1.0f, velocity);
        return juce::Decibels::decibelsToGain (-16.0f * std::pow (1.0f - v, 1.2f));
    }

    bool isStopLike (Ph p) noexcept
    {
        const auto m = phonemeInfo (p).manner;
        return m == Manner::Stop || m == Manner::Affricate;
    }

    /** Consonants after which American /u/ is fronted ("you", "to", "do", "new", "choose"). */
    bool frontsFollowingU (Ph p) noexcept
    {
        return p == Ph::Y || p == Ph::T || p == Ph::D || p == Ph::N || p == Ph::S || p == Ph::Z || p == Ph::SH
            || p == Ph::ZH || p == Ph::CH || p == Ph::JH || p == Ph::TH || p == Ph::DH;
    }

    /** Off-glide timing of a sung diphthong, or of a vowel colouring into a coda R / L: the nucleus is
        held and the glide comes late, but it completes a little before the vowel ends so that the
        off-glide target is actually reached before the consonant transition takes over. */
    void setOffGlide (PhoneEvent& e)
    {
        e.glideTime = juce::jlimit (0.07f, 0.18f, 0.2f * e.duration + 0.04f);
        e.glideStart = juce::jmax (0.0f, e.duration - 1.15f * e.glideTime);
    }
}

//==============================================================================
OfflineRenderer::OfflineRenderer (const LyricScript& s, const Score& sc, const VoiceTraits& t,
                                  double tempo, double rate, juce::uint32 seedToUse)
    : script (s), score (sc), traits (t), bpm (juce::jlimit (20.0, 400.0, tempo)),
      sampleRate (rate > 0.0 ? rate : 48000.0), seed (seedToUse)
{
    applyAccent (script, (Accent) traits.singAccent);
    setup.prepare (traits);
    plan();
}

void OfflineRenderer::plan()
{
    const double secondsPerBeat = 60.0 / bpm;
    const float tempoScale = 1.0f - 0.2f * setup.busy;

    notes.clear();
    segments.clear();

    for (const auto& n : score.notes)
    {
        TimedNote t;
        t.start = n.startBeat * secondsPerBeat;
        t.end = n.endBeat() * secondsPerBeat;
        t.midi = (float) n.midiNote;
        t.velocity = n.velocity;
        notes.push_back (t);
    }

    if (notes.empty())
        return;

    // Syllable occurrences: a syllable plus the melisma notes it is held through.
    std::vector<Occurrence> occ;
    for (int i = 0; i < (int) score.notes.size(); ++i)
    {
        const auto& n = score.notes[(size_t) i];
        if (n.continuation && ! occ.empty() && occ.back().syllable == n.syllable)
        {
            occ.back().lastNote = i;
            occ.back().noteEnd = notes[(size_t) i].end;
            continue;
        }

        Occurrence o;
        o.syllable = juce::jlimit (0, (int) script.syllables.size() - 1, n.syllable);
        o.sung = script.syllables[(size_t) o.syllable];
        o.firstNote = o.lastNote = i;
        o.vowelStart = notes[(size_t) i].start;
        o.noteEnd = notes[(size_t) i].end;
        o.breathBefore = n.breathBefore || occ.empty();
        occ.push_back (std::move (o));
    }

    if (script.syllables.empty())
        return;

    const int numOcc = (int) occ.size();
    for (int i = 0; i + 1 < numOcc; ++i)
        occ[(size_t) i].connected = ! occ[(size_t) i + 1].breathBefore
                                    && occ[(size_t) i + 1].vowelStart - occ[(size_t) i].noteEnd < 0.06;

    applyConnectedSpeech (occ);

    // Nominal consonant durations (a flap is a single quick tap). Consonants of unstressed syllables
    // are shorter in English (Klatt's duration rules: ~70 %; a little less shortening for singing).
    constexpr float tapSeconds = 0.028f, tapMinimum = 0.018f;
    for (auto& o : occ)
    {
        const auto& syl = o.sung;
        const float onsetCluster = (syl.onset.size() >= 2 ? 0.85f : 1.0f) * (syl.stress == 0 ? 0.8f : 1.0f);
        const float codaCluster = syl.coda.size() >= 2 ? 0.85f : 1.0f;
        for (auto p : syl.onset)
        {
            const auto& info = phonemeInfo (p);
            o.onset.push_back (o.onsetTap ? tapSeconds : info.onsetMs * 0.001f * tempoScale * onsetCluster);
            o.onsetMin.push_back (o.onsetTap ? tapMinimum : info.minMs * 0.001f);
        }
        for (size_t k = 0; k < syl.coda.size(); ++k)
        {
            const auto& info = phonemeInfo (syl.coda[k]);
            const bool tap = o.codaTap && k + 1 == syl.coda.size();
            o.coda.push_back (tap ? tapSeconds : info.codaMs * 0.001f * tempoScale * codaCluster);
            o.codaMin.push_back (tap ? tapMinimum : info.minMs * 0.001f);
        }
    }

    // Pass A: syllable boundaries.
    for (int i = 0; i < numOcc; ++i)
    {
        auto& o = occ[(size_t) i];
        const double lastNoteLength = notes[(size_t) o.lastNote].end - notes[(size_t) o.lastNote].start;

        if (o.connected)
        {
            auto& next = occ[(size_t) i + 1];
            const double ioi = next.vowelStart - o.vowelStart;
            const double minVowel = juce::jmax (0.045, 0.4 * ioi);

            // Detached singing leaves a little gap between words - but never inside a word ("ne-ver"
            // broken by silence sounds syllable-timed), at a flap, or where a final consonant links into
            // a vowel-initial word ("hold on"). (Vowel to vowel across words keeps the gap: a soft
            // glottal onset, "day [ʔ]a", where a seamless glide would sound like a consonant.)
            const bool linking = next.sung.wordIndex == o.sung.wordIndex || next.onsetTap || o.codaTap
                              || (next.sung.onset.empty() && ! o.sung.coda.empty());
            const double gap = linking ? 0.0 : (1.0 - setup.legato) * (1.0 - setup.legato) * juce::jmin (0.3 * lastNoteLength, 0.12);
            const double window = juce::jmax (0.01, ioi - minVowel - gap);

            std::vector<float> joint (o.coda), jointMin (o.codaMin);
            joint.insert (joint.end(), next.onset.begin(), next.onset.end());
            jointMin.insert (jointMin.end(), next.onsetMin.begin(), next.onsetMin.end());
            fitDurations (joint, jointMin, window);

            std::copy (joint.begin(), joint.begin() + (long) o.coda.size(), o.coda.begin());
            std::copy (joint.begin() + (long) o.coda.size(), joint.end(), next.onset.begin());

            next.onsetStart = next.vowelStart - total (next.onset);
            o.codaEnd = next.onsetStart - gap;
            o.vowelEnd = o.codaEnd - total (o.coda);
        }
        else
        {
            const double span = o.noteEnd - o.vowelStart;
            const double minVowel = juce::jmax (0.05, 0.5 * span);
            double limit = 1.0e9;
            if (i + 1 < numOcc)
                limit = occ[(size_t) i + 1].vowelStart - 0.1;

            const double nominal = total (o.coda);
            fitDurations (o.coda, o.codaMin, juce::jmax (0.02, juce::jmin ((span - minVowel) + 0.5 * nominal,
                                                                          limit - (o.vowelStart + minVowel))));
            const double codaLength = total (o.coda);
            o.vowelEnd = juce::jmax (o.vowelStart + minVowel, o.noteEnd - 0.5 * codaLength);
            o.codaEnd = o.vowelEnd + codaLength;
        }
    }

    // Pass B: onsets after a rest (and the very first one).
    for (int i = 0; i < numOcc; ++i)
    {
        auto& o = occ[(size_t) i];
        if (i > 0 && occ[(size_t) i - 1].connected)
            continue;

        const double earliest = i == 0 ? 0.02 : occ[(size_t) i - 1].codaEnd + 0.04;
        const double window = o.vowelStart - earliest;
        fitDurations (o.onset, o.onsetMin, window);

        const double onsetLength = total (o.onset);
        if (o.vowelStart - onsetLength < earliest)
        {
            // No room at all (a note right at the start): the vowel comes a little late.
            o.vowelStart = juce::jmin (earliest + onsetLength, o.noteEnd - 0.03);
            o.vowelEnd = juce::jmax (o.vowelEnd, o.vowelStart + 0.03);
            o.codaEnd = juce::jmax (o.codaEnd, o.vowelEnd + total (o.coda));
        }
        o.onsetStart = o.vowelStart - onsetLength;
    }

    // Segments.
    const float breathiness = juce::jlimit (0.0f, 1.0f, traits.breathiness);
    for (int i = 0; i < numOcc; ++i)
    {
        const auto& o = occ[(size_t) i];
        const auto& syl = o.sung;
        const bool previousConnected = i > 0 && occ[(size_t) i - 1].connected;

        if (o.breathBefore)
        {
            const double previousEnd = i == 0 ? 0.0 : occ[(size_t) i - 1].codaEnd;
            const double available = o.onsetStart - 0.05 - previousEnd - 0.05;
            if (available >= 0.15 && (setup.busy < 0.85 || available >= 0.3))
            {
                const double length = juce::jmin (0.42, 0.8 * available);
                Segment b;
                b.event.ph = Ph::Breath;
                b.event.duration = (float) length;
                b.start = o.onsetStart - 0.05 - length;
                b.end = o.onsetStart - 0.05;
                b.note = o.firstNote;
                b.level = 0.9f + 1.0f * breathiness;
                segments.push_back (b);
            }
        }

        double t = o.onsetStart;
        for (size_t k = 0; k < syl.onset.size(); ++k)
        {
            Segment s;
            s.event.ph = syl.onset[k];
            s.event.contextVowel = isVowel (syl.nucleus) ? syl.nucleus : Ph::AX;
            s.event.duration = o.onset[k];
            // Voiceless stops are aspirated at the start of stressed syllables and words ("tea", "today"),
            // not after S ("stay") or inside a word before an unstressed vowel ("happy").
            s.event.aspirated = ! (k > 0 && syl.onset[k - 1] == Ph::S) && (syl.stress >= 1 || syl.wordStart);
            s.event.released = true;
            s.event.tap = o.onsetTap;
            s.event.afterPause = k == 0 && ! previousConnected;
            s.start = t;
            s.end = t + o.onset[k];
            s.note = o.firstNote;
            segments.push_back (s);
            t = s.end;
        }

        Segment v;
        v.nucleus = true;
        v.event.ph = syl.nucleus;
        v.event.contextVowel = isVowel (syl.nucleus) ? syl.nucleus : Ph::AX;
        v.event.syllabic = ! isVowel (syl.nucleus);
        v.start = o.vowelStart;
        v.end = juce::jmax (o.vowelStart + 0.02, o.vowelEnd);
        v.event.duration = (float) (v.end - v.start);
        v.note = o.firstNote;
        v.level = setup.vowelLoudnessCompensation (syl.nucleus, music::midiToHz (notes[(size_t) o.firstNote].midi));
        const bool nasalCoda = ! syl.coda.empty() && phonemeInfo (syl.coda.front()).manner == Manner::Nasal;
        v.event.contextual = (syl.nucleus == Ph::UW && ! syl.onset.empty() && frontsFollowingU (syl.onset.back()))
                          || (syl.nucleus == Ph::AE && nasalCoda);
        if (! isDiphthong (syl.nucleus) && isVowel (syl.nucleus) && ! syl.coda.empty()
            && (syl.coda.front() == Ph::R || syl.coda.front() == Ph::L))
            v.event.liquidCoda = syl.coda.front();

        if (isDiphthong (syl.nucleus) || v.event.liquidCoda != Ph::Sil)
            setOffGlide (v.event);
        if (syl.nucleus == Ph::AX && v.event.liquidCoda == Ph::L)
        {
            // An unstressed schwa before L ("lit-tle", "beau-ti-ful") is dark L-coloured throughout.
            v.event.glideStart = 0.0f;
            v.event.glideTime = juce::jmin (0.1f, 0.5f * v.event.duration);
        }

        // Attack: how voicing begins at the vowel.
        const float velocity = notes[(size_t) o.firstNote].velocity;
        if (o.onsetTap)
        {
            v.attack = 0.008f;   // the voice runs on through the flap
            v.attackFloor = 0.6f;
        }
        else if (! syl.onset.empty())
        {
            const auto last = syl.onset.back();
            if (isSonorant (last) || (isVoiced (last) && phonemeInfo (last).manner == Manner::Fricative))
            {
                v.attack = 0.012f;
                v.attackFloor = 0.5f;
            }
            else
            {
                v.attack = isVoiced (last) ? 0.015f : 0.022f;
                v.attackFloor = 0.0f;
            }
        }
        else if (previousConnected)
        {
            v.attack = 0.0f;   // vowel to vowel: the voice keeps going
        }
        else
        {
            v.attack = juce::jmap (velocity, 0.075f, 0.035f) + 0.03f * breathiness;
            v.attackFloor = 0.0f;
        }

        if (! o.connected)
            v.release = syl.coda.empty() ? (float) juce::jlimit (0.06, 0.2, 0.3 * (v.end - v.start)) : 0.03f;
        segments.push_back (v);

        t = v.end;
        for (size_t k = 0; k < syl.coda.size(); ++k)
        {
            Segment s;
            s.event.ph = syl.coda[k];
            s.event.contextVowel = isVowel (syl.nucleus) ? syl.nucleus : Ph::AX;
            s.event.duration = o.coda[k];
            s.event.aspirated = false;
            s.event.coda = true;

            // English final stops are unreleased before another consonant ("kept", "good night", "hold
            // me"), released into a following vowel or W ("hold on", "stop wasting"), and released softly
            // before a pause. T and D stay unreleased before Y: released into it they run together as "ch" /
            // "j" ("can't you" -> "cantchu", heard as sloppy in a sung lyric).
            Ph following = Ph::Sil;
            const bool last = k + 1 == syl.coda.size();
            if (! last)
                following = syl.coda[k + 1];
            else if (o.connected)
            {
                const auto& next = occ[(size_t) i + 1].sung;
                following = next.onset.empty() ? next.nucleus : next.onset.front();
            }
            s.event.linked = last && o.connected && isVowel (following);
            const bool alveolarBeforeY = following == Ph::Y && (syl.coda[k] == Ph::T || syl.coda[k] == Ph::D);
            s.event.released = ! (isStopLike (syl.coda[k]) && isConsonant (following)
                                  && following != Ph::W && (following != Ph::Y || alveolarBeforeY));
            s.event.tap = last && o.codaTap;

            s.start = t;
            s.end = t + o.coda[k];
            s.note = o.lastNote;
            segments.push_back (s);
            t = s.end;
        }
    }

    std::stable_sort (segments.begin(), segments.end(), [] (const Segment& a, const Segment& b) { return a.start < b.start; });

    // Pitch notes.
    const float portamento = setup.portamentoSeconds;
    for (int i = 0; i < numOcc; ++i)
    {
        const auto& o = occ[(size_t) i];
        for (int n = o.firstNote; n <= o.lastNote; ++n)
        {
            auto& note = notes[(size_t) n];
            note.stressed = n == o.firstNote && o.sung.stress > 0;
            if (n > o.firstNote || (i > 0 && occ[(size_t) i - 1].connected))
            {
                const auto& prev = notes[(size_t) n - 1];
                const double shortest = juce::jmin (prev.end - prev.start, note.end - note.start);
                const double interval = std::abs (note.midi - prev.midi);
                const double wanted = n > o.firstNote ? portamento * 0.7 : portamento * (1.0 + 0.03 * interval);
                note.legatoFromPrevious = true;
                note.glide = juce::jmax (0.01, juce::jmin (wanted, 0.45 * shortest));
            }
            else
            {
                note.attackFrom = o.onsetStart;
            }

            if (n == o.lastNote && ! o.connected)
            {
                note.phraseEnd = true;
                note.voiceEnd = o.vowelEnd;
            }
        }
    }

    double end = notes.back().end;
    for (const auto& o : occ)
        end = juce::jmax (end, o.codaEnd);
    durationSeconds = end + 0.4;
}

//==============================================================================
void OfflineRenderer::applyConnectedSpeech (std::vector<Occurrence>& occ) const
{
    // American English connected speech. Sung without these, lyrics sound read out word by word in
    // careful "textbook" English - one of the clearest marks of a non-native singer.
    const int n = (int) occ.size();

    // Weak forms of function words on short notes that run on into the next word ("to the" [tə ðə],
    // "for you" [fɚ ju]). Long, held or phrase-final notes keep the full vowel, as singers do.
    constexpr double weakFormLongestNote = 0.42;
    for (int i = 0; i + 1 < n; ++i)
    {
        auto& o = occ[(size_t) i];
        if (! o.sung.hasWeakForm || ! o.connected || o.sung.pauseAfter || o.lastNote != o.firstNote
            || o.noteEnd - o.vowelStart > weakFormLongestNote)
            continue;

        if (o.sung.weakOnlyBeforeConsonant && occ[(size_t) i + 1].sung.onset.empty())
            continue;

        o.sung.nucleus = o.sung.weakNucleus;
        o.sung.coda = o.sung.weakCoda;
    }

    // Flapping: T / D between a vowel (or R) and an unstressed vowel inside a word ("better", "city",
    // "party"), and a final T / D before a vowel-initial word ("got it", "what about").
    for (int i = 0; i < n; ++i)
    {
        auto& o = occ[(size_t) i];
        const auto& s = o.sung;
        auto isTD = [] (Ph p) { return p == Ph::T || p == Ph::D; };

        if (i > 0 && s.onset.size() == 1 && isTD (s.onset.front()) && s.stress == 0 && ! s.wordStart
            && isVowel (s.nucleus))
        {
            const auto& prev = occ[(size_t) i - 1];
            if (prev.connected && prev.sung.wordIndex == s.wordIndex && isVowel (prev.sung.nucleus)
                && (prev.sung.coda.empty() || (prev.sung.coda.size() == 1 && prev.sung.coda.front() == Ph::R)))
                o.onsetTap = true;
        }

        if (i + 1 < n && o.connected && s.wordEnd && ! s.coda.empty() && isTD (s.coda.back()) && isVowel (s.nucleus)
            && (s.coda.size() == 1 || (s.coda.size() == 2 && s.coda.front() == Ph::R)))
        {
            const auto& next = occ[(size_t) i + 1].sung;
            if (next.wordStart && next.onset.empty() && isVowel (next.nucleus))
                o.codaTap = true;
        }
    }
}

//==============================================================================
void OfflineRenderer::buildPitchTrack (int numFrames, double frameSeconds, std::vector<float>& midi) const
{
    midi.assign ((size_t) numFrames, notes.empty() ? 60.0f : notes.front().midi);
    if (notes.empty())
        return;

    auto frameAt = [&] (double t) { return juce::jlimit (0, numFrames, (int) std::ceil (t / frameSeconds - 1.0)); };
    auto timeOf = [&] (int k) { return (k + 1) * frameSeconds; };

    // Plateaus
    for (size_t n = 0; n < notes.size(); ++n)
    {
        const auto& note = notes[n];
        const double from = note.legatoFromPrevious ? note.start : juce::jmin (note.start, note.attackFrom);
        for (int k = frameAt (from); k < numFrames; ++k)
            midi[(size_t) k] = note.midi;
    }

    // The performance follows contemporary (pop / R&B / soul) singers rather than a synthesizer: notes are
    // sung straight and only long notes bloom into vibrato late and unevenly; stressed attacks scoop up from
    // well below; phrase ends fall off; and no note is tuned to the cent. Measured on studio lead vocals:
    // ~50 % of their sung time is steady pitch (a regular vibrato on every note gave 72 %), attacks start
    // 0.3-3 semitones low, long notes carry 1-2 semitone vibrato, and melodic movement is fast.
    juce::Random random ((juce::int64) seed * 7919 + 17);
    const float scoop = juce::jlimit (0.0f, 3.0f, setup.scoopSemis * 1.8f);
    const double scoopRise = juce::jlimit (0.06, 0.25, setup.scoopSeconds * (0.6 + 0.2 * scoop));
    const float fall = juce::jlimit (0.0f, 2.5f, (setup.scoopSemis - 0.15f) * 2.0f);

    const float style = juce::jlimit (0.0f, 1.0f, scoop / 1.5f);   // how freely the genre bends notes (R&B ~0.7)

    for (size_t n = 0; n < notes.size(); ++n)
    {
        const auto& note = notes[n];

        // A note is never quite on the grid, and a held one sags or lifts a little as it goes on.
        const double offset = (random.nextDouble() - 0.5) * 0.06;
        const double bend = (random.nextDouble() - 0.5) * (0.1 + 0.12 * style);
        const double length = juce::jmax (0.05, note.end - note.start);
        for (int k = frameAt (note.start); k < frameAt (note.end); ++k)
        {
            const double x = (timeOf (k) - note.start) / length;
            midi[(size_t) k] += (float) (offset + bend * smoothstep ((x - 0.2) / 0.8));
        }

        // Soulful styles decorate a held note now and then with a quick dip to the note below and back.
        if (style > 0.3f && length >= 0.6 && ! note.phraseEnd && random.nextDouble() < 0.35 * style)
        {
            const double at = note.start + length * (0.35 + 0.3 * random.nextDouble());
            const double dip = 1.0 + random.nextDouble();
            for (int k = frameAt (at); k < frameAt (at + 0.14); ++k)
            {
                const double x = (timeOf (k) - at) / 0.14;
                midi[(size_t) k] -= (float) (dip * std::sin (juce::MathConstants<double>::pi * x));
            }
        }
    }

    for (size_t n = 0; n < notes.size(); ++n)
    {
        const auto& note = notes[n];

        if (note.legatoFromPrevious && n > 0)
        {
            // Portamento centred slightly before the beat, plus a damped overshoot on leaps.
            const auto& prev = notes[n - 1];
            const double g = note.glide;
            const double w0 = note.start - 0.4 * g;
            const double delta = note.midi - prev.midi;
            for (int k = frameAt (w0); k < numFrames; ++k)
            {
                const double t = timeOf (k);
                if (t > note.start + 0.6 * g)
                    break;
                midi[(size_t) k] = (float) (prev.midi + delta * smoothstep ((t - w0) / g));
            }

            if (std::abs (delta) >= 3.0)
            {
                const double amount = (delta > 0 ? 1.0 : -1.0) * juce::jmin (0.35, 0.05 * std::abs (delta))
                                      * (setup.busy < 0.8 ? 1.0 : 0.5);
                const double t0 = note.start + 0.2 * g;
                for (int k = frameAt (t0); k < numFrames; ++k)
                {
                    const double x = (timeOf (k) - t0) / 0.06;
                    if (x > 6.0 || timeOf (k) > note.end)
                        break;
                    if (x > 0.0)
                        midi[(size_t) k] += (float) (amount * x * std::exp (1.0 - x));
                }
            }

            // Falling to a lower note, soulful singers pass through below it and come back up.
            if (delta <= -2.0 && style > 0.3f && random.nextDouble() < 0.6 * style)
            {
                const double under = (0.4 + 0.6 * random.nextDouble()) * style;
                const double t0 = note.start + 0.3 * g;
                for (int k = frameAt (t0); k < numFrames; ++k)
                {
                    const double x = (timeOf (k) - t0) / 0.07;
                    if (x > 6.0 || timeOf (k) > note.end)
                        break;
                    if (x > 0.0)
                        midi[(size_t) k] -= (float) (under * x * std::exp (1.0 - x));
                }
            }

            // Rising onto a stressed syllable, the voice often dips below the note first and slides up into it.
            if (note.stressed && delta > 0.0 && scoop > 0.0f && random.nextDouble() < 0.4)
            {
                const double depth = 0.5 * scoop * (0.7 + 0.6 * random.nextDouble());
                for (int k = frameAt (note.start); k < numFrames; ++k)
                {
                    const double t = timeOf (k);
                    if (t > note.start + scoopRise || t > note.end)
                        break;
                    midi[(size_t) k] -= (float) (depth * (1.0 - smoothstep ((t - note.start) / scoopRise)));
                }
            }
        }
        else if (scoop > 0.0f)
        {
            // Scoop up into a fresh attack (the onset consonants sit at the scooped pitch): deep on stressed
            // syllables, slight on weak ones, never the same twice.
            const double depth = scoop * (note.stressed ? 1.0 : 0.4) * (0.7 + 0.6 * random.nextDouble());
            for (int k = frameAt (note.attackFrom); k < numFrames; ++k)
            {
                const double t = timeOf (k);
                if (t > note.start + scoopRise || t > note.end)
                    break;
                const double amount = t < note.start ? 1.0 : 1.0 - smoothstep ((t - note.start) / scoopRise);
                midi[(size_t) k] -= (float) (depth * amount);
            }
        }

        // Most phrase endings fall away as the voice lets go.
        if (note.phraseEnd && fall > 0.0f && random.nextDouble() < 0.75)
        {
            const double amount = fall * (0.7 + 0.5 * random.nextDouble());
            const double f0 = note.voiceEnd - 0.12;
            for (int k = frameAt (f0); k < numFrames; ++k)
            {
                const double t = timeOf (k);
                if (t > note.voiceEnd + 0.3)
                    break;
                midi[(size_t) k] -= (float) (amount * smoothstep ((t - f0) / 0.18));
            }
        }
    }

    // Vibrato only on held notes, blooming in their second half; every cycle a little different in rate and
    // depth. Phrase-final notes always get it, others about half the time and lighter.
    const double depth = setup.vibratoDepthSemis;
    if (depth > 0.0)
    {
        for (size_t n = 0; n < notes.size(); ++n)
        {
            const auto& note = notes[n];
            const bool nextLegato = n + 1 < notes.size() && notes[n + 1].legatoFromPrevious;
            const double held = (nextLegato ? notes[n + 1].start - 0.4 * notes[n + 1].glide
                                            : (note.phraseEnd ? note.voiceEnd : note.end)) - note.start;
            const bool chosen = note.phraseEnd || random.nextDouble() < 0.5;
            if (held < 0.5 || ! chosen)
                continue;

            const double vStart = note.start + juce::jmax ((double) setup.vibratoOnsetSeconds, 0.4 * held);
            const double vEnd = note.start + held + (note.phraseEnd && ! nextLegato ? 0.1 : 0.0);
            if (vEnd - vStart < 0.2)
                continue;

            const double noteDepth = depth * (note.phraseEnd ? 1.0 : 0.6);
            double phase = 0.0, cycleRate = setup.vibratoRateHz, cycleDepth = noteDepth;
            for (int k = frameAt (vStart); k < numFrames; ++k)
            {
                const double t = timeOf (k);
                if (t >= vEnd)
                    break;

                const double fadeIn = smoothstep ((t - vStart) / 0.4);
                const double fadeOut = nextLegato ? 1.0 - smoothstep ((t - (vEnd - 0.07)) / 0.07) : 1.0;
                phase += juce::MathConstants<double>::twoPi * cycleRate * frameSeconds;
                if (phase >= juce::MathConstants<double>::twoPi)
                {
                    phase -= juce::MathConstants<double>::twoPi;
                    cycleRate = setup.vibratoRateHz * (0.93 + 0.14 * random.nextDouble());
                    cycleDepth = noteDepth * (0.75 + 0.5 * random.nextDouble());
                }
                midi[(size_t) k] += (float) (cycleDepth * fadeIn * fadeOut * std::sin (phase));
            }
        }
    }

    // Slow random drift (three incommensurate sines).
    if (setup.driftSemis > 0.0f)
    {
        double freq[3], ph[3];
        for (int i = 0; i < 3; ++i)
        {
            freq[i] = (0.6 + random.nextDouble() * 0.8) * (1 << i);
            ph[i] = random.nextDouble() * juce::MathConstants<double>::twoPi;
        }
        const double amp = setup.driftSemis * 0.8 + 0.015;
        for (int k = 0; k < numFrames; ++k)
        {
            const double t = timeOf (k);
            double v = 0.0;
            for (int i = 0; i < 3; ++i)
                v += std::sin (juce::MathConstants<double>::twoPi * freq[i] * t + ph[i]) / (1.0 + 0.5 * i);
            midi[(size_t) k] += (float) (amp * v);
        }
    }
}

//==============================================================================
bool OfflineRenderer::render (const std::atomic<bool>* cancel, const std::function<void (float)>& progress)
{
    juce::ScopedNoDenormals noDenormals;

    auto cancelled = [cancel] { return cancel != nullptr && cancel->load (std::memory_order_relaxed); };
    auto report = [&progress] (float p) { if (progress) progress (juce::jlimit (0.0f, 1.0f, p)); };

    if (notes.empty() || segments.empty())
        return ! cancelled();

    const int frameLength = juce::jmax (16, (int) std::round (sampleRate / 1000.0));
    const double frameSeconds = frameLength / sampleRate;
    const int totalSamples = (int) std::ceil (durationSeconds * sampleRate);
    const int numFrames = (totalSamples + frameLength - 1) / frameLength;

    // Raw articulation per frame.
    std::vector<std::vector<float>> tracks ((size_t) numArticParams, std::vector<float> ((size_t) numFrames, 0.0f));
    std::vector<bool> valid ((size_t) numFrames, false);
    std::vector<float> dynamics ((size_t) numFrames, 1.0f), rd ((size_t) numFrames, setup.baseRd);

    ArticFrame frame;
    size_t segIndex = 0;
    int dynamicsNote = 0;
    std::vector<int> frameSegment ((size_t) numFrames, -1);
    for (int k = 0; k < numFrames; ++k)
    {
        const double t = (k + 1) * frameSeconds;
        while (segIndex < segments.size() && segments[segIndex].end <= t)
            ++segIndex;

        const Segment* seg = nullptr;
        if (segIndex < segments.size() && segments[segIndex].start <= t)
            seg = &segments[segIndex];

        if (seg != nullptr)
        {
            frameSegment[(size_t) k] = (int) segIndex;
            const float local = (float) (t - seg->start);
            articulate (seg->event, local, setup, frame);

            float gain = seg->level;
            if (seg->nucleus)
            {
                const float length = (float) (seg->end - seg->start);
                if (seg->attack > 0.0f)
                    gain *= seg->attackFloor + (1.0f - seg->attackFloor) * (float) smoothstep (local / seg->attack);
                if (seg->release > 0.0f)
                    gain *= (float) smoothstep ((length - local) / seg->release);
                if (length > 0.8f)
                    gain *= 1.0f + 0.1f * std::sin (juce::MathConstants<float>::pi * local / length);
            }

            frame.v[apAV] *= gain;
            frame.v[apAH] *= gain;
            frame.v[apAF] *= gain;
            dynamicsNote = seg->note;

            // Within a melisma the dynamics follow the note being sung.
            if (seg->nucleus)
                while (dynamicsNote + 1 < (int) notes.size() && notes[(size_t) dynamicsNote + 1].start <= t
                       && score.notes[(size_t) dynamicsNote + 1].continuation)
                    ++dynamicsNote;
        }
        else
        {
            articulate (PhoneEvent(), 0.0f, setup, frame);
        }

        for (int p = 0; p < numArticParams; ++p)
            tracks[(size_t) p][(size_t) k] = frame.v[(size_t) p];
        valid[(size_t) k] = frame.formantsValid;

        const float velocity = notes[(size_t) juce::jlimit (0, (int) notes.size() - 1, dynamicsNote)].velocity;
        dynamics[(size_t) k] = velocityGain (velocity);
        rd[(size_t) k] = juce::jlimit (0.3f, 2.7f, setup.baseRd + 0.7f * (0.7f - velocity));
    }

    if (cancelled())
        return false;
    report (0.05f);

    // Silence has no articulation: hold the neighbouring formants so transitions stay put.
    const int shapeParams[] = { apF1, apF2, apF3, apB1, apB2, apB3, apUpperDamping, apNasalZero,
                                apFricF1, apFricB1, apFricF2, apFricB2, apFricHighpass, apF1Tuning };
    int firstValid = -1;
    for (int k = 0; k < numFrames; ++k)
    {
        if (valid[(size_t) k])
        {
            if (firstValid < 0)
                firstValid = k;
            continue;
        }
        if (k > 0)
            for (auto p : shapeParams)
                tracks[(size_t) p][(size_t) k] = tracks[(size_t) p][(size_t) k - 1];
    }
    if (firstValid > 0)
        for (int k = 0; k < firstValid; ++k)
            for (auto p : shapeParams)
                tracks[(size_t) p][(size_t) k] = tracks[(size_t) p][(size_t) firstValid];

    // Consonant-vowel transitions start at the release (and end at the closure) rather than being
    // centred on the boundary: hold the consonant's formants a little way into the vowel so that the
    // audible part of the transition carries the place-of-articulation cue.
    {
        auto frameOf = [frameSeconds, numFrames] (double t)
        {
            return juce::jlimit (0, numFrames - 1, (int) std::ceil (t / frameSeconds - 1.0));
        };
        auto holdTime = [] (const Segment& c) -> double
        {
            const auto& info = phonemeInfo (c.event.ph);
            switch (info.manner)
            {
                case Manner::Stop:      return (c.event.tap || (! info.voiced && c.event.aspirated && c.event.released)) ? 0.0 : 0.016;
                case Manner::Affricate: return 0.012;
                case Manner::Nasal:     return 0.014;
                case Manner::Fricative: return 0.010;
                case Manner::Vowel: case Manner::Diphthong: case Manner::Aspirate: case Manner::Liquid:
                case Manner::Glide: case Manner::Silence: case Manner::Breath:
                default:                return 0.0;
            }
        };
        const int formantParams[] = { apF1, apF2, apF3 };

        for (size_t i = 0; i + 1 < segments.size(); ++i)
        {
            const auto& a = segments[i];
            const auto& b = segments[i + 1];
            if (std::abs (b.start - a.end) > 0.002)
                continue;

            const bool aVowel = a.nucleus && isVowel (a.event.ph);
            const bool bVowel = b.nucleus && isVowel (b.event.ph);

            if (! aVowel && bVowel)
            {
                const double hold = holdTime (a);
                if (hold <= 0.0)
                    continue;
                const int source = frameOf (a.end - 0.5 * frameSeconds);
                for (int k = frameOf (b.start); k <= frameOf (b.start + hold) && k < numFrames; ++k)
                    if (frameSegment[(size_t) k] == (int) i + 1)
                        for (auto p : formantParams)
                            tracks[(size_t) p][(size_t) k] = tracks[(size_t) p][(size_t) source];
            }
            else if (aVowel && ! bVowel)
            {
                const double hold = holdTime (b) > 0.0 ? 0.016 : 0.0;
                if (hold <= 0.0)
                    continue;
                const int source = frameOf (b.start + 0.5 * frameSeconds);
                for (int k = frameOf (a.end - hold); k <= frameOf (a.end) && k < numFrames; ++k)
                    if (frameSegment[(size_t) k] == (int) i)
                        for (auto p : formantParams)
                            tracks[(size_t) p][(size_t) k] = tracks[(size_t) p][(size_t) source];
            }
        }
    }

    // Coarticulation: zero-phase smoothing of every parameter.
    std::vector<float> scratch;
    for (int p = 0; p < numArticParams; ++p)
    {
        auto& track = tracks[(size_t) p];
        const bool logDomain = articParamIsFrequency (p);
        if (logDomain)
            for (auto& x : track)
                x = std::log (juce::jmax (1.0f, x));

        gaussianSmooth (track, articSmoothingSeconds (p) / frameSeconds, scratch);

        if (logDomain)
            for (auto& x : track)
                x = std::exp (x);
    }
    gaussianSmooth (dynamics, 0.025 / frameSeconds, scratch);
    gaussianSmooth (rd, 0.03 / frameSeconds, scratch);

    std::vector<float> pitch;
    buildPitchTrack (numFrames, frameSeconds, pitch);

    if (cancelled())
        return false;
    report (0.1f);

    // Synthesis.
    audio.setSize (1, numFrames * frameLength);
    audio.clear();
    auto* out = audio.getWritePointer (0);

    VocalSynth synth;
    synth.prepare (sampleRate, seed);
    synth.setSourceCharacter (setup.source);

    std::vector<float> voicing ((size_t) numFrames, 0.0f), f0Frames ((size_t) numFrames, 0.0f);
    SynthFrame synthFrame;
    for (int k = 0; k < numFrames; ++k)
    {
        for (int p = 0; p < numArticParams; ++p)
            frame.v[(size_t) p] = tracks[(size_t) p][(size_t) k];

        const float g = dynamics[(size_t) k];
        frame.v[apAV] *= g;
        frame.v[apAH] *= g;
        frame.v[apAF] *= g;

        const float f0 = music::midiToHz (pitch[(size_t) k]);
        makeSynthFrame (frame, setup, f0, rd[(size_t) k], synthFrame);
        synth.process (synthFrame, out + (size_t) k * (size_t) frameLength, frameLength);

        voicing[(size_t) k] = tracks[(size_t) apAV][(size_t) k];
        f0Frames[(size_t) k] = f0;

        if ((k & 255) == 0)
        {
            if (cancelled())
                return false;
            report (0.1f + 0.85f * (float) k / (float) numFrames);
        }
    }

    audio.setSize (1, totalSamples, true);

    // Level: peaks around -12 dBFS, gentle fades at the edges.
    const float peak = audio.getMagnitude (0, 0, totalSamples);
    if (peak > 1.0e-6f)
        audio.applyGain (juce::Decibels::decibelsToGain (-12.0f) / peak);

    const int fadeIn = juce::jmin (totalSamples / 4, (int) (0.003 * sampleRate));
    const int fadeOut = juce::jmin (totalSamples / 4, (int) (0.03 * sampleRate));
    if (fadeIn > 0)
        audio.applyGainRamp (0, fadeIn, 0.0f, 1.0f);
    if (fadeOut > 0)
        audio.applyGainRamp (totalSamples - fadeOut, fadeOut, 1.0f, 0.0f);

    // Pitch contour at 100 frames per second (0 where unvoiced).
    const int numF0 = (int) std::ceil (durationSeconds * 100.0);
    f0Hz.assign ((size_t) numF0, 0.0f);
    for (int j = 0; j < numF0; ++j)
    {
        const double t = (j + 0.5) * 0.01;
        const int k = juce::jlimit (0, numFrames - 1, (int) std::lround (t / frameSeconds) - 1);
        if (voicing[(size_t) k] > 0.08f)
            f0Hz[(size_t) j] = f0Frames[(size_t) k];
    }

    report (1.0f);
    return true;
}

} // namespace voxa::sing
