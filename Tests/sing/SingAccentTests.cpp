#include "sing/Lyrics.h"
#include "sing/Melody.h"
#include "sing/OfflineRenderer.h"
#include "voices/VoiceLibrary.h"
#include <tuple>

namespace voxa::sing
{

/** General American pronunciation: connected speech, allophones and stress (the guide vocal must not
    sound like a non-native singer). */
class SingAccentTests : public juce::UnitTest
{
public:
    SingAccentTests() : juce::UnitTest ("Text to Sing: American English", "Sing") {}

    void runTest() override
    {
        const auto voices = createFactoryVoices();
        const auto& male = voices.front().traits;

        beginTest ("Weak forms of function words; \"the\" before a vowel");
        {
            const auto s = parseLyrics ("go to the end of the night for you");
            expect (s.syllables[1].hasWeakForm && s.syllables[1].weakNucleus == Ph::AX);      // to
            expect (s.syllables[1].weakOnlyBeforeConsonant);
            expect (s.syllables[2].nucleus == Ph::IY);                                       // the end
            expect (s.syllables[4].hasWeakForm && s.syllables[4].weakCoda == std::vector<Ph> { Ph::V });   // of
            expect (s.syllables[5].nucleus == Ph::AX);                                       // the night
            expect (s.syllables[7].hasWeakForm && s.syllables[7].weakNucleus == Ph::ER         // for -> [fɚ]
                    && s.syllables[7].weakCoda.empty());
            expect (! s.syllables[0].hasWeakForm && ! s.syllables[6].hasWeakForm);
        }

        beginTest ("Weak forms are sung on short notes only");
        {
            expect (nucleusOf (plan ("I want to go", 0.5, 120.0, male), 2) == Ph::AX);      // 0.25 s: [tə]
            expect (nucleusOf (plan ("I want to go", 2.0, 120.0, male), 2) == Ph::UW);      // 1 s: [tu]
            expect (nucleusOf (plan ("I want to ask", 0.5, 120.0, male), 2) == Ph::UW);     // before a vowel
        }

        beginTest ("Flaps: T / D between vowels before an unstressed vowel, and across words");
        {
            const auto better = plan ("better", 1.0, 100.0, male);
            const auto* t = findSegment (better, Ph::T);
            expect (t != nullptr && t->event.tap && t->end - t->start < 0.04);

            const auto party = plan ("party", 1.0, 100.0, male);
            expect (findSegment (party, Ph::T) != nullptr && findSegment (party, Ph::T)->event.tap);

            const auto gotIt = plan ("got it", 1.0, 100.0, male);
            const auto* coda = findSegment (gotIt, Ph::T);
            expect (coda != nullptr && coda->event.coda && coda->event.tap);

            // Not before a stressed vowel, not word-initially: aspirated stops.
            for (auto word : { "tonight", "attack" })
            {
                const auto r = plan (word, 1.0, 100.0, male);
                const auto* stop = findSegment (r, Ph::T);
                expect (stop != nullptr && ! stop->event.tap && stop->event.aspirated, word);
            }

            // The voice carries on through the flap.
            auto r = plan ("better", 1.0, 100.0, male);
            expect (r.render (nullptr, {}));
            const auto* tap = findSegment (r, Ph::T);
            int voiced = 0, frames = 0;
            for (double x = tap->start; x < tap->end; x += 0.01, ++frames)
                voiced += r.f0Hz[(size_t) juce::jlimit (0, (int) r.f0Hz.size() - 1, (int) (x * 100.0))] > 0.0f ? 1 : 0;
            expect (frames > 0 && voiced == frames, "voicing must continue through a flap");
        }

        beginTest ("No yod coalescence: can't you, hold you keep their T / D (unreleased) and Y");
        {
            for (auto [lyric, stop, merged] : { std::tuple<const char*, Ph, Ph> { "can't you", Ph::T, Ph::CH },
                                                { "hold you", Ph::D, Ph::JH } })
            {
                const auto r = plan (lyric, 1.0, 100.0, male);
                const auto* s = findSegment (r, stop);
                expect (s != nullptr && ! s->event.released, juce::String (lyric) + ": unreleased stop");
                expect (findSegment (r, Ph::Y) != nullptr && findSegment (r, merged) == nullptr, lyric);
            }
            // other stops still release into the glide
            const auto* k = findSegment (plan ("back you", 1.0, 100.0, male), Ph::K);
            expect (k != nullptr && k->event.released);
        }

        beginTest ("Final stops: soft before a pause, unreleased before a consonant, linked into a vowel");
        {
            const auto had = plan ("had", 1.0, 100.0, male);
            const auto* d = findSegment (had, Ph::D);
            expect (d != nullptr && d->event.coda && d->event.released && ! d->event.linked);

            const auto holdOn = plan ("hold on", 1.0, 100.0, male);
            expect (findSegment (holdOn, Ph::D)->event.linked);

            const auto goodNight = plan ("good night", 1.0, 100.0, male);
            expect (! findSegment (goodNight, Ph::D)->event.released);

            // The soft release of a final voiced stop has no voiced "-uh" after the burst.
            VoiceSetup setup;
            setup.prepare (male);
            PhoneEvent e;
            e.ph = Ph::D;
            e.coda = true;
            e.duration = 0.08f;
            ArticFrame frame;
            articulate (e, 0.077f, setup, frame);
            expectEquals (frame.v[apAV], 0.0f);
            e.linked = true;
            articulate (e, 0.077f, setup, frame);
            expect (frame.v[apAV] > 0.2f);
        }

        beginTest ("Aspiration only where English aspirates");
        {
            expect (! findSegment (plan ("happy", 1.0, 100.0, male), Ph::P)->event.aspirated);
            expect (findSegment (plan ("pay", 1.0, 100.0, male), Ph::P)->event.aspirated);
            expect (! findSegment (plan ("stay", 1.0, 100.0, male), Ph::T)->event.aspirated);
        }

        beginTest ("Detached singing never breaks a word with silence");
        {
            auto detached = male;
            detached.singLegato = 0.0f;
            const auto r = plan ("never love me", 1.0, 90.0, detached);
            const auto* eh = findSegment (r, Ph::EH);
            const auto* v = findSegment (r, Ph::V);           // ne-ver
            expect (eh != nullptr && v != nullptr && std::abs (v->start - eh->end) < 1.0e-6);

            const Segment* loveV = nullptr;                   // love | me: a gap between the words
            const Segment* m = nullptr;
            for (const auto& s : r.getSegments())
            {
                if (s.event.ph == Ph::V && s.event.coda)
                    loveV = &s;
                if (s.event.ph == Ph::M)
                    m = &s;
            }
            expect (loveV != nullptr && m != nullptr && m->start - loveV->end > 0.005);
        }

        beginTest ("American vowel targets and off-glides");
        {
            VoiceSetup setup;
            setup.prepare (male);
            auto formants = [&setup] (Ph vowel, float glide, bool contextual = false, Ph liquid = Ph::Sil)
            {
                std::array<float, 3> f {};
                setup.vowelFormants (vowel, glide, contextual, liquid, f.data());
                return f;
            };

            // /ow/ and /ey/ are diphthongs, not pure [o] and [e].
            expect (formants (Ph::OW, 0.0f)[0] - formants (Ph::OW, 1.0f)[0] > 120.0f);
            expect (formants (Ph::EY, 0.0f)[0] - formants (Ph::EY, 1.0f)[0] > 120.0f);
            expect (formants (Ph::EY, 1.0f)[1] - formants (Ph::EY, 0.0f)[1] > 250.0f);

            // /ae/ is a front vowel, raised before nasals; /ao/ is more open than /ow/; /u/ fronts after coronals.
            expect (formants (Ph::AE, 0.0f)[1] > 1800.0f);
            expect (formants (Ph::AE, 0.0f, true)[0] < formants (Ph::AE, 0.0f)[0]);
            expect (formants (Ph::AO, 0.0f)[0] > formants (Ph::OW, 0.0f)[0] + 60.0f);
            expect (formants (Ph::UW, 0.0f, true)[1] > formants (Ph::UW, 0.0f)[1] + 200.0f);

            // A coda R lowers F3 at the end of the vowel; a coda L lowers F2.
            expect (formants (Ph::IY, 1.0f, false, Ph::R)[2] < formants (Ph::IY, 0.0f)[2] - 800.0f);
            expect (formants (Ph::IY, 1.0f, false, Ph::L)[1] < formants (Ph::IY, 0.0f)[1] - 800.0f);

            // The off-glide comes late but completes before the vowel ends.
            const auto go = plan ("go home", 4.0, 100.0, male);
            const auto* ow = findSegment (go, Ph::OW);
            expect (ow != nullptr);
            const float glideEnd = ow->event.glideStart + ow->event.glideTime;
            expect (ow->event.glideStart > 0.6f * ow->event.duration);
            expect (glideEnd < ow->event.duration - 0.01f && glideEnd > ow->event.duration - 0.05f);

            const auto here = plan ("here", 4.0, 100.0, male);
            expect (findSegment (here, Ph::IY)->event.liquidCoda == Ph::R);
            const auto you = plan ("you", 4.0, 100.0, male);
            expect (findSegment (you, Ph::UW)->event.contextual);
        }

        beginTest ("Auto melody: the stressed syllable of a feminine ending gets the long note");
        {
            const auto lyrics = parseLyrics ("Hold on to the feeling\nEvery little thing is magic\nKeep us together");
            int phrases = 0, stressedLonger = 0;
            for (juce::uint32 seed = 1; seed <= 10; ++seed)
            {
                const auto score = melody::compose (lyrics, male, 0, ScaleType::Major, seed);
                for (size_t i = 1; i < score.notes.size(); ++i)
                {
                    const bool phraseEnd = i + 1 == score.notes.size() || score.notes[i + 1].breathBefore;
                    if (! phraseEnd)
                        continue;
                    const auto& last = score.notes[i];
                    const auto& penult = score.notes[i - 1];
                    const auto& weak = lyrics.syllables[(size_t) last.syllable];
                    const auto& strong = lyrics.syllables[(size_t) penult.syllable];
                    if (last.continuation || penult.continuation || weak.stress != 0 || strong.stress != 1
                        || weak.wordIndex != strong.wordIndex)
                        continue;
                    ++phrases;
                    stressedLonger += penult.lengthBeats >= last.lengthBeats ? 1 : 0;
                    expect (last.lengthBeats <= 1.5 + 1.0e-9, "weak ending is moderate");
                }
            }
            logMessage ("stressed syllable at least as long as the weak ending in " + juce::String (stressedLonger)
                        + " of " + juce::String (phrases) + " phrases");
            expect (stressedLonger * 10 >= phrases * 8);
        }
    }

private:
    using Segment = OfflineRenderer::Segment;

    /** Plans the lyrics with every syllable on its own `beats`-long note, joined legato. */
    static OfflineRenderer plan (const juce::String& lyrics, double beats, double bpm, const VoiceTraits& traits)
    {
        static std::vector<std::unique_ptr<std::pair<LyricScript, Score>>> keepAlive;   // the renderer keeps references
        auto data = std::make_unique<std::pair<LyricScript, Score>>();
        data->first = parseLyrics (lyrics);
        double beat = 1.0;
        for (int i = 0; i < data->first.getNumSyllables(); ++i)
        {
            ScoreNote n;
            n.startBeat = beat;
            n.lengthBeats = beats;
            n.midiNote = 55;
            n.syllable = i;
            n.breathBefore = i == 0;
            data->second.notes.push_back (n);
            beat += beats;
        }
        keepAlive.push_back (std::move (data));
        return OfflineRenderer (keepAlive.back()->first, keepAlive.back()->second, traits, bpm, 48000.0, 1);
    }

    static const Segment* findSegment (const OfflineRenderer& r, Ph ph)
    {
        for (const auto& s : r.getSegments())
            if (s.event.ph == ph)
                return &s;
        return nullptr;
    }

    /** Nucleus of the n-th syllable as planned. */
    static Ph nucleusOf (const OfflineRenderer& r, int n)
    {
        int count = 0;
        for (const auto& s : r.getSegments())
            if (s.nucleus && count++ == n)
                return s.event.ph;
        return Ph::Count;
    }
};

static SingAccentTests singAccentTests;

} // namespace voxa::sing
