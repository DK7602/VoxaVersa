#include "core/MusicTheory.h"

namespace voxa
{

class MusicTheoryTests : public juce::UnitTest
{
public:
    MusicTheoryTests() : juce::UnitTest ("Music theory", "Core") {}

    void runTest() override
    {
        using namespace music;

        beginTest ("Note conversions");
        expectWithinAbsoluteError (midiToHz (69.0f), 440.0f, 0.001f);
        expectWithinAbsoluteError (hzToMidi (261.6256f), 60.0f, 0.001f);
        expectEquals (noteName (60), juce::String ("C4"));

        beginTest ("Scale snapping");
        expectEquals (nearestScaleNote (61.0f, 0, ScaleType::Major), 60);      // C# -> C (tie resolves down)
        expectEquals (nearestScaleNote (60.9f, 0, ScaleType::Major), 60);
        expectEquals (nearestScaleNote (61.6f, 0, ScaleType::Major), 62);
        expectEquals (nearestScaleNote (66.2f, 7, ScaleType::Major), 66);      // F# is in G major
        expectEquals (nearestScaleNote (63.0f, 0, ScaleType::Chromatic), 63);

        beginTest ("Diatonic harmony");
        expectEquals (harmonyNote (60, HarmonyInterval::ThirdUp, 0, ScaleType::Major), 64);   // C -> E
        expectEquals (harmonyNote (62, HarmonyInterval::ThirdUp, 0, ScaleType::Major), 65);   // D -> F (minor third)
        expectEquals (harmonyNote (60, HarmonyInterval::FifthUp, 0, ScaleType::Major), 67);
        expectEquals (harmonyNote (71, HarmonyInterval::FifthUp, 0, ScaleType::Major), 77);   // B -> F (diminished)
        expectEquals (harmonyNote (60, HarmonyInterval::ThirdDown, 0, ScaleType::Major), 57);
        expectEquals (harmonyNote (60, HarmonyInterval::OctaveUp, 0, ScaleType::Major), 72);
        expectEquals (harmonyNote (60, HarmonyInterval::FifthUp, 0, ScaleType::MajorPentatonic), 67);
        expectEquals (harmonyNote (60, HarmonyInterval::Off, 0, ScaleType::Major), -1);

        beginTest ("Key detection");
        KeyDetector kd;
        for (int rep = 0; rep < 20; ++rep)
            for (int n : { 67, 69, 71, 72, 74, 76, 78, 79, 67, 74, 71 })   // G major melody
                kd.addPitch ((float) n, 1.0f);
        kd.update();
        expectEquals (kd.getKeyRoot(), 7);
        expect (kd.getScale() == ScaleType::Major);
    }
};

static MusicTheoryTests musicTheoryTests;

} // namespace voxa
