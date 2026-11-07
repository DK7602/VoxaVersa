#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

/** Runs every registered juce::UnitTest. Usage: VoxaVersaTests [category] */
int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juceInit;

    juce::UnitTestRunner runner;
    runner.setAssertOnFailure (false);

    if (argc > 1)
        runner.runTestsInCategory (argv[1]);
    else
        runner.runAllTests();

    int failures = 0;
    for (int i = 0; i < runner.getNumResults(); ++i)
        if (auto* r = runner.getResult (i))
        {
            failures += r->failures;
            // On Windows the runner's own log goes to the debugger only: print what failed where CI can read it.
            if (r->failures > 0)
            {
                std::printf ("\nFAILED in %s / %s:\n", r->unitTestName.toRawUTF8(), r->subcategoryName.toRawUTF8());
                for (const auto& m : r->messages)
                    std::printf ("  %s\n", m.toRawUTF8());
            }
        }

    std::printf ("\n%s: %d test group(s), %d failure(s)\n", failures == 0 ? "PASSED" : "FAILED",
                 runner.getNumResults(), failures);
    return failures == 0 ? 0 : 1;
}
