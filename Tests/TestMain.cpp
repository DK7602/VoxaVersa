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
            failures += r->failures;

    std::printf ("\n%s: %d test group(s), %d failure(s)\n", failures == 0 ? "PASSED" : "FAILED",
                 runner.getNumResults(), failures);
    return failures == 0 ? 0 : 1;
}
