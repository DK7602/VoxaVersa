#include "ui/UserManual.h"
#include "voices/VoiceLibrary.h"

#include <juce_core/juce_core.h>

namespace voxa
{

class UserManualTests : public juce::UnitTest
{
public:
    UserManualTests() : juce::UnitTest ("User manual", "Core") {}

    void runTest() override
    {
        beginTest ("The manual is built in, with this build's version, and opens from a written file");
        {
            const auto html = ui::UserManual::getHtml();
            expect (html.contains ("<title>VoxaVersa User Manual</title>"));
            expect (! html.contains ("{{VERSION}}"));
            expect (html.contains (VOXAVERSA_VERSION_STRING));

            const auto dir = juce::File::createTempFile ("manual");
            const auto file = ui::UserManual::write (dir);
            expect (file.existsAsFile());
            expectEquals (file.loadFileAsString(), html);
            expect (ui::UserManual::write (dir) == file);   // unchanged: written once
            dir.deleteRecursively();
        }

        beginTest ("The manual names every voice style and specialty voice");
        {
            const auto html = ui::UserManual::getHtml();
            VoiceLibrary library;
            for (const auto& v : library.getVoices())
            {
                if (! v.isFactory)
                    continue;
                for (const auto& sound : v.sounds)
                    expect (html.contains ("<b>" + sound.name + "</b>"), v.name + ": style " + sound.name + " missing from the manual");
                if (v.genre == "Specialty" || v.id.startsWith ("neutral-"))
                    expect (html.contains (v.name), v.name + " missing from the manual");
            }
        }
    }
};

static UserManualTests userManualTests;

} // namespace voxa
