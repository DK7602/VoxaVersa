#include "ui/UserManual.h"
#include "ui/ManualView.h"
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

        beginTest ("The manual shows inside VoxaVersa: every section listed and reachable");
        {
            ui::ManualView view;
            view.setSize (1500, 905);
            expectGreaterThan (view.getNumBlocks(), 100);
            const auto html = ui::UserManual::getHtml();
            int sectionsInPage = 0;
            for (int i = html.indexOf ("<section id="); i >= 0; i = html.indexOf (i + 1, "<section id="))
                ++sectionsInPage;
            expectEquals ((int) view.getSections().size(), sectionsInPage);
            for (const auto& [id, title] : view.getSections())
            {
                expect (title.isNotEmpty(), id);
                expect (view.showSection (id), id);
            }
            expect (! view.showSection ("no-such-section"));
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
