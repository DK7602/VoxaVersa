#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>
#include <memory>
#include <vector>

namespace voxa::ui
{

/** The user manual inside VoxaVersa's own window (menu > User Manual): the built-in manual page (UserManual)
    drawn natively in the editor's colours, its sections listed on the left, the text scrolling on the right.
    It reads the page's <main>: headings, paragraphs, lists, tables, notes, tips and the FAQ, with bold, italic,
    code and links inside them. "Open in Browser" still opens the page itself; Close or Escape hides it. */
class ManualView : public juce::Component,
                   private juce::ScrollBar::Listener
{
public:
    ManualView();
    ~ManualView() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    bool keyPressed (const juce::KeyPress&) override;

    /** Shows the top of the section with this id ("styles", "faq"...); false if there is none. */
    bool showSection (const juce::String& id);

    /** Called by Close / Escape (the owner hides the view). */
    std::function<void()> onClose;

    /** One laid-out piece of the manual (exposed for tests). */
    struct Block;
    /** The manual's sections, in order: id and title. */
    const std::vector<std::pair<juce::String, juce::String>>& getSections() const noexcept { return sections; }
    /** How many blocks the manual parsed into (0: the page is missing or unreadable). */
    int getNumBlocks() const noexcept;

private:
    void scrollBarMoved (juce::ScrollBar*, double) override;

    class Content;
    class SectionList;

    std::unique_ptr<Content> content;
    std::unique_ptr<SectionList> sectionList;
    juce::Viewport viewport;
    juce::TextButton closeButton { "Close" }, browserButton { "Open in Browser" };
    std::vector<std::pair<juce::String, juce::String>> sections;
    juce::String version;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ManualView)
};

} // namespace voxa::ui
