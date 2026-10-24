#pragma once

#include "EditorContext.h"
#include "GlassPanel.h"
#include "Buttons.h"
#include "Knob.h"

namespace voxa::ui
{

/** Tabbed pro panel: Advanced (the four feature toggles + intensity), Expression, Tune,
    Harmony, FX and Space. */
class AdvancedPanel : public GlassPanel
{
public:
    explicit AdvancedPanel (EditorContext&);
    ~AdvancedPanel() override;

    void paintStatic (juce::Graphics&, juce::Rectangle<float>) override;
    void resized() override;
    void tick (float dt);

    void showTab (int index);
    bool showTabNamed (const juce::String& name);
    int getCurrentTab() const noexcept { return currentTab; }

    static const juce::StringArray& getTabNames();

    class Page : public juce::Component
    {
    public:
        virtual void tick (float) {}
    };

private:
    EditorContext& ctx;
    juce::OwnedArray<TabButton> tabs;
    juce::OwnedArray<Page> pages;
    int currentTab = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedPanel)
};

} // namespace voxa::ui
