#pragma once

#include "Theme.h"
#include "Icons.h"
#include <optional>

namespace voxa::ui
{

/** A frosted-glass card. The glass itself is rendered into the editor's cached background
    (see MainView); the component paints only dynamic emphasis on top. */
class GlassPanel : public juce::Component
{
public:
    explicit GlassPanel (const juce::String& title, std::optional<Icon> titleIcon = std::nullopt);

    const juce::String& getPanelTitle() const noexcept { return title; }

    /** Static decoration drawn into the cached background (local coordinates): title row etc. */
    virtual void paintStatic (juce::Graphics& g, juce::Rectangle<float> area);

    /** Brighter glowing border (e.g. Text to Sing in sing mode). */
    void setEmphasis (bool shouldEmphasise);
    bool hasEmphasis() const noexcept { return emphasis; }

    void paint (juce::Graphics&) override;

    static constexpr float titleHeight = 46.0f;
    static constexpr float cornerRadius = 14.0f;

    /** Paints a frosted glass card. `backdrop` (optional) is a blurred copy of what lies behind,
        drawn so that its pixel (0, 0) lands at `backdropOrigin` in the current coordinate space. */
    static void paintGlassCard (juce::Graphics& g, juce::Rectangle<float> area, const juce::Image* backdrop,
                                juce::Point<float> backdropOrigin, float backdropScale, float corner = cornerRadius);

protected:
    juce::Rectangle<int> getContentBounds() const { return getLocalBounds().withTrimmedTop ((int) titleHeight); }
    void paintTitle (juce::Graphics& g, juce::Rectangle<float> titleArea);

    juce::String title;
    std::optional<Icon> titleIcon;
    bool emphasis = false;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GlassPanel)
};

} // namespace voxa::ui
