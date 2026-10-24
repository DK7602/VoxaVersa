#pragma once

#include "Theme.h"
#include <map>

namespace voxa::ui
{

/** Slider property keys understood by VoxaLookAndFeel. */
namespace SliderProps
{
    inline constexpr const char* knobStyle = "voxaKnobStyle";   // "large" | "medium" | "small"
    inline constexpr const char* accent    = "voxaAccent";      // ARGB int
    inline constexpr const char* bipolar   = "voxaBipolar";     // bool
}

/** Glassmorphism look: chrome knobs with glowing arcs, navy glass controls, blue glow states.
    Shared between editor instances (use through juce::SharedResourcePointer). */
class VoxaLookAndFeel : public juce::LookAndFeel_V4
{
public:
    VoxaLookAndFeel();
    ~VoxaLookAndFeel() override;

    /** Cached chrome knob cap at a physical pixel diameter. */
    const juce::Image& getChromeCap (int diameterPx);

    //==============================================================================
    juce::Typeface::Ptr getTypefaceForFont (const juce::Font&) override;

    void drawRotarySlider (juce::Graphics&, int x, int y, int width, int height, float sliderPos,
                           float rotaryStartAngle, float rotaryEndAngle, juce::Slider&) override;
    void drawLinearSlider (juce::Graphics&, int x, int y, int width, int height, float sliderPos,
                           float minSliderPos, float maxSliderPos, juce::Slider::SliderStyle, juce::Slider&) override;
    int getSliderThumbRadius (juce::Slider&) override;
    juce::Label* createSliderTextBox (juce::Slider&) override;

    void drawButtonBackground (juce::Graphics&, juce::Button&, const juce::Colour& backgroundColour,
                               bool isHighlighted, bool isDown) override;
    void drawButtonText (juce::Graphics&, juce::TextButton&, bool isHighlighted, bool isDown) override;
    juce::Font getTextButtonFont (juce::TextButton&, int buttonHeight) override;
    void drawToggleButton (juce::Graphics&, juce::ToggleButton&, bool isHighlighted, bool isDown) override;

    void drawComboBox (juce::Graphics&, int width, int height, bool isButtonDown, int buttonX, int buttonY,
                       int buttonW, int buttonH, juce::ComboBox&) override;
    juce::Font getComboBoxFont (juce::ComboBox&) override;
    void positionComboBoxText (juce::ComboBox&, juce::Label&) override;
    juce::PopupMenu::Options getOptionsForComboBoxPopupMenu (juce::ComboBox&, juce::Label&) override;

    void drawPopupMenuBackground (juce::Graphics&, int width, int height) override;
    void drawPopupMenuItem (juce::Graphics&, const juce::Rectangle<int>& area, bool isSeparator, bool isActive,
                            bool isHighlighted, bool isTicked, bool hasSubMenu, const juce::String& text,
                            const juce::String& shortcutKeyText, const juce::Drawable* icon,
                            const juce::Colour* textColour) override;
    void drawPopupMenuSectionHeader (juce::Graphics&, const juce::Rectangle<int>&, const juce::String&) override;
    juce::Font getPopupMenuFont() override;
    void getIdealPopupMenuItemSize (const juce::String& text, bool isSeparator, int standardMenuItemHeight,
                                    int& idealWidth, int& idealHeight) override;
    int getPopupMenuBorderSize() override { return 4; }

    void drawTooltip (juce::Graphics&, const juce::String& text, int width, int height) override;
    juce::Rectangle<int> getTooltipBounds (const juce::String& tipText, juce::Point<int> screenPos,
                                           juce::Rectangle<int> parentArea) override;

    void fillTextEditorBackground (juce::Graphics&, int width, int height, juce::TextEditor&) override;
    void drawTextEditorOutline (juce::Graphics&, int width, int height, juce::TextEditor&) override;

    void drawScrollbar (juce::Graphics&, juce::ScrollBar&, int x, int y, int width, int height, bool isVertical,
                        int thumbStartPosition, int thumbSize, bool isMouseOver, bool isMouseDown) override;
    int getDefaultScrollbarWidth() override { return 8; }

    void drawAlertBox (juce::Graphics&, juce::AlertWindow&, const juce::Rectangle<int>& textArea, juce::TextLayout&) override;
    juce::Font getAlertWindowTitleFont() override;
    juce::Font getAlertWindowMessageFont() override;
    juce::Font getAlertWindowFont() override;
    int getAlertWindowButtonHeight() override { return 32; }

    juce::Font getLabelFont (juce::Label&) override;

    void drawCallOutBoxBackground (juce::CallOutBox&, juce::Graphics&, const juce::Path&, juce::Image&) override;
    void drawCornerResizer (juce::Graphics&, int w, int h, bool isMouseOver, bool isMouseDragging) override;

private:
    const juce::TextLayout& getTooltipLayout (const juce::String& text);

    std::map<int, juce::Image> chromeCaps;
    juce::String tooltipText;
    juce::TextLayout tooltipLayout;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoxaLookAndFeel)
};

} // namespace voxa::ui
