#pragma once

#include "Theme.h"
#include <juce_audio_processors/juce_audio_processors.h>

namespace voxa::ui
{

//==============================================================================
/** Editable value read-out linked to a slider (double-click to type a value). */
class ValueBox : public juce::Label
{
public:
    explicit ValueBox (juce::Slider& slider);
    void update();
    void paint (juce::Graphics&) override;
    void setFontHeight (float h);

private:
    void textWasEdited() override;
    juce::Slider& slider;
};

//==============================================================================
/** A chrome knob bound to a parameter, with name, range captions and value read-outs. */
class Knob : public juce::Component
{
public:
    enum class Size { Large, Medium, Small };

    struct Options
    {
        Size size = Size::Medium;
        juce::String name;
        juce::String leftCaption, rightCaption;
        bool valueAbove = false;     // live value text above the knob
        bool valueBox = false;       // editable value box under the name
        bool valueBelow = false;     // small value text under the name (small knobs)
        juce::Colour accent { Palette::glow };
        std::function<juce::String (double)> formatter;   // overrides the parameter's text
        juce::String tooltip;
    };

    Knob (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, Options options);
    ~Knob() override;

    void paint (juce::Graphics&) override;
    void resized() override;

    juce::Slider& getSlider() noexcept { return slider; }
    juce::Rectangle<int> getKnobBounds() const noexcept { return slider.getBounds(); }

private:
    juce::String getValueText();

    Options options;
    juce::Slider slider;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
    std::unique_ptr<ValueBox> valueBox;
    juce::Rectangle<float> valueAboveArea, leftCaptionArea, rightCaptionArea, nameArea, valueBelowArea;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Knob)
};

//==============================================================================
/** Numeric box with up/down arrows, drag and wheel editing (used for the sing tempo). */
class NumberBox : public juce::Component,
                  public juce::SettableTooltipClient,
                  private juce::Timer
{
public:
    NumberBox (juce::RangedAudioParameter& parameter, juce::UndoManager* undoManager, const juce::String& suffix = {});
    ~NumberBox() override;

    /** Shows a read-only value instead of the parameter (e.g. the host tempo when synced). */
    void setOverride (std::optional<double> value);

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseDoubleClick (const juce::MouseEvent&) override;
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails&) override;
    bool keyPressed (const juce::KeyPress&) override;
    std::unique_ptr<juce::AccessibilityHandler> createAccessibilityHandler() override;

private:
    void timerCallback() override;
    void step (int direction);
    void setValue (double v, bool asGesture);
    double getValue() const;

    juce::RangedAudioParameter& param;
    juce::ParameterAttachment attachment;
    juce::String suffix;
    double value = 0.0, dragStartValue = 0.0;
    std::optional<double> overrideValue;
    juce::Rectangle<float> upArea, downArea, textArea;
    int heldDirection = 0, repeatCount = 0;
    bool dragging = false;
    std::unique_ptr<juce::TextEditor> editor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NumberBox)
};

} // namespace voxa::ui
