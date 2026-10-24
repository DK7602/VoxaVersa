#include "Knob.h"
#include "VoxaLookAndFeel.h"
#include "Icons.h"

namespace voxa::ui
{

//==============================================================================
ValueBox::ValueBox (juce::Slider& s) : slider (s)
{
    setEditable (false, true, false);
    setJustificationType (juce::Justification::centred);
    setFont (ui::font (17.0f, Weight::SemiBold));
    setColour (juce::Label::textColourId, Palette::text);
    setTitle (s.getTitle() + " value");
    setTooltip ("Double-click to type a value");
    update();
}

void ValueBox::setFontHeight (float h)
{
    setFont (ui::font (h, Weight::SemiBold));
}

void ValueBox::update()
{
    setText (slider.getTextFromValue (slider.getValue()), juce::dontSendNotification);
}

void ValueBox::textWasEdited()
{
    const auto v = slider.getValueFromText (getText());
    slider.setValue (v, juce::sendNotificationSync);
    update();
}

void ValueBox::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat();
    fillInset (g, r.reduced (0.5f), 6.0f);

    if (! isBeingEdited())
    {
        g.setFont (getFont());
        g.setColour (findColour (juce::Label::textColourId));
        g.drawText (getText(), r.reduced (4.0f, 0.0f), juce::Justification::centred, true);
    }
}

//==============================================================================
Knob::Knob (juce::AudioProcessorValueTreeState& state, const juce::String& paramId, Options o)
    : options (std::move (o))
{
    slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    slider.setRotaryParameters (juce::MathConstants<float>::pi * 1.25f, juce::MathConstants<float>::pi * 2.75f, true);
    slider.setMouseDragSensitivity (options.size == Size::Small ? 180 : 240);
    slider.setScrollWheelEnabled (true);
    slider.setMouseCursor (juce::MouseCursor::UpDownResizeCursor);
    slider.getProperties().set (SliderProps::knobStyle, options.size == Size::Large ? "large"
                                                         : options.size == Size::Small ? "small" : "medium");
    slider.getProperties().set (SliderProps::accent, (juce::int64) options.accent.getARGB());
    addAndMakeVisible (slider);

    slider.onValueChange = [this]
    {
        if (valueBox != nullptr)
            valueBox->update();
        if (options.valueAbove || options.valueBelow)
            repaint();
    };
    slider.onDragStart = [this] { repaint(); };
    slider.onDragEnd = [this] { repaint(); };

    attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (state, paramId, slider);

    if (auto* p = state.getParameter (paramId))
    {
        slider.setDoubleClickReturnValue (true, p->convertFrom0to1 (p->getDefaultValue()));
        if (options.name.isEmpty())
            options.name = p->getName (64);
    }

    slider.setTitle (options.name);
    setTitle (options.name);
    juce::String tip = options.tooltip.isNotEmpty() ? options.tooltip : options.name;
    if (options.leftCaption.isNotEmpty())
        tip << "\n" << options.leftCaption << " \xe2\x86\x94 " << options.rightCaption;
    tip << "\nDouble-click to reset";
    slider.setTooltip (tip);
    slider.setDescription (options.tooltip);

    if (options.valueBox)
    {
        valueBox = std::make_unique<ValueBox> (slider);
        addAndMakeVisible (*valueBox);
    }

    setFocusContainerType (FocusContainerType::none);
}

Knob::~Knob()
{
    slider.onValueChange = nullptr;
}

juce::String Knob::getValueText()
{
    if (options.formatter)
        return options.formatter (slider.getValue());
    return slider.getTextFromValue (slider.getValue());
}

void Knob::resized()
{
    const auto b = getLocalBounds().toFloat();
    const float w = b.getWidth(), h = b.getHeight();

    switch (options.size)
    {
        case Size::Large:
        {
            const float boxH = options.valueBox ? 30.0f : 0.0f;
            const float d = juce::jmin (w * 0.66f, h - 24.0f - boxH - 6.0f);
            auto knob = juce::Rectangle<float> (d, d).withCentre ({ b.getCentreX(), d * 0.5f });
            slider.setBounds (knob.toNearestInt());
            leftCaptionArea  = juce::Rectangle<float> (44.0f, 18.0f).withCentre ({ knob.getCentreX() - d * 0.52f, knob.getBottom() - d * 0.04f });
            rightCaptionArea = juce::Rectangle<float> (44.0f, 18.0f).withCentre ({ knob.getCentreX() + d * 0.52f, knob.getBottom() - d * 0.04f });
            nameArea = { b.getX(), knob.getBottom() + 2.0f, w, 22.0f };
            if (valueBox != nullptr)
                valueBox->setBounds (juce::Rectangle<float> (94.0f, boxH - 2.0f).withCentre ({ b.getCentreX(), nameArea.getBottom() + 4.0f + boxH * 0.5f }).toNearestInt());
            break;
        }

        case Size::Medium:
        {
            const float aboveH = options.valueAbove ? 18.0f : 0.0f;
            const float d = juce::jmin (w * 0.82f, h - aboveH - 38.0f);
            valueAboveArea = { b.getX(), 0.0f, w, aboveH };
            auto knob = juce::Rectangle<float> (d, d).withCentre ({ b.getCentreX(), aboveH + d * 0.5f });
            slider.setBounds (knob.toNearestInt());
            const float capY = knob.getBottom() - 1.0f;
            leftCaptionArea  = { b.getX() + 5.0f, capY, w * 0.5f - 5.0f, 16.0f };
            rightCaptionArea = { b.getCentreX(), capY, w * 0.5f - 5.0f, 16.0f };
            nameArea = { b.getX(), capY + 15.0f, w, 21.0f };
            break;
        }

        case Size::Small:
        {
            const float belowH = options.valueBelow ? 14.0f : 0.0f;
            const float d = juce::jmin (w * 0.9f, h - 16.0f - belowH);
            auto knob = juce::Rectangle<float> (d, d).withCentre ({ b.getCentreX(), d * 0.5f });
            slider.setBounds (knob.toNearestInt());
            nameArea = { b.getX(), knob.getBottom(), w, 16.0f };
            valueBelowArea = { b.getX(), nameArea.getBottom() - 1.0f, w, belowH };
            break;
        }
    }
}

void Knob::paint (juce::Graphics& g)
{
    const bool dragging = slider.isMouseButtonDown();
    const float alpha = isEnabled() ? 1.0f : 0.45f;

    switch (options.size)
    {
        case Size::Large:
            drawText (g, options.leftCaption, leftCaptionArea, font (15.5f), Palette::textSoft.withMultipliedAlpha (alpha));
            drawText (g, options.rightCaption, rightCaptionArea, font (15.5f), Palette::textSoft.withMultipliedAlpha (alpha));
            drawText (g, options.name, nameArea, font (18.5f, Weight::Medium), Palette::text.withMultipliedAlpha (alpha));
            break;

        case Size::Medium:
            if (options.valueAbove)
                drawText (g, getValueText(), valueAboveArea, font (15.0f, Weight::Medium),
                          (dragging ? juce::Colours::white : Palette::textSoft).withMultipliedAlpha (alpha));
            drawText (g, options.leftCaption, leftCaptionArea, font (14.0f), Palette::textSoft.withMultipliedAlpha (alpha),
                      juce::Justification::centredLeft);
            drawText (g, options.rightCaption, rightCaptionArea, font (14.0f), Palette::textSoft.withMultipliedAlpha (alpha),
                      juce::Justification::centredRight);
            drawText (g, options.name, nameArea, font (18.0f, Weight::Medium), Palette::text.withMultipliedAlpha (alpha));
            break;

        case Size::Small:
            drawText (g, options.name, nameArea, font (14.0f, Weight::Medium), Palette::textSoft.withMultipliedAlpha (alpha));
            if (options.valueBelow)
                drawText (g, getValueText(), valueBelowArea, font (13.0f),
                          (dragging ? Palette::glowBright : Palette::textDim).withMultipliedAlpha (alpha));
            break;
    }
}

//==============================================================================
NumberBox::NumberBox (juce::RangedAudioParameter& p, juce::UndoManager* um, const juce::String& s)
    : param (p),
      attachment (p, [this] (float v) { value = v; repaint(); }, um),
      suffix (s)
{
    setWantsKeyboardFocus (true);
    setTitle (p.getName (64));
    setTooltip (p.getName (64) + "\nDrag, scroll or use the arrows. Double-click to type.");
    attachment.sendInitialUpdate();
}

NumberBox::~NumberBox() = default;

void NumberBox::setOverride (std::optional<double> v)
{
    if (overrideValue != v)
    {
        overrideValue = v;
        repaint();
    }
}

double NumberBox::getValue() const
{
    return value;
}

void NumberBox::setValue (double v, bool asGesture)
{
    const auto& range = param.getNormalisableRange();
    v = range.snapToLegalValue ((float) juce::jlimit ((double) range.start, (double) range.end, v));
    if (asGesture)
        attachment.setValueAsCompleteGesture ((float) v);
    else
        attachment.setValueAsPartOfGesture ((float) v);
}

void NumberBox::step (int direction)
{
    if (! overrideValue)
        setValue (std::round (value) + direction, true);
}

void NumberBox::resized()
{
    auto r = getLocalBounds().toFloat();
    auto arrows = r.removeFromRight (20.0f);
    upArea = arrows.removeFromTop (arrows.getHeight() * 0.5f);
    downArea = arrows;
    textArea = r;
}

void NumberBox::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat().reduced (0.5f);
    drawGlassControl (g, r, 7.0f, false, isMouseOver (true), false, isEnabled());

    if (hasKeyboardFocus (false))
    {
        g.setColour (Palette::glow.withAlpha (0.8f));
        g.drawRoundedRectangle (r.reduced (0.5f), 7.0f, 1.2f);
    }

    const double shown = overrideValue ? *overrideValue : value;
    const auto txt = juce::String (juce::roundToInt (shown)) + suffix;
    if (editor == nullptr)
        drawText (g, txt, textArea.withTrimmedLeft (10.0f), font (17.0f, Weight::Medium),
                  overrideValue ? Palette::glowBright : Palette::text, juce::Justification::centredLeft);

    const auto arrowCol = overrideValue ? Palette::textFaint : Palette::textSoft;
    Icons::draw (g, Icon::ChevronUp, upArea.withSizeKeepingCentre (11.0f, 11.0f).translated (-3.0f, 2.0f), arrowCol, 2.4f);
    Icons::draw (g, Icon::ChevronDown, downArea.withSizeKeepingCentre (11.0f, 11.0f).translated (-3.0f, -2.0f), arrowCol, 2.4f);
}

void NumberBox::mouseDown (const juce::MouseEvent& e)
{
    if (overrideValue)
        return;

    const auto p = e.position;
    if (upArea.contains (p) || downArea.contains (p))
    {
        heldDirection = upArea.contains (p) ? 1 : -1;
        repeatCount = 0;
        step (heldDirection);
        startTimer (400);
        return;
    }

    dragging = true;
    dragStartValue = value;
    attachment.beginGesture();
}

void NumberBox::mouseDrag (const juce::MouseEvent& e)
{
    if (! dragging || overrideValue)
        return;

    const double delta = -e.getDistanceFromDragStartY() / 4.0;
    attachment.setValueAsPartOfGesture ((float) juce::jlimit ((double) param.getNormalisableRange().start,
                                                              (double) param.getNormalisableRange().end,
                                                              std::round (dragStartValue + delta)));
}

void NumberBox::mouseUp (const juce::MouseEvent&)
{
    stopTimer();
    heldDirection = 0;
    if (dragging)
    {
        dragging = false;
        attachment.endGesture();
    }
}

void NumberBox::timerCallback()
{
    if (heldDirection == 0)
    {
        stopTimer();
        return;
    }
    ++repeatCount;
    step (heldDirection * (repeatCount > 12 ? 5 : 1));
    startTimer (70);
}

void NumberBox::mouseDoubleClick (const juce::MouseEvent& e)
{
    if (overrideValue || ! textArea.contains (e.position))
        return;

    editor = std::make_unique<juce::TextEditor>();
    editor->setFont (font (17.0f, Weight::Medium));
    editor->setJustification (juce::Justification::centredLeft);
    editor->setInputRestrictions (4, "0123456789.");
    editor->setText (juce::String (juce::roundToInt (value)), false);
    editor->selectAll();
    editor->setBounds (textArea.reduced (3.0f).toNearestInt());
    addAndMakeVisible (*editor);
    editor->grabKeyboardFocus();

    auto finish = [this] (bool commit)
    {
        if (editor == nullptr)
            return;
        if (commit)
            setValue (editor->getText().getDoubleValue(), true);
        juce::MessageManager::callAsync ([safe = juce::Component::SafePointer<NumberBox> (this)]
        {
            if (safe != nullptr)
            {
                safe->editor.reset();
                safe->repaint();
            }
        });
    };
    editor->onReturnKey = [finish] { finish (true); };
    editor->onFocusLost = [finish] { finish (true); };
    editor->onEscapeKey = [finish] { finish (false); };
    repaint();
}

void NumberBox::mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails& w)
{
    const float d = std::abs (w.deltaY) > std::abs (w.deltaX) ? w.deltaY : -w.deltaX;
    if (! juce::exactlyEqual (d, 0.0f))
        step (d > 0.0f ? 1 : -1);
}

bool NumberBox::keyPressed (const juce::KeyPress& k)
{
    if (k == juce::KeyPress::upKey)   { step (1);  return true; }
    if (k == juce::KeyPress::downKey) { step (-1); return true; }
    if (k == juce::KeyPress::pageUpKey)   { step (10);  return true; }
    if (k == juce::KeyPress::pageDownKey) { step (-10); return true; }
    return false;
}

} // namespace voxa::ui
