#include "VoxaLookAndFeel.h"
#include "Icons.h"

namespace voxa::ui
{

VoxaLookAndFeel::VoxaLookAndFeel()
{
    setColour (juce::ResizableWindow::backgroundColourId, Palette::panelFill);
    setColour (juce::DocumentWindow::textColourId, Palette::text);

    setColour (juce::Label::textColourId, Palette::textSoft);
    setColour (juce::Label::textWhenEditingColourId, Palette::text);
    setColour (juce::Label::backgroundWhenEditingColourId, Palette::inset);
    setColour (juce::Label::outlineWhenEditingColourId, Palette::glow);

    setColour (juce::TextEditor::backgroundColourId, Palette::inset);
    setColour (juce::TextEditor::textColourId, Palette::text);
    setColour (juce::TextEditor::highlightColourId, Palette::accent.withAlpha (0.55f));
    setColour (juce::TextEditor::highlightedTextColourId, juce::Colours::white);
    setColour (juce::TextEditor::outlineColourId, Palette::insetBorder);
    setColour (juce::TextEditor::focusedOutlineColourId, Palette::glow);
    setColour (juce::CaretComponent::caretColourId, Palette::glowBright);

    setColour (juce::ComboBox::backgroundColourId, Palette::control);
    setColour (juce::ComboBox::textColourId, Palette::text);
    setColour (juce::ComboBox::outlineColourId, Palette::insetBorder);
    setColour (juce::ComboBox::arrowColourId, Palette::textSoft);
    setColour (juce::ComboBox::focusedOutlineColourId, Palette::glow);

    setColour (juce::PopupMenu::backgroundColourId, juce::Colour (0xf20a1533));
    setColour (juce::PopupMenu::textColourId, Palette::textSoft);
    setColour (juce::PopupMenu::headerTextColourId, Palette::glowBright);
    setColour (juce::PopupMenu::highlightedBackgroundColourId, Palette::accent);
    setColour (juce::PopupMenu::highlightedTextColourId, juce::Colours::white);

    setColour (juce::TextButton::buttonColourId, Palette::control);
    setColour (juce::TextButton::buttonOnColourId, Palette::accent);
    setColour (juce::TextButton::textColourOffId, Palette::textSoft);
    setColour (juce::TextButton::textColourOnId, juce::Colours::white);

    setColour (juce::ToggleButton::textColourId, Palette::textSoft);
    setColour (juce::ToggleButton::tickColourId, Palette::glowBright);
    setColour (juce::ToggleButton::tickDisabledColourId, Palette::textFaint);

    setColour (juce::Slider::textBoxTextColourId, Palette::text);
    setColour (juce::Slider::textBoxBackgroundColourId, Palette::inset);
    setColour (juce::Slider::textBoxOutlineColourId, Palette::insetBorder);
    setColour (juce::Slider::textBoxHighlightColourId, Palette::accent.withAlpha (0.5f));

    setColour (juce::TooltipWindow::backgroundColourId, juce::Colour (0xf00a1430));
    setColour (juce::TooltipWindow::textColourId, Palette::text);
    setColour (juce::TooltipWindow::outlineColourId, Palette::glow);

    setColour (juce::AlertWindow::backgroundColourId, Palette::panelFill);
    setColour (juce::AlertWindow::textColourId, Palette::text);
    setColour (juce::AlertWindow::outlineColourId, Palette::glow);

    setColour (juce::ScrollBar::thumbColourId, Palette::glow.withAlpha (0.6f));
    setColour (juce::ScrollBar::trackColourId, juce::Colours::transparentBlack);

    setColour (juce::ListBox::backgroundColourId, Palette::inset);
    setColour (juce::ListBox::textColourId, Palette::text);

    setColour (juce::DirectoryContentsDisplayComponent::highlightColourId, Palette::accent);
    setColour (juce::DirectoryContentsDisplayComponent::textColourId, Palette::text);
    setColour (juce::FileTreeComponent::backgroundColourId, Palette::inset);

    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::windowBackground, Palette::panelFill);
    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::widgetBackground, Palette::control);
    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::menuBackground, Palette::panelFill);
    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::outline, Palette::insetBorder);
    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::defaultText, Palette::text);
    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::defaultFill, Palette::accent);
    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::highlightedText, juce::Colours::white);
    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::highlightedFill, Palette::accent);
    getCurrentColourScheme().setUIColour (ColourScheme::UIColour::menuText, Palette::text);

    setDefaultSansSerifTypeface (getTypeface (Weight::Regular));
    juce::LookAndFeel::setDefaultLookAndFeel (this);
}

VoxaLookAndFeel::~VoxaLookAndFeel()
{
    if (&juce::LookAndFeel::getDefaultLookAndFeel() == this)
        juce::LookAndFeel::setDefaultLookAndFeel (nullptr);
}

const juce::Image& VoxaLookAndFeel::getChromeCap (int diameterPx)
{
    diameterPx = juce::jlimit (8, 1024, diameterPx);
    auto it = chromeCaps.find (diameterPx);
    if (it == chromeCaps.end())
    {
        if (chromeCaps.size() > 48)
            chromeCaps.clear();
        it = chromeCaps.emplace (diameterPx, renderChromeCap (diameterPx)).first;
    }
    return it->second;
}

juce::Typeface::Ptr VoxaLookAndFeel::getTypefaceForFont (const juce::Font& f)
{
    const auto name = f.getTypefaceName();
    if (name.isEmpty() || name == juce::Font::getDefaultSansSerifFontName()
        || name == juce::Font::getDefaultSerifFontName())
    {
        const auto style = f.getTypefaceStyle();
        if (f.isBold() || style.containsIgnoreCase ("bold"))
            return getTypeface (Weight::SemiBold);
        if (style.containsIgnoreCase ("medium"))
            return getTypeface (Weight::Medium);
        return getTypeface (Weight::Regular);
    }
    return LookAndFeel_V4::getTypefaceForFont (f);
}

//==============================================================================
void VoxaLookAndFeel::drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height, float sliderPos,
                                        float startAngle, float endAngle, juce::Slider& slider)
{
    const auto style = slider.getProperties()[SliderProps::knobStyle].toString();
    const bool small = style == "small";
    const bool large = style == "large";
    const bool enabled = slider.isEnabled();
    const bool active = slider.isMouseOverOrDragging() || slider.hasKeyboardFocus (false);

    auto accent = Palette::glow;
    if (slider.getProperties().contains (SliderProps::accent))
        accent = juce::Colour ((juce::uint32) (juce::int64) slider.getProperties()[SliderProps::accent]);

    bool bipolar = false;
    if (slider.getProperties().contains (SliderProps::bipolar))
        bipolar = (bool) slider.getProperties()[SliderProps::bipolar];
    else
        bipolar = juce::approximatelyEqual (slider.getMinimum(), -slider.getMaximum());

    const auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat();
    const float R = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;
    const auto c = bounds.getCentre();
    const float angle = startAngle + sliderPos * (endAngle - startAngle);

    auto circle = [c] (float r) { return juce::Rectangle<float> (c.x - r, c.y - r, r * 2.0f, r * 2.0f); };

    const float arcRadius = R * (small ? 0.86f : 0.87f);
    const float arcWidth = juce::jmax (1.6f, R * (small ? 0.085f : 0.06f));
    const float bezelR = R * (small ? 0.74f : 0.78f);
    const float capR = R * (small ? 0.62f : 0.64f);

    // tick marks
    if (! small)
    {
        const int numTicks = large ? 21 : 11;
        g.setColour (Palette::textSoft.withAlpha (enabled ? 0.38f : 0.15f));
        for (int i = 0; i < numTicks; ++i)
        {
            const float a = startAngle + (endAngle - startAngle) * (float) i / (float) (numTicks - 1);
            const bool major = (i % (large ? 5 : 5)) == 0;
            const float r1 = R * (major ? 0.935f : 0.955f), r2 = R * 0.995f;
            g.drawLine (c.x + r1 * std::sin (a), c.y - r1 * std::cos (a), c.x + r2 * std::sin (a), c.y - r2 * std::cos (a),
                        major ? 1.3f : 0.9f);
        }
    }

    // arc track
    juce::Path track;
    track.addCentredArc (c.x, c.y, arcRadius, arcRadius, 0.0f, startAngle, endAngle, true);
    g.setColour (juce::Colour (0xff040917).withAlpha (enabled ? 0.9f : 0.5f));
    g.strokePath (track, juce::PathStrokeType (arcWidth * 1.9f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    g.setColour (Palette::insetBorder.withAlpha (0.55f));
    g.strokePath (track, juce::PathStrokeType (arcWidth * 0.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // value arc
    {
        const float from = bipolar ? (startAngle + endAngle) * 0.5f : startAngle;
        const float a0 = juce::jmin (from, angle), a1 = juce::jmax (from, angle);
        if (a1 - a0 > 0.005f)
        {
            juce::Path valueArc;
            valueArc.addCentredArc (c.x, c.y, arcRadius, arcRadius, 0.0f, a0, a1, true);
            const auto col = enabled ? (active ? accent.brighter (0.25f) : accent) : Palette::textFaint;
            strokeGlow (g, valueArc, col, arcWidth, enabled ? (active ? 1.4f : 1.0f) : 0.0f);
        }
    }

    // bezel well
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xff1c2d5c), c.x, c.y - bezelR,
                                             juce::Colour (0xff040817), c.x, c.y + bezelR, false));
    g.fillEllipse (circle (bezelR));

    // luminous blue ring around the cap (brighter where the light falls, and when active)
    {
        const float ringR = (capR + bezelR) * 0.5f + 0.5f;
        const float ringW = juce::jmax (1.2f, (bezelR - capR) * (small ? 0.55f : 0.62f));
        juce::Path ring;
        ring.addEllipse (circle (ringR));
        const float boost = enabled ? (active ? 1.0f : 0.82f) : 0.3f;

        juce::ColourGradient rg (accent.brighter (0.6f).withMultipliedAlpha (boost), c.x - ringR, c.y - ringR,
                                 accent.darker (0.9f).withMultipliedAlpha (boost), c.x + ringR * 0.6f, c.y + ringR, false);
        rg.addColour (0.45, accent.withMultipliedAlpha (boost));

        g.setColour (accent.withAlpha (0.18f * boost));
        g.strokePath (ring, juce::PathStrokeType (ringW * 2.6f));
        g.setGradientFill (rg);
        g.strokePath (ring, juce::PathStrokeType (ringW));
        g.setColour (juce::Colours::white.withAlpha (0.35f * boost));
        g.drawEllipse (circle (ringR + ringW * 0.5f), 0.6f);
    }

    // cap shadow
    g.setColour (juce::Colours::black.withAlpha (0.45f));
    g.fillEllipse (circle (capR + 0.8f).translated (0.0f, capR * 0.06f));

    // chrome cap
    {
        const float phys = physicalScale (g);
        const auto& cap = getChromeCap (juce::roundToInt (capR * 2.0f * phys));
        g.setOpacity (enabled ? 1.0f : 0.55f);
        drawCachedImage (g, cap, c - juce::Point<float> ((float) cap.getWidth(), (float) cap.getHeight()) * (0.5f / phys), phys);
        g.setOpacity (1.0f);

        if (active && enabled)
        {
            g.setColour (juce::Colours::white.withAlpha (0.06f));
            g.fillEllipse (circle (capR));
        }
    }

    // pointer
    {
        const float r1 = capR * (small ? 0.30f : 0.42f), r2 = capR * 0.9f;
        juce::Path pointer;
        pointer.startNewSubPath (c.x + r1 * std::sin (angle), c.y - r1 * std::cos (angle));
        pointer.lineTo (c.x + r2 * std::sin (angle), c.y - r2 * std::cos (angle));
        const float pw = juce::jmax (1.4f, capR * (small ? 0.12f : 0.075f));
        g.setColour (juce::Colour (0xff0a1a3e).withAlpha (0.55f));
        g.strokePath (pointer, juce::PathStrokeType (pw * 2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        strokeGlow (g, pointer, enabled ? accent.brighter (0.3f) : Palette::textFaint, pw, enabled ? 0.8f : 0.0f);
    }
}

int VoxaLookAndFeel::getSliderThumbRadius (juce::Slider& slider)
{
    const bool small = slider.getProperties()[SliderProps::knobStyle].toString() == "small";
    const int h = slider.isHorizontal() ? slider.getHeight() : slider.getWidth();
    return juce::jmin (small ? 7 : 11, juce::jmax (4, h / 2 - 1));
}

void VoxaLookAndFeel::drawLinearSlider (juce::Graphics& g, int x, int y, int width, int height, float sliderPos,
                                        float minSliderPos, float maxSliderPos, juce::Slider::SliderStyle style,
                                        juce::Slider& slider)
{
    if (style != juce::Slider::LinearHorizontal && style != juce::Slider::LinearVertical)
    {
        LookAndFeel_V4::drawLinearSlider (g, x, y, width, height, sliderPos, minSliderPos, maxSliderPos, style, slider);
        return;
    }

    juce::ignoreUnused (minSliderPos, maxSliderPos);
    const bool horizontal = style == juce::Slider::LinearHorizontal;
    const bool small = slider.getProperties()[SliderProps::knobStyle].toString() == "small";
    const bool enabled = slider.isEnabled();
    const bool active = slider.isMouseOverOrDragging() || slider.hasKeyboardFocus (false);

    auto accent = Palette::glow;
    if (slider.getProperties().contains (SliderProps::accent))
        accent = juce::Colour ((juce::uint32) (juce::int64) slider.getProperties()[SliderProps::accent]);

    bool bipolar = false;
    if (slider.getProperties().contains (SliderProps::bipolar))
        bipolar = (bool) slider.getProperties()[SliderProps::bipolar];

    const auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat();
    const float trackThickness = small ? 4.0f : 6.0f;

    juce::Rectangle<float> trackArea;
    if (horizontal)
        trackArea = { bounds.getX(), bounds.getCentreY() - trackThickness * 0.5f, bounds.getWidth(), trackThickness };
    else
        trackArea = { bounds.getCentreX() - trackThickness * 0.5f, bounds.getY(), trackThickness, bounds.getHeight() };

    // groove
    g.setColour (juce::Colour (0xff030816).withAlpha (enabled ? 0.95f : 0.5f));
    g.fillRoundedRectangle (trackArea.expanded (1.0f), trackThickness * 0.5f + 1.0f);
    g.setColour (Palette::insetBorder.withAlpha (0.6f));
    g.drawRoundedRectangle (trackArea.expanded (1.0f), trackThickness * 0.5f + 1.0f, 0.8f);

    // value fill
    {
        float from = horizontal ? trackArea.getX() : trackArea.getBottom();
        if (bipolar)
            from = horizontal ? trackArea.getCentreX() : trackArea.getCentreY();

        juce::Rectangle<float> fill;
        if (horizontal)
            fill = juce::Rectangle<float>::leftTopRightBottom (juce::jmin (from, sliderPos), trackArea.getY(),
                                                               juce::jmax (from, sliderPos), trackArea.getBottom());
        else
            fill = juce::Rectangle<float>::leftTopRightBottom (trackArea.getX(), juce::jmin (from, sliderPos),
                                                               trackArea.getRight(), juce::jmax (from, sliderPos));

        if (! fill.isEmpty())
        {
            const auto col = enabled ? accent : Palette::textFaint;
            if (enabled)
            {
                g.setColour (col.withAlpha (active ? 0.30f : 0.20f));
                g.fillRoundedRectangle (fill.expanded (3.0f), trackThickness);
            }
            g.setGradientFill (horizontal
                                   ? juce::ColourGradient (col.darker (0.4f), fill.getX(), 0.0f, col.brighter (0.5f), fill.getRight(), 0.0f, false)
                                   : juce::ColourGradient (col.darker (0.4f), 0.0f, fill.getBottom(), col.brighter (0.5f), 0.0f, fill.getY(), false));
            g.fillRoundedRectangle (fill, trackThickness * 0.5f);
            g.setColour (juce::Colours::white.withAlpha (enabled ? 0.45f : 0.1f));
            g.fillRoundedRectangle (fill.withSizeKeepingCentre (fill.getWidth() - (horizontal ? 2.0f : 0.0f), 1.2f)
                                        .withY (horizontal ? fill.getCentreY() - 0.6f : fill.getY()), 0.6f);
        }
    }

    // thumb (chrome sphere)
    {
        const float r = (float) getSliderThumbRadius (slider);
        const auto centre = horizontal ? juce::Point<float> (sliderPos, bounds.getCentreY())
                                       : juce::Point<float> (bounds.getCentreX(), sliderPos);
        const auto thumb = juce::Rectangle<float> (r * 2.0f, r * 2.0f).withCentre (centre);

        if (enabled)
        {
            g.setColour (accent.withAlpha (active ? 0.35f : 0.2f));
            g.fillEllipse (thumb.expanded (r * 0.45f));
        }
        g.setColour (juce::Colours::black.withAlpha (0.45f));
        g.fillEllipse (thumb.translated (0.0f, 1.5f));

        juce::ColourGradient grad (juce::Colours::white, centre.x - r * 0.35f, centre.y - r * 0.45f,
                                   juce::Colour (0xff8a9cc0), centre.x + r * 0.6f, centre.y + r * 0.9f, true);
        grad.addColour (0.45, juce::Colour (0xffe1e8f5));
        g.setGradientFill (grad);
        g.fillEllipse (thumb);
        g.setColour (juce::Colour (0xff3e5486));
        g.drawEllipse (thumb.reduced (0.4f), 0.8f);
        g.setColour (juce::Colours::white.withAlpha (0.8f));
        g.fillEllipse (thumb.reduced (r * 0.55f).translated (-r * 0.18f, -r * 0.22f));
    }
}

juce::Label* VoxaLookAndFeel::createSliderTextBox (juce::Slider& slider)
{
    auto* l = LookAndFeel_V4::createSliderTextBox (slider);
    l->setFont (font (14.0f, Weight::Medium));
    l->setColour (juce::Label::outlineColourId, juce::Colours::transparentBlack);
    l->setColour (juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    return l;
}

//==============================================================================
void VoxaLookAndFeel::drawButtonBackground (juce::Graphics& g, juce::Button& b, const juce::Colour&,
                                            bool isHighlighted, bool isDown)
{
    auto r = b.getLocalBounds().toFloat().reduced (1.0f);
    drawGlassControl (g, r, juce::jmin (8.0f, r.getHeight() * 0.3f), b.getToggleState(), isHighlighted, isDown, b.isEnabled());
}

juce::Font VoxaLookAndFeel::getTextButtonFont (juce::TextButton&, int buttonHeight)
{
    return font (juce::jlimit (12.0f, 17.0f, (float) buttonHeight * 0.48f), Weight::Medium);
}

void VoxaLookAndFeel::drawButtonText (juce::Graphics& g, juce::TextButton& b, bool, bool)
{
    const auto col = b.getToggleState() ? juce::Colours::white : Palette::textSoft;
    g.setFont (getTextButtonFont (b, b.getHeight()));
    g.setColour (col.withMultipliedAlpha (b.isEnabled() ? 1.0f : 0.45f));
    g.drawFittedText (b.getButtonText(), b.getLocalBounds().reduced (6, 2), juce::Justification::centred, 1);
}

void VoxaLookAndFeel::drawToggleButton (juce::Graphics& g, juce::ToggleButton& b, bool isHighlighted, bool)
{
    auto r = b.getLocalBounds().toFloat();
    const float h = juce::jmin (r.getHeight(), 18.0f);
    auto sw = juce::Rectangle<float> (h * 1.8f, h).withCentre ({ r.getX() + h * 0.9f + 2.0f, r.getCentreY() });
    const bool on = b.getToggleState();

    g.setColour (on ? Palette::accent : juce::Colour (0xff050b1e));
    g.fillRoundedRectangle (sw, h * 0.5f);
    g.setColour (on ? Palette::glowBright : Palette::insetBorder.brighter (isHighlighted ? 0.4f : 0.0f));
    g.drawRoundedRectangle (sw.reduced (0.5f), h * 0.5f, 1.0f);
    const auto knob = juce::Rectangle<float> (h - 4.0f, h - 4.0f)
                          .withCentre ({ on ? sw.getRight() - h * 0.5f : sw.getX() + h * 0.5f, sw.getCentreY() });
    g.setColour (on ? juce::Colours::white : Palette::textDim);
    g.fillEllipse (knob);

    g.setFont (font (15.0f, Weight::Medium));
    g.setColour ((on ? Palette::text : Palette::textSoft).withMultipliedAlpha (b.isEnabled() ? 1.0f : 0.45f));
    g.drawText (b.getButtonText(), r.withTrimmedLeft (sw.getRight() + 6.0f - r.getX()), juce::Justification::centredLeft, true);
}

//==============================================================================
void VoxaLookAndFeel::drawComboBox (juce::Graphics& g, int width, int height, bool isButtonDown, int, int, int, int,
                                    juce::ComboBox& box)
{
    auto r = juce::Rectangle<float> (0.0f, 0.0f, (float) width, (float) height).reduced (0.5f);
    const bool hover = box.isMouseOver (true);
    drawGlassControl (g, r, juce::jmin (7.0f, r.getHeight() * 0.25f), false, hover, isButtonDown, box.isEnabled());

    if (box.hasKeyboardFocus (true))
    {
        g.setColour (Palette::glow.withAlpha (0.8f));
        g.drawRoundedRectangle (r.reduced (0.5f), juce::jmin (7.0f, r.getHeight() * 0.25f), 1.2f);
    }

    const float arrowSize = juce::jmin (14.0f, (float) height * 0.42f);
    auto arrowArea = juce::Rectangle<float> (arrowSize, arrowSize).withCentre ({ (float) width - arrowSize * 0.5f - 8.0f, (float) height * 0.5f });
    Icons::draw (g, Icon::ChevronDown, arrowArea, Palette::textSoft.withMultipliedAlpha (box.isEnabled() ? 1.0f : 0.4f), 2.2f);
}

juce::Font VoxaLookAndFeel::getComboBoxFont (juce::ComboBox& box)
{
    return font (juce::jlimit (12.0f, 17.0f, (float) box.getHeight() * 0.47f), Weight::Medium);
}

void VoxaLookAndFeel::positionComboBoxText (juce::ComboBox& box, juce::Label& label)
{
    const float arrowSize = juce::jmin (14.0f, (float) box.getHeight() * 0.42f);
    label.setBounds (6, 1, box.getWidth() - 6 - juce::roundToInt (arrowSize) - 10, box.getHeight() - 2);
    label.setFont (getComboBoxFont (box));
    label.setColour (juce::Label::textColourId, Palette::text.withMultipliedAlpha (box.isEnabled() ? 1.0f : 0.5f));
}

juce::PopupMenu::Options VoxaLookAndFeel::getOptionsForComboBoxPopupMenu (juce::ComboBox& box, juce::Label& label)
{
    return LookAndFeel_V4::getOptionsForComboBoxPopupMenu (box, label)
               .withStandardItemHeight (26)
               .withMinimumWidth (juce::jmax (box.getWidth(), 120));
}

//==============================================================================
void VoxaLookAndFeel::drawPopupMenuBackground (juce::Graphics& g, int width, int height)
{
    auto r = juce::Rectangle<float> (0.0f, 0.0f, (float) width, (float) height);
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xff15264f), 0.0f, 0.0f, juce::Colour (0xff081230), 0.0f, (float) height, false));
    g.fillRect (r);
    g.setColour (Palette::glow.withAlpha (0.65f));
    g.drawRect (r, 1.0f);
}

void VoxaLookAndFeel::drawPopupMenuItem (juce::Graphics& g, const juce::Rectangle<int>& area, bool isSeparator,
                                         bool isActive, bool isHighlighted, bool isTicked, bool hasSubMenu,
                                         const juce::String& text, const juce::String& shortcutKeyText,
                                         const juce::Drawable* icon, const juce::Colour* textColour)
{
    if (isSeparator)
    {
        auto r = area.toFloat().reduced (8.0f, 0.0f);
        g.setColour (Palette::panelBorder.withAlpha (0.18f));
        g.fillRect (r.withHeight (1.0f).withCentre (r.getCentre()));
        return;
    }

    auto r = area.toFloat().reduced (3.0f, 1.0f);

    if (isHighlighted && isActive)
    {
        g.setGradientFill (juce::ColourGradient (juce::Colour (0xff3d82ff), r.getX(), r.getY(),
                                                 juce::Colour (0xff1a46c2), r.getX(), r.getBottom(), false));
        g.fillRoundedRectangle (r, 5.0f);
        g.setColour (Palette::glowBright.withAlpha (0.6f));
        g.drawRoundedRectangle (r.reduced (0.5f), 5.0f, 1.0f);
    }

    auto textCol = textColour != nullptr ? *textColour : (isHighlighted ? juce::Colours::white : Palette::textSoft);
    if (! isActive)
        textCol = Palette::textFaint;

    auto content = r.reduced (8.0f, 0.0f);
    auto tickArea = content.removeFromLeft (18.0f);

    if (isTicked)
    {
        auto dot = juce::Rectangle<float> (7.0f, 7.0f).withCentre (tickArea.getCentre());
        g.setColour (Palette::glow.withAlpha (0.4f));
        g.fillEllipse (dot.expanded (3.0f));
        g.setColour (Palette::glowBright);
        g.fillEllipse (dot);
    }

    if (icon != nullptr)
    {
        auto iconArea = content.removeFromLeft (20.0f).reduced (2.0f);
        icon->drawWithin (g, iconArea, juce::RectanglePlacement::centred, 1.0f);
        content.removeFromLeft (4.0f);
    }

    if (hasSubMenu)
    {
        auto arrow = content.removeFromRight (14.0f);
        Icons::draw (g, Icon::ChevronRight, arrow.withSizeKeepingCentre (11.0f, 11.0f), textCol, 2.2f);
    }

    g.setFont (getPopupMenuFont());
    g.setColour (textCol);
    g.drawFittedText (text, content.toNearestInt(), juce::Justification::centredLeft, 1);

    if (shortcutKeyText.isNotEmpty())
    {
        g.setFont (font (13.0f));
        g.setColour (Palette::textDim);
        g.drawText (shortcutKeyText, content, juce::Justification::centredRight, true);
    }
}

void VoxaLookAndFeel::drawPopupMenuSectionHeader (juce::Graphics& g, const juce::Rectangle<int>& area, const juce::String& text)
{
    auto r = area.toFloat().reduced (10.0f, 0.0f);
    g.setFont (font (12.5f, Weight::SemiBold));
    g.setColour (Palette::glowBright.withAlpha (0.85f));
    g.drawText (text.toUpperCase(), r.withTrimmedTop (4.0f), juce::Justification::centredLeft, true);
    g.setColour (Palette::panelBorder.withAlpha (0.15f));
    g.fillRect (r.withTop (r.getBottom() - 1.0f));
}

juce::Font VoxaLookAndFeel::getPopupMenuFont()
{
    return font (15.5f, Weight::Regular);
}

void VoxaLookAndFeel::getIdealPopupMenuItemSize (const juce::String& text, bool isSeparator, int standardMenuItemHeight,
                                                 int& idealWidth, int& idealHeight)
{
    if (isSeparator)
    {
        idealWidth = 50;
        idealHeight = 9;
        return;
    }

    idealHeight = standardMenuItemHeight > 0 ? standardMenuItemHeight : 27;
    idealWidth = juce::roundToInt (textWidth (getPopupMenuFont(), text)) + 60;
}

//==============================================================================
namespace
{
    juce::TextLayout layoutTooltip (const juce::String& text, float maxWidth)
    {
        juce::AttributedString s;
        s.setJustification (juce::Justification::topLeft);
        s.setLineSpacing (1.0f);

        const auto lines = juce::StringArray::fromLines (text);
        for (int i = 0; i < lines.size(); ++i)
        {
            const bool first = i == 0 && lines.size() > 1;
            s.append (lines[i] + (i < lines.size() - 1 ? "\n" : ""),
                      first ? font (15.0f, Weight::SemiBold) : font (14.0f),
                      first ? juce::Colours::white : Palette::textSoft);
        }

        juce::TextLayout tl;
        tl.createLayoutWithBalancedLineLengths (s, maxWidth);
        return tl;
    }
}

const juce::TextLayout& VoxaLookAndFeel::getTooltipLayout (const juce::String& text)
{
    if (text != tooltipText || tooltipLayout.getNumLines() == 0)
    {
        tooltipText = text;
        tooltipLayout = layoutTooltip (text, 300.0f);
    }
    return tooltipLayout;
}

juce::Rectangle<int> VoxaLookAndFeel::getTooltipBounds (const juce::String& tipText, juce::Point<int> screenPos,
                                                        juce::Rectangle<int> parentArea)
{
    const auto& tl = getTooltipLayout (tipText);
    const int w = (int) std::ceil (tl.getWidth()) + 22;
    const int h = (int) std::ceil (tl.getHeight()) + 14;

    return juce::Rectangle<int> (screenPos.x > parentArea.getCentreX() ? screenPos.x - (w + 12) : screenPos.x + 18,
                                 screenPos.y > parentArea.getCentreY() ? screenPos.y - (h + 8) : screenPos.y + 18,
                                 w, h)
        .constrainedWithin (parentArea);
}

void VoxaLookAndFeel::drawTooltip (juce::Graphics& g, const juce::String& text, int width, int height)
{
    auto r = juce::Rectangle<float> (0.0f, 0.0f, (float) width, (float) height);
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xf61a2c5a), 0.0f, 0.0f, juce::Colour (0xf6091431), 0.0f, (float) height, false));
    g.fillRoundedRectangle (r, 6.0f);
    g.setColour (Palette::glow.withAlpha (0.8f));
    g.drawRoundedRectangle (r.reduced (0.5f), 6.0f, 1.0f);

    getTooltipLayout (text).draw (g, r.reduced (11.0f, 7.0f));
}

//==============================================================================
void VoxaLookAndFeel::fillTextEditorBackground (juce::Graphics& g, int width, int height, juce::TextEditor& te)
{
    if (te.getProperties()["voxaTransparent"])
        return;
    fillInset (g, juce::Rectangle<float> (0.0f, 0.0f, (float) width, (float) height), 6.0f);
}

void VoxaLookAndFeel::drawTextEditorOutline (juce::Graphics& g, int width, int height, juce::TextEditor& te)
{
    if (te.getProperties()["voxaTransparent"])
        return;

    if (te.isEnabled() && te.hasKeyboardFocus (true) && ! te.isReadOnly())
    {
        g.setColour (Palette::glow.withAlpha (0.85f));
        g.drawRoundedRectangle (juce::Rectangle<float> (0.0f, 0.0f, (float) width, (float) height).reduced (0.75f), 6.0f, 1.5f);
    }
}

void VoxaLookAndFeel::drawScrollbar (juce::Graphics& g, juce::ScrollBar&, int x, int y, int width, int height, bool isVertical,
                                     int thumbStartPosition, int thumbSize, bool isMouseOver, bool isMouseDown)
{
    juce::Rectangle<float> thumb;
    if (isVertical)
        thumb = { (float) x + 2.0f, (float) thumbStartPosition, (float) width - 4.0f, (float) thumbSize };
    else
        thumb = { (float) thumbStartPosition, (float) y + 2.0f, (float) thumbSize, (float) height - 4.0f };

    g.setColour (Palette::glow.withAlpha (isMouseDown ? 0.9f : (isMouseOver ? 0.75f : 0.5f)));
    g.fillRoundedRectangle (thumb, juce::jmin (thumb.getWidth(), thumb.getHeight()) * 0.5f);
}

//==============================================================================
void VoxaLookAndFeel::drawAlertBox (juce::Graphics& g, juce::AlertWindow& alert, const juce::Rectangle<int>& textArea,
                                    juce::TextLayout& textLayout)
{
    auto r = alert.getLocalBounds().toFloat();
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xff1a2c5c), 0.0f, 0.0f, juce::Colour (0xff081230), 0.0f, r.getHeight(), false));
    g.fillRect (r);
    g.setColour (Palette::glow.withAlpha (0.8f));
    g.drawRect (r, 1.5f);

    textLayout.draw (g, textArea.withTrimmedLeft (4).toFloat());
}

juce::Font VoxaLookAndFeel::getAlertWindowTitleFont()   { return font (19.0f, Weight::SemiBold); }
juce::Font VoxaLookAndFeel::getAlertWindowMessageFont() { return font (15.5f); }
juce::Font VoxaLookAndFeel::getAlertWindowFont()        { return font (15.0f); }

juce::Font VoxaLookAndFeel::getLabelFont (juce::Label& label)
{
    return label.getFont();
}

void VoxaLookAndFeel::drawCornerResizer (juce::Graphics& g, int w, int h, bool isMouseOver, bool isMouseDragging)
{
    const float alpha = isMouseDragging ? 0.9f : (isMouseOver ? 0.7f : 0.35f);
    g.setColour (Palette::glowBright.withAlpha (alpha));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j)
        {
            const float x = (float) w - 4.0f - (float) j * 4.0f;
            const float y = (float) h - 4.0f - (float) (i - j) * 4.0f;
            g.fillEllipse (x - 1.0f, y - 1.0f, 2.0f, 2.0f);
        }
}

void VoxaLookAndFeel::drawCallOutBoxBackground (juce::CallOutBox&, juce::Graphics& g, const juce::Path& path, juce::Image&)
{
    g.setColour (juce::Colour (0xf20b1736));
    g.fillPath (path);
    g.setColour (Palette::glow.withAlpha (0.8f));
    g.strokePath (path, juce::PathStrokeType (1.5f));
}

} // namespace voxa::ui
