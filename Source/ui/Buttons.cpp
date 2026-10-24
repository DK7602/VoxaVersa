#include "Buttons.h"

namespace voxa::ui
{

//==============================================================================
GlassButton::GlassButton (const juce::String& buttonText, std::optional<Icon> i, Style s)
    : juce::Button (buttonText), icon (i), style (s)
{
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
    setTitle (buttonText);
}

void GlassButton::paintButton (juce::Graphics& g, bool hover, bool down)
{
    auto r = getLocalBounds().toFloat().reduced (1.5f);
    const bool on = getToggleState() || style == Style::Primary;
    const bool enabled = isEnabled();

    if (style == Style::Flat && ! on)
    {
        if (hover || down)
        {
            g.setColour (Palette::controlTop.withAlpha (down ? 0.7f : 0.45f));
            g.fillRoundedRectangle (r, corner);
            g.setColour (Palette::panelBorder.withAlpha (0.18f));
            g.drawRoundedRectangle (r.reduced (0.5f), corner, 1.0f);
        }
    }
    else
    {
        drawGlassControl (g, r, corner, on, hover, down, enabled);
    }

    if (progress >= 0.0f)
    {
        auto bar = r.reduced (3.0f);
        bar = bar.withWidth (bar.getWidth() * juce::jlimit (0.0f, 1.0f, progress));
        g.setColour (juce::Colours::white.withAlpha (0.18f));
        g.fillRoundedRectangle (bar, juce::jmax (1.0f, corner - 3.0f));
    }

    if (hasKeyboardFocus (false))
    {
        g.setColour (Palette::glowBright.withAlpha (0.8f));
        g.drawRoundedRectangle (r.reduced (1.0f), corner, 1.2f);
    }

    const auto f = font (fontHeight, Weight::Medium);
    const auto label = getButtonText();
    const float tw = label.isEmpty() ? 0.0f : textWidth (f, label);
    const float iconSize = icon ? juce::jmin (r.getHeight() * 0.55f, 22.0f) : 0.0f;
    const float trailingSize = trailingIcon ? fontHeight * 0.7f : 0.0f;
    const float gap = (icon && tw > 0.0f) ? 8.0f : 0.0f;
    const float trailingGap = trailingIcon ? 6.0f : 0.0f;
    const float total = iconSize + gap + tw + trailingGap + trailingSize;
    float x = r.getCentreX() - total * 0.5f;

    const float alpha = enabled ? 1.0f : 0.45f;
    const auto txtCol = (textColour ? *textColour : (on ? juce::Colours::white : (hover ? Palette::text : Palette::textSoft))).withMultipliedAlpha (alpha);

    if (icon)
    {
        const auto ic = (iconColour ? *iconColour : (on ? juce::Colours::white : Palette::glowBright)).withMultipliedAlpha (alpha);
        Icons::draw (g, *icon, { x, r.getCentreY() - iconSize * 0.5f, iconSize, iconSize }, ic, 1.8f, on || hover);
        x += iconSize + gap;
    }

    if (tw > 0.0f)
    {
        g.setFont (f);
        g.setColour (txtCol);
        g.drawText (label, juce::Rectangle<float> (x, r.getY(), tw + 2.0f, r.getHeight()), juce::Justification::centredLeft, false);
        x += tw + trailingGap;
    }

    if (trailingIcon)
        Icons::draw (g, *trailingIcon, { x, r.getCentreY() - trailingSize * 0.5f + 1.0f, trailingSize, trailingSize }, txtCol, 2.4f);
}

//==============================================================================
IconButton::IconButton (const juce::String& name, Icon i, Background bg)
    : juce::Button (name), icon (i), background (bg)
{
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
    setTitle (name);
    setTooltip (name);
}

void IconButton::paintButton (juce::Graphics& g, bool hover, bool down)
{
    auto r = getLocalBounds().toFloat().reduced (1.5f);
    const bool on = litWhenOn && getToggleState();
    const bool enabled = isEnabled();

    if (background == Background::Rounded)
        drawGlassControl (g, r, juce::jmin (9.0f, r.getHeight() * 0.25f), on, hover, down, enabled);
    else if (background == Background::Circle)
        drawGlassControl (g, r.withSizeKeepingCentre (juce::jmin (r.getWidth(), r.getHeight()), juce::jmin (r.getWidth(), r.getHeight())),
                          juce::jmin (r.getWidth(), r.getHeight()) * 0.5f, on, hover, down, enabled);
    else if (on)
    {
        g.setColour (Palette::glow.withAlpha (0.18f));
        g.fillRoundedRectangle (r, 6.0f);
    }

    if (hasKeyboardFocus (false))
    {
        g.setColour (Palette::glowBright.withAlpha (0.8f));
        g.drawRoundedRectangle (r.reduced (1.0f), 6.0f, 1.2f);
    }

    const float size = juce::jmin (r.getWidth(), r.getHeight()) * iconScale;
    auto col = on ? juce::Colours::white : (hover ? Palette::text : colour);
    if (down)
        col = col.darker (0.2f);
    if (! enabled)
        col = col.withMultipliedAlpha (0.35f);

    Icons::draw (g, icon, r.withSizeKeepingCentre (size, size), col, strokeWidth, enabled && (hover || on));
}

//==============================================================================
PowerButton::PowerButton() : juce::Button ("Power")
{
    setClickingTogglesState (true);
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
    setTitle ("Power");
    setTooltip ("Power: bypasses all processing when off");
}

void PowerButton::paintButton (juce::Graphics& g, bool hover, bool down)
{
    auto r = getLocalBounds().toFloat();
    const float d = juce::jmin (r.getWidth(), r.getHeight()) - 8.0f;
    auto circle = r.withSizeKeepingCentre (d, d);
    const bool on = getToggleState();

    if (on)
    {
        for (int i = 3; i >= 1; --i)
        {
            g.setColour (Palette::glow.withAlpha (0.10f * (float) (4 - i)));
            g.fillEllipse (circle.expanded ((float) i * 1.6f));
        }
        g.setGradientFill (juce::ColourGradient (juce::Colour (0xff57a0ff).brighter (hover ? 0.1f : 0.0f), circle.getCentreX(), circle.getY(),
                                                 juce::Colour (0xff1640b4), circle.getCentreX(), circle.getBottom(), false));
    }
    else
    {
        g.setGradientFill (juce::ColourGradient (Palette::controlTop.brighter (hover ? 0.15f : 0.0f), circle.getCentreX(), circle.getY(),
                                                 Palette::control, circle.getCentreX(), circle.getBottom(), false));
    }
    g.fillEllipse (circle);

    // glossy top highlight
    auto sheen = circle.reduced (d * 0.12f).withHeight (d * 0.42f).translated (0.0f, -d * 0.04f);
    g.setGradientFill (juce::ColourGradient (juce::Colours::white.withAlpha (on ? 0.35f : 0.12f), sheen.getCentreX(), sheen.getY(),
                                             juce::Colours::white.withAlpha (0.0f), sheen.getCentreX(), sheen.getBottom(), false));
    g.fillEllipse (sheen);

    g.setColour (on ? Palette::glowBright : Palette::panelBorder.withAlpha (0.4f));
    g.drawEllipse (circle.reduced (0.6f), on ? 1.6f : 1.0f);

    if (hasKeyboardFocus (false))
    {
        g.setColour (juce::Colours::white.withAlpha (0.7f));
        g.drawEllipse (circle.expanded (2.5f), 1.0f);
    }

    Icons::draw (g, Icon::Power, circle.reduced (d * 0.24f), (on ? juce::Colours::white : Palette::textDim).darker (down ? 0.2f : 0.0f), 2.2f, on);
}

//==============================================================================
ModeTab::ModeTab (const juce::String& title, const juce::String& sub, Icon i)
    : juce::Button (title), subtitle (sub), icon (i)
{
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
    setTitle (title);
    setDescription (sub);
}

void ModeTab::paintButton (juce::Graphics& g, bool hover, bool down)
{
    auto r = getLocalBounds().toFloat().reduced (3.0f);
    const bool on = getToggleState();
    drawGlassControl (g, r, 10.0f, on, hover, down, isEnabled());

    if (hasKeyboardFocus (false))
    {
        g.setColour (juce::Colours::white.withAlpha (0.6f));
        g.drawRoundedRectangle (r.reduced (2.0f), 9.0f, 1.0f);
    }

    auto content = r.reduced (14.0f, 6.0f);
    const float iconSize = juce::jmin (36.0f, content.getHeight() * 0.78f);
    auto iconArea = content.removeFromLeft (iconSize).withSizeKeepingCentre (iconSize, iconSize);
    Icons::draw (g, icon, iconArea, juce::Colours::white.withAlpha (on ? 1.0f : 0.85f), 1.7f, on);
    content.removeFromLeft (12.0f);

    auto titleArea = content.removeFromTop (content.getHeight() * 0.54f);
    drawText (g, getButtonText(), titleArea, font (21.0f, Weight::Medium), juce::Colours::white.withAlpha (on ? 1.0f : 0.92f),
              juce::Justification::bottomLeft);
    drawText (g, subtitle, content, font (16.0f), (on ? juce::Colour (0xffd8e6ff) : Palette::textSoft), juce::Justification::topLeft);
}

//==============================================================================
FeatureTile::FeatureTile (const juce::String& caption, Icon i)
    : juce::Button (caption), icon (i)
{
    setClickingTogglesState (true);
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
    setTitle (caption);
}

void FeatureTile::paintButton (juce::Graphics& g, bool hover, bool down)
{
    auto r = getLocalBounds().toFloat();
    const float captionH = 20.0f;
    const float tileSize = juce::jmin (r.getWidth() - 6.0f, r.getHeight() - captionH - 6.0f);
    auto tile = juce::Rectangle<float> (tileSize, tileSize).withCentre ({ r.getCentreX(), r.getY() + 3.0f + tileSize * 0.5f });
    const bool on = getToggleState();
    const float corner = 9.0f;

    if (on)
    {
        juce::Path outline;
        outline.addRoundedRectangle (tile, corner);
        g.setColour (Palette::glow.withAlpha (0.12f));
        g.strokePath (outline, juce::PathStrokeType (7.0f));
        g.setColour (Palette::glow.withAlpha (0.25f));
        g.strokePath (outline, juce::PathStrokeType (3.5f));
    }

    g.setGradientFill (juce::ColourGradient (juce::Colour (on ? 0xff16306a : 0xff111f45).brighter (hover ? 0.15f : 0.0f), tile.getX(), tile.getY(),
                                             juce::Colour (on ? 0xff0b1a40 : 0xff0a1430), tile.getX(), tile.getBottom(), false));
    g.fillRoundedRectangle (tile, corner);
    g.setColour (on ? Palette::glowBright.withAlpha (0.95f) : Palette::panelBorder.withAlpha (hover ? 0.45f : 0.25f));
    g.drawRoundedRectangle (tile.reduced (0.5f), corner, on ? 1.5f : 1.0f);

    if (hasKeyboardFocus (false))
    {
        g.setColour (juce::Colours::white.withAlpha (0.7f));
        g.drawRoundedRectangle (tile.expanded (2.0f), corner + 2.0f, 1.0f);
    }

    const auto iconCol = on ? juce::Colour (0xff6fb8ff) : Palette::textDim;
    Icons::draw (g, icon, tile.reduced (tileSize * 0.2f), down ? iconCol.darker (0.2f) : iconCol, 1.9f, on);

    drawText (g, getButtonText(), { r.getX() - 6.0f, tile.getBottom() + 3.0f, r.getWidth() + 12.0f, captionH },
              font (15.5f, Weight::Medium), on ? Palette::text : Palette::textSoft);
}

//==============================================================================
TabButton::TabButton (const juce::String& buttonText) : juce::Button (buttonText)
{
    setClickingTogglesState (false);
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
    setTitle (buttonText);
}

float TabButton::getIdealWidth() const
{
    return textWidth (font (fontHeight, Weight::Title), getButtonText()) + 14.0f;
}

void TabButton::paintButton (juce::Graphics& g, bool hover, bool)
{
    auto r = getLocalBounds().toFloat();
    const bool on = getToggleState();
    const auto f = font (fontHeight, Weight::Title);

    drawText (g, getButtonText(), r.withTrimmedBottom (4.0f), f,
              on ? juce::Colours::white : (hover ? Palette::textSoft : Palette::textDim));

    if (on)
    {
        const float tw = textWidth (f, getButtonText());
        auto line = juce::Rectangle<float> (tw, 2.0f).withCentre ({ r.getCentreX(), r.getBottom() - 2.5f });
        g.setColour (Palette::glow.withAlpha (0.35f));
        g.fillRoundedRectangle (line.expanded (2.0f, 2.0f), 2.0f);
        g.setColour (Palette::glowBright);
        g.fillRoundedRectangle (line, 1.0f);
    }

    if (hasKeyboardFocus (false))
    {
        g.setColour (Palette::glowBright.withAlpha (0.6f));
        g.drawRoundedRectangle (r.reduced (1.0f), 4.0f, 1.0f);
    }
}

} // namespace voxa::ui
