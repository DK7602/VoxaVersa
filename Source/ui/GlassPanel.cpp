#include "GlassPanel.h"

namespace voxa::ui
{

GlassPanel::GlassPanel (const juce::String& t, std::optional<Icon> icon)
    : title (t), titleIcon (icon)
{
    setTitle (t);
    setFocusContainerType (FocusContainerType::focusContainer);
}

void GlassPanel::paintStatic (juce::Graphics& g, juce::Rectangle<float> area)
{
    paintTitle (g, area.removeFromTop (titleHeight));
}

void GlassPanel::paintTitle (juce::Graphics& g, juce::Rectangle<float> r)
{
    auto textArea = r.reduced (22.0f, 0.0f).withTrimmedTop (2.0f);

    if (titleIcon)
    {
        Icons::draw (g, *titleIcon, textArea.removeFromLeft (26.0f).withSizeKeepingCentre (26.0f, 26.0f),
                     juce::Colours::white, 1.6f);
        textArea.removeFromLeft (10.0f);
    }

    drawText (g, title.toUpperCase(), textArea, font (24.0f, Weight::Title), juce::Colours::white,
              juce::Justification::centredLeft);

    // separator under the title row
    juce::ColourGradient cg (Palette::panelBorder.withAlpha (0.0f), r.getX() + 8.0f, 0.0f,
                             Palette::panelBorder.withAlpha (0.0f), r.getRight() - 8.0f, 0.0f, false);
    cg.addColour (0.12, Palette::panelBorder.withAlpha (0.22f));
    cg.addColour (0.88, Palette::panelBorder.withAlpha (0.22f));
    g.setGradientFill (cg);
    g.fillRect (juce::Rectangle<float> (r.getX() + 8.0f, r.getBottom() - 1.0f, r.getWidth() - 16.0f, 1.0f));
    g.setColour (juce::Colours::black.withAlpha (0.25f));
    g.fillRect (juce::Rectangle<float> (r.getX() + 8.0f, r.getBottom(), r.getWidth() - 16.0f, 1.0f));
}

void GlassPanel::setEmphasis (bool e)
{
    if (e != emphasis)
    {
        emphasis = e;
        repaint();
    }
}

void GlassPanel::paint (juce::Graphics& g)
{
    if (! emphasis)
        return;

    auto r = getLocalBounds().toFloat().reduced (1.0f);
    juce::Path outline;
    outline.addRoundedRectangle (r, cornerRadius);

    g.saveState();
    g.reduceClipRegion (outline);
    for (int i = 0; i < 4; ++i)
    {
        g.setColour (Palette::glow.withAlpha (0.10f - (float) i * 0.02f));
        g.drawRoundedRectangle (r.reduced (2.0f + (float) i * 2.5f), cornerRadius - 1.0f, 3.0f);
    }
    g.restoreState();

    g.setColour (Palette::glowBright.withAlpha (0.9f));
    g.drawRoundedRectangle (r, cornerRadius, 1.6f);
}

void GlassPanel::paintGlassCard (juce::Graphics& g, juce::Rectangle<float> area, const juce::Image* backdrop,
                                 juce::Point<float> backdropOrigin, float backdropScale, float corner)
{
    juce::Path outline;
    outline.addRoundedRectangle (area, corner);

    // soft shadow + outer glow (layered strokes / gradients: much cheaper than blurring images)
    juce::DropShadow (juce::Colour (0xff06103a).withAlpha (0.30f), 18, { 0, 8 }).drawForRectangle (g, area.toNearestInt());

    for (int i = 5; i >= 1; --i)
    {
        g.setColour (Palette::glow.withAlpha (0.035f + 0.022f * (float) (5 - i)));
        g.strokePath (outline, juce::PathStrokeType ((float) i * 4.0f));
    }
    g.setColour (juce::Colours::white.withAlpha (0.22f));
    g.strokePath (outline, juce::PathStrokeType (3.0f));

    g.saveState();
    g.reduceClipRegion (outline);

    if (backdrop != nullptr && backdrop->isValid())
        g.drawImageTransformed (*backdrop, juce::AffineTransform::scale (1.0f / backdropScale).translated (backdropOrigin));

    // translucent navy tint: lighter towards the top so the silk shows through like frosted glass
    {
        juce::ColourGradient tint (juce::Colour (0xff253a70).withAlpha (0.70f), area.getX(), area.getY(),
                                   juce::Colour (0xff0a1736).withAlpha (0.86f), area.getX(), area.getBottom(), false);
        tint.addColour (0.35, juce::Colour (0xff18295a).withAlpha (0.78f));
        g.setGradientFill (tint);
        g.fillRect (area);
    }

    // diagonal sheen
    {
        juce::ColourGradient sheen (juce::Colours::white.withAlpha (0.12f), area.getX(), area.getY(),
                                    juce::Colours::white.withAlpha (0.0f), area.getX() + area.getWidth() * 0.4f, area.getY() + area.getHeight() * 0.7f, false);
        g.setGradientFill (sheen);
        g.fillRect (area);
    }

    // inner top light
    g.setGradientFill (juce::ColourGradient (Palette::glowBright.withAlpha (0.12f), area.getX(), area.getY(),
                                             Palette::glowBright.withAlpha (0.0f), area.getX(), area.getY() + 44.0f, false));
    g.fillRect (area.withHeight (44.0f));

    // inner rim light
    g.setColour (Palette::glow.withAlpha (0.25f));
    g.drawRoundedRectangle (area.reduced (2.5f), corner - 2.0f, 2.0f);
    g.restoreState();

    // bright frosted border
    juce::ColourGradient border (juce::Colour (0xfff2f7ff).withAlpha (0.95f), area.getX(), area.getY(),
                                 Palette::glowBright.withAlpha (0.55f), area.getRight(), area.getBottom(), false);
    border.addColour (0.45, juce::Colour (0xffbcd8ff).withAlpha (0.8f));
    g.setGradientFill (border);
    g.drawRoundedRectangle (area.reduced (0.75f), corner, 1.6f);
}

} // namespace voxa::ui
