#include "GlassPanel.h"

namespace voxa::ui
{

GlassPanel::GlassPanel (const juce::String& t, std::optional<Icon> icon)
    : title (t), titleIcon (icon)
{
    setTitle (t);
    setFocusContainerType (FocusContainerType::keyboardFocusContainer);
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

    // outer glow + shadow
    drawGlow (g, outline, Palette::glow.withAlpha (0.45f), 12);
    {
        juce::Path shadow;
        shadow.addRoundedRectangle (area.translated (0.0f, 6.0f), corner);
        drawGlow (g, shadow, juce::Colours::black.withAlpha (0.35f), 14);
    }

    g.saveState();
    g.reduceClipRegion (outline);

    if (backdrop != nullptr && backdrop->isValid())
        g.drawImageTransformed (*backdrop, juce::AffineTransform::scale (1.0f / backdropScale).translated (backdropOrigin));

    // navy tint
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xff1a2d60).withAlpha (0.80f), area.getX(), area.getY(),
                                             juce::Colour (0xff09142f).withAlpha (0.90f), area.getX(), area.getBottom(), false));
    g.fillRect (area);

    // diagonal sheen
    {
        juce::ColourGradient sheen (juce::Colours::white.withAlpha (0.07f), area.getX(), area.getY(),
                                    juce::Colours::white.withAlpha (0.0f), area.getX() + area.getWidth() * 0.35f, area.getY() + area.getHeight() * 0.6f, false);
        g.setGradientFill (sheen);
        g.fillRect (area);
    }

    // inner top light
    g.setGradientFill (juce::ColourGradient (Palette::glowBright.withAlpha (0.10f), area.getX(), area.getY(),
                                             Palette::glowBright.withAlpha (0.0f), area.getX(), area.getY() + 40.0f, false));
    g.fillRect (area.withHeight (40.0f));

    // inner edge darkening
    g.setColour (juce::Colours::black.withAlpha (0.25f));
    g.drawRoundedRectangle (area.reduced (2.0f), corner - 1.5f, 2.0f);
    g.restoreState();

    // border
    juce::ColourGradient border (Palette::panelBorder.withAlpha (0.85f), area.getX(), area.getY(),
                                 Palette::panelBorder.withAlpha (0.35f), area.getRight(), area.getBottom(), false);
    border.addColour (0.5, Palette::glow.withAlpha (0.55f));
    g.setGradientFill (border);
    g.drawRoundedRectangle (area.reduced (0.6f), corner, 1.3f);
}

} // namespace voxa::ui
