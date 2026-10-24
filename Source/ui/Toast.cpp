#include "Toast.h"
#include "Icons.h"

namespace voxa::ui
{

ToastOverlay::ToastOverlay()
{
    setVisible (false);
    setAlwaysOnTop (true);
    setTitle ("Status message");
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
}

void ToastOverlay::show (const juce::String& m, ToastKind k)
{
    message = m;
    kind = k;
    age = 0.0f;
    holdTime = k == ToastKind::Error ? 6.5f : 4.0f;

    const float w = juce::jlimit (240.0f, 900.0f, textWidth (font (17.0f, Weight::Medium), message) + 74.0f);
    setBounds (juce::Rectangle<int> ((int) w, 44).withCentre ({ anchor.x, anchor.y + 22 }));
    setVisible (true);
    toFront (false);
    setDescription (message);
    repaint();

    if (auto* handler = getAccessibilityHandler())
        handler->postAnnouncement (message, k == ToastKind::Error ? juce::AccessibilityHandler::AnnouncementPriority::high
                                                                  : juce::AccessibilityHandler::AnnouncementPriority::medium);
}

void ToastOverlay::tick (float dt)
{
    if (! isVisible())
        return;

    age += dt;
    const float fadeIn = juce::jmin (1.0f, age / 0.15f);
    const float fadeOut = juce::jlimit (0.0f, 1.0f, 1.0f - (age - holdTime) / 0.6f);
    alpha = juce::jmin (fadeIn, fadeOut);

    if (alpha <= 0.0f)
    {
        setVisible (false);
        drawnAlpha = -1.0f;
        return;
    }

    if (std::abs (alpha - drawnAlpha) > 0.02f)
        repaint();
}

void ToastOverlay::paint (juce::Graphics& g)
{
    drawnAlpha = alpha;
    g.setOpacity (1.0f);
    auto r = getLocalBounds().toFloat().reduced (2.0f);

    const auto accent = kind == ToastKind::Error ? Palette::danger : (kind == ToastKind::Success ? Palette::success : Palette::glow);

    g.setColour (accent.withAlpha (0.25f * alpha));
    g.fillRoundedRectangle (r.expanded (1.5f), 12.0f);
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xff1a2c5c).withAlpha (0.96f * alpha), r.getX(), r.getY(),
                                             juce::Colour (0xff081230).withAlpha (0.96f * alpha), r.getX(), r.getBottom(), false));
    g.fillRoundedRectangle (r, 11.0f);
    g.setColour (accent.withAlpha (alpha));
    g.drawRoundedRectangle (r.reduced (0.5f), 11.0f, 1.4f);

    auto content = r.reduced (14.0f, 0.0f);
    const auto icon = kind == ToastKind::Error ? Icon::Info : (kind == ToastKind::Success ? Icon::Sparkle : Icon::Info);
    Icons::draw (g, icon, content.removeFromLeft (20.0f).withSizeKeepingCentre (18.0f, 18.0f), accent.withAlpha (alpha), 1.8f);
    content.removeFromLeft (10.0f);
    drawText (g, message, content, font (17.0f, Weight::Medium), juce::Colours::white.withAlpha (alpha), juce::Justification::centredLeft);
}

void ToastOverlay::mouseDown (const juce::MouseEvent&)
{
    setVisible (false);
}

} // namespace voxa::ui
