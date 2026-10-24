#pragma once

#include "Theme.h"
#include "Icons.h"
#include <optional>

namespace voxa::ui
{

//==============================================================================
/** Glass pill button with optional icon. Lit (blue glow) when its toggle state is on. */
class GlassButton : public juce::Button
{
public:
    enum class Style
    {
        Glass,      // dark glass, lit when on
        Flat,       // text only until hovered / on (filter tabs)
        Primary     // always lit (call-to-action)
    };

    explicit GlassButton (const juce::String& buttonText = {}, std::optional<Icon> buttonIcon = std::nullopt, Style buttonStyle = Style::Glass);

    void setIcon (std::optional<Icon> newIcon)          { icon = newIcon; repaint(); }
    void setStyle (Style s)                             { style = s; repaint(); }
    void setCornerRadius (float r)                      { corner = r; repaint(); }
    void setFontHeight (float h)                        { fontHeight = h; repaint(); }
    void setIconColour (std::optional<juce::Colour> c)  { iconColour = c; repaint(); }
    void setTextColour (std::optional<juce::Colour> c)  { textColour = c; repaint(); }
    void setTrailingIcon (std::optional<Icon> i)        { trailingIcon = i; repaint(); }
    void setProgress (float p)                          { if (! juce::approximatelyEqual (p, progress)) { progress = p; repaint(); } }

    void paintButton (juce::Graphics&, bool isHighlighted, bool isDown) override;

private:
    std::optional<Icon> icon, trailingIcon;
    Style style;
    float corner = 7.0f, fontHeight = 16.0f, progress = -1.0f;
    std::optional<juce::Colour> iconColour, textColour;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GlassButton)
};

//==============================================================================
/** Icon-only button. */
class IconButton : public juce::Button
{
public:
    enum class Background { None, Rounded, Circle };

    IconButton (const juce::String& name, Icon icon, Background bg = Background::None);

    void setIcon (Icon newIcon)                 { icon = newIcon; repaint(); }
    void setIconScale (float s)                 { iconScale = s; repaint(); }
    void setStrokeWidth (float w)               { strokeWidth = w; repaint(); }
    void setColour (juce::Colour c)             { colour = c; repaint(); }
    void setLitWhenOn (bool shouldLight)        { litWhenOn = shouldLight; repaint(); }

    void paintButton (juce::Graphics&, bool isHighlighted, bool isDown) override;

private:
    Icon icon;
    Background background;
    float iconScale = 0.6f, strokeWidth = 1.8f;
    juce::Colour colour { Palette::textSoft };
    bool litWhenOn = true;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (IconButton)
};

//==============================================================================
/** The round glowing power button. */
class PowerButton : public juce::Button
{
public:
    PowerButton();
    void paintButton (juce::Graphics&, bool isHighlighted, bool isDown) override;
};

//==============================================================================
/** Header mode tab with icon, title and subtitle. */
class ModeTab : public juce::Button
{
public:
    ModeTab (const juce::String& title, const juce::String& subtitle, Icon icon);
    void paintButton (juce::Graphics&, bool isHighlighted, bool isDown) override;

private:
    juce::String subtitle;
    Icon icon;
};

//==============================================================================
/** Square icon tile with a caption underneath (Pitch Preserve, Formant Shift, ...). */
class FeatureTile : public juce::Button
{
public:
    FeatureTile (const juce::String& caption, Icon icon);
    void paintButton (juce::Graphics&, bool isHighlighted, bool isDown) override;

private:
    Icon icon;
};

//==============================================================================
/** Text tab used by the Advanced panel and the grid header. */
class TabButton : public juce::Button
{
public:
    explicit TabButton (const juce::String& buttonText);
    void setFontHeight (float h) { fontHeight = h; repaint(); }
    float getIdealWidth() const;
    void paintButton (juce::Graphics&, bool isHighlighted, bool isDown) override;

private:
    float fontHeight = 15.0f;
};

} // namespace voxa::ui
