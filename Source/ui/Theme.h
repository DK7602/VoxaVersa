#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace voxa::ui
{

//==============================================================================
/** The VoxaVersa colour palette (deep navy glass, luminous blue accents). */
namespace Palette
{
    inline const juce::Colour canvasDeep   { 0xff050b1f };
    inline const juce::Colour canvasMid    { 0xff1b3268 };
    inline const juce::Colour canvasLight  { 0xffd9e6fb };

    inline const juce::Colour panelFill    { 0xff0b1736 };
    inline const juce::Colour panelFillTop { 0xff17295a };
    inline const juce::Colour panelBorder  { 0xffa9c9ff };

    inline const juce::Colour glow         { 0xff3d8bff };
    inline const juce::Colour glowBright   { 0xff9fd4ff };
    inline const juce::Colour accent       { 0xff2f74ff };
    inline const juce::Colour accentDeep   { 0xff163fb8 };
    inline const juce::Colour cyan         { 0xff5fd0ff };

    inline const juce::Colour inset        { 0xff050c22 };
    inline const juce::Colour insetTop     { 0xff0a1636 };
    inline const juce::Colour insetBorder  { 0xff2b4a86 };

    inline const juce::Colour control      { 0xff13234d };
    inline const juce::Colour controlTop   { 0xff22376f };

    inline const juce::Colour text         { 0xfff3f7ff };
    inline const juce::Colour textSoft     { 0xffc3d3f0 };
    inline const juce::Colour textDim      { 0xff8296bd };
    inline const juce::Colour textFaint    { 0xff5a6f99 };

    inline const juce::Colour danger       { 0xffff5b72 };
    inline const juce::Colour success      { 0xff4fe3a6 };
    inline const juce::Colour warning      { 0xffffc75a };
    inline const juce::Colour record       { 0xffff3b5c };
}

//==============================================================================
enum class Weight { Regular, Medium, SemiBold, Title };

/** Embedded Barlow fonts (see docs/UI_FONTS.md). */
juce::Font font (float height, Weight weight = Weight::Regular);

/** Text asked for smaller than this (in design pixels) is drawn larger, up to 1.45x, so that it stays
    readable on screen when the editor is scaled down. The editor sets it from its scale. */
void setMinimumTextHeight (float designHeight) noexcept;
juce::Typeface::Ptr getTypeface (Weight weight);

//==============================================================================
/** Returns the physical pixels per logical unit for this graphics context. */
float physicalScale (juce::Graphics& g);

/** Draws an image that was rendered at `scale` physical pixels per unit into `area` (logical units). */
void drawCachedImage (juce::Graphics& g, const juce::Image& image, juce::Point<float> topLeft, float scale);

/** In-place separable box blur of an ARGB / RGB image (several passes approximate a gaussian). */
void blurImage (juce::Image& image, int radius, int passes = 3);

/** Soft glow around a filled shape. */
void drawGlow (juce::Graphics& g, const juce::Path& shape, juce::Colour colour, int radius);

/** Stroke with a soft multi-layer glow (cheap, no temporary images). */
void strokeGlow (juce::Graphics& g, const juce::Path& path, juce::Colour colour, float thickness,
                 float glowAmount = 1.0f, bool roundedEnds = true);

/** Dark recessed area (scope, lyrics editor, value boxes). */
void fillInset (juce::Graphics& g, juce::Rectangle<float> area, float corner, float alpha = 1.0f);

/** Glass button / pill background. `active` = lit blue state. */
void drawGlassControl (juce::Graphics& g, juce::Rectangle<float> area, float corner,
                       bool active, bool hover, bool pressed, bool enabled = true);

/** Single-line text helper. */
void drawText (juce::Graphics& g, const juce::String& text, juce::Rectangle<float> area,
               const juce::Font& f, juce::Colour colour,
               juce::Justification just = juce::Justification::centred);

/** Width of a single line of text. */
float textWidth (const juce::Font& f, const juce::String& text);

/** A brushed-chrome knob cap (rendered pixel by pixel, cache the result). */
juce::Image renderChromeCap (int diameterPx);

/** Formats seconds as m:ss. */
juce::String formatTime (double seconds);

/** Note name for a frequency, or "--" when unvoiced. */
juce::String noteNameForHz (float hz);

} // namespace voxa::ui
