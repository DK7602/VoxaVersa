#include "Theme.h"
#include "VoxaAssets.h"
#include "../core/MusicTheory.h"

#include <cmath>
#include <map>
#include <vector>

namespace voxa::ui
{

namespace
{
    struct FontSet : public juce::DeletedAtShutdown
    {
        FontSet()
        {
            regular  = load (VoxaAssets::BarlowSemiCondensedRegular_ttf,  VoxaAssets::BarlowSemiCondensedRegular_ttfSize);
            medium   = load (VoxaAssets::BarlowSemiCondensedMedium_ttf,   VoxaAssets::BarlowSemiCondensedMedium_ttfSize);
            semibold = load (VoxaAssets::BarlowSemiCondensedSemiBold_ttf, VoxaAssets::BarlowSemiCondensedSemiBold_ttfSize);
            title    = load (VoxaAssets::BarlowMedium_ttf,                VoxaAssets::BarlowMedium_ttfSize);
        }

        ~FontSet() override { clearSingletonInstance(); }

        static juce::Typeface::Ptr load (const char* data, int size)
        {
            return juce::Typeface::createSystemTypefaceFor (data, (size_t) size);
        }

        juce::Typeface::Ptr get (Weight w) const
        {
            switch (w)
            {
                case Weight::Medium:   return medium;
                case Weight::SemiBold: return semibold;
                case Weight::Title:    return title;
                case Weight::Regular:
                default:               return regular;
            }
        }

        juce::Typeface::Ptr regular, medium, semibold, title;
        std::map<int, juce::Font> cache;   // fonts are reused: constructing one allocates

        JUCE_DECLARE_SINGLETON_INLINE (FontSet, false)
    };
}

juce::Typeface::Ptr getTypeface (Weight weight)
{
    return FontSet::getInstance()->get (weight);
}

juce::Font font (float height, Weight weight)
{
    auto* set = FontSet::getInstance();
    const int key = juce::roundToInt (height * 16.0f) * 8 + (int) weight;

    if (auto it = set->cache.find (key); it != set->cache.end())
        return it->second;

    auto tf = set->get (weight);
    auto f = tf != nullptr ? juce::Font (juce::FontOptions (tf).withHeight (height))
                           : juce::Font (juce::FontOptions (height));
    if (set->cache.size() < 512)
        set->cache.emplace (key, f);
    return f;
}

//==============================================================================
float physicalScale (juce::Graphics& g)
{
    return juce::jmax (0.1f, g.getInternalContext().getPhysicalPixelScaleFactor());
}

void drawCachedImage (juce::Graphics& g, const juce::Image& image, juce::Point<float> topLeft, float scale)
{
    if (! image.isValid())
        return;

    // The image was rendered at the physical resolution, so the combined transform is a pure
    // translation. With low resampling quality the software renderer then takes its fast blit path
    // (snapping to the nearest pixel) instead of interpolating every pixel.
    g.saveState();
    g.setImageResamplingQuality (juce::Graphics::lowResamplingQuality);
    g.drawImageTransformed (image, juce::AffineTransform::scale (1.0f / scale).translated (topLeft));
    g.restoreState();
}

void blurImage (juce::Image& image, int radius, int passes)
{
    if (radius < 1 || ! image.isValid())
        return;

    juce::Image::BitmapData bd (image, juce::Image::BitmapData::readWrite);
    const int w = bd.width, h = bd.height, ps = bd.pixelStride, ls = bd.lineStride;
    const int channels = juce::jmin (ps, 4);
    const int window = radius * 2 + 1;
    std::vector<juce::uint8> line ((size_t) juce::jmax (w, h) * (size_t) ps);

    auto blurLine = [&] (juce::uint8* start, int count, int stride)
    {
        for (int i = 0; i < count; ++i)
            std::memcpy (&line[(size_t) (i * ps)], start + i * stride, (size_t) ps);

        for (int c = 0; c < channels; ++c)
        {
            int sum = 0;
            for (int k = -radius; k <= radius; ++k)
                sum += line[(size_t) (juce::jlimit (0, count - 1, k) * ps + c)];

            for (int i = 0; i < count; ++i)
            {
                start[i * stride + c] = (juce::uint8) (sum / window);
                const int addIdx = juce::jmin (i + radius + 1, count - 1);
                const int subIdx = juce::jmax (i - radius, 0);
                sum += line[(size_t) (addIdx * ps + c)] - line[(size_t) (subIdx * ps + c)];
            }
        }
    };

    for (int pass = 0; pass < passes; ++pass)
    {
        for (int y = 0; y < h; ++y)
            blurLine (bd.getLinePointer (y), w, ps);

        for (int x = 0; x < w; ++x)
            blurLine (bd.data + x * ps, h, ls);
    }
}

void drawGlow (juce::Graphics& g, const juce::Path& shape, juce::Colour colour, int radius)
{
    if (radius > 0)
        juce::DropShadow (colour, radius, {}).drawForPath (g, shape);
}

void strokeGlow (juce::Graphics& g, const juce::Path& path, juce::Colour colour, float thickness,
                 float glowAmount, bool roundedEnds)
{
    const auto cap = roundedEnds ? juce::PathStrokeType::rounded : juce::PathStrokeType::butt;
    const auto join = juce::PathStrokeType::curved;

    struct Layer { float width, alpha; };
    static constexpr Layer layers[] = { { 5.0f, 0.05f }, { 3.4f, 0.09f }, { 2.2f, 0.18f } };

    if (glowAmount > 0.0f)
    {
        for (auto& l : layers)
        {
            g.setColour (colour.withMultipliedAlpha (l.alpha * glowAmount));
            g.strokePath (path, juce::PathStrokeType (thickness * l.width, join, cap));
        }
    }

    g.setColour (colour);
    g.strokePath (path, juce::PathStrokeType (thickness, join, cap));
    g.setColour (colour.interpolatedWith (juce::Colours::white, 0.6f).withMultipliedAlpha (0.9f));
    g.strokePath (path, juce::PathStrokeType (thickness * 0.4f, join, cap));
}

void fillInset (juce::Graphics& g, juce::Rectangle<float> area, float corner, float alpha)
{
    g.setGradientFill (juce::ColourGradient (Palette::inset.withMultipliedAlpha (alpha), area.getX(), area.getY(),
                                             Palette::insetTop.withMultipliedAlpha (alpha), area.getX(), area.getBottom(), false));
    g.fillRoundedRectangle (area, corner);

    // inner shadow along the top edge
    juce::Path p;
    p.addRoundedRectangle (area, corner);
    g.saveState();
    g.reduceClipRegion (p);
    g.setColour (juce::Colours::black.withAlpha (0.35f * alpha));
    g.drawRoundedRectangle (area.translated (0.0f, 1.5f).expanded (0.5f), corner, 2.5f);
    g.restoreState();

    g.setGradientFill (juce::ColourGradient (Palette::insetBorder.withAlpha (0.45f * alpha), area.getX(), area.getY(),
                                             Palette::panelBorder.withAlpha (0.28f * alpha), area.getX(), area.getBottom(), false));
    g.drawRoundedRectangle (area.reduced (0.5f), corner, 1.0f);
}

void drawGlassControl (juce::Graphics& g, juce::Rectangle<float> area, float corner,
                       bool active, bool hover, bool pressed, bool enabled)
{
    const float alpha = enabled ? 1.0f : 0.45f;

    if (active)
    {
        juce::Path outline;
        outline.addRoundedRectangle (area, corner);
        g.setColour (Palette::glow.withAlpha (0.10f * alpha));
        g.strokePath (outline, juce::PathStrokeType (7.0f));
        g.setColour (Palette::glow.withAlpha (0.22f * alpha));
        g.strokePath (outline, juce::PathStrokeType (3.5f));

        auto top = juce::Colour (0xff4a8dff).withMultipliedAlpha (alpha);
        auto bottom = juce::Colour (0xff1840b8).withMultipliedAlpha (alpha);
        if (pressed) { top = top.darker (0.2f); bottom = bottom.darker (0.2f); }
        else if (hover) { top = top.brighter (0.12f); }

        g.setGradientFill (juce::ColourGradient (top, area.getX(), area.getY(), bottom, area.getX(), area.getBottom(), false));
        g.fillRoundedRectangle (area, corner);

        // inner sheen
        auto sheen = area.reduced (1.5f).withHeight (area.getHeight() * 0.48f);
        g.setGradientFill (juce::ColourGradient (juce::Colours::white.withAlpha (0.22f * alpha), sheen.getX(), sheen.getY(),
                                                 juce::Colours::white.withAlpha (0.0f), sheen.getX(), sheen.getBottom(), false));
        g.fillRoundedRectangle (sheen, juce::jmax (1.0f, corner - 1.5f));

        g.setColour (Palette::glowBright.withAlpha (0.95f * alpha));
        g.drawRoundedRectangle (area.reduced (0.5f), corner, 1.4f);
        return;
    }

    auto top = Palette::controlTop.withAlpha (0.92f * alpha);
    auto bottom = Palette::control.withAlpha (0.92f * alpha);
    if (pressed) { top = top.darker (0.25f); bottom = bottom.darker (0.25f); }
    else if (hover) { top = top.brighter (0.18f); bottom = bottom.brighter (0.12f); }

    g.setGradientFill (juce::ColourGradient (top, area.getX(), area.getY(), bottom, area.getX(), area.getBottom(), false));
    g.fillRoundedRectangle (area, corner);

    auto sheen = area.reduced (1.5f).withHeight (area.getHeight() * 0.45f);
    g.setGradientFill (juce::ColourGradient (juce::Colours::white.withAlpha (0.07f * alpha), sheen.getX(), sheen.getY(),
                                             juce::Colours::white.withAlpha (0.0f), sheen.getX(), sheen.getBottom(), false));
    g.fillRoundedRectangle (sheen, juce::jmax (1.0f, corner - 1.5f));

    g.setGradientFill (juce::ColourGradient (Palette::panelBorder.withAlpha ((hover ? 0.55f : 0.32f) * alpha), area.getX(), area.getY(),
                                             Palette::panelBorder.withAlpha ((hover ? 0.30f : 0.14f) * alpha), area.getX(), area.getBottom(), false));
    g.drawRoundedRectangle (area.reduced (0.5f), corner, 1.0f);
}

void drawText (juce::Graphics& g, const juce::String& text, juce::Rectangle<float> area,
               const juce::Font& f, juce::Colour colour, juce::Justification just)
{
    g.setFont (f);
    g.setColour (colour);
    g.drawText (text, area, just, true);
}

float textWidth (const juce::Font& f, const juce::String& text)
{
    return juce::GlyphArrangement::getStringWidth (f, text);
}

//==============================================================================
juce::Image renderChromeCap (int d)
{
    d = juce::jmax (8, d);
    juce::Image img (juce::Image::ARGB, d, d, true);
    juce::Image::BitmapData bd (img, juce::Image::BitmapData::writeOnly);

    const float R = (float) d * 0.5f;
    const float lightAngle = -2.35f;   // light from the upper left

    auto hash = [] (int n)
    {
        n = (n << 13) ^ n;
        return 1.0f - (float) ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f;
    };

    for (int y = 0; y < d; ++y)
    {
        for (int x = 0; x < d; ++x)
        {
            const float dx = ((float) x + 0.5f - R) / R;
            const float dy = ((float) y + 0.5f - R) / R;
            const float r = std::sqrt (dx * dx + dy * dy);
            const float coverage = juce::jlimit (0.0f, 1.0f, (1.0f - r) * R + 0.5f);

            if (coverage <= 0.0f)
            {
                bd.setPixelColour (x, y, juce::Colours::transparentBlack);
                continue;
            }

            const float theta = std::atan2 (dy, dx);

            // conic brushed-metal reflections
            float L = 0.60f
                    + 0.20f * std::cos (2.0f * (theta - 0.70f))
                    + 0.10f * std::cos (4.0f * theta + 0.9f)
                    + 0.16f * std::pow (juce::jmax (0.0f, std::cos (2.0f * (theta + 0.35f))), 12.0f)
                    + 0.08f * std::pow (juce::jmax (0.0f, std::cos (theta - lightAngle)), 3.0f);

            // concentric brushing
            const int ring = (int) (r * R * 2.2f);
            L += 0.035f * hash (ring) + 0.02f * hash (ring * 7 + (int) ((theta + 3.2f) * 40.0f));

            // gentle dome shading
            L += 0.08f * (-dy) * (1.0f - r);

            // bevelled rim
            const float rim = juce::jlimit (0.0f, 1.0f, (r - 0.86f) / 0.12f);
            L += rim * 0.32f * std::cos (theta - lightAngle);
            L -= rim * 0.10f;

            // centre dimple
            if (r < 0.14f)
                L += 0.05f * std::cos (theta - lightAngle) * (r / 0.14f);

            L = juce::jlimit (0.04f, 1.0f, L);

            const float blueTint = 0.06f + 0.10f * juce::jmax (0.0f, dy);
            const float red   = juce::jlimit (0.0f, 1.0f, L * 0.90f);
            const float green = juce::jlimit (0.0f, 1.0f, L * 0.94f + blueTint * 0.2f);
            const float blue  = juce::jlimit (0.0f, 1.0f, L * 1.00f + blueTint);

            bd.setPixelColour (x, y, juce::Colour::fromFloatRGBA (red, green, blue, coverage));
        }
    }

    return img;
}

juce::String formatTime (double seconds)
{
    const int s = juce::jmax (0, (int) std::floor (seconds));
    return juce::String (s / 60) + ":" + juce::String (s % 60).paddedLeft ('0', 2);
}

juce::String noteNameForHz (float hz)
{
    if (hz <= 1.0f || ! std::isfinite (hz))
        return "--";
    return music::noteName (juce::roundToInt (music::hzToMidi (hz)));
}

} // namespace voxa::ui
