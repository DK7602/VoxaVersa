#include "Meters.h"

namespace voxa::ui
{

//==============================================================================
LevelMeter::LevelMeter (LevelMeterSource& s) : source (s)
{
    setInterceptsMouseClicks (false, false);
    setOpaque (false);
}

float LevelMeter::dbToPosition (float db) noexcept
{
    static constexpr float knotsDb[]  = { 0.0f, -6.0f, -12.0f, -24.0f, -48.0f };
    static constexpr float knotsPos[] = { 0.0f, 0.22f, 0.44f, 0.70f, 1.0f };

    if (db >= 0.0f)
        return 0.0f;
    if (db <= -48.0f)
        return 1.0f;

    for (int i = 1; i < 5; ++i)
    {
        if (db >= knotsDb[i])
        {
            const float t = (db - knotsDb[i - 1]) / (knotsDb[i] - knotsDb[i - 1]);
            return knotsPos[i - 1] + t * (knotsPos[i] - knotsPos[i - 1]);
        }
    }
    return 1.0f;
}

int LevelMeter::segmentsFor (float db) const noexcept
{
    if (db <= -48.0f)
        return 0;
    const float fromBottom = 1.0f - dbToPosition (db);
    return juce::jlimit (0, numSegments, (int) std::ceil (fromBottom * (float) numSegments - 0.25f));
}

void LevelMeter::tick (float dt)
{
    bool changed = false;

    for (int ch = 0; ch < 2; ++ch)
    {
        const float peak = source.consumePeak (ch);
        const float db = peak > 1.0e-6f ? 20.0f * std::log10 (peak) : -100.0f;

        levelDb[(size_t) ch] = juce::jmax (db, levelDb[(size_t) ch] - 26.0f * dt);

        if (db >= peakDb[(size_t) ch])
        {
            peakDb[(size_t) ch] = db;
            peakAge[(size_t) ch] = 0.0f;
        }
        else
        {
            peakAge[(size_t) ch] += dt;
            if (peakAge[(size_t) ch] > 1.2f)
                peakDb[(size_t) ch] -= 30.0f * dt;
        }

        if (peak >= 0.999f)
            clipAge[(size_t) ch] = 0.0f;
        else
            clipAge[(size_t) ch] += dt;

        const int lit = segmentsFor (levelDb[(size_t) ch]);
        const int pk = segmentsFor (peakDb[(size_t) ch]);
        const bool clip = clipAge[(size_t) ch] < 1.5f;

        if (lit != drawnLit[(size_t) ch] || pk != drawnPeak[(size_t) ch] || clip != drawnClip[(size_t) ch])
            changed = true;
    }

    if (changed)
        repaint();
}

void LevelMeter::resized()
{
    auto r = getLocalBounds().toFloat();
    scaleArea = r.removeFromRight (r.getWidth() * 0.42f);
    r = r.reduced (0.0f, 8.0f);
    const float barW = juce::jmin (16.0f, (r.getWidth() - 6.0f) * 0.5f);
    const float x0 = r.getRight() - (barW * 2.0f + 6.0f);
    bars[0] = { x0, r.getY(), barW, r.getHeight() };
    bars[1] = { x0 + barW + 6.0f, r.getY(), barW, r.getHeight() };
}

void LevelMeter::paint (juce::Graphics& g)
{
    const float gap = 1.6f;

    for (int ch = 0; ch < 2; ++ch)
    {
        const auto bar = bars[(size_t) ch];
        const float segH = (bar.getHeight() - gap * (float) (numSegments - 1)) / (float) numSegments;
        const int lit = segmentsFor (levelDb[(size_t) ch]);
        const int pk = segmentsFor (peakDb[(size_t) ch]);
        const bool clip = clipAge[(size_t) ch] < 1.5f;

        drawnLit[(size_t) ch] = lit;
        drawnPeak[(size_t) ch] = pk;
        drawnClip[(size_t) ch] = clip;

        // housing
        g.setColour (juce::Colour (0xff030817).withAlpha (0.9f));
        g.fillRoundedRectangle (bar.expanded (2.0f, 3.0f), 3.0f);
        g.setColour (Palette::insetBorder.withAlpha (0.55f));
        g.drawRoundedRectangle (bar.expanded (2.0f, 3.0f), 3.0f, 0.8f);

        for (int i = 0; i < numSegments; ++i)
        {
            const float y = bar.getBottom() - (float) (i + 1) * segH - (float) i * gap;
            const auto seg = juce::Rectangle<float> (bar.getX(), y, bar.getWidth(), segH);
            const float t = (float) i / (float) (numSegments - 1);
            const bool isLit = i < lit || (i == pk - 1 && pk > 0);
            const bool top = i == numSegments - 1;

            if (isLit)
            {
                auto col = top && clip ? Palette::danger
                                       : juce::Colour (0xff1d5cff).interpolatedWith (juce::Colour (0xff5fb4ff), juce::jmin (1.0f, t * 1.4f))
                                                                  .interpolatedWith (juce::Colour (0xffc8ecff), juce::jmax (0.0f, t - 0.75f) * 3.0f);
                g.setColour (col.withAlpha (0.28f));
                g.fillRect (seg.expanded (1.5f, 1.0f));
                g.setColour (col);
                g.fillRect (seg);
                g.setColour (juce::Colours::white.withAlpha (0.25f));
                g.fillRect (seg.withHeight (juce::jmax (0.8f, segH * 0.3f)));
            }
            else
            {
                g.setColour ((top && clip) ? Palette::danger.withAlpha (0.8f) : juce::Colour (0xff17295a).withAlpha (0.75f));
                g.fillRect (seg);
            }
        }
    }

    // scale
    const auto f = font (15.0f);
    const auto& bar = bars[0];
    for (int db : { 0, -6, -12, -24, -48 })
    {
        const float y = bar.getY() + dbToPosition ((float) db) * bar.getHeight();
        drawText (g, juce::String (db), { scaleArea.getX() + 6.0f, y - 9.0f, scaleArea.getWidth() - 6.0f, 18.0f }, f,
                  Palette::textSoft, juce::Justification::centredLeft);
    }
}

//==============================================================================
WaveformScope::WaveformScope (ScopeBuffer& b) : buffer (b)
{
    setInterceptsMouseClicks (false, false);
    buildColourTable();
}

void WaveformScope::resized()
{
    numPoints = juce::jlimit (32, maxPoints, getWidth() / 3);
    std::fill (drawnAmps.begin(), drawnAmps.end(), -1.0f);
    background = {};
}

void WaveformScope::setHighlighted (bool h)
{
    if (h != highlighted)
    {
        highlighted = h;
        repaint();
    }
}

void WaveformScope::setPlaceholder (const juce::String& text)
{
    if (text != placeholder)
    {
        placeholder = text;
        repaint();
    }
}

void WaveformScope::setPlayPosition (double p)
{
    if (std::abs (p - playPosition) > 0.0005)
    {
        playPosition = p;
        repaint();
    }
}

void WaveformScope::tick()
{
    buffer.read (mins.data(), maxs.data(), numPoints);

    bool anySignal = false;
    float diff = 0.0f;
    for (int i = 0; i < numPoints; ++i)
    {
        const float a = juce::jmax (std::abs (mins[(size_t) i]), std::abs (maxs[(size_t) i]));
        const float shown = juce::jmin (1.0f, std::pow (juce::jmin (1.0f, a), 0.6f));
        amps[(size_t) i] = shown;
        anySignal = anySignal || shown > 0.004f;
        diff += std::abs (shown - drawnAmps[(size_t) i]);
    }

    if (! anySignal && silent)
        return;

    silent = ! anySignal;
    if (diff > 0.01f)
        repaint();
}


void WaveformScope::buildColourTable()
{
    // centre of each bar is near-white, fading through light blue to deep blue at its tips
    const juce::Colour core (0xfff0f8ff), mid (0xff86c2ff), body (0xff3f8dff), tip (0xff2563ec);
    for (int i = 0; i < 256; ++i)
    {
        const float t = (float) i / 255.0f;
        juce::Colour c;
        if (t < 0.3f)       c = core.interpolatedWith (mid, t / 0.3f);
        else if (t < 0.6f)  c = mid.interpolatedWith (body, (t - 0.3f) / 0.3f);
        else                c = body.interpolatedWith (tip, (t - 0.6f) / 0.4f).withMultipliedAlpha (1.0f - (t - 0.6f) * 0.5f);
        colourTable[(size_t) i] = c.getPixelARGB();
    }
    glowPixel = juce::Colour (0xff4f9bff).withAlpha (0.42f).getPixelARGB();
}

void WaveformScope::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat();
    const float phys = physicalScale (g);
    const int pw = juce::jmax (1, juce::roundToInt (r.getWidth() * phys));
    const int ph = juce::jmax (1, juce::roundToInt (r.getHeight() * phys));

    // static inset + centre line, cached at the physical resolution
    if (! background.isValid() || std::abs (backgroundScale - phys) > 0.001f
        || background.getWidth() != pw || background.getHeight() != ph)
    {
        backgroundScale = phys;
        background = juce::Image (juce::Image::ARGB, pw, ph, true, juce::SoftwareImageType());
        frame = juce::Image (juce::Image::ARGB, pw, ph, true, juce::SoftwareImageType());

        juce::Graphics bg (background);
        bg.addTransform (juce::AffineTransform::scale (phys));
        fillInset (bg, r, 10.0f);
        auto in = r.reduced (10.0f, 8.0f);
        juce::ColourGradient cg (Palette::glow.withAlpha (0.0f), in.getX(), in.getCentreY(), Palette::glow.withAlpha (0.0f), in.getRight(), in.getCentreY(), false);
        cg.addColour (0.5, Palette::glowBright.withAlpha (0.7f));
        bg.setGradientFill (cg);
        bg.fillRect (juce::Rectangle<float> (in.getX(), in.getCentreY() - 0.6f, in.getWidth(), 1.2f));
    }
    drawCachedImage (g, background, {}, backgroundScale);

    if (highlighted)
    {
        g.setColour (Palette::glow.withAlpha (0.25f));
        g.fillRoundedRectangle (r.reduced (2.0f), 9.0f);
        g.setColour (Palette::glowBright);
        g.drawRoundedRectangle (r.reduced (1.0f), 10.0f, 2.0f);
    }

    const auto inner = r.reduced (10.0f, 8.0f);
    if (silent && placeholder.isNotEmpty())
        drawText (g, placeholder, inner, font (15.0f), Palette::textDim);

    // Bars are written straight into a pixel buffer (no path rasterisation): a colour table gives
    // every bar a bright core and deep-blue tips, flanked by soft glow columns.
    frame.clear (frame.getBounds());
    {
        juce::Image::BitmapData bd (frame, juce::Image::BitmapData::readWrite);
        const auto in = inner * phys;
        const float step = in.getWidth() / (float) numPoints;
        const int lineW = juce::jmax (1, juce::roundToInt (step * 0.5f));
        const int cy = juce::roundToInt (in.getCentreY());
        const float halfH = in.getHeight() * 0.5f;
        const float edge = (float) numPoints * 0.14f;

        for (int i = 0; i < numPoints; ++i)
        {
            const float fi = (float) i;
            const float taper = juce::jmin (1.0f, juce::jmin (fi, (float) numPoints - 1.0f - fi) / edge);
            const float a = amps[(size_t) i] * (0.25f + 0.75f * taper * taper);
            drawnAmps[(size_t) i] = amps[(size_t) i];
            if (a < 0.01f)
                continue;

            const int x = (int) (in.getX() + fi * step);
            const int hh = juce::jlimit (1, ph / 2 - 1, juce::roundToInt (a * halfH));
            const int y0 = juce::jmax (0, cy - hh), y1 = juce::jmin (ph - 1, cy + hh);
            const float invH = 255.0f / (float) hh;

            for (int col = x - 1; col <= x + lineW; ++col)
            {
                if (! juce::isPositiveAndBelow (col, pw))
                    continue;

                const bool glowColumn = col < x || col >= x + lineW;
                for (int y = glowColumn ? y0 + hh / 4 : y0; y <= (glowColumn ? y1 - hh / 4 : y1); ++y)
                {
                    auto* p = reinterpret_cast<juce::PixelARGB*> (bd.getPixelPointer (col, y));
                    if (glowColumn)
                        p->blend (glowPixel);
                    else
                        p->set (colourTable[(size_t) juce::jmin (255, (int) ((float) std::abs (y - cy) * invH))]);
                }
            }
        }
    }
    g.setOpacity (1.0f);
    drawCachedImage (g, frame, {}, backgroundScale);

    if (playPosition >= 0.0)
    {
        const float x = inner.getX() + (float) playPosition * inner.getWidth();
        g.setColour (Palette::glowBright.withAlpha (0.25f));
        g.fillRect (juce::Rectangle<float> (x - 2.0f, inner.getY(), 4.0f, inner.getHeight()));
        g.setColour (juce::Colours::white.withAlpha (0.85f));
        g.fillRect (juce::Rectangle<float> (x - 0.5f, inner.getY(), 1.0f, inner.getHeight()));
    }
}

} // namespace voxa::ui
