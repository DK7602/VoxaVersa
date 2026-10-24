#include "CentreDisplay.h"
#include "core/MusicTheory.h"

namespace voxa::ui
{

CentreDisplay::CentreDisplay (EditorContext& c) : ctx (c)
{
    setInterceptsMouseClicks (false, false);
    setTitle ("Voice orb");
    setDescription ("Shows the detected input pitch, the output pitch, the key and the voice engine status");
    updateReadouts();
}

void CentreDisplay::resized()
{
    orbArea = { 94.0f, 8.0f, 424.0f, 424.0f };
    wordmarkArea = { 70.0f, 384.0f, 472.0f, 106.0f };
    leftPill   = { 6.0f, 140.0f, 80.0f, 64.0f };
    leftPill2  = { 6.0f, 214.0f, 80.0f, 64.0f };
    rightPill  = { (float) getWidth() - 86.0f, 140.0f, 80.0f, 64.0f };
    rightPill2 = { (float) getWidth() - 86.0f, 214.0f, 80.0f, 64.0f };
    haloImage = {};
    for (auto& p : pills)
        p.dirty = true;

    // tiles covering the halo ring (0.9 .. 1.3 x the orb radius)
    ringTiles.clear();
    const auto halo = orbArea.expanded (orbArea.getWidth() * 0.15f);
    const auto c = orbArea.getCentre();
    const float radius = orbArea.getWidth() * 0.5f;
    const int grid = 8;
    const float tw = halo.getWidth() / (float) grid, th = halo.getHeight() / (float) grid;
    for (int ty = 0; ty < grid; ++ty)
    {
        for (int tx = 0; tx < grid; ++tx)
        {
            const juce::Rectangle<float> tile (halo.getX() + (float) tx * tw, halo.getY() + (float) ty * th, tw, th);
            const float nearX = juce::jlimit (tile.getX(), tile.getRight(), c.x), nearY = juce::jlimit (tile.getY(), tile.getBottom(), c.y);
            const float farX = std::abs (tile.getX() - c.x) > std::abs (tile.getRight() - c.x) ? tile.getX() : tile.getRight();
            const float farY = std::abs (tile.getY() - c.y) > std::abs (tile.getBottom() - c.y) ? tile.getY() : tile.getBottom();
            const float minDist = c.getDistanceFrom ({ nearX, nearY });
            const float maxDist = c.getDistanceFrom ({ farX, farY });
            if (maxDist > radius * 0.88f && minDist < radius * 1.3f)
                ringTiles.add (tile.getSmallestIntegerContainer());
        }
    }

    const float spacing = orbArea.getWidth() * 0.028f;
    barsArea = juce::Rectangle<float> (spacing * (float) (numBars + 2), orbArea.getHeight() * 0.42f)
                   .withCentre ({ orbArea.getCentreX(), orbArea.getCentreY() + orbArea.getHeight() * 0.025f })
                   .getSmallestIntegerContainer();
}

void CentreDisplay::updateReadouts()
{
    auto& p = ctx.processor;
    const float inHz = p.detectedPitchHz.load();
    const float outHz = p.targetPitchHz.load();

    auto newIn = noteNameForHz (inHz);
    auto newOut = noteNameForHz (outHz > 1.0f ? outHz : 0.0f);

    juce::String newKey, newKeySub;
    if (ctx.getChoice (ParamID::key) == 0)
    {
        const int root = p.detectedKeyRoot.load();
        const int scale = p.detectedScale.load();
        if (root >= 0)
        {
            newKey = music::getKeyNames()[root];
            newKeySub = juce::isPositiveAndBelow (scale, Choices::scale().size()) ? Choices::scale()[scale] : juce::String();
            newKeySub = newKeySub.replace ("Natural Minor", "Minor") + " (auto)";
        }
        else
        {
            newKey = "Auto";
            newKeySub = "listening...";
        }
    }
    else
    {
        newKey = Choices::key()[ctx.getChoice (ParamID::key)];
        newKeySub = Choices::scale()[ctx.getChoice (ParamID::scale)].replace ("Natural Minor", "Minor");
    }

    juce::String newEngine, newEngineSub;
    bool lit = false;
    switch (p.getNeuralStatus())
    {
        case VoxaVersaProcessor::NeuralStatus::Active:          newEngine = "AI";  newEngineSub = "neural voice"; lit = true; break;
        case VoxaVersaProcessor::NeuralStatus::Loading:         newEngine = "AI";  newEngineSub = "loading..."; break;
        case VoxaVersaProcessor::NeuralStatus::Error:           newEngine = "DSP"; newEngineSub = "AI error"; break;
        case VoxaVersaProcessor::NeuralStatus::NoModelForVoice: newEngine = "DSP"; newEngineSub = "no AI model"; break;
        case VoxaVersaProcessor::NeuralStatus::Unavailable:
        default:                                                newEngine = "DSP"; newEngineSub = ctx.getBool (ParamID::aiEngine) ? "AI unavailable" : "AI off"; break;
    }

    if (newIn != inNote || newOut != outNote || newKey != keyText || newKeySub != keySub
        || newEngine != engineText || newEngineSub != engineSub || lit != engineLit)
    {
        inNote = newIn;
        outNote = newOut;
        keyText = newKey;
        keySub = newKeySub;
        engineText = newEngine;
        engineSub = newEngineSub;
        engineLit = lit;
        for (auto& p : pills)
            p.dirty = true;
        repaint (leftPill.getUnion (leftPill2).expanded (8.0f).toNearestInt());
        repaint (rightPill.getUnion (rightPill2).expanded (8.0f).toNearestInt());
    }
}

void CentreDisplay::tick (float dt)
{
    if (++readoutCounter >= 3)
    {
        readoutCounter = 0;
        updateReadouts();
    }

    auto& p = ctx.processor;
    const float rms = 0.5f * (p.outputMeter.getRms (0) + p.outputMeter.getRms (1));
    const float target = juce::jlimit (0.0f, 1.0f, std::pow (rms * 5.0f, 0.7f));
    level += (target - level) * (target > level ? 0.5f : 0.08f);
    phase = std::fmod (phase + dt * juce::MathConstants<float>::twoPi * 0.22f, juce::MathConstants<float>::twoPi);

    // voice bars from the most recent output samples
    p.outputScope.read (scopeMin.data(), scopeMax.data(), (int) scopeMin.size());
    const int per = (int) scopeMin.size() / numBars;
    float barChange = 0.0f;
    for (int i = 0; i < numBars; ++i)
    {
        float a = 0.0f;
        for (int k = 0; k < per; ++k)
        {
            const auto idx = (size_t) (i * per + k);
            a = juce::jmax (a, std::abs (scopeMin[idx]), std::abs (scopeMax[idx]));
        }
        const float centre = 1.0f - std::abs ((float) i - (float) (numBars - 1) * 0.5f) / ((float) numBars * 0.5f);
        const float v = juce::jlimit (0.0f, 1.0f, std::pow (a, 0.6f) * (0.35f + 0.65f * centre));
        bars[(size_t) i] = juce::jmax (v, bars[(size_t) i] * 0.82f);
        barChange += std::abs (bars[(size_t) i] - drawnBars[(size_t) i]);
    }

    const float glow = 0.42f + 0.12f * std::sin (phase) + 0.46f * level;
    if (std::abs (glow - drawnGlow) > 0.012f)
    {
        drawnGlow = glow;
        for (auto& tile : ringTiles)
            repaint (tile);
    }
    if (barChange > 0.05f)
        repaint (barsArea);
}

void CentreDisplay::paintPill (juce::Graphics& g, juce::Rectangle<float> r, const juce::String& label,
                               const juce::String& value, const juce::String& sub, bool lit)
{
    g.setGradientFill (juce::ColourGradient (juce::Colour (0xe0182b5c), r.getX(), r.getY(), juce::Colour (0xe8081330), r.getX(), r.getBottom(), false));
    g.fillRoundedRectangle (r, 10.0f);
    g.setColour (lit ? Palette::glowBright.withAlpha (0.9f) : Palette::panelBorder.withAlpha (0.45f));
    g.drawRoundedRectangle (r.reduced (0.5f), 10.0f, lit ? 1.4f : 1.0f);

    auto inner = r.reduced (4.0f, 5.0f);
    drawText (g, label, inner.removeFromTop (14.0f), font (12.5f, Weight::SemiBold), Palette::textDim);
    drawText (g, value, inner.removeFromTop (26.0f), font (25.0f, Weight::SemiBold), lit ? juce::Colours::white : Palette::text);
    drawText (g, sub, inner, font (12.5f), Palette::textSoft);
}

void CentreDisplay::drawPill (juce::Graphics& g, int index, juce::Rectangle<float> area, const juce::String& label,
                              const juce::String& value, const juce::String& sub, bool lit, float phys)
{
    if (! g.clipRegionIntersects (area.getSmallestIntegerContainer()))
        return;

    auto& cache = pills[(size_t) index];
    if (cache.dirty || std::abs (cache.scale - phys) > 0.001f || ! cache.image.isValid())
    {
        cache.dirty = false;
        cache.scale = phys;
        cache.image = juce::Image (juce::Image::ARGB, juce::roundToInt (area.getWidth() * phys) + 2,
                                   juce::roundToInt (area.getHeight() * phys) + 2, true);
        juce::Graphics pg (cache.image);
        pg.addTransform (juce::AffineTransform::scale (phys));
        paintPill (pg, area.withPosition (1.0f / phys, 1.0f / phys), label, value, sub, lit);
    }
    g.setOpacity (1.0f);
    drawCachedImage (g, cache.image, area.getTopLeft() - juce::Point<float> (1.0f / phys, 1.0f / phys), phys);
}

void CentreDisplay::paint (juce::Graphics& g)
{
    const float phys = physicalScale (g);
    const auto haloArea = orbArea.expanded (orbArea.getWidth() * 0.15f);

    if (! haloImage.isValid() || std::abs (haloScale - phys) > 0.01f)
    {
        haloScale = phys;
        const int w = juce::roundToInt (haloArea.getWidth() * phys), h = juce::roundToInt (haloArea.getHeight() * phys);
        haloImage = juce::Image (juce::Image::ARGB, juce::jmax (1, w), juce::jmax (1, h), true);
        juce::Graphics hg (haloImage);
        const float cx = (float) w * 0.5f, cy = (float) h * 0.5f;
        const float R = orbArea.getWidth() * 0.5f * phys;
        juce::ColourGradient cg (juce::Colours::transparentWhite, cx, cy, juce::Colours::transparentWhite, cx + R * 1.3f, cy, true);
        cg.addColour (0.70, juce::Colour (0x00bfe3ff));
        cg.addColour (0.76, juce::Colour (0x553d8bff));
        cg.addColour (0.775, juce::Colour (0xccd8f0ff));
        cg.addColour (0.80, juce::Colour (0x804fa8ff));
        cg.addColour (0.88, juce::Colour (0x303d8bff));
        hg.setGradientFill (cg);
        hg.fillEllipse (0.0f, 0.0f, (float) w, (float) h);
    }

    g.setOpacity (juce::jlimit (0.0f, 1.0f, drawnGlow));
    drawCachedImage (g, haloImage, haloArea.getTopLeft(), phys);
    g.setOpacity (1.0f);

    // voice bars across the orb's centre
    {
        const float cx = orbArea.getCentreX(), cy = orbArea.getCentreY() + orbArea.getHeight() * 0.025f;
        const float spacing = orbArea.getWidth() * 0.028f;
        const float maxH = orbArea.getHeight() * 0.17f;
        const float barW = spacing * 0.42f;

        for (int i = 0; i < numBars; ++i)
        {
            drawnBars[(size_t) i] = bars[(size_t) i];
            const float v = bars[(size_t) i];
            if (v < 0.03f)
                continue;
            const float x = cx + ((float) i - (float) (numBars - 1) * 0.5f) * spacing;
            const float h = juce::jmax (barW, v * maxH);
            auto bar = juce::Rectangle<float> (barW, h * 2.0f).withCentre ({ x, cy });
            g.setColour (Palette::glow.withAlpha (0.22f * v));
            g.fillRoundedRectangle (bar.expanded (barW * 1.4f, barW), barW * 1.5f);
            g.setColour (Palette::glowBright.withAlpha (0.55f + 0.45f * v));
            g.fillRoundedRectangle (bar, barW * 0.5f);
            g.setColour (juce::Colours::white.withAlpha (0.8f * v));
            g.fillRoundedRectangle (bar.reduced (barW * 0.25f, barW * 0.5f), barW * 0.3f);
        }
    }

    drawPill (g, 0, leftPill, "PITCH IN", inNote, "your voice", inNote != "--", phys);
    drawPill (g, 1, leftPill2, "KEY", keyText, keySub, false, phys);
    drawPill (g, 2, rightPill, "PITCH OUT", outNote, "new voice", outNote != "--", phys);
    drawPill (g, 3, rightPill2, "ENGINE", engineText, engineSub, engineLit, phys);
}

} // namespace voxa::ui
