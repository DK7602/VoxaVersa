#include "MainView.h"
#include "VoxaAssets.h"

namespace voxa::ui
{

MainView::MainView (EditorContext& c)
    : ctx (c), header (c), input (c), output (c), centre (c), library (c), character (c), advanced (c), sing (c)
{
    setOpaque (true);
    setTitle ("VoxaVersa");
    setFocusContainerType (FocusContainerType::keyboardFocusContainer);

    orbImage = juce::ImageCache::getFromMemory (VoxaAssets::orb_png, VoxaAssets::orb_pngSize);
    wordmarkImage = juce::ImageCache::getFromMemory (VoxaAssets::wordmark_png, VoxaAssets::wordmark_pngSize);

    for (auto* comp : std::initializer_list<juce::Component*> { &centre, &header, &input, &output, &library, &character, &advanced, &sing })
        addAndMakeVisible (comp);
    addChildComponent (toast);

    library.onExpandedChanged = [this] (bool expanded)
    {
        library.setBounds (expanded ? libraryExpandedBounds : libraryBounds);
        if (expanded)
            library.toFront (false);
        toast.toFront (false);
    };

    addMouseListener (this, true);
    setSize (designWidth, designHeight);
}

MainView::~MainView()
{
    removeMouseListener (this);
}

std::vector<GlassPanel*> MainView::getGlassPanels()
{
    return { &input, &output, &library, &character, &advanced, &sing };
}

void MainView::resized()
{
    header.setBounds (0, 0, designWidth, 96);
    centre.setBounds (462, 84, 612, 500);
    input.setBounds (18, 112, 442, 450);
    output.setBounds (1076, 112, 442, 450);
    library.setBounds (library.isExpanded() ? libraryExpandedBounds : libraryBounds);
    character.setBounds (18, 797, 600, 208);
    advanced.setBounds (627, 797, 395, 208);
    sing.setBounds (1031, 797, 487, 208);
    toast.setAnchor ({ designWidth / 2, 100 });
}

void MainView::mouseDown (const juce::MouseEvent&)
{
    // Every gesture starts a new undo transaction (parameter changes are recorded by the APVTS).
    ctx.undo.beginNewTransaction();
}

void MainView::noteScaleChanged()
{
    lastScaleChange = juce::Time::getMillisecondCounter();
}

void MainView::tick (float dt)
{
    if (refreshPending && juce::Time::getMillisecondCounter() - lastScaleChange > 250)
    {
        refreshPending = false;
        repaint();
    }

    header.tick();

    if (! library.isExpanded())
    {
        input.tick (dt);
        output.tick (dt);
        centre.tick (dt);
    }

    library.tick (dt);
    character.tick();
    advanced.tick (dt);
    sing.tick (dt);
    toast.tick (dt);
}

//==============================================================================
void MainView::paint (juce::Graphics& g)
{
    const float scale = physicalScale (g);
    const bool scaleChanged = std::abs (scale - backgroundScale) > 0.001f;

    if (! background.isValid())
    {
        renderBackground (scale);
    }
    else if (scaleChanged)
    {
        if (juce::Time::getMillisecondCounter() - lastScaleChange > 250)
            renderBackground (scale);
        else
            refreshPending = true;
    }

    if (std::abs (scale - backgroundScale) > 0.001f)
        g.setImageResamplingQuality (juce::Graphics::lowResamplingQuality);

    drawCachedImage (g, background, {}, backgroundScale);
}

void MainView::renderBackground (float scale)
{
    const int w = juce::roundToInt ((float) designWidth * scale);
    const int h = juce::roundToInt ((float) designHeight * scale);
    juce::Image img (juce::Image::RGB, juce::jmax (1, w), juce::jmax (1, h), false);

    {
        juce::Graphics g (img);
        g.addTransform (juce::AffineTransform::scale (scale));
        paintBackdrop (g);
    }

    // softened silk, then crisp highlights on top
    blurImage (img, juce::jmax (1, juce::roundToInt (5.0f * scale)), 2);

    {
        juce::Graphics g (img);
        g.addTransform (juce::AffineTransform::scale (scale));

        juce::Random rng (77);
        for (int i = 0; i < 16; ++i)
        {
            const float rx = 300.0f + rng.nextFloat() * 520.0f;
            const float ry = rx * (0.42f + rng.nextFloat() * 0.3f);
            const float start = rng.nextFloat() * juce::MathConstants<float>::twoPi;
            juce::Path p;
            p.addCentredArc (768.0f + (rng.nextFloat() - 0.5f) * 200.0f, 330.0f + (rng.nextFloat() - 0.5f) * 160.0f,
                             rx, ry, (rng.nextFloat() - 0.5f) * 0.9f, start, start + 0.8f + rng.nextFloat() * 1.6f, true);
            g.setColour (juce::Colours::white.withAlpha (0.25f + rng.nextFloat() * 0.4f));
            g.strokePath (p, juce::PathStrokeType (0.8f + rng.nextFloat() * 1.6f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        }

        // sparkles
        for (auto [x, y, s] : { std::tuple { 1000.0f, 152.0f, 1.0f }, { 538.0f, 126.0f, 0.7f }, { 1010.0f, 470.0f, 0.55f },
                                { 250.0f, 86.0f, 0.5f }, { 1290.0f, 96.0f, 0.45f } })
        {
            g.setGradientFill (juce::ColourGradient (juce::Colours::white.withAlpha (0.9f), x, y, juce::Colours::white.withAlpha (0.0f), x + 22.0f * s, y, true));
            g.fillEllipse (x - 22.0f * s, y - 22.0f * s, 44.0f * s, 44.0f * s);
            juce::ColourGradient hx (juce::Colours::white.withAlpha (0.0f), x - 46.0f * s, y, juce::Colours::white.withAlpha (0.0f), x + 46.0f * s, y, false);
            hx.addColour (0.5, juce::Colours::white);
            g.setGradientFill (hx);
            g.fillRect (juce::Rectangle<float> (x - 46.0f * s, y - 0.8f, 92.0f * s, 1.6f));
            juce::ColourGradient vy (juce::Colours::white.withAlpha (0.0f), x, y - 30.0f * s, juce::Colours::white.withAlpha (0.0f), x, y + 30.0f * s, false);
            vy.addColour (0.5, juce::Colours::white);
            g.setGradientFill (vy);
            g.fillRect (juce::Rectangle<float> (x - 0.8f, y - 30.0f * s, 1.6f, 60.0f * s));
        }
    }

    // frosted backdrop for the glass cards
    const int bw = juce::jmax (8, w / 6), bh = juce::jmax (8, h / 6);
    auto blurred = img.rescaled (bw, bh, juce::Graphics::mediumResamplingQuality);
    blurImage (blurred, 3, 3);
    const float blurredScale = (float) bw / (float) designWidth;

    {
        juce::Graphics g (img);
        g.addTransform (juce::AffineTransform::scale (scale));
        g.setImageResamplingQuality (juce::Graphics::highResamplingQuality);

        // orb + wordmarks
        const auto centreOrigin = centre.getPosition().toFloat();
        if (orbImage.isValid())
        {
            const auto orbArea = centre.getOrbArea() + centreOrigin;
            // pre-shrink in two steps for quality
            const int target = juce::roundToInt (orbArea.getWidth() * scale);
            auto scaled = orbImage.getWidth() > target * 2 ? orbImage.rescaled (target * 2, target * 2, juce::Graphics::highResamplingQuality) : orbImage;
            g.drawImage (scaled, orbArea, juce::RectanglePlacement::centred);
        }
        if (wordmarkImage.isValid())
        {
            g.drawImage (wordmarkImage, centre.getWordmarkArea() + centreOrigin, juce::RectanglePlacement::centred);
            g.drawImage (wordmarkImage, header.getWordmarkArea() + header.getPosition().toFloat(), juce::RectanglePlacement::centred);
        }

        for (auto* p : getGlassPanels())
        {
            const auto b = (p == &library ? libraryBounds : p->getBounds()).toFloat();
            GlassPanel::paintGlassCard (g, b, &blurred, {}, blurredScale);
            g.saveState();
            g.setOrigin (b.getTopLeft().toInt());
            p->paintStatic (g, { 0.0f, 0.0f, b.getWidth(), b.getHeight() });
            g.restoreState();
        }
    }

    background = img;
    backgroundScale = scale;
}

void MainView::paintBackdrop (juce::Graphics& g)
{
    const auto bounds = juce::Rectangle<float> (0.0f, 0.0f, (float) designWidth, (float) designHeight);
    const juce::Point<float> orbCentre (768.0f, 304.0f);

    // luminous base
    {
        juce::ColourGradient base (juce::Colour (0xfff6f9ff), orbCentre.x, orbCentre.y, juce::Colour (0xff8297c8), orbCentre.x + 1050.0f, orbCentre.y, true);
        base.addColour (0.22, juce::Colour (0xffe2eafa));
        base.addColour (0.45, juce::Colour (0xffbccbea));
        base.addColour (0.75, juce::Colour (0xff9cb0da));
        g.setGradientFill (base);
        g.fillRect (bounds);
    }

    // silk folds: dark shadow band + bright band, softened by the blur that follows
    juce::Random rng (2024);
    for (int i = 0; i < 34; ++i)
    {
        const float rx = 260.0f + rng.nextFloat() * 560.0f;
        const float ry = rx * (0.38f + rng.nextFloat() * 0.34f);
        const float start = rng.nextFloat() * juce::MathConstants<float>::twoPi;
        const float len = 0.9f + rng.nextFloat() * 1.9f;
        juce::Path p;
        p.addCentredArc (orbCentre.x + (rng.nextFloat() - 0.5f) * 260.0f, orbCentre.y + 30.0f + (rng.nextFloat() - 0.5f) * 220.0f,
                         rx, ry, (rng.nextFloat() - 0.5f) * 1.0f, start, start + len, true);

        const float width = 10.0f + rng.nextFloat() * 52.0f;
        const float offset = width * 0.45f;
        g.setColour (juce::Colour (0xff2552c0).withAlpha (0.12f + rng.nextFloat() * 0.26f));
        g.strokePath (p, juce::PathStrokeType (width, juce::PathStrokeType::curved, juce::PathStrokeType::rounded),
                      juce::AffineTransform::translation (offset, offset * 0.6f));
        g.setColour (juce::Colours::white.withAlpha (0.22f + rng.nextFloat() * 0.40f));
        g.strokePath (p, juce::PathStrokeType (width * 0.55f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    // deep blue swirls towards the outer edges
    for (int i = 0; i < 10; ++i)
    {
        const bool left = (i % 2) == 0;
        juce::Path p;
        const float x0 = left ? -60.0f : 1596.0f;
        const float y0 = 180.0f + rng.nextFloat() * 840.0f;
        p.startNewSubPath (x0, y0);
        p.cubicTo (left ? 160.0f + rng.nextFloat() * 200.0f : 1376.0f - rng.nextFloat() * 200.0f, y0 - 200.0f - rng.nextFloat() * 300.0f,
                   left ? 380.0f + rng.nextFloat() * 160.0f : 1156.0f - rng.nextFloat() * 160.0f, y0 + rng.nextFloat() * 200.0f,
                   left ? 520.0f + rng.nextFloat() * 120.0f : 1016.0f - rng.nextFloat() * 120.0f, y0 - 420.0f - rng.nextFloat() * 260.0f);
        g.setColour (juce::Colour (0xff1a45b0).withAlpha (0.12f + rng.nextFloat() * 0.16f));
        g.strokePath (p, juce::PathStrokeType (20.0f + rng.nextFloat() * 60.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        g.setColour (juce::Colours::white.withAlpha (0.25f + rng.nextFloat() * 0.3f));
        g.strokePath (p, juce::PathStrokeType (6.0f + rng.nextFloat() * 14.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded),
                      juce::AffineTransform::translation (-12.0f, -8.0f));
    }

    // radiant glow behind the orb
    {
        juce::ColourGradient glow (juce::Colours::white, orbCentre.x, orbCentre.y, juce::Colours::white.withAlpha (0.0f), orbCentre.x + 420.0f, orbCentre.y, true);
        glow.addColour (0.45, juce::Colours::white.withAlpha (0.85f));
        glow.addColour (0.62, juce::Colour (0xffd8e8ff).withAlpha (0.5f));
        g.setGradientFill (glow);
        g.fillEllipse (orbCentre.x - 420.0f, orbCentre.y - 420.0f, 840.0f, 840.0f);

        juce::ColourGradient blueRim (juce::Colour (0xff6aa8ff).withAlpha (0.0f), orbCentre.x, orbCentre.y, juce::Colour (0xff6aa8ff).withAlpha (0.0f), orbCentre.x + 300.0f, orbCentre.y, true);
        blueRim.addColour (0.68, juce::Colour (0xff6aa8ff).withAlpha (0.0f));
        blueRim.addColour (0.76, juce::Colour (0xff4f8fff).withAlpha (0.45f));
        blueRim.addColour (0.88, juce::Colour (0xff9cc8ff).withAlpha (0.2f));
        g.setGradientFill (blueRim);
        g.fillEllipse (orbCentre.x - 300.0f, orbCentre.y - 300.0f, 600.0f, 600.0f);
    }

    // gentle edge vignette
    {
        juce::ColourGradient vig (juce::Colour (0xff10224f).withAlpha (0.0f), orbCentre.x, 520.0f, juce::Colour (0xff10224f).withAlpha (0.28f), orbCentre.x + 980.0f, 520.0f, true);
        vig.addColour (0.6, juce::Colour (0xff10224f).withAlpha (0.0f));
        g.setGradientFill (vig);
        g.fillRect (bounds);
    }
}

} // namespace voxa::ui
