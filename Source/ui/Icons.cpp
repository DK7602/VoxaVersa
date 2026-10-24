#include "Icons.h"

#include <map>

namespace voxa::ui::Icons
{

namespace
{
    juce::Path svg (const char* data)
    {
        return juce::Drawable::parseSVGPath (data);
    }

    juce::Path makeGear()
    {
        juce::Path p;
        const int teeth = 8;
        const float cx = 12.0f, cy = 12.0f, rOuter = 9.6f, rInner = 7.3f;
        const float toothHalf = juce::MathConstants<float>::pi / (float) teeth * 0.42f;
        const float rootHalf = juce::MathConstants<float>::pi / (float) teeth * 0.62f;

        for (int i = 0; i < teeth; ++i)
        {
            const float a = juce::MathConstants<float>::twoPi * (float) i / (float) teeth;
            auto pt = [&] (float r, float ang) { return juce::Point<float> (cx + r * std::sin (ang), cy - r * std::cos (ang)); };

            const auto p1 = pt (rInner, a - rootHalf);
            const auto p2 = pt (rOuter, a - toothHalf);
            const auto p3 = pt (rOuter, a + toothHalf);
            const auto p4 = pt (rInner, a + rootHalf);

            if (i == 0)
                p.startNewSubPath (p1);
            else
                p.lineTo (p1);

            p.lineTo (p2);
            p.lineTo (p3);
            p.lineTo (p4);
        }
        p.closeSubPath();
        p.addEllipse (cx - 3.2f, cy - 3.2f, 6.4f, 6.4f);
        return p;
    }

    juce::Path makeStar (float outer, float inner)
    {
        juce::Path p;
        p.addStar ({ 12.0f, 12.6f }, 5, inner, outer, 0.0f);
        return p;
    }

    Shape build (Icon icon)
    {
        Shape s;
        switch (icon)
        {
            case Icon::Gear:          s.stroke = makeGear(); break;
            case Icon::Mic:           s.stroke = svg ("M9 6.2a3 3 0 0 1 6 0v4.8a3 3 0 0 1 -6 0z M5.6 11a6.4 6.4 0 0 0 12.8 0 M12 17.4v3.6 M8.6 21h6.8"); break;
            case Icon::Document:      s.stroke = svg ("M6 3h8.2l3.8 3.8V21H6z M14 3v4h4 M9 11h6 M9 14.2h6 M9 17.4h4"); break;
            case Icon::Save:          s.stroke = svg ("M5 4h11.2L19 6.8V20H5z M8 4v4.6h7V4 M8 20v-6h8v6"); break;
            case Icon::Undo:          s.stroke = svg ("M9 5.5L4.8 9.7 9 13.9 M4.8 9.7H14.2A4.8 4.8 0 0 1 19 14.5V18"); break;
            case Icon::Redo:          s.stroke = svg ("M15 5.5L19.2 9.7 15 13.9 M19.2 9.7H9.8A4.8 4.8 0 0 0 5 14.5V18"); break;
            case Icon::Power:         s.stroke = svg ("M12 3v8.4 M7.2 6.4A7.4 7.4 0 1 0 16.8 6.4"); break;
            case Icon::Link:          s.stroke = svg ("M10.2 13.8a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0 -5.66 -5.66l-1.2 1.2 M13.8 10.2a4 4 0 0 0 -5.66 0l-3 3a4 4 0 0 0 5.66 5.66l1.2 -1.2"); break;
            case Icon::Star:          s.stroke = makeStar (9.4f, 4.2f); break;
            case Icon::StarFilled:    s.fill = makeStar (9.4f, 4.2f); break;
            case Icon::Search:        s.stroke.addEllipse (4.0f, 4.0f, 12.4f, 12.4f); s.stroke.startNewSubPath (14.6f, 14.6f); s.stroke.lineTo (20.0f, 20.0f); break;
            case Icon::List:          s.stroke = svg ("M9 6.5h11 M9 12h11 M9 17.5h11 M4.2 6.5h0.6 M4.2 12h0.6 M4.2 17.5h0.6"); break;
            case Icon::Grid:          s.stroke = svg ("M4 4h6.5v6.5H4z M13.5 4H20v6.5h-6.5z M4 13.5h6.5V20H4z M13.5 13.5H20V20h-6.5z"); break;
            case Icon::Play:          s.fill = svg ("M8 4.8v14.4a0.8 0.8 0 0 0 1.2 0.7l11.4-7.2a0.8 0.8 0 0 0 0 -1.4L9.2 4.1A0.8 0.8 0 0 0 8 4.8z"); break;
            case Icon::Stop:          s.fill.addRoundedRectangle (6.0f, 6.0f, 12.0f, 12.0f, 1.8f); break;
            case Icon::Pause:         s.fill.addRoundedRectangle (6.5f, 5.0f, 4.0f, 14.0f, 1.2f); s.fill.addRoundedRectangle (13.5f, 5.0f, 4.0f, 14.0f, 1.2f); break;
            case Icon::Record:        s.fill.addEllipse (5.5f, 5.5f, 13.0f, 13.0f); break;
            case Icon::Note:
                s.stroke = svg ("M9.2 17.6V6.2l10.4-2.2v11.4 M9.2 9.4l10.4-2.2");
                s.fill.addEllipse (3.6f, 15.1f, 5.8f, 4.8f);
                s.fill.addEllipse (14.0f, 13.0f, 5.8f, 4.8f);
                break;
            case Icon::ChevronLeft:   s.stroke = svg ("M15 4.5L7.5 12 15 19.5"); break;
            case Icon::ChevronRight:  s.stroke = svg ("M9 4.5L16.5 12 9 19.5"); break;
            case Icon::ChevronDown:   s.stroke = svg ("M5.5 9l6.5 6.5L18.5 9"); break;
            case Icon::ChevronUp:     s.stroke = svg ("M5.5 15l6.5-6.5L18.5 15"); break;
            case Icon::AutoLevel:     s.stroke = svg ("M1.5 12h3.6l1.6-3.8 2.6 9.6 3-14 2.8 11.6 1.8-3.4h5.6"); break;
            case Icon::PitchPreserve: s.stroke = svg ("M2.5 13.5c2.2 0 2.6-7.5 5.4-7.5s3.2 12 6.2 12 2.8-6 5.2-6c1 0 1.6 0.8 2.2 1.6"); break;
            case Icon::FormantShift:  s.stroke = svg ("M2.5 15.5c1.6 0 2.4-9.5 4.8-9.5 2.6 0 2.8 12.5 5.6 12.5 2.4 0 2.6-8 4.6-8 1.6 0 2.2 3 4 3"); break;
            case Icon::Expression:    s.stroke = svg ("M8.2 3.8c3.2 0 5.4 2.2 5.4 5.4 0 0.6 1.6 2.6 1.6 3.1 0 0.4-0.9 0.6-1.4 0.8v2.3c0 1.2-0.9 1.9-2 1.9H10v2.9 M8 8.6h0.8 M17.8 9.4a3.4 3.4 0 0 1 0 5.2 M20.2 7.4a6.4 6.4 0 0 1 0 9.2"); break;
            case Icon::Vibe:          s.stroke = svg ("M1.5 12h2 M5 9.5v5 M8 6v12 M11 3v18 M14 7v10 M17 9v6 M20 10.8v2.4 M21 12h1.5"); break;
            case Icon::Drag:          s.stroke = svg ("M12 3.5v11.5 M7.4 10.6L12 15.2l4.6-4.6 M4.5 15.5v4h15v-4"); break;
            case Icon::Folder:        s.stroke = svg ("M3 6.5V18.5h18V8.5H11.5L9.5 6.5z"); break;
            case Icon::Close:         s.stroke = svg ("M6.5 6.5l11 11 M17.5 6.5l-11 11"); break;
            case Icon::Sync:          s.stroke = svg ("M19.5 12a7.5 7.5 0 0 1 -13.3 4.8 M4.5 12a7.5 7.5 0 0 1 13.3 -4.8 M18.3 3.2v4.2h-4.2 M5.7 20.8v-4.2h4.2"); break;
            case Icon::Midi:
                s.stroke.addEllipse (3.0f, 3.0f, 18.0f, 18.0f);
                s.stroke.startNewSubPath (10.6f, 18.4f); s.stroke.lineTo (13.4f, 18.4f);
                for (auto [x, y] : { std::pair { 7.0f, 12.0f }, { 17.0f, 12.0f }, { 8.6f, 8.2f }, { 15.4f, 8.2f }, { 12.0f, 6.6f } })
                    s.fill.addEllipse (x - 1.1f, y - 1.1f, 2.2f, 2.2f);
                break;
            case Icon::Export:        s.stroke = svg ("M12 15V3.8 M7.6 8.2L12 3.8l4.4 4.4 M5 13v7h14v-7"); break;
            case Icon::Sparkle:       s.fill = svg ("M12 2.5l2 6.2 6.5 2.3-6.5 2.3L12 19.5l-2-6.2-6.5-2.3 6.5-2.3z M19 15.5l0.9 2.4 2.4 0.9-2.4 0.9L19 22l-0.9-2.3-2.4-0.9 2.4-0.9z"); break;
            case Icon::TuningFork:    s.stroke = svg ("M8 3v6a4 4 0 0 0 8 0V3 M12 13v8"); break;
            case Icon::Info:          s.stroke.addEllipse (3.0f, 3.0f, 18.0f, 18.0f); s.stroke.startNewSubPath (12.0f, 11.0f); s.stroke.lineTo (12.0f, 16.5f); s.fill.addEllipse (11.0f, 6.8f, 2.0f, 2.0f); break;
            case Icon::Trash:         s.stroke = svg ("M4.5 6.5h15 M9.5 6.5V4h5v2.5 M6.5 6.5l1 13.5h9l1-13.5 M10 10.5v6 M14 10.5v6"); break;
            case Icon::Plus:          s.stroke = svg ("M12 5v14 M5 12h14"); break;
            case Icon::Menu:          s.stroke = svg ("M4 7h16 M4 12h16 M4 17h16"); break;
            case Icon::Waveform:      s.stroke = svg ("M3 12h1.5 M6 9v6 M9 5v14 M12 8v8 M15 4v16 M18 9v6 M21 12h-1.5"); break;
        }
        return s;
    }
}

const Shape& get (Icon icon)
{
    static std::map<Icon, Shape> cache;
    auto it = cache.find (icon);
    if (it == cache.end())
        it = cache.emplace (icon, build (icon)).first;
    return it->second;
}

void draw (juce::Graphics& g, Icon icon, juce::Rectangle<float> area, juce::Colour colour, float strokeWidth, bool glow)
{
    const auto& shape = get (icon);
    const float size = juce::jmin (area.getWidth(), area.getHeight());
    const float scale = size / 24.0f;
    const auto t = juce::AffineTransform::scale (scale).translated (area.getCentreX() - 12.0f * scale,
                                                                     area.getCentreY() - 12.0f * scale);

    const juce::PathStrokeType stroke (strokeWidth * scale, juce::PathStrokeType::curved, juce::PathStrokeType::rounded);

    if (glow)
    {
        for (auto [w, a] : { std::pair { 3.4f, 0.10f }, { 2.2f, 0.22f } })
        {
            g.setColour (colour.withMultipliedAlpha (a));
            if (! shape.stroke.isEmpty())
                g.strokePath (shape.stroke, juce::PathStrokeType (strokeWidth * scale * w, juce::PathStrokeType::curved,
                                                                  juce::PathStrokeType::rounded), t);
            if (! shape.fill.isEmpty())
                g.strokePath (shape.fill, juce::PathStrokeType (strokeWidth * scale * (w - 1.0f)), t);
        }
    }

    g.setColour (colour);
    if (! shape.stroke.isEmpty())
        g.strokePath (shape.stroke, stroke, t);
    if (! shape.fill.isEmpty())
        g.fillPath (shape.fill, t);
}

} // namespace voxa::ui::Icons
