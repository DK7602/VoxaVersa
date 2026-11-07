#include "ManualView.h"
#include "Theme.h"
#include "UserManual.h"

namespace voxa::ui
{

namespace
{
    // sizes in design pixels (the editor scales the whole canvas)
    // (at UI size 100 % the canvas is drawn at 78 %: 21 here is ~16 px on screen)
    constexpr float bodySize = 21.0f, smallSize = 19.0f, titleSize = 38.0f, headingSize = 31.0f, subheadingSize = 24.0f;
    constexpr float headerHeight = 70.0f, navWidth = 300.0f, contentPad = 34.0f;

    const juce::String& bullet()
    {
        static const juce::String b = juce::String::charToString ((juce::juce_wchar) 0x2022);
        return b;
    }

    struct Style
    {
        float size = bodySize;
        Weight weight = Weight::Regular;
        juce::Colour colour = Palette::textSoft;
        bool mono = false;
    };

    juce::Font makeFont (const Style& s)
    {
        if (s.mono)
            return juce::Font (juce::FontOptions (juce::Font::getDefaultMonospacedFontName(), s.size * 0.88f, juce::Font::plain));
        return font (s.size, s.weight);
    }

    struct Run
    {
        juce::String text;
        Style style;
    };

    bool isList (const juce::XmlElement& e) { return e.hasTagName ("ul") || e.hasTagName ("ol"); }

    /** The inline text of an element (bold, italic, code, links), lists inside it left out. */
    void collectRuns (const juce::XmlElement& e, Style style, std::vector<Run>& runs)
    {
        for (auto* child : e.getChildIterator())
        {
            if (child->isTextElement())
            {
                runs.push_back ({ child->getText(), style });
                continue;
            }
            if (isList (*child))
                continue;
            auto s = style;
            if (child->hasTagName ("b") || child->hasTagName ("strong") || child->hasTagName ("dt"))
            {
                s.weight = Weight::SemiBold;
                s.colour = child->getStringAttribute ("class") == "ui" ? Palette::glowBright : Palette::text;
            }
            else if (child->hasTagName ("i") || child->hasTagName ("em"))
                s.colour = Palette::text;
            else if (child->hasTagName ("code") || child->hasTagName ("kbd"))
            {
                s.mono = true;
                s.colour = Palette::cyan;
            }
            else if (child->hasTagName ("a"))
                s.colour = Palette::glowBright;
            collectRuns (*child, s, runs);
        }
    }

    /** HTML whitespace: every run of spaces and line breaks is one space; none at the start or the end. */
    juce::AttributedString toAttributed (std::vector<Run> runs, float lineSpacing)
    {
        bool lastWasSpace = true;
        for (auto& r : runs)
        {
            juce::String out;
            for (auto c : r.text)
            {
                if (juce::CharacterFunctions::isWhitespace (c))
                {
                    if (! lastWasSpace)
                        out << ' ';
                    lastWasSpace = true;
                }
                else
                {
                    out << juce::String::charToString (c);
                    lastWasSpace = false;
                }
            }
            r.text = out;
        }
        while (! runs.empty() && runs.back().text.isEmpty())
            runs.pop_back();
        if (! runs.empty())
            runs.back().text = runs.back().text.trimEnd();

        juce::AttributedString a;
        a.setWordWrap (juce::AttributedString::byWord);
        a.setJustification (juce::Justification::topLeft);
        a.setLineSpacing (lineSpacing);
        for (const auto& r : runs)
            if (r.text.isNotEmpty())
                a.append (r.text, makeFont (r.style), r.style.colour);
        return a;
    }

    juce::AttributedString inlineText (const juce::XmlElement& e, Style style, float lineSpacing = 5.0f)
    {
        std::vector<Run> runs;
        collectRuns (e, style, runs);
        return toAttributed (std::move (runs), lineSpacing);
    }
}

//==============================================================================
struct ManualView::Block
{
    enum class Kind { Title, Lead, Heading, Subheading, Paragraph, Item, Table, Note, Tip, Question, Answer, Footer };
    Kind kind = Kind::Paragraph;
    juce::AttributedString text;
    int level = 0;
    juce::String marker;
    juce::String anchor;
    std::vector<std::vector<juce::AttributedString>> rows;
    bool headerRow = false;

    // layout
    float y = 0.0f, height = 0.0f, x = 0.0f, width = 0.0f;
    juce::TextLayout layout;
    std::vector<std::vector<juce::TextLayout>> cells;
    std::vector<float> rowHeights, columnWidths;
};

//==============================================================================
class ManualView::Content : public juce::Component
{
public:
    std::vector<Block> blocks;

    void parse (const juce::String& html, std::vector<std::pair<juce::String, juce::String>>& sections)
    {
        blocks.clear();
        sections.clear();
        const int a = html.indexOf ("<main>"), b = html.indexOf ("</main>");
        if (a < 0 || b < a)
            return;
        const auto main = juce::XmlDocument::parse (html.substring (a, b + 7));
        if (main == nullptr)
            return;

        for (auto* e : main->getChildIterator())
        {
            if (e->hasTagName ("h1"))
                add (Block::Kind::Title, inlineText (*e, { titleSize, Weight::Title, Palette::text }));
            else if (e->hasTagName ("p"))
                add (Block::Kind::Lead, inlineText (*e, { 23.0f, Weight::Regular, Palette::textSoft }));
            else if (e->hasTagName ("section"))
                parseSection (*e, sections);
            else if (e->hasTagName ("footer"))
                add (Block::Kind::Footer, inlineText (*e, { 17.0f, Weight::Regular, Palette::textDim }));
        }
    }

    void layoutFor (float width)
    {
        float y = 8.0f;
        const float textWidth = width - 2.0f * contentPad;
        for (auto& b : blocks)
        {
            b.x = contentPad;
            b.width = textWidth;
            float before = 0.0f, after = 12.0f;
            switch (b.kind)
            {
                case Block::Kind::Title:      after = 4.0f; break;
                case Block::Kind::Lead:       after = 10.0f; break;
                case Block::Kind::Heading:    before = 30.0f; after = 12.0f; break;
                case Block::Kind::Subheading: before = 14.0f; after = 6.0f; break;
                case Block::Kind::Item:       after = 7.0f; b.x += 32.0f + 28.0f * (float) b.level; b.width -= 32.0f + 28.0f * (float) b.level; break;
                case Block::Kind::Note:
                case Block::Kind::Tip:        before = 4.0f; after = 16.0f; break;
                case Block::Kind::Question:   before = 10.0f; after = 3.0f; break;
                case Block::Kind::Answer:     after = 8.0f; b.x += 16.0f; b.width -= 16.0f; break;
                case Block::Kind::Table:      after = 16.0f; break;
                case Block::Kind::Footer:     before = 24.0f; break;
                case Block::Kind::Paragraph:  break;
            }
            y += before;
            b.y = y;

            if (b.kind == Block::Kind::Table)
                layoutTable (b);
            else if (b.kind == Block::Kind::Note || b.kind == Block::Kind::Tip)
            {
                b.layout.createLayout (b.text, b.width - 36.0f);
                b.height = b.layout.getHeight() + 28.0f;
            }
            else
            {
                b.layout.createLayout (b.text, b.width);
                b.height = b.layout.getHeight();
            }
            y += b.height + after;
        }
        setSize ((int) width, (int) std::ceil (y + 40.0f));
    }

    float yOfAnchor (const juce::String& id) const
    {
        for (const auto& b : blocks)
            if (b.anchor == id)
                return b.y - 22.0f;
        return -1.0f;
    }

    juce::String anchorAt (float y) const
    {
        juce::String current;
        for (const auto& b : blocks)
        {
            if (b.anchor.isNotEmpty() && b.y - 40.0f <= y)
                current = b.anchor;
            if (b.y > y + 40.0f)
                break;
        }
        return current;
    }

    void paint (juce::Graphics& g) override
    {
        const auto clip = g.getClipBounds().toFloat();
        for (const auto& b : blocks)
        {
            if (b.y > clip.getBottom() || b.y + b.height + 30.0f < clip.getY())
                continue;
            const juce::Rectangle<float> area (b.x, b.y, b.width, b.height);
            switch (b.kind)
            {
                case Block::Kind::Heading:
                    if (&b != firstHeading())
                    {
                        g.setColour (Palette::insetBorder.withAlpha (0.8f));
                        g.fillRect (area.getX(), area.getY() - 16.0f, area.getWidth(), 1.0f);
                    }
                    b.layout.draw (g, area);
                    break;
                case Block::Kind::Item:
                {
                    g.setColour (Palette::glowBright);
                    g.setFont (font (bodySize, b.marker == bullet() ? Weight::Regular : Weight::SemiBold));
                    g.drawText (b.marker, juce::Rectangle<float> (area.getX() - 34.0f, area.getY(), 28.0f, bodySize * 1.3f),
                                juce::Justification::topRight);
                    b.layout.draw (g, area);
                    break;
                }
                case Block::Kind::Note:
                case Block::Kind::Tip:
                {
                    const auto tint = b.kind == Block::Kind::Note ? Palette::warning : Palette::glow;
                    g.setColour (tint.withAlpha (0.12f));
                    g.fillRoundedRectangle (area, 8.0f);
                    g.setColour (tint);
                    g.fillRoundedRectangle (area.withWidth (4.0f), 2.0f);
                    b.layout.draw (g, area.reduced (18.0f, 14.0f).withTrimmedLeft (4.0f));
                    break;
                }
                case Block::Kind::Table:
                    paintTable (g, b);
                    break;
                default:
                    b.layout.draw (g, area);
                    break;
            }
        }
    }

private:
    const Block* firstHeading() const
    {
        for (const auto& b : blocks)
            if (b.kind == Block::Kind::Heading)
                return &b;
        return nullptr;
    }

    void add (Block::Kind kind, juce::AttributedString text, int level = 0, const juce::String& marker = {}, const juce::String& anchor = {})
    {
        Block b;
        b.kind = kind;
        b.text = std::move (text);
        b.level = level;
        b.marker = marker;
        b.anchor = anchor;
        blocks.push_back (std::move (b));
    }

    void parseSection (const juce::XmlElement& section, std::vector<std::pair<juce::String, juce::String>>& sections)
    {
        const auto id = section.getStringAttribute ("id");
        for (auto* e : section.getChildIterator())
        {
            if (e->hasTagName ("h2"))
            {
                sections.push_back ({ id, e->getAllSubText().trim() });
                add (Block::Kind::Heading, inlineText (*e, { headingSize, Weight::Title, Palette::text }), 0, {}, id);
            }
            else if (e->hasTagName ("h3"))
                add (Block::Kind::Subheading, inlineText (*e, { subheadingSize, Weight::SemiBold, Palette::glowBright }));
            else if (e->hasTagName ("p"))
                add (Block::Kind::Paragraph, inlineText (*e, {}));
            else if (isList (*e))
                parseList (*e, 0);
            else if (e->hasTagName ("div") && e->getStringAttribute ("class") == "tablewrap")
            {
                if (auto* t = e->getChildByName ("table"))
                    parseTable (*t);
            }
            else if (e->hasTagName ("table"))
                parseTable (*e);
            else if (e->hasTagName ("div"))
                add (e->getStringAttribute ("class") == "note" ? Block::Kind::Note : Block::Kind::Tip, inlineText (*e, {}));
            else if (e->hasTagName ("dl"))
                for (auto* d : e->getChildIterator())
                {
                    if (d->hasTagName ("dt"))
                        add (Block::Kind::Question, inlineText (*d, { bodySize, Weight::SemiBold, Palette::text }));
                    else if (d->hasTagName ("dd"))
                        add (Block::Kind::Answer, inlineText (*d, {}));
                }
        }
    }

    void parseList (const juce::XmlElement& list, int level)
    {
        const bool numbered = list.hasTagName ("ol");
        int n = 1;
        for (auto* li : list.getChildIterator())
        {
            if (! li->hasTagName ("li"))
                continue;
            add (Block::Kind::Item, inlineText (*li, {}), level, numbered ? juce::String (n++) + "." : bullet());
            for (auto* inner : li->getChildIterator())
                if (isList (*inner))
                    parseList (*inner, level + 1);
        }
    }

    void parseTable (const juce::XmlElement& table)
    {
        Block b;
        b.kind = Block::Kind::Table;
        for (auto* tr : table.getChildIterator())
        {
            if (! tr->hasTagName ("tr"))
                continue;
            std::vector<juce::AttributedString> row;
            bool header = false;
            for (auto* cell : tr->getChildIterator())
            {
                if (cell->hasTagName ("th"))
                {
                    header = true;
                    row.push_back (inlineText (*cell, { 17.0f, Weight::SemiBold, Palette::textDim }, 3.0f));
                }
                else if (cell->hasTagName ("td"))
                    row.push_back (inlineText (*cell, { smallSize }, 3.0f));
            }
            if (b.rows.empty())
                b.headerRow = header;
            b.rows.push_back (std::move (row));
        }
        blocks.push_back (std::move (b));
    }

    void layoutTable (Block& b)
    {
        size_t columns = 0;
        for (const auto& r : b.rows)
            columns = std::max (columns, r.size());
        b.columnWidths.clear();
        if (columns == 2)
            b.columnWidths = { 0.3f, 0.7f };
        else if (columns == 3)
            b.columnWidths = { 0.22f, 0.39f, 0.39f };
        else
            b.columnWidths.assign (columns, columns > 0 ? 1.0f / (float) columns : 1.0f);
        for (auto& w : b.columnWidths)
            w *= b.width;

        b.cells.clear();
        b.rowHeights.clear();
        b.height = 0.0f;
        for (const auto& r : b.rows)
        {
            std::vector<juce::TextLayout> cells;
            float h = 0.0f;
            for (size_t c = 0; c < r.size(); ++c)
            {
                juce::TextLayout l;
                l.createLayout (r[c], b.columnWidths[c] - 18.0f);
                h = std::max (h, l.getHeight());
                cells.push_back (std::move (l));
            }
            b.cells.push_back (std::move (cells));
            b.rowHeights.push_back (h + 16.0f);
            b.height += h + 16.0f;
        }
    }

    void paintTable (juce::Graphics& g, const Block& b) const
    {
        float y = b.y;
        for (size_t r = 0; r < b.cells.size(); ++r)
        {
            if (r == 0 && b.headerRow)
            {
                g.setColour (Palette::inset.withAlpha (0.6f));
                g.fillRect (b.x, y, b.width, b.rowHeights[r]);
            }
            float x = b.x;
            for (size_t c = 0; c < b.cells[r].size(); ++c)
            {
                b.cells[r][c].draw (g, { x + 9.0f, y + 8.0f, b.columnWidths[c] - 18.0f, b.rowHeights[r] - 16.0f });
                x += b.columnWidths[c];
            }
            y += b.rowHeights[r];
            g.setColour (Palette::insetBorder.withAlpha (0.6f));
            g.fillRect (b.x, y - 1.0f, b.width, 1.0f);
        }
    }
};

//==============================================================================
class ManualView::SectionList : public juce::Component
{
public:
    std::vector<std::pair<juce::String, juce::String>> items;
    juce::String current;
    std::function<void (const juce::String&)> onPick;

    static constexpr float rowHeight = 42.0f;

    void paint (juce::Graphics& g) override
    {
        for (size_t i = 0; i < items.size(); ++i)
        {
            const juce::Rectangle<float> r (0.0f, (float) i * rowHeight, (float) getWidth(), rowHeight - 4.0f);
            const bool on = items[i].first == current, over = (int) i == hover;
            if (on || over)
            {
                g.setColour ((on ? Palette::accent : Palette::control).withAlpha (on ? 0.35f : 0.8f));
                g.fillRoundedRectangle (r, 7.0f);
            }
            drawText (g, items[i].second, r.withTrimmedLeft (14.0f), font (19.5f, on ? Weight::SemiBold : Weight::Regular),
                      on ? Palette::text : Palette::textSoft, juce::Justification::centredLeft);
        }
    }

    void mouseMove (const juce::MouseEvent& e) override { setHover (rowAt (e.position.y)); }
    void mouseExit (const juce::MouseEvent&) override { setHover (-1); }
    void mouseUp (const juce::MouseEvent& e) override
    {
        const int i = rowAt (e.position.y);
        if (i >= 0 && onPick)
            onPick (items[(size_t) i].first);
    }

private:
    int hover = -1;
    int rowAt (float y) const { const int i = (int) (y / rowHeight); return juce::isPositiveAndBelow (i, (int) items.size()) ? i : -1; }
    void setHover (int i) { if (i != hover) { hover = i; repaint(); } }
};

//==============================================================================
ManualView::ManualView()
    : content (std::make_unique<Content>()), sectionList (std::make_unique<SectionList>())
{
    setTitle ("User Manual");
    setWantsKeyboardFocus (true);

    content->parse (UserManual::getHtml(), sections);
    version = juce::String (VOXAVERSA_VERSION_STRING) + " (build " + VOXAVERSA_BUILD_STRING + ")";

    viewport.setViewedComponent (content.get(), false);
    viewport.setScrollBarsShown (true, false);
    viewport.setScrollBarThickness (10);
    viewport.setSingleStepSizes (40, 40);
    addAndMakeVisible (viewport);

    sectionList->items = sections;
    if (! sections.empty())
        sectionList->current = sections.front().first;
    sectionList->onPick = [this] (const juce::String& id) { showSection (id); };
    addAndMakeVisible (*sectionList);

    viewport.getVerticalScrollBar().addListener (this);

    closeButton.onClick = [this] { if (onClose) onClose(); };
    browserButton.onClick = [] { UserManual::open(); };
    closeButton.setTooltip ("Close the manual (Esc)");
    browserButton.setTooltip ("Open this manual in your web browser");
    addAndMakeVisible (closeButton);
    addAndMakeVisible (browserButton);
}

ManualView::~ManualView()
{
    viewport.getVerticalScrollBar().removeListener (this);
}

int ManualView::getNumBlocks() const noexcept
{
    return (int) content->blocks.size();
}

void ManualView::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    g.setColour (Palette::panelFill);
    g.fillRoundedRectangle (r, 14.0f);
    g.setColour (Palette::insetBorder);
    g.drawRoundedRectangle (r.reduced (0.5f), 14.0f, 1.0f);

    drawText (g, "User Manual", { 30.0f, 0.0f, 300.0f, headerHeight }, font (30.0f, Weight::Title), Palette::text,
              juce::Justification::centredLeft);
    drawText (g, "VoxaVersa " + version, { 250.0f, 0.0f, 460.0f, headerHeight }, font (18.0f), Palette::textDim,
              juce::Justification::centredLeft);
    g.setColour (Palette::insetBorder);
    g.fillRect (0.0f, headerHeight, r.getWidth(), 1.0f);
    g.fillRect (navWidth, headerHeight, 1.0f, r.getHeight() - headerHeight);

    if (content->blocks.empty())
        drawText (g, "The manual couldn't be loaded.", r.withTrimmedTop (headerHeight), font (18.0f), Palette::textSoft);
}

void ManualView::resized()
{
    auto r = getLocalBounds();
    auto top = r.removeFromTop ((int) headerHeight).reduced (18, 13);
    closeButton.setBounds (top.removeFromRight (120));
    top.removeFromRight (12);
    browserButton.setBounds (top.removeFromRight (200));

    auto nav = r.removeFromLeft ((int) navWidth).reduced (14, 16);
    sectionList->setBounds (nav.withHeight ((int) (SectionList::rowHeight * (float) sections.size())));

    viewport.setBounds (r.reduced (2, 2));
    content->layoutFor ((float) viewport.getWidth() - (float) viewport.getScrollBarThickness());
}

bool ManualView::keyPressed (const juce::KeyPress& key)
{
    if (key == juce::KeyPress::escapeKey)
    {
        if (onClose)
            onClose();
        return true;
    }
    return false;
}

bool ManualView::showSection (const juce::String& id)
{
    const float y = content->yOfAnchor (id);
    if (y < 0.0f)
        return false;
    viewport.setViewPosition (0, juce::jmax (0, (int) y));
    sectionList->current = id;
    sectionList->repaint();
    return true;
}

void ManualView::scrollBarMoved (juce::ScrollBar*, double)
{
    const auto id = content->anchorAt ((float) viewport.getViewPositionY());
    if (id.isNotEmpty() && id != sectionList->current)
    {
        sectionList->current = id;
        sectionList->repaint();
    }
}

} // namespace voxa::ui
