#include "VoiceLibraryPanel.h"
#include "core/MusicTheory.h"

namespace voxa::ui
{

namespace
{
    juce::String tooltipFor (const VoiceProfile& v, bool favourite)
    {
        juce::String t;
        t << v.name << (v.hasNeuralModel() ? "  [AI]" : "") << "\n"
          << v.tagline << "\n"
          << "Genre: " << v.genre << "   Range: " << music::noteName (v.traits.rangeLowMidi)
          << " - " << music::noteName (v.traits.rangeHighMidi) << "\n";
        if (v.styles.size() > 0)
            t << "Style: " << v.styles.joinIntoString (", ") << "\n";
        if (v.hasNeuralModel())
            t << "AI neural voice model included\n";
        if (! v.isFactory && v.author.isNotEmpty())
            t << "By " << v.author << "\n";
        t << (favourite ? "Favourite - " : "") << "Click to select, right-click for options";
        return t;
    }
}

//==============================================================================
VoiceCard::VoiceCard (EditorContext& c, AvatarCache& a) : ctx (c), avatars (a)
{
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
}

const VoiceProfile* VoiceCard::getVoice() const
{
    const auto& voices = ctx.processor.getVoiceLibrary().getVoices();
    return juce::isPositiveAndBelow (voiceIndex, (int) voices.size()) ? &voices[(size_t) voiceIndex] : nullptr;
}

void VoiceCard::setVoiceIndex (int index)
{
    voiceIndex = index;
    if (auto* v = getVoice())
    {
        setTitle (v->name);
        setDescription (v->tagline);
        setTooltip (tooltipFor (*v, favourite));
    }
    repaint();
}

void VoiceCard::setState (bool sel, bool bl, bool fav)
{
    if (sel != selected || bl != blend || fav != favourite)
    {
        selected = sel;
        blend = bl;
        if (fav != favourite)
        {
            favourite = fav;
            if (auto* v = getVoice())
                setTooltip (tooltipFor (*v, favourite));
        }
        repaint();
    }
}

juce::Rectangle<float> VoiceCard::getStarArea() const
{
    auto r = getLocalBounds().toFloat().reduced (4.0f);
    return { r.getRight() - 28.0f, r.getY() + 3.0f, 25.0f, 25.0f };
}

void VoiceCard::paint (juce::Graphics& g)
{
    auto* v = getVoice();
    if (v == nullptr)
        return;

    auto r = getLocalBounds().toFloat().reduced (4.0f);
    const float corner = 9.0f;

    if (selected)
    {
        juce::Path outline;
        outline.addRoundedRectangle (r, corner);
        for (auto [w, a] : { std::pair { 8.0f, 0.10f }, { 5.0f, 0.18f }, { 2.6f, 0.35f } })
        {
            g.setColour (Palette::glow.withAlpha (a));
            g.strokePath (outline, juce::PathStrokeType (w));
        }
    }

    juce::Path clip;
    clip.addRoundedRectangle (r, corner);

    g.saveState();
    g.reduceClipRegion (clip);
    const float phys = physicalScale (g);
    const auto& img = avatars.get (*v, juce::roundToInt (r.getWidth() * phys), juce::roundToInt (r.getHeight() * phys));
    g.setOpacity (1.0f);
    drawCachedImage (g, img, r.getTopLeft(), phys);

    if (hover)
    {
        g.setColour (juce::Colours::white.withAlpha (0.06f));
        g.fillRect (r);
    }
    if (selected)
    {
        g.setGradientFill (juce::ColourGradient (Palette::glow.withAlpha (0.16f), r.getX(), r.getBottom(),
                                                 Palette::glow.withAlpha (0.0f), r.getX(), r.getCentreY(), false));
        g.fillRect (r);
    }
    g.restoreState();

    // name
    auto nameArea = r.withTop (r.getBottom() - 30.0f);
    const auto f = font (r.getWidth() < 110.0f ? 15.0f : 17.0f, Weight::SemiBold);
    drawText (g, v->name, nameArea.translated (0.0f, 1.0f), f, juce::Colours::black.withAlpha (0.6f));
    drawText (g, v->name, nameArea, f, juce::Colours::white);

    // border
    if (selected)
    {
        g.setColour (Palette::glowBright);
        g.drawRoundedRectangle (r, corner, 2.2f);
        g.setColour (juce::Colours::white.withAlpha (0.6f));
        g.drawRoundedRectangle (r.reduced (1.5f), corner - 1.0f, 0.8f);
    }
    else
    {
        g.setColour (Palette::panelBorder.withAlpha (hover ? 0.75f : 0.35f));
        g.drawRoundedRectangle (r.reduced (0.5f), corner, 1.0f);
    }

    if (hasKeyboardFocus (false))
    {
        g.setColour (juce::Colours::white.withAlpha (0.85f));
        g.drawRoundedRectangle (r.reduced (2.5f), corner - 2.0f, 1.2f);
    }

    // badges
    float badgeY = r.getY() + 5.0f;
    auto badge = [&] (const juce::String& text, juce::Colour colour, std::optional<Icon> icon)
    {
        const auto bf = font (12.0f, Weight::SemiBold);
        const float w = textWidth (bf, text) + (icon ? 22.0f : 12.0f);
        auto b = juce::Rectangle<float> (r.getX() + 5.0f, badgeY, w, 17.0f);
        g.setColour (juce::Colour (0xd0050b1e));
        g.fillRoundedRectangle (b, 8.5f);
        g.setColour (colour);
        g.drawRoundedRectangle (b.reduced (0.5f), 8.5f, 1.0f);
        auto content = b.reduced (5.0f, 0.0f);
        if (icon)
            Icons::draw (g, *icon, content.removeFromLeft (11.0f).withSizeKeepingCentre (11.0f, 11.0f), colour, 1.6f);
        drawText (g, text, content, bf, juce::Colours::white);
        badgeY += 20.0f;
    };

    if (v->hasNeuralModel())
        badge ("AI", Palette::cyan, Icon::Sparkle);
    if (blend)
        badge ("BLEND", juce::Colour (0xffc58bff), std::nullopt);
    if (! v->isFactory && ! v->hasNeuralModel())
        badge ("USER", Palette::textSoft, std::nullopt);

    // favourite star
    if (favourite || hover)
    {
        auto star = getStarArea();
        g.setColour (juce::Colour (0xb0050b1e));
        g.fillEllipse (star.reduced (1.0f));
        if (favourite)
            Icons::draw (g, Icon::StarFilled, star.reduced (5.0f), juce::Colour (0xffffd66b), 1.5f, overStar);
        else
            Icons::draw (g, Icon::Star, star.reduced (5.0f), overStar ? juce::Colours::white : Palette::textSoft, 1.6f, overStar);
    }
}

void VoiceCard::mouseEnter (const juce::MouseEvent& e)
{
    hover = true;
    overStar = getStarArea().contains (e.position);
    repaint();
}

void VoiceCard::mouseExit (const juce::MouseEvent&)
{
    hover = false;
    overStar = false;
    repaint();
}

void VoiceCard::mouseMove (const juce::MouseEvent& e)
{
    const bool s = getStarArea().contains (e.position);
    if (s != overStar)
    {
        overStar = s;
        repaint();
    }
}

void VoiceCard::mouseDown (const juce::MouseEvent& e)
{
    if (e.mods.isPopupMenu())
    {
        if (onContextMenu)
            onContextMenu (voiceIndex);
        return;
    }

    if (getStarArea().contains (e.position))
    {
        if (onToggleFavourite)
            onToggleFavourite (voiceIndex);
        return;
    }

    if (onSelect)
        onSelect (voiceIndex);
}

void VoiceCard::mouseDoubleClick (const juce::MouseEvent& e)
{
    if (! getStarArea().contains (e.position) && onDoubleClick)
        onDoubleClick (voiceIndex);
}

bool VoiceCard::keyPressed (const juce::KeyPress& k)
{
    if (k == juce::KeyPress::returnKey || k == juce::KeyPress::spaceKey)
    {
        if (onSelect) onSelect (voiceIndex);
        return true;
    }
    if (k.getTextCharacter() == 'f' || k.getTextCharacter() == 'F')
    {
        if (onToggleFavourite) onToggleFavourite (voiceIndex);
        return true;
    }
    if (k == juce::KeyPress (juce::KeyPress::F10Key, juce::ModifierKeys::shiftModifier, 0) || k.getTextCharacter() == 'm')
    {
        if (onContextMenu) onContextMenu (voiceIndex);
        return true;
    }
    return false;
}

std::unique_ptr<juce::AccessibilityHandler> VoiceCard::createAccessibilityHandler()
{
    return std::make_unique<juce::AccessibilityHandler> (*this, juce::AccessibilityRole::button,
                                                         juce::AccessibilityActions()
                                                             .addAction (juce::AccessibilityActionType::press, [this] { if (onSelect) onSelect (voiceIndex); })
                                                             .addAction (juce::AccessibilityActionType::showMenu, [this] { if (onContextMenu) onContextMenu (voiceIndex); }));
}

//==============================================================================
VoiceCarousel::VoiceCarousel (EditorContext& c, AvatarCache& a) : ctx (c), avatars (a)
{
    setFocusContainerType (FocusContainerType::focusContainer);
    setTitle ("Voice carousel");
}

void VoiceCarousel::setVoices (const std::vector<int>& indices)
{
    while (cards.size() > indices.size())
        cards.pop_back();

    while (cards.size() < indices.size())
    {
        auto card = std::make_unique<VoiceCard> (ctx, avatars);
        if (configureCard)
            configureCard (*card);
        addAndMakeVisible (*card);
        cards.push_back (std::move (card));
    }

    for (size_t i = 0; i < indices.size(); ++i)
        cards[i]->setVoiceIndex (indices[i]);

    targetOffset = juce::jlimit (0.0f, maxOffset(), targetOffset);
    offset = juce::jlimit (0.0f, maxOffset(), offset);
    updateStates();
    layoutCards();
}

void VoiceCarousel::updateStates()
{
    const auto& lib = ctx.processor.getVoiceLibrary();
    const auto selected = ctx.processor.getSelectedVoiceId();
    const auto blend = ctx.processor.getBlendVoiceId();
    const auto& voices = lib.getVoices();

    for (auto& c : cards)
    {
        const int idx = c->getVoiceIndex();
        if (! juce::isPositiveAndBelow (idx, (int) voices.size()))
            continue;
        const auto& id = voices[(size_t) idx].id;
        c->setState (id == selected, id == blend, lib.isFavourite (id));
    }
    layoutCards();
}

int VoiceCarousel::getVisibleCount() const
{
    return juce::jmax (1, juce::roundToInt (((float) getWidth() + gap) / (132.0f + gap)));
}

float VoiceCarousel::maxOffset() const
{
    return (float) juce::jmax (0, (int) cards.size() - getVisibleCount());
}

bool VoiceCarousel::canScroll (int direction) const
{
    return direction < 0 ? targetOffset > 0.01f : targetOffset < maxOffset() - 0.01f;
}

void VoiceCarousel::resized()
{
    const int n = getVisibleCount();
    cardW = ((float) getWidth() - gap * (float) (n - 1)) / (float) n;
    targetOffset = juce::jlimit (0.0f, maxOffset(), targetOffset);
    offset = juce::jlimit (0.0f, maxOffset(), offset);
    layoutCards();
}

void VoiceCarousel::layoutCards()
{
    const float h = (float) getHeight();
    const float step = cardW + gap;
    VoiceCard* selectedCard = nullptr;

    for (size_t i = 0; i < cards.size(); ++i)
    {
        auto& c = *cards[i];
        const float x = ((float) i - offset) * step;
        const bool visible = x + cardW > -step && x < (float) getWidth() + step;
        c.setVisible (visible);
        if (! visible)
            continue;

        auto b = juce::Rectangle<float> (x, 0.0f, cardW, h);
        c.setBounds (b.toNearestInt());

        const auto& voices = ctx.processor.getVoiceLibrary().getVoices();
        if (juce::isPositiveAndBelow (c.getVoiceIndex(), (int) voices.size())
            && voices[(size_t) c.getVoiceIndex()].id == ctx.processor.getSelectedVoiceId())
            selectedCard = &c;
    }

    if (selectedCard != nullptr)
        selectedCard->toFront (false);
}

void VoiceCarousel::scrollPages (int delta)
{
    const int n = getVisibleCount();
    targetOffset = juce::jlimit (0.0f, maxOffset(), std::round (targetOffset) + (float) (delta * juce::jmax (1, n - 1)));
}

void VoiceCarousel::scrollToVoice (int voiceIndex, bool animate)
{
    for (size_t i = 0; i < cards.size(); ++i)
    {
        if (cards[i]->getVoiceIndex() != voiceIndex)
            continue;

        const int n = getVisibleCount();
        const float pos = (float) i;
        if (pos < targetOffset || pos > targetOffset + (float) n - 1.0f)
            targetOffset = juce::jlimit (0.0f, maxOffset(), pos - (float) (n / 2));
        if (! animate)
        {
            offset = targetOffset;
            layoutCards();
        }
        return;
    }
}

void VoiceCarousel::tick (float dt)
{
    if (std::abs (offset - targetOffset) < 0.002f)
    {
        if (! juce::exactlyEqual (offset, targetOffset))
        {
            offset = targetOffset;
            layoutCards();
        }
        return;
    }

    offset += (targetOffset - offset) * juce::jmin (1.0f, dt * 11.0f);
    layoutCards();
}

void VoiceCarousel::mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails& w)
{
    const float d = std::abs (w.deltaX) > std::abs (w.deltaY) ? w.deltaX : w.deltaY;
    targetOffset = juce::jlimit (0.0f, maxOffset(), targetOffset - d * 6.0f);
}

//==============================================================================
VoiceGrid::VoiceGrid (EditorContext& c, AvatarCache& a) : ctx (c), avatars (a)
{
    setFocusContainerType (FocusContainerType::focusContainer);
    setTitle ("All voices");
}

void VoiceGrid::setVoices (const std::vector<int>& indices)
{
    while (cards.size() > indices.size())
        cards.pop_back();

    while (cards.size() < indices.size())
    {
        auto card = std::make_unique<VoiceCard> (ctx, avatars);
        if (configureCard)
            configureCard (*card);
        addAndMakeVisible (*card);
        cards.push_back (std::move (card));
    }

    for (size_t i = 0; i < indices.size(); ++i)
        cards[i]->setVoiceIndex (indices[i]);

    updateStates();
    layoutForWidth (getWidth());
}

void VoiceGrid::updateStates()
{
    const auto& lib = ctx.processor.getVoiceLibrary();
    const auto selected = ctx.processor.getSelectedVoiceId();
    const auto blend = ctx.processor.getBlendVoiceId();
    const auto& voices = lib.getVoices();

    for (auto& c : cards)
    {
        const int idx = c->getVoiceIndex();
        if (juce::isPositiveAndBelow (idx, (int) voices.size()))
        {
            const auto& id = voices[(size_t) idx].id;
            c->setState (id == selected, id == blend, lib.isFavourite (id));
        }
    }
}

void VoiceGrid::layoutForWidth (int width)
{
    if (width <= 0)
        return;

    const float gap = 12.0f, pad = 6.0f;
    const int cols = juce::jmax (1, (int) (((float) width - 2.0f * pad + gap) / (128.0f + gap)));
    const float cardW = ((float) width - 2.0f * pad - gap * (float) (cols - 1)) / (float) cols;
    const float cardH = cardW * 1.02f;
    const int rows = ((int) cards.size() + cols - 1) / cols;

    setSize (width, juce::jmax (10, juce::roundToInt (pad * 2.0f + (float) rows * (cardH + gap) - gap)));

    for (size_t i = 0; i < cards.size(); ++i)
    {
        const int col = (int) i % cols, row = (int) i / cols;
        cards[i]->setBounds (juce::Rectangle<float> (pad + (float) col * (cardW + gap), pad + (float) row * (cardH + gap), cardW, cardH).toNearestInt());
    }
}

//==============================================================================
SearchBox::SearchBox()
{
    editor.getProperties().set ("voxaTransparent", true);
    editor.setFont (font (17.0f));
    editor.setTextToShowWhenEmpty ("Search voices...", Palette::textDim);
    editor.setColour (juce::TextEditor::backgroundColourId, juce::Colours::transparentBlack);
    editor.setColour (juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
    editor.setColour (juce::TextEditor::focusedOutlineColourId, juce::Colours::transparentBlack);
    editor.setIndents (2, 0);
    editor.setJustification (juce::Justification::centredLeft);
    editor.setTitle ("Search voices");
    editor.setTooltip ("Search by name, genre, style or description");
    addAndMakeVisible (editor);
}

void SearchBox::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat().reduced (0.5f);
    g.setColour (juce::Colour (0xc0060d24));
    g.fillRoundedRectangle (r, 8.0f);
    g.setColour (editor.hasKeyboardFocus (true) ? Palette::glow.withAlpha (0.9f) : Palette::panelBorder.withAlpha (0.3f));
    g.drawRoundedRectangle (r, 8.0f, 1.0f);
    Icons::draw (g, Icon::Search, { 10.0f, r.getCentreY() - 10.0f, 20.0f, 20.0f }, Palette::textSoft, 2.0f);
}

void SearchBox::resized()
{
    editor.setBounds (getLocalBounds().withTrimmedLeft (38).reduced (0, 3));
}

//==============================================================================
VoiceLibraryPanel::VoiceLibraryPanel (EditorContext& c)
    : GlassPanel ("Voice Library"), ctx (c), carousel (c, avatars), grid (c, avatars)
{
    for (auto* comp : std::initializer_list<juce::Component*> { &allTab, &maleTab, &femaleTab, &stylesButton, &genresButton,
                                                                 &favouritesButton, &search, &gridButton, &leftArrow, &rightArrow,
                                                                 &carousel })
        addAndMakeVisible (comp);
    addChildComponent (gridViewport);

    for (auto* b : { &allTab, &maleTab, &femaleTab })
    {
        b->setClickingTogglesState (true);
        b->setRadioGroupId (2001);
        b->setStyle (GlassButton::Style::Flat);
        b->setFontHeight (18.0f);
        b->setCornerRadius (8.0f);
    }
    allTab.setToggleState (true, juce::dontSendNotification);
    allTab.setTooltip ("Show all voices");
    maleTab.setTooltip ("Show male voices");
    femaleTab.setTooltip ("Show female voices");
    allTab.onClick    = [this] { if (allTab.getToggleState())    { filter.gender = VoiceLibrary::Filter::Gender::Any;    refilter(); } };
    maleTab.onClick   = [this] { if (maleTab.getToggleState())   { filter.gender = VoiceLibrary::Filter::Gender::Male;   refilter(); } };
    femaleTab.onClick = [this] { if (femaleTab.getToggleState()) { filter.gender = VoiceLibrary::Filter::Gender::Female; refilter(); } };

    for (auto* b : { &stylesButton, &genresButton, &favouritesButton })
    {
        b->setStyle (GlassButton::Style::Flat);
        b->setFontHeight (18.0f);
        b->setCornerRadius (8.0f);
    }
    stylesButton.setTrailingIcon (Icon::ChevronDown);
    genresButton.setTrailingIcon (Icon::ChevronDown);
    stylesButton.setTooltip ("Filter by vocal style");
    genresButton.setTooltip ("Filter by genre");
    stylesButton.onClick = [this] { showStylesMenu(); };
    genresButton.onClick = [this] { showGenresMenu(); };

    favouritesButton.setClickingTogglesState (true);
    favouritesButton.setIconColour (Palette::text);
    favouritesButton.setTooltip ("Show only your favourite voices (click the star on a card)");
    favouritesButton.onClick = [this]
    {
        filter.favouritesOnly = favouritesButton.getToggleState();
        favouritesButton.setIcon (filter.favouritesOnly ? Icon::StarFilled : Icon::Star);
        refilter();
    };

    search.editor.onTextChange = [this] { refilter(); };
    search.editor.onEscapeKey = [this] { search.editor.clear(); refilter(); };

    gridButton.setIconScale (0.62f);
    gridButton.setStrokeWidth (2.0f);
    gridButton.setColour (Palette::text);
    gridButton.setTooltip ("Show all voices in a grid");
    gridButton.onClick = [this] { setExpanded (! expanded); };

    for (auto* a : { &leftArrow, &rightArrow })
    {
        a->setIconScale (0.8f);
        a->setStrokeWidth (2.0f);
        a->setColour (Palette::text);
    }
    leftArrow.onClick = [this] { carousel.scrollPages (-1); };
    rightArrow.onClick = [this] { carousel.scrollPages (1); };

    carousel.configureCard = [this] (VoiceCard& card) { configureCard (card); };
    grid.configureCard = [this] (VoiceCard& card) { configureCard (card); };

    gridViewport.setViewedComponent (&grid, false);
    gridViewport.setScrollBarsShown (true, false);
    gridViewport.setScrollBarThickness (8);

    knownVoiceCount = ctx.processor.getVoiceLibrary().getVoices().size();
    ctx.processor.getVoiceLibrary().addChangeListener (this);
    refilter();
}

VoiceLibraryPanel::~VoiceLibraryPanel()
{
    ctx.processor.getVoiceLibrary().removeChangeListener (this);
}

void VoiceLibraryPanel::configureCard (VoiceCard& card)
{
    card.onSelect = [this] (int index) { selectVoice (index); };
    card.onDoubleClick = [this] (int index)
    {
        selectVoice (index);
        if (expanded)
            setExpanded (false);
    };
    card.onToggleFavourite = [this] (int index)
    {
        const auto& voices = ctx.processor.getVoiceLibrary().getVoices();
        if (juce::isPositiveAndBelow (index, (int) voices.size()))
            ctx.processor.getVoiceLibrary().toggleFavourite (voices[(size_t) index].id);
    };
    card.onContextMenu = [this] (int index) { showCardMenu (index); };
}

void VoiceLibraryPanel::changeListenerCallback (juce::ChangeBroadcaster*)
{
    // Favourite toggles also land here: only drop the rendered portraits when the voice set changed.
    const auto numVoices = ctx.processor.getVoiceLibrary().getVoices().size();
    if (numVoices != knownVoiceCount)
    {
        knownVoiceCount = numVoices;
        avatars.clear();
    }
    refilter();
}

void VoiceLibraryPanel::refilter()
{
    filter.searchText = search.editor.getText();
    filtered = ctx.processor.getVoiceLibrary().filter (filter);
    carousel.setVoices (filtered);
    grid.setVoices (filtered);
    grid.layoutForWidth (gridViewport.getMaximumVisibleWidth());

    const int sel = ctx.processor.getVoiceLibrary().indexOf (ctx.processor.getSelectedVoiceId());
    carousel.scrollToVoice (sel, false);
    updateFilterButtons();
    repaint();
}

void VoiceLibraryPanel::updateFilterButtons()
{
    stylesButton.setButtonText (filter.style.isNotEmpty() ? filter.style : juce::String ("Styles"));
    stylesButton.setToggleState (filter.style.isNotEmpty(), juce::dontSendNotification);
    genresButton.setButtonText (filter.genre.isNotEmpty() ? filter.genre : juce::String ("Genres"));
    genresButton.setToggleState (filter.genre.isNotEmpty(), juce::dontSendNotification);
    leftArrow.setEnabled (carousel.canScroll (-1));
    rightArrow.setEnabled (carousel.canScroll (1));
}

void VoiceLibraryPanel::selectVoice (int index)
{
    const auto& voices = ctx.processor.getVoiceLibrary().getVoices();
    if (! juce::isPositiveAndBelow (index, (int) voices.size()))
        return;

    const auto& v = voices[(size_t) index];
    if (v.id == ctx.processor.getSelectedVoiceId())
        return;

    ctx.undo.beginNewTransaction ("Select voice");
    ctx.processor.selectVoice (v.id);
    if (v.id == ctx.processor.getBlendVoiceId())
        ctx.processor.setBlendVoice ({});
}

void VoiceLibraryPanel::showCardMenu (int index)
{
    auto& lib = ctx.processor.getVoiceLibrary();
    const auto& voices = lib.getVoices();
    if (! juce::isPositiveAndBelow (index, (int) voices.size()))
        return;

    const auto v = voices[(size_t) index];
    const bool isSelected = v.id == ctx.processor.getSelectedVoiceId();
    const bool isBlend = v.id == ctx.processor.getBlendVoiceId();
    const bool hasBlend = ctx.processor.getBlendVoiceId().isNotEmpty();
    const bool fav = lib.isFavourite (v.id);

    juce::PopupMenu m;
    m.addSectionHeader (v.name);
    m.addItem (1, "Select Voice", ! isSelected, isSelected);
    m.addItem (2, fav ? "Remove from Favorites" : "Add to Favorites");
    m.addSeparator();
    m.addItem (3, "Set as Blend Voice", ! isSelected, isBlend);
    m.addItem (4, "Clear Blend Voice", hasBlend);
    if (! v.isFactory && v.packDirectory.isDirectory())
    {
        m.addSeparator();
        m.addItem (5, "Show Voice Pack Folder");
    }

    m.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (this).withMousePosition(), [this, v, index] (int r)
    {
        switch (r)
        {
            case 1: selectVoice (index); break;
            case 2: ctx.processor.getVoiceLibrary().toggleFavourite (v.id); break;
            case 3:
            {
                ctx.undo.beginNewTransaction ("Set blend voice");
                ctx.processor.setBlendVoice (v.id);
                if (ctx.getValue (ParamID::voiceBlend) < 1.0f)
                    if (auto* p = ctx.state.getParameter (ParamID::voiceBlend))
                        p->setValueNotifyingHost (p->convertTo0to1 (50.0f));
                ctx.toast ("Blending " + ctx.processor.getSelectedVoice().name + " with " + v.name
                           + " - adjust Voice Blend in Voice Character");
                break;
            }
            case 4:
                ctx.undo.beginNewTransaction ("Clear blend voice");
                ctx.processor.setBlendVoice ({});
                break;
            case 5: v.packDirectory.startAsProcess(); break;
            default: break;
        }
    });
}

void VoiceLibraryPanel::showStylesMenu()
{
    const auto styles = ctx.processor.getVoiceLibrary().getStyles();
    juce::PopupMenu m;
    m.addItem (1, "All Styles", true, filter.style.isEmpty());
    m.addSeparator();
    for (int i = 0; i < styles.size(); ++i)
        m.addItem (100 + i, styles[i], true, styles[i] == filter.style);

    m.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&stylesButton).withMinimumNumColumns (3).withMaximumNumColumns (3), [this, styles] (int r)
    {
        if (r == 1)
            filter.style = {};
        else if (r >= 100)
            filter.style = styles[r - 100];
        else
            return;
        refilter();
    });
}

void VoiceLibraryPanel::showGenresMenu()
{
    const auto genres = ctx.processor.getVoiceLibrary().getGenres();
    juce::PopupMenu m;
    m.addItem (1, "All Genres", true, filter.genre.isEmpty());
    m.addSeparator();
    for (int i = 0; i < genres.size(); ++i)
        m.addItem (100 + i, genres[i], true, genres[i] == filter.genre);

    m.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&genresButton).withMinimumNumColumns (2).withMaximumNumColumns (2), [this, genres] (int r)
    {
        if (r == 1)
            filter.genre = {};
        else if (r >= 100)
            filter.genre = genres[r - 100];
        else
            return;
        refilter();
    });
}

void VoiceLibraryPanel::setExpanded (bool e)
{
    if (e == expanded)
        return;

    expanded = e;
    gridButton.setToggleState (e, juce::dontSendNotification);
    gridButton.setTooltip (e ? "Close the voice grid" : "Show all voices in a grid");
    gridButton.setIcon (e ? Icon::Close : Icon::List);

    if (onExpandedChanged)
        onExpandedChanged (e);

    carousel.setVisible (! e);
    leftArrow.setVisible (! e);
    rightArrow.setVisible (! e);
    gridViewport.setVisible (e);
    resized();
    repaint();
}

void VoiceLibraryPanel::resized()
{
    allTab.setBounds (218, 6, 92, 38);
    maleTab.setBounds (316, 6, 92, 38);
    femaleTab.setBounds (414, 6, 104, 38);
    stylesButton.setBounds (532, 6, 128, 38);
    genresButton.setBounds (668, 6, 140, 38);
    favouritesButton.setBounds (816, 6, 140, 38);
    search.setBounds (getWidth() - 392, 6, 300, 38);
    gridButton.setBounds (getWidth() - 66, 6, 50, 38);

    const int top = (int) titleHeight + 8;
    leftArrow.setBounds (6, top + 36, 40, 66);
    rightArrow.setBounds (getWidth() - 46, top + 36, 40, 66);
    carousel.setBounds (52, top, getWidth() - 104, getHeight() - top - 10);

    gridViewport.setBounds (16, top, getWidth() - 32, getHeight() - top - 14);
    grid.layoutForWidth (gridViewport.getMaximumVisibleWidth());
}

void VoiceLibraryPanel::paintStatic (juce::Graphics& g, juce::Rectangle<float> area)
{
    GlassPanel::paintStatic (g, area);
}

void VoiceLibraryPanel::paint (juce::Graphics& g)
{
    if (expanded)
    {
        auto r = getLocalBounds().toFloat().reduced (1.0f);
        g.setColour (juce::Colour (0xff081230));
        g.fillRoundedRectangle (r, cornerRadius);
        paintGlassCard (g, r, nullptr, {}, 1.0f);
        paintTitle (g, r.withHeight (titleHeight));
        drawText (g, juce::String ((int) filtered.size()) + " voices", { (float) search.getX() - 130.0f, 6.0f, 120.0f, 38.0f },
                  font (15.0f), Palette::textDim, juce::Justification::centredRight);
    }

    GlassPanel::paint (g);

    if (filtered.empty())
        drawText (g, "No voices match your filters", carousel.getBounds().toFloat(), font (18.0f), Palette::textSoft);
}

bool VoiceLibraryPanel::keyPressed (const juce::KeyPress& k)
{
    if (k == juce::KeyPress::escapeKey && expanded)
    {
        setExpanded (false);
        return true;
    }
    return false;
}

void VoiceLibraryPanel::tick (float dt)
{
    const auto sel = ctx.processor.getSelectedVoiceId();
    const auto blend = ctx.processor.getBlendVoiceId();
    if (sel != shownSelected || blend != shownBlend)
    {
        const bool selectionChanged = sel != shownSelected;
        shownSelected = sel;
        shownBlend = blend;
        carousel.updateStates();
        grid.updateStates();
        if (selectionChanged)
            carousel.scrollToVoice (ctx.processor.getVoiceLibrary().indexOf (sel), true);
    }

    carousel.tick (dt);
    leftArrow.setEnabled (carousel.canScroll (-1));
    rightArrow.setEnabled (carousel.canScroll (1));
}

} // namespace voxa::ui
