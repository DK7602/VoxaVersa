#pragma once

#include "EditorContext.h"
#include "GlassPanel.h"
#include "Buttons.h"
#include "VoiceAvatar.h"

namespace voxa::ui
{

//==============================================================================
/** One voice in the library: stylised portrait, name, favourite star, AI / blend badges. */
class VoiceCard : public juce::Component,
                  public juce::SettableTooltipClient
{
public:
    VoiceCard (EditorContext&, AvatarCache&);

    void setVoiceIndex (int index);
    int getVoiceIndex() const noexcept { return voiceIndex; }
    void setState (bool selected, bool blend, bool favourite);

    std::function<void (int)> onSelect, onToggleFavourite, onContextMenu, onDoubleClick;

    void paint (juce::Graphics&) override;
    void mouseEnter (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDoubleClick (const juce::MouseEvent&) override;
    bool keyPressed (const juce::KeyPress&) override;
    void focusGained (FocusChangeType) override { repaint(); }
    void focusLost (FocusChangeType) override { repaint(); }
    std::unique_ptr<juce::AccessibilityHandler> createAccessibilityHandler() override;

private:
    juce::Rectangle<float> getStarArea() const;
    const VoiceProfile* getVoice() const;

    EditorContext& ctx;
    AvatarCache& avatars;
    int voiceIndex = -1;
    bool selected = false, blend = false, favourite = false, hover = false, overStar = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoiceCard)
};

//==============================================================================
/** Horizontally scrolling row of voice cards. */
class VoiceCarousel : public juce::Component
{
public:
    VoiceCarousel (EditorContext&, AvatarCache&);

    void setVoices (const std::vector<int>& voiceIndices);
    void updateStates();
    void scrollPages (int delta);
    void scrollToVoice (int voiceIndex, bool animate);
    bool canScroll (int direction) const;
    void tick (float dt);

    std::function<void (VoiceCard&)> configureCard;

    void resized() override;
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails&) override;

private:
    void layoutCards();
    int getVisibleCount() const;
    float maxOffset() const;

    EditorContext& ctx;
    AvatarCache& avatars;
    std::vector<std::unique_ptr<VoiceCard>> cards;
    float offset = 0.0f, targetOffset = 0.0f;
    float cardW = 130.0f, gap = 10.0f, margin = 0.0f;
};

//==============================================================================
/** The expanded grid of all (filtered) voices. */
class VoiceGrid : public juce::Component
{
public:
    VoiceGrid (EditorContext&, AvatarCache&);

    void setVoices (const std::vector<int>& voiceIndices);
    void updateStates();
    void layoutForWidth (int width);

    std::function<void (VoiceCard&)> configureCard;

private:
    EditorContext& ctx;
    AvatarCache& avatars;
    std::vector<std::unique_ptr<VoiceCard>> cards;
};

//==============================================================================
class SearchBox : public juce::Component
{
public:
    SearchBox();
    void paint (juce::Graphics&) override;
    void resized() override;
    juce::TextEditor editor;
};

//==============================================================================
class VoiceLibraryPanel : public GlassPanel,
                          private juce::ChangeListener
{
public:
    explicit VoiceLibraryPanel (EditorContext&);
    ~VoiceLibraryPanel() override;

    void paint (juce::Graphics&) override;
    void paintStatic (juce::Graphics&, juce::Rectangle<float>) override;
    void resized() override;
    bool keyPressed (const juce::KeyPress&) override;
    void tick (float dt);

    bool isExpanded() const noexcept { return expanded; }
    void setExpanded (bool shouldExpand);

    /** Called when the grid is opened/closed so the owner can resize this panel. */
    std::function<void (bool expanded)> onExpandedChanged;

private:
    void changeListenerCallback (juce::ChangeBroadcaster*) override;
    void refilter();
    void configureCard (VoiceCard&);
    void selectVoice (int index);
    void showCardMenu (int index);
    void showStylesMenu();
    void showGenresMenu();
    void updateFilterButtons();

    EditorContext& ctx;
    AvatarCache avatars;

    GlassButton allTab { "All" }, maleTab { "Male" }, femaleTab { "Female" };
    GlassButton stylesButton { "Styles" }, genresButton { "Genres" };
    GlassButton favouritesButton { "Favorites", Icon::Star };
    SearchBox search;
    IconButton gridButton { "Show all voices", Icon::List };
    IconButton leftArrow { "Previous voices", Icon::ChevronLeft };
    IconButton rightArrow { "Next voices", Icon::ChevronRight };

    VoiceCarousel carousel;
    juce::Viewport gridViewport;
    VoiceGrid grid;

    VoiceLibrary::Filter filter;
    std::vector<int> filtered;
    juce::String shownSelected, shownBlend;
    size_t knownVoiceCount = 0;
    bool expanded = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoiceLibraryPanel)
};

} // namespace voxa::ui
