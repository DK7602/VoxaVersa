#pragma once

#include "EditorContext.h"
#include "HeaderBar.h"
#include "IOPanels.h"
#include "CentreDisplay.h"
#include "VoiceLibraryPanel.h"
#include "VoiceCharacterPanel.h"
#include "AdvancedPanel.h"
#include "TextToSingPanel.h"
#include "Toast.h"

namespace voxa::ui
{

/** The whole UI laid out on the 1536 x 1024 design canvas. The editor scales it uniformly.
    Static art (backdrop, frosted glass cards, orb, wordmarks, titles) is cached in one image
    rendered at the physical resolution; only dynamic regions repaint at runtime. */
class MainView : public juce::Component
{
    EditorContext& ctx;   // declared first: initialised before the panels

public:
    static constexpr int designWidth = 1536, designHeight = 1024;

    explicit MainView (EditorContext&);
    ~MainView() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails&) override;

    /** Animations, meters and polling (called by the editor's timer). */
    void tick (float dt);

    /** Call when the on-screen scale changes; the cached background is re-rendered once resizing settles. */
    void noteScaleChanged();

    HeaderBar header;
    InputPanel input;
    OutputPanel output;
    CentreDisplay centre;
    VoiceLibraryPanel library;
    VoiceCharacterPanel character;
    AdvancedPanel advanced;
    TextToSingPanel sing;
    ToastOverlay toast;

private:
    void renderBackground (float scale);
    void paintBackdrop (juce::Graphics&);
    static void drawTagline (juce::Graphics&, juce::Rectangle<float> area);
    std::vector<GlassPanel*> getGlassPanels();

    juce::Image background;
    float backgroundScale = 0.0f;
    juce::uint32 lastScaleChange = 0;
    bool refreshPending = false;
    juce::uint32 lastWheelTime = 0;

    juce::Image orbImage, wordmarkImage;
    juce::Rectangle<int> libraryBounds { 18, 583, 1500, 204 }, libraryExpandedBounds { 18, 100, 1500, 687 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainView)
};

} // namespace voxa::ui
