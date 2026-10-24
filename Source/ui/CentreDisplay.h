#pragma once

#include "EditorContext.h"

namespace voxa::ui
{

/** The centre: glowing orb (static art lives in the cached background, the animated halo and
    voice bars are drawn here), the pitch read-outs and engine status. */
class CentreDisplay : public juce::Component
{
public:
    explicit CentreDisplay (EditorContext&);

    void paint (juce::Graphics&) override;
    void resized() override;
    void tick (float dt);

    /** Orb and wordmark positions in this component's coordinates (for the cached background). */
    juce::Rectangle<float> getOrbArea() const noexcept       { return orbArea; }
    juce::Rectangle<float> getWordmarkArea() const noexcept  { return wordmarkArea; }

private:
    void updateReadouts();
    void paintPill (juce::Graphics&, juce::Rectangle<float> area, const juce::String& label,
                    const juce::String& value, const juce::String& sub, bool lit);
    void drawPill (juce::Graphics&, int index, juce::Rectangle<float> area, const juce::String& label,
                   const juce::String& value, const juce::String& sub, bool lit, float phys);

    EditorContext& ctx;
    juce::Rectangle<float> orbArea, wordmarkArea, leftPill, rightPill, leftPill2, rightPill2;

    juce::Image haloImage;
    float haloScale = 0.0f;
    juce::Array<juce::Rectangle<int>> ringTiles;    // the halo repaints only where the ring is
    juce::Rectangle<int> barsArea;

    struct PillCache { juce::Image image; float scale = 0.0f; bool dirty = true; };
    std::array<PillCache, 4> pills;

    float level = 0.0f, phase = 0.0f, drawnGlow = -1.0f;
    static constexpr int numBars = 17;
    std::array<float, numBars> bars {}, drawnBars {};
    std::array<float, 64> scopeMin {}, scopeMax {};

    juce::String inNote, outNote, keyText, keySub, engineText, engineSub;
    bool engineLit = false;
    int readoutCounter = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CentreDisplay)
};

} // namespace voxa::ui
