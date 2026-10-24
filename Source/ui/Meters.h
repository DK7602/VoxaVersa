#pragma once

#include "Theme.h"
#include "core/AudioFeeds.h"
#include <array>

namespace voxa::ui
{

//==============================================================================
/** Stereo LED bar meter with a dB scale (0, -6, -12, -24, -48). */
class LevelMeter : public juce::Component
{
public:
    explicit LevelMeter (LevelMeterSource& source);

    /** Pulls the latest levels and repaints when the display changed. */
    void tick (float deltaSeconds);

    void paint (juce::Graphics&) override;
    void resized() override;

    /** Maps a level in dB to 0 (top, 0 dB) .. 1 (bottom, -48 dB). */
    static float dbToPosition (float db) noexcept;

private:
    LevelMeterSource& source;
    std::array<float, 2> levelDb { -100.0f, -100.0f }, peakDb { -100.0f, -100.0f },
                         peakAge { 0.0f, 0.0f }, clipAge { 10.0f, 10.0f };
    std::array<int, 2> drawnLit { -1, -1 }, drawnPeak { -1, -1 };
    std::array<bool, 2> drawnClip { false, false };
    std::array<juce::Rectangle<float>, 2> bars;
    juce::Rectangle<float> scaleArea;

    static constexpr int numSegments = 26;
    int segmentsFor (float db) const noexcept;
};

//==============================================================================
/** Glowing scrolling waveform of a ScopeBuffer. */
class WaveformScope : public juce::Component
{
public:
    explicit WaveformScope (ScopeBuffer& buffer);

    void tick();
    void setHighlighted (bool shouldHighlight);
    void setPlaceholder (const juce::String& text);
    void setPlayPosition (double position01);   // < 0 hides the marker

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    ScopeBuffer& buffer;
    static constexpr int maxPoints = 512;
    std::array<float, maxPoints> mins {}, maxs {}, amps {}, drawnAmps {};
    int numPoints = 160;
    bool highlighted = false, silent = true;
    double playPosition = -1.0;
    juce::String placeholder;
    juce::Image background, frame;   // cached inset / reused bar buffer (physical resolution)
    float backgroundScale = 0.0f;
    std::array<juce::PixelARGB, 256> colourTable;
    juce::PixelARGB glowPixel;

    void buildColourTable();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (WaveformScope)
};

} // namespace voxa::ui
