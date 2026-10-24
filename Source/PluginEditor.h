#pragma once

#include "PluginProcessor.h"
#include "ui/EditorContext.h"
#include "ui/VoxaLookAndFeel.h"

namespace voxa
{

namespace ui { class MainView; }

/** VoxaVersa's editor: hosts the 1536 x 1024 design canvas (ui::MainView) scaled uniformly
    to the window size (fixed 3:2 aspect ratio, 900 x 600 .. 1920 x 1280). */
class VoxaVersaEditor : public juce::AudioProcessorEditor,
                        private juce::Timer
{
public:
    explicit VoxaVersaEditor (VoxaVersaProcessor&);
    ~VoxaVersaEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    bool keyPressed (const juce::KeyPress&) override;

    static constexpr int defaultWidth = 1200, minWidth = 900, maxWidth = 1920;

private:
    void timerCallback() override;
    void setUiWidth (int width);
    int loadSavedWidth() const;
    void saveWidthToSettings() const;

    struct Snapshot;

    VoxaVersaProcessor& processor;
    juce::SharedResourcePointer<ui::VoxaLookAndFeel> lookAndFeel;
    ui::EditorContext context;
    std::unique_ptr<ui::MainView> view;
    std::unique_ptr<juce::TooltipWindow> tooltips;
    std::unique_ptr<Snapshot> snapshot;
    double lastTickMs = 0.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoxaVersaEditor)
};

} // namespace voxa
