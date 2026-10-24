#pragma once

#include "EditorContext.h"
#include "GlassPanel.h"
#include "Buttons.h"
#include "Knob.h"
#include "Meters.h"

namespace voxa::ui
{

//==============================================================================
/** A button that drags a file out of the plugin into the host (DAW) or the desktop. */
class DragFileButton : public GlassButton
{
public:
    DragFileButton (const juce::String& buttonText, std::optional<Icon> buttonIcon = Icon::Drag);

    /** Returns the file to drag (may create it on demand); empty File = nothing to drag. */
    std::function<juce::File()> getFileToDrag;

    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;

private:
    bool dragging = false;
};

//==============================================================================
class InputPanel : public GlassPanel,
                   public juce::FileDragAndDropTarget
{
public:
    explicit InputPanel (EditorContext&);
    ~InputPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void tick (float dt);

    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void fileDragEnter (const juce::StringArray&, int, int) override;
    void fileDragExit (const juce::StringArray&) override;
    void filesDropped (const juce::StringArray& files, int, int) override;

    static bool isAudioFile (const juce::String& path);

private:
    void loadSample (const juce::File&);
    void showSourceMenu();
    void updateSampleControls();

    EditorContext& ctx;
    IconButton linkButton { "Input source", Icon::Link };
    WaveformScope scope;
    Knob gain;
    LevelMeter meter;
    GlassButton autoLevel { "Auto Level", Icon::AutoLevel };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> autoLevelAttachment;

    IconButton playButton { "Play sample", Icon::Play, IconButton::Background::Circle };
    GlassButton renderButton { "Render", Icon::Export };
    DragFileButton dragButton { "Drag" };
    IconButton clearButton { "Remove sample", Icon::Close };
    GlassButton recordButton { "Rec", Icon::Record };

    juce::String caption, shownSampleName;
    bool hadSample = false, wasRecording = false, dropHover = false;
    int shownCaptionSignature = -1, shownRecordSignature = -1;
    juce::Rectangle<float> captionArea;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (InputPanel)
};

//==============================================================================
class OutputPanel : public GlassPanel
{
public:
    explicit OutputPanel (EditorContext&);
    ~OutputPanel() override;

    void paint (juce::Graphics&) override;
    void paintStatic (juce::Graphics&, juce::Rectangle<float>) override;
    void resized() override;
    void tick (float dt);

private:
    void showOutputMenu();

    EditorContext& ctx;
    IconButton linkButton { "Output options", Icon::Link };
    WaveformScope scope;
    Knob gain;
    LevelMeter meter;
    GlassButton mixLabel { "Mix" };
    juce::Slider mixSlider;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> mixAttachment;
    ValueBox mixValue { mixSlider };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (OutputPanel)
};

} // namespace voxa::ui
