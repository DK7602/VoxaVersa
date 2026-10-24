#pragma once

#include "EditorContext.h"
#include "GlassPanel.h"
#include "Buttons.h"
#include "Knob.h"
#include "IOPanels.h"

namespace voxa::ui
{

/** The big glowing Generate button (shows progress while singing is rendered). */
class GenerateButton : public juce::Button
{
public:
    GenerateButton();
    void setProgress (std::optional<float> progress01);
    void paintButton (juce::Graphics&, bool isHighlighted, bool isDown) override;

private:
    std::optional<float> progress;
};

//==============================================================================
/** Lyrics editor, melody source, key, tempo, Generate, and the rendered take's preview/export. */
class TextToSingPanel : public GlassPanel
{
public:
    explicit TextToSingPanel (EditorContext&);
    ~TextToSingPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void tick (float dt);

    /** Splits lyrics into approximate syllable ranges (words, split at hyphens). */
    static juce::Array<juce::Range<int>> approximateSyllables (const juce::String& lyrics);

private:
    void generate();
    void updateSourceControls();
    void updateTakeControls();
    void showMenu();
    void loadMidiFile();
    juce::File exportMidi (bool quiet);

    EditorContext& ctx;

    juce::TextEditor lyrics;
    juce::ComboBox melodySource, key;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> melodyAttachment, keyAttachment;
    std::unique_ptr<NumberBox> tempo;
    IconButton syncButton { "Sync tempo to host", Icon::Sync };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> syncAttachment;
    GenerateButton generateButton;
    IconButton menuButton { "Text to Sing options", Icon::Link };

    // melody-source specific
    GlassButton captureButton { "Capture", Icon::Midi };
    IconButton clearCaptureButton { "Clear captured notes", Icon::Close };
    GlassButton loadMidiButton { "Load MIDI...", Icon::Folder };
    GlassButton humButton { "Hum", Icon::Mic };
    juce::ComboBox autoScale;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> autoScaleAttachment;

    // rendered take
    IconButton previewButton { "Preview take", Icon::Play, IconButton::Background::Circle };
    DragFileButton renderDragButton { "Render & Drag" };
    DragFileButton midiButton { "MIDI", Icon::Export };

    juce::Rectangle<float> lyricsBox, sourceInfoArea, stripArea;
    juce::String sourceInfo, timeText, liveText;
    int shownSource = -1, shownMode = -1;
    VoxaVersaProcessor::SingState lastState = VoxaVersaProcessor::SingState::Idle;
    juce::File exportedMidi;
    int lastSyllable = -2;
    juce::int64 infoSignature = -1;
    int shownSyncSignature = -1, shownTimeSignature = -1, shownRenderSignature = -1;
    juce::String lastLyricsSeen;
    juce::Array<juce::Range<int>> syllables;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TextToSingPanel)
};

} // namespace voxa::ui
