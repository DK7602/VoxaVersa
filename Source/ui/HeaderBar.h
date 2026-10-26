#pragma once

#include "EditorContext.h"
#include "Buttons.h"

namespace voxa::ui
{

/** Top bar: settings, mode tabs, preset browser, A/B, save, undo/redo and power. */
class HeaderBar : public juce::Component,
                  private juce::ChangeListener
{
public:
    /** Window widths offered by Settings > UI Size (3:2). */
    static constexpr int uiWidths[] = { 900, 1050, 1200, 1350, 1536, 1728, 1920 };

    explicit HeaderBar (EditorContext&);
    ~HeaderBar() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;

    /** Updates undo/redo enablement and the preset name; called from the editor timer. */
    void tick();

    /** The header's tagline and the area it is drawn into (header coordinates; rendered in the cached background). */
    static const char* const tagline;
    juce::Rectangle<float> getTaglineArea() const;

    std::function<void (int width)> onUiSizeChosen;
    std::function<int()> getUiWidth;

    void showSettingsMenu();
    void showPresetMenu();
    void savePresetDialog();

private:
    class PresetNameButton;

    void changeListenerCallback (juce::ChangeBroadcaster*) override;
    void updateModeButtons (int mode);
    void updatePresetLabel();
    void confirmDeletePreset();
    void importPreset();
    void importVoicePack();
    void showAbout();
    void showABMenu();
    juce::Component* getDialogParent() { return getParentComponent() != nullptr ? getParentComponent() : this; }

    EditorContext& ctx;

    IconButton gearButton { "Settings", Icon::Gear, IconButton::Background::Rounded };
    ModeTab voiceTab { "Voice Transform", "Use your vocal", Icon::Mic };
    ModeTab singTab { "Text to Sing", "Type lyrics", Icon::Document };
    std::unique_ptr<juce::ParameterAttachment> modeAttachment;

    IconButton prevPreset { "Previous preset", Icon::ChevronLeft };
    IconButton nextPreset { "Next preset", Icon::ChevronRight };
    std::unique_ptr<PresetNameButton> presetName;
    GlassButton slotA { "A" }, slotB { "B" };
    IconButton saveButton { "Save preset", Icon::Save, IconButton::Background::Rounded };
    IconButton undoButton { "Undo", Icon::Undo };
    IconButton redoButton { "Redo", Icon::Redo };
    PowerButton powerButton;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> powerAttachment;

    juce::Rectangle<float> presetPill, abPill, undoPill;
    std::unique_ptr<juce::AlertWindow> dialog;
    juce::String shownPresetName;
    int shownSlot = -1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HeaderBar)
};

} // namespace voxa::ui
