#include "HeaderBar.h"
#include "voices/VoiceLibrary.h"

namespace voxa::ui
{

//==============================================================================
class HeaderBar::PresetNameButton : public juce::Button
{
public:
    PresetNameButton() : juce::Button ("Preset")
    {
        setWantsKeyboardFocus (true);
        setMouseCursor (juce::MouseCursor::PointingHandCursor);
        setTitle ("Preset browser");
        setTooltip ("Browse presets by category");
    }

    void setPresetName (const juce::String& n)
    {
        if (n != name)
        {
            name = n;
            setDescription ("Current preset: " + n);
            repaint();
        }
    }

    void paintButton (juce::Graphics& g, bool hover, bool down) override
    {
        auto r = getLocalBounds().toFloat().reduced (6.0f, 0.0f);
        if (hover || down)
        {
            g.setColour (juce::Colours::white.withAlpha (down ? 0.10f : 0.05f));
            g.fillRoundedRectangle (r.reduced (0.0f, 6.0f), 6.0f);
        }
        if (hasKeyboardFocus (false))
        {
            g.setColour (Palette::glowBright.withAlpha (0.7f));
            g.drawRoundedRectangle (r.reduced (0.0f, 6.0f), 6.0f, 1.0f);
        }

        auto chevron = r.removeFromRight (22.0f);
        Icons::draw (g, Icon::ChevronDown, chevron.withSizeKeepingCentre (14.0f, 14.0f), Palette::text, 2.4f);
        drawText (g, name, r.withTrimmedLeft (6.0f), font (17.5f, Weight::Medium), hover ? juce::Colours::white : Palette::text,
                  juce::Justification::centredLeft);
    }

private:
    juce::String name;
};

//==============================================================================
HeaderBar::HeaderBar (EditorContext& c) : ctx (c)
{
    presetName = std::make_unique<PresetNameButton>();

    for (auto* comp : std::initializer_list<juce::Component*> { &gearButton, &voiceTab, &singTab, &prevPreset, presetName.get(),
                                                                 &nextPreset, &slotA, &slotB, &saveButton, &undoButton,
                                                                 &redoButton, &powerButton })
        addAndMakeVisible (comp);

    gearButton.setIconScale (0.55f);
    gearButton.setTooltip ("Settings: UI size, quality, AI engine, voice packs, presets");
    gearButton.onClick = [this] { showSettingsMenu(); };

    voiceTab.setTooltip ("Voice Transform: turn your own vocal into the selected voice");
    singTab.setTooltip ("Text to Sing: type lyrics and let the selected voice sing them");
    voiceTab.setRadioGroupId (1001);
    singTab.setRadioGroupId (1001);

    if (auto* modeParam = ctx.state.getParameter (ParamID::mode))
    {
        modeAttachment = std::make_unique<juce::ParameterAttachment> (*modeParam, [this] (float v) { updateModeButtons (juce::roundToInt (v)); }, &ctx.undo);
        modeAttachment->sendInitialUpdate();
    }
    voiceTab.onClick = [this] { if (modeAttachment) modeAttachment->setValueAsCompleteGesture (0.0f); };
    singTab.onClick = [this] { if (modeAttachment) modeAttachment->setValueAsCompleteGesture (1.0f); };

    prevPreset.setIconScale (0.5f);
    nextPreset.setIconScale (0.5f);
    prevPreset.setStrokeWidth (2.2f);
    nextPreset.setStrokeWidth (2.2f);
    prevPreset.onClick = [this] { ctx.processor.getPresetManager().loadPrevious(); updatePresetLabel(); };
    nextPreset.onClick = [this] { ctx.processor.getPresetManager().loadNext(); updatePresetLabel(); };
    presetName->onClick = [this] { showPresetMenu(); };

    slotA.setTooltip ("A/B comparison: slot A (right-click to copy between slots)");
    slotB.setTooltip ("A/B comparison: slot B (right-click to copy between slots)");
    slotA.setTitle ("Compare slot A");
    slotB.setTitle ("Compare slot B");
    slotA.setCornerRadius (6.0f);
    slotB.setCornerRadius (6.0f);
    slotA.setFontHeight (17.0f);
    slotB.setFontHeight (17.0f);
    slotA.setStyle (GlassButton::Style::Flat);
    slotB.setStyle (GlassButton::Style::Flat);
    slotA.onClick = [this] { ctx.processor.getPresetManager().switchToSlot (0); updatePresetLabel(); };
    slotB.onClick = [this] { ctx.processor.getPresetManager().switchToSlot (1); updatePresetLabel(); };
    slotA.addMouseListener (this, false);
    slotB.addMouseListener (this, false);

    saveButton.setIconScale (0.5f);
    saveButton.setTooltip ("Save the current settings as a user preset");
    saveButton.onClick = [this] { savePresetDialog(); };

    undoButton.setIconScale (0.62f);
    redoButton.setIconScale (0.62f);
    undoButton.setStrokeWidth (2.0f);
    redoButton.setStrokeWidth (2.0f);
    undoButton.setColour (Palette::text);
    redoButton.setColour (Palette::text);
    undoButton.setTooltip ("Undo (Ctrl/Cmd+Z)");
    redoButton.setTooltip ("Redo (Ctrl/Cmd+Shift+Z)");
    undoButton.onClick = [this] { ctx.undo.undo(); };
    redoButton.onClick = [this] { ctx.undo.redo(); };

    powerAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (ctx.state, ParamID::power, powerButton);

    ctx.processor.getPresetManager().addChangeListener (this);
    updatePresetLabel();
    tick();
}

HeaderBar::~HeaderBar()
{
    ctx.processor.getPresetManager().removeChangeListener (this);
    slotA.removeMouseListener (this);
    slotB.removeMouseListener (this);
}

juce::Rectangle<float> HeaderBar::getWordmarkArea() const
{
    return { 606.0f, 8.0f, 324.0f, 76.0f };
}

void HeaderBar::resized()
{
    gearButton.setBounds (16, 22, 50, 50);
    voiceTab.setBounds (74, 9, 234, 74);
    singTab.setBounds (310, 9, 204, 74);

    presetPill = { 956.0f, 24.0f, 288.0f, 48.0f };
    prevPreset.setBounds (960, 28, 34, 40);
    presetName->setBounds (994, 24, 214, 48);
    nextPreset.setBounds (1208, 28, 34, 40);

    abPill = { 1252.0f, 24.0f, 72.0f, 48.0f };
    slotA.setBounds (1255, 27, 33, 42);
    slotB.setBounds (1288, 27, 33, 42);

    saveButton.setBounds (1331, 23, 50, 50);

    undoPill = { 1388.0f, 24.0f, 72.0f, 48.0f };
    undoButton.setBounds (1390, 28, 34, 40);
    redoButton.setBounds (1424, 28, 34, 40);

    powerButton.setBounds (1464, 15, 66, 66);
}

void HeaderBar::paint (juce::Graphics& g)
{
    // dark glass tray holding the two mode tabs
    {
        const auto tray = juce::Rectangle<float> (70.0f, 5.0f, 448.0f, 82.0f);
        g.setGradientFill (juce::ColourGradient (juce::Colour (0xff1b2c58).withAlpha (0.55f), tray.getX(), tray.getY(),
                                                 juce::Colour (0xff081230).withAlpha (0.7f), tray.getX(), tray.getBottom(), false));
        g.fillRoundedRectangle (tray, 14.0f);
        g.setColour (juce::Colours::white.withAlpha (0.4f));
        g.drawRoundedRectangle (tray.reduced (0.5f), 14.0f, 1.0f);
    }

    drawGlassControl (g, presetPill, 10.0f, false, presetName->isMouseOver(), false);
    fillInset (g, presetName->getBounds().toFloat().reduced (2.0f, 5.0f), 7.0f, 0.8f);
    drawGlassControl (g, abPill, 10.0f, false, false, false);
    drawGlassControl (g, undoPill, 10.0f, false, false, false);

    g.setColour (Palette::panelBorder.withAlpha (0.18f));
    g.fillRect (juce::Rectangle<float> (presetPill.getX() + 38.0f, presetPill.getY() + 10.0f, 1.0f, presetPill.getHeight() - 20.0f));
    g.fillRect (juce::Rectangle<float> (presetPill.getRight() - 38.0f, presetPill.getY() + 10.0f, 1.0f, presetPill.getHeight() - 20.0f));
}

void HeaderBar::tick()
{
    undoButton.setEnabled (ctx.undo.canUndo());
    redoButton.setEnabled (ctx.undo.canRedo());

    const auto slot = ctx.processor.getPresetManager().getActiveSlot();
    if (slot != shownSlot)
    {
        shownSlot = slot;
        slotA.setToggleState (slot == 0, juce::dontSendNotification);
        slotB.setToggleState (slot == 1, juce::dontSendNotification);
    }

    updatePresetLabel();
}

void HeaderBar::updateModeButtons (int mode)
{
    voiceTab.setToggleState (mode == 0, juce::dontSendNotification);
    singTab.setToggleState (mode == 1, juce::dontSendNotification);
}

void HeaderBar::updatePresetLabel()
{
    const auto name = ctx.processor.getPresetManager().getCurrentName();
    if (name != shownPresetName)
    {
        shownPresetName = name;
        presetName->setPresetName (name);
    }
}

void HeaderBar::changeListenerCallback (juce::ChangeBroadcaster*)
{
    updatePresetLabel();
}

void HeaderBar::mouseDown (const juce::MouseEvent& e)
{
    if (e.mods.isPopupMenu() && (e.eventComponent == &slotA || e.eventComponent == &slotB))
        showABMenu();
}

void HeaderBar::showABMenu()
{
    auto& pm = ctx.processor.getPresetManager();
    const bool aActive = pm.getActiveSlot() == 0;
    juce::PopupMenu m;
    m.addSectionHeader ("A/B Compare");
    m.addItem (1, aActive ? "Copy A to B" : "Copy B to A");
    m.addItem (2, aActive ? "Switch to B" : "Switch to A");
    m.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&slotA), [this, safeThis = juce::Component::SafePointer<juce::Component> (this)] (int r)
    {
        if (safeThis == nullptr)
            return;

        auto& presets = ctx.processor.getPresetManager();
        if (r == 1)
        {
            presets.copyActiveToOtherSlot();
            ctx.toast ("Copied the active slot to the other slot");
        }
        else if (r == 2)
        {
            presets.switchToSlot (1 - presets.getActiveSlot());
        }
    });
}

//==============================================================================
void HeaderBar::showPresetMenu()
{
    auto& pm = ctx.processor.getPresetManager();
    const auto& presets = pm.getPresets();
    const int current = pm.getCurrentIndex();

    juce::StringArray categories;
    for (auto& e : presets)
        categories.addIfNotAlreadyThere (e.category);

    juce::PopupMenu menu;
    menu.addSectionHeader ("Presets");
    for (auto& cat : categories)
    {
        juce::PopupMenu sub;
        bool containsCurrent = false;
        for (int i = 0; i < (int) presets.size(); ++i)
        {
            if (presets[(size_t) i].category == cat)
            {
                sub.addItem (i + 1, presets[(size_t) i].name, true, i == current);
                containsCurrent = containsCurrent || i == current;
            }
        }
        menu.addSubMenu (cat, sub, true, nullptr, containsCurrent);
    }

    const bool isUser = juce::isPositiveAndBelow (current, (int) presets.size()) && ! presets[(size_t) current].isFactory;
    menu.addSeparator();
    menu.addItem (10001, "Save Preset...");
    menu.addItem (10002, "Import Preset...");
    menu.addItem (10003, "Delete Preset", isUser);
    menu.addItem (10004, "Open Presets Folder");

    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (presetName.get()).withMinimumWidth (220),
                        [this, safeThis = juce::Component::SafePointer<juce::Component> (this)] (int result)
    {
        if (safeThis == nullptr)
            return;

        auto& presetManager = ctx.processor.getPresetManager();
        if (result > 0 && result <= (int) presetManager.getPresets().size())
        {
            presetManager.loadPreset (result - 1);
            updatePresetLabel();
        }
        else if (result == 10001) savePresetDialog();
        else if (result == 10002) importPreset();
        else if (result == 10003) confirmDeletePreset();
        else if (result == 10004)
        {
            auto dir = PresetManager::getUserPresetDirectory();
            dir.createDirectory();
            dir.startAsProcess();
        }
    });
}

void HeaderBar::savePresetDialog()
{
    auto& pm = ctx.processor.getPresetManager();
    const auto& presets = pm.getPresets();
    const int current = pm.getCurrentIndex();
    const bool isUser = juce::isPositiveAndBelow (current, (int) presets.size()) && ! presets[(size_t) current].isFactory;

    dialog = std::make_unique<juce::AlertWindow> ("Save Preset", "Enter a name for your preset:", juce::MessageBoxIconType::NoIcon, getDialogParent());
    dialog->addTextEditor ("name", isUser ? pm.getCurrentName() : juce::String ("My Preset"), "Name");
    dialog->addButton ("Save", 1, juce::KeyPress (juce::KeyPress::returnKey));
    dialog->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));

    if (auto* te = dialog->getTextEditor ("name"))
    {
        te->setFont (font (16.0f));
        te->selectAll();
    }

    dialog->enterModalState (true, juce::ModalCallbackFunction::create ([this, safeThis = juce::Component::SafePointer<juce::Component> (this)] (int result)
    {
        if (safeThis == nullptr)
            return;

        if (dialog == nullptr)
            return;

        const auto name = dialog->getTextEditorContents ("name").trim();
        dialog.reset();

        if (result != 1)
            return;

        juce::String error;
        if (ctx.processor.getPresetManager().saveUserPreset (name, error))
            ctx.toast ("Saved preset \"" + name + "\"", ToastKind::Success);
        else
            ctx.toast (error.isNotEmpty() ? error : juce::String ("Could not save the preset"), ToastKind::Error);
        updatePresetLabel();
    }), false);
}

void HeaderBar::confirmDeletePreset()
{
    auto& pm = ctx.processor.getPresetManager();
    const int index = pm.getCurrentIndex();
    if (! juce::isPositiveAndBelow (index, (int) pm.getPresets().size()) || pm.getPresets()[(size_t) index].isFactory)
    {
        ctx.toast ("Factory presets cannot be deleted", ToastKind::Error);
        return;
    }

    const auto name = pm.getPresets()[(size_t) index].name;
    auto options = juce::MessageBoxOptions()
                       .withIconType (juce::MessageBoxIconType::WarningIcon)
                       .withTitle ("Delete Preset")
                       .withMessage ("Delete the user preset \"" + name + "\"? This cannot be undone.")
                       .withButton ("Delete")
                       .withButton ("Cancel")
                       .withAssociatedComponent (getDialogParent());

    juce::AlertWindow::showAsync (options, [this, index, name, safeThis = juce::Component::SafePointer<juce::Component> (this)] (int result)
    {
        if (safeThis == nullptr)
            return;

        if (result != 1)
            return;
        if (ctx.processor.getPresetManager().deleteUserPreset (index))
            ctx.toast ("Deleted preset \"" + name + "\"");
        else
            ctx.toast ("Could not delete the preset", ToastKind::Error);
        updatePresetLabel();
    });
}

void HeaderBar::importPreset()
{
    ctx.chooseFile ("Import Preset", juce::String ("*") + PresetManager::fileExtension, false, [this] (const juce::File& f)
    {
        juce::String error;
        if (ctx.processor.getPresetManager().importPresetFile (f, error))
            ctx.toast ("Imported " + f.getFileName(), ToastKind::Success);
        else
            ctx.toast (error.isNotEmpty() ? error : juce::String ("Could not import the preset"), ToastKind::Error);
    });
}

void HeaderBar::importVoicePack()
{
    // Packs (.vxpack), a folder or .zip of many packs (e.g. Google Drive's download of the trained voices),
    // the voice assignment and the AI content encoder (contentvec-768.onnx) - one or several at once.
    ctx.chooseFiles ("Import Voices (.vxpack, a .zip or folder of voices, contentvec-768.onnx)",
                     "*.zip;*.vxpack;*.onnx;voice.json;voice_assignment*.json", true, [this] (const juce::Array<juce::File>& chosen)
    {
        VoiceLibrary::ImportResult total;
        for (auto f : chosen)
        {
            if (f.getFileName() == "voice.json")
                f = f.getParentDirectory();
            const auto r = ctx.processor.getVoiceLibrary().importVoices (f);
            total.installed.addArray (r.installed);
            total.encoderInstalled = total.encoderInstalled || r.encoderInstalled;
            total.assignedPacks += r.assignedPacks;
            total.errors.addArray (r.errors);
        }
        ctx.toast (total.summary(), total.errors.isEmpty() ? ToastKind::Success
                                                           : (total.succeeded() ? ToastKind::Info : ToastKind::Error));
    });
}

void HeaderBar::showAbout()
{
    const auto& voices = ctx.processor.getVoiceLibrary().getVoices();
    juce::String msg;
    msg << "VoxaVersa " << VOXAVERSA_VERSION_STRING << "\n"
        << "AI vocal transformation and text-to-sing.\n\n"
        << (int) voices.size() << " voices installed\n"
        << "Engine: " << ctx.processor.getNeuralStatusMessage() << "\n"
        << "Latency: " << juce::String (1000.0 * ctx.processor.getCurrentLatencySamples() / juce::jmax (1.0, ctx.processor.getCurrentSampleRate()), 1) << " ms\n\n"
        << "Voice packs: " << VoiceLibrary::getUserVoicesDirectory().getFullPathName() << "\n"
        << "Interface font: Barlow (SIL Open Font License)";

    juce::AlertWindow::showAsync (juce::MessageBoxOptions()
                                      .withIconType (juce::MessageBoxIconType::InfoIcon)
                                      .withTitle ("About VoxaVersa")
                                      .withMessage (msg)
                                      .withButton ("OK")
                                      .withAssociatedComponent (getDialogParent()),
                                  nullptr);
}

//==============================================================================
void HeaderBar::showSettingsMenu()
{
    enum
    {
        sizeBase = 100, qualityBase = 200, sourceBase = 300,
        aiToggle = 400, importVoices = 401, openVoices = 402, rescanVoices = 403, deviceAuto = 404, deviceCpu = 405,
        importPresetId = 500, deletePresetId = 501, openPresets = 502, about = 600
    };

    const auto& widths = uiWidths;
    const int currentWidth = getUiWidth ? getUiWidth() : 1200;

    juce::PopupMenu sizes;
    for (int i = 0; i < (int) std::size (uiWidths); ++i)
    {
        const int w = widths[i];
        sizes.addItem (sizeBase + i, juce::String (juce::roundToInt (100.0 * w / 1200.0)) + "%  (" + juce::String (w) + " x "
                                         + juce::String (w * 2 / 3) + ")", true, std::abs (currentWidth - w) < 8);
    }

    juce::PopupMenu quality;
    const int q = ctx.getChoice (ParamID::quality);
    for (int i = 0; i < Choices::quality().size(); ++i)
        quality.addItem (qualityBase + i, Choices::quality()[i], true, i == q);

    juce::PopupMenu source;
    const int s = ctx.getChoice (ParamID::sourceVoice);
    for (int i = 0; i < Choices::sourceVoice().size(); ++i)
        source.addItem (sourceBase + i, Choices::sourceVoice()[i], true, i == s);

    juce::PopupMenu menu;
    menu.addSectionHeader ("Settings");
    menu.addSubMenu ("UI Size", sizes);
    menu.addSubMenu ("Quality: " + Choices::quality()[q], quality);
    menu.addSubMenu ("Source Voice: " + Choices::sourceVoice()[s], source);
    menu.addSeparator();
    menu.addSectionHeader ("AI Voice Engine");
    menu.addItem (aiToggle, "Use AI Voice Engine", true, ctx.getBool (ParamID::aiEngine));
    {
        const bool cpuOnly = ctx.processor.isNeuralCpuOnly();
        juce::PopupMenu device;
        device.addItem (deviceCpu, "CPU (recommended)", true, cpuOnly);
        device.addItem (deviceAuto, "Graphics card (GPU) - experimental, may add noise", true, ! cpuOnly);
        menu.addSubMenu (juce::String ("AI Device: ") + (cpuOnly ? "CPU" : "GPU (experimental)"), device);
    }
    menu.addItem (-1, "Status: " + ctx.processor.getNeuralStatusMessage(), false, false);
    menu.addItem (importVoices, "Import Voices...");
    menu.addItem (openVoices, "Open Voices Folder");
    menu.addItem (rescanVoices, "Rescan Voices");
    menu.addSeparator();
    menu.addSectionHeader ("Presets");
    menu.addItem (importPresetId, "Import Preset...");
    {
        auto& pm = ctx.processor.getPresetManager();
        const int idx = pm.getCurrentIndex();
        const bool isUser = juce::isPositiveAndBelow (idx, (int) pm.getPresets().size()) && ! pm.getPresets()[(size_t) idx].isFactory;
        menu.addItem (deletePresetId, "Delete Preset", isUser);
    }
    menu.addItem (openPresets, "Open Presets Folder");
    menu.addSeparator();
    menu.addItem (about, "About VoxaVersa...");

    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&gearButton).withMinimumWidth (240), [this, safeThis = juce::Component::SafePointer<juce::Component> (this)] (int r)
    {
        if (safeThis == nullptr)
            return;

        if (r >= sizeBase && r < sizeBase + (int) std::size (uiWidths))
        {
            if (onUiSizeChosen)
                onUiSizeChosen (uiWidths[r - sizeBase]);
        }
        else if (r >= qualityBase && r < qualityBase + 10)  ctx.setValue (ParamID::quality, (float) (r - qualityBase));
        else if (r >= sourceBase && r < sourceBase + 10)    ctx.setValue (ParamID::sourceVoice, (float) (r - sourceBase));
        else if (r == aiToggle)                             ctx.setValue (ParamID::aiEngine, ctx.getBool (ParamID::aiEngine) ? 0.0f : 1.0f);
        else if (r == importVoices)                         importVoicePack();
        else if (r == deviceAuto || r == deviceCpu)         ctx.processor.setNeuralCpuOnly (r == deviceCpu);
        else if (r == openVoices)
        {
            auto dir = VoiceLibrary::getUserVoicesDirectory();
            dir.createDirectory();
            dir.startAsProcess();
        }
        else if (r == rescanVoices)
        {
            ctx.processor.getVoiceLibrary().rescanUserVoices();
            ctx.toast ("Voices rescanned: " + juce::String ((int) ctx.processor.getVoiceLibrary().getVoices().size()) + " available");
        }
        else if (r == importPresetId)                       importPreset();
        else if (r == deletePresetId)                       confirmDeletePreset();
        else if (r == openPresets)
        {
            auto dir = PresetManager::getUserPresetDirectory();
            dir.createDirectory();
            dir.startAsProcess();
        }
        else if (r == about)                                showAbout();
    });
}

} // namespace voxa::ui
