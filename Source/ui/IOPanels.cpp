#include "IOPanels.h"
#include "VoxaLookAndFeel.h"

namespace voxa::ui
{

//==============================================================================
DragFileButton::DragFileButton (const juce::String& buttonText, std::optional<Icon> buttonIcon)
    : GlassButton (buttonText, buttonIcon)
{
    setMouseCursor (juce::MouseCursor::DraggingHandCursor);
}

void DragFileButton::mouseDrag (const juce::MouseEvent& e)
{
    GlassButton::mouseDrag (e);

    if (dragging || e.getDistanceFromDragStart() < 6 || ! getFileToDrag)
        return;

    const auto file = getFileToDrag();
    if (! file.existsAsFile())
        return;

    dragging = true;
    juce::DragAndDropContainer::performExternalDragDropOfFiles ({ file.getFullPathName() }, false, this,
                                                               [safe = juce::Component::SafePointer<DragFileButton> (this)]
    {
        if (safe != nullptr)
            safe->dragging = false;
    });
}

void DragFileButton::mouseUp (const juce::MouseEvent& e)
{
    GlassButton::mouseUp (e);
    dragging = false;
}

//==============================================================================
InputPanel::InputPanel (EditorContext& c)
    : GlassPanel ("Input"),
      ctx (c),
      scope (c.processor.inputScope),
      gain (c.state, ParamID::inputGain, { Knob::Size::Large, "Input Gain", "-24", "+24", false, true, false, Palette::glow, {},
                                           "Input gain applied before the voice engine" }),
      meter (c.processor.inputMeter)
{
    for (auto* comp : std::initializer_list<juce::Component*> { &linkButton, &scope, &gain, &meter, &autoLevel, &playButton,
                                                                 &renderButton, &dragButton, &clearButton, &recordButton })
        addChildComponent (comp);

    for (auto* comp : std::initializer_list<juce::Component*> { &linkButton, &scope, &gain, &meter, &autoLevel })
        comp->setVisible (true);

    linkButton.setIconScale (0.75f);
    linkButton.setTooltip ("Input source: live input or a voice sample");
    linkButton.onClick = [this] { showSourceMenu(); };

    scope.setTitle ("Input waveform");
    scope.setPlaceholder ("Sing, or drop a vocal file here");

    autoLevel.setClickingTogglesState (true);
    autoLevel.setTooltip ("Auto Level: keeps your input at an ideal level for the voice engine");
    autoLevel.setFontHeight (18.0f);
    autoLevelAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (ctx.state, ParamID::autoLevel, autoLevel);

    playButton.setIconScale (0.5f);
    playButton.setTooltip ("Play / stop the loaded voice sample through VoxaVersa");
    playButton.onClick = [this] { ctx.processor.setSamplePlaying (! ctx.processor.isSamplePlaying()); };

    renderButton.setFontHeight (15.0f);
    renderButton.setTooltip ("Render the sample through the current voice into a WAV file");
    renderButton.onClick = [this]
    {
        if (ctx.isRendering (EditorContext::RenderTarget::Sample))
            ctx.processor.cancelOfflineRender();
        else if (ctx.isAnyRenderPending())
            ctx.toast ("Another render is in progress", ToastKind::Error);
        else
            ctx.startRender (EditorContext::RenderTarget::Sample);
    };

    dragButton.setFontHeight (15.0f);
    dragButton.setTooltip ("Drag the rendered WAV into your DAW");
    dragButton.getFileToDrag = [this] { return ctx.getRenderedFile (EditorContext::RenderTarget::Sample); };
    dragButton.onClick = [this]
    {
        auto f = ctx.getRenderedFile (EditorContext::RenderTarget::Sample);
        if (f.existsAsFile())
            f.revealToUser();
        else
            ctx.toast ("Render the sample first, then drag it into your DAW");
    };

    clearButton.setIconScale (0.5f);
    clearButton.onClick = [this]
    {
        ctx.processor.clearSample();
        ctx.invalidateRender (EditorContext::RenderTarget::Sample);
        updateSampleControls();
    };

    recordButton.setClickingTogglesState (false);
    recordButton.setIconColour (Palette::record);
    recordButton.setFontHeight (16.0f);
    recordButton.setTooltip ("Record the transformed output to a WAV file");
    recordButton.onClick = [this]
    {
        if (ctx.processor.isRecording())
            ctx.processor.stopRecording();
        else
            ctx.processor.startRecording();
    };
    recordButton.setVisible (ctx.processor.isStandaloneApp());

    updateSampleControls();
}

InputPanel::~InputPanel() = default;

bool InputPanel::isAudioFile (const juce::String& path)
{
    return juce::File (path).hasFileExtension ("wav;aif;aiff;flac;mp3;ogg;m4a;caf;wma");
}

void InputPanel::resized()
{
    linkButton.setBounds (getWidth() - 50, 8, 36, 32);
    scope.setBounds (16, 54, getWidth() - 32, 100);
    captionArea = { 16.0f, 156.0f, (float) getWidth() - 32.0f, 30.0f };

    playButton.setBounds (16, 156, 30, 30);
    clearButton.setBounds (getWidth() - 44, 156, 28, 30);
    dragButton.setBounds (clearButton.getX() - 74, 155, 70, 32);
    renderButton.setBounds (dragButton.getX() - 92, 155, 88, 32);

    gain.setBounds (70, 196, 180, 186);
    meter.setBounds (300, 190, 126, 160);
    autoLevel.setBounds (128, 392, 186, 46);
    recordButton.setBounds (getWidth() - 116, 395, 100, 40);
}

void InputPanel::paint (juce::Graphics& g)
{
    GlassPanel::paint (g);

    const bool sample = ctx.processor.hasSample();
    auto area = captionArea;
    juce::String text = caption;

    if (dropHover)
        text = "Drop to load as voice sample";

    if (sample)
    {
        area = area.withLeft ((float) playButton.getRight() + 8.0f).withRight ((float) renderButton.getX() - 6.0f);
        drawText (g, text, area, font (16.0f, Weight::Medium), Palette::text, juce::Justification::centredLeft);
    }
    else
    {
        drawText (g, text, area, font (17.0f), Palette::text);
    }
}

void InputPanel::updateSampleControls()
{
    const bool sample = ctx.processor.hasSample();
    hadSample = sample;
    shownSampleName = ctx.processor.getSampleName();

    playButton.setVisible (sample);
    renderButton.setVisible (sample);
    dragButton.setVisible (sample);
    clearButton.setVisible (sample);

    caption = sample ? shownSampleName : juce::String ("Your Vocal Track");
    repaint (captionArea.toNearestInt().expanded (2));
}

void InputPanel::tick (float dt)
{
    scope.tick();
    meter.tick (dt);

    const bool sample = ctx.processor.hasSample();
    if (sample != hadSample || (sample && ctx.processor.getSampleName() != shownSampleName))
        updateSampleControls();

    if (sample)
    {
        const bool playing = ctx.processor.isSamplePlaying();
        playButton.setIcon (playing ? Icon::Stop : Icon::Play);
        playButton.setToggleState (playing, juce::dontSendNotification);
        scope.setPlayPosition (playing ? ctx.processor.getSamplePlayPosition() : -1.0);

        const auto len = ctx.processor.getSampleLengthSeconds();
        auto newCaption = shownSampleName;
        if (len > 0.0)
            newCaption << "  " << formatTime (playing ? ctx.processor.getSamplePlayPosition() * len : 0.0) << " / " << formatTime (len);
        if (newCaption != caption)
        {
            caption = newCaption;
            repaint (captionArea.toNearestInt().expanded (2));
        }

        const bool rendering = ctx.isRendering (EditorContext::RenderTarget::Sample);
        renderButton.setButtonText (rendering ? "Cancel" : "Render");
        renderButton.setProgress (rendering ? ctx.getRenderProgress() : -1.0f);
        renderButton.setToggleState (rendering, juce::dontSendNotification);

        const bool rendered = ctx.getRenderedFile (EditorContext::RenderTarget::Sample).existsAsFile();
        dragButton.setEnabled (rendered);
        dragButton.setToggleState (rendered, juce::dontSendNotification);
    }
    else
    {
        scope.setPlayPosition (-1.0);
    }

    if (recordButton.isVisible())
    {
        const bool rec = ctx.processor.isRecording();
        recordButton.setToggleState (rec, juce::dontSendNotification);
        recordButton.setButtonText (rec ? formatTime (ctx.processor.getRecordingSeconds()) : juce::String ("Rec"));
        recordButton.setIcon (rec ? Icon::Stop : Icon::Record);
        recordButton.setIconColour (rec ? juce::Colours::white : Palette::record);

        if (wasRecording && ! rec)
        {
            const auto f = ctx.processor.getLastRecording();
            if (f.existsAsFile())
                ctx.toast ("Recording saved: " + f.getFileName(), ToastKind::Success);
        }
        wasRecording = rec;
    }
}

void InputPanel::loadSample (const juce::File& file)
{
    juce::String error;
    ctx.invalidateRender (EditorContext::RenderTarget::Sample);
    if (ctx.processor.loadSample (file, error))
    {
        ctx.toast ("Loaded voice sample " + file.getFileName(), ToastKind::Success);
        ctx.lastFolder = file.getParentDirectory();
    }
    else
    {
        ctx.toast ("Could not load " + file.getFileName() + (error.isNotEmpty() ? ": " + error : juce::String()), ToastKind::Error);
    }
    updateSampleControls();
}

bool InputPanel::isInterestedInFileDrag (const juce::StringArray& files)
{
    for (auto& f : files)
        if (isAudioFile (f))
            return true;
    return false;
}

void InputPanel::fileDragEnter (const juce::StringArray&, int, int)
{
    dropHover = true;
    scope.setHighlighted (true);
    repaint();
}

void InputPanel::fileDragExit (const juce::StringArray&)
{
    dropHover = false;
    scope.setHighlighted (false);
    repaint();
}

void InputPanel::filesDropped (const juce::StringArray& files, int, int)
{
    dropHover = false;
    scope.setHighlighted (false);
    repaint();

    for (auto& f : files)
    {
        if (isAudioFile (f))
        {
            loadSample (juce::File (f));
            return;
        }
    }
}

void InputPanel::showSourceMenu()
{
    const bool sample = ctx.processor.hasSample();
    juce::PopupMenu m;
    m.addSectionHeader ("Input Source");
    m.addItem (1, "Live Input", true, ! sample);
    m.addItem (2, sample ? "Voice Sample: " + ctx.processor.getSampleName() : juce::String ("Voice Sample..."), true, sample);
    m.addSeparator();
    m.addItem (3, "Load Voice Sample...");
    m.addItem (4, "Remove Sample", sample);
    if (ctx.processor.isStandaloneApp())
    {
        m.addSeparator();
        m.addItem (5, ctx.processor.isRecording() ? "Stop Recording" : "Start Recording");
        m.addItem (6, "Reveal Last Recording", ctx.processor.getLastRecording().existsAsFile());
    }

    m.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&linkButton), [this] (int r)
    {
        auto chooseSample = [this]
        {
            ctx.chooseFile ("Load Voice Sample", "*.wav;*.aif;*.aiff;*.flac;*.mp3;*.ogg;*.m4a", false,
                            [this] (const juce::File& f) { loadSample (f); });
        };

        switch (r)
        {
            case 1:
                if (ctx.processor.hasSample())
                {
                    ctx.processor.clearSample();
                    ctx.invalidateRender (EditorContext::RenderTarget::Sample);
                    updateSampleControls();
                }
                break;
            case 2: if (! ctx.processor.hasSample()) chooseSample(); break;
            case 3: chooseSample(); break;
            case 4:
                ctx.processor.clearSample();
                ctx.invalidateRender (EditorContext::RenderTarget::Sample);
                updateSampleControls();
                break;
            case 5:
                if (ctx.processor.isRecording()) ctx.processor.stopRecording();
                else ctx.processor.startRecording();
                break;
            case 6: ctx.processor.getLastRecording().revealToUser(); break;
            default: break;
        }
    });
}

//==============================================================================
OutputPanel::OutputPanel (EditorContext& c)
    : GlassPanel ("Output"),
      ctx (c),
      scope (c.processor.outputScope),
      gain (c.state, ParamID::outputGain, { Knob::Size::Large, "Output Gain", "-24", "+24", false, true, false, Palette::glow, {},
                                            "Output gain after the voice engine" }),
      meter (c.processor.outputMeter)
{
    for (auto* comp : std::initializer_list<juce::Component*> { &linkButton, &scope, &gain, &meter, &mixLabel, &mixSlider, &mixValue })
        addAndMakeVisible (comp);

    linkButton.setIconScale (0.75f);
    linkButton.setTooltip ("Output: export folder, renders and recordings");
    linkButton.onClick = [this] { showOutputMenu(); };

    scope.setTitle ("Output waveform");

    mixLabel.setStyle (GlassButton::Style::Primary);
    mixLabel.setFontHeight (19.0f);
    mixLabel.setCornerRadius (8.0f);
    mixLabel.setTooltip ("Dry/Wet mix. Click to reset to 100% wet.");
    mixLabel.onClick = [this] { ctx.setValue (ParamID::mix, 100.0f); };

    mixSlider.setSliderStyle (juce::Slider::LinearHorizontal);
    mixSlider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    mixSlider.setTitle ("Dry/Wet Mix");
    mixSlider.setTooltip ("Dry/Wet mix between your original and the transformed voice");
    mixAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (ctx.state, ParamID::mix, mixSlider);
    mixSlider.setDoubleClickReturnValue (true, 100.0);
    mixSlider.onValueChange = [this] { mixValue.update(); };
    mixValue.setFontHeight (18.0f);
    mixValue.update();
}

OutputPanel::~OutputPanel()
{
    mixSlider.onValueChange = nullptr;
}

void OutputPanel::resized()
{
    linkButton.setBounds (getWidth() - 50, 8, 36, 32);
    scope.setBounds (16, 54, getWidth() - 32, 100);
    gain.setBounds (70, 196, 180, 186);
    meter.setBounds (300, 190, 126, 160);

    mixLabel.setBounds (18, 384, 70, 50);
    mixSlider.setBounds (100, 390, 236, 32);
    mixValue.setBounds (344, 386, 82, 44);
}

void OutputPanel::paintStatic (juce::Graphics& g, juce::Rectangle<float> area)
{
    GlassPanel::paintStatic (g, area);
    drawText (g, "Transformed Vocal", { 16.0f, 156.0f, area.getWidth() - 32.0f, 30.0f }, font (17.0f), Palette::text);
    drawText (g, "Dry", { 96.0f, 420.0f, 60.0f, 20.0f }, font (15.0f), Palette::textSoft, juce::Justification::centredLeft);
    drawText (g, "Wet", { 280.0f, 420.0f, 60.0f, 20.0f }, font (15.0f), Palette::textSoft, juce::Justification::centredRight);
}

void OutputPanel::paint (juce::Graphics& g)
{
    GlassPanel::paint (g);
}

void OutputPanel::tick (float dt)
{
    scope.tick();
    meter.tick (dt);
}

void OutputPanel::showOutputMenu()
{
    const auto rendered = ctx.processor.getLastRenderedFile();
    const auto recording = ctx.processor.getLastRecording();

    juce::PopupMenu m;
    m.addSectionHeader ("Output");
    m.addItem (1, "Open Export Folder");
    m.addItem (2, "Reveal Last Render", rendered.existsAsFile());
    m.addItem (3, "Reveal Last Recording", recording.existsAsFile());
    m.addSeparator();
    m.addItem (4, "Reset Mix to 100% Wet");

    m.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&linkButton), [this] (int r)
    {
        if (r == 1)
        {
            auto dir = VoxaVersaProcessor::getExportDirectory();
            dir.createDirectory();
            dir.startAsProcess();
        }
        else if (r == 2) ctx.processor.getLastRenderedFile().revealToUser();
        else if (r == 3) ctx.processor.getLastRecording().revealToUser();
        else if (r == 4) ctx.setValue (ParamID::mix, 100.0f);
    });
}

} // namespace voxa::ui
