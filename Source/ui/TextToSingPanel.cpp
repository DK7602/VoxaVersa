#include "TextToSingPanel.h"
#include "VoxaLookAndFeel.h"

namespace voxa::ui
{

//==============================================================================
GenerateButton::GenerateButton() : juce::Button ("Generate")
{
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
    setTitle ("Generate singing");
    setTooltip ("Generate: the selected voice sings your lyrics");
}

void GenerateButton::setProgress (std::optional<float> p)
{
    if (p != progress)
    {
        progress = p;
        repaint();
    }
}

void GenerateButton::paintButton (juce::Graphics& g, bool hover, bool down)
{
    auto r = getLocalBounds().toFloat().reduced (5.0f);
    const float corner = 12.0f;
    const bool enabled = isEnabled();

    juce::Path outline;
    outline.addRoundedRectangle (r, corner);
    for (auto [w, a] : { std::pair { 10.0f, 0.08f }, { 6.0f, 0.16f }, { 3.0f, 0.3f } })
    {
        g.setColour (Palette::glow.withAlpha (a * (hover ? 1.4f : 1.0f) * (enabled ? 1.0f : 0.3f)));
        g.strokePath (outline, juce::PathStrokeType (w));
    }

    auto top = juce::Colour (0xff4f97ff), bottom = juce::Colour (0xff1846c6);
    if (hover) { top = top.brighter (0.12f); bottom = bottom.brighter (0.1f); }
    if (down)  { top = top.darker (0.2f); bottom = bottom.darker (0.2f); }
    g.setGradientFill (juce::ColourGradient (top, r.getX(), r.getY(), bottom, r.getX(), r.getBottom(), false));
    g.fillRoundedRectangle (r, corner);

    if (progress)
    {
        g.saveState();
        g.reduceClipRegion (outline);
        g.setColour (juce::Colours::white.withAlpha (0.22f));
        g.fillRect (r.withWidth (r.getWidth() * juce::jlimit (0.0f, 1.0f, *progress)));
        g.restoreState();
    }

    auto sheen = r.reduced (2.0f).withHeight (r.getHeight() * 0.45f);
    g.setGradientFill (juce::ColourGradient (juce::Colours::white.withAlpha (0.28f), sheen.getX(), sheen.getY(),
                                             juce::Colours::white.withAlpha (0.0f), sheen.getX(), sheen.getBottom(), false));
    g.fillRoundedRectangle (sheen, corner - 2.0f);

    g.setColour (Palette::glowBright);
    g.drawRoundedRectangle (r.reduced (0.5f), corner, 1.6f);

    if (hasKeyboardFocus (false))
    {
        g.setColour (juce::Colours::white.withAlpha (0.8f));
        g.drawRoundedRectangle (r.reduced (3.0f), corner - 2.0f, 1.0f);
    }

    auto content = r.reduced (12.0f, 8.0f);
    auto iconArea = content.removeFromLeft (content.getHeight()).withSizeKeepingCentre (34.0f, 34.0f);
    g.setColour (juce::Colours::white.withAlpha (0.9f));
    g.drawRoundedRectangle (iconArea, 9.0f, 1.6f);
    Icons::draw (g, Icon::Note, iconArea.reduced (6.0f), juce::Colours::white, 1.8f, true);
    content.removeFromLeft (10.0f);

    if (progress)
    {
        drawText (g, "Singing " + juce::String (juce::roundToInt (*progress * 100.0f)) + "%", content.removeFromTop (content.getHeight() * 0.6f),
                  font (19.0f, Weight::SemiBold), juce::Colours::white, juce::Justification::bottomLeft);
        drawText (g, "click to cancel", content, font (13.0f), juce::Colour (0xffd6e6ff), juce::Justification::topLeft);
    }
    else
    {
        drawText (g, getButtonText(), content, font (24.0f, Weight::SemiBold), juce::Colours::white.withAlpha (enabled ? 1.0f : 0.5f),
                  juce::Justification::centredLeft);
    }
}

//==============================================================================
juce::Array<juce::Range<int>> TextToSingPanel::approximateSyllables (const juce::String& text)
{
    juce::Array<juce::Range<int>> result;
    const int n = text.length();
    int start = -1;

    auto isSep = [] (juce::juce_wchar c) { return juce::CharacterFunctions::isWhitespace (c) || c == '-' || c == '_'; };

    for (int i = 0; i <= n; ++i)
    {
        const auto c = i < n ? text[i] : (juce::juce_wchar) ' ';
        if (isSep (c))
        {
            if (start >= 0)
            {
                // trim punctuation from the ends
                int s = start, e = i;
                while (s < e && ! juce::CharacterFunctions::isLetterOrDigit (text[s])) ++s;
                while (e > s && ! juce::CharacterFunctions::isLetterOrDigit (text[e - 1])) --e;
                if (e > s)
                    result.add ({ s, e });
                start = -1;
            }
        }
        else if (start < 0)
        {
            start = i;
        }
    }
    return result;
}

TextToSingPanel::TextToSingPanel (EditorContext& c)
    : GlassPanel ("Text to Sing", Icon::Document), ctx (c)
{
    lyrics.setMultiLine (true, true);
    lyrics.setReturnKeyStartsNewLine (true);
    lyrics.setScrollbarsShown (true);
    lyrics.setInputRestrictions (VoxaVersaProcessor::maxLyricsLength);
    lyrics.setFont (font (17.0f));
    lyrics.setTextToShowWhenEmpty ("Type your lyrics here...", Palette::textSoft);
    lyrics.getProperties().set ("voxaTransparent", true);
    lyrics.setColour (juce::TextEditor::backgroundColourId, juce::Colours::transparentBlack);
    lyrics.setColour (juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
    lyrics.setColour (juce::TextEditor::focusedOutlineColourId, juce::Colours::transparentBlack);
    lyrics.setIndents (4, 4);
    lyrics.setTitle ("Lyrics");
    lyrics.setDescription ("Lyrics to sing. Use hyphens to split syllables, e.g. sing-ing.");
    lyrics.setTooltip ("Type the lyrics to sing (up to " + juce::String (VoxaVersaProcessor::maxLyricsLength)
                       + " characters). Hyphens split syllables: to-geth-er");
    lyrics.setText (ctx.processor.getLyrics(), false);
    lastLyricsSeen = lyrics.getText();
    syllables = approximateSyllables (lastLyricsSeen);
    lyrics.onTextChange = [this]
    {
        lastLyricsSeen = lyrics.getText();
        ctx.processor.setLyrics (lastLyricsSeen);
        syllables = approximateSyllables (lastLyricsSeen);
        lastSyllable = -2;
        repaint (stripArea.toNearestInt().expanded (4));
    };
    addAndMakeVisible (lyrics);

    melodySource.addItemList (Choices::melodySource(), 1);
    melodySource.setTitle ("Melody Source");
    melodySource.setTooltip ("Melody source: play MIDI, auto-compose, load a MIDI file or hum the tune");
    melodyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (ctx.state, ParamID::melodySource, melodySource);
    addAndMakeVisible (melodySource);

    key.addItemList (Choices::key(), 1);
    key.setTitle ("Key");
    key.setTooltip ("Key of the sung melody (Auto = detect / choose automatically)");
    keyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (ctx.state, ParamID::key, key);
    addAndMakeVisible (key);

    if (auto* p = ctx.state.getParameter (ParamID::singTempo))
    {
        tempo = std::make_unique<NumberBox> (*p, &ctx.undo);
        addAndMakeVisible (*tempo);
    }

    syncButton.setClickingTogglesState (true);
    syncButton.setIconScale (0.8f);
    syncButton.setStrokeWidth (2.0f);
    syncButton.setTooltip ("Sync the sing tempo to the host tempo");
    syncAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (ctx.state, ParamID::singSyncHost, syncButton);
    addAndMakeVisible (syncButton);

    generateButton.onClick = [this] { generate(); };
    addAndMakeVisible (generateButton);

    menuButton.setIconScale (0.75f);
    menuButton.setTooltip ("Lyrics and export options");
    menuButton.onClick = [this] { showMenu(); };
    addAndMakeVisible (menuButton);

    // source specific
    captureButton.setClickingTogglesState (false);
    captureButton.setFontHeight (15.0f);
    captureButton.setTooltip ("Capture MIDI notes played into VoxaVersa as the melody");
    captureButton.onClick = [this] { ctx.processor.setMidiCaptureEnabled (! ctx.processor.isMidiCaptureEnabled()); };
    clearCaptureButton.setIconScale (0.5f);
    clearCaptureButton.onClick = [this] { ctx.processor.clearCapturedMidi(); };

    loadMidiButton.setFontHeight (15.0f);
    loadMidiButton.setTooltip ("Load a melody from a MIDI file");
    loadMidiButton.onClick = [this] { loadMidiFile(); };

    humButton.setFontHeight (15.0f);
    humButton.setTooltip ("Record yourself humming the melody");
    humButton.onClick = [this] { ctx.processor.setHumRecording (! ctx.processor.isHumRecording()); };

    autoScale.addItemList (Choices::scale(), 1);
    autoScale.setTitle ("Scale");
    autoScale.setTooltip ("Scale for the auto-composed melody");
    autoScaleAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (ctx.state, ParamID::scale, autoScale);

    for (auto* comp : std::initializer_list<juce::Component*> { &captureButton, &clearCaptureButton, &loadMidiButton, &humButton, &autoScale })
        addChildComponent (comp);

    // take controls
    previewButton.setIconScale (0.48f);
    previewButton.setTooltip ("Preview the sung take");
    previewButton.onClick = [this] { ctx.processor.setSingPreviewPlaying (! ctx.processor.isSingPreviewPlaying()); };

    renderDragButton.setFontHeight (14.0f);
    renderDragButton.setTooltip ("Render the take through the full voice chain, then drag the WAV into your DAW");
    renderDragButton.getFileToDrag = [this] { return ctx.getRenderedFile (EditorContext::RenderTarget::SingTake); };
    renderDragButton.onClick = [this]
    {
        using RT = EditorContext::RenderTarget;
        if (ctx.isRendering (RT::SingTake))
            ctx.processor.cancelOfflineRender();
        else if (ctx.getRenderedFile (RT::SingTake).existsAsFile())
            ctx.toast ("Drag this button into your DAW to import the take");
        else if (ctx.isAnyRenderPending())
            ctx.toast ("Another render is in progress", ToastKind::Error);
        else
            ctx.startRender (RT::SingTake);
    };

    midiButton.setFontHeight (14.0f);
    midiButton.setTooltip ("Export the melody with lyrics as a MIDI file (click), or drag it into your DAW");
    midiButton.getFileToDrag = [this] { return exportMidi (true); };
    midiButton.onClick = [this] { exportMidi (false); };

    for (auto* comp : std::initializer_list<juce::Component*> { &previewButton, &renderDragButton, &midiButton })
        addChildComponent (comp);

    updateSourceControls();
    updateTakeControls();
}

TextToSingPanel::~TextToSingPanel() = default;

void TextToSingPanel::resized()
{
    const int w = getWidth();
    menuButton.setBounds (w - 50, 8, 36, 32);

    // title row: source specific controls
    sourceInfoArea = { 318.0f, 8.0f, (float) w - 372.0f, 30.0f };
    captureButton.setBounds (206, 7, 104, 32);
    clearCaptureButton.setBounds (w - 78, 9, 26, 28);
    loadMidiButton.setBounds (206, 7, 108, 32);
    humButton.setBounds (206, 7, 84, 32);
    autoScale.setBounds (246, 8, w - 304, 30);

    lyricsBox = { 14.0f, 52.0f, (float) w - 28.0f, 82.0f };
    stripArea = { lyricsBox.getX() + 6.0f, lyricsBox.getBottom() - 26.0f, lyricsBox.getWidth() - 12.0f, 24.0f };
    lyrics.setBounds (juce::Rectangle<float> (lyricsBox.getX() + 4.0f, lyricsBox.getY() + 2.0f, lyricsBox.getWidth() - 8.0f,
                                              lyricsBox.getHeight() - 30.0f).toNearestInt());

    previewButton.setBounds (juce::Rectangle<float> (stripArea.getX(), stripArea.getY(), 24.0f, 24.0f).toNearestInt());
    renderDragButton.setBounds (juce::Rectangle<float> (stripArea.getX() + 118.0f, stripArea.getY() - 2.0f, 118.0f, 28.0f).toNearestInt());
    midiButton.setBounds (juce::Rectangle<float> (stripArea.getX() + 240.0f, stripArea.getY() - 2.0f, 70.0f, 28.0f).toNearestInt());

    melodySource.setBounds (14, 158, 118, 38);
    key.setBounds (138, 158, 80, 38);
    if (tempo != nullptr)
        tempo->setBounds (224, 158, 84, 38);
    syncButton.setBounds (270, 136, 22, 22);
    generateButton.setBounds (314, 138, w - 324, 62);
}

void TextToSingPanel::paint (juce::Graphics& g)
{
    GlassPanel::paint (g);

    fillInset (g, lyricsBox, 9.0f);
    if (lyrics.hasKeyboardFocus (true))
    {
        g.setColour (Palette::glow.withAlpha (0.8f));
        g.drawRoundedRectangle (lyricsBox.reduced (0.5f), 9.0f, 1.3f);
    }

    // strip: take info / live syllable + counter
    const auto counter = juce::String (lyrics.getTotalNumChars()) + "/" + juce::String (VoxaVersaProcessor::maxLyricsLength);
    drawText (g, counter, stripArea, font (14.0f), Palette::textSoft, juce::Justification::centredRight);

    if (previewButton.isVisible())
        drawText (g, timeText, { stripArea.getX() + 30.0f, stripArea.getY(), 86.0f, stripArea.getHeight() }, font (14.0f, Weight::Medium),
                  Palette::text, juce::Justification::centredLeft);
    else if (liveText.isNotEmpty())
        drawText (g, liveText, stripArea.withTrimmedRight (80.0f), font (14.5f, Weight::Medium), Palette::glowBright,
                  juce::Justification::centredLeft);

    // labels
    const auto lf = font (16.0f, Weight::Medium);
    drawText (g, "Melody Source", { 16.0f, 136.0f, 120.0f, 22.0f }, lf, Palette::text, juce::Justification::centredLeft);
    drawText (g, "Key", { 140.0f, 136.0f, 60.0f, 22.0f }, lf, Palette::text, juce::Justification::centredLeft);
    drawText (g, "Tempo", { 226.0f, 136.0f, 46.0f, 22.0f }, lf, Palette::text, juce::Justification::centredLeft);

    if (sourceInfo.isNotEmpty())
        drawText (g, sourceInfo, sourceInfoArea, font (14.5f), Palette::textSoft, juce::Justification::centredLeft);
}

//==============================================================================
void TextToSingPanel::updateSourceControls()
{
    const int source = ctx.getChoice (ParamID::melodySource);
    shownSource = source;

    captureButton.setVisible (source == (int) MelodySource::UseMidi);
    clearCaptureButton.setVisible (source == (int) MelodySource::UseMidi);
    autoScale.setVisible (source == (int) MelodySource::AutoMelody);
    loadMidiButton.setVisible (source == (int) MelodySource::MidiFile);
    humButton.setVisible (source == (int) MelodySource::HumMelody);

    switch (source)
    {
        case (int) MelodySource::UseMidi:   sourceInfoArea.setX (316.0f); break;
        case (int) MelodySource::MidiFile:  sourceInfoArea.setX (320.0f); break;
        case (int) MelodySource::HumMelody: sourceInfoArea.setX (296.0f); break;
        case (int) MelodySource::AutoMelody:
        default:                            sourceInfoArea.setX (206.0f); break;
    }
    sourceInfoArea.setRight ((float) getWidth() - (source == (int) MelodySource::UseMidi ? 80.0f : 54.0f));
    repaint (0, 0, getWidth(), (int) titleHeight);
}

void TextToSingPanel::updateTakeControls()
{
    using State = VoxaVersaProcessor::SingState;
    const bool ready = ctx.processor.getSingState() == State::Ready;
    previewButton.setVisible (ready);
    renderDragButton.setVisible (ready);
    midiButton.setVisible (ready);
}

void TextToSingPanel::tick (float)
{
    using State = VoxaVersaProcessor::SingState;
    auto& p = ctx.processor;

    // keep the editor in sync with the state (preset load, session restore, undo)
    if (! lyrics.hasKeyboardFocus (true))
    {
        const auto stored = p.getLyrics();
        if (stored != lastLyricsSeen)
        {
            lastLyricsSeen = stored;
            lyrics.setText (stored, false);
            syllables = approximateSyllables (stored);
            lastSyllable = -2;
            repaint (stripArea.toNearestInt().expanded (4));
        }
    }

    const int source = ctx.getChoice (ParamID::melodySource);
    if (source != shownSource)
        updateSourceControls();

    const int mode = ctx.getChoice (ParamID::mode);
    if (mode != shownMode)
    {
        shownMode = mode;
        setEmphasis (mode == (int) Mode::TextToSing);
    }

    // source info text (strings are only rebuilt when something changed)
    {
        juce::int64 signature = source * 1000003;
        switch (source)
        {
            case (int) MelodySource::UseMidi:
            {
                const bool capturing = p.isMidiCaptureEnabled();
                const int notes = p.getNumCapturedNotes();
                signature += notes * 2 + (capturing ? 1 : 0);
                if (signature != infoSignature)
                {
                    captureButton.setToggleState (capturing, juce::dontSendNotification);
                    captureButton.setButtonText (capturing ? "Capturing" : "Capture");
                    clearCaptureButton.setEnabled (notes > 0);
                    sourceInfo = juce::String (notes) + (notes == 1 ? " note" : " notes");
                }
                break;
            }
            case (int) MelodySource::MidiFile:
            {
                const auto name = p.getMelodyFileName();
                signature += name.hashCode64();
                if (signature != infoSignature)
                    sourceInfo = name.isNotEmpty() ? name : juce::String ("no file loaded");
                break;
            }
            case (int) MelodySource::HumMelody:
            {
                const bool rec = p.isHumRecording();
                const double secs = p.getHumRecordedSeconds();
                signature += (juce::int64) (secs * 10.0) * 2 + (rec ? 1 : 0);
                if (signature != infoSignature)
                {
                    humButton.setToggleState (rec, juce::dontSendNotification);
                    humButton.setIcon (rec ? Icon::Stop : Icon::Mic);
                    humButton.setButtonText (rec ? "Stop" : "Hum");
                    sourceInfo = rec ? "recording " + juce::String (secs, 1) + " s"
                                     : (secs > 0.0 ? juce::String (secs, 1) + " s recorded" : juce::String ("hum the tune"));
                }
                break;
            }
            case (int) MelodySource::AutoMelody:
            default:
                if (signature != infoSignature)
                    sourceInfo = {};
                break;
        }

        if (signature != infoSignature)
        {
            infoSignature = signature;
            repaint (sourceInfoArea.toNearestInt().expanded (2));
        }
    }

    // tempo sync
    if (tempo != nullptr)
    {
        const double bpm = p.hostBpm.load();
        const bool syncOn = ctx.getBool (ParamID::singSyncHost);
        const bool synced = syncOn && bpm > 0.0;
        tempo->setOverride (synced ? std::optional<double> (bpm) : std::nullopt);

        const int syncSignature = (synced ? 2 : 0) + (syncOn ? 1 : 0) + juce::roundToInt (bpm * 10.0) * 4;
        if (syncSignature != shownSyncSignature)
        {
            shownSyncSignature = syncSignature;
            syncButton.setColour (synced ? Palette::success : Palette::textSoft);
            syncButton.setTooltip (synced ? "Following the host tempo (" + juce::String (bpm, 1) + " BPM). Click to use your own tempo."
                                          : syncOn ? juce::String ("Sync to host tempo is on (no host tempo available)")
                                                   : juce::String ("Sync the sing tempo to the host tempo"));
        }
    }

    // generation state
    const auto state = p.getSingState();
    if (state != lastState)
    {
        if (state == State::Rendering)
        {
            ctx.invalidateRender (EditorContext::RenderTarget::SingTake);
            exportedMidi = juce::File();
        }
        else if (lastState == State::Rendering && state == State::Ready)
        {
            ctx.toast ("Your take is ready (" + formatTime (p.getSingTakeLengthSeconds()) + ") - press play to preview", ToastKind::Success);
        }
        else if (state == State::Failed)
        {
            const auto err = p.getSingError();
            ctx.toast ("Singing failed" + (err.isNotEmpty() ? ": " + err : juce::String()), ToastKind::Error);
        }
        lastState = state;
        updateTakeControls();
    }

    generateButton.setProgress (state == State::Rendering ? std::optional<float> (p.getSingProgress()) : std::nullopt);

    if (state == State::Ready)
    {
        const bool playing = p.isSingPreviewPlaying();
        previewButton.setIcon (playing ? Icon::Stop : Icon::Play);
        previewButton.setToggleState (playing, juce::dontSendNotification);

        const double len = p.getSingTakeLengthSeconds();
        const int timeSignature = (int) (p.getSingPlayPosition() * len) * 100000 + (int) len;
        if (timeSignature != shownTimeSignature)
        {
            shownTimeSignature = timeSignature;
            timeText = formatTime (p.getSingPlayPosition() * len) + " / " + formatTime (len);
            repaint (stripArea.toNearestInt().expanded (2));
        }

        using RT = EditorContext::RenderTarget;
        const bool rendering = ctx.isRendering (RT::SingTake);
        const bool rendered = ctx.hasRender (RT::SingTake);
        const int percent = juce::roundToInt (ctx.getRenderProgress() * 100.0f);
        const int renderSignature = rendering ? 1000 + percent : (rendered ? 1 : 0);
        if (renderSignature != shownRenderSignature)
        {
            shownRenderSignature = renderSignature;
            renderDragButton.setButtonText (rendering ? "Rendering " + juce::String (percent) + "%"
                                                      : rendered ? juce::String ("Drag to DAW") : juce::String ("Render & Drag"));
            renderDragButton.setIcon (rendering ? std::optional<Icon> (Icon::Close) : std::optional<Icon> (rendered ? Icon::Drag : Icon::Export));
            renderDragButton.setToggleState (rendered, juce::dontSendNotification);
            renderDragButton.setProgress (rendering ? ctx.getRenderProgress() : -1.0f);
        }
    }

    // live MIDI singing: show (and highlight) the next syllable
    const bool live = mode == (int) Mode::TextToSing && source == (int) MelodySource::UseMidi && state != State::Ready
                      && ! syllables.isEmpty();
    const int idx = live ? juce::jlimit (0, syllables.size() - 1, p.getLiveSyllableIndex()) : -1;
    if (idx != lastSyllable)
    {
        lastSyllable = idx;
        if (idx >= 0)
        {
            const auto range = syllables[idx];
            liveText = juce::String (juce::CharPointer_UTF8 ("\xe2\x99\xaa ")) + "Next: \"" + lastLyricsSeen.substring (range.getStart(), range.getEnd())
                       + "\"  (" + juce::String (idx + 1) + "/" + juce::String (syllables.size()) + ")";
            if (! lyrics.hasKeyboardFocus (true))
                lyrics.setHighlightedRegion (range);
        }
        else
        {
            liveText = {};
        }
        repaint (stripArea.toNearestInt().expanded (2));
    }
}

//==============================================================================
void TextToSingPanel::generate()
{
    using State = VoxaVersaProcessor::SingState;
    auto& p = ctx.processor;

    if (p.getSingState() == State::Rendering)
    {
        p.cancelSinging();
        ctx.toast ("Generation cancelled");
        return;
    }

    if (lyrics.getText().trim().isEmpty())
    {
        ctx.toast ("Type some lyrics first", ToastKind::Error);
        lyrics.grabKeyboardFocus();
        return;
    }

    const int source = ctx.getChoice (ParamID::melodySource);
    if (source == (int) MelodySource::MidiFile && p.getMelodyFileName().isEmpty())
    {
        ctx.toast ("Load a MIDI file for the melody first", ToastKind::Error);
        return;
    }
    if (source == (int) MelodySource::HumMelody && p.getHumRecordedSeconds() <= 0.0)
    {
        ctx.toast ("Record the hummed melody first (press Hum)", ToastKind::Error);
        return;
    }
    if (source == (int) MelodySource::UseMidi && p.getNumCapturedNotes() == 0)
        ctx.toast ("No MIDI notes captured: press Capture and play your melody, or choose Auto Melody");

    p.setLyrics (lyrics.getText());
    p.setSingPreviewPlaying (false);
    p.generateSinging();
}

void TextToSingPanel::loadMidiFile()
{
    ctx.chooseFile ("Load Melody MIDI File", "*.mid;*.midi", false, [this] (const juce::File& f)
    {
        juce::String error;
        if (ctx.processor.loadMelodyMidiFile (f, error))
            ctx.toast ("Melody loaded: " + f.getFileName(), ToastKind::Success);
        else
            ctx.toast ("Could not load " + f.getFileName() + (error.isNotEmpty() ? ": " + error : juce::String()), ToastKind::Error);
    });
}

juce::File TextToSingPanel::exportMidi (bool quiet)
{
    if (exportedMidi.existsAsFile())
    {
        if (! quiet)
            exportedMidi.revealToUser();
        return exportedMidi;
    }

    const auto f = ctx.processor.exportSingMidi();
    if (f.existsAsFile())
    {
        exportedMidi = f;
        if (! quiet)
            ctx.toast ("MIDI exported: " + f.getFileName() + " (drag the MIDI button into your DAW)", ToastKind::Success);
    }
    else if (! quiet)
    {
        ctx.toast ("Could not export the MIDI file", ToastKind::Error);
    }
    return exportedMidi;
}

void TextToSingPanel::showMenu()
{
    using State = VoxaVersaProcessor::SingState;
    const bool ready = ctx.processor.getSingState() == State::Ready;

    juce::PopupMenu m;
    m.addSectionHeader ("Text to Sing");
    m.addItem (1, "Load Lyrics from Text File...");
    m.addItem (2, "Clear Lyrics", lyrics.getTotalNumChars() > 0);
    m.addSeparator();
    m.addItem (3, "Sync Tempo to Host", true, ctx.getBool (ParamID::singSyncHost));
    m.addItem (4, "Load Melody MIDI File...");
    m.addSeparator();
    m.addItem (5, "Export MIDI", ready);
    m.addItem (6, "Render Take to WAV", ready && ! ctx.isAnyRenderPending());
    m.addItem (7, "Open Export Folder");

    m.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&menuButton), [this, safeThis = juce::Component::SafePointer<juce::Component> (this)] (int r)
    {
        if (safeThis == nullptr)
            return;

        switch (r)
        {
            case 1:
                ctx.chooseFile ("Load Lyrics", "*.txt;*.lrc;*.md", false, [this] (const juce::File& f)
                {
                    auto text = f.loadFileAsString().substring (0, VoxaVersaProcessor::maxLyricsLength);
                    lyrics.setText (text, true);
                    ctx.processor.setLyrics (text);
                });
                break;
            case 2:
                lyrics.clear();
                ctx.processor.setLyrics ({});
                break;
            case 3: ctx.setValue (ParamID::singSyncHost, ctx.getBool (ParamID::singSyncHost) ? 0.0f : 1.0f); break;
            case 4: loadMidiFile(); break;
            case 5: exportMidi (false); break;
            case 6: ctx.startRender (EditorContext::RenderTarget::SingTake); break;
            case 7:
            {
                auto dir = VoxaVersaProcessor::getExportDirectory();
                dir.createDirectory();
                dir.startAsProcess();
                break;
            }
            default: break;
        }
    });
}

} // namespace voxa::ui
