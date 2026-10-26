#include "EditorContext.h"

namespace voxa::ui
{

EditorContext::EditorContext (VoxaVersaProcessor& p)
    : processor (p), state (p.getState()), undo (p.getUndoManager())
{
    lastFolder = juce::File::getSpecialLocation (juce::File::userMusicDirectory);
    lastNeuralStatus = p.getNeuralStatus();
    lastMissedChunks = p.getNeuralMissedChunks();
}

float EditorContext::getValue (const char* paramId) const
{
    if (auto* raw = state.getRawParameterValue (paramId))
        return raw->load();
    return 0.0f;
}

int EditorContext::getChoice (const char* paramId) const
{
    return juce::roundToInt (getValue (paramId));
}

void EditorContext::setValue (const char* paramId, float value)
{
    if (auto* p = state.getParameter (paramId))
    {
        undo.beginNewTransaction();
        p->beginChangeGesture();
        p->setValueNotifyingHost (p->convertTo0to1 (value));
        p->endChangeGesture();
    }
}

//==============================================================================
void EditorContext::startRender (RenderTarget target)
{
    if (pending.has_value())
        return;

    previousRenderFile = processor.getLastRenderedFile();
    previousRenderTime = previousRenderFile.existsAsFile() ? previousRenderFile.getLastModificationTime() : juce::Time();
    pending = target;
    pendingTicks = 0;
    sawRendering = false;
    invalidateRender (target);
    processor.startOfflineRender (target);
}

juce::File EditorContext::getRenderedFile (RenderTarget target) const
{
    const auto& f = target == RenderTarget::Sample ? renderedSample : renderedTake;
    return f.existsAsFile() ? f : juce::File();
}

void EditorContext::invalidateRender (RenderTarget target)
{
    (target == RenderTarget::Sample ? renderedSample : renderedTake) = juce::File();
}

void EditorContext::tick()
{
    // surface AI engine state changes (e.g. a model failed to load)
    const auto status = processor.getNeuralStatus();
    if (status != lastNeuralStatus)
    {
        using NS = VoxaVersaProcessor::NeuralStatus;
        if (status == NS::Error)
            toast ("AI voice engine: " + processor.getNeuralStatusMessage(), ToastKind::Error);
        else if (status == NS::Active)
            toast ("AI voice engine active: " + processor.getNeuralStatusMessage(), ToastKind::Success);
        lastNeuralStatus = status;
    }

    // the AI voice can't keep up live on this computer: say what happens and what to do (at most once a minute)
    const auto missed = processor.getNeuralMissedChunks();
    if (missed != lastMissedChunks)
    {
        const auto now = juce::Time::getMillisecondCounterHiRes();
        if (missed > lastMissedChunks && status == VoxaVersaProcessor::NeuralStatus::Active && now - lastOverloadToastMs > 60000.0)
        {
            toast ("This computer can't run the AI voice fully live, so short gaps play the DSP voice. "
                   "Export or Render in Place gives the complete AI voice.", ToastKind::Info);
            lastOverloadToastMs = now;
        }
        lastMissedChunks = missed;
    }

    if (! pending.has_value())
        return;

    ++pendingTicks;
    if (processor.isOfflineRendering())
    {
        sawRendering = true;
        return;
    }

    // Wait briefly in case the render thread has not started yet.
    if (! sawRendering && pendingTicks < 12)
        return;

    const auto target = *pending;
    pending.reset();

    const auto file = processor.getLastRenderedFile();
    const auto error = processor.getLastRenderError();
    const bool isNew = file.existsAsFile()
                       && (file != previousRenderFile || file.getLastModificationTime() > previousRenderTime);

    if (error.isNotEmpty() || ! isNew)
    {
        toast ("Render failed" + (error.isNotEmpty() ? ": " + error : juce::String (" (no audio was produced)")), ToastKind::Error);
        return;
    }

    (target == RenderTarget::Sample ? renderedSample : renderedTake) = file;
    toast ("Rendered " + file.getFileName() + " \xe2\x80\x93 drag it into your DAW", ToastKind::Success);
}

//==============================================================================
void EditorContext::chooseFile (const juce::String& title, const juce::String& patterns, bool directoriesToo,
                                std::function<void (const juce::File&)> onChosen, bool saveMode)
{
    chooser = std::make_unique<juce::FileChooser> (title, lastFolder, patterns, true);

    int flags = saveMode ? juce::FileBrowserComponent::saveMode : juce::FileBrowserComponent::openMode;
    flags |= juce::FileBrowserComponent::canSelectFiles;
    if (directoriesToo)
        flags |= juce::FileBrowserComponent::canSelectDirectories;

    chooser->launchAsync (flags, [this, cb = std::move (onChosen)] (const juce::FileChooser& fc)
    {
        const auto result = fc.getResult();
        if (result != juce::File())
        {
            lastFolder = result.getParentDirectory();
            if (cb)
                cb (result);
        }
    });
}

void EditorContext::chooseFiles (const juce::String& title, const juce::String& patterns, bool directoriesToo,
                                 std::function<void (const juce::Array<juce::File>&)> onChosen)
{
    chooser = std::make_unique<juce::FileChooser> (title, lastFolder, patterns, true);

    int flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles
              | juce::FileBrowserComponent::canSelectMultipleItems;
    if (directoriesToo)
        flags |= juce::FileBrowserComponent::canSelectDirectories;

    chooser->launchAsync (flags, [this, cb = std::move (onChosen)] (const juce::FileChooser& fc)
    {
        const auto results = fc.getResults();
        if (! results.isEmpty())
        {
            lastFolder = results.getFirst().getParentDirectory();
            if (cb)
                cb (results);
        }
    });
}

} // namespace voxa::ui
