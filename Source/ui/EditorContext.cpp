#include "EditorContext.h"

namespace voxa::ui
{

EditorContext::EditorContext (VoxaVersaProcessor& p)
    : processor (p), state (p.getState()), undo (p.getUndoManager())
{
    lastFolder = juce::File::getSpecialLocation (juce::File::userMusicDirectory);
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

} // namespace voxa::ui
