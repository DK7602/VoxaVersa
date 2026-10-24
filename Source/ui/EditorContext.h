#pragma once

#include "PluginProcessor.h"
#include "Theme.h"
#include <optional>

namespace voxa::ui
{

enum class ToastKind { Info, Success, Error };

/** Shared services for the editor's panels (message thread only). */
class EditorContext
{
public:
    using RenderTarget = VoxaVersaProcessor::RenderTarget;

    explicit EditorContext (VoxaVersaProcessor& p);

    VoxaVersaProcessor& processor;
    juce::AudioProcessorValueTreeState& state;
    juce::UndoManager& undo;

    /** Shows a transient message in the editor's toast area. */
    std::function<void (const juce::String&, ToastKind)> showToast;

    void toast (const juce::String& message, ToastKind kind = ToastKind::Info) const
    {
        if (showToast)
            showToast (message, kind);
    }

    //==============================================================================
    // Parameters
    float getValue (const char* paramId) const;
    int getChoice (const char* paramId) const;
    bool getBool (const char* paramId) const { return getValue (paramId) >= 0.5f; }
    /** Sets a parameter as one complete, undoable gesture. */
    void setValue (const char* paramId, float denormalisedValue);

    //==============================================================================
    // Offline rendering (one render slot is shared by samples and sung takes)
    void startRender (RenderTarget target);
    bool isRendering (RenderTarget target) const noexcept { return pending.has_value() && *pending == target; }
    bool isAnyRenderPending() const noexcept { return pending.has_value(); }
    float getRenderProgress() const noexcept { return processor.getOfflineRenderProgress(); }
    juce::File getRenderedFile (RenderTarget target) const;
    void invalidateRender (RenderTarget target);

    /** Polls render completion; called from the editor's timer. */
    void tick();

    //==============================================================================
    /** Async file chooser (kept alive by the context). */
    void chooseFile (const juce::String& title, const juce::String& patterns, bool directoriesToo,
                     std::function<void (const juce::File&)> onChosen, bool saveMode = false);

    /** Last folder used by file choosers. */
    juce::File lastFolder;

private:
    std::optional<RenderTarget> pending;
    int pendingTicks = 0;
    bool sawRendering = false;
    juce::File previousRenderFile;
    juce::Time previousRenderTime;
    juce::File renderedSample, renderedTake;
    std::unique_ptr<juce::FileChooser> chooser;
};

} // namespace voxa::ui
