#pragma once

#include "EditorContext.h"

namespace voxa::ui
{

/** Transient status / error message shown under the header. Click to dismiss. */
class ToastOverlay : public juce::Component
{
public:
    ToastOverlay();

    void show (const juce::String& message, ToastKind kind);
    void tick (float dt);

    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent&) override;

    /** Horizontal centre and top edge (parent coordinates) where toasts appear. */
    void setAnchor (juce::Point<int> topCentre) { anchor = topCentre; }

private:
    juce::String message;
    ToastKind kind = ToastKind::Info;
    float age = 0.0f, holdTime = 4.0f, alpha = 0.0f, drawnAlpha = -1.0f;
    juce::Point<int> anchor { 768, 98 };
};

} // namespace voxa::ui
