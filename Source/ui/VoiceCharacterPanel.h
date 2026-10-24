#pragma once

#include "EditorContext.h"
#include "GlassPanel.h"
#include "Buttons.h"
#include "Knob.h"

namespace voxa::ui
{

/** Pitch (deeper / higher), fine tune, gender, age, tone, style and the voice blend morph. */
class VoiceCharacterPanel : public GlassPanel
{
public:
    explicit VoiceCharacterPanel (EditorContext&);
    ~VoiceCharacterPanel() override;

    void paint (juce::Graphics&) override;
    void paintStatic (juce::Graphics&, juce::Rectangle<float>) override;
    void resized() override;
    void tick();

private:
    static Knob::Options characterOptions (const juce::String& name, const juce::String& left, const juce::String& right,
                                           const juce::String& tooltip);

    EditorContext& ctx;
    Knob pitch, fineTune, gender, age, tone, style;

    juce::Slider blendSlider;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> blendAttachment;
    IconButton clearBlend { "Clear blend voice", Icon::Close };
    juce::String blendFrom, blendTo, shownBlendId, shownVoiceId;
    juce::Rectangle<float> blendFromArea, blendToArea;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoiceCharacterPanel)
};

} // namespace voxa::ui
