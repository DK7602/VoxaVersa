#pragma once

#include "PluginProcessor.h"

namespace voxa
{

class VoxaVersaEditor : public juce::AudioProcessorEditor
{
public:
    explicit VoxaVersaEditor (VoxaVersaProcessor&);
    ~VoxaVersaEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    VoxaVersaProcessor& processor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoxaVersaEditor)
};

} // namespace voxa
