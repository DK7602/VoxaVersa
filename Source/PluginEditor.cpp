// TEMPORARY SKELETON - the full UI replaces this file.
#include "PluginEditor.h"

namespace voxa
{

VoxaVersaEditor::VoxaVersaEditor (VoxaVersaProcessor& p)
    : AudioProcessorEditor (p), processor (p)
{
    setSize (1280, 853);
}

VoxaVersaEditor::~VoxaVersaEditor() = default;

void VoxaVersaEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff0b1630));
    g.setColour (juce::Colours::white);
    g.drawFittedText ("VoxaVersa", getLocalBounds(), juce::Justification::centred, 1);
}

void VoxaVersaEditor::resized() {}

} // namespace voxa
