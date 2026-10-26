#include "PresetManager.h"
#include "../Parameters.h"
#include "../voices/VoiceLibrary.h"

namespace voxa
{

PresetManager::PresetManager (juce::AudioProcessorValueTreeState& state)
    : apvts (state), factoryPresets (createFactoryPresets())
{
    rescan();
}

juce::File PresetManager::getUserPresetDirectory()
{
    return VoiceLibrary::getUserDataDirectory().getChildFile ("Presets");
}

void PresetManager::rescan()
{
    const auto currentName = getCurrentName();
    entries.clear();

    for (size_t i = 0; i < factoryPresets.size(); ++i)
        entries.push_back ({ factoryPresets[i].name, factoryPresets[i].category, true, (int) i, {} });

    auto dir = getUserPresetDirectory();
    if (dir.isDirectory())
    {
        juce::Array<juce::File> files;
        for (const auto& e : juce::RangedDirectoryIterator (dir, false, juce::String ("*") + fileExtension))
            files.add (e.getFile());
        files.sort();
        for (auto& f : files)
            entries.push_back ({ f.getFileNameWithoutExtension(), "User", false, -1, f });
    }

    currentIndex = 0;
    for (size_t i = 0; i < entries.size(); ++i)
        if (entries[i].name == currentName)
            currentIndex = (int) i;

    sendChangeMessage();
}

juce::String PresetManager::getCurrentName() const
{
    return apvts.state.getProperty (StateID::presetName, "Init Preset").toString();
}

void PresetManager::resetSoundParametersToDefault()
{
    static const juce::StringArray sessionSettings { ParamID::power, ParamID::mode, ParamID::inputGain, ParamID::outputGain,
                                                     ParamID::key, ParamID::scale, ParamID::midiControl, ParamID::singTempo,
                                                     ParamID::singSyncHost, ParamID::melodySource };
    for (auto* p : apvts.processor.getParameters())
        if (auto* rp = dynamic_cast<juce::RangedAudioParameter*> (p))
            if (! sessionSettings.contains (rp->paramID))
                rp->setValueNotifyingHost (rp->getDefaultValue());
}

void PresetManager::applyFactoryPreset (const FactoryPreset& preset)
{
    resetSoundParametersToDefault();

    for (const auto& [id, value] : preset.values)
        if (auto* p = apvts.getParameter (id))
            p->setValueNotifyingHost (p->convertTo0to1 (value));

    apvts.state.setProperty (StateID::voiceId, preset.voiceId, apvts.undoManager);
    apvts.state.setProperty (StateID::blendVoiceId, "", apvts.undoManager);
    apvts.state.setProperty (StateID::presetName, preset.name, apvts.undoManager);
}

void PresetManager::applyStateTree (const juce::ValueTree& tree)
{
    if (! tree.isValid())
        return;

    // Keep session-specific content that presets should never overwrite.
    const auto lyrics = apvts.state.getProperty (StateID::lyrics);
    const auto melodyFile = apvts.state.getProperty (StateID::melodyFile);
    const auto samplePath = apvts.state.getProperty (StateID::samplePath);

    // Parameters: set through the parameter objects so the host is notified.
    for (int i = 0; i < tree.getNumChildren(); ++i)
    {
        const auto child = tree.getChild (i);
        if (! child.hasType ("PARAM"))
            continue;
        const auto id = child.getProperty ("id").toString();
        if (id == ParamID::power)
            continue;
        if (auto* p = apvts.getParameter (id))
            p->setValueNotifyingHost (p->convertTo0to1 ((float) child.getProperty ("value")));
    }

    for (auto* key : { StateID::voiceId, StateID::blendVoiceId, StateID::presetName })
        if (tree.hasProperty (key))
            apvts.state.setProperty (key, tree.getProperty (key), apvts.undoManager);

    apvts.state.setProperty (StateID::lyrics, lyrics, nullptr);
    apvts.state.setProperty (StateID::melodyFile, melodyFile, nullptr);
    apvts.state.setProperty (StateID::samplePath, samplePath, nullptr);
}

void PresetManager::loadPreset (int index)
{
    if (! juce::isPositiveAndBelow (index, (int) entries.size()))
        return;

    if (apvts.undoManager != nullptr)
        apvts.undoManager->beginNewTransaction ("Load preset");

    const auto& e = entries[(size_t) index];
    if (e.isFactory)
    {
        applyFactoryPreset (factoryPresets[(size_t) e.factoryIndex]);
    }
    else if (auto xml = juce::parseXML (e.file))
    {
        auto tree = juce::ValueTree::fromXml (*xml);
        tree.setProperty (StateID::presetName, e.name, nullptr);
        applyStateTree (tree);
    }

    currentIndex = index;
    sendChangeMessage();
}

void PresetManager::loadNext()
{
    if (! entries.empty())
        loadPreset ((currentIndex + 1) % (int) entries.size());
}

void PresetManager::loadPrevious()
{
    if (! entries.empty())
        loadPreset ((currentIndex - 1 + (int) entries.size()) % (int) entries.size());
}

void PresetManager::loadInit()
{
    loadPreset (0);
}

bool PresetManager::saveUserPreset (const juce::String& name, juce::String& error)
{
    const auto cleanName = juce::File::createLegalFileName (name.trim());
    if (cleanName.isEmpty())
    {
        error = "Please enter a preset name";
        return false;
    }

    auto dir = getUserPresetDirectory();
    if (! dir.createDirectory())
    {
        error = "Could not create " + dir.getFullPathName();
        return false;
    }

    auto state = apvts.copyState();
    state.setProperty (StateID::presetName, cleanName, nullptr);
    state.removeProperty (StateID::lyrics, nullptr);
    state.removeProperty (StateID::melodyFile, nullptr);
    state.removeProperty (StateID::samplePath, nullptr);

    auto file = dir.getChildFile (cleanName + fileExtension);
    if (auto xml = state.createXml(); xml == nullptr || ! xml->writeTo (file))
    {
        error = "Could not write " + file.getFullPathName();
        return false;
    }

    apvts.state.setProperty (StateID::presetName, cleanName, nullptr);
    rescan();
    return true;
}

bool PresetManager::deleteUserPreset (int index)
{
    if (! juce::isPositiveAndBelow (index, (int) entries.size()) || entries[(size_t) index].isFactory)
        return false;

    const bool ok = entries[(size_t) index].file.deleteFile();
    rescan();
    return ok;
}

bool PresetManager::importPresetFile (const juce::File& file, juce::String& error)
{
    auto xml = juce::parseXML (file);
    if (xml == nullptr)
    {
        error = "Not a valid preset file";
        return false;
    }

    auto dir = getUserPresetDirectory();
    dir.createDirectory();
    auto dest = dir.getChildFile (file.getFileNameWithoutExtension() + fileExtension);
    if (! xml->writeTo (dest))
    {
        error = "Could not copy preset";
        return false;
    }
    rescan();
    return true;
}

void PresetManager::switchToSlot (int slot)
{
    slot = juce::jlimit (0, 1, slot);
    if (slot == activeSlot)
        return;

    slots[activeSlot] = apvts.copyState();
    if (slots[slot].isValid())
        applyStateTree (slots[slot]);

    activeSlot = slot;
    sendChangeMessage();
}

void PresetManager::copyActiveToOtherSlot()
{
    slots[1 - activeSlot] = apvts.copyState();
    sendChangeMessage();
}

void PresetManager::syncWithState()
{
    rescan();
}

} // namespace voxa
