#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <map>
#include <vector>

namespace voxa
{

struct FactoryPreset
{
    juce::String name;
    juce::String category;
    juce::String voiceId;
    std::map<juce::String, float> values;   // parameter id -> real (denormalised) value
};

std::vector<FactoryPreset> createFactoryPresets();

/** Factory + user presets, prev/next navigation and A/B comparison. Message-thread only. */
class PresetManager : public juce::ChangeBroadcaster
{
public:
    explicit PresetManager (juce::AudioProcessorValueTreeState&);

    struct Entry
    {
        juce::String name;
        juce::String category;
        bool isFactory = true;
        int factoryIndex = -1;
        juce::File file;
    };

    const std::vector<Entry>& getPresets() const noexcept { return entries; }
    int getCurrentIndex() const noexcept { return currentIndex; }
    juce::String getCurrentName() const;

    void loadPreset (int index);
    void loadNext();
    void loadPrevious();
    void loadInit();

    bool saveUserPreset (const juce::String& name, juce::String& errorMessage);
    bool deleteUserPreset (int index);
    bool importPresetFile (const juce::File& file, juce::String& errorMessage);
    void rescan();

    static juce::File getUserPresetDirectory();
    static constexpr const char* fileExtension = ".vxpreset";

    // A/B comparison
    int getActiveSlot() const noexcept { return activeSlot; }
    void switchToSlot (int slot);          // 0 = A, 1 = B
    void copyActiveToOtherSlot();

    /** Called after the host restores a session so the displayed name matches the state. */
    void syncWithState();

private:
    void applyFactoryPreset (const FactoryPreset&);
    void applyStateTree (const juce::ValueTree&);
    /** Resets the sound to defaults. Keeps power and the session's own settings: mode, input / output gain,
        the song's key and scale, MIDI pitch control and the Text to Sing tempo and melody source. */
    void resetSoundParametersToDefault();

    juce::AudioProcessorValueTreeState& apvts;
    std::vector<FactoryPreset> factoryPresets;
    std::vector<Entry> entries;
    int currentIndex = 0;

    int activeSlot = 0;
    juce::ValueTree slots[2];
};

} // namespace voxa
