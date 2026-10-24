#pragma once

#include "VoiceProfile.h"
#include <juce_events/juce_events.h>
#include <vector>

namespace voxa
{

/** All built-in voices plus user voice packs. Message-thread only. */
class VoiceLibrary : public juce::ChangeBroadcaster
{
public:
    VoiceLibrary();

    const std::vector<VoiceProfile>& getVoices() const noexcept { return voices; }
    const VoiceProfile* findById (const juce::String& id) const;
    int indexOf (const juce::String& id) const;
    const VoiceProfile& getDefaultVoice() const;

    /** Genres / style tags present in the library, sorted, without duplicates. */
    juce::StringArray getGenres() const;
    juce::StringArray getStyles() const;

    struct Filter
    {
        enum class Gender { Any, Male, Female };
        Gender gender = Gender::Any;
        juce::String genre;        // empty = any
        juce::String style;        // empty = any
        bool favouritesOnly = false;
        juce::String searchText;   // matches name, genre, tags, tagline
    };

    /** Indices into getVoices() that match the filter, in library order. */
    std::vector<int> filter (const Filter&) const;

    // Favourites (persisted in the user settings file)
    bool isFavourite (const juce::String& id) const;
    void setFavourite (const juce::String& id, bool shouldBeFavourite);
    void toggleFavourite (const juce::String& id) { setFavourite (id, ! isFavourite (id)); }

    // User voice packs
    static juce::File getUserDataDirectory();      // .../VoxaVersa
    static juce::File getUserVoicesDirectory();    // .../VoxaVersa/Voices
    static juce::File getEncodersDirectory();      // .../VoxaVersa/Encoders
    void rescanUserVoices();
    /** Installs a voice pack folder or .zip/.vxpack archive into the user voices directory. */
    bool importVoicePack (const juce::File& folderOrArchive, juce::String& errorMessage);

    /** Parses a voice pack directory (containing voice.json). Returns false if invalid. */
    static bool loadVoicePack (const juce::File& directory, VoiceProfile& result, juce::String& errorMessage);

private:
    void loadFavourites();
    void saveFavourites() const;

    std::vector<VoiceProfile> voices;
    size_t numFactoryVoices = 0;
    juce::StringArray favourites;
};

/** The built-in voice catalogue (defined in FactoryVoices.cpp). */
std::vector<VoiceProfile> createFactoryVoices();

} // namespace voxa
