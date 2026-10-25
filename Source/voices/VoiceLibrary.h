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
    void rescanUserVoices()   { rescanUserVoices (getUserVoicesDirectory()); }
    /** Reloads the voice packs from `voicesDirectory` (built-in voices pick up AI models listed by packs). */
    void rescanUserVoices (const juce::File& voicesDirectory);
    /** Installs a voice pack folder or .zip/.vxpack archive into the user voices directory. */
    bool importVoicePack (const juce::File& folderOrArchive, juce::String& errorMessage);

    struct ImportResult
    {
        juce::StringArray installed;     // names of the voices installed
        bool encoderInstalled = false;   // a content encoder (e.g. contentvec-768.onnx) was copied to Encoders/
        juce::StringArray errors;

        bool succeeded() const noexcept   { return installed.size() > 0 || encoderInstalled; }
        juce::String summary() const;
    };

    /** Imports one voice pack, or a folder / .zip holding any number of .vxpack files and optionally the
        content encoder: e.g. Google Drive's download of the training "voices" folder (+ contentvec-768.onnx).
        Only the packs and the encoder are taken from a big archive; everything else in it is ignored. */
    ImportResult importVoices (const juce::File& folderOrArchive)   { return importVoices (folderOrArchive, getUserDataDirectory()); }
    ImportResult importVoices (const juce::File& folderOrArchive, const juce::File& userDataDirectory);

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
