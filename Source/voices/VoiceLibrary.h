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
        int assignedPacks = 0;           // packs described by an imported voice assignment
        juce::StringArray errors;

        bool succeeded() const noexcept   { return installed.size() > 0 || encoderInstalled || assignedPacks > 0; }
        juce::String summary() const;
    };

    /** A voice assignment (voice_assignment.json, written by the training queue next to its packs) says, per
        pack id, which built-in voices the pack powers and how it is presented, overriding its voice.json:

            { "packs": { "<pack id>": { "factoryVoices": ["female-pop", ...], "listed": false,
                                        "name": "...", "description": "...", "license": "..." } } }

        Imported assignments are merged (by pack id) into <Voices>/voice_assignment.json, which every rescan
        applies, so a pack never has to be rewritten when more singers finish and the assignment changes. */
    static constexpr const char* assignmentFileName = "voice_assignment.json";

    /** Imports one voice pack, or a folder / .zip holding any number of .vxpack files and optionally the
        content encoder and a voice assignment: e.g. Google Drive's download of the training "voices" folder
        (+ contentvec-768.onnx). Only those are taken from a big archive; everything else in it is ignored. */
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
