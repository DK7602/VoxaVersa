#pragma once

#include <juce_core/juce_core.h>

namespace voxa
{

/** How a trained singer is presented when it has a card of its own in the Voice Library (it sings no built-in
    voice): a style title, genre, description and tags instead of "VocalSet Male 3". Its portrait is
    Resources/voices/singer-<pack id>.jpg when there is one. */
struct SingerCard
{
    const char* packId;    // the pack's id as in voice.json, e.g. "vocalset-male3"
    const char* title;     // "Lyric Tenor"
    const char* genre;
    const char* tagline;
    const char* styles;    // comma separated
};

/** The card for a pack id ("vocalset-male3" or "user-vocalset-male3"), or nullptr. */
const SingerCard* findSingerCard (const juce::String& packId);

/** "singer-<pack id>.jpg": the bundled portrait file name of a singer's card. */
juce::String singerPortraitFileName (const juce::String& packId);

} // namespace voxa
