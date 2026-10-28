#include "SingerCards.h"

namespace voxa
{

namespace
{
    // The VocalSet singers (CC BY 4.0) are classically trained; their titles follow each one's measured colour
    // (register, brightness, breathiness, ring - tools/training/factory_voice_assignment.json).
    const SingerCard cards[] = {
        { "vocalset-female1", "Lyric Soprano",     "Classical", "Clear, singing classical soprano",            "Clean,Classical" },
        { "vocalset-female2", "Crystal Soprano",   "Classical", "Pure, glassy classical soprano",              "Clean,Classical,Airy" },
        { "vocalset-female3", "Silver Soprano",    "Classical", "Bright, polished concert soprano",            "Clean,Classical" },
        { "vocalset-female4", "Velvet Mezzo",      "Classical", "Dark, rounded mezzo-soprano",                 "Warm,Classical" },
        { "vocalset-female5", "Pearl Soprano",     "Classical", "Smooth, gentle classical soprano",            "Clean,Classical,Soft" },
        { "vocalset-female6", "Halo Soprano",      "Classical", "Light, floating head-voice soprano",          "Airy,Classical" },
        { "vocalset-female7", "Cathedral Soprano", "Classical", "Dark, ringing choral soprano",                "Powerful,Classical" },
        { "vocalset-female8", "Aria Diva",         "Classical", "Brilliant, ringing operatic soprano",         "Powerful,Classical,Bright" },
        { "vocalset-female9", "Stage Soprano",     "Classical", "Bright, open stage soprano",                  "Clean,Classical,Bright" },
        { "vocalset-male1",   "Opera Baritone",    "Classical", "Ringing operatic baritone",                   "Powerful,Classical" },
        { "vocalset-male2",   "Concert Baritone",  "Classical", "Dark, resonant concert baritone",             "Powerful,Classical,Warm" },
        { "vocalset-male3",   "Lyric Tenor",       "Classical", "Bright, easy lyric tenor",                    "Clean,Classical,Bright" },
        { "vocalset-male4",   "Broadway Baritone", "Classical", "Open, ringing stage baritone",                "Powerful,Classical" },
        { "vocalset-male5",   "Chamber Tenor",     "Classical", "Soft, refined chamber tenor",                 "Clean,Classical,Soft" },
        { "vocalset-male6",   "Velvet Baritone",   "Classical", "Dark, smooth baritone",                       "Warm,Classical" },
        { "vocalset-male7",   "Falsetto Tenor",    "Classical", "Tenor with a high, easy falsetto",            "Clean,Classical,Airy" },
        { "vocalset-male8",   "Basso Profundo",    "Classical", "Deep, dark classical bass",                   "Deep,Classical" },
        { "vocalset-male9",   "Countertenor",      "Classical", "Pure, high countertenor",                     "Clean,Classical,Airy" },
        { "vocalset-male10",  "Hushed Tenor",      "Classical", "Breathy, intimate tenor",                     "Breathy,Classical,Soft" },
        { "vocalset-male11",  "Clarion Tenor",     "Classical", "Bright, ringing tenor with a breathy edge",   "Bright,Classical,Breathy" },
        { "cymatics-female-blend", "Studio Diva",  "Pop",       "Warm, polished studio pop & soul voice",      "Clean,Warm" },
    };
}

const SingerCard* findSingerCard (const juce::String& packId)
{
    const auto id = packId.startsWith ("user-") ? packId.substring (5) : packId;
    for (const auto& c : cards)
        if (id == c.packId)
            return &c;
    return nullptr;
}

juce::String singerPortraitFileName (const juce::String& packId)
{
    return "singer-" + (packId.startsWith ("user-") ? packId.substring (5) : packId) + ".jpg";
}

} // namespace voxa
