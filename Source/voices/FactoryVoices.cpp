#include "VoiceLibrary.h"

namespace voxa
{

namespace
{
    struct GenreStyle
    {
        const char* genre;
        const char* idPart;
        juce::uint32 colourA, colourB;
        const char* maleTagline;
        const char* femaleTagline;
        juce::StringArray styles;
        // Common traits applied to both genders
        float tiltDb, warmthDb, presenceDb, airDb, nasalDb, breathiness, rasp;
        float vibRate, vibDepth, vibOnset, scoop, jitter;
        float comp, deEss, sat, revMix, revSize, dlyMix; int dlyDiv; float doubler, lowCut;
        float legato, busy;
        // Register & formant per gender
        int maleLow, maleHigh;   float maleFormant;
        int femaleLow, femaleHigh; float femaleFormant;
    };

    // Each genre produces a male and a female voice.
    const GenreStyle genreStyles[] = {
        // genre        id           colours                    male tagline                                   female tagline
        { "Pop", "pop", 0xff3a7bff, 0xffb8d8ff, "Bright, polished radio-pop tenor", "Sparkling, airy chart-topping pop",
          { "Polished", "Bright", "Smooth" },
          1.5f, 0.0f, 2.0f, 2.5f, 0.0f, 0.15f, 0.0f,   5.6f, 18.0f, 300.0f, 25.0f, 3.0f,
          0.45f, 0.35f, 0.05f, 0.14f, 0.45f, 0.06f, 1, 0.25f, 90.0f,  0.65f, 0.5f,
          48, 69, -0.6f,  57, 79, 1.3f },

        { "Rap", "rap", 0xffff8a3d, 0xffffd29a, "Punchy, confident hip-hop flow", "Sharp, rhythmic rap delivery",
          { "Punchy", "Rhythmic", "Bold" },
          0.5f, 1.5f, 2.5f, 1.0f, 0.0f, 0.05f, 0.1f,   5.0f, 0.0f, 600.0f, 0.0f, 6.0f,
          0.65f, 0.4f, 0.12f, 0.06f, 0.3f, 0.0f, 1, 0.35f, 90.0f,  0.25f, 0.95f,
          43, 60, -0.9f,  53, 70, 0.9f },

        { "R&B", "rnb", 0xffa04dff, 0xffe0b8ff, "Silky neo-soul tenor with falsetto", "Velvet R&B with lush runs",
          { "Smooth", "Silky", "Soulful" },
          0.5f, 1.5f, 1.0f, 2.0f, 0.0f, 0.25f, 0.0f,   5.8f, 30.0f, 200.0f, 60.0f, 5.0f,
          0.45f, 0.35f, 0.05f, 0.2f, 0.55f, 0.08f, 2, 0.15f, 80.0f,  0.8f, 0.55f,
          47, 72, -0.7f,  55, 81, 1.1f },

        { "Country", "country", 0xffd9a441, 0xfff6e2b0, "Warm Nashville baritone with twang", "Heartfelt country with bright twang",
          { "Twang", "Warm", "Heartfelt" },
          0.5f, 1.0f, 1.5f, 1.0f, 2.5f, 0.1f, 0.1f,   5.2f, 25.0f, 280.0f, 80.0f, 5.0f,
          0.4f, 0.3f, 0.08f, 0.16f, 0.45f, 0.1f, 4, 0.2f, 80.0f,  0.6f, 0.45f,
          43, 67, -1.0f,  55, 76, 0.9f },

        { "Gospel", "gospel", 0xfff2c14e, 0xfffff0c4, "Powerful church baritone", "Soaring gospel powerhouse",
          { "Powerful", "Soulful", "Choir" },
          0.5f, 2.0f, 2.0f, 1.5f, 0.0f, 0.1f, 0.15f,   6.0f, 45.0f, 180.0f, 100.0f, 6.0f,
          0.5f, 0.3f, 0.06f, 0.32f, 0.75f, 0.0f, 1, 0.3f, 70.0f,  0.75f, 0.45f,
          45, 72, -1.1f,  53, 81, 0.8f },

        { "Soul", "soul", 0xffc8553d, 0xfff3b49f, "Gritty vintage soul crooner", "Warm, smoky classic soul",
          { "Warm", "Gritty", "Vintage" },
          -0.5f, 2.5f, 1.0f, 0.0f, 0.0f, 0.15f, 0.3f,   5.6f, 35.0f, 220.0f, 70.0f, 6.0f,
          0.5f, 0.25f, 0.2f, 0.18f, 0.5f, 0.06f, 2, 0.1f, 70.0f,  0.7f, 0.5f,
          45, 70, -1.0f,  52, 76, 0.7f },

        { "Rock", "rock", 0xffe23b3b, 0xffff9f9f, "Raspy arena-rock frontman", "Fierce rock vocalist with edge",
          { "Gritty", "Powerful", "Raspy" },
          1.0f, 1.0f, 3.0f, 1.0f, 0.5f, 0.05f, 0.45f,   5.8f, 25.0f, 260.0f, 40.0f, 7.0f,
          0.6f, 0.3f, 0.35f, 0.16f, 0.6f, 0.08f, 1, 0.45f, 100.0f,  0.6f, 0.55f,
          45, 72, -0.8f,  55, 79, 0.9f },

        { "Jazz", "jazz", 0xff2fb5a3, 0xffa8eee3, "Velvet lounge crooner", "Smoky late-night jazz singer",
          { "Smooth", "Warm", "Intimate" },
          -1.0f, 2.0f, 0.0f, 0.5f, 0.0f, 0.2f, 0.05f,   5.0f, 30.0f, 350.0f, 50.0f, 6.0f,
          0.35f, 0.25f, 0.08f, 0.2f, 0.4f, 0.0f, 1, 0.0f, 60.0f,  0.8f, 0.4f,
          41, 65, -1.3f,  53, 74, 0.7f },

        { "Opera", "opera", 0xff8c6cff, 0xffd9ceff, "Classical operatic tenor", "Classical coloratura soprano",
          { "Powerful", "Classical", "Resonant" },
          0.0f, 1.0f, 3.5f, 0.5f, 0.0f, 0.02f, 0.0f,   5.8f, 70.0f, 120.0f, 20.0f, 3.0f,
          0.25f, 0.2f, 0.0f, 0.35f, 0.85f, 0.0f, 1, 0.0f, 60.0f,  0.9f, 0.3f,
          48, 72, -0.8f,  60, 84, 1.4f },

        { "EDM", "edm", 0xff00d1ff, 0xffa6f3ff, "Anthemic festival topline", "Euphoric dance-floor diva",
          { "Bright", "Airy", "Anthemic" },
          2.0f, -0.5f, 2.0f, 3.5f, 0.0f, 0.2f, 0.0f,   5.5f, 10.0f, 400.0f, 0.0f, 2.0f,
          0.6f, 0.4f, 0.1f, 0.28f, 0.7f, 0.18f, 2, 0.4f, 120.0f,  0.6f, 0.55f,
          48, 70, -0.6f,  60, 81, 1.4f },

        { "Indie Folk", "indie", 0xff7bbf6a, 0xffd5f2c9, "Breathy, intimate indie storyteller", "Delicate, breathy folk voice",
          { "Breathy", "Intimate", "Organic" },
          -0.5f, 1.0f, 0.5f, 1.5f, 0.0f, 0.4f, 0.05f,   5.0f, 10.0f, 400.0f, 15.0f, 5.0f,
          0.35f, 0.3f, 0.06f, 0.15f, 0.4f, 0.04f, 1, 0.3f, 80.0f,  0.7f, 0.45f,
          45, 67, -0.9f,  55, 74, 1.0f },

        { "Latin", "latin", 0xffff5f7e, 0xffffc2cf, "Passionate Latin pop tenor", "Fiery reggaeton & Latin pop",
          { "Passionate", "Bright", "Rhythmic" },
          1.0f, 0.5f, 2.0f, 1.5f, 0.5f, 0.1f, 0.05f,   6.0f, 30.0f, 250.0f, 40.0f, 5.0f,
          0.5f, 0.35f, 0.08f, 0.16f, 0.5f, 0.08f, 2, 0.25f, 85.0f,  0.6f, 0.7f,
          47, 69, -0.7f,  57, 77, 1.1f },

        { "K-Pop", "kpop", 0xffff7ad9, 0xffffd0f3, "Clean, bright idol vocal", "Crisp, airy idol vocal",
          { "Bright", "Airy", "Polished" },
          2.0f, -0.5f, 2.0f, 3.0f, 0.0f, 0.25f, 0.0f,   5.6f, 15.0f, 320.0f, 20.0f, 3.0f,
          0.55f, 0.4f, 0.04f, 0.15f, 0.45f, 0.06f, 1, 0.35f, 100.0f,  0.6f, 0.65f,
          50, 72, -0.4f,  59, 81, 1.5f },

        { "Metal", "metal", 0xff8a8f99, 0xffd5d9e0, "Aggressive metal vocalist", "Powerful symphonic metal voice",
          { "Aggressive", "Gritty", "Powerful" },
          1.0f, 0.5f, 3.5f, 0.0f, 1.0f, 0.0f, 0.7f,   6.0f, 20.0f, 300.0f, 20.0f, 8.0f,
          0.7f, 0.3f, 0.45f, 0.18f, 0.65f, 0.05f, 1, 0.5f, 110.0f,  0.5f, 0.6f,
          45, 74, -0.9f,  55, 83, 1.0f },

        { "Blues", "blues", 0xff3d6bd9, 0xff9db6f2, "Raspy, weathered blues voice", "Smoky, soulful blues belter",
          { "Raspy", "Soulful", "Vintage" },
          -1.0f, 2.0f, 1.5f, -0.5f, 0.5f, 0.12f, 0.4f,   5.2f, 35.0f, 260.0f, 90.0f, 7.0f,
          0.5f, 0.25f, 0.25f, 0.2f, 0.5f, 0.08f, 2, 0.0f, 70.0f,  0.65f, 0.45f,
          43, 67, -1.2f,  52, 74, 0.6f },

        { "Reggae", "reggae", 0xff2ecc71, 0xfff7dc6f, "Laid-back roots reggae voice", "Sunny, soulful reggae vocal",
          { "Laid-back", "Warm", "Rhythmic" },
          0.0f, 1.5f, 1.0f, 0.5f, 1.0f, 0.12f, 0.1f,   5.4f, 20.0f, 300.0f, 40.0f, 5.0f,
          0.45f, 0.3f, 0.1f, 0.18f, 0.5f, 0.15f, 4, 0.1f, 80.0f,  0.55f, 0.6f,
          45, 67, -0.9f,  55, 74, 0.9f },

        { "Afrobeats", "afrobeats", 0xffffa62b, 0xffffe0a8, "Smooth Afro-fusion melodic flow", "Silky Afrobeats & Afro-pop vocal",
          { "Smooth", "Rhythmic", "Melodic" },
          1.0f, 1.0f, 1.5f, 2.0f, 0.0f, 0.15f, 0.0f,   5.5f, 15.0f, 300.0f, 30.0f, 4.0f,
          0.5f, 0.35f, 0.06f, 0.14f, 0.45f, 0.1f, 2, 0.3f, 85.0f,  0.65f, 0.7f,
          47, 69, -0.7f,  57, 76, 1.1f },

        { "Trap", "trap", 0xff6a5cff, 0xffc3bdff, "Melodic trap with hard auto-tune", "Moody melodic trap vocal",
          { "Auto-Tuned", "Punchy", "Moody" },
          1.0f, 1.0f, 2.5f, 2.0f, 0.0f, 0.1f, 0.05f,   5.0f, 0.0f, 600.0f, 0.0f, 0.0f,
          0.65f, 0.4f, 0.12f, 0.16f, 0.55f, 0.15f, 1, 0.45f, 100.0f,  0.55f, 0.8f,
          43, 65, -0.9f,  55, 72, 1.0f },

        { "Musical Theatre", "theatre", 0xffe0457b, 0xfff5b5cb, "Bold Broadway leading man", "Brilliant Broadway belter",
          { "Powerful", "Clear", "Dramatic" },
          1.0f, 0.5f, 3.0f, 1.0f, 0.0f, 0.05f, 0.0f,   6.0f, 40.0f, 200.0f, 25.0f, 4.0f,
          0.45f, 0.3f, 0.04f, 0.25f, 0.7f, 0.0f, 1, 0.0f, 80.0f,  0.75f, 0.5f,
          47, 70, -0.7f,  57, 81, 1.2f },

        { "Lo-Fi", "lofi", 0xff9d8df1, 0xffe4ddff, "Mellow bedroom-pop murmur", "Dreamy lo-fi bedroom vocal",
          { "Intimate", "Breathy", "Dreamy" },
          -2.0f, 1.0f, -1.0f, -2.0f, 0.0f, 0.4f, 0.05f,   4.8f, 8.0f, 400.0f, 10.0f, 5.0f,
          0.5f, 0.3f, 0.3f, 0.22f, 0.5f, 0.12f, 1, 0.2f, 140.0f,  0.7f, 0.45f,
          43, 64, -1.0f,  55, 72, 0.9f },
    };

    /** Each genre's own vocal colour, chosen so that same-gender voices are clearly different:
        vocal-tract size (formants), an upper "ring" resonance, a lower "body" resonance, and how
        breathy / gritty the voice is. Applied on top of the genre's tonal EQ, which is also widened. */
    struct VoiceColour
    {
        const char* idPart;
        float maleFormant, femaleFormant;
        float ringHz, ringDb, bodyHz, bodyDb;
        float breathScale, raspAdd;
    };

    constexpr float eqWidening = 1.6f;

    const VoiceColour voiceColours[] = {
        { "pop",       -0.4f, 1.4f, 3200.0f, 3.5f, 250.0f, -1.5f, 1.0f, 0.00f },
        { "rap",       -0.6f, 0.6f, 1800.0f, 3.0f, 160.0f,  2.5f, 0.7f, 0.05f },
        { "rnb",       -0.7f, 0.9f, 2600.0f, 2.0f, 300.0f,  2.0f, 1.6f, 0.00f },
        { "country",   -1.1f, 1.2f, 2300.0f, 4.0f, 400.0f, -1.0f, 0.8f, 0.05f },
        { "gospel",    -1.3f, 0.5f, 2900.0f, 4.5f, 220.0f,  3.0f, 0.6f, 0.10f },
        { "soul",      -1.2f, 0.5f, 1600.0f, 3.5f, 320.0f,  2.0f, 1.0f, 0.15f },
        { "rock",      -0.8f, 0.9f, 1700.0f, 5.0f, 500.0f, -2.0f, 0.6f, 0.15f },
        { "jazz",      -1.6f, 0.3f,  700.0f, 2.5f, 200.0f,  3.0f, 1.4f, 0.05f },
        { "opera",     -1.3f, 1.6f, 3000.0f, 6.0f, 350.0f,  1.5f, 0.3f, 0.00f },
        { "edm",       -0.3f, 1.6f, 5200.0f, 4.0f, 400.0f, -3.0f, 1.2f, 0.00f },
        { "indie",     -0.7f, 1.1f, 1500.0f, -2.0f, 250.0f, 1.0f, 1.8f, 0.00f },
        { "latin",     -0.5f, 1.4f, 2500.0f, 3.0f, 300.0f, -2.0f, 1.1f, 0.00f },
        { "kpop",      -0.1f, 2.0f, 2900.0f, 3.5f, 220.0f, -1.0f, 1.3f, 0.00f },
        { "metal",     -1.1f, 0.7f, 1200.0f, 5.5f, 180.0f,  2.5f, 0.4f, 0.25f },
        { "blues",     -1.6f, 0.1f,  800.0f, 3.0f, 180.0f,  2.5f, 1.0f, 0.20f },
        { "reggae",    -1.1f, 0.7f, 1300.0f, 2.5f, 350.0f,  2.0f, 1.1f, 0.05f },
        { "afrobeats", -0.8f, 0.9f, 3900.0f, 3.5f, 200.0f,  2.5f, 1.1f, 0.00f },
        { "trap",      -1.4f, 0.9f, 2300.0f, 2.0f, 140.0f,  4.0f, 1.0f, 0.00f },
        { "theatre",   -0.8f, 1.3f, 3300.0f, 5.0f, 400.0f, -1.0f, 0.4f, 0.00f },
        { "lofi",      -0.9f, 0.7f, 1000.0f, 2.0f, 300.0f,  1.5f, 1.5f, 0.05f },
    };

    const VoiceColour* findColour (const char* idPart)
    {
        for (const auto& c : voiceColours)
            if (juce::String (c.idPart) == idPart)
                return &c;
        return nullptr;
    }

    VoiceProfile makeVoice (const GenreStyle& g, VoiceGender gender)
    {
        const bool male = gender == VoiceGender::Male;

        VoiceProfile p;
        p.id = juce::String (male ? "male-" : "female-") + g.idPart;
        p.name = juce::String (male ? "Male " : "Female ") + g.genre;
        p.tagline = male ? g.maleTagline : g.femaleTagline;
        p.genre = g.genre;
        p.styles = g.styles;
        p.accentA = juce::Colour (g.colourA);
        p.accentB = juce::Colour (g.colourB);

        auto& t = p.traits;
        t.gender = gender;
        t.rangeLowMidi  = male ? g.maleLow : g.femaleLow;
        t.rangeHighMidi = male ? g.maleHigh : g.femaleHigh;
        t.formantSemis  = male ? g.maleFormant : g.femaleFormant;
        t.tiltDb = g.tiltDb + (male ? -0.5f : 0.5f);
        t.warmthDb = g.warmthDb + (male ? 0.5f : -0.5f);
        t.presenceDb = g.presenceDb;
        t.airDb = g.airDb + (male ? 0.0f : 0.5f);
        t.nasalDb = g.nasalDb;
        t.breathiness = juce::jlimit (0.0f, 1.0f, g.breathiness + (male ? 0.0f : 0.05f));
        t.rasp = juce::jlimit (0.0f, 1.0f, g.rasp * (male ? 1.0f : 0.8f));
        if (const auto* c = findColour (g.idPart))
        {
            t.formantSemis = male ? c->maleFormant : c->femaleFormant;
            t.ringHz = c->ringHz * (male ? 0.92f : 1.0f);
            t.ringDb = c->ringDb;
            t.bodyHz = c->bodyHz * (male ? 0.85f : 1.0f);
            t.bodyDb = c->bodyDb;
            t.breathiness = juce::jlimit (0.0f, 1.0f, t.breathiness * c->breathScale);
            t.rasp = juce::jlimit (0.0f, 1.0f, t.rasp + c->raspAdd);
        }
        t.tiltDb *= eqWidening;
        t.warmthDb *= eqWidening;
        t.presenceDb *= eqWidening;
        t.airDb *= eqWidening;
        t.nasalDb *= eqWidening;

        t.vibratoRateHz = g.vibRate + (male ? -0.1f : 0.1f);
        t.vibratoDepthCents = g.vibDepth;
        t.vibratoOnsetMs = g.vibOnset;
        t.scoopCents = g.scoop;
        t.jitterCents = g.jitter;
        t.vibeCompressor = g.comp;
        t.vibeDeEsser = g.deEss;
        t.vibeSaturation = g.sat;
        t.vibeReverbMix = g.revMix;
        t.vibeReverbSize = g.revSize;
        t.vibeDelayMix = g.dlyMix;
        t.vibeDelayDivision = g.dlyDiv;
        t.vibeDoubler = g.doubler;
        t.vibeLowCutHz = g.lowCut + (male ? 0.0f : 20.0f);
        t.singLegato = g.legato;
        t.singRhythmBusy = g.busy;
        return p;
    }

    VoiceProfile makeSpecialty (const char* id, const char* name, const char* tagline, VoiceGender gender,
                                juce::StringArray styles, juce::uint32 a, juce::uint32 b,
                                std::function<void (VoiceTraits&)> setup)
    {
        VoiceProfile p;
        p.id = id;
        p.name = name;
        p.tagline = tagline;
        p.genre = "Specialty";
        p.styles = std::move (styles);
        p.accentA = juce::Colour (a);
        p.accentB = juce::Colour (b);
        p.traits.gender = gender;
        setup (p.traits);
        return p;
    }
}

std::vector<VoiceProfile> createFactoryVoices()
{
    std::vector<VoiceProfile> voices;

    // Male voices first, then female, in genre order - matches the library carousel ordering.
    for (const auto& g : genreStyles)
        voices.push_back (makeVoice (g, VoiceGender::Male));
    for (const auto& g : genreStyles)
        voices.push_back (makeVoice (g, VoiceGender::Female));

    voices.push_back (makeSpecialty ("male-deep-bass", "Deep Bass", "Cavernous low bass voice", VoiceGender::Male,
                                     { "Deep", "Resonant", "Warm" }, 0xff1f3a93, 0xff6f8fd9, [] (VoiceTraits& t)
    {
        t.rangeLowMidi = 33; t.rangeHighMidi = 57; t.formantSemis = -2.6f;
        t.ringHz = 2400.0f; t.ringDb = 2.0f; t.bodyHz = 120.0f; t.bodyDb = 4.0f;
        t.tiltDb = -2.0f; t.warmthDb = 3.0f; t.presenceDb = 0.5f; t.airDb = -1.0f;
        t.vibratoDepthCents = 15.0f; t.vibratoRateHz = 5.0f; t.jitterCents = 4.0f;
        t.vibeCompressor = 0.5f; t.vibeReverbMix = 0.15f; t.vibeLowCutHz = 40.0f;
    }));

    voices.push_back (makeSpecialty ("female-angelic", "Angelic Soprano", "Pure, ethereal head voice", VoiceGender::Female,
                                     { "Airy", "Ethereal", "Classical" }, 0xffbfd9ff, 0xffffffff, [] (VoiceTraits& t)
    {
        t.rangeLowMidi = 64; t.rangeHighMidi = 88; t.formantSemis = 1.9f;
        t.ringHz = 3600.0f; t.ringDb = 4.0f; t.bodyHz = 300.0f; t.bodyDb = -3.0f;
        t.tiltDb = 2.0f; t.airDb = 4.0f; t.breathiness = 0.3f;
        t.vibratoDepthCents = 25.0f; t.vibratoRateHz = 5.4f; t.vibratoOnsetMs = 300.0f;
        t.vibeReverbMix = 0.4f; t.vibeReverbSize = 0.9f; t.vibeDoubler = 0.2f; t.vibeLowCutHz = 140.0f;
    }));

    voices.push_back (makeSpecialty ("female-smoky-alto", "Smoky Alto", "Low, husky contralto", VoiceGender::Female,
                                     { "Smoky", "Warm", "Intimate" }, 0xff7a4e8f, 0xffd2b4e0, [] (VoiceTraits& t)
    {
        t.rangeLowMidi = 50; t.rangeHighMidi = 70; t.formantSemis = 0.1f;
        t.ringHz = 2200.0f; t.ringDb = 2.0f; t.bodyHz = 250.0f; t.bodyDb = 3.5f;
        t.tiltDb = -1.0f; t.warmthDb = 2.0f; t.breathiness = 0.4f; t.rasp = 0.15f;
        t.vibratoDepthCents = 25.0f; t.vibratoRateHz = 5.2f; t.scoopCents = 40.0f;
        t.vibeReverbMix = 0.18f; t.vibeSaturation = 0.1f;
    }));

    voices.push_back (makeSpecialty ("neutral-whisper", "Whisper", "Close, breathy ASMR whisper", VoiceGender::Neutral,
                                     { "Breathy", "Intimate", "Airy" }, 0xff8fa3bf, 0xffe1e8f2, [] (VoiceTraits& t)
    {
        t.rangeLowMidi = 48; t.rangeHighMidi = 72; t.formantSemis = 0.2f;
        t.breathiness = 0.9f; t.airDb = 4.0f; t.presenceDb = 1.0f; t.vibratoDepthCents = 0.0f;
        t.ringHz = 6000.0f; t.ringDb = 3.0f; t.bodyHz = 250.0f; t.bodyDb = -3.0f;
        t.vibeCompressor = 0.6f; t.vibeReverbMix = 0.08f; t.vibeDeEsser = 0.5f;
    }));

    voices.push_back (makeSpecialty ("neutral-robot", "Hard Tune Robot", "Zero-latency robotic pitch snap", VoiceGender::Neutral,
                                     { "Auto-Tuned", "Robotic", "Creative" }, 0xff00e5c7, 0xffb2fff3, [] (VoiceTraits& t)
    {
        t.rangeLowMidi = 45; t.rangeHighMidi = 72; t.formantSemis = 0.0f;
        t.tiltDb = 3.0f; t.presenceDb = 3.0f; t.vibratoDepthCents = 0.0f; t.jitterCents = 0.0f;
        t.ringHz = 2000.0f; t.ringDb = 5.0f; t.bodyHz = 400.0f; t.bodyDb = -2.0f;
        t.vibeCompressor = 0.6f; t.vibeSaturation = 0.2f; t.vibeDelayMix = 0.15f; t.vibeReverbMix = 0.15f;
    }));

    voices.push_back (makeSpecialty ("neutral-choir", "Choir Ensemble", "Massive stacked choir section", VoiceGender::Neutral,
                                     { "Choir", "Powerful", "Lush" }, 0xffffd479, 0xfffff4d6, [] (VoiceTraits& t)
    {
        t.rangeLowMidi = 48; t.rangeHighMidi = 76; t.formantSemis = 0.3f;
        t.breathiness = 0.15f; t.vibratoDepthCents = 25.0f; t.vibratoRateHz = 5.6f;
        t.ringHz = 2800.0f; t.ringDb = 3.0f; t.bodyHz = 300.0f; t.bodyDb = 2.0f;
        t.vibeDoubler = 0.9f; t.vibeReverbMix = 0.38f; t.vibeReverbSize = 0.85f; t.vibeCompressor = 0.4f;
    }));

    return voices;
}

} // namespace voxa
