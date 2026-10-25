#include "VoiceAvatar.h"
#include "VoxaAssets.h"

#include <map>

namespace voxa::ui
{

namespace
{
    //==============================================================================
    // All artwork is designed in a 100 x 100 box, figure facing right (female portraits are mirrored).

    const char* const malePath =
        "M 0 100 L 0 90 C 6 84 16 80 27 78 C 33 76.5 36.5 73 37.5 68 C 38 63 36.5 58 34 55 C 29 50 27 43 27.5 36 "
        "C 28 24 37 15 49 14 C 58 13.5 65 18 67 26 C 67.8 29.5 67.6 31.5 68.2 33 C 68.6 34 68.2 35 67.8 35.8 "
        "C 69.5 38.5 72 41.5 73.6 43.6 C 74.4 44.6 73.6 45.8 71.8 45.9 C 70.6 46 69.8 46.2 69.6 46.8 "
        "C 70.1 47.8 70.2 48.6 69.2 49.2 C 68.8 49.4 68.8 49.8 69.3 50.3 C 69.7 51 69.4 52 68.3 52.4 "
        "C 67.9 53 68.2 54.2 68.6 55.5 C 69 57.5 68.4 59.4 66.2 60.2 C 63 61.2 59 61.6 56 62.6 "
        "C 56.2 66 57 70 58.5 74 C 66 78 80 82 90 88 C 95 91 99 95 100 100 Z";

    const char* const femalePath =
        "M 0 100 L 0 93 C 7 87 17 83.5 29 81.5 C 34.5 80.5 37.8 77 38.6 71.5 C 39 66 37.5 60.5 35 57 "
        "C 30 51.5 28.5 44 29 37 C 29.5 25.5 38 16.5 49.5 16 C 58.5 15.6 64.5 20 66.2 27.5 C 66.8 30.5 66.6 32.5 67 33.8 "
        "C 67.2 34.6 67 35.3 66.7 36 C 68.2 38.6 70.4 41.2 71.8 43.2 C 72.5 44.2 71.8 45.2 70.3 45.3 "
        "C 69.3 45.4 68.7 45.6 68.6 46.2 C 69.4 47.2 69.8 48.2 68.6 48.9 C 68.2 49.1 68.2 49.5 68.7 49.9 "
        "C 69.6 50.8 69.2 52.2 67.8 52.6 C 67.3 53.1 67.4 54.2 67.6 55.2 C 67.9 57 66.9 58.7 64.8 59.3 "
        "C 61.5 60.2 58.5 60.8 56 61.8 C 56.5 66 57.5 71 59 76 C 66 80 78 84 88 89 C 94 92 98 96 100 100 Z";

    enum class Figure { Male, Female, Robot, Choir };
    enum class Hair { None, Short, Quiff, Fringe, Slick, Afro, LongStraight, LongWavy, Curly, Bob, Updo, SpaceBuns, Dreads, LongMale };
    enum class Hat { None, Cap, Fedora, Cowboy, Beanie, Tam, Headwrap, TopHat, Hood };

    struct Recipe
    {
        Figure figure = Figure::Male;
        Hair hair = Hair::Short;
        Hat hat = Hat::None;
        bool headphones = false, sunglasses = false, hoops = false, stud = false, beard = false, stubble = false,
             moustache = false, bowTie = false, stole = false, halo = false, tiara = false, flower = false,
             jacket = false, chain = false, bangs = false;
    };

    Recipe recipeFor (const VoiceProfile& v)
    {
        Recipe r;
        const bool female = v.getGender() == VoiceGender::Female;
        const bool neutral = v.getGender() == VoiceGender::Neutral;
        r.figure = (female || neutral) ? Figure::Female : Figure::Male;
        r.hair = (female || neutral) ? Hair::LongStraight : Hair::Short;

        auto is = [&v] (const char* s) { return v.genre.containsIgnoreCase (s) || v.id.containsIgnoreCase (s); };

        if (v.id == "male-deep-bass")         { r.hair = Hair::None; r.beard = true; r.moustache = true; r.jacket = true; return r; }
        if (v.id == "female-angelic")         { r.hair = Hair::LongWavy; r.halo = true; return r; }
        if (v.id == "female-smoky-alto")      { r.hair = Hair::Bob; r.hoops = true; return r; }
        if (v.id == "neutral-whisper")        { r.hair = Hair::Bob; r.hat = Hat::Hood; return r; }
        if (v.id == "neutral-robot")          { r.figure = Figure::Robot; return r; }
        if (v.id == "neutral-choir")          { r.figure = Figure::Choir; r.stole = true; return r; }

        if (is ("k-pop") || is ("kpop"))      { if (female) { r.hair = Hair::LongStraight; r.bangs = true; r.stud = true; } else { r.hair = Hair::Fringe; r.stud = true; } }
        else if (is ("pop"))                  { if (female) { r.hair = Hair::LongWavy; r.stud = true; } else r.hair = Hair::Quiff; }
        else if (is ("trap"))                 { if (female) { r.hair = Hair::SpaceBuns; r.hoops = true; } else { r.hat = Hat::Beanie; r.chain = true; r.sunglasses = true; } }
        else if (is ("rap") || is ("hip"))    { if (female) { r.hair = Hair::LongStraight; r.hoops = true; } else { r.hat = Hat::Cap; r.chain = true; } }
        else if (is ("r&b") || is ("rnb"))    { if (female) { r.hair = Hair::Curly; r.stud = true; } else { r.hair = Hair::Short; r.sunglasses = true; r.stubble = true; } }
        else if (is ("country"))              { r.hat = Hat::Cowboy; if (female) r.hair = Hair::LongWavy; else r.stubble = true; }
        else if (is ("gospel"))               { r.stole = true; if (female) r.hair = Hair::Curly; else r.hair = Hair::Short; }
        else if (is ("soul"))                 { r.hair = Hair::Afro; if (female) r.hoops = true; else r.moustache = true; }
        else if (is ("rock"))                 { r.jacket = true; if (female) r.hair = Hair::LongStraight; else { r.hair = Hair::LongMale; r.stubble = true; } }
        else if (is ("jazz"))                 { if (female) { r.hair = Hair::Updo; r.stud = true; } else { r.hat = Hat::Fedora; r.bowTie = true; } }
        else if (is ("opera") || is ("classical")) { if (female) { r.hair = Hair::Updo; r.tiara = true; } else { r.hair = Hair::Slick; r.bowTie = true; } }
        else if (is ("edm") || is ("dance"))  { r.headphones = true; if (female) r.hair = Hair::LongStraight; else r.hair = Hair::Quiff; }
        else if (is ("indie") || is ("folk")) { if (female) { r.hair = Hair::Bob; r.flower = true; } else { r.hat = Hat::Beanie; r.beard = true; } }
        else if (is ("latin"))                { if (female) { r.hair = Hair::LongWavy; r.hoops = true; r.flower = true; } else { r.hair = Hair::Slick; r.chain = true; r.stubble = true; } }
        else if (is ("metal"))                { r.jacket = true; if (female) r.hair = Hair::LongWavy; else { r.hair = Hair::LongMale; r.beard = true; } }
        else if (is ("blues"))                { if (female) { r.hair = Hair::Curly; r.hoops = true; } else { r.hat = Hat::Fedora; r.beard = true; } }
        else if (is ("reggae"))               { if (female) { r.hat = Hat::Headwrap; r.hoops = true; r.hair = Hair::None; } else { r.hair = Hair::Dreads; r.hat = Hat::Tam; } }
        else if (is ("afro"))                 { if (female) { r.hat = Hat::Headwrap; r.hair = Hair::None; r.hoops = true; } else { r.hair = Hair::Short; r.chain = true; } }
        else if (is ("theatre") || is ("musical")) { if (female) { r.hair = Hair::Updo; r.stud = true; } else { r.hat = Hat::TopHat; r.bowTie = true; } }
        else if (is ("lo-fi") || is ("lofi")) { r.headphones = true; if (female) r.hair = Hair::Bob; else r.hat = Hat::Beanie; }

        return r;
    }

    //==============================================================================
    juce::Path svg (const char* d) { return juce::Drawable::parseSVGPath (d); }

    const std::map<Hair, const char*>& femaleHairShapes()
    {
        static const std::map<Hair, const char*> t {
            { Hair::LongStraight, "M 64.5 22 C 62 15 55 11 47 11 C 36 11 26 17 23 30 C 20.5 42 22 52 20 64 C 18.5 75 14 86 10 100 L 46 100 C 44 92 42.5 82 43 72 C 43.5 64 47.5 57 51.5 51 C 54.5 46 56.5 39 58 33 C 59.2 28.5 61.5 24.5 64.5 22 Z" },
            { Hair::LongWavy, "M 64.5 22 C 62.5 14.5 55 10 46.5 10 C 34 10 24 17 21.5 30 C 19 42 23 48 19.5 58 C 16.5 66 21 72 17.5 81 C 15 88 12 94 8 100 L 47 100 C 43 93 46 86 42.5 78 C 40 71 45.5 63 48.5 56 C 51 50.5 55.5 45 56.8 38.5 C 57.8 32 60.5 25.5 64.5 22 Z" },
            { Hair::Curly, "M 65 23 C 66 12 54 5 42 6.5 C 28 8 18 19 17 35 C 16 50 19 62 15.5 75 C 24 80 35 78 44 72 C 46 64 49 56 53 49 C 56 43 57 36 58.5 31 C 60 27.5 62.5 24.5 65 23 Z" },
            { Hair::Afro, "M 64.5 24 C 70 14 64 3 50 1.5 C 34 0 19 8 15 24 C 11 40 16 54 26 60 C 34 64 42 60 46 54 C 49 48 51 42 54 36 C 57 30 60.5 26.5 64.5 24 Z" },
            { Hair::Bob, "M 64.5 22 C 62 14 55 10.5 46.5 10.5 C 35 10.5 25 18 23.5 31 C 22.5 41 23.5 50 25 58 C 26 62 29 64.5 33.5 64 L 53 61.5 C 52.5 56 53.6 48.5 56 41 C 58 34 61 26 64.5 22 Z" },
            { Hair::Updo, "M 64.5 22.5 C 62 15.5 55.5 12 47.5 12 C 37.5 12 29.8 17 27.8 27.5 C 26.5 35 28 43 32 49.5 C 35 53.5 39.5 54.5 42.5 52 C 45.5 49 47.5 44 50.5 40 C 54 35.5 57.5 29.5 64.5 22.5 Z M 37.5 19.5 A 8.5 8.5 0 1 0 20.5 19.5 A 8.5 8.5 0 1 0 37.5 19.5 Z" },
            { Hair::SpaceBuns, "M 64.5 22.5 C 62 15.5 55.5 12 47.5 12 C 37.5 12 29.8 17 27.8 27.5 C 26.5 35 28 43 32 49.5 C 35 53.5 39.5 54.5 42.5 52 C 45.5 49 47.5 44 50.5 40 C 54 35.5 57.5 29.5 64.5 22.5 Z M 43 9.5 A 7 7 0 1 0 29 9.5 A 7 7 0 1 0 43 9.5 Z M 60 8.5 A 6.5 6.5 0 1 0 47 8.5 A 6.5 6.5 0 1 0 60 8.5 Z" },
        };
        return t;
    }

    const std::map<Hair, const char*>& maleHairShapes()
    {
        static const std::map<Hair, const char*> t {
            { Hair::Short, "M 66.8 26.2 C 68.6 23.4 68.2 20 66.2 17.8 C 66.9 15.8 65.6 13.6 63.4 13.2 C 62.2 10.9 59.4 9.9 56.8 10.6 C 54.4 8.9 50.6 8.7 47.8 9.7 C 44.6 8.9 40.6 9.9 38.2 12 C 34.8 12.2 31.9 14.2 30.6 17.2 C 28.2 19.2 27 22.6 27.3 26 C 26 29.2 26.2 33.4 27.4 36.8 C 27.4 39.6 28 42.4 28.8 44.5 C 33 44 37 42 40.5 39 C 43 37 45.5 37.5 47.5 40 L 49.6 45.6 L 51.6 45.2 C 51.4 39.4 52.6 33.6 55.2 29.6 C 57.6 27.6 60.8 27.8 62.8 27 C 64.2 27.8 65.6 27.4 66.8 26.2 Z" },
            { Hair::Quiff, "M 68.2 25 C 71.8 22.4 72.4 17.6 70 14.2 C 70.4 11 67.6 7.8 63.8 7.6 C 61.4 5.2 57 4.6 53.6 6 C 50 4.8 45.4 5.6 42.6 8 C 38.6 7.8 34.6 10 32.6 13.4 C 29.6 15.4 27.6 19.4 27.6 23.6 C 26 27.4 26.2 32.6 27.6 36.8 C 27.6 39.6 28.1 42.4 28.8 44.5 C 33 44 37 42 40.5 39 C 43 37 45.5 37.5 47.5 40 L 49.6 45.6 L 51.6 45.2 C 51.4 39.4 52.6 33.6 55.2 29.6 C 58 27 62.4 26.4 65 26.6 C 66.2 26.4 67.4 25.8 68.2 25 Z" },
            { Hair::Fringe, "M 68 31.5 C 69.5 22 63 10 49 10 C 37 10 28 16 26.6 27.5 C 25.9 33.5 26.6 39 28.8 44.5 C 33 44 37 42 40.5 39 C 43 37 45.5 37.5 47.5 40 L 49.6 45.6 L 51.6 45.2 C 51.6 39.5 53 34.5 56 31.5 C 60 29.5 64 31 68 31.5 Z" },
            { Hair::Slick, "M 66 24 C 65 16.5 58 11.5 48.5 11.5 C 38 11.5 29.5 16 27.4 26 C 26.4 32 27 38 29 42.5 C 33 43 36 42 39 40 C 42 38 45 38 47.5 40.5 L 49.6 45.6 L 51.6 45.2 C 51.6 38 53 32 56 28 C 59 25.6 62.4 24.6 66 24 Z" },
            { Hair::Afro, "M 67 27 C 72 20 70 7 58 3.5 C 46 0 30 2.5 22 12 C 15 21 16 34 21 43 C 24.5 48.5 30 50 35 47 C 39 44.5 43 42 47 41.5 L 50 45.5 L 52 45 C 52.5 38 54.5 32.5 58 29.5 C 61 28 64 27.6 67 27 Z" },
            { Hair::LongMale, "M 66.5 25 C 67 16.5 59.5 10 48.5 10 C 36 10 26.5 17 24.5 30 C 23 41 24 52 22.5 63 C 21.5 70 19 77 16 84 L 24 82 L 27 86 L 33 80 L 38 84 L 41 74 C 42 66 44 58 47.5 52 C 50 47.5 51.6 42 52.2 36 C 53.5 30.5 59.5 26.5 66.5 25 Z" },
            { Hair::Dreads, "M 66 25.5 C 66.5 17 59.5 11 48.5 11 C 37 11 28 16.5 26.6 27.5 C 25.9 33.5 26.6 39 28.8 44.5 C 33 44 37 42 40.5 39 C 43 37 45.5 37.5 47.5 40 L 49.6 45.6 L 51.6 45.2 C 51.4 39.4 52.6 33.6 55.2 29.6 C 58.2 26.8 62 26 66 25.5 Z" },
        };
        return t;
    }

    const std::map<Hat, const char*>& hatShapes()
    {
        static const std::map<Hat, const char*> t {
            { Hat::Cap, "M 64 24.5 C 64.5 14 56 8 46 8.5 C 35.5 9 27.5 15 26.2 26.5 L 26.4 30 C 38 28 51 26.2 64 24.5 Z" },
            { Hat::Fedora, "M 33.5 20 C 33 12 38.5 6 48 6 C 57.5 6 62.5 12 63 20 L 63.3 23 L 33.2 23.5 Z" },
            { Hat::Cowboy, "M 31.5 21.5 C 30.5 11 36 4.5 44 4.6 C 47 4.6 48.5 6.5 50.5 6.4 C 53 6.2 54.5 4.4 57.5 4.8 C 63 5.6 65 12 64.5 21.5 Z" },
            { Hat::Beanie, "M 66.5 27 C 67.5 15 60 6 48 6 C 35.5 6 26.5 14 25.8 27 L 26.2 31 C 40 29.4 53.5 28.2 66.5 27 Z" },
            { Hat::Tam, "M 67 27.5 C 68 18 63 8 50 5 C 38 2.5 22 6 15 16 C 11 22 13 29 19 31 C 30 33 48 29.5 67 27.5 Z" },
            { Hat::Headwrap, "M 66 26 C 68 18 64 9 56 5 C 50 2 40 1 32 4 C 24 7 19 14 21 22 C 22.5 29 25 35 28.5 40 C 33 38 38 34 42 30 C 47 27 56 26 66 26 Z" },
            { Hat::TopHat, "M 34 22 L 35.5 -1 L 60.5 -1 L 62 22 Z" },
            { Hat::Hood, "M 69 27 C 67.5 13 57 5 45 5 C 30 5 20 17 19 33 C 18 50 21 63 27.5 73 C 31.5 79 40 81 46 78.5 L 51 70 C 51.5 62 53.5 52 57 45 C 60 38.5 64 32 69 27 Z" },
        };
        return t;
    }

    const std::map<Hat, const char*>& brimShapes()
    {
        static const std::map<Hat, const char*> t {
            { Hat::Cap, "M 60.5 23.6 C 68 21.8 76.5 21.8 82.5 24 C 83.6 24.8 82.6 26 80.6 26 C 74 26.2 67.5 26.6 61 27.6 Z" },
            { Hat::Fedora, "M 21 23.5 C 30 21.2 52 20.6 77 22.4 C 79.2 22.8 79 24.4 76.8 24.8 C 60 25.6 40 26.4 23 26.8 C 20.4 26.8 19.6 24.4 21 23.5 Z" },
            { Hat::Cowboy, "M 13 16.5 C 18 22 30 23.6 47.5 23.6 C 64 23.6 76 22 83.5 16.8 C 85.5 15.8 87 17.4 85.4 19.6 C 79.5 26.4 66 28.8 47.5 28.8 C 29.5 28.8 16.5 26.2 11.5 18.8 C 10.6 17.2 11.8 15.8 13 16.5 Z" },
            { Hat::Beanie, "M 26 25 C 40 23.4 54 22.4 67 21.6 L 66.6 27.4 C 53.6 28.4 40 29.6 26.2 31.2 Z" },
            { Hat::TopHat, "M 25 22.5 C 35 20.5 58 20 72 21.5 C 74 21.8 74 23.6 72 24 C 58 25 38 25.4 26.5 25.6 C 24.2 25.6 23.6 23.2 25 22.5 Z" },
        };
        return t;
    }

    const std::map<Hat, const char*>& bandShapes()
    {
        static const std::map<Hat, const char*> t {
            { Hat::Fedora, "M 33.3 18.6 L 63.1 18 L 63.3 22.6 L 33.2 23.2 Z" },
            { Hat::Cowboy, "M 31.3 18.2 L 64.6 18.2 L 64.5 21.8 L 31.5 21.8 Z" },
            { Hat::TopHat, "M 34.2 17 L 61.7 17 L 62 21.6 L 34 21.6 Z" },
        };
        return t;
    }

    template <typename Key>
    juce::Path shapeFrom (const std::map<Key, const char*>& table, Key key)
    {
        const auto it = table.find (key);
        return it != table.end() ? svg (it->second) : juce::Path();
    }

    juce::Path hairPath (Hair hair, bool female)
    {
        if (female)
            if (auto p = shapeFrom (femaleHairShapes(), hair); ! p.isEmpty())
                return p;
        return shapeFrom (maleHairShapes(), hair);
    }

    juce::Path hatPath (Hat hat)  { return shapeFrom (hatShapes(), hat); }
    juce::Path hatBrim (Hat hat)  { return shapeFrom (brimShapes(), hat); }
    juce::Path hatBand (Hat hat)  { return shapeFrom (bandShapes(), hat); }

    //==============================================================================
    struct Painter
    {
        juce::Graphics& g;
        const VoiceProfile& voice;
        Recipe recipe;
        juce::Colour a, b;          // accents
        juce::Colour lit, dark, hairDark, hairLit;
        juce::Random rng;

        Painter (juce::Graphics& gr, const VoiceProfile& v)
            : g (gr), voice (v), recipe (recipeFor (v)), a (v.accentA), b (v.accentB),
              rng ((juce::int64) v.id.hashCode64())
        {
            const auto h = (juce::uint64) v.id.hashCode64();
            setPerson ((int) (h % 7), (int) ((h >> 8) % 6), recipe.figure == Figure::Female);
        }

        /** Skin tone and natural hair colour, lit by the voice's accent colours. */
        void setPerson (int skinIndex, int hairIndex, bool female)
        {
            static const juce::uint32 skins[] = { 0xfff3d1bb, 0xffe8b896, 0xffd09a72, 0xffb27a55, 0xff8d5a3b, 0xff6a412b, 0xff4d2f20 };
            static const juce::uint32 hairs[] = { 0xff17110f, 0xff2e1d14, 0xff533520, 0xff7a3b1f, 0xffc39a55, 0xffd8cdbd };

            // darker skin tones get darker hair more often; platinum / blonde mostly for lighter tones
            if (skinIndex >= 4 && hairIndex >= 3)
                hairIndex = (hairIndex + skinIndex) % 3;
            if (! female && hairIndex == 5)
                hairIndex = 1;
            if ((recipe.hair == Hair::Curly || recipe.hair == Hair::Afro || recipe.hair == Hair::Dreads) && hairIndex >= 4)
                hairIndex = hairIndex - 3;

            const juce::Colour skin (skins[juce::jlimit (0, 6, skinIndex)]);
            const juce::Colour hairColour (hairs[juce::jlimit (0, 5, hairIndex)]);

            lit = skin.interpolatedWith (b, 0.12f).withMultipliedBrightness (0.96f);
            dark = skin.darker (1.6f).interpolatedWith (juce::Colour (0xff0a1030), 0.55f);
            hairLit = hairColour.brighter (0.35f).interpolatedWith (b, 0.18f);
            hairDark = hairColour.darker (0.8f).interpolatedWith (juce::Colour (0xff05081a), 0.35f);
        }

        juce::ColourGradient rimGradient (float alpha) const
        {
            juce::ColourGradient cg (b.withAlpha (alpha), 82.0f, 18.0f, b.withAlpha (0.0f), 38.0f, 66.0f, false);
            cg.addColour (0.35, b.withAlpha (alpha * 0.85f));
            return cg;
        }

        void rim (const juce::Path& p, float width = 0.8f, float alpha = 1.0f)
        {
            for (auto [w, al] : { std::pair { 4.2f, 0.10f }, { 2.4f, 0.22f } })
            {
                g.setGradientFill (rimGradient (alpha * al));
                g.strokePath (p, juce::PathStrokeType (width * w, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            }
            g.setGradientFill (rimGradient (alpha));
            g.strokePath (p, juce::PathStrokeType (width, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        }

        void fillFigure (const juce::Path& body, float brightness = 1.0f)
        {
            juce::ColourGradient cg (lit.withMultipliedBrightness (brightness), 76.0f, 36.0f, dark, 30.0f, 54.0f, false);
            cg.addColour (0.32, lit.interpolatedWith (dark, 0.35f).withMultipliedBrightness (brightness));
            cg.addColour (0.62, lit.interpolatedWith (dark, 0.75f).withMultipliedBrightness (brightness));
            g.setGradientFill (cg);
            g.fillPath (body);

            g.saveState();
            g.reduceClipRegion (body);

            // key light on the face
            g.setGradientFill (juce::ColourGradient (b.interpolatedWith (juce::Colours::white, 0.3f).withAlpha (0.3f * brightness), 66.0f, 40.0f,
                                                     b.withAlpha (0.0f), 47.0f, 40.0f, true));
            g.fillRect (juce::Rectangle<float> (30.0f, 14.0f, 50.0f, 56.0f));

            // soft shadow under the jaw
            g.setGradientFill (juce::ColourGradient (juce::Colours::black.withAlpha (0.35f), 58.0f, 64.0f,
                                                     juce::Colours::black.withAlpha (0.0f), 58.0f, 72.0f, false));
            g.fillRect (juce::Rectangle<float> (40.0f, 60.0f, 26.0f, 14.0f));

            // clothing
            juce::Path garment = svg ("M -2 102 L -2 79 C 18 82 34 78 44 81 C 51 83 57 82 63 77.5 C 75 79 90 82 102 86 L 102 102 Z");
            juce::ColourGradient clothes (a.withMultipliedBrightness (0.45f).withMultipliedSaturation (0.9f), 70.0f, 80.0f,
                                          juce::Colour (0xff04081a), 20.0f, 100.0f, false);
            g.setGradientFill (clothes);
            g.fillPath (garment);
            g.setColour (b.withAlpha (0.35f));
            g.strokePath (svg ("M 44 81 C 51 83 57 82 63 77.5"), juce::PathStrokeType (0.6f));

            // bottom shade
            g.setGradientFill (juce::ColourGradient (juce::Colours::black.withAlpha (0.0f), 0.0f, 70.0f,
                                                     juce::Colours::black.withAlpha (0.55f), 0.0f, 100.0f, false));
            g.fillRect (juce::Rectangle<float> (-5.0f, 60.0f, 110.0f, 45.0f));
            g.restoreState();
        }

        void face (bool female)
        {
            const auto featureCol = dark.darker (0.4f).withAlpha (0.9f);
            const float dy = female ? 0.6f : 0.0f;
            const float dx = female ? -0.8f : 0.0f;
            auto t = juce::AffineTransform::translation (dx, dy);
            const auto round = [] (float w) { return juce::PathStrokeType (w, juce::PathStrokeType::curved, juce::PathStrokeType::rounded); };

            // eye socket shade, closed eye + lashes
            g.setGradientFill (juce::ColourGradient (dark.withAlpha (0.35f), 64.8f + dx, 37.0f + dy, dark.withAlpha (0.0f), 68.0f + dx, 37.0f + dy, true));
            g.fillEllipse (60.8f + dx, 34.2f + dy, 8.0f, 5.6f);
            g.setColour (featureCol);
            g.strokePath (svg ("M 63.0 37.3 C 64.2 38.8 65.9 38.9 67.0 37.6"), round (0.75f), t);
            if (female)
                for (float x : { 63.6f, 64.6f, 65.6f, 66.5f })
                    g.drawLine (x + dx, 38.3f + dy, x + dx - 0.5f, 39.4f + dy, 0.4f);

            // brow
            g.strokePath (svg (female ? "M 62.2 34.2 C 63.8 33.0 65.6 32.9 67.1 33.6" : "M 61.8 34.5 C 63.6 33.2 65.8 33.0 67.7 33.8"),
                          round (female ? 0.65f : 1.1f), t);

            // ear
            g.setColour (dark.withAlpha (0.6f));
            g.strokePath (svg ("M 48.6 38.6 C 51.8 37.8 53.2 40.8 52.7 43.4 C 52.2 46 50.6 47.2 48.9 46.6"), round (0.9f), t);
            g.setColour (lit.brighter (0.3f).withAlpha (0.5f));
            g.strokePath (svg ("M 49.4 39.6 C 51.4 39.4 52 41.4 51.6 43.2"), round (0.5f), t);

            // mouth line + lips sheen
            g.setColour (featureCol.withAlpha (0.7f));
            g.strokePath (svg (female ? "M 68.4 49.6 L 66.2 49.9" : "M 69.0 49.9 L 66.6 50.1"), round (0.45f), t);
            g.setColour (b.interpolatedWith (juce::Colours::white, 0.4f).withAlpha (female ? 0.6f : 0.3f));
            g.fillEllipse (juce::Rectangle<float> (67.6f + dx, 47.5f + dy, 1.5f, 0.8f));
            g.fillEllipse (juce::Rectangle<float> (67.3f + dx, 50.7f + dy, 1.5f, 0.9f));

            // nose-tip and cheekbone highlights
            g.setColour (juce::Colours::white.withAlpha (0.35f));
            g.fillEllipse (juce::Rectangle<float> ((female ? 70.2f : 71.6f) + dx, 42.8f + dy, 1.4f, 1.0f));
            g.setGradientFill (juce::ColourGradient (juce::Colours::white.withAlpha (0.18f), 62.0f + dx, 42.0f + dy,
                                                     juce::Colours::white.withAlpha (0.0f), 66.0f + dx, 42.0f + dy, true));
            g.fillEllipse (57.0f + dx, 38.0f + dy, 10.0f, 8.0f);
        }

        void hair (const juce::Path& p, bool female)
        {
            if (p.isEmpty())
                return;

            juce::ColourGradient cg (hairLit, 60.0f, 12.0f, hairDark, 28.0f, 58.0f, false);
            cg.addColour (0.45, hairLit.interpolatedWith (hairDark, 0.6f));
            g.setGradientFill (cg);
            g.fillPath (p);

            const bool curly = recipe.hair == Hair::Curly || recipe.hair == Hair::Afro;
            juce::Path silhouette (p), curlRim;

            // curls for voluminous styles
            if (curly)
            {
                juce::Path curls;
                std::vector<juce::Point<float>> centres;
                const float len = p.getLength();
                for (float d = 0.0f; d < len; d += 3.4f)
                {
                    const auto pt = p.getPointAlongPath (d);
                    if (pt.x > (female ? 60.0f : 62.0f) && pt.y > 18.0f)
                        continue;
                    const float r = 2.4f + rng.nextFloat() * 1.5f;
                    curls.addEllipse (pt.x - r, pt.y - r, r * 2.0f, r * 2.0f);
                    centres.push_back (pt);
                    curlRim.addCentredArc (pt.x, pt.y, r, r, 0.0f,
                                           std::atan2 (pt.x - 46.0f, -(pt.y - 34.0f)) - 1.2f,
                                           std::atan2 (pt.x - 46.0f, -(pt.y - 34.0f)) + 1.2f, true);
                }
                g.setGradientFill (cg);
                g.fillPath (curls);
                silhouette.addPath (curls);

                // tiny highlights give the curls texture
                g.saveState();
                g.reduceClipRegion (silhouette);
                for (int i = 0; i < 70; ++i)
                {
                    const auto bounds = p.getBounds();
                    const float x = bounds.getX() + rng.nextFloat() * bounds.getWidth();
                    const float y = bounds.getY() + rng.nextFloat() * bounds.getHeight();
                    juce::Path arc;
                    arc.addCentredArc (x, y, 1.8f, 1.8f, 0.0f, -2.4f, -0.4f, true);
                    g.setColour (b.withAlpha (0.10f + rng.nextFloat() * 0.18f));
                    g.strokePath (arc, juce::PathStrokeType (0.45f));
                }
                g.restoreState();
            }

            g.saveState();
            g.reduceClipRegion (silhouette);

            // glossy sheen near the crown
            g.setGradientFill (juce::ColourGradient (b.withAlpha (0.38f), female ? 54.0f : 55.0f, 15.0f, b.withAlpha (0.0f), female ? 68.0f : 70.0f, 24.0f, true));
            g.fillRect (silhouette.getBounds());

            // flowing strands, from the parting towards the back and down
            if (! curly)
            {
                const auto bounds = p.getBounds();
                for (int i = 0; i < 11; ++i)
                {
                    juce::Path s;
                    const float t = (float) i / 10.0f;
                    const float sx = 64.0f - t * 6.0f, sy = 16.0f + t * 4.0f;
                    s.startNewSubPath (sx, sy);
                    s.cubicTo (sx - 12.0f, sy - 6.0f + rng.nextFloat() * 3.0f,
                               bounds.getX() + 2.0f + rng.nextFloat() * 6.0f, sy + 8.0f + t * 10.0f,
                               bounds.getX() + 4.0f + rng.nextFloat() * 10.0f, bounds.getBottom() - rng.nextFloat() * 10.0f);
                    g.setColour ((i % 3 == 0 ? b : hairLit.brighter (0.4f)).withAlpha (0.10f + rng.nextFloat() * 0.12f));
                    g.strokePath (s, juce::PathStrokeType (0.35f + rng.nextFloat() * 0.35f));
                }
            }
            g.restoreState();

            rim (curly ? curlRim : silhouette, 0.7f, 0.9f);
        }

        void beard (bool full)
        {
            auto p = svg ("M 51.5 46 C 53 51.5 56 55.5 60 56.6 C 63.2 57.4 66 56.6 68.4 55.5 C 69 57.5 68.8 60 66.5 62 C 62 65.5 56 66 52 63.5 C 49.5 61.5 48.5 55 51.5 46 Z");
            g.setGradientFill (juce::ColourGradient (hairLit.brighter (0.2f).withAlpha (full ? 1.0f : 0.5f), 66.0f, 52.0f,
                                                     hairDark.withAlpha (full ? 1.0f : 0.4f), 52.0f, 62.0f, false));
            g.fillPath (p);

            // stubble / beard texture
            g.saveState();
            g.reduceClipRegion (p);
            for (int i = 0; i < (full ? 60 : 40); ++i)
            {
                const float x = 50.0f + rng.nextFloat() * 20.0f, y = 46.0f + rng.nextFloat() * 20.0f;
                g.setColour (b.withAlpha (0.10f + rng.nextFloat() * 0.15f));
                g.fillEllipse (x, y, 0.5f, 0.5f);
            }
            g.restoreState();

            if (full)
                rim (p, 0.5f, 0.7f);
        }

        void moustache()
        {
            auto p = svg ("M 69.8 46.6 C 70.4 47.4 70 48.2 68.9 48.5 C 67.2 48.8 65.6 48.2 64.6 47.2 C 66.4 46.6 68.2 46.3 69.8 46.6 Z");
            g.setColour (hairDark);
            g.fillPath (p);
        }

        void hat (Hat h, bool female)
        {
            if (h == Hat::None)
                return;

            g.saveState();
            if (female)
                g.addTransform (juce::AffineTransform::translation (-0.6f, 2.2f));

            auto crown = hatPath (h);
            auto brim = hatBrim (h);
            auto band = hatBand (h);

            juce::Colour base = a.withMultipliedBrightness (0.55f);
            if (h == Hat::Cowboy)
                base = juce::Colour (0xffc8a06a).interpolatedWith (a, 0.25f).withMultipliedBrightness (0.85f);
            else if (h == Hat::Fedora || h == Hat::TopHat)
                base = juce::Colour (0xff1a1f33).interpolatedWith (a, 0.15f);
            else if (h == Hat::Hood)
                base = juce::Colour (0xff1b2340).interpolatedWith (a, 0.25f);

            juce::ColourGradient cg (base.brighter (0.35f), 64.0f, 6.0f, base.darker (0.9f), 30.0f, 30.0f, false);

            if (h == Hat::Tam)
            {
                g.saveState();
                g.reduceClipRegion (crown);
                const juce::Colour stripes[] = { juce::Colour (0xff2ecc71), juce::Colour (0xfff7dc6f), juce::Colour (0xffe74c3c), juce::Colour (0xff111111) };
                for (int i = 0; i < 4; ++i)
                {
                    g.setColour (stripes[i].withMultipliedBrightness (0.75f));
                    g.fillRect (juce::Rectangle<float> (0.0f, 24.0f - (float) i * 6.5f, 100.0f, 6.6f));
                }
                g.setGradientFill (juce::ColourGradient (juce::Colours::white.withAlpha (0.25f), 64.0f, 6.0f, juce::Colours::black.withAlpha (0.4f), 20.0f, 30.0f, false));
                g.fillPath (crown);
                g.restoreState();
            }
            else if (h == Hat::Headwrap)
            {
                juce::ColourGradient wrap (b.withMultipliedBrightness (0.9f), 64.0f, 8.0f, a.withMultipliedBrightness (0.5f), 24.0f, 30.0f, false);
                g.setGradientFill (wrap);
                g.fillPath (crown);
                g.saveState();
                g.reduceClipRegion (crown);
                g.setColour (juce::Colours::black.withAlpha (0.25f));
                for (int i = 0; i < 5; ++i)
                {
                    juce::Path fold;
                    fold.startNewSubPath (22.0f + (float) i * 3.0f, 32.0f - (float) i * 5.0f);
                    fold.quadraticTo (44.0f, 18.0f - (float) i * 3.0f, 66.0f, 22.0f - (float) i * 4.0f);
                    g.strokePath (fold, juce::PathStrokeType (0.9f));
                }
                g.restoreState();
            }
            else
            {
                g.setGradientFill (cg);
                g.fillPath (crown);
            }

            if (h == Hat::Cap)
            {
                g.setColour (b.withAlpha (0.5f));
                g.fillEllipse (44.8f, 7.4f, 2.4f, 2.0f);
                g.strokePath (svg ("M 46 9 C 44 16 44 22 45 27.5"), juce::PathStrokeType (0.4f));
            }

            if (! band.isEmpty())
            {
                g.setColour (h == Hat::Cowboy ? juce::Colour (0xff3a2615) : a.withMultipliedBrightness (0.8f));
                g.fillPath (band);
            }

            if (! brim.isEmpty())
            {
                juce::ColourGradient bg (base.brighter (0.25f), 70.0f, 18.0f, base.darker (1.0f), 20.0f, 28.0f, false);
                g.setGradientFill (bg);
                g.fillPath (brim);
                rim (brim, 0.6f, 0.9f);
            }

            rim (crown, 0.7f, 0.9f);
            g.restoreState();
        }

        void headphones (bool female)
        {
            g.saveState();
            if (female)
                g.addTransform (juce::AffineTransform::translation (-0.6f, 2.0f));

            auto band = svg ("M 45.5 32 C 44 22 44.6 13.5 47.5 7.2");
            g.setColour (juce::Colour (0xff0b1020));
            g.strokePath (band, juce::PathStrokeType (4.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            g.setColour (juce::Colour (0xff8a95b5));
            g.strokePath (band, juce::PathStrokeType (1.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            auto cup = juce::Rectangle<float> (38.5f, 31.0f, 13.5f, 17.5f);
            g.setGradientFill (juce::ColourGradient (juce::Colour (0xff3a4566), cup.getRight(), cup.getY(), juce::Colour (0xff070a14), cup.getX(), cup.getBottom(), false));
            g.fillRoundedRectangle (cup, 6.0f);
            g.setColour (b.withAlpha (0.25f));
            g.fillEllipse (cup.reduced (3.0f, 4.0f).expanded (1.2f));
            g.setColour (b);
            g.drawEllipse (cup.reduced (3.0f, 4.0f), 0.9f);
            juce::Path cupPath;
            cupPath.addRoundedRectangle (cup, 6.0f);
            rim (cupPath, 0.6f, 0.8f);
            g.restoreState();
        }

        void sunglasses()
        {
            auto lens = svg ("M 60.5 33 C 64 32.4 67.5 32.6 69.2 33.6 C 69.6 36 68.6 39.2 66 39.6 C 63 40 60.8 38.2 60.4 35.6 Z");
            g.setColour (juce::Colour (0xff05070f));
            g.strokePath (svg ("M 61 34.4 L 48.5 36.4"), juce::PathStrokeType (1.1f));
            g.setGradientFill (juce::ColourGradient (b.withAlpha (0.9f), 68.0f, 33.0f, juce::Colour (0xff05070f), 62.0f, 39.0f, false));
            g.fillPath (lens);
            g.setColour (juce::Colours::white.withAlpha (0.7f));
            g.strokePath (svg ("M 66.5 34 L 68.4 34.3"), juce::PathStrokeType (0.5f));
        }

        void jewellery (bool female)
        {
            const float dx = female ? -0.8f : 0.0f, dy = female ? 0.6f : 0.0f;
            const juce::Colour gold (0xffffd27a);

            if (recipe.hoops)
            {
                g.setColour (gold.withAlpha (0.35f));
                g.drawEllipse (juce::Rectangle<float> (47.6f + dx, 46.2f + dy, 5.2f, 7.6f), 1.8f);
                g.setColour (gold);
                g.drawEllipse (juce::Rectangle<float> (47.6f + dx, 46.2f + dy, 5.2f, 7.6f), 0.8f);
            }

            if (recipe.stud)
            {
                g.setColour (juce::Colours::white.withAlpha (0.35f));
                g.fillEllipse (48.4f + dx, 45.8f + dy, 2.6f, 2.6f);
                g.setColour (juce::Colours::white);
                g.fillEllipse (49.0f + dx, 46.4f + dy, 1.4f, 1.4f);
            }

            if (recipe.chain)
            {
                juce::Path chainPath = svg ("M 41 79 C 45 88 54 90.5 61 81");
                const float len = chainPath.getLength();
                for (float d = 0.0f; d < len; d += 1.3f)
                {
                    const auto pt = chainPath.getPointAlongPath (d);
                    g.setColour (gold.withAlpha (0.9f));
                    g.fillEllipse (pt.x - 0.55f, pt.y - 0.55f, 1.1f, 1.1f);
                }
                g.setColour (gold);
                g.fillEllipse (49.0f, 87.0f, 3.2f, 3.2f);
            }

            if (recipe.bowTie)
            {
                juce::Path bow = svg ("M 52.5 73.8 L 58.2 76 L 52.5 78.2 Z M 63.8 73.8 L 58.2 76 L 63.8 78.2 Z");
                g.setColour (juce::Colour (0xff06070d));
                g.fillPath (bow);
                g.setColour (b.withAlpha (0.8f));
                g.strokePath (bow, juce::PathStrokeType (0.4f));
                g.fillEllipse (57.2f, 75.0f, 2.0f, 2.0f);
            }

            if (recipe.stole)
            {
                auto stole = svg ("M 44 101 L 56.5 75.5 L 63.5 78.5 L 57.5 101 Z");
                g.setGradientFill (juce::ColourGradient (gold.withMultipliedBrightness (0.95f), 62.0f, 78.0f, gold.withMultipliedBrightness (0.35f), 46.0f, 100.0f, false));
                g.fillPath (stole);
                g.setColour (juce::Colours::white.withAlpha (0.4f));
                g.strokePath (svg ("M 58.5 79 L 51 100"), juce::PathStrokeType (0.4f));
            }

            if (recipe.jacket)
            {
                auto lapel = svg ("M 56.5 75 C 60.5 80 63 88 64 101 L 73 101 C 71 90 66.5 82 60.5 76 Z");
                g.setGradientFill (juce::ColourGradient (juce::Colour (0xff2a3048), 70.0f, 80.0f, juce::Colour (0xff05060c), 60.0f, 100.0f, false));
                g.fillPath (lapel);
                rim (lapel, 0.45f, 0.6f);
            }

            if (recipe.flower)
            {
                const juce::Point<float> c (female ? 38.0f : 40.0f, female ? 22.0f : 20.0f);
                for (int i = 0; i < 5; ++i)
                {
                    const float ang = juce::MathConstants<float>::twoPi * (float) i / 5.0f;
                    g.setColour (b.interpolatedWith (juce::Colour (0xffff9ccf), 0.5f));
                    g.fillEllipse (c.x + 2.0f * std::cos (ang) - 1.6f, c.y + 2.0f * std::sin (ang) - 1.6f, 3.2f, 3.2f);
                }
                g.setColour (juce::Colour (0xffffe08a));
                g.fillEllipse (c.x - 1.0f, c.y - 1.0f, 2.0f, 2.0f);
            }

            if (recipe.tiara)
            {
                auto tiara = svg ("M 40 16.5 L 42.4 12.8 L 44.6 15.2 L 47.4 10.6 L 50.2 15 L 52.4 12.4 L 54.6 16 Z");
                g.setColour (juce::Colours::white.withAlpha (0.3f));
                g.strokePath (tiara, juce::PathStrokeType (1.6f));
                g.setColour (juce::Colour (0xfffff1c4));
                g.fillPath (tiara);
            }

            if (recipe.halo)
            {
                auto halo = juce::Rectangle<float> (33.0f, 3.0f, 30.0f, 6.0f);
                for (auto [w, al] : { std::pair { 3.5f, 0.15f }, { 2.0f, 0.3f }, { 0.8f, 1.0f } })
                {
                    g.setColour (juce::Colour (0xfffff3c8).withAlpha (al));
                    g.drawEllipse (halo, w);
                }
            }
        }

        void dreads()
        {
            g.setColour (hairDark);
            for (auto [x0, y0, x1, y1] : { std::tuple { 30.0f, 28.0f, 22.0f, 78.0f }, { 34.0f, 32.0f, 28.0f, 84.0f },
                                           { 39.0f, 36.0f, 35.0f, 82.0f }, { 27.0f, 24.0f, 17.0f, 70.0f }, { 43.0f, 38.0f, 41.0f, 76.0f } })
            {
                juce::Path s;
                s.startNewSubPath (x0, y0);
                s.quadraticTo ((x0 + x1) * 0.5f - 3.0f, (y0 + y1) * 0.5f, x1, y1);
                g.setColour (hairDark);
                g.strokePath (s, juce::PathStrokeType (3.4f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
                g.setColour (b.withAlpha (0.25f));
                g.strokePath (s, juce::PathStrokeType (0.5f));
            }
        }

        //==============================================================================
        void person (bool female, float brightness)
        {
            auto body = svg (female ? femalePath : malePath);

            if (recipe.hair == Hair::Dreads)
                dreads();

            fillFigure (body, brightness);
            face (female);

            if (recipe.beard || recipe.stubble)
                beard (recipe.beard);
            if (recipe.moustache)
                moustache();

            rim (body, 0.8f, brightness);

            jewellery (female);
            hair (hairPath (recipe.hair, female), female);

            if (recipe.bangs && female)
            {
                auto bangs = svg ("M 64.5 22 C 67 25 68 29.5 67.2 32.5 C 64 30 60 29.5 57.5 31.5 C 59 27 61.5 23.8 64.5 22 Z");
                g.setGradientFill (juce::ColourGradient (hairLit, 64.0f, 20.0f, hairDark, 58.0f, 32.0f, false));
                g.fillPath (bangs);
            }

            if (recipe.sunglasses)
                sunglasses();
            if (recipe.headphones)
                headphones (female);

            hat (recipe.hat, female);
        }

        void robot()
        {
            auto shoulders = svg ("M 4 101 C 6 82 22 73 50 73 C 78 73 94 82 96 101 Z");
            auto neck = juce::Rectangle<float> (42.0f, 55.0f, 16.0f, 20.0f);
            auto head = juce::Rectangle<float> (27.0f, 14.0f, 46.0f, 44.0f);
            auto metal = [this] (juce::Rectangle<float> r)
            {
                juce::ColourGradient cg (juce::Colour (0xffc9d4ea), r.getX(), r.getY(), juce::Colour (0xff1d2540), r.getRight(), r.getBottom(), false);
                cg.addColour (0.4, a.interpolatedWith (juce::Colour (0xff6c7898), 0.6f));
                return cg;
            };

            g.setGradientFill (metal ({ 4.0f, 73.0f, 92.0f, 28.0f }));
            g.fillPath (shoulders);
            g.setGradientFill (metal (neck));
            g.fillRect (neck);
            g.setColour (juce::Colours::black.withAlpha (0.4f));
            for (int i = 0; i < 4; ++i)
                g.fillRect (neck.getX(), neck.getY() + 4.0f + (float) i * 4.0f, neck.getWidth(), 1.0f);

            // antenna
            g.setColour (juce::Colour (0xff9aa6c4));
            g.drawLine (50.0f, 14.0f, 50.0f, 5.0f, 1.4f);
            g.setColour (b.withAlpha (0.4f));
            g.fillEllipse (46.5f, 0.5f, 7.0f, 7.0f);
            g.setColour (b);
            g.fillEllipse (48.0f, 2.0f, 4.0f, 4.0f);

            // ears
            g.setGradientFill (metal ({ 22.0f, 28.0f, 56.0f, 16.0f }));
            g.fillRoundedRectangle (22.0f, 28.0f, 6.0f, 15.0f, 2.0f);
            g.fillRoundedRectangle (72.0f, 28.0f, 6.0f, 15.0f, 2.0f);

            g.setGradientFill (metal (head));
            g.fillRoundedRectangle (head, 13.0f);

            // visor
            auto visor = juce::Rectangle<float> (32.0f, 27.0f, 36.0f, 12.0f);
            g.setColour (juce::Colour (0xff03050c));
            g.fillRoundedRectangle (visor, 6.0f);
            g.setColour (b.withAlpha (0.35f));
            g.fillRoundedRectangle (visor.reduced (2.0f, 3.5f).expanded (2.0f, 1.5f), 4.0f);
            g.setColour (b);
            g.fillRoundedRectangle (visor.reduced (3.5f, 4.8f), 2.0f);

            // mouth grille
            g.setColour (juce::Colours::black.withAlpha (0.45f));
            for (int i = 0; i < 5; ++i)
                g.fillRoundedRectangle (40.0f + (float) i * 4.4f, 46.0f, 2.2f, 6.0f, 1.0f);

            juce::Path outline;
            outline.addRoundedRectangle (head, 13.0f);
            rim (outline, 0.8f);
            rim (shoulders, 0.8f, 0.7f);
        }

        void choir()
        {
            struct Member { float x, y, s, bright; bool female; };
            const Member members[] = { { -18.0f, 16.0f, 0.78f, 0.55f, true }, { 36.0f, 18.0f, 0.74f, 0.5f, false },
                                       { 6.0f, 10.0f, 0.9f, 1.0f, true } };
            for (auto& m : members)
            {
                g.saveState();
                g.addTransform (juce::AffineTransform::scale (m.s).translated (m.x, m.y));
                recipe.hair = m.female ? Hair::Curly : Hair::Short;
                setPerson ((int) (&m - members) * 3 % 7, (int) (&m - members) % 4, m.female);
                person (m.female, m.bright);
                g.restoreState();
            }
        }

        //==============================================================================
        void background (float w, float h)
        {
            g.setGradientFill (juce::ColourGradient (juce::Colour (0xff0b1434), 0.0f, 0.0f, juce::Colour (0xff04071a), 0.0f, h, false));
            g.fillAll();

            auto blob = [this] (float cx, float cy, float r, juce::Colour c, float alpha)
            {
                g.setGradientFill (juce::ColourGradient (c.withAlpha (alpha), cx, cy, c.withAlpha (0.0f), cx + r, cy, true));
                g.fillEllipse (cx - r, cy - r, r * 2.0f, r * 2.0f);
            };

            blob (w * (0.25f + rng.nextFloat() * 0.5f), h * (0.2f + rng.nextFloat() * 0.25f), w * 0.75f, a, 0.55f);
            blob (w * rng.nextFloat(), h * rng.nextFloat() * 0.6f, w * 0.5f, b, 0.28f);
            blob (w * (rng.nextBool() ? 0.05f : 0.95f), h * (0.1f + rng.nextFloat() * 0.5f), w * 0.55f, juce::Colour (0xff5b3cff), 0.30f);
            blob (w * 0.5f, h * 0.35f, w * 0.35f, b, 0.18f);

            for (int i = 0; i < 46; ++i)
            {
                const float x = rng.nextFloat() * w, y = rng.nextFloat() * h * 0.85f;
                const float s = (0.4f + rng.nextFloat() * rng.nextFloat() * 1.6f) * w / 120.0f;
                g.setColour (juce::Colours::white.withAlpha (0.25f + rng.nextFloat() * 0.65f));
                g.fillEllipse (x - s * 0.5f, y - s * 0.5f, s, s);

                if (i % 11 == 0)
                {
                    g.setColour (b.withAlpha (0.5f));
                    g.fillRect (juce::Rectangle<float> (x - s * 2.5f, y - 0.25f * s, s * 5.0f, 0.5f * s));
                    g.fillRect (juce::Rectangle<float> (x - 0.25f * s, y - s * 2.5f, 0.5f * s, s * 5.0f));
                }
            }
        }
    };
}

//==============================================================================
juce::Image getFactoryPortrait (const VoiceProfile& voice)
{
    if (! voice.isFactory || voice.id.isEmpty())
        return {};

    // BinaryData mangles resource names ("female-pop.jpg" -> "femalepop_jpg"), so look the photo up by its original filename
    const auto wanted = voice.id + ".jpg";

    for (int i = 0; i < VoxaAssets::namedResourceListSize; ++i)
    {
        if (wanted != VoxaAssets::originalFilenames[i])
            continue;

        int size = 0;
        if (const auto* data = VoxaAssets::getNamedResource (VoxaAssets::namedResourceList[i], size); data != nullptr && size > 0)
            return juce::ImageCache::getFromMemory (data, size);
    }

    return {};
}

namespace
{
    /** Draws a photo so it covers the area, halving it first while it is more than twice the target size
        so the final resample does not alias. */
    void drawPhoto (juce::Graphics& g, juce::Image photo, int w, int h, juce::RectanglePlacement placement)
    {
        while (photo.getWidth() / 2 >= w && photo.getHeight() / 2 >= h)
            photo = photo.rescaled (photo.getWidth() / 2, photo.getHeight() / 2, juce::Graphics::highResamplingQuality);

        g.drawImage (photo, juce::Rectangle<float> (0.0f, 0.0f, (float) w, (float) h), placement);
    }
}

juce::Image renderVoiceAvatar (const VoiceProfile& voice, int w, int h)
{
    w = juce::jmax (8, w);
    h = juce::jmax (8, h);
    juce::Image img (juce::Image::ARGB, w, h, true);
    juce::Graphics g (img);
    g.setImageResamplingQuality (juce::Graphics::highResamplingQuality);

    juce::Image custom;
    if (voice.avatarImage.existsAsFile())
        custom = juce::ImageFileFormat::loadFrom (voice.avatarImage);

    const auto portrait = custom.isValid() ? juce::Image() : getFactoryPortrait (voice);

    Painter p (g, voice);

    if (custom.isValid())
    {
        drawPhoto (g, custom, w, h, juce::RectanglePlacement::fillDestination);
    }
    else if (portrait.isValid())
    {
        // anchored at the top so a squarer card never crops the head
        drawPhoto (g, portrait, w, h, juce::RectanglePlacement::xMid | juce::RectanglePlacement::yTop
                                          | juce::RectanglePlacement::fillDestination);
    }
    else
    {
        p.background ((float) w, (float) h);

        // spotlight behind the head
        g.setGradientFill (juce::ColourGradient (voice.accentB.withAlpha (0.28f), (float) w * 0.5f, (float) h * 0.38f,
                                                 voice.accentB.withAlpha (0.0f), (float) w * 0.5f + (float) h * 0.45f, (float) h * 0.38f, true));
        g.fillRect (0, 0, w, h);

        const float s = (float) h / 100.0f * 1.02f;
        const bool mirrored = p.recipe.figure == Figure::Female;
        auto t = juce::AffineTransform::scale (s).translated (((float) w - 100.0f * s) * 0.5f, (float) h * 0.02f);
        if (mirrored)
            t = juce::AffineTransform::scale (-1.0f, 1.0f).translated (100.0f, 0.0f).followedBy (t);

        g.saveState();
        g.addTransform (t);

        switch (p.recipe.figure)
        {
            case Figure::Robot:  p.robot(); break;
            case Figure::Choir:  p.choir(); break;
            case Figure::Female: p.person (true, 1.0f); break;
            case Figure::Male:
            default:             p.person (false, 1.0f); break;
        }

        g.restoreState();
    }

    // name vignette
    g.setGradientFill (juce::ColourGradient (juce::Colours::black.withAlpha (0.0f), 0.0f, (float) h * 0.58f,
                                             juce::Colour (0xe6030616), 0.0f, (float) h, false));
    g.fillRect (0, h / 2, w, h - h / 2);

    return img;
}

const juce::Image& AvatarCache::get (const VoiceProfile& voice, int w, int h)
{
    const auto key = voice.id + "@" + juce::String (w) + "x" + juce::String (h);
    auto it = images.find (key);
    if (it == images.end())
    {
        if (images.size() > 200)
            images.clear();
        it = images.emplace (key, renderVoiceAvatar (voice, w, h)).first;
    }
    return it->second;
}

} // namespace voxa::ui
