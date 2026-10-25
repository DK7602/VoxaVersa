#include "VoiceLibrary.h"

namespace voxa
{

//==============================================================================
VoiceTraits VoiceTraits::blend (const VoiceTraits& a, const VoiceTraits& b, float t) noexcept
{
    t = juce::jlimit (0.0f, 1.0f, t);
    auto lerp = [t] (float x, float y) { return x + (y - x) * t; };
    auto lerpi = [t] (int x, int y) { return juce::roundToInt ((float) x + (float) (y - x) * t); };

    VoiceTraits r = t < 0.5f ? a : b;
    r.rangeLowMidi = lerpi (a.rangeLowMidi, b.rangeLowMidi);
    r.rangeHighMidi = lerpi (a.rangeHighMidi, b.rangeHighMidi);
    r.formantSemis = lerp (a.formantSemis, b.formantSemis);
    r.tiltDb = lerp (a.tiltDb, b.tiltDb);
    r.warmthDb = lerp (a.warmthDb, b.warmthDb);
    r.presenceDb = lerp (a.presenceDb, b.presenceDb);
    r.airDb = lerp (a.airDb, b.airDb);
    r.nasalDb = lerp (a.nasalDb, b.nasalDb);
    r.ringHz = std::exp (lerp (std::log (a.ringHz), std::log (b.ringHz)));
    r.ringDb = lerp (a.ringDb, b.ringDb);
    r.bodyHz = std::exp (lerp (std::log (a.bodyHz), std::log (b.bodyHz)));
    r.bodyDb = lerp (a.bodyDb, b.bodyDb);
    r.breathiness = lerp (a.breathiness, b.breathiness);
    r.rasp = lerp (a.rasp, b.rasp);
    r.vibratoRateHz = lerp (a.vibratoRateHz, b.vibratoRateHz);
    r.vibratoDepthCents = lerp (a.vibratoDepthCents, b.vibratoDepthCents);
    r.vibratoOnsetMs = lerp (a.vibratoOnsetMs, b.vibratoOnsetMs);
    r.scoopCents = lerp (a.scoopCents, b.scoopCents);
    r.scoopMs = lerp (a.scoopMs, b.scoopMs);
    r.jitterCents = lerp (a.jitterCents, b.jitterCents);
    r.vibeCompressor = lerp (a.vibeCompressor, b.vibeCompressor);
    r.vibeDeEsser = lerp (a.vibeDeEsser, b.vibeDeEsser);
    r.vibeSaturation = lerp (a.vibeSaturation, b.vibeSaturation);
    r.vibeReverbMix = lerp (a.vibeReverbMix, b.vibeReverbMix);
    r.vibeReverbSize = lerp (a.vibeReverbSize, b.vibeReverbSize);
    r.vibeDelayMix = lerp (a.vibeDelayMix, b.vibeDelayMix);
    r.vibeDoubler = lerp (a.vibeDoubler, b.vibeDoubler);
    r.vibeLowCutHz = lerp (a.vibeLowCutHz, b.vibeLowCutHz);
    r.singLegato = lerp (a.singLegato, b.singLegato);
    r.singRhythmBusy = lerp (a.singRhythmBusy, b.singRhythmBusy);
    r.singPortamentoMs = lerp (a.singPortamentoMs, b.singPortamentoMs);
    return r;
}

juce::String VoiceProfile::getGenderName() const
{
    switch (traits.gender)
    {
        case VoiceGender::Male:   return "Male";
        case VoiceGender::Female: return "Female";
        case VoiceGender::Neutral:
        default:                  return "Neutral";
    }
}

//==============================================================================
VoiceLibrary::VoiceLibrary()
{
    voices = createFactoryVoices();
    numFactoryVoices = voices.size();
    loadFavourites();
    rescanUserVoices();
}

const VoiceProfile* VoiceLibrary::findById (const juce::String& id) const
{
    for (auto& v : voices)
        if (v.id == id)
            return &v;
    return nullptr;
}

int VoiceLibrary::indexOf (const juce::String& id) const
{
    for (size_t i = 0; i < voices.size(); ++i)
        if (voices[i].id == id)
            return (int) i;
    return -1;
}

const VoiceProfile& VoiceLibrary::getDefaultVoice() const
{
    if (auto* v = findById ("female-pop"))
        return *v;
    return voices.front();
}

juce::StringArray VoiceLibrary::getGenres() const
{
    juce::StringArray result;
    for (auto& v : voices)
        result.addIfNotAlreadyThere (v.genre);
    return result;   // keep library (curated) order rather than alphabetical
}

juce::StringArray VoiceLibrary::getStyles() const
{
    juce::StringArray result;
    for (auto& v : voices)
        for (auto& s : v.styles)
            result.addIfNotAlreadyThere (s);
    result.sort (true);
    return result;
}

std::vector<int> VoiceLibrary::filter (const Filter& f) const
{
    std::vector<int> result;
    const auto search = f.searchText.trim().toLowerCase();
    const auto searchTerms = juce::StringArray::fromTokens (search, " ", "\"");

    for (size_t i = 0; i < voices.size(); ++i)
    {
        const auto& v = voices[i];

        if (f.gender == Filter::Gender::Male && v.traits.gender != VoiceGender::Male)     continue;
        if (f.gender == Filter::Gender::Female && v.traits.gender != VoiceGender::Female) continue;
        if (f.genre.isNotEmpty() && ! v.genre.equalsIgnoreCase (f.genre))                 continue;
        if (f.style.isNotEmpty() && ! v.styles.contains (f.style, true))                  continue;
        if (f.favouritesOnly && ! isFavourite (v.id))                                     continue;

        if (search.isNotEmpty())
        {
            const auto haystack = (v.name + " " + v.genre + " " + v.tagline + " " + v.styles.joinIntoString (" ")
                                   + " " + v.getGenderName()).toLowerCase();
            bool all = true;
            for (auto& term : searchTerms)
                if (term.isNotEmpty() && ! haystack.contains (term))
                    all = false;
            if (! all)
                continue;
        }

        result.push_back ((int) i);
    }
    return result;
}

//==============================================================================
static juce::File getSettingsFile()
{
    return VoiceLibrary::getUserDataDirectory().getChildFile ("settings.xml");
}

bool VoiceLibrary::isFavourite (const juce::String& id) const
{
    return favourites.contains (id);
}

void VoiceLibrary::setFavourite (const juce::String& id, bool shouldBeFavourite)
{
    if (shouldBeFavourite == isFavourite (id))
        return;

    if (shouldBeFavourite)
        favourites.add (id);
    else
        favourites.removeString (id);

    saveFavourites();
    sendChangeMessage();
}

void VoiceLibrary::loadFavourites()
{
    favourites.clear();
    if (auto xml = juce::parseXML (getSettingsFile()))
        if (auto* fav = xml->getChildByName ("Favourites"))
            favourites.addTokens (fav->getStringAttribute ("ids"), ",", {});
    favourites.removeEmptyStrings();
}

void VoiceLibrary::saveFavourites() const
{
    auto file = getSettingsFile();
    std::unique_ptr<juce::XmlElement> xml (juce::parseXML (file));
    if (xml == nullptr || ! xml->hasTagName ("VoxaVersaSettings"))
        xml = std::make_unique<juce::XmlElement> ("VoxaVersaSettings");

    auto* fav = xml->getChildByName ("Favourites");
    if (fav == nullptr)
        fav = xml->createNewChildElement ("Favourites");
    fav->setAttribute ("ids", favourites.joinIntoString (","));

    file.getParentDirectory().createDirectory();
    xml->writeTo (file);
}

//==============================================================================
juce::File VoiceLibrary::getUserDataDirectory()
{
    return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
#if JUCE_MAC
        .getChildFile ("Application Support")
#endif
        .getChildFile ("VoxaVersa");
}

juce::File VoiceLibrary::getUserVoicesDirectory()  { return getUserDataDirectory().getChildFile ("Voices"); }
juce::File VoiceLibrary::getEncodersDirectory()    { return getUserDataDirectory().getChildFile ("Encoders"); }

static VoiceGender parseGender (const juce::String& s)
{
    if (s.equalsIgnoreCase ("male"))   return VoiceGender::Male;
    if (s.equalsIgnoreCase ("female")) return VoiceGender::Female;
    return VoiceGender::Neutral;
}

bool VoiceLibrary::loadVoicePack (const juce::File& directory, VoiceProfile& p, juce::String& error)
{
    const auto manifestFile = directory.getChildFile ("voice.json");
    if (! manifestFile.existsAsFile())
    {
        error = "Missing voice.json in " + directory.getFullPathName();
        return false;
    }

    const auto json = juce::JSON::parse (manifestFile);
    auto* obj = json.getDynamicObject();
    if (obj == nullptr)
    {
        error = "voice.json is not a valid JSON object";
        return false;
    }

    auto get = [obj] (const char* key, const juce::var& def = {}) { return obj->hasProperty (key) ? obj->getProperty (key) : def; };

    p = VoiceProfile();
    p.isFactory = false;
    p.packDirectory = directory;
    p.name = get ("name", directory.getFileName()).toString();
    p.id = "user-" + get ("id", p.name).toString().toLowerCase().replaceCharacters (" _/\\", "----");
    p.tagline = get ("description", "User voice pack").toString();
    p.genre = get ("genre", "Custom").toString();
    p.author = get ("author", "").toString();
    p.license = get ("license", "").toString();

    if (auto* tags = get ("styles").getArray())
        for (auto& t : *tags)
            p.styles.add (t.toString());

    auto& t = p.traits;
    t.gender = parseGender (get ("gender", "neutral").toString());
    const bool male = t.gender == VoiceGender::Male;
    t.formantSemis = (float) (double) get ("formant", male ? -1.0 : (t.gender == VoiceGender::Female ? 1.0 : 0.0));

    if (auto* range = get ("range").getArray(); range != nullptr && range->size() == 2)
    {
        t.rangeLowMidi = (int) (*range)[0];
        t.rangeHighMidi = (int) (*range)[1];
    }
    else
    {
        t.rangeLowMidi = male ? 45 : 55;
        t.rangeHighMidi = male ? 69 : 79;
    }

    t.breathiness = (float) (double) get ("breathiness", 0.1);
    t.rasp = (float) (double) get ("rasp", 0.0);
    t.vibratoDepthCents = (float) (double) get ("vibratoDepth", 20.0);
    t.vibratoRateHz = (float) (double) get ("vibratoRate", 5.5);

    if (auto colour = get ("colour").toString(); colour.isNotEmpty())
        p.accentA = juce::Colour::fromString (colour.startsWith ("#") ? "ff" + colour.substring (1) : colour);
    p.accentB = p.accentA.brighter (0.8f);

    if (auto avatar = get ("avatar").toString(); avatar.isNotEmpty())
        p.avatarImage = directory.getChildFile (avatar);

    if (auto* ids = get ("factoryVoices").getArray())
        for (auto& id : *ids)
            p.factoryVoices.addIfNotAlreadyThere (id.toString().trim());
    p.listed = (bool) get ("listed", true);

    if (auto* model = obj->getProperty ("model").getDynamicObject())
    {
        p.neuralModelFile = directory.getChildFile (model->getProperty ("file").toString());
        p.neuralEncoder = model->getProperty ("encoder").toString();
        p.neuralSampleRate = (int) model->getProperty ("sampleRate");
        if (! p.neuralModelFile.existsAsFile())
        {
            error = "Model file not found: " + p.neuralModelFile.getFullPathName();
            return false;
        }
    }

    return true;
}

void VoiceLibrary::rescanUserVoices (const juce::File& dir)
{
    voices.resize (numFactoryVoices);
    for (auto& v : voices)
    {
        v.packDirectory = juce::File();
        v.neuralModelFile = juce::File();
        v.neuralEncoder.clear();
        v.neuralSampleRate = 0;
        v.aiPackName.clear();
    }

    if (dir.isDirectory())
    {
        auto packs = dir.findChildFiles (juce::File::findDirectories, false);
        packs.sort();   // deterministic: when two packs claim a built-in voice, the first by name wins

        for (const auto& packDir : packs)
        {
            VoiceProfile p;
            juce::String error;
            if (! loadVoicePack (packDir, p, error))
            {
                DBG ("VoxaVersa: skipping voice pack: " << error);
                continue;
            }

            if (p.hasNeuralModel())
            {
                for (const auto& id : p.factoryVoices)
                {
                    const int index = indexOf (id);
                    if (index < 0 || index >= (int) numFactoryVoices || voices[(size_t) index].hasNeuralModel())
                        continue;
                    auto& v = voices[(size_t) index];
                    v.packDirectory = p.packDirectory;
                    v.neuralModelFile = p.neuralModelFile;
                    v.neuralEncoder = p.neuralEncoder;
                    v.neuralSampleRate = p.neuralSampleRate;
                    v.aiPackName = p.name;
                }
            }

            if (p.listed && indexOf (p.id) < 0)
                voices.push_back (std::move (p));
        }
    }

    sendChangeMessage();
}

bool VoiceLibrary::importVoicePack (const juce::File& source, juce::String& error)
{
    const auto voicesDir = getUserVoicesDirectory();
    if (! voicesDir.createDirectory())
    {
        error = "Could not create " + voicesDir.getFullPathName();
        return false;
    }

    juce::File installed;

    if (source.isDirectory())
    {
        installed = voicesDir.getChildFile (source.getFileName());
        if (installed.exists())
            installed.deleteRecursively();
        if (! source.copyDirectoryTo (installed))
        {
            error = "Could not copy voice pack";
            return false;
        }
    }
    else
    {
        juce::ZipFile zip (source);
        if (zip.getNumEntries() == 0)
        {
            error = "Not a valid voice pack archive";
            return false;
        }

        installed = voicesDir.getChildFile (source.getFileNameWithoutExtension());
        if (installed.exists())
            installed.deleteRecursively();

        if (auto result = zip.uncompressTo (installed, true); result.failed())
        {
            error = result.getErrorMessage();
            return false;
        }

        // Archives often contain a single top-level folder.
        if (! installed.getChildFile ("voice.json").existsAsFile())
        {
            auto subdirs = installed.findChildFiles (juce::File::findDirectories, false);
            if (subdirs.size() == 1 && subdirs[0].getChildFile ("voice.json").existsAsFile())
            {
                auto tmp = voicesDir.getNonexistentChildFile ("import", "");
                subdirs[0].moveFileTo (tmp);
                installed.deleteRecursively();
                tmp.moveFileTo (installed);
            }
        }
    }

    VoiceProfile p;
    if (! loadVoicePack (installed, p, error))
    {
        installed.deleteRecursively();
        return false;
    }

    rescanUserVoices();
    return true;
}

} // namespace voxa
