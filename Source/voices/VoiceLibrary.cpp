#include "VoiceLibrary.h"
#include "SingerCards.h"

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
    r.aiCentreMidi = a.aiCentreMidi > 0.0f && b.aiCentreMidi > 0.0f ? lerp (a.aiCentreMidi, b.aiCentreMidi)
                                                                       : (t < 0.5f ? a.aiCentreMidi : b.aiCentreMidi);
    if (a.aiLowMidi > 0 && b.aiLowMidi > 0)
    {
        r.aiLowMidi = lerpi (a.aiLowMidi, b.aiLowMidi);
        r.aiHighMidi = lerpi (a.aiHighMidi, b.aiHighMidi);
    }
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
        t.aiCentreMidi = 0.5f * (float) (t.rangeLowMidi + t.rangeHighMidi);   // the singer's own range
        t.aiLowMidi = t.rangeLowMidi;
        t.aiHighMidi = t.rangeHighMidi;
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

/** The "packs" object of a voice assignment file (or an empty var). */
static juce::var readAssignment (const juce::File& file)
{
    if (! file.existsAsFile())
        return {};

    const auto json = juce::JSON::parse (file);
    auto* obj = json.getDynamicObject();
    return obj != nullptr && obj->getProperty ("packs").getDynamicObject() != nullptr ? obj->getProperty ("packs") : juce::var();
}

/** The assignment entry for a pack, matched by its id (as in voice.json) or its folder name. */
/** Letters and digits only, lower case: "VocalSet Male 3", "vocalset_male3" and "vocalset-male3" all agree. */
static juce::String looseName (const juce::String& s)
{
    return s.toLowerCase().retainCharacters ("abcdefghijklmnopqrstuvwxyz0123456789");
}

static const juce::var* findAssignment (const juce::var& packs, const VoiceProfile& p)
{
    auto* obj = packs.getDynamicObject();
    if (obj == nullptr)
        return nullptr;

    // by pack id or folder name; then, for a pack whose id was written differently, by id or name spelled loosely
    for (const auto& entry : obj->getProperties())
    {
        const auto key = entry.name.toString();
        if ("user-" + key.toLowerCase().replaceCharacters (" _/\\", "----") == p.id || key == p.packDirectory.getFileName())
            return &entry.value;
    }

    const auto id = looseName (p.id.fromFirstOccurrenceOf ("user-", false, false));
    const auto folder = looseName (p.packDirectory.getFileName());
    const auto name = looseName (p.name);
    for (const auto& entry : obj->getProperties())
    {
        const auto key = looseName (entry.name.toString());
        const auto entryName = looseName (entry.value.getProperty ("name", {}).toString());
        if (key == id || key == folder || (name.isNotEmpty() && (key == name || entryName == name)))
            return &entry.value;
    }

    return nullptr;
}

static void applyAssignment (VoiceProfile& p, const juce::var& entry)
{
    auto* obj = entry.getDynamicObject();
    if (obj == nullptr)
        return;

    if (obj->hasProperty ("name"))        p.name = obj->getProperty ("name").toString();
    if (obj->hasProperty ("description")) p.tagline = obj->getProperty ("description").toString();
    if (obj->hasProperty ("license"))     p.license = obj->getProperty ("license").toString();
    if (obj->hasProperty ("listed"))      p.listed = (bool) obj->getProperty ("listed");

    // the singer's measured range (3rd - 97th percentile) and median note
    if (auto* range = obj->getProperty ("range").getArray(); range != nullptr && range->size() == 2)
    {
        p.traits.rangeLowMidi = (int) (*range)[0];
        p.traits.rangeHighMidi = (int) (*range)[1];
        p.traits.aiCentreMidi = obj->hasProperty ("median") ? (float) (double) obj->getProperty ("median")
                                                            : 0.5f * (float) (p.traits.rangeLowMidi + p.traits.rangeHighMidi);
        p.traits.aiLowMidi = p.traits.rangeLowMidi;
        p.traits.aiHighMidi = p.traits.rangeHighMidi;
    }

    if (auto* ids = obj->getProperty ("factoryVoices").getArray())
    {
        p.factoryVoices.clear();
        for (auto& id : *ids)
            p.factoryVoices.addIfNotAlreadyThere (id.toString().trim());
    }
}

/** A built-in voice sung by `singer` (a pack with a voice model). */
static void giveModel (VoiceProfile& v, const VoiceProfile& singer)
{
    v.packDirectory = singer.packDirectory;
    v.neuralModelFile = singer.neuralModelFile;
    v.neuralEncoder = singer.neuralEncoder;
    v.neuralSampleRate = singer.neuralSampleRate;
    v.aiPackName = singer.name;
    v.aiSingerId = singer.id;
    v.traits.aiCentreMidi = singer.traits.aiCentreMidi;
    v.traits.aiLowMidi = singer.traits.aiLowMidi;
    v.traits.aiHighMidi = singer.traits.aiHighMidi;
}

void VoiceLibrary::rescanUserVoices (const juce::File& dir)
{
    scannedDirectory = dir;
    aiSingers.clear();
    voices.resize (numFactoryVoices);
    for (auto& v : voices)
    {
        v.packDirectory = juce::File();
        v.neuralModelFile = juce::File();
        v.neuralEncoder.clear();
        v.neuralSampleRate = 0;
        v.aiPackName.clear();
        v.aiSingerId.clear();
        v.aiSingerMale.clear();
        v.aiSingerFemale.clear();
        v.traits.aiCentreMidi = 0.0f;
        v.traits.aiLowMidi = v.traits.aiHighMidi = 0;
    }

    if (dir.isDirectory())
    {
        auto packs = dir.findChildFiles (juce::File::findDirectories, false);
        packs.sort();   // deterministic: when two packs claim a built-in voice, the first by name wins
        const auto assignment = readAssignment (dir.getChildFile (assignmentFileName));

        std::vector<VoiceProfile> singers;   // packs made to power built-in voices

        for (const auto& packDir : packs)
        {
            VoiceProfile p;
            juce::String error;
            if (! loadVoicePack (packDir, p, error))
            {
                DBG ("VoxaVersa: skipping voice pack: " << error);
                continue;
            }

            if (const auto* entry = findAssignment (assignment, p))
                applyAssignment (p, *entry);
            if (const auto* card = findSingerCard (p.id))
            {
                // a credit line (e.g. VocalSet's CC BY) stays in the description after the style's own
                p.tagline = juce::String (card->tagline) + (p.tagline.isNotEmpty() ? " - " + p.tagline : juce::String());
                p.name = card->title;
                p.genre = card->genre;
                p.styles = juce::StringArray::fromTokens (card->styles, ",", {});
            }
            if (p.hasNeuralModel())
                aiSingers.push_back (p);

            if (p.hasNeuralModel() && ! p.factoryVoices.isEmpty())
            {
                for (const auto& id : p.factoryVoices)
                {
                    const int index = indexOf (id);
                    if (index >= 0 && index < (int) numFactoryVoices && ! voices[(size_t) index].hasNeuralModel())
                        giveModel (voices[(size_t) index], p);
                }
                singers.push_back (p);
            }

            if (p.listed && indexOf (p.id) < 0)
                voices.push_back (std::move (p));
        }

        // A built-in voice whose own singer is not installed (yet) borrows the installed singer of its gender
        // whose register is closest, so every male and female voice has the AI engine.
        for (size_t i = 0; i < numFactoryVoices; ++i)
        {
            auto& v = voices[i];
            if (v.hasNeuralModel() || v.traits.gender == VoiceGender::Neutral)
                continue;

            const VoiceProfile* best = nullptr;
            float bestDistance = 0.0f;
            for (const auto& singer : singers)
            {
                if (singer.traits.gender != v.traits.gender)
                    continue;
                const float centre = singer.traits.aiCentreMidi > 0.0f ? singer.traits.aiCentreMidi : (float) singer.traits.centreMidi();
                const float distance = std::abs (centre - (float) v.traits.centreMidi());
                if (best == nullptr || distance < bestDistance)
                {
                    best = &singer;
                    bestDistance = distance;
                }
            }
            if (best != nullptr)
                giveModel (v, *best);
        }

        // Neutral voices without a singer of their own get one per input gender: Hard Tune Robot the pop
        // singers', Choir Ensemble the gospel singers'.
        for (size_t i = 0; i < numFactoryVoices; ++i)
        {
            auto& v = voices[i];
            if (v.traits.gender != VoiceGender::Neutral || v.hasNeuralModel())
                continue;
            const juce::String partner = v.id == "neutral-choir" ? "gospel" : (v.id == "neutral-robot" ? "pop" : "");
            if (partner.isEmpty())
                continue;
            if (const auto* m = findById ("male-" + partner))
                v.aiSingerMale = m->aiSingerId;
            if (const auto* f = findById ("female-" + partner))
                v.aiSingerFemale = f->aiSingerId;
        }

        // Singers chosen by ear (right-click a voice > AI Singer) come last, over the assignment.
        const auto choices = juce::JSON::parse (dir.getChildFile (choicesFileName));   // kept alive while read
        if (auto* chosen = choices.getDynamicObject())
        {
            for (const auto& entry : chosen->getProperties())
            {
                const int index = indexOf (entry.name.toString());
                const auto* singer = findSinger (entry.value.toString());
                if (index < 0 || index >= (int) numFactoryVoices || singer == nullptr)
                    continue;
                auto& v = voices[(size_t) index];
                giveModel (v, *singer);
                if (v.traits.gender == VoiceGender::Neutral && (v.aiSingerMale.isNotEmpty() || v.aiSingerFemale.isNotEmpty()))
                    v.aiSingerMale = v.aiSingerFemale = singer->id;
            }
        }
        applyNeutralSingers();
    }

    sendChangeMessage();
}

static bool installPack (const juce::File& source, const juce::File& voicesDir, juce::String& error, juce::String& name)
{
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
    if (! VoiceLibrary::loadVoicePack (installed, p, error))
    {
        installed.deleteRecursively();
        return false;
    }

    name = p.name;
    return true;
}

const VoiceProfile* VoiceLibrary::findSinger (const juce::String& packId) const
{
    for (const auto& s : aiSingers)
        if (s.id == packId)
            return &s;
    return nullptr;
}

juce::String VoiceLibrary::getSingerChoice (const juce::String& voiceId) const
{
    const auto json = juce::JSON::parse (scannedDirectory.getChildFile (choicesFileName));
    return json.getProperty (voiceId, {}).toString();
}

void VoiceLibrary::setSingerChoice (const juce::String& voiceId, const juce::String& singerPackId)
{
    const auto dir = scannedDirectory;
    const auto file = dir.getChildFile (choicesFileName);
    auto json = juce::JSON::parse (file);
    if (json.getDynamicObject() == nullptr)
        json = juce::var (new juce::DynamicObject());
    if (singerPackId.isEmpty())
        json.getDynamicObject()->removeProperty (voiceId);
    else
        json.getDynamicObject()->setProperty (voiceId, singerPackId);
    if (dir.createDirectory())
        file.replaceWithText (juce::JSON::toString (json));
    rescanUserVoices (dir);
}

bool VoiceLibrary::setSourceGender (VoiceGender g)
{
    if (g == VoiceGender::Neutral || g == sourceGender)
        return false;
    sourceGender = g;
    return applyNeutralSingers();
}

bool VoiceLibrary::applyNeutralSingers()
{
    bool changed = false;
    for (size_t i = 0; i < numFactoryVoices; ++i)
    {
        auto& v = voices[i];
        if (v.aiSingerMale.isEmpty() && v.aiSingerFemale.isEmpty())
            continue;
        const bool female = sourceGender == VoiceGender::Female;
        const auto& id = (female ? v.aiSingerFemale : v.aiSingerMale).isNotEmpty() ? (female ? v.aiSingerFemale : v.aiSingerMale)
                                                                                 : (female ? v.aiSingerMale : v.aiSingerFemale);
        if (const auto* singer = findSinger (id); singer != nullptr && singer->neuralModelFile != v.neuralModelFile)
        {
            giveModel (v, *singer);
            changed = true;
        }
    }
    return changed;
}

bool VoiceLibrary::importVoicePack (const juce::File& source, juce::String& error)
{
    juce::String name;
    if (! installPack (source, getUserVoicesDirectory(), error, name))
        return false;

    rescanUserVoices();
    return true;
}

static bool isAssignmentFile (const juce::String& fileName)
{
    // inside folders and archives; browsers rename repeated downloads ("voice_assignment (1).json")
    return fileName.containsIgnoreCase ("assignment") && fileName.endsWithIgnoreCase (".json");
}

/** Merges an imported assignment into <Voices>/voice_assignment.json (newer entries replace older ones).
    Returns the number of packs it describes, or -1 if it isn't a voice assignment. */
static int mergeAssignment (const juce::File& imported, const juce::File& voicesDir)
{
    const auto incoming = readAssignment (imported);
    auto* in = incoming.getDynamicObject();
    if (in == nullptr || ! voicesDir.createDirectory())
        return -1;

    const auto target = voicesDir.getChildFile (VoiceLibrary::assignmentFileName);
    auto merged = readAssignment (target);
    if (merged.getDynamicObject() == nullptr)
        merged = juce::var (new juce::DynamicObject());

    for (const auto& entry : in->getProperties())
        merged.getDynamicObject()->setProperty (entry.name, entry.value);

    auto root = std::make_unique<juce::DynamicObject>();
    root->setProperty ("packs", merged);
    if (! target.replaceWithText (juce::JSON::toString (juce::var (root.release()))))
        return -1;

    return in->getProperties().size();
}

static bool isPitchTrackerFile (const juce::String& fileName)
{
    const auto n = fileName.toLowerCase();
    return n.startsWith ("fcpe") && n.endsWith (".onnx");   // "fcpe (1).onnx" after a repeated download
}

static bool isEncoderFile (const juce::String& fileName)
{
    const auto n = fileName.toLowerCase();
    return n == "contentvec-768.onnx" || n == "vec-768-layer-12.onnx" || n == "contentvec-256.onnx" || n == "vec-256-layer-9.onnx"
        || isPitchTrackerFile (fileName);
}

juce::String VoiceLibrary::ImportResult::summary() const
{
    juce::String s;
    if (installed.size() == 1)
        s << "Voice installed: " << installed[0];
    else if (installed.size() > 1)
        s << installed.size() << " voices installed";
    if (encoderInstalled)
        s << (s.isEmpty() ? "AI content encoder installed" : " + AI content encoder");
    if (pitchTrackerInstalled)
        s << (s.isEmpty() ? "AI pitch tracker installed (AI singers now follow runs)" : " + AI pitch tracker");
    if (assignedPacks > 0)
    {
        s << (s.isEmpty() ? "Voice assignment updated" : " + voice assignment") << " (" << assignedPacks
          << (assignedPacks == 1 ? " singer)" : " singers)");
        auto names = placed;
        names.removeDuplicates (true);
        const int shown = juce::jmin (3, names.size());
        s << (names.isEmpty() ? juce::String (". None of them is installed here yet: import their voice folders too")
                              : ". Placed in the built-in voices: " + juce::StringArray (names.begin(), shown).joinIntoString (", ")
                                  + (names.size() > shown ? " and " + juce::String (names.size() - shown) + " more" : juce::String()));
    }
    if (errors.size() > 0)
        s << (s.isEmpty() ? "Import failed: " : " (") << errors[0]
          << (errors.size() > 1 ? " and " + juce::String (errors.size() - 1) + " more" : juce::String())
          << (succeeded() ? ")" : "");
    return s.isEmpty() ? juce::String ("No voice packs (.vxpack), voice assignment or AI encoder found there") : s;
}

VoiceLibrary::ImportResult VoiceLibrary::importVoices (const juce::File& source, const juce::File& userDataDirectory)
{
    ImportResult r;
    const auto voicesDir = userDataDirectory.getChildFile ("Voices");
    const auto encodersDir = userDataDirectory.getChildFile ("Encoders");

    auto pack = [&] (const juce::File& f)
    {
        juce::String error, name;
        if (installPack (f, voicesDir, error, name))
            r.installed.add (name);
        else
            r.errors.add (f.getFileName() + ": " + error);
    };

    auto encoder = [&] (const juce::File& f)
    {
        const bool pitch = isPitchTrackerFile (f.getFileName());
        const auto dest = encodersDir.getChildFile (pitch ? juce::String ("fcpe.onnx") : f.getFileName());
        if (encodersDir.createDirectory() && (dest == f || f.copyFileTo (dest)))
            (pitch ? r.pitchTrackerInstalled : r.encoderInstalled) = true;
        else
            r.errors.add ("Could not copy " + f.getFileName() + " to " + encodersDir.getFullPathName());
    };

    // assignments are merged after the packs, so they describe the packs that are installed by then
    juce::Array<juce::File> assignments;

    auto assignment = [&] (const juce::File& f)
    {
        const int n = mergeAssignment (f, voicesDir);
        if (n < 0)
        {
            r.errors.add (f.getFileName() + ": not a valid voice assignment");
            return;
        }
        r.assignedPacks += n;

        const auto incoming = readAssignment (f);
        for (const auto& dir : voicesDir.findChildFiles (juce::File::findDirectories, false))
        {
            VoiceProfile p;
            juce::String error;
            if (loadVoicePack (dir, p, error))
                if (const auto* entry = findAssignment (incoming, p))
                    if (entry->getProperty ("factoryVoices", {}).size() > 0)
                        r.placed.addIfNotAlreadyThere (entry->getProperty ("name", p.name).toString());
        }
    };

    auto scan = [&] (const juce::File& dir)
    {
        for (const auto& f : dir.findChildFiles (juce::File::findFiles, true, "*"))
        {
            if (f.hasFileExtension ("vxpack"))
                pack (f);
            else if (f.getFileName() == "voice.json")
                pack (f.getParentDirectory());
            else if (isEncoderFile (f.getFileName()))
                encoder (f);
            else if (isAssignmentFile (f.getFileName()))
                assignments.add (f);
        }

        for (const auto& f : assignments)
            assignment (f);
        assignments.clear();
    };

    if (source.isDirectory())
    {
        if (source.getChildFile ("voice.json").existsAsFile())
            pack (source);
        else
            scan (source);
    }
    else if (isEncoderFile (source.getFileName()))
    {
        encoder (source);
    }
    else if (source.hasFileExtension ("json") && source.getFileName() != "voice.json")
    {
        assignment (source);   // a voice assignment picked on its own, whatever the download called it
    }
    else
    {
        juce::ZipFile zip (source);
        juce::Array<int> wanted;
        for (int i = 0; i < zip.getNumEntries(); ++i)
        {
            const auto entryName = zip.getEntry (i)->filename;
            const auto base = entryName.fromLastOccurrenceOf ("/", false, false);
            if (base.endsWithIgnoreCase (".vxpack") || isEncoderFile (base) || isAssignmentFile (base))
                wanted.add (i);
        }

        if (zip.getNumEntries() == 0)
        {
            r.errors.add (source.getFileName() + ": not a voice pack or zip archive");
        }
        else if (wanted.isEmpty())
        {
            pack (source);    // a single voice pack
        }
        else
        {
            // A collection: unpack only the packs, the encoder and the assignment, then install them.
            auto temp = userDataDirectory.getChildFile ("ImportTemp");
            temp.deleteRecursively();
            for (int i : wanted)
                if (auto result = zip.uncompressEntry (i, temp, true); result.failed())
                    r.errors.add (result.getErrorMessage());
            scan (temp);
            temp.deleteRecursively();
        }
    }

    if (r.installed.size() > 0 || r.assignedPacks > 0)
        rescanUserVoices (voicesDir);
    return r;
}

} // namespace voxa
