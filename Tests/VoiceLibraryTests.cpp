#include "voices/VoiceLibrary.h"

namespace voxa
{

class VoiceLibraryTests : public juce::UnitTest
{
public:
    VoiceLibraryTests() : juce::UnitTest ("Voice library", "Core") {}

    void runTest() override
    {
        auto root = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("voxaversa-library-tests");
        root.deleteRecursively();

        auto makePack = [&root] (const juce::String& folder, const juce::String& json)
        {
            auto dir = root.getChildFile (folder);
            dir.createDirectory();
            dir.getChildFile ("model.onnx").replaceWithText ("not a real model");
            dir.getChildFile ("voice.json").replaceWithText (json);
        };

        beginTest ("Voice packs give built-in voices their AI model");
        {
            makePack ("a-hidden", R"({ "name": "Singer A", "gender": "female", "listed": false,
                                       "factoryVoices": ["female-pop", "female-latin", "no-such-voice"],
                                       "model": { "file": "model.onnx", "encoder": "contentvec-768", "sampleRate": 40000 } })");
            makePack ("b-listed", R"({ "name": "Singer B", "gender": "female",
                                       "factoryVoices": ["female-pop", "female-jazz"],
                                       "model": { "file": "model.onnx", "encoder": "contentvec-768", "sampleRate": 48000 } })");

            VoiceLibrary library;
            const auto factoryCount = createFactoryVoices().size();
            library.rescanUserVoices (root);

            const auto* pop = library.findById ("female-pop");
            const auto* latin = library.findById ("female-latin");
            const auto* jazz = library.findById ("female-jazz");
            const auto* rock = library.findById ("female-rock");
            expect (pop != nullptr && latin != nullptr && jazz != nullptr && rock != nullptr);
            if (pop == nullptr || latin == nullptr || jazz == nullptr || rock == nullptr)
                return;

            // The first pack by folder name wins a voice both packs claim.
            expect (pop->hasNeuralModel() && pop->aiPackName == "Singer A", "pop: " + pop->aiPackName);
            expectEquals (pop->neuralSampleRate, 40000);
            expect (pop->isFactory);
            expect (latin->hasNeuralModel() && latin->aiPackName == "Singer A");
            expect (jazz->hasNeuralModel() && jazz->aiPackName == "Singer B");
            expectEquals (jazz->neuralSampleRate, 48000);
            expect (jazz->packDirectory == root.getChildFile ("b-listed"));

            // A built-in voice nobody claims borrows an installed singer of its gender; there is no male one.
            expect (rock->hasNeuralModel() && (rock->aiPackName == "Singer A" || rock->aiPackName == "Singer B"), rock->aiPackName);
            expect (! library.findById ("male-rock")->hasNeuralModel());
            // Hard Tune Robot borrows the pop singers (here only a female one, so any input gets her).
            expect (library.findById ("neutral-robot")->hasNeuralModel() && library.findById ("neutral-robot")->aiPackName == "Singer A");

            // "listed": false hides the pack itself; a listed pack is also its own voice.
            expectEquals ((int) library.getVoices().size(), (int) factoryCount + 1);
            expect (library.findById ("user-singer-b") != nullptr);
            expect (library.findById ("user-singer-a") == nullptr);

            // Removing the packs removes the built-in voices' AI models again.
            root.deleteRecursively();
            library.rescanUserVoices (root);
            expect (! library.findById ("female-pop")->hasNeuralModel());
            expect (library.findById ("female-pop")->aiPackName.isEmpty());
            expectEquals ((int) library.getVoices().size(), (int) factoryCount);
        }

        beginTest ("Importing a collection: Google Drive's zip of the trained voices + the encoder");
        {
            auto work = root.getChildFile ("import");
            auto user = work.getChildFile ("user");
            work.createDirectory();

            // Two packs as .vxpack archives (voice.json + model at the root).
            auto makeVxpack = [&work] (const juce::String& id, const juce::String& name)
            {
                auto dir = work.getChildFile ("src-" + id);
                dir.createDirectory();
                dir.getChildFile ("model.onnx").replaceWithText ("model " + id);
                dir.getChildFile ("voice.json").replaceWithText (R"({ "name": ")" + name + R"(", "id": ")" + id
                    + R"(", "gender": "female", "factoryVoices": ["female-pop"], "model": { "file": "model.onnx", "sampleRate": 40000 } })");
                juce::ZipFile::Builder b;
                b.addFile (dir.getChildFile ("voice.json"), 9, "voice.json");
                b.addFile (dir.getChildFile ("model.onnx"), 9, "model.onnx");
                auto out = work.getChildFile (id + ".vxpack");
                juce::FileOutputStream os (out);
                b.writeToStream (os, nullptr);
                return out;
            };
            const auto a = makeVxpack ("singer-a", "Singer A");
            const auto b = makeVxpack ("singer-b", "Singer B");
            auto encoderFile = work.getChildFile ("contentvec-768.onnx");
            encoderFile.replaceWithText ("encoder");
            auto noise = work.getChildFile ("report.md");
            noise.replaceWithText ("# report");

            // What Google Drive gives for "Download" on the voices folder + the encoder: one zip, packs in subfolders.
            auto driveZip = work.getChildFile ("voices-20261025T190000Z-001.zip");
            {
                juce::ZipFile::Builder zb;
                zb.addFile (a, 0, "voices/studio_female_a/singer-a.vxpack");
                zb.addFile (noise, 9, "voices/studio_female_a/report.md");
                zb.addFile (b, 0, "voices/studio_female_b/singer-b.vxpack");
                zb.addFile (encoderFile, 9, "contentvec-768.onnx");
                juce::FileOutputStream os (driveZip);
                zb.writeToStream (os, nullptr);
            }

            VoiceLibrary library;
            const auto r = library.importVoices (driveZip, user);
            expectEquals (r.installed.size(), 2, r.summary());
            expect (r.encoderInstalled, r.summary());
            expect (r.errors.isEmpty(), r.summary());
            expect (r.summary().contains ("2 voices installed"), r.summary());
            expect (user.getChildFile ("Voices/singer-a/voice.json").existsAsFile());
            expect (user.getChildFile ("Voices/singer-b/model.onnx").existsAsFile());
            expect (user.getChildFile ("Encoders/contentvec-768.onnx").existsAsFile());
            expect (! user.getChildFile ("ImportTemp").exists(), "temporary files left behind");
            expect (library.findById ("user-singer-a") != nullptr);
            expect (library.findById ("female-pop")->hasNeuralModel(), "factoryVoices applied");

            // The same as an unzipped folder, a single .vxpack, and the encoder on its own.
            auto folder = work.getChildFile ("unzipped");
            juce::ZipFile (driveZip).uncompressTo (folder);
            auto user2 = work.getChildFile ("user2");
            const auto r2 = library.importVoices (folder, user2);
            expect (r2.installed.size() == 2 && r2.encoderInstalled, r2.summary());

            auto user3 = work.getChildFile ("user3");
            const auto r3 = library.importVoices (a, user3);
            expect (r3.installed.size() == 1 && r3.installed[0] == "Singer A" && ! r3.encoderInstalled, r3.summary());
            expect (library.importVoices (encoderFile, user3).encoderInstalled);

            // The AI pitch tracker, also under a browser's "(1)" name: installed as Encoders/fcpe.onnx.
            auto tracker = work.getChildFile ("fcpe (1).onnx");
            expect (tracker.replaceWithText ("onnx"));
            const auto rt = library.importVoices (tracker, user3);
            expect (rt.pitchTrackerInstalled && ! rt.encoderInstalled && rt.succeeded(), rt.summary());
            expect (user3.getChildFile ("Encoders/fcpe.onnx").existsAsFile());
            expect (rt.summary().contains ("pitch tracker"), rt.summary());

            // Nothing usable: a clear message, nothing installed.
            const auto r4 = library.importVoices (noise, work.getChildFile ("user4"));
            expect (! r4.succeeded(), r4.summary());

            beginTest ("An update of an installed singer may leave out its model; a pack that doesn't load changes nothing");
            {
                // singer-a's feature index: voice.json naming it and the model already installed, no model
                auto dir = work.getChildFile ("update-singer-a");
                dir.createDirectory();
                dir.getChildFile ("features.npy").replaceWithText ("index");
                dir.getChildFile ("voice.json").replaceWithText (R"({ "name": "Singer A", "id": "singer-a", "gender": "female",
                    "factoryVoices": ["female-pop"], "model": { "file": "model.onnx", "sampleRate": 40000, "index": "features.npy" } })");
                auto update = work.getChildFile ("update/singer-a.vxpack");
                update.getParentDirectory().createDirectory();
                {
                    juce::ZipFile::Builder zb;
                    zb.addFile (dir.getChildFile ("voice.json"), 9, "voice.json");
                    zb.addFile (dir.getChildFile ("features.npy"), 9, "features.npy");
                    juce::FileOutputStream os (update);
                    zb.writeToStream (os, nullptr);
                }

                const auto ru = library.importVoices (update, user);
                expect (ru.succeeded() && ru.installed.size() == 1, ru.summary());
                const auto installed = user.getChildFile ("Voices/singer-a");
                expectEquals (installed.getChildFile ("model.onnx").loadFileAsString(), juce::String ("model singer-a"),
                              "the installed model is kept");
                expect (installed.getChildFile ("features.npy").existsAsFile());
                expect (installed.getChildFile ("voice.json").loadFileAsString().contains ("features.npy"));
                auto leftovers = user.getChildFile ("Voices").findChildFiles (juce::File::findDirectories, false, "import*");
                expect (leftovers.isEmpty(), "staging folders left behind");

                // the same update for a singer who isn't installed: says so, installs nothing
                const auto fresh = work.getChildFile ("user5");
                const auto rf = library.importVoices (update, fresh);
                expect (! rf.succeeded() && rf.summary().contains ("import Singer A itself first"), rf.summary());
                expect (! fresh.getChildFile ("Voices/singer-a").exists());

                // a damaged pack over an installed singer leaves the singer as it was
                auto broken = work.getChildFile ("broken/singer-a.vxpack");
                broken.getParentDirectory().createDirectory();
                {
                    auto bad = work.getChildFile ("bad-voice.json");
                    bad.replaceWithText ("{ not json");
                    juce::ZipFile::Builder zb;
                    zb.addFile (bad, 9, "voice.json");
                    juce::FileOutputStream os (broken);
                    zb.writeToStream (os, nullptr);
                }
                expect (! library.importVoices (broken, user).succeeded());
                expect (installed.getChildFile ("model.onnx").existsAsFile() && installed.getChildFile ("features.npy").existsAsFile(),
                        "the installed singer is untouched");
                leftovers = user.getChildFile ("Voices").findChildFiles (juce::File::findDirectories, false, "import*");
                expect (leftovers.isEmpty(), "staging folders left behind");

                // "update": true - found by its id whatever the archive is called; its fields over the installed
                // voice.json, "model" key by key, so it needs nothing but what changes
                auto patchDir = work.getChildFile ("patch");
                patchDir.createDirectory();
                patchDir.getChildFile ("a_index.npy").replaceWithText ("index 2");
                patchDir.getChildFile ("voice.json").replaceWithText (R"({ "update": true, "id": "Singer_A", "name": "Renamed?",
                    "model": { "index": "a_index.npy", "indexRate": 0.6 } })");
                auto patch = work.getChildFile ("patches/whatever-it-is-called.vxpack");
                patch.getParentDirectory().createDirectory();
                {
                    juce::ZipFile::Builder zb;
                    zb.addFile (patchDir.getChildFile ("voice.json"), 9, "voice.json");
                    zb.addFile (patchDir.getChildFile ("a_index.npy"), 9, "a_index.npy");
                    juce::FileOutputStream os (patch);
                    zb.writeToStream (os, nullptr);
                }
                const auto rp = library.importVoices (patch, user);
                expect (rp.succeeded() && rp.installed.isEmpty() && rp.updated == juce::StringArray ("Singer A"), rp.summary());
                expect (rp.summary().contains ("Singer updated: Singer A"), rp.summary());
                const auto json = juce::JSON::parse (installed.getChildFile ("voice.json"));
                expectEquals (json.getProperty ("name", {}).toString(), juce::String ("Singer A"), "her name stays");
                expectEquals (json.getProperty ("model", {}).getProperty ("file", {}).toString(), juce::String ("model.onnx"));
                expectEquals (json.getProperty ("model", {}).getProperty ("index", {}).toString(), juce::String ("a_index.npy"));
                expectWithinAbsoluteError ((float) (double) json.getProperty ("model", {}).getProperty ("indexRate", 0.0), 0.6f, 1.0e-6f);
                expect (json.getProperty ("factoryVoices", {}).isArray(), "the rest of her voice.json stays");
                expectEquals (installed.getChildFile ("model.onnx").loadFileAsString(), juce::String ("model singer-a"));
                expect (installed.getChildFile ("a_index.npy").existsAsFile());
                expect (! user.getChildFile ("Voices/whatever-it-is-called").exists(), "no second singer");
                expect (library.findById ("user-singer-a") != nullptr);

                // for a singer who isn't installed: says so
                patchDir.getChildFile ("voice.json").replaceWithText (R"({ "update": true, "id": "singer-z", "name": "Singer Z",
                    "model": { "index": "a_index.npy" } })");
                const auto rz = library.importVoices (patchDir, user);
                expect (! rz.succeeded() && rz.summary().contains ("import Singer Z itself first"), rz.summary());
                leftovers = user.getChildFile ("Voices").findChildFiles (juce::File::findDirectories, false, "import*");
                expect (leftovers.isEmpty(), "staging folders left behind");
            }
            library.rescanUserVoices (root.getChildFile ("none"));
        }

        beginTest ("A voice assignment decides which built-in voices a pack powers and how it is shown");
        {
            auto work = root.getChildFile ("assignment");
            auto user = work.getChildFile ("user");
            auto src = work.getChildFile ("src");
            src.createDirectory();

            // a pack as the training produced it before: no assignment inside, so it shows up as its own voice
            src.getChildFile ("model.onnx").replaceWithText ("model");
            src.getChildFile ("voice.json").replaceWithText (R"({ "name": "Old Name A", "id": "pack-a", "gender": "female",
                                                                   "description": "old description",
                                                                   "model": { "file": "model.onnx", "sampleRate": 40000 } })");
            VoiceLibrary library;
            expect (library.importVoices (src, user).installed.size() == 1);
            expect (library.findById ("user-pack-a") != nullptr, "unassigned pack is listed");
            expect (! library.findById ("female-jazz")->hasNeuralModel());

            // the assignment, downloaded on its own (and renamed by the download)
            auto assignmentFile = work.getChildFile ("687e842a-9204-4293-a1e5-90c01c44df9a.json");
            assignmentFile.replaceWithText (R"({ "packs": { "pack-a": { "name": "Studio Female A", "listed": false,
                                                                        "description": "RVC v2 voice trained on dry studio vocals",
                                                                        "factoryVoices": ["female-jazz", "female-rock"] } } })");
            const auto r = library.importVoices (assignmentFile, user);
            expectEquals (r.assignedPacks, 1, r.summary());
            expect (r.succeeded() && r.summary().contains ("Voice assignment updated"), r.summary());
            expect (user.getChildFile ("Voices").getChildFile (VoiceLibrary::assignmentFileName).existsAsFile());
            expect (library.findById ("user-pack-a") == nullptr, "assigned pack is hidden");
            expect (library.findById ("female-jazz")->hasNeuralModel());
            expect (library.findById ("female-jazz")->aiPackName == "Studio Female A", library.findById ("female-jazz")->aiPackName);
            expect (library.findById ("female-rock")->hasNeuralModel());

            // it also applies to packs installed later (re-downloading a pack doesn't undo it) ...
            expect (library.importVoices (src, user).installed.size() == 1);
            expect (library.findById ("female-jazz")->aiPackName == "Studio Female A");
            expect (library.findById ("user-pack-a") == nullptr);

            // ... and a newer assignment (in Google Drive's zip of the voices folder) replaces it
            auto newer = work.getChildFile ("newer.json");
            newer.replaceWithText (R"({ "packs": { "pack-a": { "name": "Studio Female A", "listed": false, "factoryVoices": ["female-pop"] } } })");
            auto driveZip = work.getChildFile ("voices-002.zip");
            {
                juce::ZipFile::Builder zb;
                zb.addFile (newer, 9, "voices/voice_assignment.json");
                zb.addFile (src.getChildFile ("voice.json"), 9, "voices/notes/readme.json");
                juce::FileOutputStream os (driveZip);
                zb.writeToStream (os, nullptr);
            }
            const auto r2 = library.importVoices (driveZip, user);
            expectEquals (r2.assignedPacks, 1, r2.summary());
            expect (library.findById ("female-pop")->hasNeuralModel());
            expect (library.findById ("female-jazz")->aiPackName == "Studio Female A", "borrowed: the only female singer");
            expect (! library.findById ("male-jazz")->hasNeuralModel());

            // a pack whose id was spelled differently by an older training run is still found (by its name)
            auto other = work.getChildFile ("src-m3");
            other.createDirectory();
            other.getChildFile ("model.onnx").replaceWithText ("model");
            other.getChildFile ("voice.json").replaceWithText (R"({ "name": "VocalSet Male 3", "gender": "male",
                                                                     "model": { "file": "model.onnx", "sampleRate": 40000 } })");
            expect (library.importVoices (other, user).installed.size() == 1);
            expect (library.findById ("user-vocalset-male-3") != nullptr);
            auto m3 = work.getChildFile ("voice_assignment (2).json");
            m3.replaceWithText (R"({ "packs": { "vocalset-male3": { "name": "VocalSet Male 3", "listed": false,
                                                                    "factoryVoices": ["male-jazz", "male-blues"] } } })");
            const auto r3 = library.importVoices (m3, user);
            expect (r3.summary().contains ("Placed in the built-in voices: VocalSet Male 3"), r3.summary());
            expect (library.findById ("user-vocalset-male-3") == nullptr, "assigned pack is hidden");
            expect (library.findById ("male-jazz")->aiPackName == "VocalSet Male 3");

            // not an assignment
            auto bogus = work.getChildFile ("voice_assignment.json");
            bogus.replaceWithText ("[1, 2]");
            expect (! library.importVoices (bogus, user).succeeded());
            library.rescanUserVoices (root.getChildFile ("none"));
        }

        beginTest ("An assignment can rename installed singers; what refers to them follows");
        {
            auto work = root.getChildFile ("rename");
            auto user = work.getChildFile ("user");
            auto voicesDir = user.getChildFile ("Voices");
            auto src = work.getChildFile ("old-singer");
            src.createDirectory();
            src.getChildFile ("model.onnx").replaceWithText ("model");
            src.getChildFile ("voice.json").replaceWithText (R"({ "name": "Vendor Singer", "id": "old_singer", "gender": "female",
                                                                   "model": { "file": "model.onnx", "sampleRate": 40000 } })");
            VoiceLibrary library;
            expect (library.importVoices (src, user).installed.size() == 1);

            auto first = work.getChildFile ("first.json");
            first.replaceWithText (R"({ "packs": { "old-singer": { "name": "Studio Singer", "listed": false, "range": [55, 70], "median": 62,
                                                                   "factoryVoices": ["female-jazz"] } } })");
            expectEquals (library.importVoices (first, user).assignedPacks, 1);
            library.setSingerChoice ("female-rock", "user-old-singer");
            expect (library.findById ("female-rock")->aiSingerId == "user-old-singer");

            // the rename, with only the fields that change: the range stays
            auto rename = work.getChildFile ("voice_assignment.json");
            rename.replaceWithText (R"({ "packs": { "new-singer": { "factoryVoices": ["female-pop"] } },
                                         "renamed": { "old-singer": "new-singer" } })");
            const auto r = library.importVoices (rename, user);
            expect (r.succeeded() && r.renamedPacks == 1 && r.summary().contains ("1 singer renamed"), r.summary());
            expect (! voicesDir.getChildFile ("old-singer").exists());
            const auto manifest = juce::JSON::parse (voicesDir.getChildFile ("new-singer").getChildFile ("voice.json"));
            expect (manifest.getProperty ("id", {}).toString() == "new-singer");
            expect (manifest.getProperty ("name", {}).toString() == "Studio Singer", "the assignment's name, not the vendor's");
            const auto* singer = library.findSinger ("user-new-singer");
            expect (singer != nullptr && singer->traits.rangeLowMidi == 55 && singer->traits.rangeHighMidi == 70);
            expect (library.findById ("female-pop")->aiSingerId == "user-new-singer");
            expect (library.findById ("female-rock")->aiSingerId == "user-new-singer", "the singer chosen by ear follows");
            expect (library.getSingerChoice ("female-rock") == "user-new-singer");
            const auto stored = juce::JSON::parse (voicesDir.getChildFile (VoiceLibrary::assignmentFileName));
            expect (! stored["packs"].hasProperty ("old-singer") && stored["packs"].hasProperty ("new-singer"), "the old entry moved");

            // the old pack imported again (an older download) becomes the new one, and isn't installed twice
            expect (library.importVoices (src, user).installed.size() == 1);
            expect (! voicesDir.getChildFile ("old-singer").exists());
            expect (library.findSinger ("user-new-singer") != nullptr && library.findSinger ("user-old-singer") == nullptr);
            expect (library.findById ("female-pop")->aiSingerId == "user-new-singer");

            // a song saved with the old id still finds the singer when it is listed
            auto listed = work.getChildFile ("listed.json");
            listed.replaceWithText (R"({ "packs": { "new-singer": { "listed": true } } })");
            library.importVoices (listed, user);
            expect (library.findById ("user-old-singer") != nullptr && library.findById ("user-old-singer")->id == "user-new-singer");
            library.rescanUserVoices (root.getChildFile ("none"));
        }

        beginTest ("Robot and choir follow the input's gender; a singer chosen by ear sticks; spare singers get a card");
        {
            auto dir = root.getChildFile ("gendered");
            auto pack = [&dir] (const juce::String& folder, const juce::String& json)
            {
                auto d = dir.getChildFile (folder);
                d.createDirectory();
                d.getChildFile ("model.onnx").replaceWithText ("not a real model");
                d.getChildFile ("voice.json").replaceWithText (json);
            };
            pack ("f", R"({ "id": "pop-f", "name": "Pop F", "gender": "female", "listed": false, "factoryVoices": ["female-pop"],
                            "model": { "file": "model.onnx", "encoder": "contentvec-768", "sampleRate": 40000 } })");
            pack ("m", R"({ "id": "pop-m", "name": "Pop M", "gender": "male", "listed": false, "factoryVoices": ["male-pop", "male-theatre"],
                            "model": { "file": "model.onnx", "encoder": "contentvec-768", "sampleRate": 40000 } })");
            pack ("m2", R"({ "id": "rock-m", "name": "Rock M", "gender": "male", "listed": false, "factoryVoices": ["male-rock"],
                             "model": { "file": "model.onnx", "encoder": "contentvec-768", "sampleRate": 40000 } })");
            pack ("v", R"({ "id": "vocalset-male3", "name": "VocalSet Male 3", "gender": "male",
                            "model": { "file": "model.onnx", "encoder": "contentvec-768", "sampleRate": 40000 } })");

            VoiceLibrary library;
            library.rescanUserVoices (dir);
            expectEquals (library.findById ("neutral-robot")->aiPackName, juce::String ("Pop M"));   // a male input by default
            expect (library.setSourceGender (VoiceGender::Female));
            expectEquals (library.findById ("neutral-robot")->aiPackName, juce::String ("Pop F"));
            expect (! library.setSourceGender (VoiceGender::Female), "no change, nothing to reload");
            expect (library.findById ("neutral-choir")->hasNeuralModel(), "the choir borrows the gospel singers");
            {
                const auto* choir = library.findById ("neutral-choir");
                const auto* partner = library.findSinger (choir->choirPartnerSinger);
                expect (partner != nullptr && partner->traits.gender == VoiceGender::Male, "a woman's choir lead, a man's section with it");
                expect (library.setSourceGender (VoiceGender::Male));
                partner = library.findSinger (library.findById ("neutral-choir")->choirPartnerSinger);
                expect (partner != nullptr && partner->traits.gender == VoiceGender::Female, "and the other way round");
                library.setSourceGender (VoiceGender::Female);
            }

            expectEquals ((int) library.getAiSingers().size(), 4);
            library.setSingerChoice ("male-rock", "user-pop-m");
            expectEquals (library.findById ("male-rock")->aiPackName, juce::String ("Pop M"));
            expectEquals (library.getSingerChoice ("male-rock"), juce::String ("user-pop-m"));
            library.rescanUserVoices (dir);   // e.g. after an import: the choice stays
            expectEquals (library.findById ("male-rock")->aiPackName, juce::String ("Pop M"));
            library.setSingerChoice ("male-rock", {});
            expectEquals (library.findById ("male-rock")->aiPackName, juce::String ("Rock M"));

            const auto* card = library.findById ("user-vocalset-male3");
            expect (card != nullptr && card->name == "Lyric Tenor" && card->genre == "Classical", card != nullptr ? card->name : "missing");
        }

        beginTest ("Whisper keeps the DSP whisper; a singer assigned only to it is a voice of its own");
        {
            auto dir = root.getChildFile ("whisper");
            auto d = dir.getChildFile ("w");
            d.createDirectory();
            d.getChildFile ("model.onnx").replaceWithText ("not a real model");
            d.getChildFile ("voice.json").replaceWithText (R"({ "id": "vocalset-male11", "name": "VocalSet Male 11", "gender": "male",
                "listed": false, "factoryVoices": ["neutral-whisper"],
                "model": { "file": "model.onnx", "encoder": "contentvec-768", "sampleRate": 40000 } })");
            VoiceLibrary library;
            library.rescanUserVoices (dir);
            expect (! library.findById ("neutral-whisper")->hasNeuralModel(), "an AI singer cannot whisper");
            const auto* own = library.findById ("user-vocalset-male11");
            expect (own != nullptr && own->name == "Clarion Tenor", own != nullptr ? own->name : "missing");
        }

        beginTest ("A voice's sound chosen by ear sets its traits and lasts");
        {
            auto dir = root.getChildFile ("sounds");
            dir.createDirectory();
            VoiceLibrary library;
            library.rescanUserVoices (dir);
            const auto* rap = library.findById ("male-rap");
            expect (rap->sounds.size() == 3 && rap->soundName == "Street", rap->soundName);
            expectWithinAbsoluteError (rap->traits.aiGrit, 0.2f, 1.0e-6f);
            const float body = rap->traits.bodyDb;

            library.setSoundChoice ("male-rap", "Hard");
            expectEquals (library.findById ("male-rap")->soundName, juce::String ("Hard"));
            expectWithinAbsoluteError (library.findById ("male-rap")->traits.aiGrit, 0.0f, 1.0e-6f);
            expectWithinAbsoluteError (library.findById ("male-rap")->traits.presenceDb, 5.5f, 1.0e-6f);
            library.rescanUserVoices (dir);   // e.g. after an import, or the next session: the choice stays
            expectEquals (library.findById ("male-rap")->soundName, juce::String ("Hard"));
            library.setSoundChoice ("male-rap", "Growl");   // any sound can follow any
            expectWithinAbsoluteError (library.findById ("male-rap")->traits.aiGrit, 0.3f, 1.0e-6f);
            expectWithinAbsoluteError (library.findById ("male-rap")->traits.bodyDb, body, 1.0e-6f);

            // Male Blues: its growl strength (listening audit rounds 40 - 42), the strongest by default
            const auto* blues = library.findById ("male-blues");
            expect (blues->sounds.size() == 3 && blues->soundName == "Strong", blues->soundName);
            expectWithinAbsoluteError (blues->traits.aiGrowl, 0.35f * 0.85f * 0.6f, 1.0e-6f);
            expectWithinAbsoluteError (blues->traits.aiGrit, 0.0f, 1.0e-6f);
            library.setSoundChoice ("male-blues", "Light");
            expectWithinAbsoluteError (library.findById ("male-blues")->traits.aiGrowl, 0.35f * 0.85f * 0.35f, 1.0e-6f);
            library.setSoundChoice ("male-blues", "Medium");
            expectWithinAbsoluteError (library.findById ("male-blues")->traits.aiGrowl, 0.35f * 0.85f * 0.45f, 1.0e-6f);

            // Jazz: the dig-in growl as a rasp, both genders; Reggae: on the beat or laid back
            for (auto* id : { "male-jazz", "female-jazz" })
            {
                const auto* jazz = library.findById (id);
                expect (jazz->sounds.size() == 3 && jazz->soundName == "Smoky", id);
                expectWithinAbsoluteError (jazz->traits.aiGrowl, 0.35f * 0.85f * 0.65f, 1.0e-6f);
                expectWithinAbsoluteError (jazz->traits.aiGrowlSpread, 1.0f, 1.0e-6f);
                expectWithinAbsoluteError (jazz->traits.aiBreath, 0.3f, 1.0e-6f);
                expectWithinAbsoluteError (jazz->traits.tiltDb, -4.0f, 1.0e-6f);
            }
            library.setSoundChoice ("female-jazz", "Velvet");
            expectWithinAbsoluteError (library.findById ("female-jazz")->traits.aiBreath, 0.0f, 1.0e-6f);
            expectWithinAbsoluteError (library.findById ("female-jazz")->traits.tiltDb, -4.0f, 1.0e-6f);
            library.setSoundChoice ("female-jazz", "Raspy");   // the voice's own tone back
            expectGreaterThan (library.findById ("female-jazz")->traits.tiltDb, -3.0f);
            expectWithinAbsoluteError (library.findById ("female-jazz")->traits.aiGrowlSpread, 1.0f, 1.0e-6f);
            for (auto* id : { "male-reggae", "female-reggae" })
            {
                expectEquals (library.findById (id)->soundName, juce::String ("On the beat"));
                expectWithinAbsoluteError (library.findById (id)->traits.laidBackMs, 0.0f, 1.0e-6f);
            }
            library.setSoundChoice ("male-reggae", "Laid back");
            expectWithinAbsoluteError (library.findById ("male-reggae")->traits.laidBackMs, 30.0f, 1.0e-6f);

            // Female Gospel and Female Pop: darker and fuller (listening audits 43 and 46)
            for (auto* id : { "female-gospel", "female-pop" })
            {
                const auto& t = library.findById (id)->traits;
                expectWithinAbsoluteError (t.tiltDb, -0.5f, 1.0e-6f, id);
                expectWithinAbsoluteError (t.bodyDb, 4.0f, 1.0e-6f, id);
                expectWithinAbsoluteError (t.warmthDb, 3.5f, 1.0e-6f, id);
            }

            expectEquals (library.findById ("male-deep-bass")->soundName, juce::String ("Natural"));
            expectEquals (library.findById ("neutral-whisper")->soundName, juce::String ("Natural"));
            for (const auto& v : library.getVoices())
                for (const auto& sound : v.sounds)
                    expect (sound.apply != nullptr && sound.name.isNotEmpty() && sound.description.isNotEmpty(), v.id);
        }

        beginTest ("Built-in voices have unique ids and sensible ranges");
        {
            VoiceLibrary library;
            juce::StringArray ids;
            for (const auto& v : library.getVoices())
            {
                if (! v.isFactory)
                    continue;
                expect (! ids.contains (v.id), "duplicate id " + v.id);
                ids.add (v.id);
                expect (v.traits.rangeLowMidi < v.traits.rangeHighMidi, v.id);
                expect (v.traits.ringHz > 100.0f && v.traits.ringHz < 12000.0f, v.id + " ring " + juce::String (v.traits.ringHz));
                expect (v.traits.bodyHz > 50.0f && v.traits.bodyHz < 1000.0f, v.id + " body " + juce::String (v.traits.bodyHz));
            }
            expect (ids.size() >= 46, "factory voices: " + juce::String (ids.size()));
        }

        root.deleteRecursively();
    }
};

static VoiceLibraryTests voiceLibraryTests;

} // namespace voxa
