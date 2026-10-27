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
            expect (! library.findById ("neutral-robot")->hasNeuralModel());

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
