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
            expect (! rock->hasNeuralModel());

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
                zb.addFile (a, 0, "voices/cymatics_female_a/singer-a.vxpack");
                zb.addFile (noise, 9, "voices/cymatics_female_a/report.md");
                zb.addFile (b, 0, "voices/cymatics_female_b/singer-b.vxpack");
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

            // Nothing usable: a clear message, nothing installed.
            const auto r4 = library.importVoices (noise, work.getChildFile ("user4"));
            expect (! r4.succeeded(), r4.summary());
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
