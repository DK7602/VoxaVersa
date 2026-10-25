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
