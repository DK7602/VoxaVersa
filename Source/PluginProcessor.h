#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_events/juce_events.h>

#include "Parameters.h"
#include "core/AudioFeeds.h"
#include "voices/VoiceLibrary.h"
#include "presets/PresetManager.h"
#include "sing/SingEngine.h"

namespace voxa
{
class VoiceChain;
namespace neural { class NeuralVoiceModel; }

/** The VoxaVersa audio processor (shared by the effect, the instrument variant and the standalone app).

    Threading: every public method that is not marked "audio thread" or "any thread" must be called
    from the message thread. */
class VoxaVersaProcessor : public juce::AudioProcessor,
                           private juce::ValueTree::Listener,
                           private juce::AsyncUpdater
{
public:
    VoxaVersaProcessor();
    ~VoxaVersaProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout&) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override;
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 4.0; }

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int) override;
    const juce::String getProgramName (int) override;
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    //==============================================================================
    juce::AudioProcessorValueTreeState& getState() noexcept     { return apvts; }
    juce::UndoManager& getUndoManager() noexcept                { return undoManager; }
    VoiceLibrary& getVoiceLibrary() noexcept                    { return voiceLibrary; }
    PresetManager& getPresetManager() noexcept                  { return presetManager; }

    /** True for the "VoxaVersa Singer" instrument build. */
    static bool isInstrumentBuild() noexcept;
    bool isStandaloneApp() const noexcept { return wrapperType == wrapperType_Standalone; }

    //==============================================================================
    // Visual feeds (any thread)
    ScopeBuffer inputScope, outputScope;
    LevelMeterSource inputMeter, outputMeter;
    std::atomic<float> detectedPitchHz { 0.0f };   // source pitch (0 = unvoiced)
    std::atomic<float> targetPitchHz { 0.0f };     // pitch after correction / transposition
    std::atomic<int> detectedKeyRoot { -1 };       // auto-key estimate (-1 = unknown)
    std::atomic<int> detectedScale { 0 };
    std::atomic<float> autoLevelGainDb { 0.0f };
    std::atomic<int> sourceGenderEstimate { 0 };   // 0 unknown, 1 male, 2 female

    int getCurrentLatencySamples() const noexcept { return getLatencySamples(); }
    double getCurrentSampleRate() const noexcept  { return currentSampleRate; }

    //==============================================================================
    // Voices
    juce::String getSelectedVoiceId() const;
    const VoiceProfile& getSelectedVoice() const;
    void selectVoice (const juce::String& voiceId);
    juce::String getBlendVoiceId() const;
    void setBlendVoice (const juce::String& voiceIdOrEmpty);

    enum class NeuralStatus { Unavailable, NoModelForVoice, Loading, Active, Error };
    NeuralStatus getNeuralStatus() const noexcept { return neuralStatus.load(); }
    juce::String getNeuralStatusMessage() const;

    //==============================================================================
    // Input source: live input or a loaded voice sample (drag & drop onto the input panel)
    bool loadSample (const juce::File& audioFile, juce::String& errorMessage);
    void clearSample();
    bool hasSample() const noexcept;
    juce::String getSampleName() const;
    double getSampleLengthSeconds() const noexcept;
    void setSamplePlaying (bool shouldPlay);
    bool isSamplePlaying() const noexcept { return samplePlaying.load(); }
    double getSamplePlayPosition() const noexcept;     // 0..1

    //==============================================================================
    // Offline rendering (sample or sung take through the full voice chain) for export / drag-out
    enum class RenderTarget { Sample, SingTake };
    void startOfflineRender (RenderTarget);
    bool isOfflineRendering() const noexcept { return offlineRendering.load(); }
    float getOfflineRenderProgress() const noexcept { return offlineProgress.load(); }
    void cancelOfflineRender();
    /** Last rendered file (WAV, in the temp/export folder), or File() if none. */
    juce::File getLastRenderedFile() const;
    juce::String getLastRenderError() const;
    static juce::File getExportDirectory();

    //==============================================================================
    // Text to Sing
    void setLyrics (const juce::String& lyrics);
    juce::String getLyrics() const;
    static constexpr int maxLyricsLength = 2000;

    bool loadMelodyMidiFile (const juce::File& midiFile, juce::String& errorMessage);
    juce::String getMelodyFileName() const;

    /** Captures MIDI notes played into the plugin (Use MIDI source) so Generate can use them. */
    void setMidiCaptureEnabled (bool shouldCapture);
    bool isMidiCaptureEnabled() const noexcept { return midiCapture.load(); }
    int getNumCapturedNotes() const;
    void clearCapturedMidi();

    /** Records the input for hum-to-melody (Hum Melody source). */
    void setHumRecording (bool shouldRecord);
    bool isHumRecording() const noexcept { return humRecording.load(); }
    double getHumRecordedSeconds() const noexcept;

    enum class SingState { Idle, Rendering, Ready, Failed };
    /** Renders the lyrics on a background thread using the current voice, key, scale, tempo & melody source. */
    void generateSinging();
    void cancelSinging();
    SingState getSingState() const noexcept { return singState.load(); }
    float getSingProgress() const noexcept { return singProgress.load(); }
    juce::String getSingError() const;
    std::vector<sing::SungNote> getSungNotes() const;
    double getSingTakeLengthSeconds() const;

    /** Preview playback of the rendered take (standalone, or when the host is stopped). */
    void setSingPreviewPlaying (bool shouldPlay);
    bool isSingPreviewPlaying() const noexcept { return singPreviewPlaying.load(); }
    double getSingPlayPosition() const noexcept;   // 0..1

    /** Writes the take's melody (with lyric events) as a .mid file into the export directory. */
    juce::File exportSingMidi();

    /** Live MIDI singing: index of the syllable that the next note will sing. */
    int getLiveSyllableIndex() const noexcept;

    //==============================================================================
    // Recording of the processed output (mainly for the standalone app)
    void startRecording();
    void stopRecording();
    bool isRecording() const noexcept { return recording.load(); }
    double getRecordingSeconds() const noexcept;
    juce::File getLastRecording() const;

private:
    //==============================================================================
    void valueTreePropertyChanged (juce::ValueTree&, const juce::Identifier&) override;
    void valueTreeRedirected (juce::ValueTree&) override;
    void handleAsyncUpdate() override;

    void updateVoiceTraits();
    void updateNeuralModel();
    void updateLatency();

    struct Impl;
    std::unique_ptr<Impl> impl;

    juce::UndoManager undoManager;
    juce::AudioProcessorValueTreeState apvts;
    VoiceLibrary voiceLibrary;
    PresetManager presetManager;

    double currentSampleRate = 48000.0;

    std::atomic<NeuralStatus> neuralStatus { NeuralStatus::Unavailable };
    std::atomic<bool> samplePlaying { false }, offlineRendering { false }, midiCapture { false },
                      humRecording { false }, singPreviewPlaying { false }, recording { false };
    std::atomic<float> offlineProgress { 0.0f }, singProgress { 0.0f };
    std::atomic<SingState> singState { SingState::Idle };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoxaVersaProcessor)
};

} // namespace voxa
