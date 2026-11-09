#pragma once

#include "RenderVoice.h"
#include <juce_core/juce_core.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace voxa::dsp
{

/** Audio on the host's timeline (sample positions from 0), kept in fixed pages, with a note of which 256-sample
    cells were written. write() allocates pages; writeRealtime() takes them from a small pool that the message
    thread keeps topped up (topUp()), so the audio thread never allocates. */
class TimelineAudio
{
public:
    static constexpr int pageShift = 15, pageSize = 1 << pageShift;
    static constexpr int cellShift = 8, cellSize = 1 << cellShift, cellsPerPage = pageSize / cellSize;

    /** `maxSamples`: positions from that one on are ignored. `maxPages` (0: no limit) caps the memory. */
    TimelineAudio (int numChannels, juce::int64 maxSamples, int maxPages = 0);
    ~TimelineAudio();

    int getNumChannels() const noexcept { return channels; }

    /** Writes `n` samples per channel at timeline position `start` (allocates; not on the audio thread). */
    void write (juce::int64 start, const float* const* data, int n);
    /** Audio thread: as write(), with pages from the pool. Returns false where a page was needed and none was left
        (that stretch is not written). */
    bool writeRealtime (juce::int64 start, const float* const* data, int n) noexcept;
    /** A copy (allocates; not on the audio thread). */
    std::unique_ptr<TimelineAudio> clone() const;
    /** Message thread: allocates pages until `spare` wait in the pool (none past maxPages). */
    void topUp (int spare);

    bool covers (juce::int64 t) const noexcept;
    float sample (int channel, juce::int64 t) const noexcept;   // 0 where nothing was written
    void read (int channel, juce::int64 start, float* dest, int n) const noexcept;
    /** Stretches of consecutive written cells, in timeline order. */
    std::vector<juce::Range<juce::int64>> getRuns() const;
    juce::int64 getNumSamplesWritten() const noexcept { return (juce::int64) cellsWritten.load() * cellSize; }

private:
    struct Page
    {
        std::unique_ptr<float[]> data;
        uint8_t valid[cellsPerPage] {};
        explicit Page (int numChannels) : data (new float[(size_t) numChannels * pageSize]()) {}
    };

    Page* pageFor (juce::int64 pageIndex, bool realtime) noexcept;
    Page* allocatePage();   // nullptr past maxPages
    bool writeImpl (juce::int64 start, const float* const* data, int n, bool realtime) noexcept;

    // The page table is written by one writer at a time and read once its writing is over (TrackRender hands a
    // capture to the render thread only after the audio thread has stopped writing to it).
    const int channels, maxPages;
    const juce::int64 numTablePages;
    std::unique_ptr<Page*[]> table;
    std::mutex ownedLock;                       // topUp (message thread) vs. allocating writes
    std::vector<std::unique_ptr<Page>> owned;   // every page allocated
    std::vector<Page*> poolSlots;
    juce::AbstractFifo pool { 64 };
    std::atomic<int> cellsWritten { 0 };

    JUCE_DECLARE_NON_COPYABLE (TimelineAudio)
};

/** Full-quality rendering of the track the plug-in is on, played back in place of the live voice.

    Capture: while armed, the input of every played stretch is kept on the host's timeline. When the transport
    stops, it is rendered on a background thread with renderVoice() (the AI singer converting each stretch at once,
    the DSP at High Quality); from then on playback over those stretches uses the render (wet signal, before Mix and
    Output Gain), crossfading to the live voice outside them. A change of voice or settings re-renders the captured
    input by itself; a changed vocal needs capturing again. */
class TrackRender
{
public:
    enum class State { Off, Capturing, Rendering, Ready, Failed };

    /** The settings of a render (message thread snapshot), and a signature of everything it depends on. */
    struct Request
    {
        VoiceRenderJob job;
        float inputGain = 1.0f;
        bool autoLevel = false;
        juce::String signature;
        juce::String notReady;   // why it can't start yet (e.g. the AI voice is still loading): wait
        juce::String unsupported;   // why this voice can't be rendered: the live voice plays
    };

    TrackRender();
    ~TrackRender();   // cancels the render and waits for it

    //==========================================================================
    // Message thread
    /** Arms capturing: from scratch (a previous render is dropped) or, with `addToCapture`, adding to what was
        captured before (stretches played again are replaced); the render plays on until the capture ends. */
    void startCapture (double sampleRate, bool addToCapture = false);
    /** Back to the live voice; the capture is forgotten. */
    void turnOff();
    /** Called ~20 Hz: keeps pages ready for capturing, starts a render when a capture ended, and re-renders when
        `signature` (everything the render depends on) changed. `makeRequest` builds the settings of a new render. */
    void service (const juce::String& signature, const std::function<Request()>& makeRequest);
    /** The host's sample rate changed: a render made at another rate is dropped. */
    void setSampleRate (double sampleRate);

    State getState() const noexcept           { return state.load(); }
    float getProgress() const noexcept        { return progress.load(); }
    double getCapturedSeconds() const noexcept;
    double getRenderedSeconds() const noexcept;
    /** Why the last render failed, or a note about it (e.g. the AI conversion fell back to the DSP voice). */
    juce::String getMessage() const;
    /** True while playback is using the render (the last block). */
    bool isPlayingRender() const noexcept     { return playingRender.load(); }

    /** The captured input as saved for the project (a FLAC file in `folder`, written when a capture is first
        rendered), so reopening the project renders it again. Any thread. */
    struct SavedCapture
    {
        juce::String file, runs;   // runs: "start-end;..." in samples on the host's timeline
        double sampleRate = 0.0;
        bool isValid() const noexcept { return file.isNotEmpty() && sampleRate > 0.0; }
    };
    SavedCapture getSavedCapture() const;
    /** Restores a saved capture (any thread): it is loaded and rendered once the host's sample rate is known and
        matches. */
    void restoreCapture (const SavedCapture&);
    /** Where captures are saved (set once). */
    void setCaptureFolder (const juce::File& folder)   { captureFolder = folder; }

    //==========================================================================
    // Audio thread
    /** The live input (before Input Gain) at host time `hostTime`. `nonRealtime`: the host renders offline (an
        export), so pages may be allocated here instead of coming from the pool. */
    void capture (juce::int64 hostTime, const float* mono, int n, bool playing, bool nonRealtime) noexcept;
    /** Replaces `wetL/R` with the render where it covers input time `inputTime` (the host time minus the latency). */
    void play (juce::int64 inputTime, float* wetL, float* wetR, int n, bool playing) noexcept;

    /** How long the capture may get (memory: 4 bytes per sample captured, 8 per sample rendered). */
    static constexpr double maxCaptureMinutes = 30.0;
    static constexpr double maxTimelineHours = 4.0;

private:
    void startRender (const Request&);
    void loadRestoredCapture();
    static bool saveCapture (const TimelineAudio&, double sampleRate, const juce::File&, juce::String& runs);
    void cancelRender();
    void publish (std::shared_ptr<TimelineAudio>);
    void unpublish();
    void retire (std::shared_ptr<TimelineAudio>);
    void collectGarbage();
    int spareCapturePages() const noexcept;

    std::atomic<State> state { State::Off };
    std::atomic<float> progress { 0.0f };
    double sampleRate = 48000.0;

    // capture (audio thread writes while `capturing`)
    std::shared_ptr<TimelineAudio> input;
    bool renderWanted = false;   // a capture ended or the settings changed: render when the request is ready
    std::atomic<TimelineAudio*> captureTarget { nullptr };
    std::atomic<bool> capturing { false }, captureEnded { false };
    bool wasPlayingWhileCapturing = false;   // audio thread

    // render (one background thread)
    juce::ThreadPool pool { 1 };
    std::shared_ptr<std::atomic<bool>> cancel;
    juce::String activeSignature;    // of the render playing or being made
    double signatureChangedAt = 0.0;
    mutable std::mutex resultLock;
    std::shared_ptr<TimelineAudio> finished;   // handed over by the render thread
    std::shared_ptr<std::atomic<bool>> finishedFor;   // ... for the render with this cancel flag
    bool renderDone = false;
    juce::String message;

    // saving with the project
    juce::File captureFolder, captureFile;   // captureFile: where the current capture is (to be) saved
    bool captureSaved = false;
    SavedCapture saved, pendingRestore;   // under resultLock
    int captureGeneration = 0;            // under resultLock: a new capture (or none) since a save started

    // playback
    std::shared_ptr<TimelineAudio> rendered;
    std::atomic<TimelineAudio*> playable { nullptr };
    std::vector<std::pair<double, std::shared_ptr<TimelineAudio>>> retired;   // freed once the audio thread is done
    float renderGain = 0.0f;   // audio thread: crossfade live -> render
    std::atomic<bool> playingRender { false };
    std::atomic<double> rate { 48000.0 };

    JUCE_DECLARE_NON_COPYABLE (TrackRender)
};

} // namespace voxa::dsp
