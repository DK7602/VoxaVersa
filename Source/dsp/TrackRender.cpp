#include "TrackRender.h"
#include "VocalFx.h"

namespace voxa::dsp
{

//==============================================================================
TimelineAudio::TimelineAudio (int numChannels, juce::int64 maxSamples, int maxPagesToUse)
    : channels (juce::jmax (1, numChannels)), maxPages (juce::jmax (0, maxPagesToUse)),
      numTablePages ((juce::jmax ((juce::int64) 0, maxSamples) + pageSize - 1) >> pageShift),
      table (new Page*[(size_t) juce::jmax ((juce::int64) 1, numTablePages)]())
{
    poolSlots.resize ((size_t) pool.getTotalSize(), nullptr);
}

TimelineAudio::~TimelineAudio() = default;

TimelineAudio::Page* TimelineAudio::allocatePage()
{
    const std::lock_guard<std::mutex> lock (ownedLock);
    if (maxPages > 0 && (int) owned.size() >= maxPages)
        return nullptr;
    owned.push_back (std::make_unique<Page> (channels));
    return owned.back().get();
}

void TimelineAudio::topUp (int spare)
{
    while (pool.getNumReady() < spare)
    {
        int s1, n1, s2, n2;
        pool.prepareToWrite (1, s1, n1, s2, n2);
        if (n1 < 1)
            break;
        auto* page = allocatePage();
        if (page == nullptr)
            break;
        poolSlots[(size_t) s1] = page;
        pool.finishedWrite (1);
    }
}

TimelineAudio::Page* TimelineAudio::pageFor (juce::int64 index, bool realtime) noexcept
{
    if (index < 0 || index >= numTablePages)
        return nullptr;
    if (auto* p = table[(size_t) index])
        return p;

    Page* page = nullptr;
    if (realtime)
    {
        int s1, n1, s2, n2;
        pool.prepareToRead (1, s1, n1, s2, n2);
        if (n1 < 1)
            return nullptr;
        page = poolSlots[(size_t) s1];
        pool.finishedRead (1);
    }
    else
    {
        page = allocatePage();
    }
    table[(size_t) index] = page;
    return page;
}

bool TimelineAudio::writeImpl (juce::int64 start, const float* const* data, int n, bool realtime) noexcept
{
    bool complete = true;
    int done = 0;
    while (done < n)
    {
        const juce::int64 t = start + done;
        const int offset = (int) (t & (pageSize - 1));
        const int count = juce::jmin (n - done, pageSize - offset);
        if (t >= 0)
        {
            if (auto* page = pageFor (t >> pageShift, realtime))
            {
                for (int ch = 0; ch < channels; ++ch)
                    std::memcpy (page->data.get() + (size_t) ch * pageSize + offset, data[ch] + done, sizeof (float) * (size_t) count);
                for (int c = offset >> cellShift; c <= (offset + count - 1) >> cellShift; ++c)
                    if (page->valid[c] == 0)
                    {
                        page->valid[c] = 1;
                        cellsWritten.fetch_add (1, std::memory_order_relaxed);
                    }
            }
            else if ((t >> pageShift) < numTablePages)
            {
                complete = false;
            }
        }
        done += count;
    }
    return complete;
}

void TimelineAudio::write (juce::int64 start, const float* const* data, int n)   { writeImpl (start, data, n, false); }
bool TimelineAudio::writeRealtime (juce::int64 start, const float* const* data, int n) noexcept { return writeImpl (start, data, n, true); }

bool TimelineAudio::covers (juce::int64 t) const noexcept
{
    if (t < 0 || (t >> pageShift) >= numTablePages)
        return false;
    const auto* page = table[(size_t) (t >> pageShift)];
    return page != nullptr && page->valid[(t & (pageSize - 1)) >> cellShift] != 0;
}

float TimelineAudio::sample (int ch, juce::int64 t) const noexcept
{
    if (t < 0 || (t >> pageShift) >= numTablePages || ! juce::isPositiveAndBelow (ch, channels))
        return 0.0f;
    const auto* page = table[(size_t) (t >> pageShift)];
    return page != nullptr ? page->data[(size_t) ch * pageSize + (size_t) (t & (pageSize - 1))] : 0.0f;
}

void TimelineAudio::read (int ch, juce::int64 start, float* dest, int n) const noexcept
{
    int done = 0;
    while (done < n)
    {
        const juce::int64 t = start + done;
        const int offset = (int) (t & (pageSize - 1));
        const int count = juce::jmin (n - done, pageSize - offset);
        const Page* page = (t >= 0 && (t >> pageShift) < numTablePages) ? table[(size_t) (t >> pageShift)] : nullptr;
        if (page != nullptr && juce::isPositiveAndBelow (ch, channels))
            std::memcpy (dest + done, page->data.get() + (size_t) ch * pageSize + offset, sizeof (float) * (size_t) count);
        else
            std::fill (dest + done, dest + done + count, 0.0f);
        done += count;
    }
}

std::vector<juce::Range<juce::int64>> TimelineAudio::getRuns() const
{
    std::vector<juce::Range<juce::int64>> runs;
    juce::int64 runStart = -1;
    for (juce::int64 p = 0; p < numTablePages; ++p)
    {
        const auto* page = table[(size_t) p];
        for (int c = 0; c < cellsPerPage; ++c)
        {
            const bool valid = page != nullptr && page->valid[c] != 0;
            const juce::int64 t = (p << pageShift) + ((juce::int64) c << cellShift);
            if (valid && runStart < 0)
                runStart = t;
            else if (! valid && runStart >= 0)
            {
                runs.emplace_back (runStart, t);
                runStart = -1;
            }
        }
    }
    if (runStart >= 0)
        runs.emplace_back (runStart, numTablePages << pageShift);
    return runs;
}

std::unique_ptr<TimelineAudio> TimelineAudio::clone() const
{
    auto copy = std::make_unique<TimelineAudio> (channels, numTablePages << pageShift, maxPages);
    std::vector<float> buffer;
    std::vector<const float*> ptrs ((size_t) channels);
    for (const auto& run : getRuns())
    {
        const int len = (int) run.getLength();
        buffer.resize ((size_t) len * (size_t) channels);
        for (int ch = 0; ch < channels; ++ch)
        {
            read (ch, run.getStart(), buffer.data() + (size_t) ch * (size_t) len, len);
            ptrs[(size_t) ch] = buffer.data() + (size_t) ch * (size_t) len;
        }
        copy->write (run.getStart(), ptrs.data(), len);
    }
    return copy;
}

//==============================================================================
TrackRender::TrackRender() = default;

TrackRender::~TrackRender()
{
    cancelRender();
    pool.removeAllJobs (true, -1);   // waited for, never killed (a thread killed inside ONNX Runtime takes the host down)
}

int TrackRender::spareCapturePages() const noexcept
{
    return (int) std::ceil (12.0 * sampleRate / TimelineAudio::pageSize);   // ~12 s of input
}

void TrackRender::cancelRender()
{
    if (cancel != nullptr)
        cancel->store (true);
    cancel = nullptr;
    renderWanted = false;
}

void TrackRender::retire (std::shared_ptr<TimelineAudio> t)
{
    if (t != nullptr)
        retired.emplace_back (juce::Time::getMillisecondCounterHiRes(), std::move (t));
}

void TrackRender::collectGarbage()
{
    const double now = juce::Time::getMillisecondCounterHiRes();
    retired.erase (std::remove_if (retired.begin(), retired.end(), [now] (const auto& r) { return now - r.first > 2000.0; }),
                   retired.end());
}

void TrackRender::publish (std::shared_ptr<TimelineAudio> t)
{
    playable.store (t.get(), std::memory_order_release);
    retire (std::move (rendered));
    rendered = std::move (t);
}

void TrackRender::unpublish()
{
    playable.store (nullptr, std::memory_order_release);
    retire (std::move (rendered));
    rendered = nullptr;
}

void TrackRender::startCapture (double newRate, bool addToCapture)
{
    cancelRender();
    capturing.store (false, std::memory_order_release);
    captureTarget.store (nullptr, std::memory_order_release);
    captureEnded.store (false);

    const bool keep = addToCapture && input != nullptr && juce::approximatelyEqual (newRate, sampleRate);
    std::shared_ptr<TimelineAudio> target;
    sampleRate = newRate;
    rate.store (newRate);
    if (keep)
    {
        target = input->clone();   // the render thread may still be reading the old one
    }
    else
    {
        unpublish();
        target = std::make_shared<TimelineAudio> (1, (juce::int64) (maxTimelineHours * 3600.0 * newRate),
                                                  (int) std::ceil (maxCaptureMinutes * 60.0 * newRate / TimelineAudio::pageSize));
    }
    target->topUp (spareCapturePages());
    retire (std::move (input));
    input = std::move (target);
    activeSignature.clear();
    {
        const std::lock_guard<std::mutex> lock (resultLock);
        message.clear();
        finished = nullptr;
        renderDone = false;
        saved = {};
        pendingRestore = {};
        ++captureGeneration;
    }
    captureFile = juce::File();
    captureSaved = false;
    progress.store (0.0f);
    state.store (State::Capturing);
    captureTarget.store (input.get(), std::memory_order_release);
    capturing.store (true, std::memory_order_release);
}

void TrackRender::turnOff()
{
    cancelRender();
    capturing.store (false, std::memory_order_release);
    captureTarget.store (nullptr, std::memory_order_release);
    captureEnded.store (false);
    unpublish();
    retire (std::move (input));
    input = nullptr;
    activeSignature.clear();
    progress.store (0.0f);
    captureFile = juce::File();
    captureSaved = false;
    {
        const std::lock_guard<std::mutex> lock (resultLock);
        saved = {};
        pendingRestore = {};
        ++captureGeneration;
    }
    state.store (State::Off);
}

TrackRender::SavedCapture TrackRender::getSavedCapture() const
{
    const std::lock_guard<std::mutex> lock (resultLock);
    return saved;
}

void TrackRender::restoreCapture (const SavedCapture& s)
{
    const std::lock_guard<std::mutex> lock (resultLock);
    pendingRestore = s;
}

bool TrackRender::saveCapture (const TimelineAudio& audio, double sr, const juce::File& file, juce::String& runsText)
{
    const auto runs = audio.getRuns();
    if (runs.empty())
        return false;
    const auto origin = runs.front().getStart();
    const auto end = runs.back().getEnd();
    if (! file.getParentDirectory().createDirectory())
        return false;

    juce::FlacAudioFormat flac;
    std::unique_ptr<juce::OutputStream> stream (file.createOutputStream());
    if (stream == nullptr)
        return false;
    std::unique_ptr<juce::AudioFormatWriter> writer (flac.createWriterFor (stream.get(), sr, 1, 24, {}, 5));
    if (writer == nullptr)
        return false;
    stream.release();   // the writer owns it now

    juce::AudioBuffer<float> chunk (1, 1 << 16);
    for (juce::int64 t = origin; t < end; t += chunk.getNumSamples())
    {
        const int n = (int) juce::jmin ((juce::int64) chunk.getNumSamples(), end - t);
        audio.read (0, t, chunk.getWritePointer (0), n);   // gaps between the runs are silence
        if (! writer->writeFromAudioSampleBuffer (chunk, 0, n))
            return false;
    }
    runsText.clear();
    for (const auto& r : runs)
        runsText << r.getStart() << '-' << r.getEnd() << ';';
    return true;
}

void TrackRender::loadRestoredCapture()
{
    SavedCapture s;
    {
        const std::lock_guard<std::mutex> lock (resultLock);
        if (! pendingRestore.isValid() || ! juce::approximatelyEqual (pendingRestore.sampleRate, sampleRate))
            return;
        s = pendingRestore;
        pendingRestore = {};
    }

    const juce::File file (s.file);
    juce::FlacAudioFormat flac;
    std::unique_ptr<juce::AudioFormatReader> reader (file.existsAsFile() ? flac.createReaderFor (file.createInputStream().release(), true)
                                                                         : nullptr);
    if (reader == nullptr)
        return;   // the file is gone: the live voice plays

    std::vector<juce::Range<juce::int64>> runs;
    for (const auto& token : juce::StringArray::fromTokens (s.runs, ";", {}))
        if (token.containsChar ('-'))
            runs.emplace_back (token.upToFirstOccurrenceOf ("-", false, false).getLargeIntValue(),
                               token.fromFirstOccurrenceOf ("-", false, false).getLargeIntValue());
    if (runs.empty())
        return;

    auto timeline = std::make_shared<TimelineAudio> (1, (juce::int64) (maxTimelineHours * 3600.0 * sampleRate),
                                                     (int) std::ceil (maxCaptureMinutes * 60.0 * sampleRate / TimelineAudio::pageSize));
    const auto origin = runs.front().getStart();
    juce::AudioBuffer<float> chunk (1, 1 << 16);
    for (const auto& run : runs)
        for (juce::int64 t = run.getStart(); t < run.getEnd(); t += chunk.getNumSamples())
        {
            const int n = (int) juce::jmin ((juce::int64) chunk.getNumSamples(), run.getEnd() - t);
            reader->read (&chunk, 0, n, t - origin, true, false);
            const float* data[1] = { chunk.getReadPointer (0) };
            timeline->write (t, data, n);
        }

    cancelRender();
    unpublish();
    retire (std::move (input));
    input = std::move (timeline);
    captureFile = file;
    captureSaved = true;
    {
        const std::lock_guard<std::mutex> lock (resultLock);
        saved = s;
        message.clear();
        ++captureGeneration;
    }
    activeSignature.clear();
    renderWanted = true;
    progress.store (0.0f);
    state.store (State::Rendering);
}

void TrackRender::setSampleRate (double newRate)
{
    if (! juce::approximatelyEqual (newRate, sampleRate) && state.load() != State::Off)
        turnOff();
    sampleRate = newRate;
    rate.store (newRate);
}

double TrackRender::getCapturedSeconds() const noexcept
{
    return input != nullptr ? (double) input->getNumSamplesWritten() / sampleRate : 0.0;
}

double TrackRender::getRenderedSeconds() const noexcept
{
    return rendered != nullptr ? (double) rendered->getNumSamplesWritten() / sampleRate : 0.0;
}

juce::String TrackRender::getMessage() const
{
    const std::lock_guard<std::mutex> lock (resultLock);
    return message;
}

void TrackRender::service (const juce::String& signature, const std::function<Request()>& makeRequest)
{
    collectGarbage();

    if (capturing.load())
        if (input != nullptr)
            input->topUp (spareCapturePages());

    // a project reopened with a render: its captured input
    if (! capturing.load())
        loadRestoredCapture();

    // the transport stopped after something was captured: render it (and save it with the project)
    if (captureEnded.exchange (false))
    {
        renderWanted = true;
        state.store (State::Rendering);
        captureSaved = false;
        captureFile = captureFolder != juce::File() ? captureFolder.getChildFile ("Capture " + juce::Uuid().toString() + ".flac")
                                                    : juce::File();
    }

    // a render finished
    {
        std::shared_ptr<TimelineAudio> done;
        bool doneFlag = false;
        {
            const std::lock_guard<std::mutex> lock (resultLock);
            if (renderDone && finishedFor == cancel && cancel != nullptr)
            {
                done = std::move (finished);
                doneFlag = true;
            }
            if (renderDone)
            {
                renderDone = false;
                finished = nullptr;
            }
        }
        if (doneFlag)
        {
            cancel = nullptr;
            if (done != nullptr)
            {
                publish (std::move (done));
                state.store (State::Ready);
            }
            else
            {
                state.store (State::Failed);
            }
        }
    }

    const auto st = state.load();
    if (st == State::Off || st == State::Capturing || input == nullptr)
        return;

    // the voice or a setting changed: the render no longer matches, the live voice plays until it is made again
    const double now = juce::Time::getMillisecondCounterHiRes();
    if (! renderWanted && signature != activeSignature)
    {
        if (! activeSignature.isEmpty())
            unpublish();
        cancelRender();
        renderWanted = true;
        signatureChangedAt = now;
        activeSignature = signature;
        state.store (State::Rendering);
        progress.store (0.0f);
    }
    else if (renderWanted && signature != activeSignature)
    {
        activeSignature = signature;
        signatureChangedAt = now;   // still changing (a knob being turned): wait until it settles
    }

    if (renderWanted && now - signatureChangedAt >= 400.0)
    {
        const auto request = makeRequest();
        if (request.unsupported.isNotEmpty())
        {
            renderWanted = false;
            activeSignature = signature;
            const std::lock_guard<std::mutex> lock (resultLock);
            message = request.unsupported;
            state.store (State::Failed);
        }
        else if (request.notReady.isEmpty())
        {
            renderWanted = false;
            startRender (request);
        }
        else
        {
            const std::lock_guard<std::mutex> lock (resultLock);
            message = request.notReady;
        }
    }
}

void TrackRender::startRender (const Request& request)
{
    cancelRender();
    cancel = std::make_shared<std::atomic<bool>> (false);
    activeSignature = request.signature;
    state.store (State::Rendering);
    progress.store (0.0f);
    {
        const std::lock_guard<std::mutex> lock (resultLock);
        message.clear();
    }

    auto source = input;
    auto flag = cancel;
    const double sr = sampleRate;
    const auto saveTo = captureSaved ? juce::File() : captureFile;
    captureSaved = captureSaved || saveTo != juce::File();
    int generation;
    {
        const std::lock_guard<std::mutex> lock (resultLock);
        generation = captureGeneration;
    }
    pool.addJob ([this, source, flag, request, sr, saveTo, generation]
    {
        if (saveTo != juce::File())
        {
            juce::String runs;
            const bool ok = saveCapture (*source, sr, saveTo, runs);
            const std::lock_guard<std::mutex> lock (resultLock);
            if (ok && generation == captureGeneration)
                saved = { saveTo.getFullPathName(), runs, sr };
        }

        const auto runs = source->getRuns();
        juce::int64 total = 0;
        for (const auto& r : runs)
            total += r.getLength();

        auto out = std::make_shared<TimelineAudio> (2, (juce::int64) (maxTimelineHours * 3600.0 * sr));
        juce::String note;
        juce::int64 done = 0;
        auto job = request.job;
        job.tailSeconds = 0.0;   // after a stretch the live voice carries on
        for (const auto& run : runs)
        {
            if (flag->load())
                break;
            const int len = (int) run.getLength();
            juce::AudioBuffer<float> src (1, len);
            source->read (0, run.getStart(), src.getWritePointer (0), len);
            src.applyGain (request.inputGain);
            if (request.autoLevel)
            {
                AutoLevel level;
                level.prepare (sr);
                level.process (src.getWritePointer (0), len);
            }

            juce::AudioBuffer<float> wet;
            juce::String error;
            const auto before = done;
            if (! renderVoice (src, job, *flag,
                               [this, flag, before, len, total] (float p)
                               {
                                   if (! flag->load() && total > 0)
                                       progress.store ((float) ((double) before + p * (double) len) / (float) total);
                               },
                               wet, error))
                break;
            if (error.isNotEmpty())
                note = "The AI voice could not be rendered (" + error + "): the render uses the DSP voice";
            const float* data[2] = { wet.getReadPointer (0), wet.getReadPointer (1) };
            out->write (run.getStart(), data, juce::jmin (len, wet.getNumSamples()));
            done += len;
        }

        const std::lock_guard<std::mutex> lock (resultLock);
        if (flag->load())
            return;   // cancelled: a newer render (or none) is wanted
        finished = total > 0 ? out : nullptr;
        if (total == 0)
            note = "Nothing was captured: play the track while Render is capturing";
        message = note;
        finishedFor = flag;
        renderDone = true;
        progress.store (1.0f);
    });
}

//==============================================================================
void TrackRender::capture (juce::int64 hostTime, const float* mono, int n, bool playing, bool nonRealtime) noexcept
{
    if (! capturing.load (std::memory_order_acquire))
    {
        wasPlayingWhileCapturing = false;
        return;
    }
    auto* target = captureTarget.load (std::memory_order_acquire);
    if (target == nullptr)
        return;

    if (playing && hostTime >= 0)
    {
        const float* data[1] = { mono };
        if (nonRealtime)
            target->write (hostTime, data, n);   // an offline export: no time limit, pages may be allocated here
        else
            target->writeRealtime (hostTime, data, n);
        wasPlayingWhileCapturing = true;
    }
    else if (wasPlayingWhileCapturing)
    {
        wasPlayingWhileCapturing = false;
        if (target->getNumSamplesWritten() > 0)
        {
            // stop writing before the message thread hands the capture to the render thread
            capturing.store (false, std::memory_order_release);
            captureEnded.store (true, std::memory_order_release);
        }
    }
}

void TrackRender::play (juce::int64 inputTime, float* wetL, float* wetR, int n, bool playing) noexcept
{
    auto* r = playable.load (std::memory_order_acquire);
    if (r == nullptr && renderGain <= 0.0f)
    {
        playingRender.store (false);
        return;
    }

    const double sr = rate.load();
    const int fade = juce::jmax (1, (int) (0.01 * sr));   // 10 ms crossfades
    const float step = 1.0f / (float) fade;
    bool used = false;
    for (int i = 0; i < n; ++i)
    {
        const auto t = inputTime + i;
        // the render ends a fade early, so the live voice is back in full where the rendered stretch ends
        const bool has = r != nullptr && playing && r->covers (t) && r->covers (t + fade);
        renderGain = has ? juce::jmin (1.0f, renderGain + step) : juce::jmax (0.0f, renderGain - step);
        if (renderGain > 0.0f && r != nullptr)
        {
            wetL[i] += renderGain * (r->sample (0, t) - wetL[i]);
            wetR[i] += renderGain * (r->sample (1, t) - wetR[i]);
            used = used || renderGain >= 0.5f;
        }
    }
    if (r == nullptr)
        renderGain = 0.0f;
    playingRender.store (used);
}

} // namespace voxa::dsp
