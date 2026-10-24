#include "SingEngine.h"
#include "Lyrics.h"
#include "Melody.h"
#include "OfflineRenderer.h"
#include "PitchTracker.h"

namespace voxa::sing
{

struct SingEngine::Impl {};

SingEngine::SingEngine() : impl (std::make_unique<Impl>()) {}
SingEngine::~SingEngine() = default;

//==============================================================================
SingResult SingEngine::render (const SingRequest& request, const std::atomic<bool>* cancel,
                               std::function<void (float)> progress)
{
    SingResult result;
    result.sampleRate = request.sampleRate;
    result.f0FrameRate = 100.0;

    auto isCancelled = [cancel] { return cancel != nullptr && cancel->load(); };
    auto report = [&progress] (float p) { if (progress) progress (juce::jlimit (0.0f, 1.0f, p)); };

    if (isCancelled())
    {
        result.errorMessage = "Cancelled";
        return result;
    }

    if (! (request.sampleRate >= 8000.0 && request.sampleRate <= 384000.0))
    {
        result.errorMessage = "Unsupported sample rate";
        return result;
    }

    if (! (request.bpm > 0.0) || ! std::isfinite (request.bpm))
    {
        result.errorMessage = "Invalid tempo";
        return result;
    }

    const auto script = parseLyrics (request.lyrics);
    if (script.isEmpty())
    {
        result.errorMessage = "There are no lyrics to sing - type some words first.";
        return result;
    }

    report (0.01f);

    Score score;
    if (request.melodyMode == MelodyMode::Provided)
    {
        if (request.melody.empty())
        {
            result.errorMessage = "No melody notes to sing: load a MIDI file, play notes in, hum a melody "
                                  "or switch to an automatic melody.";
            return result;
        }
        score = melody::fitProvided (script, request.melody, request.voice.traits, request.keyRoot,
                                     request.scale, request.bpm);
    }
    else
    {
        score = melody::compose (script, request.voice.traits, request.keyRoot, request.scale, request.seed);
    }

    if (score.notes.empty())
    {
        result.errorMessage = "Could not fit the lyrics to the melody.";
        return result;
    }

    if (isCancelled())
    {
        result.errorMessage = "Cancelled";
        return result;
    }

    report (0.03f);

    OfflineRenderer renderer (script, score, request.voice.traits, request.bpm, request.sampleRate, request.seed);
    const bool finished = renderer.render (cancel, [&report] (float p) { report (0.03f + 0.97f * p); });

    if (! finished || isCancelled())
    {
        result.errorMessage = "Cancelled";
        return result;
    }

    result.audio = std::move (renderer.audio);
    result.f0Hz = std::move (renderer.f0Hz);
    result.keyRoot = score.keyRoot;
    result.scale = score.scale;

    const double secondsPerBeat = 60.0 / request.bpm;
    for (const auto& n : score.notes)
    {
        SungNote sn;
        sn.startSeconds = n.startBeat * secondsPerBeat;
        sn.endSeconds = n.endBeat() * secondsPerBeat;
        sn.midiNote = n.midiNote;
        if (n.syllable >= 0 && n.syllable < script.getNumSyllables())
        {
            const auto& syl = script.syllables[(size_t) n.syllable];
            sn.syllable = n.continuation ? juce::String ("_") : syl.text;
            sn.wordIndex = syl.wordIndex;
        }
        result.notes.push_back (sn);
    }

    if (result.audio.getNumSamples() == 0)
        result.errorMessage = "Rendering produced no audio.";

    report (1.0f);
    return result;
}

int SingEngine::countSyllables (const juce::String& lyrics)
{
    return parseLyrics (lyrics).getNumSyllables();
}

//==============================================================================
std::vector<Note> SingEngine::melodyFromMidiFile (const juce::File& file, double& bpmOut, juce::String& errorMessage)
{
    bpmOut = 120.0;
    errorMessage.clear();

    juce::FileInputStream in (file);
    if (! in.openedOk())
    {
        errorMessage = "Could not open " + file.getFileName();
        return {};
    }

    juce::MidiFile midiFile;
    if (! midiFile.readFrom (in))
    {
        errorMessage = file.getFileName() + " is not a valid MIDI file";
        return {};
    }

    // Tempo: the earliest tempo event in any track.
    double tempoTime = 1.0e300;
    for (int t = 0; t < midiFile.getNumTracks(); ++t)
    {
        const auto* track = midiFile.getTrack (t);
        for (int i = 0; i < track->getNumEvents(); ++i)
        {
            const auto& m = track->getEventPointer (i)->message;
            if (m.isTempoMetaEvent() && m.getTimeStamp() < tempoTime)
            {
                const double secondsPerQuarter = m.getTempoSecondsPerQuarterNote();
                if (secondsPerQuarter > 0.0)
                {
                    tempoTime = m.getTimeStamp();
                    bpmOut = juce::jlimit (20.0, 400.0, 60.0 / secondsPerQuarter);
                }
                break;
            }
        }
    }

    const short timeFormat = midiFile.getTimeFormat();
    if (timeFormat <= 0)
        midiFile.convertTimestampTicksToSeconds();   // SMPTE: go via seconds

    for (int t = 0; t < midiFile.getNumTracks(); ++t)
    {
        const auto* track = midiFile.getTrack (t);
        bool hasNotes = false;
        for (int i = 0; i < track->getNumEvents() && ! hasNotes; ++i)
            hasNotes = track->getEventPointer (i)->message.isNoteOn();

        if (! hasNotes)
            continue;

        juce::MidiMessageSequence beats;
        for (int i = 0; i < track->getNumEvents(); ++i)
        {
            auto m = track->getEventPointer (i)->message;
            if (! m.isNoteOnOrOff())
                continue;

            const double ts = m.getTimeStamp();
            m.setTimeStamp (timeFormat > 0 ? ts / timeFormat : ts * bpmOut / 60.0);
            beats.addEvent (m);
        }
        beats.updateMatchedPairs();

        auto melody = melodyFromSequence (beats);
        if (melody.empty())
            errorMessage = "The MIDI file contains no usable notes";
        return melody;
    }

    errorMessage = "The MIDI file contains no notes";
    return {};
}

std::vector<Note> SingEngine::melodyFromSequence (const juce::MidiMessageSequence& sequenceInBeats)
{
    juce::MidiMessageSequence seq (sequenceInBeats);
    seq.updateMatchedPairs();

    std::vector<Note> notes;
    for (int i = 0; i < seq.getNumEvents(); ++i)
    {
        const auto* holder = seq.getEventPointer (i);
        const auto& m = holder->message;
        if (! m.isNoteOn())
            continue;

        double end = m.getTimeStamp() + 1.0;
        if (holder->noteOffObject != nullptr)
            end = holder->noteOffObject->message.getTimeStamp();
        else
        {
            // Unterminated note: until the next note starts (or one beat).
            for (int j = i + 1; j < seq.getNumEvents(); ++j)
            {
                const auto& next = seq.getEventPointer (j)->message;
                if (next.isNoteOn() && next.getTimeStamp() > m.getTimeStamp())
                {
                    end = next.getTimeStamp();
                    break;
                }
            }
        }

        Note n;
        n.startBeat = m.getTimeStamp();
        n.lengthBeats = end - m.getTimeStamp();
        n.midiNote = m.getNoteNumber();
        n.velocity = m.getFloatVelocity();
        notes.push_back (n);
    }

    return melody::makeMonophonic (std::move (notes));
}

std::vector<Note> SingEngine::melodyFromAudio (const juce::AudioBuffer<float>& audio, double sampleRate, double bpm)
{
    const int numSamples = audio.getNumSamples();
    const int numChannels = audio.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0 || sampleRate <= 0.0)
        return {};

    if (numChannels == 1)
        return transcribeMelody (audio.getReadPointer (0), numSamples, sampleRate, bpm);

    std::vector<float> mono ((size_t) numSamples, 0.0f);
    for (int c = 0; c < numChannels; ++c)
        juce::FloatVectorOperations::addWithMultiply (mono.data(), audio.getReadPointer (c), 1.0f / (float) numChannels, numSamples);
    return transcribeMelody (mono.data(), numSamples, sampleRate, bpm);
}

juce::MidiFile SingEngine::toMidiFile (const SingResult& result, double bpm)
{
    constexpr int ticksPerQuarter = 960;
    bpm = juce::jlimit (20.0, 400.0, bpm > 0.0 ? bpm : 120.0);
    const double ticksPerSecond = ticksPerQuarter * bpm / 60.0;

    juce::MidiMessageSequence track;
    track.addEvent (juce::MidiMessage::textMetaEvent (3, "VoxaVersa Vocal"), 0.0);
    track.addEvent (juce::MidiMessage::tempoMetaEvent ((int) std::lround (60.0e6 / bpm)), 0.0);
    track.addEvent (juce::MidiMessage::timeSignatureMetaEvent (4, 4), 0.0);

    double lastTick = 0.0;
    const auto& notes = result.notes;
    for (size_t i = 0; i < notes.size(); ++i)
    {
        const auto& n = notes[i];
        const double start = std::round (n.startSeconds * ticksPerSecond);
        const double end = juce::jmax (start + 1.0, std::round (n.endSeconds * ticksPerSecond));

        if (n.syllable.isNotEmpty() && n.syllable != "_")
        {
            // Karaoke convention: a trailing hyphen means the word continues on the next syllable.
            auto text = n.syllable;
            for (size_t j = i + 1; j < notes.size(); ++j)
            {
                if (notes[j].syllable == "_")
                    continue;
                if (notes[j].wordIndex == n.wordIndex)
                    text << "-";
                break;
            }
            track.addEvent (juce::MidiMessage::textMetaEvent (5, text), start);
        }

        const int noteNumber = juce::jlimit (0, 127, n.midiNote);
        track.addEvent (juce::MidiMessage::noteOn (1, noteNumber, (juce::uint8) 100), start);
        track.addEvent (juce::MidiMessage::noteOff (1, noteNumber), end);
        lastTick = juce::jmax (lastTick, end);
    }

    track.updateMatchedPairs();
    track.addEvent (juce::MidiMessage::endOfTrack(), lastTick);

    juce::MidiFile file;
    file.setTicksPerQuarterNote (ticksPerQuarter);
    file.addTrack (track);
    return file;
}

} // namespace voxa::sing
