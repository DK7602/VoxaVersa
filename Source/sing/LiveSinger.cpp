#include "SingEngine.h"
#include "Articulation.h"
#include "Lyrics.h"

namespace voxa::sing
{

namespace
{
    constexpr int maxClusterLength = 8;

    /** A lyric item as the live singer needs it: fixed-size, no further parsing on the audio thread. */
    struct LiveItem
    {
        std::array<Ph, maxClusterLength> onset {};
        int numOnset = 0;
        Ph nucleus = Ph::AA;
        std::array<Ph, maxClusterLength> coda {};
        int numCoda = 0;
        bool melisma = false;
        int syllableIndex = 0;
    };

    struct LiveLyrics
    {
        std::vector<LiveItem> items;
        int numSyllables = 0;
    };

    enum class Role : std::uint8_t
    {
        Onset,      // consonant before the vowel
        Hold,       // vowel held while the key is down
        Release,    // vowel fading out (with diphthong off-glide) after note-off
        Coda,       // consonant after the vowel
        Silence
    };

    struct LivePhase
    {
        Ph ph = Ph::Sil;
        Role role = Role::Silence;
        float duration = -1.0f;   // seconds, negative = until replaced
        bool aspirated = true;
        bool released = true;
    };

    float smoothstep (float x) noexcept
    {
        x = juce::jlimit (0.0f, 1.0f, x);
        return x * x * (3.0f - 2.0f * x);
    }
}

//==============================================================================
struct LiveSinger::Impl
{
    Impl()
    {
        setup.prepare (VoiceTraits());
        prepare (48000.0);
    }

    ~Impl()
    {
        drainRetired();
        delete pending.exchange (nullptr);
        delete active;
    }

    //==============================================================================
    // Lyrics hand-over: the message thread publishes through `pending`; the audio thread adopts it
    // and hands the old object back through a lock-free FIFO; only the message thread deletes.
    std::atomic<LiveLyrics*> pending { nullptr };
    LiveLyrics* active = nullptr;
    juce::AbstractFifo retiredFifo { 16 };
    std::array<LiveLyrics*, 16> retired {};

    std::atomic<int> numSyllables { 0 };
    std::atomic<int> currentSyllable { 0 };
    std::atomic<bool> soundingFlag { false };
    int nextItem = 0;

    void drainRetired()
    {
        const auto scope = retiredFifo.read (retiredFifo.getNumReady());
        scope.forEach ([this] (int index)
        {
            delete retired[(size_t) index];
            retired[(size_t) index] = nullptr;
        });
    }

    void setLyrics (const juce::String& text)
    {
        drainRetired();

        const auto script = parseLyrics (text);
        auto fresh = std::make_unique<LiveLyrics>();
        fresh->numSyllables = script.getNumSyllables();

        for (const auto& e : script.events)
        {
            if (e.type != LyricEvent::Type::Syllable && e.type != LyricEvent::Type::Melisma)
                continue;

            const auto& syl = script.syllables[(size_t) e.syllable];
            LiveItem item;
            item.nucleus = syl.nucleus;
            item.syllableIndex = e.syllable;
            item.melisma = e.type == LyricEvent::Type::Melisma;

            if (! item.melisma)
            {
                for (auto p : syl.onset)
                    if (item.numOnset < maxClusterLength)
                        item.onset[(size_t) item.numOnset++] = p;
                for (auto p : syl.coda)
                    if (item.numCoda < maxClusterLength)
                        item.coda[(size_t) item.numCoda++] = p;
            }
            fresh->items.push_back (item);
        }

        numSyllables.store (fresh->numSyllables);
        currentSyllable.store (0);

        // An unconsumed earlier version was never seen by the audio thread: safe to delete here.
        delete pending.exchange (fresh.release(), std::memory_order_acq_rel);
    }

    void adoptPendingLyrics() noexcept
    {
        if (pending.load (std::memory_order_relaxed) == nullptr)
            return;

        if (auto* fresh = pending.exchange (nullptr, std::memory_order_acq_rel))
        {
            if (active != nullptr)
            {
                const auto scope = retiredFifo.write (1);
                if (scope.blockSize1 > 0)
                    retired[(size_t) scope.startIndex1] = active;
                else if (scope.blockSize2 > 0)
                    retired[(size_t) scope.startIndex2] = active;
                // (A full FIFO would leak rather than free on the audio thread; it holds at most one.)
            }
            active = fresh;
            nextItem = 0;
            publishSyllable();
        }
    }

    void publishSyllable() noexcept
    {
        if (active == nullptr || active->items.empty())
        {
            currentSyllable.store (0, std::memory_order_relaxed);
            return;
        }
        nextItem %= (int) active->items.size();
        currentSyllable.store (active->items[(size_t) nextItem].syllableIndex, std::memory_order_relaxed);
    }

    LiveItem takeNextItem() noexcept
    {
        if (active == nullptr || active->items.empty())
            return {};   // neutral "ah"

        nextItem %= (int) active->items.size();
        const auto item = active->items[(size_t) nextItem];
        ++nextItem;
        publishSyllable();
        return item;
    }

    //==============================================================================
    double sampleRate = 48000.0;
    int frameLength = 48;
    float frameSeconds = 0.001f;
    int samplesUntilFrame = 0;

    VocalSynth synth;
    VoiceSetup setup;

    void prepare (double newSampleRate)
    {
        sampleRate = newSampleRate > 0.0 ? newSampleRate : 48000.0;
        frameLength = juce::jmax (16, (int) std::round (sampleRate / 1000.0));
        frameSeconds = (float) (frameLength / sampleRate);
        synth.prepare (sampleRate, 0x5eed);
        synth.setSourceCharacter (setup.source);

        for (int p = 0; p < numArticParams; ++p)
            smoothingCoefficient[(size_t) p] = 1.0f - std::exp (-frameSeconds / juce::jmax (1.0e-4f, 0.6f * articSmoothingSeconds (p)));

        reset();
    }

    void setVoice (const VoiceTraits& traits) noexcept
    {
        setup.prepare (traits);
        synth.setSourceCharacter (setup.source);
    }

    //==============================================================================
    // Notes (monophonic, last-note priority)
    std::array<int, 32> held {};
    int numHeld = 0;
    int currentNote = -1;
    bool gate = false;
    float velocity = 0.8f;
    LiveItem currentItem;

    void removeHeld (int note) noexcept
    {
        int w = 0;
        for (int r = 0; r < numHeld; ++r)
            if (held[(size_t) r] != note)
                held[(size_t) w++] = held[(size_t) r];
        numHeld = w;
    }

    void noteOn (int note, float vel) noexcept
    {
        adoptPendingLyrics();
        note = juce::jlimit (0, 127, note);

        removeHeld (note);
        if (numHeld == (int) held.size())
        {
            for (int i = 1; i < numHeld; ++i)
                held[(size_t) i - 1] = held[(size_t) i];
            --numHeld;
        }
        held[(size_t) numHeld++] = note;

        const bool legato = gate && currentNote >= 0;
        const auto item = takeNextItem();
        velocity = juce::jlimit (0.05f, 1.0f, vel);
        currentNote = note;
        gate = true;
        vowelCompensation = setup.vowelLoudnessCompensation (item.nucleus, music::midiToHz ((float) note));

        if (legato)
        {
            startGlide ((float) note);
            if (! item.melisma)
            {
                queueSize = 0;
                queuePos = 0;
                phaseTime = 0.0f;
                appendCluster (currentItem.coda.data(), currentItem.numCoda, Role::Coda, 0.6f, 0.07f);
                appendCluster (item.onset.data(), item.numOnset, Role::Onset, 0.6f, 0.08f);
                pushPhase ({ item.nucleus, Role::Hold, -1.0f });
                currentItem = item;
                attackTime = 0.015f;
                attackFloor = 0.6f;
            }
        }
        else
        {
            startAttackPitch ((float) note);
            queueSize = 0;
            queuePos = 0;
            phaseTime = 0.0f;
            appendCluster (item.onset.data(), item.numOnset, Role::Onset, 0.7f, 0.09f);
            pushPhase ({ item.nucleus, Role::Hold, -1.0f });
            currentItem = item;

            if (item.numOnset == 0)
            {
                attackTime = juce::jmap (velocity, 0.07f, 0.03f) + 0.03f * setup.source.breathiness;
                attackFloor = 0.0f;
            }
            else
            {
                const auto last = item.onset[(size_t) item.numOnset - 1];
                attackTime = isSonorant (last) ? 0.012f : 0.02f;
                attackFloor = isSonorant (last) ? 0.5f : 0.0f;
            }
            envelope = juce::jmin (envelope, attackFloor);
        }
    }

    void noteOff (int note) noexcept
    {
        removeHeld (note);
        if (note != currentNote || ! gate)
            return;

        if (numHeld > 0)
        {
            // Fall back to the most recent key still held: glide there on the same vowel.
            currentNote = held[(size_t) numHeld - 1];
            startGlide ((float) currentNote);
            return;
        }

        gate = false;
        startRelease();
    }

    void allNotesOff() noexcept
    {
        numHeld = 0;
        if (gate)
        {
            gate = false;
            startRelease();
        }
    }

    void startRelease() noexcept
    {
        // Keep any onset consonants that are still to come, then fade the vowel and sing the coda.
        LivePhase remaining[maxClusterLength];
        int numRemaining = 0;
        if (queuePos < queueSize && queue[(size_t) queuePos].role == Role::Onset)
            for (int i = queuePos; i < queueSize && queue[(size_t) i].role == Role::Onset && numRemaining < maxClusterLength; ++i)
                remaining[numRemaining++] = queue[(size_t) i];

        const float elapsed = numRemaining > 0 ? phaseTime : 0.0f;
        queueSize = 0;
        queuePos = 0;
        for (int i = 0; i < numRemaining; ++i)
            pushPhase (remaining[i]);
        phaseTime = elapsed;

        const bool hasCoda = currentItem.numCoda > 0;
        const float releaseTime = hasCoda ? 0.06f : juce::jmap (setup.legato, 0.1f, 0.2f);
        pushPhase ({ currentItem.nucleus, Role::Release, releaseTime });
        appendCluster (currentItem.coda.data(), currentItem.numCoda, Role::Coda, 0.9f, 0.3f);
        pushPhase ({ Ph::Sil, Role::Silence, -1.0f });
        releaseFloor = hasCoda ? 0.35f : 0.0f;
        // Released before the vowel was reached (a quick tap): still sing a short vowel.
        releaseStart = numRemaining > 0 ? juce::jmax (envelope, 0.8f) : envelope;
        fallActive = true;
        fallElapsed = 0.0f;
    }

    //==============================================================================
    // Phase queue
    std::array<LivePhase, 40> queue {};
    int queueSize = 0, queuePos = 0;
    float phaseTime = 0.0f;

    void pushPhase (const LivePhase& p) noexcept
    {
        if (queueSize < (int) queue.size())
            queue[(size_t) queueSize++] = p;
    }

    void appendCluster (const Ph* phones, int count, Role role, float scale, float maxTotal) noexcept
    {
        if (count <= 0)
            return;

        float total = 0.0f;
        for (int i = 0; i < count; ++i)
        {
            const auto& info = phonemeInfo (phones[i]);
            total += (role == Role::Coda ? info.codaMs : info.onsetMs) * 0.001f * scale;
        }
        const float squeeze = total > maxTotal ? maxTotal / total : 1.0f;

        for (int i = 0; i < count; ++i)
        {
            const auto& info = phonemeInfo (phones[i]);
            LivePhase p;
            p.ph = phones[i];
            p.role = role;
            p.duration = juce::jmax (0.6f * info.minMs * 0.001f,
                                     (role == Role::Coda ? info.codaMs : info.onsetMs) * 0.001f * scale * squeeze);
            p.aspirated = role == Role::Onset && ! (i > 0 && phones[i - 1] == Ph::S);
            p.released = true;
            pushPhase (p);
        }
    }

    //==============================================================================
    // Envelope
    float envelope = 0.0f, attackTime = 0.05f, attackFloor = 0.0f;
    float vowelCompensation = 1.0f;
    float releaseStart = 1.0f, releaseFloor = 0.0f;

    //==============================================================================
    // Pitch
    float pitch = 60.0f;            // base pitch without ornaments (semitones)
    float glideFrom = 60.0f, glideTo = 60.0f, glideTime = 0.0f, glideElapsed = 1.0f;
    float overshoot = 0.0f, overshootElapsed = 10.0f;
    float scoopElapsed = 10.0f;
    float noteTime = 0.0f, vibratoPhase = 0.0f;
    float driftA = 0.0f, driftB = 0.0f;
    bool fallActive = false;
    float fallElapsed = 0.0f;
    NoiseSource driftNoise;

    void startGlide (float target) noexcept
    {
        glideFrom = pitch;
        glideTo = target;
        const float interval = std::abs (target - pitch);
        glideTime = juce::jmax (0.01f, setup.portamentoSeconds * (1.0f + 0.03f * interval));
        glideElapsed = 0.0f;
        overshoot = interval >= 3.0f ? (target > pitch ? 1.0f : -1.0f) * juce::jmin (0.35f, 0.05f * interval) : 0.0f;
        overshootElapsed = -glideTime * 0.8f;
        noteTime = 0.0f;
        fallActive = false;
    }

    void startAttackPitch (float target) noexcept
    {
        pitch = glideFrom = glideTo = target;
        glideElapsed = 1.0f;
        glideTime = 0.0f;
        overshoot = 0.0f;
        scoopElapsed = 0.0f;
        noteTime = 0.0f;
        vibratoPhase = 0.0f;
        fallActive = false;
    }

    float updatePitch (float dt) noexcept
    {
        if (glideElapsed < glideTime)
        {
            glideElapsed += dt;
            pitch = glideFrom + (glideTo - glideFrom) * smoothstep (glideElapsed / glideTime);
        }
        else
        {
            pitch = glideTo;
        }

        float result = pitch;

        if (std::abs (overshoot) > 0.0f)
        {
            overshootElapsed += dt;
            const float x = overshootElapsed / 0.06f;
            if (x > 0.0f)
                result += overshoot * x * std::exp (1.0f - x);
            if (x > 6.0f)
                overshoot = 0.0f;
        }

        if (setup.scoopSemis > 0.0f && scoopElapsed < 2.0f)
        {
            scoopElapsed += dt;
            result -= setup.scoopSemis * std::exp (-scoopElapsed / (setup.scoopSeconds / 3.0f));
        }

        noteTime += dt;
        vibratoPhase += juce::MathConstants<float>::twoPi * setup.vibratoRateHz * (1.0f + 0.04f * driftA * 10.0f) * dt;
        if (vibratoPhase > juce::MathConstants<float>::twoPi)
            vibratoPhase -= juce::MathConstants<float>::twoPi;
        const float vibratoEnvelope = smoothstep ((noteTime - setup.vibratoOnsetSeconds) / 0.3f);
        result += setup.vibratoDepthSemis * vibratoEnvelope * std::sin (vibratoPhase);

        // Slow drift: doubly low-passed noise.
        const float c = 1.0f - std::exp (-dt * 6.0f);
        driftA += c * (driftNoise.next() * 2.5f - driftA);
        driftB += c * (driftA - driftB);
        result += setup.driftSemis * 3.0f * driftB;

        if (fallActive)
        {
            fallElapsed += dt;
            result -= setup.fallSemis * smoothstep ((fallElapsed - 0.05f) / 0.15f);
        }
        return result;
    }

    //==============================================================================
    std::array<float, numArticParams> smoothingCoefficient {};
    ArticFrame stage1, stage2, raw;
    SynthFrame synthFrame;
    bool smootherPrimed = false;

    void controlFrame() noexcept
    {
        const float dt = frameSeconds;

        if (queueSize == 0)
            pushPhase ({ Ph::Sil, Role::Silence, -1.0f });

        phaseTime += dt;
        while (queuePos + 1 < queueSize && queue[(size_t) queuePos].duration >= 0.0f
               && phaseTime >= queue[(size_t) queuePos].duration)
        {
            phaseTime -= queue[(size_t) queuePos].duration;
            ++queuePos;
        }

        const auto& phase = queue[(size_t) queuePos];
        const bool vowelRole = phase.role == Role::Hold || phase.role == Role::Release;

        PhoneEvent e;
        e.ph = phase.ph;
        e.contextVowel = isVowel (currentItem.nucleus) ? currentItem.nucleus : Ph::AX;
        e.duration = phase.duration > 0.0f ? phase.duration : 10.0f;
        e.aspirated = phase.aspirated;
        e.released = phase.released;
        e.syllabic = vowelRole && ! isVowel (phase.ph);
        if (phase.role == Role::Release && isDiphthong (phase.ph))
        {
            e.glideStart = 0.0f;
            e.glideTime = juce::jmax (0.04f, 0.9f * phase.duration);
        }

        articulate (e, phaseTime, setup, raw);

        // Vowel envelope
        if (phase.role == Role::Hold)
        {
            envelope = juce::jmin (1.0f, envelope + dt / juce::jmax (0.005f, attackTime) * (1.0f - attackFloor + 0.05f));
            envelope = juce::jmax (envelope, attackFloor);
        }
        else if (phase.role == Role::Release)
        {
            const float x = phaseTime / juce::jmax (0.01f, phase.duration);
            envelope = releaseFloor + (releaseStart - releaseFloor) * (1.0f - smoothstep (x));
        }
        else if (phase.role == Role::Onset)
        {
            envelope = attackFloor;
        }

        const float level = juce::Decibels::decibelsToGain (-16.0f * std::pow (1.0f - velocity, 1.2f));
        const float vowelGain = vowelRole ? envelope * vowelCompensation : 1.0f;
        raw.v[apAV] *= vowelGain * level;
        raw.v[apAH] *= vowelGain * level;
        raw.v[apAF] *= level;

        if (! smootherPrimed)
        {
            stage1 = stage2 = raw;
            stage1.v[apAV] = stage2.v[apAV] = 0.0f;
            stage1.v[apAH] = stage2.v[apAH] = 0.0f;
            stage1.v[apAF] = stage2.v[apAF] = 0.0f;
            smootherPrimed = true;
        }

        for (int p = 0; p < numArticParams; ++p)
        {
            const auto index = (size_t) p;
            const bool shape = ! (p == apAV || p == apAH || p == apAF || p == apNasality
                                  || p == apFricG1 || p == apFricG2 || p == apFricBypass);
            float target = raw.v[index];
            if (! raw.formantsValid && shape)
                target = stage2.v[index];    // silence: hold the last articulation

            const float c = smoothingCoefficient[index];
            stage1.v[index] += c * (target - stage1.v[index]);
            stage2.v[index] += c * (stage1.v[index] - stage2.v[index]);
        }

        const float f0 = music::midiToHz (updatePitch (dt));
        const float rd = juce::jlimit (0.3f, 2.7f, setup.baseRd + 0.7f * (0.7f - velocity));
        makeSynthFrame (stage2, setup, f0, rd, synthFrame);

        const bool silentPhase = phase.role == Role::Silence;
        const float activity = stage2.v[apAV] + stage2.v[apAH] + stage2.v[apAF];
        soundingFlag.store (! silentPhase || activity > 1.0e-4f, std::memory_order_relaxed);
    }

    void reset() noexcept
    {
        queueSize = 0;
        queuePos = 0;
        phaseTime = 0.0f;
        pushPhase ({ Ph::Sil, Role::Silence, -1.0f });
        numHeld = 0;
        gate = false;
        currentNote = -1;
        envelope = 0.0f;
        smootherPrimed = false;
        samplesUntilFrame = 0;
        synth.reset();
        soundingFlag.store (false);
        driftA = driftB = 0.0f;
    }

    void render (float* output, int numSamples) noexcept
    {
        juce::ScopedNoDenormals noDenormals;
        adoptPendingLyrics();

        int done = 0;
        while (done < numSamples)
        {
            if (samplesUntilFrame <= 0)
            {
                controlFrame();
                samplesUntilFrame = frameLength;
            }

            const int chunk = juce::jmin (numSamples - done, samplesUntilFrame);
            if (soundingFlag.load (std::memory_order_relaxed))
            {
                synth.process (synthFrame, output + done, chunk);
                juce::FloatVectorOperations::multiply (output + done, outputGain, chunk);
            }
            else
            {
                juce::FloatVectorOperations::clear (output + done, chunk);
            }

            done += chunk;
            samplesUntilFrame -= chunk;
        }
    }

    // Puts a sustained vowel at velocity ~0.8 around -12 dBFS peak (same calibration as offline).
    static constexpr float outputGain = 0.27f;
};

//==============================================================================
LiveSinger::LiveSinger() : impl (std::make_unique<Impl>()) {}
LiveSinger::~LiveSinger() = default;

void LiveSinger::prepare (double sampleRate, int)    { impl->prepare (sampleRate); }
void LiveSinger::reset()                              { impl->reset(); }
void LiveSinger::setLyrics (const juce::String& l)    { impl->setLyrics (l); }
void LiveSinger::setVoice (const VoiceTraits& t)      { impl->setVoice (t); }
void LiveSinger::noteOn (int note, float velocity)    { impl->noteOn (note, velocity); }
void LiveSinger::noteOff (int note)                   { impl->noteOff (note); }
void LiveSinger::allNotesOff()                        { impl->allNotesOff(); }
void LiveSinger::render (float* output, int n)        { impl->render (output, n); }

void LiveSinger::restartLyrics()
{
    impl->adoptPendingLyrics();
    impl->nextItem = 0;
    impl->publishSyllable();
}

int LiveSinger::getCurrentSyllableIndex() const noexcept   { return impl->currentSyllable.load (std::memory_order_relaxed); }
int LiveSinger::getNumSyllables() const noexcept           { return impl->numSyllables.load (std::memory_order_relaxed); }
bool LiveSinger::isSounding() const noexcept               { return impl->soundingFlag.load (std::memory_order_relaxed); }

} // namespace voxa::sing
