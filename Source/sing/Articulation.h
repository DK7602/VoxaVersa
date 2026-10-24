#pragma once

#include "Phonemes.h"
#include "VocalSynth.h"
#include "../voices/VoiceProfile.h"

namespace voxa::sing
{

/** Parameters that are interpolated between phonemes (coarticulation). */
enum ArticParam
{
    apF1, apF2, apF3,             // formant frequencies (smoothed in the log domain)
    apB1, apB2, apB3,             // formant bandwidths
    apUpperDamping,               // bandwidth multiplier for F4..F8 (muffled closures, nasal murmur)
    apNasalZero, apNasality,      // nasal anti-resonance
    apAV, apAH, apAF,             // voicing, aspiration and frication amplitudes
    apFricF1, apFricB1, apFricG1, // frication spectrum: two resonances and a flat bypass
    apFricF2, apFricB2, apFricG2,
    apFricBypass,
    apFricHighpass,               // high-pass corner of the frication noise (steep sibilant spectra)
    numArticParams
};

struct ArticFrame
{
    std::array<float, numArticParams> v {};
    bool formantsValid = true;    // false in silence: formants take the neighbours' values
};

/** Smoothing time (seconds) of each parameter; the offline renderer uses it as a Gaussian sigma
    (zero-phase), the live singer as the time constant of a causal two-pole smoother. */
float articSmoothingSeconds (int param) noexcept;
bool articParamIsFrequency (int param) noexcept;

/** Everything about a voice that the synthesizer needs, derived from VoiceTraits.
    Plain data: rebuilding it is real-time safe. */
struct VoiceSetup
{
    enum VowelTarget
    {
        vAA, vAE, vAH, vAO, vAX, vEH, vER, vIH, vIY, vUH, vUW,
        vEYa, vEYb, vOWa, vOWb, vAYa, vAYb, vAWa, vAWb, vOYa, vOYb,
        numVowelTargets
    };

    std::array<std::array<float, 3>, numVowelTargets> vowels {};   // F1..F3 in Hz, voice-scaled
    std::array<float, SynthFrame::numFormants> upperFormants {};   // F4..F8 at [3..7]
    std::array<float, SynthFrame::numFormants> upperBandwidths {};
    float consonantScale = 1.0f;    // scales consonant loci and noise spectra
    float bandwidthScale = 1.0f;

    SourceCharacter source;
    float baseRd = 1.0f;
    float voicingGain = 1.0f;       // < 1 for whispery voices
    float aspirationLevel = 0.0f;   // extra aspiration relative to voicing

    // Performance
    float vibratoRateHz = 5.5f;
    float vibratoDepthSemis = 0.2f;
    float vibratoOnsetSeconds = 0.25f;
    float scoopSemis = 0.0f;
    float scoopSeconds = 0.08f;
    float driftSemis = 0.03f;       // slow random pitch wander (RMS)
    float portamentoSeconds = 0.06f;
    float legato = 0.6f;
    float busy = 0.5f;
    float fallSemis = 0.3f;         // pitch drop when a phrase ends
    float loudnessReferenceDb = 0.0f;

    void prepare (const VoiceTraits&) noexcept;

    /** Gain that partly evens out how loud different vowels come out at a given pitch (the vocal
        tract amplifies harmonics that fall near formants, so "ee" at some pitches is far quieter
        than "ah"). Returns 1 for non-vowels. */
    float vowelLoudnessCompensation (Ph vowel, float f0) const noexcept;
    float vowelPowerDb (Ph vowel, float f0) const noexcept;

    void vowelFormants (Ph vowel, float glide, float* f) const noexcept;
};

/** One phoneme occurrence as the articulator sees it. */
struct PhoneEvent
{
    Ph ph = Ph::Sil;
    Ph contextVowel = Ph::AX;     // adjacent vowel: next for onsets, previous for codas
    float duration = 0.1f;        // seconds (a large value for held live vowels)
    float glideStart = 1.0e9f;    // diphthongs: seconds after the start where the off-glide begins
    float glideTime = 0.12f;
    bool released = true;         // stops: false = unreleased (no burst)
    bool aspirated = true;        // voiceless stops: long voice-onset time
    bool syllabic = false;        // nasal / liquid sung as the nucleus ("hmm")
};

/** Articulatory targets of a phoneme `t` seconds after its start. */
void articulate (const PhoneEvent&, float t, const VoiceSetup&, ArticFrame&) noexcept;

/** Assembles a synthesizer frame from (smoothed) articulation and prosody. */
void makeSynthFrame (const ArticFrame&, const VoiceSetup&, float f0, float rd, SynthFrame&) noexcept;

/** Formant-tuning used by trained singers: F1 is raised above f0 on high notes. */
float tunedFirstFormant (float f1, float f0) noexcept;

} // namespace voxa::sing
