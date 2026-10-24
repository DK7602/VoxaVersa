#include "PresetManager.h"
#include "../Parameters.h"

namespace voxa
{

std::vector<FactoryPreset> createFactoryPresets()
{
    namespace P = ParamID;

    // Choice parameters take their index; bools take 0/1.
    return {
        { "Init Preset", "Basics", "female-pop", {} },

        { "Natural Male to Female", "Gender Swap", "female-pop",
          { { P::pitchPreserve, 0.0f }, { P::intensity, 90.0f }, { P::sourceVoice, 1.0f } } },

        { "Natural Female to Male", "Gender Swap", "male-pop",
          { { P::pitchPreserve, 0.0f }, { P::intensity, 90.0f }, { P::sourceVoice, 2.0f } } },

        { "Radio Pop Star", "Pop", "female-pop",
          { { P::tuneOn, 1.0f }, { P::retuneSpeed, 30.0f }, { P::humanize, 40.0f },
            { P::doublerOn, 1.0f }, { P::doublerAmount, 35.0f }, { P::presence, 2.0f }, { P::air, 3.0f },
            { P::delayMix, 12.0f }, { P::delayTime, 1.0f }, { P::reverbMix, 18.0f } } },

        { "Trap Hard Tune", "Hip-Hop", "male-trap",
          { { P::tuneOn, 1.0f }, { P::retuneSpeed, 0.0f }, { P::humanize, 0.0f },
            { P::delayMix, 15.0f }, { P::delayTime, 0.0f }, { P::reverbMix, 16.0f }, { P::compressor, 55.0f } } },

        { "Silky R&B Harmonies", "R&B", "female-rnb",
          { { P::harmonyOn, 1.0f }, { P::harm1Interval, 7.0f }, { P::harm1Level, 55.0f },
            { P::harm2Interval, 9.0f }, { P::harm2Level, 45.0f }, { P::reverbMix, 24.0f }, { P::vibratoDepth, 10.0f } } },

        { "Gospel Choir Stack", "Gospel", "female-gospel",
          { { P::harmonyOn, 1.0f }, { P::harm1Interval, 7.0f }, { P::harm2Interval, 9.0f }, { P::harm3Interval, 1.0f },
            { P::harm3Level, 45.0f }, { P::doublerOn, 1.0f }, { P::doublerAmount, 60.0f }, { P::reverbMix, 32.0f },
            { P::reverbSize, 80.0f } } },

        { "Country Twang", "Country", "male-country",
          { { P::presence, 1.5f }, { P::delayMix, 10.0f }, { P::delayTime, 3.0f }, { P::reverbMix, 15.0f } } },

        { "Deep Voice", "Character", "male-deep-bass",
          { { P::pitch, -5.0f }, { P::gender, -60.0f }, { P::tone, 30.0f } } },

        { "Chipmunk", "Character", "female-kpop",
          { { P::pitch, 12.0f }, { P::gender, 100.0f }, { P::age, -100.0f }, { P::tone, -40.0f } } },

        { "Robot Hard Tune", "Character", "neutral-robot",
          { { P::tuneOn, 1.0f }, { P::retuneSpeed, 0.0f }, { P::humanize, 0.0f }, { P::scale, 2.0f },
            { P::style, 100.0f }, { P::saturation, 25.0f } } },

        { "Lo-Fi Bedroom", "Lo-Fi", "female-lofi",
          { { P::lowCut, 160.0f }, { P::air, -4.0f }, { P::saturation, 30.0f }, { P::reverbMix, 22.0f } } },

        { "Stadium Rock", "Rock", "male-rock",
          { { P::saturation, 35.0f }, { P::doublerOn, 1.0f }, { P::doublerAmount, 60.0f },
            { P::reverbMix, 28.0f }, { P::reverbSize, 75.0f }, { P::compressor, 55.0f } } },

        { "Opera Hall", "Classical", "female-opera",
          { { P::reverbMix, 42.0f }, { P::reverbSize, 90.0f }, { P::vibratoDepth, 15.0f } } },

        { "Whisper ASMR", "Character", "neutral-whisper",
          { { P::breath, 70.0f }, { P::compressor, 60.0f }, { P::reverbMix, 6.0f } } },

        { "Classic Soul", "Soul", "female-soul",
          { { P::saturation, 15.0f }, { P::reverbMix, 20.0f }, { P::rasp, 15.0f } } },

        { "K-Pop Idol", "Pop", "female-kpop",
          { { P::tuneOn, 1.0f }, { P::retuneSpeed, 25.0f }, { P::doublerOn, 1.0f }, { P::doublerAmount, 40.0f },
            { P::air, 3.0f } } },

        { "Afrobeats Vibe", "World", "male-afrobeats",
          { { P::tuneOn, 1.0f }, { P::retuneSpeed, 20.0f }, { P::delayMix, 14.0f }, { P::delayTime, 2.0f } } },

        { "Rap Double", "Hip-Hop", "male-rap",
          { { P::doublerOn, 1.0f }, { P::doublerAmount, 70.0f }, { P::doublerWidth, 100.0f }, { P::compressor, 60.0f } } },

        { "Jazz Club", "Jazz", "male-jazz",
          { { P::reverbMix, 24.0f }, { P::reverbSize, 40.0f }, { P::tone, 20.0f } } },

        { "Angelic Choir", "Choir", "neutral-choir",
          { { P::harmonyOn, 1.0f }, { P::harm1Interval, 7.0f }, { P::harm2Interval, 11.0f }, { P::harm3Interval, 1.0f },
            { P::harm3Level, 40.0f }, { P::doublerOn, 1.0f }, { P::doublerAmount, 80.0f }, { P::reverbMix, 45.0f },
            { P::reverbSize, 95.0f } } },

        { "Live Stage (Low Latency)", "Performance", "female-pop",
          { { P::quality, 0.0f }, { P::reverbMix, 10.0f } } },
    };
}

} // namespace voxa
