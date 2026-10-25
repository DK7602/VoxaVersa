#pragma once

#include "Theme.h"
#include "voices/VoiceProfile.h"
#include <map>

namespace voxa::ui
{

/** Renders the card artwork for a voice: the voice pack's avatar image when it has one, the bundled
    photo portrait of a factory voice (Resources/voices/<id>.jpg), or otherwise a stylised, rim-lit
    vector portrait over a starry nebula. */
juce::Image renderVoiceAvatar (const VoiceProfile& voice, int widthPx, int heightPx);

/** The bundled photo portrait of a factory voice, or an invalid image when there is none. */
juce::Image getFactoryPortrait (const VoiceProfile& voice);

/** Per-editor cache of rendered avatars, keyed by voice id and pixel size. */
class AvatarCache
{
public:
    const juce::Image& get (const VoiceProfile& voice, int widthPx, int heightPx);
    void clear() { images.clear(); }

private:
    std::map<juce::String, juce::Image> images;
};

} // namespace voxa::ui
