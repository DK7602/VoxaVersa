#pragma once

#include "Theme.h"
#include "voices/VoiceProfile.h"
#include <map>

namespace voxa::ui
{

/** Renders a stylised, rim-lit vector portrait for a voice over a starry nebula
    (or the voice pack's avatar image when it has one). */
juce::Image renderVoiceAvatar (const VoiceProfile& voice, int widthPx, int heightPx);

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
