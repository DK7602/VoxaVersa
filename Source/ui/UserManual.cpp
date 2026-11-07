#include "UserManual.h"
#include "voices/VoiceLibrary.h"
#include "VoxaAssets.h"

namespace voxa::ui
{

juce::String UserManual::getHtml()
{
    int size = 0;
    const auto* data = VoxaAssets::getNamedResource ("VoxaVersa_Manual_html", size);
    if (data == nullptr || size <= 0)
        return {};
    const juce::String version = juce::String (VOXAVERSA_VERSION_STRING) + " (build " + VOXAVERSA_BUILD_STRING + ")";
    return juce::String::fromUTF8 (data, size).replace ("{{VERSION}}", version);
}

juce::File UserManual::write (const juce::File& directory)
{
    const auto html = getHtml();
    if (html.isEmpty() || ! directory.createDirectory())
        return {};
    const auto file = directory.getChildFile ("VoxaVersa User Manual.html");
    if (file.existsAsFile() && file.loadFileAsString() == html)
        return file;
    return file.replaceWithText (html, false, false, "\n") ? file : juce::File();
}

bool UserManual::open()
{
    const auto file = write (VoiceLibrary::getUserDataDirectory().getChildFile ("Manual"));
    return file.existsAsFile() && file.startAsProcess();
}

} // namespace voxa::ui
