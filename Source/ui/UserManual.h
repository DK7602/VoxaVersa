#pragma once

#include <juce_core/juce_core.h>

namespace voxa::ui
{

/** The user manual (Resources/manual/VoxaVersa_Manual.html, built into the plug-in): menu > User Manual shows it
    inside VoxaVersa (ManualView); its "Open in Browser" writes it to <app data>/VoxaVersa/Manual/VoxaVersa User
    Manual.html, with this build's version, and opens it in the default browser. One self-contained page. */
struct UserManual
{
    /** The manual's page with the version filled in; empty when the build has none. */
    static juce::String getHtml();
    /** Writes the page (when it differs from what is there) and returns it; an empty File when that fails. */
    static juce::File write (const juce::File& directory);
    /** Writes it to the app data folder and opens it in the default browser. False when that fails. */
    static bool open();
};

} // namespace voxa::ui
