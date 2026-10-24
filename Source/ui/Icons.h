#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace voxa::ui
{

/** Vector line icons, designed on a 24 x 24 grid. */
enum class Icon
{
    Gear, Mic, Document, Save, Undo, Redo, Power, Link, Star, StarFilled, Search, List, Grid,
    Play, Stop, Pause, Record, Note, ChevronLeft, ChevronRight, ChevronDown, ChevronUp,
    AutoLevel, PitchPreserve, FormantShift, Expression, Vibe, Drag, Folder, Close, Sync,
    Midi, Export, Sparkle, TuningFork, Info, Trash, Plus, Menu, Waveform
};

namespace Icons
{
    struct Shape
    {
        juce::Path stroke;   // drawn with a stroke
        juce::Path fill;     // filled
    };

    /** The icon's shapes in a 24 x 24 box (cached). */
    const Shape& get (Icon);

    /** Draws an icon fitted (centred, aspect preserved) into `area`. */
    void draw (juce::Graphics& g, Icon icon, juce::Rectangle<float> area, juce::Colour colour,
               float strokeWidth = 1.7f, bool glow = false);
}

} // namespace voxa::ui
