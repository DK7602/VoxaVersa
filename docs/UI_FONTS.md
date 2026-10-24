# UI fonts

Every `*.ttf` / `*.otf` file in `Resources/fonts/` is compiled into the `VoxaAssets` binary data
target by CMake and loaded by `Source/ui/Theme.cpp` (`voxa::ui::font()`), so the plugin renders
identically on every platform without depending on system fonts.

| File | Family / weight | Used for | Licence |
| --- | --- | --- | --- |
| `Resources/fonts/BarlowSemiCondensed-Regular.ttf` | Barlow Semi Condensed Regular | body text, captions, values | SIL OFL 1.1 |
| `Resources/fonts/BarlowSemiCondensed-Medium.ttf` | Barlow Semi Condensed Medium | labels, buttons, knob names | SIL OFL 1.1 |
| `Resources/fonts/BarlowSemiCondensed-SemiBold.ttf` | Barlow Semi Condensed SemiBold | emphasis, badges, read-outs | SIL OFL 1.1 |
| `Resources/fonts/Barlow-Medium.ttf` | Barlow Medium | upper-case panel titles and tabs | SIL OFL 1.1 |

- Designer: Jeremy Tribby — Copyright 2017 The Barlow Project Authors (https://github.com/jpt/barlow)
- Source: the `google/fonts` repository (`ofl/barlow`, `ofl/barlowsemicondensed`)
- Licence: SIL Open Font License, Version 1.1 — full text in `Resources/fonts/OFL-Barlow.txt`

The OFL allows the fonts to be embedded in and distributed with the plugin (including
commercially); the font software itself may not be sold on its own, and the licence text must
accompany it.

Barlow was chosen because its slightly condensed, technical grotesque shapes match the approved
mockup (`docs/images/ui-mockup.png`) closely at small sizes.
