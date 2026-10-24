# Building VoxaVersa

VoxaVersa is a C++20 / [JUCE 8](https://juce.com) project built with CMake. One build produces:

| Target | Formats | What it is |
|---|---|---|
| **VoxaVersa** | VST3, AU (macOS), LV2 (Linux), AAX (with SDK), **Standalone app** | Audio effect with MIDI input: Voice Transform + Text to Sing |
| **VoxaVersa Singer** | VST3, AU, LV2, AAX | Instrument variant: put it on a MIDI/instrument track, it sings your lyrics |
| **VoxaVersaTests** | console app | Unit + integration tests |

## Requirements

- CMake ≥ 3.22 and a C++20 compiler
  - **Windows:** Visual Studio 2022 (Desktop C++ workload)
  - **macOS:** Xcode 15+ (Command Line Tools), macOS 11+ deployment target, builds Universal (arm64 + x86_64)
  - **Linux:** GCC 11+ or Clang 14+, plus the JUCE dependencies:
    ```bash
    sudo apt-get install ninja-build libasound2-dev libjack-jackd2-dev ladspa-sdk libcurl4-openssl-dev \
      libfreetype6-dev libfontconfig1-dev libx11-dev libxcomposite-dev libxcursor-dev libxext-dev \
      libxinerama-dev libxrandr-dev libxrender-dev libgl1-mesa-dev
    ```
- Internet access on first configure (JUCE 8.0.9 is downloaded automatically), or pass
  `-DFETCHCONTENT_SOURCE_DIR_JUCE=/path/to/JUCE` to use a local checkout.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release          # add -G Ninja on macOS/Linux if you like
cmake --build build --config Release --parallel
```

Artefacts end up in `build/VoxaVersa_artefacts/Release/<Format>/` and
`build/VoxaVersaSinger_artefacts/Release/<Format>/`.

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `VOXAVERSA_BUILD_SINGER` | `ON` | Also build the instrument variant |
| `VOXAVERSA_BUILD_TESTS` | `ON` | Build the test runner |
| `VOXAVERSA_AAX_SDK_PATH` | – | Path to the Avid AAX SDK; enables the AAX format (Pro Tools). Shipping AAX also requires PACE signing. |
| `FETCHCONTENT_SOURCE_DIR_JUCE` | – | Use a local JUCE checkout instead of downloading |

## Run the tests

```bash
./build/VoxaVersaTests_artefacts/Release/VoxaVersaTests          # everything
./build/VoxaVersaTests_artefacts/Release/VoxaVersaTests DSP      # one category: Core, DSP, Processor, Sing, Neural
```

## Install the plugins

Copy the bundles to your system plug-in folders (or set `COPY_PLUGIN_AFTER_BUILD TRUE` in `CMakeLists.txt`):

| OS | VST3 | AU | LV2 | AAX |
|---|---|---|---|---|
| Windows | `C:\Program Files\Common Files\VST3` | – | – | `C:\Program Files\Common Files\Avid\Audio\Plug-Ins` |
| macOS | `~/Library/Audio/Plug-Ins/VST3` | `~/Library/Audio/Plug-Ins/Components` | – | `/Library/Application Support/Avid/Audio/Plug-Ins` |
| Linux | `~/.vst3` | – | `~/.lv2` | – |

The standalone app runs directly (`VoxaVersa.app`, `VoxaVersa.exe`, or `VoxaVersa` on Linux). Use
**Options → Audio/MIDI Settings** in its window to pick your microphone, interface and MIDI keyboard.

## Continuous integration

`.github/workflows/build.yml` builds and tests on Windows, macOS and Linux for every push and uploads
the plug-ins as workflow artefacts. Code signing / notarization (macOS) and installers are not part of
the workflow yet. Add them with your Apple Developer ID / Windows certificate before distributing.

## Licensing notes

- JUCE 8 is dual-licensed (AGPLv3 or a commercial JUCE licence). Closed-source commercial releases need
  the appropriate JUCE licence.
- The VST3 SDK bundled with JUCE is covered by Steinberg's licence (see `VST3_SDK/LICENSE.txt` inside JUCE). Check
  Steinberg's current terms before distributing. AAX requires an Avid developer agreement.
- See `docs/THIRD_PARTY.md` for bundled data and fonts.
