# Installing VoxaVersa

Every build on GitHub produces ready-made installers. They include the AI voice engine (ONNX Runtime; on
Windows the DirectML version, which uses any DirectX 12 graphics card and falls back to the CPU), so there is
nothing else to install for AI voices except the voices themselves.

## Get the installer

1. Open the repository's **Actions** tab on GitHub, click the newest green **Build & Test** run.
2. Under **Artifacts**, download **VoxaVersa-Windows-Installer** (or **VoxaVersa-macOS-Installer**) and unzip it.

## Windows

1. Close your DAW (a running DAW keeps the old plug-in locked; Setup offers to close it), then run
   `VoxaVersa-<version>.<build>-Windows-Setup.exe`.
2. Windows may show **"Windows protected your PC"** because the installer is not code-signed yet: click
   **More info > Run anyway**.
3. Keep **Everything** selected and click through. It installs:
   * the VST3 plug-ins **VoxaVersa** (effect) and **VoxaVersa Singer** (instrument) into
     `C:\Program Files\Common Files\VST3`
   * the **VoxaVersa** app (Start menu, optional desktop shortcut)
   * the Microsoft Visual C++ runtime, if your PC doesn't have a recent one
4. In your DAW, rescan plug-ins.

Which build is running: menu (gear) > **About VoxaVersa...** shows `1.0.0 (build N)`; N is the number of
the GitHub Actions run the installer came from. If the DAW shows an older build than the standalone app,
the DAW still has the old plug-in: close it completely and run the installer again, and check that there is
no second copy of `VoxaVersa.vst3` in another plug-in folder (in Cubase: *Studio > VST Plug-in Manager*,
select VoxaVersa, the info panel shows the file it loads).

Uninstall from *Settings > Apps*. Your voices, presets and settings in `%APPDATA%\VoxaVersa` are kept.

## macOS

1. Right-click `VoxaVersa-<version>-macOS.pkg` > **Open** (needed once, because it is not notarised yet), then
   follow the installer. It installs the VST3 and AU plug-ins into `/Library/Audio/Plug-Ins` and the app into
   `/Applications`.
2. If a DAW refuses to load the plug-in, open *System Settings > Privacy & Security* and allow it there.

## Add your AI voices

Menu (top right) > **Import Voices...** and pick the zip of trained voices from Google Drive: see
[COLAB.md, part 3](COLAB.md#3-installing-the-voices-in-voxaversa). You can also pick single `.vxpack` files.

The display in the middle of the window shows the engine a voice uses: **AI** (*neural voice*) or **DSP**
(*no AI model* / *AI off*); *About* in the menu shows whether the AI engine runs on the GPU (DirectML) or the
CPU. If live AI can't keep up on your computer, use **Render**: it gives the full AI quality without real-time
limits.

## For distribution (not done yet)

* Code signing: an Authenticode certificate for Windows, an Apple Developer ID + notarisation for macOS.
* AAX (Pro Tools) needs Avid's SDK and PACE signing.
* The Windows installer is built with Inno Setup, whose current versions are free for non-commercial use only:
  buy its commercial licence before selling VoxaVersa, or switch the installer to a free tool (WiX, NSIS).
* Licences of trained voices: VocalSet voices must credit VocalSet (CC BY 4.0).
