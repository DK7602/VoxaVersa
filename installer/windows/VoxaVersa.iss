; VoxaVersa Windows installer (Inno Setup 6). Built by CI (.github/workflows/build.yml):
;
;   iscc /DAppVersion=1.0.0 /DSourceDir=<dist> /DRedistDir=<folder with vc_redist.x64.exe> [/DIconFile=<icon.ico>] VoxaVersa.iss
;
; <dist> is the CI artefact layout: VST3\VoxaVersa.vst3, VST3\VoxaVersa Singer.vst3 and Standalone\VoxaVersa.exe,
; each with ONNX Runtime (DirectML build) + DirectML.dll next to the binary. Installs:
;   - the VST3 effect and instrument into C:\Program Files\Common Files\VST3
;   - the standalone app into C:\Program Files\VoxaVersa (+ Start menu, optional desktop shortcut)
;   - the Microsoft Visual C++ runtime that ONNX Runtime needs (skipped if already up to date)
; Voices, presets and settings live in %APPDATA%\VoxaVersa and are kept on uninstall.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist"
#endif
#ifndef RedistDir
  #define RedistDir "redist"
#endif

[Setup]
AppId={{B7E5C2A4-5E2F-4C55-9D1E-6A0F3C7B2D91}
AppName=VoxaVersa
AppVersion={#AppVersion}
AppVerName=VoxaVersa {#AppVersion}
AppPublisher=VoxaVersa
DefaultDirName={autopf}\VoxaVersa
DefaultGroupName=VoxaVersa
DisableProgramGroupPage=yes
OutputBaseFilename=VoxaVersa-{#AppVersion}-Windows-Setup
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\VoxaVersa.exe
UninstallDisplayName=VoxaVersa
#ifdef IconFile
SetupIconFile={#IconFile}
#endif

[Types]
Name: "full"; Description: "Everything (recommended)"
Name: "custom"; Description: "Choose components"; Flags: iscustom

[Components]
Name: "vst3"; Description: "VST3 plug-ins: VoxaVersa (effect) and VoxaVersa Singer (instrument)"; Types: full custom
Name: "app"; Description: "VoxaVersa standalone app"; Types: full custom

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut for the standalone app"; Components: app; Flags: unchecked

[Files]
Source: "{#SourceDir}\VST3\VoxaVersa.vst3\*"; DestDir: "{commoncf64}\VST3\VoxaVersa.vst3"; Components: vst3; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\VST3\VoxaVersa Singer.vst3\*"; DestDir: "{commoncf64}\VST3\VoxaVersa Singer.vst3"; Components: vst3; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\Standalone\*"; DestDir: "{app}"; Components: app; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#RedistDir}\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[InstallDelete]
; a previous version's files inside the plug-in bundles (the bundles are replaced as a whole)
Type: filesandordirs; Name: "{commoncf64}\VST3\VoxaVersa.vst3"; Components: vst3
Type: filesandordirs; Name: "{commoncf64}\VST3\VoxaVersa Singer.vst3"; Components: vst3

[Icons]
Name: "{group}\VoxaVersa"; Filename: "{app}\VoxaVersa.exe"; Components: app
Name: "{group}\Uninstall VoxaVersa"; Filename: "{uninstallexe}"
Name: "{autodesktop}\VoxaVersa"; Filename: "{app}\VoxaVersa.exe"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing the Microsoft Visual C++ runtime..."; Flags: waituntilterminated; Check: NeedsVCRedist
Filename: "{app}\VoxaVersa.exe"; Description: "Open VoxaVersa now"; Components: app; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{commoncf64}\VST3\VoxaVersa.vst3"
Type: filesandordirs; Name: "{commoncf64}\VST3\VoxaVersa Singer.vst3"

[Messages]
FinishedLabel=VoxaVersa is installed.%n%nIn your DAW, rescan plug-ins to find VoxaVersa (effect) and VoxaVersa Singer (instrument).%n%nTo add your trained AI voices: open VoxaVersa, click the menu (top right) > Import Voices..., and pick the zip you downloaded from Google Drive.

[Code]
// ONNX Runtime needs the Visual C++ 2015-2022 runtime 14.40 or newer.
function NeedsVCRedist: Boolean;
var
  Installed, Major, Minor: Cardinal;
begin
  Result := True;
  if RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1) and
     RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Major', Major) and
     RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Minor', Minor) then
    Result := (Major < 14) or ((Major = 14) and (Minor < 40));
end;
