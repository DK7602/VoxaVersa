#!/usr/bin/env bash
# Builds VoxaVersa-<version>-macOS.pkg from the CI artefact layout (dist/VST3, dist/AU, dist/Standalone):
#   VST3 -> /Library/Audio/Plug-Ins/VST3, AU -> /Library/Audio/Plug-Ins/Components, app -> /Applications
#
#   installer/macos/build_pkg.sh <dist> <version> <out.pkg>
#
# The package and the bundles are not notarised (that needs an Apple Developer ID): the first time, open the
# .pkg with right-click > Open. Voices, presets and settings live in ~/Library/Application Support/VoxaVersa.
set -euo pipefail

dist="$1"
version="$2"
out="$3"
work="$(mktemp -d)"
root="$work/root"

mkdir -p "$root/Library/Audio/Plug-Ins/VST3" "$root/Library/Audio/Plug-Ins/Components" "$root/Applications"
[ -d "$dist/VST3" ] && cp -R "$dist/VST3/"*.vst3 "$root/Library/Audio/Plug-Ins/VST3/"
[ -d "$dist/AU" ] && cp -R "$dist/AU/"*.component "$root/Library/Audio/Plug-Ins/Components/"
[ -d "$dist/Standalone" ] && cp -R "$dist/Standalone/"*.app "$root/Applications/"

# Bundles must stay where we put them (pkgbuild would otherwise "relocate" them to an older copy found elsewhere).
pkgbuild --analyze --root "$root" "$work/components.plist"
count=$(/usr/libexec/PlistBuddy -c "Print" "$work/components.plist" | grep -c "RootRelativeBundlePath" || true)
for ((i = 0; i < count; i++)); do
    /usr/libexec/PlistBuddy -c "Set :$i:BundleIsRelocatable false" "$work/components.plist"
done

pkgbuild --root "$root" --component-plist "$work/components.plist" --identifier com.voxaversa.pkg \
         --version "$version" --install-location / "$out"
rm -rf "$work"
echo "built $out"
