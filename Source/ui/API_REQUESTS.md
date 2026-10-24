# Processor API requests from the UI

The editor is written strictly against `Source/PluginProcessor.h`. These are the places where the
UI had to work around a missing API; each item lists the workaround currently in place.

1. **Editor size persistence** – there is no documented way to store the editor size.
   *Workaround:* the editor writes a non-parameter property `uiWidth` on `getState().state`
   (saved with the session) and the default for new instances into
   `VoiceLibrary::getUserDataDirectory()/settings.xml` as `<Ui width="..."/>`.
   *Request:* reserve `StateID::uiWidth` (and keep it out of presets), or add
   `get/setEditorSize()`.

2. **Advanced tab persistence** – the selected Advanced tab is stored as the non-parameter state
   property `uiAdvancedTab`. *Request:* reserve the ID / exclude UI-only properties from presets.

3. **Syllables for live MIDI singing** – only `getLiveSyllableIndex()` exists, so the UI splits the
   lyrics itself (words, split at hyphens) to show and highlight the next syllable.
   *Request:* `juce::Array<juce::Range<int>> getSyllableRanges()` (character ranges in the lyrics as
   parsed by `SingEngine`), so the highlight always matches the engine.

4. **Explicit input source** – there is no "live input vs. voice sample" switch; the UI treats
   `hasSample()` as "sample source" and uses `clearSample()` for "Live input".
   *Request:* `setInputSource (Live | Sample)` / `getInputSource()` so a sample can stay loaded
   while monitoring the live input.

5. **Render completion / target** – one offline-render slot is shared by samples and sung takes and
   there is no completion callback, so the UI remembers which target it started and polls
   `isOfflineRendering()`; the file is accepted if `getLastRenderedFile()` changed.
   *Request:* `getLastRenderTarget()` plus a monotonically increasing render id (or a
   `ChangeBroadcaster`) so stale files are never offered for drag-out.

6. **Take identity** – nothing identifies a sung take, so the UI invalidates its rendered WAV and
   exported MIDI whenever `getSingState()` enters `Rendering`.
   *Request:* `getSingTakeId()`.

7. **Preset modified flag** – the preset name is shown without a "modified" marker because the
   `PresetManager` has no dirty state. *Request:* `isCurrentPresetModified()`.

8. **Error / status notifications** – errors are discovered by polling (`getSingError()`,
   `getLastRenderError()`, `getNeuralStatusMessage()`). *Request:* a message-thread
   `ChangeBroadcaster` (or listener) for status changes so the toast area can react without
   polling.

9. **Snapshot / demo hook** – for screenshots the editor pushes a synthetic signal into the public
   `inputScope` / `outputScope` / meters and pitch atomics, only when the opt-in environment
   variables `VOXAVERSA_SNAPSHOT` / `VOXAVERSA_DEMO` are set. *Request (optional):* a
   `setDemoMode (bool)` on the processor so this does not race with the audio thread.
