# VoxaVersa user guide

## 1. Two ways to use it

| | **VoxaVersa** (effect + standalone) | **VoxaVersa Singer** (instrument) |
|---|---|---|
| Put it on | a vocal audio track (insert) | an instrument / MIDI track |
| Voice Transform | ✅ your track is the input | ✅ via sidechain input |
| Text to Sing | ✅ generate takes; MIDI routed to the effect sings live | ✅ MIDI notes on the track sing your lyrics |

The **standalone app** is the effect version with its own audio/MIDI settings (Options → Audio/MIDI
Settings): plug in a mic, pick a voice, sing.

## 2. Voice Transform

1. Choose a voice in the **Voice Library** (filter by Male / Female / Styles / Genres, ★ favourites, search).
2. Sing or play back your vocal. The **Input** panel shows your vocal, the **Output** panel the transformed result.
3. Shape it with **Voice Character**:
   - **Pitch**: higher/deeper in semitones (±24). **Fine** for cents.
   - **Gender**: moves the formants (vocal-tract size) from male to female.
   - **Age**: young (smaller, brighter, breathier) to mature (darker, wider vibrato, a little grit).
   - **Tone**: bright to dark.
   - **Style**: natural to creative (scales vibrato, scoops, grit and saturation).
4. **Advanced** toggles:
   - **Pitch Preserve** *on* keeps your melody exactly. *Off* moves your performance into the
     selected voice's natural register (e.g. a male vocal is lifted an octave into a female voice's range).
     With an AI voice the octave is the one that keeps the song inside its singer's range, including the
     song's high notes (above the singer's top a voice turns thin and "chipmunk"), from the first phrase on.
     *Deep Bass* sings an octave down when the song allows it; *Smoky Alto* and *Female Trap* sit low.
   - **Formant Shift** applies the voice's vocal-tract character relative to your own voice (VoxaVersa
     detects whether your input is a male or female voice. Override with *Source Voice* in settings).
   - **Expression** adds the genre's scoops, vibrato and micro-pitch life.
   - **Vibe** adds the genre's production chain (compression, de-essing, saturation, space, doubling).
   - **Intensity**: how strongly the voice's character is applied.
5. **Mix** blends the original and transformed vocal. **Output** sets the level.

### Voice sample instead of live input
Drag an audio file onto the **Input** panel. Press ▶ to audition it through the current settings, then
**Render** to create a WAV and drag it straight into your DAW (or find it in `Documents/VoxaVersa/Exports`).

### Pro tools
- **Tune**: pitch correction to a key/scale (or **Auto** key detection; *Chromatic* snaps to the nearest
  semitone in any key), retune speed from hard "robot" (0 ms) to natural, humanize, and **MIDI Pitch Control**
  (the notes you play set the pitch).
- **Harmony**: up to three scale-aware harmony voices (3rds, 4ths, 5ths, 6ths, octaves) with level & pan.
- **Doubler**: natural double-tracking (detune and timing drift), stereo width. With an AI voice the
  doubles are copies of the AI singer itself.
- **FX**: low cut, de-esser, compressor, presence, air, saturation.
- **Space**: tempo-synced ping-pong delay (ducked under the vocal) and reverb.
- **Voice Blend**: right-click a second voice → *Set as Blend Voice* to morph between two voices.
- **AI Singer**: right-click a built-in voice → *AI Singer* to choose which trained singer sings it (*Automatic*
  is the one the voice assignment picked). Your choice is kept in `Voices/singer_choices.json`, so importing a new
  assignment never undoes it. *Hard Tune Robot* and *Choir Ensemble* are sung by a singer of your own voice's
  gender (the pop singers and the gospel singers), following *Source Voice* or the detected gender.
  *Hard Tune Robot* snaps every note at once by itself (the hard-tune sound, with Tune off too; the AI singer
  gets the exact note). *Choir Ensemble* is a section: the lead singer, a gospel singer of the other gender an
  octave away (a second AI model, so about twice the AI engine's work), and six copies of the two, each a little
  late, slowly detuned and coloured differently, spread across the stereo field. The Doubler's amount and width
  scale the section.
- **Singer cards**: a trained singer that sings no built-in voice is a voice of its own, with a style title and
  portrait (e.g. *Lyric Tenor*, *Velvet Mezzo*: the classically trained VocalSet singers).
- **Presets**: factory and user presets, A/B comparison, undo/redo. A factory preset keeps your session's
  mode, input/output level, key and scale, MIDI Pitch Control and Text to Sing tempo. Presets built on a male
  or female voice turn **Pitch Preserve** off so any singer lands in that voice's register; *Chipmunk*,
  *Whisper ASMR* and *Live Stage (Low Latency)* use the DSP engine (AI off).
- **Quality**: *Low Latency* (~35 ms) for live monitoring, *Balanced*, *High Quality* for mixing
  (latency is reported to the DAW and compensated automatically). An AI voice adds ~0.85 s, reported as soon
  as it is chosen, so playback and exports line up. Switching between an AI voice and a DSP-only voice changes
  the latency: some hosts (Cubase) only pick that up after you stop and restart playback, or switch the
  plug-in off and on in its slot. When the DAW stops, VoxaVersa falls silent with the other tracks instead of
  singing out the last ~0.9 s it was still processing. While you monitor live through an AI voice you hear
  yourself ~0.9 s late; record with the AI engine off (or *Low Latency*) and switch it on for playback.
- **Power** is the DAW's bypass: the power button, the bypass button of the plug-in window and the channel
  insert's bypass are the same switch. Bypassed, VoxaVersa passes your voice through with its latency kept, so
  it stays in time.

## 3. Text to Sing

1. Switch to **Text to Sing** and type your lyrics.
   - New line = new phrase (a breath is added).
   - `beau-ti-ful`: explicit syllable splits. `_` holds the previous syllable over the next note (melisma).
   - `[br]` inserts a breath/rest.
2. Choose the **Melody Source**:
   - **Auto Melody**: VoxaVersa composes a melody in your **Key**/scale and **Tempo**, shaped by the
     selected voice's genre (rap = dense rhythm, ballads = long notes).
   - **MIDI File**: load a `.mid` melody.
   - **Use MIDI**: play notes into VoxaVersa. Each note sings the next syllable *live*. Enable
     **Capture** to record what you play, then Generate a polished take from it.
   - **Hum Melody**: hum or sing the tune into your mic, VoxaVersa transcribes it.
3. Press **Generate**. Preview it, **Render & Drag** the WAV into your DAW, or **Export MIDI** (with
   lyrics as MIDI lyric events). In a DAW the take also plays in sync from the start of the song (bar 1).

The voice's register, formants, breathiness, vibrato and phrasing come from the selected voice, and all
Voice Character / FX controls still apply. When an AI singer sings the take, the melody is written in the range
that singer actually sang (from the voice assignment), where the model sounds natural.

## 4. Routing MIDI to the effect version

| DAW | How |
|---|---|
| Ableton Live | Create a MIDI track, set *MIDI To* → the vocal track → *VoxaVersa* |
| Logic Pro | Load VoxaVersa as a MIDI-controlled effect (*AU MIDI-controlled Effects*) and pick its side-chain MIDI |
| FL Studio | Use **VoxaVersa Singer** on a channel, or route MIDI Out → VoxaVersa's MIDI input port |
| Cubase / Nuendo | MIDI track output → *VoxaVersa* (it appears as a MIDI destination) |
| Reaper | Send MIDI from a track to the track hosting VoxaVersa (route window → MIDI send) |
| Studio One | Add VoxaVersa Singer as an instrument, or set the MIDI track's output to VoxaVersa |
| Pro Tools | Use VoxaVersa Singer as an instrument plug-in (AAX build) |

When in doubt, use **VoxaVersa Singer** on an instrument track. MIDI just works there.

## 5. Getting the most realistic results

VoxaVersa has two voice engines:

1. **DSP engine (built in, always available)**: pitch, formants, tone, expression and production are
   modelled with real-time signal processing. It responds instantly and is great for pitch/gender changes,
   tuning, harmonies and character. It reshapes *your* voice: listeners will hear a convincingly
   different register and timbre, but the result is still recognisably processed when pushed to extremes.
2. **AI engine (neural voice packs)**: a trained voice model re-synthesises your performance in a
   different singer's timbre. This is the route to vocals that are indistinguishable from a real singer.
   VoxaVersa runs RVC-compatible ONNX models. See `docs/NEURAL_ENGINE.md` to install the ONNX
   Runtime and to build voice packs from recordings you own or have permission to use.

Tips:
- Record dry, close and clean. Reverb or background noise on the input confuses pitch tracking.
- Sing in a comfortable range and let **Pitch Preserve off** move you into the target register.
- Keep **Intensity** around 60–80% for natural results. Push it for character effects.
- For gender changes, combine the register shift with **Gender** ±30–60 rather than maxing one control.
- Use *High Quality* when mixing and *Low Latency* only when you need to monitor live.
- **AI voice on a slower computer.** The AI engine converts in 0.4 s chunks. If your computer can't keep
  up during playback, VoxaVersa plays the DSP voice for the moments it missed (you'll see a message), so
  playback never drops out. The final result is always complete: **Export Audio Mixdown**, **Render in
  Place** or **Freeze** wait for every chunk and use the AI engine's best quality. The status line (menu,
  top right) shows the engine (CPU or GPU) and its *real-time load*; below 80% plays live without gaps.

## 6. Files & folders

| What | Where |
|---|---|
| User presets | `<app data>/VoxaVersa/Presets/*.vxpreset` |
| Voice packs | `<app data>/VoxaVersa/Voices/<pack>/voice.json` |
| ONNX Runtime (optional) | next to the plug-in, or `<app data>/VoxaVersa/Runtime/` |
| Renders & MIDI exports | `Documents/VoxaVersa/Exports` |
| Recordings (standalone) | `Documents/VoxaVersa/Recordings` |

`<app data>` is `%APPDATA%` on Windows, `~/Library/Application Support` on macOS, `~/.config` on Linux.

## 7. Disclosure of AI and transformed vocals

Every WAV that VoxaVersa renders or records carries a short machine-readable statement in its metadata
(RIFF INFO comment and software fields, plus the Broadcast WAV description), for example
*"AI-generated vocal: re-sung by the VoxaVersa AI voice model ..."* or *"Vocal digitally transformed with
VoxaVersa"*. This supports the EU AI Act's requirement (Article 50) that synthetic audio is marked as
such. Please keep these notes intact and disclose AI vocals where your distributor or platform requires it.

In addition, AI-voice and Text to Sing output carries an **inaudible watermark**, both live (so DAW bounces
are marked too) and in exported files. It sits about 30 dB under the vocal between 1 and 8 kHz, follows the
vocal's level, disappears in silence and adds no latency. It survives MP3/AAC encoding, resampling, level
changes and cropping, but not pitch shifting or time stretching. A plain DSP transform of your own voice is
not marked. To check a file: `VoxaVersaRender <file.wav> --detect` (built with `-DVOXAVERSA_BUILD_TOOLS=ON`).
A few seconds of solo vocal are enough; in a busy full mix it needs longer.
