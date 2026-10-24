#include "AdvancedPanel.h"
#include "VoxaLookAndFeel.h"
#include "core/MusicTheory.h"

namespace voxa::ui
{

namespace
{
    using APVTS = juce::AudioProcessorValueTreeState;

    //==============================================================================
    struct ParamToggle : public GlassButton
    {
        ParamToggle (EditorContext& ctx, const char* paramId, const juce::String& buttonText, std::optional<Icon> buttonIcon, const juce::String& tip)
            : GlassButton (buttonText, buttonIcon)
        {
            setClickingTogglesState (true);
            setTooltip (tip);
            setFontHeight (15.5f);
            attachment = std::make_unique<APVTS::ButtonAttachment> (ctx.state, paramId, *this);
        }

        std::unique_ptr<APVTS::ButtonAttachment> attachment;
    };

    struct ParamCombo : public juce::ComboBox
    {
        ParamCombo (EditorContext& ctx, const char* paramId, const juce::StringArray& items, const juce::String& tip)
        {
            addItemList (items, 1);
            setTooltip (tip);
            if (auto* p = ctx.state.getParameter (paramId))
                setTitle (p->getName (64));
            attachment = std::make_unique<APVTS::ComboBoxAttachment> (ctx.state, paramId, *this);
        }

        std::unique_ptr<APVTS::ComboBoxAttachment> attachment;
    };

    struct ParamSlider : public juce::Slider
    {
        ParamSlider (EditorContext& ctx, const char* paramId, bool bipolar, const juce::String& tip)
            : juce::Slider (juce::Slider::LinearHorizontal, juce::Slider::NoTextBox)
        {
            getProperties().set (SliderProps::knobStyle, "small");
            getProperties().set (SliderProps::bipolar, bipolar);
            setTooltip (tip);
            attachment = std::make_unique<APVTS::SliderAttachment> (ctx.state, paramId, *this);
            if (auto* p = ctx.state.getParameter (paramId))
            {
                setTitle (p->getName (64));
                setDoubleClickReturnValue (true, p->convertFrom0to1 (p->getDefaultValue()));
            }
        }

        std::unique_ptr<APVTS::SliderAttachment> attachment;
    };

    Knob::Options smallKnob (const juce::String& name, const juce::String& tip, juce::Colour accent = Palette::glow)
    {
        Knob::Options o;
        o.size = Knob::Size::Small;
        o.name = name;
        o.valueBelow = true;
        o.tooltip = tip;
        o.accent = accent;
        return o;
    }

    void sectionLabel (juce::Graphics& g, const juce::String& text, juce::Rectangle<float> r)
    {
        drawText (g, text, r, font (12.5f, Weight::SemiBold), Palette::glowBright.withAlpha (0.8f));
    }

    //==============================================================================
    class AdvancedPage : public AdvancedPanel::Page
    {
    public:
        explicit AdvancedPage (EditorContext& c)
        {
            struct T { FeatureTile* tile; const char* id; const char* tip; };
            for (auto t : { T { &pitchPreserve, ParamID::pitchPreserve, "Pitch Preserve: keeps your melody while the timbre changes" },
                            T { &formantShift, ParamID::formantShift, "Formant Shift: reshapes the vocal tract to match the target voice" },
                            T { &expression, ParamID::expression, "Expression: adds the voice's vibrato, scoops and phrasing" },
                            T { &vibe, ParamID::vibe, "Vibe: applies the genre's production chain (compression, space, saturation)" } })
            {
                addAndMakeVisible (t.tile);
                t.tile->setTooltip (t.tip);
                attachments.add (new APVTS::ButtonAttachment (c.state, t.id, *t.tile));
            }

            intensity.setSliderStyle (juce::Slider::LinearHorizontal);
            intensity.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
            intensity.setTitle ("Intensity");
            intensity.setTooltip ("Intensity: overall strength of the transformation");
            addAndMakeVisible (intensity);
            intensityAttachment = std::make_unique<APVTS::SliderAttachment> (c.state, ParamID::intensity, intensity);
            intensity.setDoubleClickReturnValue (true, 70.0);
            intensity.onValueChange = [this] { repaint (valueArea.toNearestInt().expanded (2)); };
        }

        void resized() override
        {
            const float w = (float) getWidth() / 4.0f;
            int i = 0;
            for (auto* t : { &pitchPreserve, &formantShift, &expression, &vibe })
            {
                t->setBounds (juce::Rectangle<float> ((float) i * w, 0.0f, w, 94.0f).toNearestInt());
                ++i;
            }

            pill = { 4.0f, 104.0f, (float) getWidth() - 8.0f, 44.0f };
            intensity.setBounds (juce::Rectangle<float> (pill.getX() + 86.0f, pill.getY() + 6.0f, pill.getWidth() - 160.0f, 32.0f).toNearestInt());
            valueArea = { pill.getRight() - 66.0f, pill.getY() + 6.0f, 58.0f, 32.0f };
        }

        void paint (juce::Graphics& g) override
        {
            g.setColour (juce::Colour (0x90040a1e));
            g.fillRoundedRectangle (pill, 10.0f);
            g.setColour (Palette::panelBorder.withAlpha (0.2f));
            g.drawRoundedRectangle (pill.reduced (0.5f), 10.0f, 1.0f);
            drawText (g, "Intensity", pill.withWidth (84.0f).withTrimmedLeft (12.0f), font (17.0f, Weight::Medium), Palette::text,
                      juce::Justification::centredLeft);
            fillInset (g, valueArea, 7.0f, 0.8f);
            drawText (g, juce::String (juce::roundToInt (intensity.getValue())) + "%", valueArea, font (17.0f, Weight::Medium), Palette::text);
        }

    private:
        FeatureTile pitchPreserve { "Pitch Preserve", Icon::PitchPreserve };
        FeatureTile formantShift { "Formant Shift", Icon::FormantShift };
        FeatureTile expression { "Expression", Icon::Expression };
        FeatureTile vibe { "Vibe", Icon::Vibe };
        juce::OwnedArray<APVTS::ButtonAttachment> attachments;
        juce::Slider intensity;
        std::unique_ptr<APVTS::SliderAttachment> intensityAttachment;
        juce::Rectangle<float> pill, valueArea;
    };

    //==============================================================================
    class ExpressionPage : public AdvancedPanel::Page
    {
    public:
        explicit ExpressionPage (EditorContext& c)
            : vibratoDepth (c.state, ParamID::vibratoDepth, smallKnob ("Vibrato", "Vibrato depth in cents (adds to the voice's own vibrato)")),
              vibratoRate (c.state, ParamID::vibratoRate, smallKnob ("Rate", "Vibrato rate")),
              breath (c.state, ParamID::breath, smallKnob ("Breath", "Breathiness / air in the voice")),
              rasp (c.state, ParamID::rasp, smallKnob ("Rasp", "Grit and rasp"))
        {
            for (auto* k : { &vibratoDepth, &vibratoRate, &breath, &rasp })
                addAndMakeVisible (k);
        }

        void resized() override
        {
            const float w = (float) getWidth() / 4.0f;
            int i = 0;
            for (auto* k : { &vibratoDepth, &vibratoRate, &breath, &rasp })
                k->setBounds (juce::Rectangle<float> ((float) i++ * w + 6.0f, 20.0f, w - 12.0f, 100.0f).toNearestInt());
        }

        void paint (juce::Graphics& g) override
        {
            sectionLabel (g, "VIBRATO", { 0.0f, 0.0f, (float) getWidth() * 0.5f, 16.0f });
            sectionLabel (g, "TEXTURE", { (float) getWidth() * 0.5f, 0.0f, (float) getWidth() * 0.5f, 16.0f });
            drawText (g, "Adds to the selected voice's natural expression", { 0.0f, 128.0f, (float) getWidth(), 18.0f },
                      font (13.5f), Palette::textDim);
        }

    private:
        Knob vibratoDepth, vibratoRate, breath, rasp;
    };

    //==============================================================================
    class LiveTuner : public juce::Component
    {
    public:
        explicit LiveTuner (EditorContext& c) : ctx (c)
        {
            setInterceptsMouseClicks (false, false);
            setTitle ("Live tuner");
        }

        void tick()
        {
            const float hz = ctx.processor.detectedPitchHz.load();
            const float target = ctx.processor.targetPitchHz.load();
            juce::String newNote = "--";
            int newCents = 0;
            bool voiced = hz > 1.0f;

            if (voiced)
            {
                const float midi = music::hzToMidi (hz);
                const int nearest = juce::roundToInt (midi);
                newNote = music::noteName (nearest);
                newCents = juce::roundToInt ((midi - (float) nearest) * 100.0f);
            }

            const auto newTarget = target > 1.0f ? noteNameForHz (target) : juce::String ("--");
            if (newNote != note || std::abs (newCents - cents) >= 1 || voiced != isVoiced || newTarget != targetNote)
            {
                note = newNote;
                cents = newCents;
                isVoiced = voiced;
                targetNote = newTarget;
                repaint();
            }
        }

        void paint (juce::Graphics& g) override
        {
            auto r = getLocalBounds().toFloat();
            fillInset (g, r, 8.0f);
            auto inner = r.reduced (8.0f, 6.0f);

            auto top = inner.removeFromTop (28.0f);
            const bool inTune = isVoiced && std::abs (cents) <= 8;
            drawText (g, note, top.removeFromLeft (top.getWidth() * 0.5f), font (24.0f, Weight::SemiBold),
                      isVoiced ? (inTune ? Palette::success : juce::Colours::white) : Palette::textFaint,
                      juce::Justification::centredLeft);
            drawText (g, (cents > 0 ? "+" : "") + juce::String (isVoiced ? cents : 0) + " ct", top, font (14.0f), Palette::textSoft,
                      juce::Justification::centredRight);

            // cents scale
            auto scale = inner.removeFromTop (16.0f).reduced (2.0f, 4.0f);
            g.setColour (Palette::insetBorder);
            g.fillRoundedRectangle (scale, 3.0f);
            g.setColour (Palette::success.withAlpha (0.35f));
            g.fillRect (scale.withSizeKeepingCentre (scale.getWidth() * 0.16f, scale.getHeight()));
            for (int i = 0; i <= 10; ++i)
            {
                const float x = scale.getX() + scale.getWidth() * (float) i / 10.0f;
                g.setColour (Palette::textFaint);
                g.fillRect (juce::Rectangle<float> (x - 0.5f, scale.getY() - (i % 5 == 0 ? 3.0f : 1.0f), 1.0f, scale.getHeight() + (i % 5 == 0 ? 6.0f : 2.0f)));
            }
            if (isVoiced)
            {
                const float x = scale.getCentreX() + (float) juce::jlimit (-50, 50, cents) / 50.0f * scale.getWidth() * 0.5f;
                juce::Path needle;
                needle.startNewSubPath (x, scale.getY() - 4.0f);
                needle.lineTo (x, scale.getBottom() + 4.0f);
                strokeGlow (g, needle, inTune ? Palette::success : Palette::glowBright, 2.0f);
            }

            drawText (g, "Target  " + targetNote, inner, font (13.5f), Palette::textSoft, juce::Justification::centredLeft);
        }

    private:
        EditorContext& ctx;
        juce::String note { "--" }, targetNote { "--" };
        int cents = 0;
        bool isVoiced = false;
    };

    class TunePage : public AdvancedPanel::Page
    {
    public:
        explicit TunePage (EditorContext& c)
            : tuneOn (c, ParamID::tuneOn, "Correction", Icon::TuningFork, "Pitch correction on/off"),
              key (c, ParamID::key, Choices::key(), "Key for pitch correction and harmonies (Auto = detected)"),
              scale (c, ParamID::scale, Choices::scale(), "Scale for pitch correction and harmonies"),
              midi (c, ParamID::midiControl, "MIDI", Icon::Midi, "MIDI Pitch Control: incoming MIDI notes set the target pitch"),
              retune (c.state, ParamID::retuneSpeed, smallKnob ("Speed", "Retune speed: 0 ms = hard tune, slower = natural")),
              humanize (c.state, ParamID::humanize, smallKnob ("Humanize", "Keeps natural variation on sustained notes")),
              tuner (c)
        {
            for (auto* comp : std::initializer_list<juce::Component*> { &tuneOn, &key, &scale, &midi, &retune, &humanize, &tuner })
                addAndMakeVisible (comp);
        }

        void resized() override
        {
            const int w = getWidth();
            tuneOn.setBounds (0, 0, 118, 34);
            key.setBounds (124, 0, 70, 34);
            scale.setBounds (200, 0, w - 200, 34);
            retune.setBounds (0, 44, 62, 98);
            humanize.setBounds (66, 44, 62, 98);
            tuner.setBounds (136, 42, w - 136, 64);
            midi.setBounds (w - 92, 112, 92, 34);
        }

        void paint (juce::Graphics& g) override
        {
            drawText (g, "Notes from MIDI", { 136.0f, 112.0f, (float) getWidth() - 236.0f, 34.0f }, font (13.5f), Palette::textDim,
                      juce::Justification::centredLeft);
        }

        void tick (float) override { tuner.tick(); }

    private:
        ParamToggle tuneOn;
        ParamCombo key, scale;
        ParamToggle midi;
        Knob retune, humanize;
        LiveTuner tuner;
    };

    //==============================================================================
    class HarmonyPage : public AdvancedPanel::Page
    {
    public:
        explicit HarmonyPage (EditorContext& c)
            : harmonyOn (c, ParamID::harmonyOn, "Harmony", std::nullopt, "Adds up to three scale-aware harmony voices")
        {
            addAndMakeVisible (harmonyOn);

            const char* intervals[] = { ParamID::harm1Interval, ParamID::harm2Interval, ParamID::harm3Interval };
            const char* levels[] = { ParamID::harm1Level, ParamID::harm2Level, ParamID::harm3Level };
            const char* pans[] = { ParamID::harm1Pan, ParamID::harm2Pan, ParamID::harm3Pan };

            for (int i = 0; i < 3; ++i)
            {
                auto* combo = interval.add (new ParamCombo (c, intervals[i], Choices::harmonyInterval(), "Harmony voice " + juce::String (i + 1) + " interval (in the current key)"));
                auto* lvl = level.add (new ParamSlider (c, levels[i], false, "Harmony voice " + juce::String (i + 1) + " level"));
                auto* pn = pan.add (new ParamSlider (c, pans[i], true, "Harmony voice " + juce::String (i + 1) + " pan"));
                for (auto* comp : std::initializer_list<juce::Component*> { combo, lvl, pn })
                    addAndMakeVisible (comp);
            }
        }

        void resized() override
        {
            const int w = getWidth();
            harmonyOn.setBounds (0, 0, 104, 30);
            for (int i = 0; i < 3; ++i)
            {
                const int y = 40 + i * 36;
                interval[i]->setBounds (26, y, 112, 30);
                level[i]->setBounds (146, y, w - 146 - 92, 30);
                pan[i]->setBounds (w - 84, y, 84, 30);
            }
        }

        void paint (juce::Graphics& g) override
        {
            const auto f = font (12.5f, Weight::SemiBold);
            const float w = (float) getWidth();
            drawText (g, "LEVEL", { 146.0f, 8.0f, w - 238.0f, 24.0f }, f, Palette::textDim);
            drawText (g, "PAN", { w - 84.0f, 8.0f, 84.0f, 24.0f }, f, Palette::textDim);
            for (int i = 0; i < 3; ++i)
                drawText (g, juce::String (i + 1), { 0.0f, 40.0f + (float) i * 36.0f, 20.0f, 30.0f }, font (16.0f, Weight::SemiBold), Palette::glowBright);
        }

    private:
        ParamToggle harmonyOn;
        juce::OwnedArray<ParamCombo> interval;
        juce::OwnedArray<ParamSlider> level, pan;
    };

    //==============================================================================
    class FxPage : public AdvancedPanel::Page
    {
    public:
        explicit FxPage (EditorContext& c)
        {
            struct K { const char* id; const char* name; const char* tip; };
            for (auto k : { K { ParamID::lowCut, "Low Cut", "Low cut filter (removes rumble)" },
                            K { ParamID::deEsser, "De-Esser", "Tames harsh 's' sounds" },
                            K { ParamID::compressor, "Comp", "Vocal compressor amount" },
                            K { ParamID::presence, "Presence", "Presence boost / cut (3-5 kHz)" },
                            K { ParamID::air, "Air", "High shelf 'air' (12 kHz)" },
                            K { ParamID::saturation, "Saturate", "Warm analogue saturation" } })
                addAndMakeVisible (knobs.add (new Knob (c.state, k.id, smallKnob (k.name, k.tip))));
        }

        void resized() override
        {
            const float w = (float) getWidth() / 3.0f;
            for (int i = 0; i < knobs.size(); ++i)
                knobs[i]->setBounds (juce::Rectangle<float> ((float) (i % 3) * w + 8.0f, (float) (i / 3) * 76.0f, w - 16.0f, 74.0f).toNearestInt());
        }

    private:
        juce::OwnedArray<Knob> knobs;
    };

    //==============================================================================
    class SpacePage : public AdvancedPanel::Page
    {
    public:
        explicit SpacePage (EditorContext& c)
            : delayMix (c.state, ParamID::delayMix, smallKnob ("Mix", "Delay mix")),
              delayFeedback (c.state, ParamID::delayFeedback, smallKnob ("Feedback", "Delay feedback")),
              delayTime (c, ParamID::delayTime, Choices::delayTime(), "Delay time (synced to the host tempo)"),
              reverbMix (c.state, ParamID::reverbMix, smallKnob ("Mix", "Reverb mix")),
              reverbSize (c.state, ParamID::reverbSize, smallKnob ("Size", "Reverb size")),
              doublerOn (c, ParamID::doublerOn, "On", std::nullopt, "Doubler: stereo double-tracking"),
              doublerAmount (c.state, ParamID::doublerAmount, smallKnob ("Amount", "Doubler amount")),
              doublerWidth (c.state, ParamID::doublerWidth, smallKnob ("Width", "Doubler stereo width"))
        {
            for (auto* comp : std::initializer_list<juce::Component*> { &delayMix, &delayFeedback, &delayTime, &reverbMix, &reverbSize,
                                                                         &doublerOn, &doublerAmount, &doublerWidth })
                addAndMakeVisible (comp);
        }

        void resized() override
        {
            const float colW = (float) getWidth() / 3.0f;
            auto col = [colW] (int c) { return juce::Rectangle<float> ((float) c * colW, 0.0f, colW, 150.0f); };

            auto d = col (0).reduced (4.0f, 0.0f);
            delayMix.setBounds (juce::Rectangle<float> (d.getX(), 20.0f, d.getWidth() * 0.5f, 84.0f).toNearestInt());
            delayFeedback.setBounds (juce::Rectangle<float> (d.getCentreX(), 20.0f, d.getWidth() * 0.5f, 84.0f).toNearestInt());
            delayTime.setBounds (juce::Rectangle<float> (d.getX() + 4.0f, 112.0f, d.getWidth() - 8.0f, 30.0f).toNearestInt());

            auto r = col (1).reduced (4.0f, 0.0f);
            reverbMix.setBounds (juce::Rectangle<float> (r.getX(), 20.0f, r.getWidth() * 0.5f, 84.0f).toNearestInt());
            reverbSize.setBounds (juce::Rectangle<float> (r.getCentreX(), 20.0f, r.getWidth() * 0.5f, 84.0f).toNearestInt());

            auto dbl = col (2).reduced (4.0f, 0.0f);
            doublerAmount.setBounds (juce::Rectangle<float> (dbl.getX(), 20.0f, dbl.getWidth() * 0.5f, 84.0f).toNearestInt());
            doublerWidth.setBounds (juce::Rectangle<float> (dbl.getCentreX(), 20.0f, dbl.getWidth() * 0.5f, 84.0f).toNearestInt());
            doublerOn.setBounds (juce::Rectangle<float> (dbl.getX() + 10.0f, 112.0f, dbl.getWidth() - 20.0f, 30.0f).toNearestInt());
        }

        void paint (juce::Graphics& g) override
        {
            const float colW = (float) getWidth() / 3.0f;
            sectionLabel (g, "DELAY", { 0.0f, 0.0f, colW, 16.0f });
            sectionLabel (g, "REVERB", { colW, 0.0f, colW, 16.0f });
            sectionLabel (g, "DOUBLER", { colW * 2.0f, 0.0f, colW, 16.0f });

            g.setColour (Palette::panelBorder.withAlpha (0.12f));
            g.fillRect (juce::Rectangle<float> (colW, 4.0f, 1.0f, 138.0f));
            g.fillRect (juce::Rectangle<float> (colW * 2.0f, 4.0f, 1.0f, 138.0f));
        }

    private:
        Knob delayMix, delayFeedback;
        ParamCombo delayTime;
        Knob reverbMix, reverbSize;
        ParamToggle doublerOn;
        Knob doublerAmount, doublerWidth;
    };
}

//==============================================================================
const juce::StringArray& AdvancedPanel::getTabNames()
{
    static const juce::StringArray names { "Advanced", "Expression", "Tune", "Harmony", "FX", "Space" };
    return names;
}

AdvancedPanel::AdvancedPanel (EditorContext& c) : GlassPanel ("Advanced"), ctx (c)
{
    pages.add (new AdvancedPage (c));
    pages.add (new ExpressionPage (c));
    pages.add (new TunePage (c));
    pages.add (new HarmonyPage (c));
    pages.add (new FxPage (c));
    pages.add (new SpacePage (c));

    const auto& names = getTabNames();
    for (int i = 0; i < names.size(); ++i)
    {
        auto* tab = tabs.add (new TabButton (names[i].toUpperCase()));
        tab->setTitle (names[i] + " tab");
        tab->setTooltip (names[i]);
        tab->onClick = [this, i] { showTab (i); };
        addAndMakeVisible (tab);

        pages[i]->setTitle (names[i]);
        pages[i]->setFocusContainerType (FocusContainerType::focusContainer);
        addChildComponent (pages[i]);
    }

    const int saved = (int) ctx.state.state.getProperty ("uiAdvancedTab", 0);
    showTab (juce::jlimit (0, pages.size() - 1, saved));
}

AdvancedPanel::~AdvancedPanel() = default;

void AdvancedPanel::showTab (int index)
{
    currentTab = juce::jlimit (0, pages.size() - 1, index);
    for (int i = 0; i < pages.size(); ++i)
    {
        pages[i]->setVisible (i == currentTab);
        tabs[i]->setToggleState (i == currentTab, juce::dontSendNotification);
    }
    ctx.state.state.setProperty ("uiAdvancedTab", currentTab, nullptr);
}

bool AdvancedPanel::showTabNamed (const juce::String& name)
{
    const int idx = getTabNames().indexOf (name, true);
    if (idx < 0)
        return false;
    showTab (idx);
    return true;
}

void AdvancedPanel::resized()
{
    // Fit the tab labels into the title row.
    const float avail = (float) getWidth() - 20.0f;
    float fontH = 17.0f;
    float total = 0.0f;
    for (; fontH > 11.0f; fontH -= 0.5f)
    {
        total = 0.0f;
        for (auto* t : tabs)
        {
            t->setFontHeight (fontH);
            total += t->getIdealWidth();
        }
        if (total <= avail)
            break;
    }

    const float extra = juce::jmax (0.0f, (avail - total) / (float) tabs.size());
    float x = 10.0f;
    for (auto* t : tabs)
    {
        const float w = t->getIdealWidth() + extra;
        t->setBounds (juce::Rectangle<float> (x, 6.0f, w, titleHeight - 8.0f).toNearestInt());
        x += w;
    }

    for (auto* p : pages)
        p->setBounds (getLocalBounds().withTrimmedTop ((int) titleHeight + 8).reduced (14, 0).withTrimmedBottom (8));
}

void AdvancedPanel::paintStatic (juce::Graphics& g, juce::Rectangle<float> area)
{
    // Tabs replace the title text; keep the separator line.
    auto r = area.removeFromTop (titleHeight);
    juce::ColourGradient cg (Palette::panelBorder.withAlpha (0.0f), r.getX() + 8.0f, 0.0f,
                             Palette::panelBorder.withAlpha (0.0f), r.getRight() - 8.0f, 0.0f, false);
    cg.addColour (0.12, Palette::panelBorder.withAlpha (0.22f));
    cg.addColour (0.88, Palette::panelBorder.withAlpha (0.22f));
    g.setGradientFill (cg);
    g.fillRect (juce::Rectangle<float> (r.getX() + 8.0f, r.getBottom() - 1.0f, r.getWidth() - 16.0f, 1.0f));
}

void AdvancedPanel::tick (float dt)
{
    if (juce::isPositiveAndBelow (currentTab, pages.size()))
        pages[currentTab]->tick (dt);
}

} // namespace voxa::ui
