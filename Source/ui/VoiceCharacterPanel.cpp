#include "VoiceCharacterPanel.h"
#include "VoxaLookAndFeel.h"

namespace voxa::ui
{

namespace
{
    std::function<juce::String (double)> bipolarFormatter (juce::String left, juce::String right)
    {
        return [left, right] (double v)
        {
            const int n = juce::roundToInt (v);
            if (n == 0)
                return juce::String ("Neutral");
            return (n < 0 ? left : right) + " " + juce::String (std::abs (n));
        };
    }
}

Knob::Options VoiceCharacterPanel::characterOptions (const juce::String& name, const juce::String& left,
                                                     const juce::String& right, const juce::String& tooltip)
{
    Knob::Options o;
    o.size = Knob::Size::Medium;
    o.name = name;
    o.leftCaption = left;
    o.rightCaption = right;
    o.valueAbove = true;
    o.formatter = bipolarFormatter (left, right);
    o.tooltip = tooltip;
    return o;
}

VoiceCharacterPanel::VoiceCharacterPanel (EditorContext& c)
    : GlassPanel ("Voice Character"),
      ctx (c),
      pitch (c.state, ParamID::pitch, [] {
          Knob::Options o;
          o.size = Knob::Size::Medium;
          o.name = "Pitch";
          o.leftCaption = "Deeper";
          o.rightCaption = "Higher";
          o.valueAbove = true;
          o.accent = Palette::cyan;
          o.formatter = [] (double v)
          {
              const int st = juce::roundToInt (v);
              if (st == 0)
                  return juce::String ("0 st");
              if (std::abs (st) % 12 == 0)
                  return juce::String (st > 0 ? "+" : "-") + juce::String (std::abs (st) / 12) + " oct";
              return (st > 0 ? "+" : "") + juce::String (st) + " st";
          };
          o.tooltip = "Pitch: make the voice deeper or higher (up to two octaves either way)";
          return o;
      }()),
      fineTune (c.state, ParamID::fineTune, [] {
          Knob::Options o;
          o.size = Knob::Size::Small;
          o.name = "Fine";
          o.valueBelow = true;
          o.accent = Palette::cyan;
          o.tooltip = "Fine tune in cents";
          return o;
      }()),
      gender (c.state, ParamID::gender, characterOptions ("Gender", "Male", "Female", "Gender: shifts the vocal character from male to female")),
      age (c.state, ParamID::age, characterOptions ("Age", "Young", "Mature", "Age: younger or more mature voice")),
      tone (c.state, ParamID::tone, characterOptions ("Tone", "Bright", "Dark", "Tone: brighter or darker timbre")),
      style (c.state, ParamID::style, characterOptions ("Style", "Natural", "Creative", "Style: natural or creative / stylised delivery"))
{
    for (auto* comp : std::initializer_list<juce::Component*> { &pitch, &fineTune, &gender, &age, &tone, &style })
        addAndMakeVisible (comp);

    addChildComponent (blendSlider);
    addChildComponent (clearBlend);

    blendSlider.setSliderStyle (juce::Slider::LinearHorizontal);
    blendSlider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    blendSlider.getProperties().set (SliderProps::knobStyle, "small");
    blendSlider.getProperties().set (SliderProps::accent, (juce::int64) juce::Colour (0xffb489ff).getARGB());
    blendSlider.setTitle ("Voice Blend");
    blendAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (ctx.state, ParamID::voiceBlend, blendSlider);
    blendSlider.setDoubleClickReturnValue (true, 50.0);

    clearBlend.setIconScale (0.5f);
    clearBlend.onClick = [this]
    {
        ctx.undo.beginNewTransaction ("Clear blend voice");
        ctx.processor.setBlendVoice ({});
    };

    tick();
}

VoiceCharacterPanel::~VoiceCharacterPanel() = default;

void VoiceCharacterPanel::resized()
{
    pitch.setBounds (14, 50, 120, 152);
    fineTune.setBounds (130, 104, 46, 86);

    const int x0 = 184, w = (getWidth() - x0 - 8) / 4;
    int x = x0;
    for (auto* k : { &gender, &age, &tone, &style })
    {
        k->setBounds (x, 50, w, 152);
        x += w;
    }

    // blend control lives in the title row
    blendFromArea = { 222.0f, 8.0f, 96.0f, 30.0f };
    blendSlider.setBounds (334, 9, 134, 28);
    blendToArea = { 484.0f, 8.0f, 90.0f, 30.0f };
    clearBlend.setBounds (getWidth() - 30, 10, 24, 26);
}

void VoiceCharacterPanel::paintStatic (juce::Graphics& g, juce::Rectangle<float> area)
{
    GlassPanel::paintStatic (g, area);

    // divider between pitch and character controls
    juce::ColourGradient cg (Palette::panelBorder.withAlpha (0.0f), 0.0f, 60.0f, Palette::panelBorder.withAlpha (0.0f), 0.0f, 196.0f, false);
    cg.addColour (0.5, Palette::panelBorder.withAlpha (0.2f));
    g.setGradientFill (cg);
    g.fillRect (juce::Rectangle<float> (180.0f, 60.0f, 1.0f, 136.0f));
}

void VoiceCharacterPanel::paint (juce::Graphics& g)
{
    GlassPanel::paint (g);

    if (blendSlider.isVisible())
    {
        auto pill = juce::Rectangle<float> (blendFromArea.getX() - 6.0f, 6.0f, (float) getWidth() - blendFromArea.getX() - 2.0f, 34.0f);
        g.setColour (juce::Colour (0x80050b1e));
        g.fillRoundedRectangle (pill, 8.0f);
        g.setColour (juce::Colour (0xffb489ff).withAlpha (0.5f));
        g.drawRoundedRectangle (pill.reduced (0.5f), 8.0f, 1.0f);

        const auto f = font (14.5f, Weight::Medium);
        drawText (g, blendFrom, blendFromArea, f, Palette::text, juce::Justification::centredRight);
        drawText (g, blendTo, blendToArea, f, Palette::text, juce::Justification::centredLeft);
        Icons::draw (g, Icon::ChevronLeft, { (float) blendSlider.getX() - 14.0f, 16.0f, 12.0f, 12.0f }, Palette::textSoft, 2.4f);
        Icons::draw (g, Icon::ChevronRight, { (float) blendSlider.getRight() + 2.0f, 16.0f, 12.0f, 12.0f }, Palette::textSoft, 2.4f);
    }
}

void VoiceCharacterPanel::tick()
{
    const auto blendId = ctx.processor.getBlendVoiceId();
    const auto voiceId = ctx.processor.getSelectedVoiceId();
    if (blendId == shownBlendId && voiceId == shownVoiceId)
        return;

    shownBlendId = blendId;
    shownVoiceId = voiceId;

    const auto* blendVoice = ctx.processor.getVoiceLibrary().findById (blendId);
    const bool show = blendVoice != nullptr;
    blendSlider.setVisible (show);
    clearBlend.setVisible (show);

    if (show)
    {
        blendFrom = ctx.processor.getSelectedVoice().name;
        blendTo = blendVoice->name;
        blendSlider.setTooltip ("Voice Blend: morph from " + blendFrom + " to " + blendTo);
    }
    repaint (0, 0, getWidth(), (int) titleHeight);
}

} // namespace voxa::ui
