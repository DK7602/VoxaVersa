#include "PluginEditor.h"
#include "ui/MainView.h"
#include "ui/VoiceAvatar.h"

namespace voxa
{

namespace
{
    juce::File getSettingsFile()
    {
        return VoiceLibrary::getUserDataDirectory().getChildFile ("settings.xml");
    }

    juce::String env (const char* name)
    {
        return juce::SystemStats::getEnvironmentVariable (name, {});
    }
}

//==============================================================================
/** Opt-in screenshot hook used for visual verification:
      VOXAVERSA_SNAPSHOT=<file.png>        render the editor at 1536 x 1024 after a delay, save it, quit (standalone)
      VOXAVERSA_SNAPSHOT_MODE=sing|voice   switch the mode first
      VOXAVERSA_SNAPSHOT_TAB=<tab>         select an Advanced tab (Advanced, Expression, Tune, Harmony, FX, Space)
      VOXAVERSA_SNAPSHOT_GRID=1            open the voice grid
      VOXAVERSA_SNAPSHOT_DELAY=<ms>        delay before capturing (default 1500)
      VOXAVERSA_SNAPSHOT_DEMO=0            do not feed the demo signal into the scopes / meters
      VOXAVERSA_SNAPSHOT_SIZE=<width>      editor width for the capture (default 1536)
      VOXAVERSA_AVATAR_SHEET=<file.png>    render every voice portrait into a contact sheet and quit
      VOXAVERSA_DEMO=1                     only animate the UI with the demo signal (UI tests / profiling) */
struct VoxaVersaEditor::Snapshot
{
    juce::File output, avatarSheet;
    double startMs = 0.0, delayMs = 1500.0;
    bool demo = true, done = false, reportedBounds = false;
    double time = 0.0;
    std::vector<float> left, right;
    juce::Random random { 42 };

    void feed (VoxaVersaProcessor& p, float dt)
    {
        if (! demo)
            return;

        const double sr = 48000.0;
        const int n = juce::jlimit (1, 8192, (int) (sr * dt));
        left.resize ((size_t) n);
        right.resize ((size_t) n);

        for (int i = 0; i < n; ++i)
        {
            const double t = time + (double) i / sr;
            const double syll = (0.3 + 0.7 * std::pow (std::abs (std::sin (juce::MathConstants<double>::twoPi * 1.3 * t)), 0.7))
                              * (0.6 + 0.4 * std::sin (juce::MathConstants<double>::twoPi * 0.37 * t));
            // harmonic series via the Chebyshev recurrence sin(k x) = 2 cos(x) sin((k-1) x) - sin((k-2) x)
            const double phi = juce::MathConstants<double>::twoPi * 220.0 * t;
            const double c2 = 2.0 * std::cos (phi);
            double sPrev = 0.0, sCur = std::sin (phi), s = 0.0;
            for (int k = 1; k <= 6; ++k)
            {
                s += sCur / (double) k;
                const double next = c2 * sCur - sPrev;
                sPrev = sCur;
                sCur = next;
            }
            const double noise = (random.nextDouble() - 0.5) * 0.3;
            const auto block = (juce::int64) (t * 220.0);
            const double spike = 0.25 + 0.75 * (double) ((block * 2654435761LL) % 1000) / 1000.0;
            left[(size_t) i] = (float) (0.5 * syll * spike * (s * 0.6 + noise));
            right[(size_t) i] = (float) (0.55 * std::pow (syll, 0.8) * std::sqrt (spike) * (s * 0.55 + noise * 1.4));
        }
        time += (double) n / sr;

        const float* in[] = { left.data(), left.data() };
        const float* out[] = { right.data(), right.data() };
        p.inputScope.push (left.data(), n);
        p.inputMeter.update (in, 2, n);
        p.outputScope.push (right.data(), n);
        p.outputMeter.update (out, 2, n);
        p.detectedPitchHz = 220.0f;
        p.targetPitchHz = 277.18f;
        p.detectedKeyRoot = 1;
        p.detectedScale = 1;
    }
};

//==============================================================================
VoxaVersaEditor::VoxaVersaEditor (VoxaVersaProcessor& p)
    : AudioProcessorEditor (p), processor (p), context (p)
{
    setLookAndFeel (&lookAndFeel.get());
    setOpaque (true);
    setTitle ("VoxaVersa");
    setDescription ("AI vocal transformation and text-to-sing");

    view = std::make_unique<ui::MainView> (context);
    addAndMakeVisible (*view);
    context.showToast = [this] (const juce::String& message, ui::ToastKind kind)
    {
        if (view != nullptr)
            view->toast.show (message, kind);
    };

    view->header.onUiSizeChosen = [this] (int w) { setUiWidth (w); };
    view->header.getUiWidth = [this] { return getWidth(); };

    tooltips = std::make_unique<juce::TooltipWindow> (view.get(), 650);
    tooltips->setLookAndFeel (&lookAndFeel.get());

    setResizable (true, true);
    setResizeLimits (minWidth, minWidth * 2 / 3, maxWidth, maxWidth * 2 / 3);
    if (auto* c = getConstrainer())
        c->setFixedAspectRatio ((double) ui::MainView::designWidth / (double) ui::MainView::designHeight);

    int width = loadSavedWidth();

    // Screenshot / visual-verification hook (opt-in through environment variables).
    const auto snapPath = env ("VOXAVERSA_SNAPSHOT");
    const auto sheetPath = env ("VOXAVERSA_AVATAR_SHEET");
    const bool demoOnly = env ("VOXAVERSA_DEMO") == "1";
    if (snapPath.isNotEmpty() || sheetPath.isNotEmpty() || demoOnly)
    {
        snapshot = std::make_unique<Snapshot>();
        snapshot->output = juce::File (snapPath);
        snapshot->avatarSheet = juce::File (sheetPath);
        snapshot->startMs = juce::Time::getMillisecondCounterHiRes();
        snapshot->delayMs = env ("VOXAVERSA_SNAPSHOT_DELAY").isNotEmpty() ? env ("VOXAVERSA_SNAPSHOT_DELAY").getDoubleValue() : 1500.0;
        snapshot->demo = env ("VOXAVERSA_SNAPSHOT_DEMO") != "0";
        if (! demoOnly)
        {
            width = ui::MainView::designWidth;
            tooltips->setMillisecondsBeforeTipAppears (1000000);   // keep screenshots free of tooltips
        }
        if (env ("VOXAVERSA_SNAPSHOT_SIZE").isNotEmpty())
            width = juce::jlimit (minWidth, maxWidth, env ("VOXAVERSA_SNAPSHOT_SIZE").getIntValue());

        const auto mode = env ("VOXAVERSA_SNAPSHOT_MODE");
        if (mode.isNotEmpty())
        {
            const bool sing = mode.equalsIgnoreCase ("sing");
            if (auto* param = processor.getState().getParameter (ParamID::mode))
                param->setValueNotifyingHost (param->convertTo0to1 (sing ? 1.0f : 0.0f));
            if (sing && processor.getLyrics().isEmpty())
                processor.setLyrics ("City lights are call-ing out my name\nDanc-ing through the night, we'll nev-er be the same");
        }

        if (auto tab = env ("VOXAVERSA_SNAPSHOT_TAB"); tab.isNotEmpty())
            view->advanced.showTabNamed (tab);
        if (env ("VOXAVERSA_SNAPSHOT_GRID") == "1")
            view->library.setExpanded (true);
        if (snapshot->demo)
        {
            processor.inputScope.prepare (48000.0);
            processor.outputScope.prepare (48000.0);
        }
    }

    setSize (width, juce::roundToInt (width * 2.0 / 3.0));
    lastTickMs = juce::Time::getMillisecondCounterHiRes();
    startTimerHz (30);
}

VoxaVersaEditor::~VoxaVersaEditor()
{
    stopTimer();
    saveWidthToSettings();
    tooltips->setLookAndFeel (nullptr);
    tooltips.reset();
    view.reset();
    setLookAndFeel (nullptr);
}

void VoxaVersaEditor::paint (juce::Graphics& g)
{
    g.fillAll (ui::Palette::canvasDeep);
}

void VoxaVersaEditor::resized()
{
    const float scale = (float) getWidth() / (float) ui::MainView::designWidth;
    view->setTransform (juce::AffineTransform::scale (scale));
    view->setBounds (0, 0, ui::MainView::designWidth, ui::MainView::designHeight);
    view->noteScaleChanged();

    if (snapshot == nullptr)
        processor.getState().state.setProperty ("uiWidth", getWidth(), nullptr);
}

void VoxaVersaEditor::setUiWidth (int w)
{
    w = juce::jlimit (minWidth, maxWidth, w);
    setSize (w, juce::roundToInt (w * 2.0 / 3.0));
    saveWidthToSettings();
}

int VoxaVersaEditor::loadSavedWidth() const
{
    const auto stored = processor.getState().state.getProperty ("uiWidth");
    if (! stored.isVoid() && (int) stored > 0)
        return juce::jlimit (minWidth, maxWidth, (int) stored);

    if (auto xml = juce::parseXML (getSettingsFile()))
        if (auto* ui = xml->getChildByName ("Ui"))
            return juce::jlimit (minWidth, maxWidth, ui->getIntAttribute ("width", defaultWidth));

    return defaultWidth;
}

void VoxaVersaEditor::saveWidthToSettings() const
{
    if (snapshot != nullptr)
        return;

    const auto file = getSettingsFile();
    std::unique_ptr<juce::XmlElement> xml (juce::parseXML (file));
    if (xml == nullptr || ! xml->hasTagName ("VoxaVersaSettings"))
        xml = std::make_unique<juce::XmlElement> ("VoxaVersaSettings");

    auto* ui = xml->getChildByName ("Ui");
    if (ui == nullptr)
        ui = xml->createNewChildElement ("Ui");

    if (ui->getIntAttribute ("width") == getWidth())
        return;

    ui->setAttribute ("width", getWidth());
    file.getParentDirectory().createDirectory();
    xml->writeTo (file);
}

bool VoxaVersaEditor::keyPressed (const juce::KeyPress& key)
{
    const auto mods = key.getModifiers();
    const int code = key.getKeyCode();

    if (mods.isCommandDown() && (code == 'Z' || code == 'z'))
    {
        processor.getUndoManager().beginNewTransaction();
        if (mods.isShiftDown())
            processor.getUndoManager().redo();
        else
            processor.getUndoManager().undo();
        return true;
    }

    if (mods.isCommandDown() && (code == 'Y' || code == 'y'))
    {
        processor.getUndoManager().redo();
        return true;
    }

    if (mods.isCommandDown() && (code == 'S' || code == 's'))
    {
        view->header.savePresetDialog();
        return true;
    }

    return false;
}

void VoxaVersaEditor::timerCallback()
{
    const double now = juce::Time::getMillisecondCounterHiRes();
    const float dt = (float) juce::jlimit (0.0, 0.2, (now - lastTickMs) * 0.001);
    lastTickMs = now;

    if (snapshot != nullptr && ! snapshot->done)
    {
        snapshot->feed (processor, dt);

        if (! snapshot->reportedBounds && isShowing())
        {
            // lets UI test scripts find the editor on screen
            snapshot->reportedBounds = true;
            std::fprintf (stderr, "VOXAVERSA_EDITOR_BOUNDS %s\n", getScreenBounds().toString().toRawUTF8());
        }
    }

    context.tick();
    view->tick (dt);

    if (snapshot != nullptr && ! snapshot->done && now - snapshot->startMs > snapshot->delayMs
        && (snapshot->output != juce::File() || snapshot->avatarSheet != juce::File()))
    {
        snapshot->done = true;

        if (snapshot->avatarSheet != juce::File())
        {
            const auto& voices = processor.getVoiceLibrary().getVoices();
            const int cols = 8, cw = 170, ch = 176;
            const int rows = ((int) voices.size() + cols - 1) / cols;
            juce::Image sheet (juce::Image::RGB, cols * cw, rows * ch, true);
            juce::Graphics g (sheet);
            g.fillAll (juce::Colour (0xff0a1430));
            for (int i = 0; i < (int) voices.size(); ++i)
            {
                auto img = ui::renderVoiceAvatar (voices[(size_t) i], cw - 10, ch - 10);
                const int x = (i % cols) * cw + 5, y = (i / cols) * ch + 5;
                g.drawImageAt (img, x, y);
                ui::drawText (g, voices[(size_t) i].name, juce::Rectangle<float> ((float) x, (float) (y + ch - 40), (float) (cw - 10), 26.0f),
                              ui::font (16.0f, ui::Weight::SemiBold), juce::Colours::white);
            }
            snapshot->avatarSheet.deleteFile();
            juce::FileOutputStream out (snapshot->avatarSheet);
            juce::PNGImageFormat().writeImageToStream (sheet, out);
        }

        if (snapshot->output != juce::File())
        {
            const auto image = createComponentSnapshot (getLocalBounds(), true, 1.0f);
            snapshot->output.deleteFile();
            juce::FileOutputStream out (snapshot->output);
            juce::PNGImageFormat().writeImageToStream (image, out);
        }

        if (processor.isStandaloneApp())
            juce::JUCEApplicationBase::quit();
    }
}

} // namespace voxa
