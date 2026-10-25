cmake_minimum_required(VERSION 3.22)

project(VoxaVersa VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Universal binary")
endif()

option(VOXAVERSA_BUILD_SINGER "Also build the 'VoxaVersa Singer' instrument variant" ON)
option(VOXAVERSA_BUILD_TESTS  "Build the unit test runner" ON)
option(VOXAVERSA_BUILD_TOOLS  "Build developer tools (offline render CLI)" OFF)
set(VOXAVERSA_AAX_SDK_PATH "" CACHE PATH "Path to the Avid AAX SDK (enables the AAX format)")

# ---------------------------------------------------------------------------------------------
# JUCE. Pass -DFETCHCONTENT_SOURCE_DIR_JUCE=/path/to/JUCE to use a local checkout.
# ---------------------------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.9
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(JUCE)

if(VOXAVERSA_AAX_SDK_PATH)
    juce_set_aax_sdk_path(${VOXAVERSA_AAX_SDK_PATH})
endif()

set(VOXA_FORMATS VST3 Standalone)
if(APPLE)
    list(APPEND VOXA_FORMATS AU)
endif()
if(UNIX AND NOT APPLE)
    list(APPEND VOXA_FORMATS LV2)
endif()
if(VOXAVERSA_AAX_SDK_PATH)
    list(APPEND VOXA_FORMATS AAX)
endif()

# ---------------------------------------------------------------------------------------------
# Shared sources & assets
# ---------------------------------------------------------------------------------------------
file(GLOB_RECURSE VOXA_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Source/*.h)

file(GLOB VOXA_ASSETS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/Resources/images/*.png
    ${CMAKE_CURRENT_SOURCE_DIR}/Resources/voices/*.jpg
    ${CMAKE_CURRENT_SOURCE_DIR}/Resources/fonts/*.ttf
    ${CMAKE_CURRENT_SOURCE_DIR}/Resources/fonts/*.otf
    ${CMAKE_CURRENT_SOURCE_DIR}/Resources/sing/*.gz)

juce_add_binary_data(VoxaVersaAssets
    NAMESPACE VoxaAssets
    HEADER_NAME VoxaAssets.h
    SOURCES ${VOXA_ASSETS})

set(VOXA_COMMON_DEFINITIONS
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_STRICT_REFCOUNTEDPOINTER=1
    JUCE_SILENCE_XCODE_15_LINKER_WARNING=1
    VOXAVERSA_VERSION_STRING="${PROJECT_VERSION}")

set(VOXA_JUCE_MODULES
    juce::juce_audio_utils
    juce::juce_audio_formats
    juce::juce_dsp)

function(voxa_configure_target target)
    target_sources(${target} PRIVATE ${VOXA_SOURCES})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source)
    target_compile_definitions(${target} PUBLIC ${VOXA_COMMON_DEFINITIONS})
    target_link_libraries(${target}
        PRIVATE
            VoxaVersaAssets
            ${VOXA_JUCE_MODULES}
            ${CMAKE_DL_LIBS}
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_lto_flags
            juce::juce_recommended_warning_flags)
endfunction()

# ---------------------------------------------------------------------------------------------
# VoxaVersa: audio effect (with MIDI input) + standalone app
# ---------------------------------------------------------------------------------------------
juce_add_plugin(VoxaVersa
    PRODUCT_NAME                "VoxaVersa"
    COMPANY_NAME                "VoxaVersa"
    COMPANY_WEBSITE             "https://voxaversa.com"
    BUNDLE_ID                   com.voxaversa.voxaversa
    PLUGIN_MANUFACTURER_CODE    Voxa
    PLUGIN_CODE                 VxVs
    FORMATS                     ${VOXA_FORMATS}
    IS_SYNTH                    FALSE
    NEEDS_MIDI_INPUT            TRUE
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD     FALSE
    VST3_CATEGORIES             Fx Pitch Shift
    AU_MAIN_TYPE                kAudioUnitType_MusicEffect
    AAX_CATEGORY                AAX_ePlugInCategory_PitchShift
    LV2URI                      https://voxaversa.com/plugins/voxaversa
    ICON_BIG                    ${CMAKE_CURRENT_SOURCE_DIR}/Resources/images/icon.png
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT  "VoxaVersa needs microphone access to record and transform your voice.")

voxa_configure_target(VoxaVersa)

# ---------------------------------------------------------------------------------------------
# VoxaVersa Singer: instrument variant (text-to-sing driven by MIDI, optional sidechain input)
# ---------------------------------------------------------------------------------------------
if(VOXAVERSA_BUILD_SINGER)
    set(VOXA_SINGER_FORMATS ${VOXA_FORMATS})
    list(REMOVE_ITEM VOXA_SINGER_FORMATS Standalone)

    juce_add_plugin(VoxaVersaSinger
        PRODUCT_NAME                "VoxaVersa Singer"
        COMPANY_NAME                "VoxaVersa"
        COMPANY_WEBSITE             "https://voxaversa.com"
        BUNDLE_ID                   com.voxaversa.voxaversasinger
        PLUGIN_MANUFACTURER_CODE    Voxa
        PLUGIN_CODE                 VxSg
        FORMATS                     ${VOXA_SINGER_FORMATS}
        IS_SYNTH                    TRUE
        NEEDS_MIDI_INPUT            TRUE
        NEEDS_MIDI_OUTPUT           FALSE
        IS_MIDI_EFFECT              FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS TRUE
        COPY_PLUGIN_AFTER_BUILD     FALSE
        VST3_CATEGORIES             Instrument Synth
        AU_MAIN_TYPE                kAudioUnitType_MusicDevice
        AAX_CATEGORY                AAX_ePlugInCategory_SWGenerators
        LV2URI                      https://voxaversa.com/plugins/voxaversa-singer
        ICON_BIG                    ${CMAKE_CURRENT_SOURCE_DIR}/Resources/images/icon.png)

    voxa_configure_target(VoxaVersaSinger)
endif()

# Console targets compile the plug-in sources without a plug-in wrapper.
function(voxa_configure_console_target target)
    voxa_configure_target(${target})
    target_compile_definitions(${target} PRIVATE
        VOXAVERSA_TESTS=1
        JUCE_MODAL_LOOPS_PERMITTED=1
        JucePlugin_Name="VoxaVersa"
        JucePlugin_IsSynth=0
        JucePlugin_WantsMidiInput=1
        JucePlugin_ProducesMidiOutput=0
        JucePlugin_IsMidiEffect=0
        JucePlugin_EditorRequiresKeyboardFocus=1)
endfunction()

# ---------------------------------------------------------------------------------------------
# Unit tests (JUCE UnitTest framework, no external dependencies)
# ---------------------------------------------------------------------------------------------
if(VOXAVERSA_BUILD_TESTS)
    enable_testing()
    file(GLOB_RECURSE VOXA_TEST_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/*.h)

    juce_add_console_app(VoxaVersaTests PRODUCT_NAME "VoxaVersaTests")
    target_sources(VoxaVersaTests PRIVATE ${VOXA_TEST_SOURCES})
    voxa_configure_console_target(VoxaVersaTests)
    target_include_directories(VoxaVersaTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Tests)
    add_test(NAME VoxaVersaTests COMMAND VoxaVersaTests)
endif()

# ---------------------------------------------------------------------------------------------
# Developer tools
# ---------------------------------------------------------------------------------------------
if(VOXAVERSA_BUILD_TOOLS)
    juce_add_console_app(VoxaVersaRender PRODUCT_NAME "VoxaVersaRender")
    target_sources(VoxaVersaRender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/render/RenderMain.cpp)
    voxa_configure_console_target(VoxaVersaRender)
endif()

# ---------------------------------------------------------------------------------------------
# Optional: copy ONNX Runtime (AI voice engine) next to the built binaries.
# Opt-in with -DVOXAVERSA_ONNXRUNTIME_DIR=<extracted ONNX Runtime release>; see the file for details.
# ---------------------------------------------------------------------------------------------
include(cmake/BundleOnnxRuntime.cmake)
