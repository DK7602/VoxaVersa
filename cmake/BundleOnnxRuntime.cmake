# =============================================================================================
# BundleOnnxRuntime.cmake - optional: ship ONNX Runtime next to the VoxaVersa binaries
#
# VoxaVersa never links against ONNX Runtime: the AI voice engine loads the shared library at run
# time and simply falls back to the DSP engine when it is missing (see docs/NEURAL_ENGINE.md).
# To bundle a runtime with your builds, download an official release archive from
# https://github.com/microsoft/onnxruntime/releases (1.16 or newer), extract it and configure with
#
#     cmake -S . -B build -DVOXAVERSA_ONNXRUNTIME_DIR=/path/to/onnxruntime-<platform>-<version>
#
# After every build the runtime library (plus any provider libraries in the archive) is copied
# next to each plugin / app binary, or into Contents/Frameworks for macOS bundles - the first
# place VoxaVersa looks. Without VOXAVERSA_ONNXRUNTIME_DIR this file does nothing.
#
# macOS: copying a file into a signed bundle invalidates its signature, so (re-)sign the bundles
# after building, e.g. `codesign --force --deep --sign - VoxaVersa.vst3` (or your Developer ID).
# =============================================================================================

set(VOXAVERSA_ONNXRUNTIME_DIR "" CACHE PATH
    "Extracted ONNX Runtime release whose shared library is copied next to the built binaries (optional)")

if(NOT VOXAVERSA_ONNXRUNTIME_DIR)
    return()
endif()

if(EXISTS "${VOXAVERSA_ONNXRUNTIME_DIR}/lib")
    set(_voxa_ort_lib_dir "${VOXAVERSA_ONNXRUNTIME_DIR}/lib")
else()
    set(_voxa_ort_lib_dir "${VOXAVERSA_ONNXRUNTIME_DIR}")
endif()

# Main library: file name inside the archive -> name VoxaVersa searches for
if(WIN32)
    set(_voxa_ort_main_source "${_voxa_ort_lib_dir}/onnxruntime.dll")
    set(_voxa_ort_main_name   "onnxruntime.dll")
    file(GLOB _voxa_ort_extras
        "${_voxa_ort_lib_dir}/onnxruntime_providers_*.dll"
        "${_voxa_ort_lib_dir}/DirectML.dll")
elseif(APPLE)
    set(_voxa_ort_main_source "${_voxa_ort_lib_dir}/libonnxruntime.dylib")
    set(_voxa_ort_main_name   "libonnxruntime.dylib")
    file(GLOB _voxa_ort_extras "${_voxa_ort_lib_dir}/libonnxruntime_providers_*.dylib")
else()
    set(_voxa_ort_main_source "${_voxa_ort_lib_dir}/libonnxruntime.so")
    set(_voxa_ort_main_name   "libonnxruntime.so")
    file(GLOB _voxa_ort_extras "${_voxa_ort_lib_dir}/libonnxruntime_providers_*.so")
endif()

if(NOT EXISTS "${_voxa_ort_main_source}")
    message(FATAL_ERROR "VOXAVERSA_ONNXRUNTIME_DIR: ${_voxa_ort_main_source} not found "
                        "(expected an extracted ONNX Runtime release archive)")
endif()

# Release archives ship symlink chains (libonnxruntime.so -> .so.1 -> .so.1.20.1): copy the real file once.
file(REAL_PATH "${_voxa_ort_main_source}" _voxa_ort_main_real)

set(_voxa_ort_copy_commands
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_voxa_ort_main_real}" "@DEST@/${_voxa_ort_main_name}")

foreach(_voxa_ort_extra IN LISTS _voxa_ort_extras)
    file(REAL_PATH "${_voxa_ort_extra}" _voxa_ort_extra_real)
    get_filename_component(_voxa_ort_extra_name "${_voxa_ort_extra}" NAME)
    list(APPEND _voxa_ort_copy_commands
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_voxa_ort_extra_real}" "@DEST@/${_voxa_ort_extra_name}")
endforeach()

set(_voxa_ort_bundled_targets "")

foreach(_voxa_product IN ITEMS VoxaVersa VoxaVersaSinger)
    foreach(_voxa_format IN ITEMS Standalone VST3 AU AUv3 LV2 AAX)
        set(_voxa_target "${_voxa_product}_${_voxa_format}")

        if(NOT TARGET "${_voxa_target}")
            continue()
        endif()

        get_target_property(_voxa_is_bundle "${_voxa_target}" BUNDLE)              # plugins (CFBundle)
        get_target_property(_voxa_is_app_bundle "${_voxa_target}" MACOSX_BUNDLE)   # standalone .app

        if(APPLE AND (_voxa_is_bundle OR _voxa_is_app_bundle))
            set(_voxa_dest "$<TARGET_BUNDLE_CONTENT_DIR:${_voxa_target}>/Frameworks")
        else()
            set(_voxa_dest "$<TARGET_FILE_DIR:${_voxa_target}>")
        endif()

        string(REPLACE "@DEST@" "${_voxa_dest}" _voxa_commands "${_voxa_ort_copy_commands}")

        add_custom_command(TARGET "${_voxa_target}" POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_voxa_dest}"
            ${_voxa_commands}
            COMMENT "Bundling ONNX Runtime with ${_voxa_target}"
            VERBATIM)

        list(APPEND _voxa_ort_bundled_targets "${_voxa_target}")
    endforeach()
endforeach()

message(STATUS "VoxaVersa: bundling ONNX Runtime (${_voxa_ort_main_real}) with: ${_voxa_ort_bundled_targets}")
