#pragma once

// Includes the vendored ONNX Runtime C API header (types and the OrtApi function table only - VoxaVersa
// never links against ONNX Runtime; see OnnxRuntime in NeuralVoice.h).

#include <juce_core/juce_core.h>

JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wshadow", "-Wshadow-all", "-Wconversion", "-Wsign-conversion",
                                     "-Wzero-as-null-pointer-constant", "-Wpedantic", "-Wextra-semi",
                                     "-Wmissing-prototypes", "-Wnullable-to-nonnull-conversion",
                                     "-Wdeprecated", "-Wunused-parameter", "-Wold-style-cast",
                                     "-Wnon-virtual-dtor", "-Wredundant-decls", "-Wundef",
                                     "-Wdocumentation", "-Wlanguage-extension-token")
JUCE_BEGIN_IGNORE_WARNINGS_MSVC (4100 4201 4505 4996)

#include "third_party/onnxruntime/onnxruntime_c_api.h"

JUCE_END_IGNORE_WARNINGS_MSVC
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

namespace voxa::neural
{
/** The C API version VoxaVersa requests: 16 == ONNX Runtime 1.16. The OrtApi struct is append-only, so any
    newer runtime serves this version too. Only call functions that exist in this version! */
inline constexpr uint32_t requiredOrtApiVersion = 16;
inline constexpr const char* minimumOrtVersionString = "1.16";
} // namespace voxa::neural
