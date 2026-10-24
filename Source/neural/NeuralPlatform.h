#pragma once

#include <juce_core/juce_core.h>

/** Minimal shared-library loading used by the neural engine (kept separate so that <windows.h> is
    confined to a single translation unit). Libraries opened here are never closed: ONNX Runtime owns
    global state and thread pools that must stay valid until the process exits. */
namespace voxa::neural::platform
{

/** File names under which the ONNX Runtime shared library is published on this platform, in the order
    they are tried inside each search directory. */
juce::StringArray getRuntimeLibraryNames();

/** Opens a shared library. `pathOrName` is either an absolute path or a bare file name, which is then
    resolved by the system loader. Returns nullptr and fills `error` on failure. */
void* openLibrary (const juce::String& pathOrName, juce::String& error);

void* getSymbol (void* handle, const char* name);

/** Best-effort path of the file a library was loaded from (`symbol` is any symbol inside it). */
juce::File getLibraryFile (void* handle, void* symbol);

} // namespace voxa::neural::platform
