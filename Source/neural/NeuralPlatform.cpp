#include "NeuralPlatform.h"

#if JUCE_WINDOWS
 #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
 #endif
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #include <windows.h>
#else
 #include <dlfcn.h>
#endif

namespace voxa::neural::platform
{

juce::StringArray getRuntimeLibraryNames()
{
   #if JUCE_WINDOWS
    return { "onnxruntime.dll" };
   #elif JUCE_MAC
    return { "libonnxruntime.dylib" };
   #else
    return { "libonnxruntime.so", "libonnxruntime.so.1" };
   #endif
}

void* openLibrary (const juce::String& pathOrName, juce::String& error)
{
   #if JUCE_WINDOWS
    // Never show "missing DLL" message boxes inside a host; look for dependencies (e.g. DirectML.dll)
    // next to the library itself rather than next to the host executable.
    DWORD previousMode = 0;
    SetThreadErrorMode (SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &previousMode);

    const bool isPath = juce::File::isAbsolutePath (pathOrName);
    HMODULE handle = isPath ? LoadLibraryExW (pathOrName.toWideCharPointer(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)
                            : LoadLibraryW (pathOrName.toWideCharPointer());
    const auto lastError = GetLastError();
    SetThreadErrorMode (previousMode, nullptr);

    if (handle == nullptr)
        error = "Windows error " + juce::String ((int) lastError);

    return (void*) handle;
   #else
    dlerror();
    auto* handle = dlopen (pathOrName.toRawUTF8(), RTLD_NOW | RTLD_LOCAL);

    if (handle == nullptr)
    {
        const char* message = dlerror();
        error = message != nullptr ? juce::String::fromUTF8 (message) : juce::String ("unknown error");
    }

    return handle;
   #endif
}

void* getSymbol (void* handle, const char* name)
{
    if (handle == nullptr)
        return nullptr;

   #if JUCE_WINDOWS
    return (void*) GetProcAddress ((HMODULE) handle, name);
   #else
    return dlsym (handle, name);
   #endif
}

juce::File getLibraryFile (void* handle, void* symbol)
{
   #if JUCE_WINDOWS
    juce::ignoreUnused (symbol);
    std::vector<wchar_t> buffer (32768);
    const auto length = GetModuleFileNameW ((HMODULE) handle, buffer.data(), (DWORD) buffer.size());

    if (length > 0 && length < buffer.size())
        return juce::File (juce::String (buffer.data(), (size_t) length));

    return {};
   #else
    juce::ignoreUnused (handle);
    Dl_info info {};

    if (symbol != nullptr && dladdr (symbol, &info) != 0 && info.dli_fname != nullptr)
    {
        const auto path = juce::String::fromUTF8 (info.dli_fname);

        if (juce::File::isAbsolutePath (path))
            return juce::File (path);

        return juce::File::getCurrentWorkingDirectory().getChildFile (path);
    }

    return {};
   #endif
}

} // namespace voxa::neural::platform
