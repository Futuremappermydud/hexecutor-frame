#pragma once
// Public entry points for the HexecutorBackend delegate. The backend also
// self-registers from a static initializer; callers that link this archive
// statically should call ensure_registered() so the object file is kept.
#include <executorch/runtime/core/error.h>

namespace hexecutor_et {
// Returns Ok if the backend is registered under the name "HexecutorBackend".
executorch::runtime::Error ensure_registered();
} // namespace hexecutor_et
