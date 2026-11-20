// SPDX-License-Identifier: LGPL-3.0-or-later
// libhexecutor_et: HexecutorBackend, the hexecutor host runtime and the ExecuTorch runtime as one shared
// library (see CMakeLists.txt). The code comes from the whole archives it is linked from; this unit only
// gives the library a source file and keeps the backend's registrar object referenced.
#include "hexecutor_backend.hpp"

namespace hexecutor_et {
namespace {
// Referencing ensure_registered keeps hexecutor_backend.o (and its static registrar) in the library even if
// a linker drops unreferenced archive members despite the whole-archive flags.
[[maybe_unused]] auto *const g_keep = &ensure_registered;
} // namespace
} // namespace hexecutor_et
