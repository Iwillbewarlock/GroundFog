#pragma once

#include "RE/Skyrim.h"
#include "REL/Relocation.h"
#include "SKSE/SKSE.h"
#include "SKSE/Logger.h"

#include <d3d11.h>
#include <d3dcompiler.h>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace std::literals;

namespace logger = SKSE::log;

// windows.h (minwindef.h) defines near and far as empty macros: a local named "near" then fails to compile in odd
// ways (hit three times on 2026-11-08). Nothing here uses the 16-bit pointer keywords.
#undef near
#undef far
