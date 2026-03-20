#pragma once

/// Thin compatibility shim for Tracy profiler zones.
///
/// Include this header instead of <tracy/Tracy.hpp> directly.
/// When TRACY_ENABLE is defined (set by linking TracyClient with VDBSCAN_TRACY=ON),
/// the real Tracy macros are active. Otherwise all macros expand to nothing,
/// leaving zero overhead and no dependency on the Tracy headers.

#ifdef TRACY_ENABLE
#  include <tracy/Tracy.hpp>
#else
#  define ZoneScoped
#  define ZoneScopedN(name)
#  define FrameMark
#endif
