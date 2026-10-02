// include/strata/core/memory_gate.hpp - may the engine start on THIS machine, right now?
//
// WHY THIS EXISTS.  The expert arena is held resident (~46.84 GiB on the 2x2080Ti box, out of
// 60.42 GiB of RAM).  Starting the engine into a machine that is already carrying a browser and a
// desktop does not fail loudly: the kernel has to keep ~48 GiB unreclaimable while everything else
// fights over the rest, and the symptom is a frozen desktop (dsh-web / rustdesk), not an error.
// The gate turns that into a startup refusal that names the numbers.
//
// IT IS A POINT-IN-TIME GUARD, NOT A RESERVATION.  `available_memory_bytes()` reads MemAvailable
// (clamped to the tightest cgroup-v2 ancestor limit when one is visible), and another process can
// take that memory a millisecond later.  The gate's job is to refuse the OBVIOUSLY hopeless start
// ("40 GiB free, 47 GiB wanted"), not to promise anything.
//
// AND IT CAN BE TURNED OFF: `STRATA_START_HEADROOM_GIB=0` disables the margin, after which the
// engine starts whenever the arena itself fits.  A gate with no way out is a trap.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core {

struct MemoryGate {
    bool ok = true;
    uint64_t available = 0;   ///< what /proc/meminfo (or the tightest cgroup ancestor) said
    uint64_t needed = 0;      ///< the arena the caller is about to allocate
    uint64_t headroom = 0;    ///< the margin kept for everything else
    uint64_t shortfall = 0;   ///< needed + headroom - available, when !ok
    std::string message;      ///< empty when ok; the refusal otherwise
};

/// The decision, with no machine state: `available >= needed + headroom`.
///
/// The addition is SATURATING on purpose: a wrapped requirement would read as "fits", and a gate
/// that passes by arithmetic accident is worse than no gate at all.
inline MemoryGate memory_gate(uint64_t available, uint64_t needed, uint64_t headroom) {
    MemoryGate g;
    g.available = available;
    g.needed = needed;
    g.headroom = headroom;

    const uint64_t required = (needed > UINT64_MAX - headroom) ? UINT64_MAX : needed + headroom;
    if (available >= required) return g;

    g.ok = false;
    g.shortfall = required - available;  // required > available here, so this cannot wrap

    char buf[768];
    std::snprintf(buf, sizeof buf,
                  "not enough RAM to start: %.1f GiB available, but the expert arena needs "
                  "%.1f GiB and %.1f GiB is kept as headroom for the rest of the machine "
                  "(short by %.1f GiB).  The engine is REFUSED rather than started, because it "
                  "holds the arena resident and it is the MACHINE that fails, not the engine.\n"
                  "  Fix any one of, then start again:\n"
                  "   1. free memory first (close browsers / other model servers);\n"
                  "   2. make the arena smaller (a smaller pack, or --resident-budget-gib);\n"
                  "   3. start anyway, accepting the risk: STRATA_START_HEADROOM_GIB=0 "
                  "(any value; 0 disables the margin, so only the arena itself is checked).",
                  (double) available / 1073741824.0, (double) needed / 1073741824.0,
                  (double) headroom / 1073741824.0, (double) g.shortfall / 1073741824.0);
    g.message = buf;
    return g;
}

}  // namespace strata::core
