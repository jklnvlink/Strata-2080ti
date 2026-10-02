// tests/core/memory_gate_test.cpp - the startup RAM gate's decision, CPU only.
//
// WHY A PURE FUNCTION.  The gate decides whether the engine may start at all: it holds the whole
// expert arena (46.84 GiB here) resident, so starting into a machine that cannot also hold the
// desktop is how the "dsh-web / rustdesk freeze" failure begins.  A judgement that can only be
// exercised by starting a 47 GiB engine is a judgement nobody tests - so the arithmetic and the
// wording live here, with no machine state, and `generate.cpp` only feeds it numbers.
#include "strata/core/memory_gate.hpp"

#include <cstdio>
#include <string>

namespace core = strata::core;

static int g_failures = 0;

#define CHECK(cond, ...)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);     \
            std::printf(__VA_ARGS__);                           \
            std::printf("\n");                                  \
            ++g_failures;                                       \
        }                                                       \
    } while (0)

constexpr uint64_t GIB = 1ull << 30;

int main() {
    // 1. Plenty of room: ok, and no message (an ok gate must not carry advice).
    {
        const auto g = core::memory_gate(57 * GIB, 47 * GIB, 6 * GIB);
        CHECK(g.ok, "57 GiB available must pass a 47+6 GiB requirement");
        CHECK(g.shortfall == 0, "an ok gate has no shortfall, got %llu",
              (unsigned long long) g.shortfall);
        CHECK(g.message.empty(), "an ok gate must not carry a message: '%s'", g.message.c_str());
    }

    // 2. Exactly at the requirement passes: the headroom is the margin the desktop gets, so
    //    available == needed + headroom is the last admissible state, not a failure.
    {
        const auto g = core::memory_gate(53 * GIB, 47 * GIB, 6 * GIB);
        CHECK(g.ok, "available == needed + headroom must pass (the headroom is still intact)");
    }

    // 3. One byte short fails, and the shortfall names how much.
    {
        const auto g = core::memory_gate(53 * GIB - 1, 47 * GIB, 6 * GIB);
        CHECK(!g.ok, "one byte short must fail");
        CHECK(g.shortfall == 1, "shortfall is %llu, want 1", (unsigned long long) g.shortfall);
    }

    // 4. The message must name all three numbers the operator needs to act, plus a way out.
    //    This is the contract: a refusal that does not say what to do is a bug report, not a gate.
    {
        const auto g = core::memory_gate(40 * GIB, 47 * GIB, 6 * GIB);
        CHECK(!g.ok, "40 GiB must refuse a 47+6 GiB requirement");
        CHECK(g.shortfall == 13 * GIB, "shortfall is %llu GiB, want 13",
              (unsigned long long) (g.shortfall / GIB));
        for (const char* needle : {"40.0 GiB", "47.0 GiB", "6.0 GiB", "13.0 GiB",
                                   "STRATA_START_HEADROOM_GIB"}) {
            CHECK(g.message.find(needle) != std::string::npos,
                  "the refusal must mention '%s'; it says:\n%s", needle, g.message.c_str());
        }
    }

    // 5. A zero headroom disables the gate's margin: it then only refuses when the arena itself
    //    does not fit - that is the documented way to opt out (CLAUDE.md's "can turn it off").
    {
        const auto g = core::memory_gate(50 * GIB, 47 * GIB, 0);
        CHECK(g.ok, "with headroom 0, 50 GiB must pass a 47 GiB requirement");
        const auto f = core::memory_gate(46 * GIB, 47 * GIB, 0);
        CHECK(!f.ok, "with headroom 0, 46 GiB must still refuse a 47 GiB requirement");
        CHECK(f.headroom == 0, "headroom must be reported as 0");
    }

    // 6. needed + headroom overflow must not wrap into a passing gate.  A gate that says "ok"
    //    because the arithmetic wrapped is worse than no gate.
    {
        const auto g = core::memory_gate(1 * GIB, UINT64_MAX - 1024, 4096);
        CHECK(!g.ok, "an overflowing requirement must refuse, not wrap");
    }

    if (g_failures != 0) {
        std::printf("memory_gate_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("memory_gate_test: ok\n");
    return 0;
}
