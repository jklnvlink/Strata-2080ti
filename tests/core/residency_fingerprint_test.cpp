// tests/core/residency_fingerprint_test.cpp - the residency fingerprint, CPU only.
//
// WHY IT TAKES A CALLABLE.  Which experts sit in the VRAM cache decides, per expert, whether the
// GPU's quantized kernel or the CPU path computes it - and the two round differently, so two runs
// that chose different resident sets can reply differently.  A statement of the resident set is
// therefore part of reproducibility, not a nicety.  `ExpertCache::open` allocates DEVICE memory, so
// a signature that needed one could only be tested on a GPU; the callable makes it testable here.
#include "strata/core/residency_fingerprint.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>

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

constexpr int32_t kNotRes = -1;

int main() {
    // 1. An empty geometry fingerprints to the offset basis and reports nothing resident.
    {
        const auto r = core::residency_fingerprint(0, 0, [](int64_t, int64_t) { return kNotRes; });
        CHECK(r.fp == core::kFnvOffset, "empty fp is %016llx, want the offset basis",
              (unsigned long long) r.fp);
        CHECK(r.resident == 0, "empty resident is %lld, want 0", (long long) r.resident);
    }

    // 2. The same mapping fingerprints the same; a changed slot changes it.
    {
        std::map<std::pair<int64_t, int64_t>, int32_t> m = {{{0, 0}, 0}, {{0, 1}, 1}, {{1, 0}, 2}};
        auto slot_of = [&](int64_t l, int64_t e) {
            const auto it = m.find({l, e});
            return it == m.end() ? kNotRes : it->second;
        };
        const auto a = core::residency_fingerprint(2, 2, slot_of);
        const auto b = core::residency_fingerprint(2, 2, slot_of);
        CHECK(a.fp == b.fp, "the same mapping must fingerprint the same");
        CHECK(a.resident == 3, "resident is %lld, want 3", (long long) a.resident);

        m[{1, 1}] = 3;
        const auto c = core::residency_fingerprint(2, 2, slot_of);
        CHECK(c.fp != a.fp, "one changed slot must change the fp");
        CHECK(c.resident == 4, "resident is %lld, want 4", (long long) c.resident);
    }

    // 3. Slot IDENTITY matters, not just which pairs are resident: the same pairs in different
    //    slots must fingerprint differently, because a slot's expert is what a kernel reads.
    {
        auto by_pair = [](int64_t l, int64_t e) { return (int32_t) (l * 2 + e); };
        auto by_other = [](int64_t l, int64_t e) { return (int32_t) (7 - (l * 2 + e)); };
        CHECK(core::residency_fingerprint(2, 2, by_pair).fp !=
                  core::residency_fingerprint(2, 2, by_other).fp,
              "different slot assignment must fingerprint differently");
    }

    // 4. The log line is a contract: `<TAG> slots=<n> resident=<n> fp=<16 hex>` after the
    //    caller's own prefix, so a parser can split it on spaces.
    {
        const core::ResidencyFingerprint r{0x0123456789abcdefull, 8435};
        const std::string s = core::residency_fingerprint_line("CUDA0", 8435, r);
        CHECK(s == "RESIDENCY_FP CUDA0 slots=8435 resident=8435 fp=0123456789abcdef",
              "line format is '%s'", s.c_str());
    }

    if (g_failures != 0) {
        std::printf("residency_fingerprint_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("residency_fingerprint_test: ok\n");
    return 0;
}
