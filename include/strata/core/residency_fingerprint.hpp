// include/strata/core/residency_fingerprint.hpp - a statement of WHICH expert sits in WHICH slot.
//
// WHY THIS EXISTS.  The engine says it itself at startup: "the GPU computes the experts in the
// cache; it rounds differently from the CPU, so a reply can differ slightly from a run without the
// cache".  Which experts are resident therefore decides the arithmetic path per expert, and the
// resident set is built from a profile plus whatever VRAM was free at launch.  Reproducibility
// needs that set stated in one comparable number, not promised to be stable.
//
// IT TAKES A CALLABLE, NOT AN ExpertCache, deliberately: `ExpertCache::open` allocates DEVICE
// memory, so a signature that needed one could only be tested on a GPU.  The callable makes the
// whole fingerprint testable on a CPU with a fake mapping.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core {

/// The offset basis and prime `generate.cpp`'s own `fnv1a` uses, repeated so this header is
/// self-contained (a fingerprint only has to be internally consistent, but matching the engine's
/// other hashes keeps the logs comparable by eye).
inline constexpr uint64_t kFnvOffset = 1469598103934665603ull;
inline constexpr uint64_t kFnvPrime = 1099511628211ull;

inline uint64_t fnv1a_mix(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

struct ResidencyFingerprint {
    uint64_t fp = kFnvOffset;
    int64_t resident = 0;  ///< pairs whose slot is >= 0
};

/// `slot_of(layer, expert)` returns the slot index, or a negative value for "not resident"
/// (`ExpertCache::kNotResident`).  Every pair is mixed in a FIXED order, so the result depends on
/// the mapping alone - not on the order the cache happened to admit the pairs.
template <class SlotOf>
ResidencyFingerprint residency_fingerprint(int64_t n_layers, int64_t n_expert, SlotOf slot_of) {
    ResidencyFingerprint r;
    for (int64_t l = 0; l < n_layers; ++l) {
        for (int64_t e = 0; e < n_expert; ++e) {
            const int32_t s = static_cast<int32_t>(slot_of(l, e));
            r.fp = fnv1a_mix(r.fp, &l, sizeof l);
            r.fp = fnv1a_mix(r.fp, &e, sizeof e);
            r.fp = fnv1a_mix(r.fp, &s, sizeof s);
            if (s >= 0) ++r.resident;
        }
    }
    return r;
}

/// `tag` names the stage ("CUDA0"/"CUDA1").  The caller adds the `strata generate: ` prefix, as
/// every other startup line does.
inline std::string residency_fingerprint_line(const char* tag, int64_t slots,
                                              const ResidencyFingerprint& r) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "RESIDENCY_FP %s slots=%lld resident=%lld fp=%016llx", tag,
                  (long long) slots, (long long) r.resident, (unsigned long long) r.fp);
    return std::string(buf);
}

}  // namespace strata::core
