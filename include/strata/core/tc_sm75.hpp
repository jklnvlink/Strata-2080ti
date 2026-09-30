// include/strata/core/tc_sm75.hpp - sm_75 (RTX 20 / Turing) tensor-core paths.
//
// Turing has no tf32 MMA and no cp.async, so three of this engine's prompt-path kernels are compiled out below
// sm_80 and fall back to slower paths (see `qsa_prompt_attn.cu`, `qsa_select.cu`).  It does have fp16 MMA
// (m16n8k16), a real tensor core, and cuBLAS has no bf16 GEMM kernel below sm_80 - it runs an FFMA one, which
// measured 19.5% of the prompt's GPU time on a 2x2080 Ti.  These switches select the Turing paths.
//
// The default is `auto`: cc 7.5 uses them, cc >= 8 keeps its own (better) paths untouched, and every new path
// keeps the old one compiled in beside it, so `STRATA_TC_SM75=0` restores the pre-patch behaviour exactly.
#pragma once

#include <cstdlib>

namespace strata::core {

inline bool tc_sm75_enabled(int cc_major) {
    const char* e = std::getenv("STRATA_TC_SM75");
    if (e != nullptr && e[0] == '0') return false;
    if (e != nullptr && e[0] == '1') return true;
    return cc_major == 7;
}

}  // namespace strata::core
