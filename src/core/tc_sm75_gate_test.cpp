// src/core/tc_sm75_gate_test.cpp - the contract behind every sm_75 (RTX 20 / Turing) optimization in this tree:
//
//   THE TURING PATHS ARE FOR COMPUTE CAPABILITY 7.5 AND FOR NOTHING ELSE.
//
// A regression here would silently move a different GPU generation onto kernels written for Turing (or take the
// 2080 Ti off them), which is exactly the kind of change that is invisible in a benchmark on one machine.  The
// test is host-only and needs no GPU: it checks the predicate every kernel and every buffer decision asks.
//
//   cc 7.5            -> on by default, off with STRATA_TC_SM75=0
//   cc 7.0 / 7.2      -> OFF (Volta is not Turing, even though its major is 7)
//   cc 8.0 / 8.6 / 8.9 / 9.0 / 12.0 -> OFF, and STRATA_TC_SM75=1 CANNOT turn them on
#include "strata/core/tc_sm75.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;

void expect(bool got, bool want, const char* what) {
    if (got != want) {
        std::printf("FAIL %s: got %d want %d\n", what, (int) got, (int) want);
        ++failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

void set_env(const char* v) {
#if defined(_WIN32)
    if (v == nullptr) _putenv_s("STRATA_TC_SM75", "");
    else _putenv_s("STRATA_TC_SM75", v);
#else
    if (v == nullptr) unsetenv("STRATA_TC_SM75");
    else setenv("STRATA_TC_SM75", v, 1);
#endif
}

}  // namespace

int main() {
    using strata::core::tc_sm75_enabled;

    // ---- default (auto): 7.5 only ----
    set_env(nullptr);
    expect(tc_sm75_enabled(7, 5), true, "default: cc 7.5 (the 2080 Ti) is ON");
    expect(tc_sm75_enabled(7, 0), false, "default: cc 7.0 (Volta) is OFF");
    expect(tc_sm75_enabled(7, 2), false, "default: cc 7.2 (Xavier) is OFF");
    expect(tc_sm75_enabled(8, 0), false, "default: cc 8.0 is OFF");
    expect(tc_sm75_enabled(8, 6), false, "default: cc 8.6 is OFF");
    expect(tc_sm75_enabled(8, 9), false, "default: cc 8.9 is OFF");
    expect(tc_sm75_enabled(9, 0), false, "default: cc 9.0 is OFF");
    expect(tc_sm75_enabled(10, 0), false, "default: cc 10.0 is OFF");
    expect(tc_sm75_enabled(12, 0), false, "default: cc 12.0 is OFF");
    expect(tc_sm75_enabled(6, 1), false, "default: cc 6.1 is OFF");
    expect(tc_sm75_enabled(5, 0), false, "default: cc 5.0 is OFF");

    // ---- STRATA_TC_SM75=0: the pre-patch behaviour, everywhere ----
    set_env("0");
    expect(tc_sm75_enabled(7, 5), false, "=0: cc 7.5 is OFF (the A/B and bisect switch)");
    expect(tc_sm75_enabled(8, 0), false, "=0: cc 8.0 is OFF");

    // ---- STRATA_TC_SM75=1: it can NOT move another generation onto the Turing paths ----
    set_env("1");
    expect(tc_sm75_enabled(7, 5), true, "=1: cc 7.5 is ON");
    expect(tc_sm75_enabled(7, 0), false, "=1: cc 7.0 stays OFF");
    expect(tc_sm75_enabled(8, 0), false, "=1: cc 8.0 stays OFF - the environment cannot reach it");
    expect(tc_sm75_enabled(8, 6), false, "=1: cc 8.6 stays OFF");
    expect(tc_sm75_enabled(8, 9), false, "=1: cc 8.9 stays OFF");
    expect(tc_sm75_enabled(9, 0), false, "=1: cc 9.0 stays OFF");
    expect(tc_sm75_enabled(12, 0), false, "=1: cc 12.0 stays OFF");

    set_env(nullptr);
    std::printf(failures == 0 ? "tc_sm75_gate_test: PASS\n" : "tc_sm75_gate_test: %d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
