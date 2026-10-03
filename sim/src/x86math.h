// x86_sinf / x86_cosf: sinf/cosf with the exact results of macOS x86_64 libm (what the game runs
// under Rosetta). arm64 libm differs on ~45M float inputs, so this is a port of the x86 algorithm.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
float x86_sinf(float a);
float x86_cosf(float a);
#ifdef __cplusplus
}
#endif
