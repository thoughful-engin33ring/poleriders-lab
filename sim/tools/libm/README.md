# x86 libm sinf/cosf, bit-exact on arm64

The game runs under Rosetta, so Box2D's `sinf/cosf` (b2Mat22::Set, every body every step) are macOS
**x86_64** libsystem_m. arm64 libm returns different bits for ~45M of the 2^32 float inputs, which
would make the native sim diverge. `sim/src/x86math.c` is a port of the x86 routine (a short
double-precision polynomial + range-reduction table), exhaustively verified over all 2^32 inputs.

- `dump_x86_libm.c` — x86_64 program that writes the libm sinf code and constants (`code.bin`, `data.bin`).
- `gen_consts.py data.bin ../../src/x86_trig_consts.h` — constants header.
- `fullcheck.c` — hash sinf/cosf over all 2^32 inputs per 2^16 block; build it x86_64 (system libm) and
  arm64 `-DPORT` with `../../src/x86math.c`, diff the outputs (0 differing blocks, 2026-09-30).
- `probe.c` — the original probe comparing system libms (x86 vs arm64); note a compiler fuses
  `sinf(a); cosf(a)` into `__sincosf_stret`, which has different results — call through volatile
  pointers.
