// Bit-exact port of macOS x86_64 libsystem_m sinf/cosf (what the game gets under Rosetta), from its
// disassembly (see sim/tools/libm/README.md). Pure IEEE double arithmetic, so it reproduces the x86
// results exactly on arm64 as long as nothing is fused: build with -ffp-contract=off.
#include "x86math.h"
#include <stdint.h>
#include <string.h>
#include "x86_trig_consts.h"

static inline double d_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static inline uint64_t u_of_d(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static inline uint32_t u_of_f(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// cvttsd2si (32-bit): truncate; out-of-range / NaN -> 0x80000000
static inline int32_t cvtt(double v) {
  if (!(v > -2147483649.0 && v < 2147483648.0)) return INT32_MIN;
  return (int32_t)v;
}

// quadrant q (after the +1 shift for cosf), reduced argument r in [-0.5, 0.5] (units of pi/2)
static inline float poly(double r, int32_t q) {
  double s = r * r;
  double a0 = (d_of(X86_POLY[0]) + s) * s + d_of(X86_POLY[6]);
  double a1 = (d_of(X86_POLY[1]) + s) * s + d_of(X86_POLY[7]);
  double b0 = s * (d_of(X86_POLY[2]) + s) + d_of(X86_POLY[8]);
  double b1 = s * (d_of(X86_POLY[3]) + s) + d_of(X86_POLY[9]);
  double c0 = d_of(X86_POLY[4]) * r, c1 = d_of(X86_POLY[5]);
  double sin_lane = b0 * (a0 * c0), cos_lane = b1 * (a1 * c1);
  float out = (q & 1) ? (float)cos_lane : (float)sin_lane;
  return f_of(u_of_f(out) ^ (((uint32_t)q << 30) & 0x80000000u));
}

static inline float trig(float x, int cos_shift) {
  uint32_t abs = u_of_f(x) & 0x7fffffffu;
  uint32_t v = abs - 0x800000u, w = v - 0x40800000u;
  double xd = (double)x, r;
  int32_t q;
  if (v <= 0x40800000u) {  // normal, |x| <= 8
    double t = xd * d_of(X86_TWO_OVER_PI);
    double k = __builtin_roundeven(t);
    r = t - k;
    q = cvtt(k);
  } else if ((int32_t)v < (int32_t)0x40800000) {  // zero / subnormal
    uint64_t base = cos_shift ? u_of_d(1.0) : u_of_d(xd);
    return (float)d_of(base - (w != 0xbf000000u));
  } else if (w < 0x3e800000u) {  // 8 < |x| < inf: per-exponent reduction table
    const uint64_t *e = X86_REDUCE[(w >> 20) >> 4];
    double hi = d_of(e[0]) * xd, lo = xd * d_of(e[1]);
    double k = __builtin_roundeven(hi);
    r = lo + (hi - k);
    q = cvtt(k);
  } else {
    return x - x;  // inf / nan
  }
  return poly(r, q + cos_shift);
}

float x86_sinf(float x) { return trig(x, 0); }
float x86_cosf(float x) { return trig(x, 1); }
