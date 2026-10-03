// Hash sinf/cosf over ALL 2^32 float bit patterns, per 2^16 block. Built twice:
//   x86_64 with the system libm (reference, run under Rosetta) and arm64 with -DPORT (x86math.c).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifdef PORT
#include "../../src/x86math.h"
static float (*volatile SIN)(float) = x86_sinf, (*volatile COS)(float) = x86_cosf;
#else
static float (*volatile SIN)(float) = sinf, (*volatile COS)(float) = cosf;
#endif
int main(void) {
  for (uint32_t b = 0; b < 65536; b++) {
    uint64_t hs = 1469598103934665603ULL, hc = hs;
    for (uint32_t lo = 0; lo < 65536; lo++) {
      uint32_t u = (b << 16) | lo, us, uc;
      float a, s, c;
      memcpy(&a, &u, 4);
      s = SIN(a), c = COS(a);
      memcpy(&us, &s, 4), memcpy(&uc, &c, 4);
      if (s != s) us = 0x7fc00000u;  // NaN payloads don't matter
      if (c != c) uc = 0x7fc00000u;
      hs = (hs ^ us) * 1099511628211ULL, hc = (hc ^ uc) * 1099511628211ULL;
    }
    printf("%u %llx %llx\n", b, (unsigned long long)hs, (unsigned long long)hc);
  }
  return 0;
}
