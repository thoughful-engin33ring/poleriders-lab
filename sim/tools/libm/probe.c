// Probe libm sinf/cosf over every float bit pattern in [lo, hi) (as uint32 ranges).
//   probe hash  -> "block hs hc" per 2^16 block      (run on both archs, diff)
//   probe dump blocks.txt out.bin -> (a, sinf, cosf) for every float of the listed blocks
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
// called through volatile pointers so the compiler cannot fuse them into __sincosf_stret (the game
// calls sinf and cosf separately; the fused routine gives different results)
static float (*volatile SIN)(float) = sinf, (*volatile COS)(float) = cosf;
static const uint32_t RANGES[][2] = {{0x00000000u, 0x49800000u}, {0x80000000u, 0xc9800000u}};  // |a| < 2^20
int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "hash")) {
    for (int r = 0; r < 2; r++)
      for (uint32_t b = RANGES[r][0] >> 16; b < RANGES[r][1] >> 16; b++) {
        uint64_t hs = 1469598103934665603ULL, hc = hs;
        for (uint32_t lo = 0; lo < 65536; lo++) {
          uint32_t u = (b << 16) | lo, us, uc;
          float a, s, c;
          memcpy(&a, &u, 4);
          s = SIN(a), c = COS(a);
          memcpy(&us, &s, 4), memcpy(&uc, &c, 4);
          hs = (hs ^ us) * 1099511628211ULL, hc = (hc ^ uc) * 1099511628211ULL;
        }
        printf("%u %llx %llx\n", b, (unsigned long long)hs, (unsigned long long)hc);
      }
  } else if (argc > 3 && !strcmp(argv[1], "dump")) {
    FILE *f = fopen(argv[2], "r"), *o = fopen(argv[3], "wb");
    unsigned b;
    while (fscanf(f, "%u", &b) == 1)
      for (uint32_t lo = 0; lo < 65536; lo++) {
        uint32_t u = (b << 16) | lo;
        float r[3];
        memcpy(&r[0], &u, 4);
        r[1] = SIN(r[0]), r[2] = COS(r[0]);
        fwrite(r, 4, 3, o);
      }
    fclose(o);
  }
  return 0;
}
