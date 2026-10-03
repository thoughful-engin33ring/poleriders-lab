// Python's `random.Random(seed).random()` (MT19937, init_by_array seeding), bit-exact, so the native
// env makes the same sudden-death reset draws as bridge/farm_env.py FarmHost (rng = random.Random(i)).
#pragma once
#include <cstdint>
#include <vector>

namespace sim {
namespace env {

class PyRandom {
 public:
  explicit PyRandom(uint64_t seed = 0) { Seed(seed); }

  // random.seed(int): key = 32-bit little-endian words of abs(seed) ([0] for 0) -> init_by_array
  void Seed(uint64_t seed) {
    std::vector<uint32_t> key;
    do {
      key.push_back(static_cast<uint32_t>(seed));
      seed >>= 32;
    } while (seed);
    InitByArray(key.data(), static_cast<int>(key.size()));
  }

  // random.random(): 53-bit float in [0, 1)
  double Random() {
    uint32_t a = Next() >> 5, b = Next() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  }

 private:
  enum { N = 624, M = 397 };
  uint32_t mt_[N];
  int mti_ = N + 1;

  void InitGenrand(uint32_t s) {
    mt_[0] = s;
    for (mti_ = 1; mti_ < N; mti_++)
      mt_[mti_] = 1812433253u * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) + static_cast<uint32_t>(mti_);
  }

  void InitByArray(const uint32_t *key, int len) {
    InitGenrand(19650218u);
    int i = 1, j = 0;
    for (int k = N > len ? N : len; k; k--) {
      mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) + key[j] + static_cast<uint32_t>(j);
      i++, j++;
      if (i >= N) mt_[0] = mt_[N - 1], i = 1;
      if (j >= len) j = 0;
    }
    for (int k = N - 1; k; k--) {
      mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) - static_cast<uint32_t>(i);
      i++;
      if (i >= N) mt_[0] = mt_[N - 1], i = 1;
    }
    mt_[0] = 0x80000000u;
    mti_ = N;
  }

  uint32_t Next() {
    static const uint32_t mag01[2] = {0u, 0x9908b0dfu};
    uint32_t y;
    if (mti_ >= N) {
      int kk;
      for (kk = 0; kk < N - M; kk++) {
        y = (mt_[kk] & 0x80000000u) | (mt_[kk + 1] & 0x7fffffffu);
        mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 1u];
      }
      for (; kk < N - 1; kk++) {
        y = (mt_[kk] & 0x80000000u) | (mt_[kk + 1] & 0x7fffffffu);
        mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 1u];
      }
      y = (mt_[N - 1] & 0x80000000u) | (mt_[0] & 0x7fffffffu);
      mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 1u];
      mti_ = 0;
    }
    y = mt_[mti_++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
  }
};

}  // namespace env
}  // namespace sim
