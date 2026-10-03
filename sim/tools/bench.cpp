// Native tick throughput: one match per thread, random held sticks, no dumps.
//   bench engine.bin [ticks=20000] [threads=1]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "../src/match.h"

static sim::EngineContext load(const char *path) {
  sim::EngineContext e{};
  FILE *f = fopen(path, "rb");
  if (!f) exit(1);
  fread(e.gamepad[0], 0x1b8, 1, f), fread(e.gamepad[1], 0x1b8, 1, f);
  fread(e.pads[0], 0x2c8, 1, f), fread(e.pads[1], 0x2c8, 1, f);
  fread(&e.session[0], 0x1c, 1, f), fread(&e.session[1], 0x1c, 1, f);
  int lv;
  if (fread(&lv, 4, 1, f) == 1) e.cfg.level = lv;
  fclose(f);
  return e;
}

int main(int argc, char **argv) {
  sim::EngineContext e = load(argv[1]);
  long ticks = argc > 2 ? atol(argv[2]) : 20000;
  int threads = argc > 3 ? atoi(argv[3]) : 1;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> th;
  for (int k = 0; k < threads; k++)
    th.emplace_back([&, k] {
      unsigned s = 12345 + k;
      auto rnd = [&] { s = s * 1103515245 + 12345; return ((s >> 8) & 0xffff) / 32768.0f - 1.0f; };
      sim::Match *m = new sim::Match(e);
      sim::PadAct a[2] = {};
      for (long t = 0; t < ticks; t++) {
        if (t % 12 == 0) a[0] = {rnd(), rnd(), rnd()}, a[1] = {rnd(), rnd(), rnd()};
        if (m->ps->gameOver) { delete m; m = new sim::Match(e); }
        m->Tick(a);
      }
      delete m;
    });
  for (auto &t : th) t.join();
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("%d threads x %ld ticks: %.2fs  -> %.1f us/tick/thread, %.0f ticks/s total\n", threads, ticks, s,
         s * 1e6 / ticks, threads * ticks / s);
}
