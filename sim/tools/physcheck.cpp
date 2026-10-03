// Pure-physics check of the native Box2D build against the game: rebuild a world from a real-game
// dump, dump it back (tick 0), step it n times with bare b2World::Step(1/60, 60, 18) — what
// `farm physstep` does in the game — and dump again.
//   physcheck world.load N out0.json outN.json
#include <cstdio>
#include <cstdlib>
#include "../src/b2dump.h"

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: physcheck world.load N out0.json outN.json\n");
    return 2;
  }
  b2World *w = b2load_world(argv[1]);
  if (!w) return 1;
  int n = atoi(argv[2]);
  FILE *f = fopen(argv[3], "w");
  fputs(b2dump_world(w, 0, {}).c_str(), f);
  fclose(f);
  for (int t = 0; t < n; t++) w->Step(1.0f / 60.0f, 60, 18);
  f = fopen(argv[4], "w");
  fputs(b2dump_world(w, n, {}).c_str(), f);
  fclose(f);
  return 0;
}
