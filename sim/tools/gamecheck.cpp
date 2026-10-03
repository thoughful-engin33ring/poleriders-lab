// Run a native match with a scripted input sequence and dump every tick (JSON lines, oracle format).
//   gamecheck engine.bin script.txt out.jsonl
// engine.bin (tools/trace_compare.py): the real shadow's initial input state — gamepad0, gamepad1
// (0x1b8 each), pad0, pad1 (0x2c8 each), session0, session1 (0x1c each, = Player+0x220), and
// optionally an int32 level id (1 Kalgoorlie, 2 Flatirons; default MatchConfig::level).
// script.txt: one line per tick, "lx0 rx0 ry0 lx1 rx1 ry1" (parsed with strtof, like the bridge).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../src/match.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: gamecheck engine.bin script.txt out.jsonl\n");
    return 2;
  }
  sim::EngineContext e{};
  FILE *f = fopen(argv[1], "rb");
  if (!f || fread(e.gamepad[0], 0x1b8, 1, f) != 1 || fread(e.gamepad[1], 0x1b8, 1, f) != 1 ||
      fread(e.pads[0], 0x2c8, 1, f) != 1 || fread(e.pads[1], 0x2c8, 1, f) != 1 ||
      fread(&e.session[0], 0x1c, 1, f) != 1 || fread(&e.session[1], 0x1c, 1, f) != 1) {
    fprintf(stderr, "bad engine.bin\n");
    return 1;
  }
  int32_t level;  // optional 7th record: the level id (PlayState+0x214) the oracle ran
  if (fread(&level, 4, 1, f) == 1) e.cfg.level = level;
  fclose(f);
  FILE *in = fopen(argv[2], "r"), *out = fopen(argv[3], "w");
  sim::Match m(e);
  fputs(m.Dump().c_str(), out);
  fputc('\n', out);
  char line[512];
  while (fgets(line, sizeof line, in)) {
    float v[6];
    char *p = line, *q;
    int n = 0;
    for (; n < 6; n++, p = q) {
      v[n] = strtof(p, &q);
      if (q == p) break;
      v[n] = v[n] < -1 ? -1 : v[n] > 1 ? 1 : v[n];  // bridge clampf
    }
    if (n < 6) continue;
    sim::PadAct act[2] = {{v[0], v[1], v[2]}, {v[3], v[4], v[5]}};
    m.Tick(act);
    fputs(m.Dump().c_str(), out);
    fputc('\n', out);
  }
  fclose(out);
  return 0;
}
