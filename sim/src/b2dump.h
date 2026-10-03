// Raw dumps of a Box2D world in the oracle format of bridge/farm.c `farm dump` (hex of every
// body/fixture/shape, joint and contact in list order), so native and real-game states can be
// diffed byte for byte (sim/tools/diff_dump.py). Also the reverse: rebuild a world from a dump.
#pragma once
#include <Box2D/Box2D.h>
#include <string>
#include <vector>

struct DumpObject {  // extra raw memory to include under "objects" (game objects)
  std::string name;
  const void *ptr;
  size_t len;
};

std::string b2dump_world(b2World *w, unsigned long frame, const std::vector<DumpObject> &objects);

// Rebuild a world from the loader text format written by tools/dump2load.py (bodies, fixtures in
// proxy-id order, joints — reproducing list orders and proxy ids). Returns nullptr on error.
b2World *b2load_world(const char *path);
