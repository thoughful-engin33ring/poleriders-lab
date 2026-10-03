// One native Pole Riders match, driven exactly like a bridge shadow (bridge/farm.c sh_create/sh_tick):
//   construct = PlayState(false) + LoadComplete();
//   tick      = per pad: gamepad axes = (lx, 0, rx, ry), setJoystickPosition(0, lx, -0),
//               setJoystickPosition(1, rx, -ry); then _UpdatePad(pad, 1/60 + 0.0002) for pads 0, 1;
//               level5unlocked = 1; PlayState::Update().
#pragma once
#include <string>
#include "arena.h"
#include "game/engine.h"
#include "game/layout.h"

namespace sim {

struct PadAct {
  float lx, rx, ry;  // SDL stick convention (+y down), as in `farm act`
};

class Match {
 public:
  // Initial engine input state (gamepad/pad bytes, session records) comes from a real-game dump
  // (objects gamepad0/1, pad0/1 and player+0x220) until package D can construct it.
  explicit Match(const EngineContext &initial);
  ~Match();
  // A match living entirely in arena `a` (the Match object itself and everything the game allocates for it):
  // snapshot / restore = Arena::Save / Restore. Destroy it with a->Clear() / Arena::Destroy (no destructor).
  static Match *CreateIn(Arena *a, const EngineContext &initial);
  // Run game code for this match from outside Tick (env curricula, test events): sets g_engine, which also
  // routes the game's allocations into the match's arena.
  struct Scope {
    EngineContext *prev;
    explicit Scope(Match *m) : prev(g_engine) { g_engine = &m->engine; }
    ~Scope() { g_engine = prev; }
  };
  void Tick(const PadAct act[2]);
  std::string Dump() const;  // oracle format (bridge `farm dump`)

  EngineContext engine;
  PoleRiders::PlayState *ps = nullptr;
  unsigned long frame = 0;
};

}  // namespace sim
