#include "match.h"

#include <cstring>
#include <new>
#include "b2dump.h"

namespace sim {

thread_local EngineContext *g_engine;

namespace {
constexpr float kSyncDt = 1.0f / 60 + 0.0002f;  // bridge g_sync_dt: the dt _UpdatePad sees
constexpr int kGamepadAxes = 0x1a4;              // GamepadSDL float[4] LX LY RX RY (bridge GP_AXES)

struct EngineScope {  // run game code against this match's engine state
  EngineContext *prev;
  explicit EngineScope(EngineContext *e) : prev(g_engine) { g_engine = e; }
  ~EngineScope() { g_engine = prev; }
};
}  // namespace

Match::Match(const EngineContext &initial) : engine(initial) {
  EngineScope scope(&engine);
  ps = new PoleRiders::PlayState(false);
  ps->LoadComplete();
}

Match *Match::CreateIn(Arena *a, const EngineContext &initial) {
  EngineContext e = initial;
  e.arena = a;
  void *mem = a->Alloc(sizeof(Match));
  return new (mem) Match(e);
}

Match::~Match() {
  EngineScope scope(&engine);
  delete ps;
}

void Match::Tick(const PadAct act[2]) {
  EngineScope scope(&engine);
  for (int i = 0; i < 2; i++) {  // bridge sh_tick
    uint8_t *gp = engine.gamepad[i];
    float *ax = reinterpret_cast<float *>(gp + kGamepadAxes);
    ax[0] = act[i].lx, ax[1] = 0, ax[2] = act[i].rx, ax[3] = act[i].ry;
    atrophy::Gamepad_setJoystickPosition(gp, 0, ax[0], -ax[1]);
    atrophy::Gamepad_setJoystickPosition(gp, 1, ax[2], -ax[3]);
  }
  for (int i = 0; i < 2; i++) Common::Input::_UpdatePad(i, kSyncDt);
  engine.cfg.level5unlocked = true;
  ps->Update();
  frame++;
}

std::string Match::Dump() const {
  std::vector<DumpObject> objs{{"playstate", ps, sizeof(PoleRiders::PlayState)}};
  for (int i = 0; i < 4; i++) {
    PoleRiders::Player *pl = ps->players[i];
    if (!pl) continue;
    std::string n = std::to_string(i);
    objs.push_back({"player" + n, pl, sizeof(PoleRiders::Player)});
    objs.push_back({"stats" + n, pl->stats, sizeof(PoleRiders::Stats)});
    objs.push_back({"pole" + n, pl->pole, sizeof(PoleRiders::Pole)});
  }
  objs.push_back({"ball", ps->ball, sizeof(PoleRiders::Ball)});
  for (int i = 0; i < 2; i++) {
    std::string n = std::to_string(i);
    objs.push_back({"gamepad" + n, engine.gamepad[i], sizeof engine.gamepad[i]});
    objs.push_back({"pad" + n, engine.pads[i], sizeof engine.pads[i]});
  }
  return b2dump_world(ps->world, frame, objs);
}

}  // namespace sim
