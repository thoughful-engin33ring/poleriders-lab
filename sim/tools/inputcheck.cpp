// Package D check: run only the engine input chain (src/game/input.cpp) over a stick script, exactly as
// sim::Match::Tick / the bridge's sh_tick drive it, and print the state after every tick.
//   inputcheck engine.bin script.txt [rumble.txt] > out.txt
//   inputcheck --default engine.bin      (does sim::DefaultInputState reproduce its gamepad/pad bytes?)
// engine.bin: as for gamecheck (gamepad0, gamepad1, pad0, pad1, session0, session1).
// script.txt: "lx0 rx0 ry0 lx1 rx1 ry1" per tick (strtof + clamp, like the bridge).
// rumble.txt (optional): per tick "a0 b0 a1 b1" hex float bits: Pad+0xf4/+0xf8 as Player::Update's
//   VirtualUpdateVibration left them (taken from the real trace; not an input-chain output).
// Output, one line per tick incl. tick 0: hex of gamepad0 gamepad1 pad0 pad1, then the float bits of
//   VirtualRightStick / VirtualLeftStick (4 floats each) for session 0 and 1 (what Player stores at
//   +0x19c / +0x1ac when active).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../src/game/engine.h"

namespace sim {
thread_local EngineContext *g_engine;
}

static void hex(const void *p, size_t n) {
  const uint8_t *b = static_cast<const uint8_t *>(p);
  for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
  putchar(' ');
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: inputcheck engine.bin script.txt [rumble.txt]\n");
    return 2;
  }
  static sim::EngineContext e{};
  const bool deflt = strcmp(argv[1], "--default") == 0;
  FILE *f = fopen(argv[deflt ? 2 : 1], "rb");
  if (!f || fread(e.gamepad[0], 0x1b8, 1, f) != 1 || fread(e.gamepad[1], 0x1b8, 1, f) != 1 ||
      fread(e.pads[0], 0x2c8, 1, f) != 1 || fread(e.pads[1], 0x2c8, 1, f) != 1 ||
      fread(&e.session[0], 0x1c, 1, f) != 1 || fread(&e.session[1], 0x1c, 1, f) != 1) {
    fprintf(stderr, "bad engine.bin\n");
    return 1;
  }
  fclose(f);
  if (deflt) {
    static sim::EngineContext d{};
    sim::DefaultInputState(d);
    const bool same = memcmp(d.gamepad[0], e.gamepad[0], 2 * 0x1b8) == 0 && memcmp(d.pads[0], e.pads[0], 2 * 0x2c8) == 0;
    printf("DefaultInputState %s the real tick-0 input state\n", same ? "matches" : "DIFFERS from");
    return same ? 0 : 1;
  }
  sim::g_engine = &e;
  FILE *in = fopen(argv[2], "r");
  FILE *rum = argc > 3 ? fopen(argv[3], "r") : nullptr;
  auto emit = [&]() {
    for (int i = 0; i < 2; i++) hex(e.gamepad[i], 0x1b8);
    for (int i = 0; i < 2; i++) hex(e.pads[i], 0x2c8);
    for (int i = 0; i < 2; i++) {
      Common::Input::Stick r = Common::Input::VirtualRightStick(e.session[i]);
      Common::Input::Stick l = Common::Input::VirtualLeftStick(e.session[i]);
      hex(&r, 16);
      hex(&l, 16);
    }
    putchar('\n');
  };
  emit();
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
    for (int i = 0; i < 2; i++) {  // = sim::Match::Tick
      uint8_t *gp = e.gamepad[i];
      float *ax = reinterpret_cast<float *>(gp + 0x1a4);
      ax[0] = v[3 * i], ax[1] = 0, ax[2] = v[3 * i + 1], ax[3] = v[3 * i + 2];
      atrophy::Gamepad_setJoystickPosition(gp, 0, ax[0], -ax[1]);
      atrophy::Gamepad_setJoystickPosition(gp, 1, ax[2], -ax[3]);
    }
    for (int i = 0; i < 2; i++) Common::Input::_UpdatePad(i, 1.0f / 60 + 0.0002f);
    if (rum) {
      unsigned u[4];
      if (fscanf(rum, "%x %x %x %x", &u[0], &u[1], &u[2], &u[3]) == 4)
        for (int i = 0; i < 2; i++) {
          memcpy(&e.pad(i)->rumble[0], &u[2 * i], 4);
          memcpy(&e.pad(i)->rumble[1], &u[2 * i + 1], 4);
        }
    }
    emit();
  }
  return 0;
}
