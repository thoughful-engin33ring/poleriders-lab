// Engine-side state the Pole Riders code reads — globals in the real game, per match here (so matches
// can run on parallel threads). Game code reaches it through `sim::g_engine` (thread_local, set by
// sim::Match before it runs any game code).
//
// Input chain (package D, src/game/input.cpp), per tick as the bridge shadow does it (farm.c sh_tick):
//   GamepadSDL.axes = (lx, 0, rx, ry); Gamepad_setJoystickPosition(gp, 0, lx, -0), (gp, 1, rx, -ry)
//   -> deadzone/clamp into GamepadSDL.sticks[s].x/y;  _UpdatePad(pad, 1/60+0.0002) for pads 0, 1
//   -> Pad.leftStick/rightStick + button/flick flags;  Player reads VirtualLeftStick/VirtualRightStick.
#pragma once
#include <cstddef>
#include <cstdint>
#include "layout.h"

// ---- atrophy::GamepadSDL (0x1b8; atrophy::sGamepadList[4] @0x10064c980) ----
// Only the fields the input chain touches are named. Byte layout identical to the game.
namespace atrophy {
struct GamepadStick {           // 0x30, at GamepadSDL+0xc + 0x30*stick
  float x;                      // +0x00 processed value (setJoystickPosition), +x right
  float y;                      // +0x04 processed value, +y UP (the bridge passes -ry)
  float z, w;                   // +0x08/+0x0c copied to Pad (+0x104/+0x108 for stick 0), never written here
  float deadZone;               // +0x10 (0.1 for the bridge's virtual pads)
  int32_t emuButton[4];         // +0x14 buttons emulated by the stick's 4 directions (-1 = none)
  int32_t emuCurrent;           // +0x24 emulated button currently pressed by this stick (-1 = none)
  float emuHalfWidth;           // +0x28 sector half-width (radians) of an emulated direction
  uint8_t emulate;              // +0x2c stick emulates buttons (d-pad); 0 for the bridge's pads
  uint8_t _pad_2d[3];
};
static_assert(sizeof(GamepadStick) == 0x30, "");

struct GamepadButton {          // 8, at GamepadSDL+0xe0 + 8*button
  uint16_t sources;             // +0 bitmask of what holds the button (stick emulation bits 1<<(4*stick+1+k))
  uint8_t value;                // +2 0..255 (setButtonState); _UpdatePad tests != 0
  uint8_t _pad_3[5];
};
static_assert(sizeof(GamepadButton) == 8, "");

struct GamepadSDL {
  int32_t id;                   // 0x000 gamepad index (event payload)
  int32_t type;                 // 0x004 1 = SDL (_UpdatePad button map A: types 1/4/5; map B: 7/8)
  uint8_t connected;            // 0x008 Input_IsGamepadConnectedAt
  uint8_t _pad_009[3];
  GamepadStick sticks[4];       // 0x00c 0 = left, 1 = right, 2/3 = triggers (+0x6c/+0x70/+0x9c)
  int32_t numSticks;            // 0x0cc (2) — _UpdatePad copies triggers only if >= 3
  const float *response;        // 0x0d0 optional stick response table (null / count 0 here)
  int32_t responseCount;        // 0x0d8
  uint8_t _pad_0dc[4];
  GamepadButton buttons[18];    // 0x0e0 (17 used)
  int32_t numButtons;           // 0x170 (17)
  float motor[2];               // 0x174 rumble (setMotorAt)
  int32_t numMotors;            // 0x17c (0)
  int32_t padColor;             // 0x180 = Pad+0x2c4 every _UpdatePad
  uint8_t _pad_184[4];
  void *sdlController;          // 0x188
  int32_t instanceId;           // 0x190
  uint8_t _pad_194[4];
  void *haptic;                 // 0x198 null for the bridge's virtual pads -> setMotorAt is a no-op
  uint8_t _pad_1a0[4];
  float axes[4];                // 0x1a4 raw SDL LX LY RX RY (the bridge writes them)
  uint8_t motorDirty;           // 0x1b4
  uint8_t _pad_1b5[3];
};
static_assert(offsetof(GamepadSDL, type) == 0x004, "");
static_assert(offsetof(GamepadSDL, connected) == 0x008, "");
static_assert(offsetof(GamepadSDL, sticks) == 0x00c, "");
static_assert(offsetof(GamepadSDL, numSticks) == 0x0cc, "");
static_assert(offsetof(GamepadSDL, response) == 0x0d0, "");
static_assert(offsetof(GamepadSDL, responseCount) == 0x0d8, "");
static_assert(offsetof(GamepadSDL, buttons) == 0x0e0, "");
static_assert(offsetof(GamepadSDL, numButtons) == 0x170, "");
static_assert(offsetof(GamepadSDL, motor) == 0x174, "");
static_assert(offsetof(GamepadSDL, numMotors) == 0x17c, "");
static_assert(offsetof(GamepadSDL, padColor) == 0x180, "");
static_assert(offsetof(GamepadSDL, sdlController) == 0x188, "");
static_assert(offsetof(GamepadSDL, instanceId) == 0x190, "");
static_assert(offsetof(GamepadSDL, haptic) == 0x198, "");
static_assert(offsetof(GamepadSDL, axes) == 0x1a4, "");
static_assert(offsetof(GamepadSDL, motorDirty) == 0x1b4, "");
static_assert(sizeof(GamepadSDL) == 0x1b8, "");
}  // namespace atrophy

// ---- Common::Input::_pads[8] entry (0x2c8; @0x100c5aca0) ----
namespace Common {
namespace Input {
enum { kNumButtons = 0x51 };    // Common::Input::Button values 0..0x50
struct Pad {  // flags are bytes (0/1 as written by the game's setne)
  uint8_t held[kNumButtons];    // 0x000 ButtonHeld(); rebuilt by every _UpdatePad
  uint8_t pressed[kNumButtons]; // 0x051 held && !held last frame
  uint8_t forced[kNumButtons];  // 0x0a2 extra "held" sources (keyboard delegate); OR-ed into held
  uint8_t _pad_0f3;
  float rumble[2];              // 0x0f4 (VirtualUpdateVibration; _UpdatePad feeds setMotorAt, then zeroes)
  float leftStick[4];           // 0x0fc = GamepadSDL.sticks[0] x, y, z, w
  float rightStick[4];          // 0x10c = GamepadSDL.sticks[1] x, y, z, w
  float triggers[4];            // 0x11c sticks[2].x, sticks[2].y, sticks[3].x, 0 (0 when numSticks < 3)
  float holdTime[kNumButtons];  // 0x12c seconds held (0 when released)
  uint8_t repeat[kNumButtons];  // 0x270 pressed || auto-repeat at 7 Hz after 0.5 s
  uint8_t _pad_2c1[3];
  int32_t padColor;             // 0x2c4 -> GamepadSDL.padColor
};
static_assert(offsetof(Pad, pressed) == 0x051, "");
static_assert(offsetof(Pad, forced) == 0x0a2, "");
static_assert(offsetof(Pad, rumble) == 0x0f4, "");
static_assert(offsetof(Pad, leftStick) == 0x0fc, "");
static_assert(offsetof(Pad, rightStick) == 0x10c, "");
static_assert(offsetof(Pad, triggers) == 0x11c, "");
static_assert(offsetof(Pad, holdTime) == 0x12c, "");
static_assert(offsetof(Pad, repeat) == 0x270, "");
static_assert(offsetof(Pad, padColor) == 0x2c4, "");
static_assert(sizeof(Pad) == 0x2c8, "");
}  // namespace Input
}  // namespace Common

namespace sim {

struct MatchConfig {
  int level = 1;                // Assets::currentLevel (1 = Kalgoorlie, the training map; 2 = Flatirons)
  int maxScore = 3;             // Assets::maxScore (goals to win)
  float roundTime = 180.0f;     // Assets::roundTime (seconds)
  bool level5unlocked = true;   // Assets::level5unlocked (the bridge forces it to 1 every tick)
  bool shouldRumble = true;     // Assets::shouldRumble (Player::Update -> VirtualUpdateVibration args; the
                                // oracle's pads do receive rumble, so it is 1 there)
};

struct EngineContext {
  alignas(8) uint8_t gamepad[4][0x1b8];  // atrophy::sGamepadList[4] (atrophy::GamepadSDL, see gp())
  alignas(8) uint8_t pads[8][0x2c8];     // Common::Input::_pads[8] (Common::Input::Pad, see pad())
  PoleRiders::SessionPlayer session[4];  // Common::Session player records (as copied into Player+0x220)
  MatchConfig cfg;

  atrophy::GamepadSDL *gp(int i) { return reinterpret_cast<atrophy::GamepadSDL *>(gamepad[i]); }
  Common::Input::Pad *pad(int i) { return reinterpret_cast<Common::Input::Pad *>(pads[i]); }
  // Training-only (not in the game): Player::Respawn builds the pole pointing straight up instead of
  // forward (sim/src/env pit drops). Off for everything that mirrors the real game.
  bool spawnPoleUp = false;
  // training-only (BatchEnv::SetRandomPole): every Respawn builds the pole at a random angle between forward (0)
  // and 150 deg (back-up) in the rider's attack frame, drawn from this per-match splitmix64 state
  bool randomPole = false;
  uint64_t poleRng = 0;
  // sim::Arena (src/arena.h) holding this match: every allocation the game code makes while this context is
  // g_engine goes there (snapshot / restore). nullptr = the plain heap (tools, the env's probe match).
  void *arena = nullptr;
};

extern thread_local EngineContext *g_engine;

// Fills gamepad[0..1] / pads[0..1] with the input state a fresh bridge shadow starts from (the bridge's
// connected virtual SDL pads, idle; = the real tick-0 dump bytes, checked by `inputcheck --default`).
// Session records are NOT touched (they come from the game's menus: heads, teams).
void DefaultInputState(EngineContext &e);  // package D, src/game/input.cpp

}  // namespace sim

// ---- input chain (package D: src/game/input.cpp) ----
namespace Common {
namespace Input {
struct Stick {  // the 16-byte value the stick readers return (Player copies it to +0x19c / +0x1ac)
  union {
    float v[4];
    struct {
      float x;  // +x right
      float y;  // +y UP (SDL y negated by the bridge before setJoystickPosition)
      float z;  // sticks: GamepadSDL stick +0x08 (0 in practice); digital fallbacks: 0
      float w;  // GamepadSDL stick +0x0c (0 in practice)
    };
  };
};
static_assert(sizeof(Stick) == 16, "");
Stick VirtualLeftStick(PoleRiders::SessionPlayer p);   // @0x10005f2c0
Stick VirtualRightStick(PoleRiders::SessionPlayer p);  // @0x10005f290
Stick LeftStick(int controller);                       // @0x10005f1a0
Stick RightStick(int controller);                      // @0x10005f0a0
bool ButtonHeld(int controller, int button);           // @0x10005e340
int GetControllerNum(int player);                      // @0x10005e810 (player = session slot)
void _UpdatePad(int controller, float dt);             // @0x10005f660
// presentation-side readers/writers of the same state (ported so pad bytes stay identical to the game):
void VirtualUpdateVibration(float a, float b, PoleRiders::SessionPlayer p);  // @0x10005e740 (Player::Update)
float VirtualLeftTrigger(PoleRiders::SessionPlayer p);  // @0x10005ed30 (Player::Taunting)
bool VirtualPRTaunt(PoleRiders::SessionPlayer p);       // @0x10005eea0 (Player::Taunting)
}  // namespace Input
}  // namespace Common
namespace atrophy {
bool Input_IsGamepadConnectedAt(int i);                // @0x10004d8e0
void *Input_GetGamepadAt(int i);                       // @0x10004d940 (null if not connected)
// atrophy::Gamepad::setJoystickPosition(int stick, float x, float y) @0x1000116ce on gamepad `gp`
void Gamepad_setJoystickPosition(void *gp, int stick, float x, float y);
// atrophy::Gamepad::setMotorAt(int motor, float v) @0x10004d78e (rumble; no-op without a haptic device)
void Gamepad_setMotorAt(void *gp, int motor, float v);
}  // namespace atrophy
