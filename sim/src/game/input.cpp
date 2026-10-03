// Package D: the engine input chain, from the bridge's stick values to what Player reads.
//   atrophy::Gamepad::setJoystickPosition -> Common::Input::_UpdatePad -> Common::Input::Virtual*Stick
// All state lives in sim::g_engine (per match): gamepad[4] = atrophy::sGamepadList, pads[8] =
// Common::Input::_pads, session[4] = Common::Session::_sessionData player records.
//
// Not ported (no effect on anything the match reads):
//  - atrophy::Event_DispatchEvent (from setJoystickPosition's emulated-button path): event types 0xb/0xc go
//    to the only input delegate, Common::Input::_delegate, whose gamepadButtonDown/Up (@0x100060bee /
//    @0x100060bf4) are empty. The emulated-button path is ported (it edits GamepadSDL.buttons[].sources).
//  - Common::Move::* (PS Move): the bridge stubs Move::Init/Update (SPRBRIDGE_NOMOVE), so the move
//    gameData stays zero and Common::Move::Active(i) (@0x100060e90, gameData[i].active) is false: the
//    Move block at the end of _UpdatePad never runs.
#include <cmath>
#include <cstring>
#include "engine.h"

using atrophy::GamepadSDL;
using atrophy::GamepadStick;
using Common::Input::Pad;
using Common::Input::Stick;

namespace {

inline uint32_t fbits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
inline float bitsf(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
// (bits(v) & 0x80000000) + 0x3f800000: +1.0f or -1.0f by v's sign bit (also for +-0 and NaN)
inline float signOne(float v) { return bitsf((fbits(v) & 0x80000000u) + 0x3f800000u); }

GamepadSDL *GP(int i) { return sim::g_engine->gp(i); }
Pad *PAD(int i) { return sim::g_engine->pad(i); }

const float kOne = 1.0f;
const float kTwoPi = 0x1.921fb6p+2f;    // 0x40c90fdb
const float kMinusTwoPi = -0x1.921fb6p+2f;
const float kPi = 0x1.921fb6p+1f;       // 0x40490fdb
// atrophy::kEmulatedButtonAngles @0x100286600
const float kEmulatedButtonAngles[4] = {0x1.921fb6p+0f, -0x1.921fb6p+0f, 0x1.921fb6p+1f, 0.0f};

// Stick-emulated button release/press (GamepadSDL.buttons[b].sources); the edge events it would dispatch
// (Event_DispatchEvent, types 0xc up / 0xb down) are no-ops for the game (see top).
// Raw offsets like the asm (0xe0 + 8*b): with emuCurrent == emuButton[k] == -1 the game really touches
// GamepadSDL+0xd8 (responseCount's low half) — `src & ~bits` of 0 leaves it 0.
inline uint16_t *sourcesAt(GamepadSDL *gp, int b) {
  return reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(gp) + 0xe0 + 8 * static_cast<long>(b));
}
inline void releaseSources(GamepadSDL *gp, int b, uint16_t bits) {
  uint16_t *s = sourcesAt(gp, b);
  *s = static_cast<uint16_t>(*s & ~bits);
}
inline void addSources(GamepadSDL *gp, int b, uint16_t bits) {
  uint16_t *s = sourcesAt(gp, b);
  *s = static_cast<uint16_t>(*s | bits);
}

// piecewise-linear response table (GamepadSDL+0xd0, count +0xd8); count is 0 on the bridge's pads.
// Reproduces the asm literally, including frac = |v|*n - n (not - i).
inline float respond(const float *tab, int n, float v) {
  const float fn = static_cast<float>(n);
  float a = std::fabs(v) * fn;
  int i = static_cast<int>(a);  // cvttss2si
  if (n < i) i = n;
  int j = i + 1;
  if (n < j) j = n;
  float frac = a - fn;
  float r = (kOne - frac) * tab[i] + frac * tab[j];
  return signOne(v) * r;
}

}  // namespace

namespace atrophy {

// atrophy::Input_IsGamepadConnectedAt(int) @0x10004d8e0
bool Input_IsGamepadConnectedAt(int i) { return GP(i)->connected != 0; }

// atrophy::Input_GetGamepadAt(int) @0x10004d940
void *Input_GetGamepadAt(int i) {
  GamepadSDL *gp = GP(i);
  return gp->connected != 0 ? gp : nullptr;
}

// atrophy::Gamepad::setJoystickPosition(int, float, float) @0x1000116ce
void Gamepad_setJoystickPosition(void *gpv, int stick, float x, float y) {
  GamepadSDL *gp = static_cast<GamepadSDL *>(gpv);
  GamepadStick &s = gp->sticks[stick];
  const float dz = s.deadZone;
  // per axis: sign(v) * fminf(1, fmaxf(0, |v| - dz) / (1 - dz))
  const float den = kOne - dz;
  float nx = signOne(x) * fminf(kOne, fmaxf(0.0f, std::fabs(x) - dz) / den);
  float x2 = nx * nx;
  float ny = signOne(y) * fminf(kOne, fmaxf(0.0f, std::fabs(y) - dz) / den);
  float l = ny * ny + x2;
  if (l > kOne) {  // ucomiss/jbe: NaN keeps the values. Divides by the SQUARED length (sic).
    ny = ny / l;
    nx = nx / l;
  }
  if (gp->responseCount != 0) {
    const int n = gp->responseCount - 1;
    ny = respond(gp->response, n, ny);
    nx = respond(gp->response, n, nx);
  }
  s.x = nx;
  s.y = ny;
  if (s.emulate == 0) return;

  // stick -> emulated d-pad buttons (4 sectors around kEmulatedButtonAngles)
  float len = sqrtf(static_cast<float>(static_cast<double>(nx) * static_cast<double>(nx) +
                                       static_cast<double>(ny) * static_cast<double>(ny)));
  if (len > 0.5f) {
    const float angle = atan2f(ny, nx);
    const int cl = 4 * stick + 1;
    const uint16_t bit = static_cast<uint16_t>(1 << cl);
    for (int k = 0; k < 4; k++) {
      const int b = s.emuButton[k];
      const uint16_t src = stick < 4 ? bit : 0;
      float a = fmodf(angle, kTwoPi);
      if (std::signbit(a)) a += kTwoPi;
      float c = fmodf(kEmulatedButtonAngles[k], kTwoPi);
      if (std::signbit(c)) c += kTwoPi;
      float d = fmodf(a - c, kTwoPi);
      if (std::signbit(d)) d += kTwoPi;
      if (d > kPi) d += kMinusTwoPi;
      d = std::fabs(d);
      const int cur = s.emuCurrent;
      if (!(s.emuHalfWidth >= d)) {  // outside this direction's sector (jb: < or unordered)
        if (cur != b) continue;
        releaseSources(gp, cur, src);
        s.emuCurrent = -1;
      } else {
        if (cur == b) continue;
        if (cur >= 0) releaseSources(gp, cur, src);
        if (b >= 0) addSources(gp, b, src);
        s.emuCurrent = b;
      }
    }
  } else {
    const int cur = s.emuCurrent;
    if (cur < 0) return;
    if (stick < 4) releaseSources(gp, cur, static_cast<uint16_t>(0xf << (4 * stick + 1)));
    s.emuCurrent = -1;
  }
}

// atrophy::Gamepad::setMotorAt(int, float) @0x10004d78e
void Gamepad_setMotorAt(void *gpv, int motor, float v) {
  GamepadSDL *gp = static_cast<GamepadSDL *>(gpv);
  if (gp->haptic == nullptr || gp->numMotors <= 0) return;
  const float cur = gp->motor[motor];
  const int32_t e = static_cast<int32_t>((fbits(cur - v) & 0x7f800000u) >> 23) - 0x7f;
  if (e < -0x13) return;  // |difference| < 2^-19
  const int32_t vi = static_cast<int32_t>(fbits(v)), ci = static_cast<int32_t>(fbits(cur));
  if ((vi ^ ci) >= 0) {  // same sign: skip if within 4 ulps
    int32_t d = static_cast<int32_t>(static_cast<uint32_t>(ci) - static_cast<uint32_t>(vi));
    if (d < 0) d = -d;
    if (d < 5) return;
  }
  gp->motor[motor] = v;
  gp->motorDirty = 1;
}

}  // namespace atrophy

namespace Common {
namespace Input {

namespace {
// GamepadSDL -> Pad button flags. Map A: gamepad types 1 (SDL), 4, 5; map B: types 7, 8.
// B(o) = GamepadSDL byte at offset o != 0 (buttons[(o-0xe0)/8].value).
void mapButtonsA(Pad *p, const uint8_t *g) {
  auto B = [g](int o) { return g[o] != 0; };
  uint8_t *P = p->held;
  P[0] = B(0x13a); P[1] = B(0x152); P[2] = B(0x142); P[3] = B(0x14a);
  P[0x16] = B(0x102) || (B(0x14a) && !B(0x13a) && !B(0x142));
  P[0x17] = B(0x10a) || (B(0x152) && !B(0x13a) && !B(0x142));
  P[0x18] = B(0x13a) && !B(0x14a) && !B(0x152);
  P[0x19] = B(0x142) && !B(0x14a) && !B(0x152);
  P[0x10] = B(0xe2); P[0x11] = B(0x102); P[0x12] = B(0x10a); P[0x13] = B(0xea);
  P[0x14] = B(0x15a); P[0x15] = B(0x162);
  P[0x1c] = P[0x1c] || B(0x102) || B(0x10a);
  P[0x1d] = P[0x1d] || B(0xe2) || B(0xea) || B(0xf2) || B(0xfa) || B(0x102) || B(0x10a) || B(0x15a) ||
            B(0x162) || B(0x12a) || B(0x122) || B(0x132) || B(0x112) || B(0x11a);
  P[4] = B(0x13a); P[5] = B(0x152); P[6] = B(0x142); P[7] = B(0x14a);
  P[8] = B(0xfa); P[9] = B(0xea); P[0xa] = B(0xe2); P[0xb] = B(0xf2);
  P[0x4a] = B(0xfa) && B(0x10a);
  P[0x49] = B(0x11a);
  P[0x21] = B(0xf2); P[0x22] = B(0xe2); P[0x23] = B(0x102); P[0x24] = B(0x10a);
  P[0x25] = B(0xfa) && B(0xea);
  P[0x32] = B(0xe2); P[0x33] = B(0xfa);
  P[0x34] = B(0x142) || B(0x13a) || B(0x14a) || B(0x152) || B(0x102);
  P[0x35] = B(0x15a) || B(0x162);
  P[0x36] = B(0xea); P[0x37] = B(0xf2);
  P[0x38] = B(0x102) || B(0x12a) || B(0x10a);
  P[0x39] = B(0xe2) || B(0xf2);
  P[0x3a] = B(0xea) || B(0xfa);
  P[0x46] = B(0x15a); P[0x47] = B(0x162); P[0x4d] = B(0x10a); P[0x4c] = B(0x162);
  P[0x4f] = B(0x102); P[0x4e] = B(0x15a);
  P[0x4b] = B(0x102) && B(0x10a);
  if (B(0x122)) P[0x1e] = 1;
}

void mapButtonsB(Pad *p, const uint8_t *g) {
  auto B = [g](int o) { return g[o] != 0; };
  uint8_t *P = p->held;
  P[0] = B(0x102); P[1] = B(0x11a); P[2] = B(0x10a); P[3] = B(0x112);
  P[0x16] = B(0x122) || (B(0x112) && !B(0x102) && !B(0x10a));
  P[0x17] = B(0x12a) || (B(0x11a) && !B(0x102) && !B(0x10a));
  P[0x18] = B(0x102) && !B(0x112) && !B(0x11a);
  P[0x19] = B(0x10a) && !B(0x112) && !B(0x11a);
  P[0x10] = B(0xe2); P[0x11] = B(0x122); P[0x12] = B(0x12a); P[0x13] = B(0xea);
  P[0x14] = B(0x132); P[0x15] = B(0x13a);
  P[0x1c] = P[0x1c] || B(0x122) || B(0x12a);
  P[0x1d] = P[0x1d] || B(0xe2) || B(0xea) || B(0xf2) || B(0xfa) || B(0x122) || B(0x12a) || B(0x132) ||
            B(0x13a) || B(0x15a) || B(0x152) || B(0x162) || B(0x142) || B(0x14a);
  P[4] = B(0x102); P[5] = B(0x11a); P[6] = B(0x10a); P[7] = B(0x112);
  P[8] = B(0xfa); P[9] = B(0xea); P[0xa] = B(0xe2); P[0xb] = B(0xf2);
  P[0x4a] = B(0xfa) && B(0x12a);
  P[0x49] = B(0x14a);
  P[0x21] = B(0xf2); P[0x22] = B(0xe2); P[0x23] = B(0x122); P[0x24] = B(0x12a);
  P[0x25] = B(0xfa) && B(0xea);
  P[0x32] = B(0xe2); P[0x33] = B(0xfa);
  P[0x34] = B(0x10a) || B(0x102) || B(0x112) || B(0x11a) || B(0x162);
  P[0x35] = B(0x13a) || B(0x132);
  P[0x36] = B(0xea); P[0x37] = B(0xf2);
  P[0x38] = B(0x12a) || B(0x122);
  P[0x39] = B(0xe2) || B(0xf2) || B(0x102);
  P[0x3a] = B(0xea) || B(0xfa) || B(0x10a);
  P[0x46] = B(0x132); P[0x47] = B(0x13a); P[0x4d] = B(0x12a); P[0x4c] = B(0x13a);
  P[0x4f] = B(0x122); P[0x4e] = B(0x132);
  P[0x4b] = B(0x122) && B(0x12a);
  if (B(0x152)) P[0x1e] = 1;
}

// digital fallback of LeftStick/RightStick: x = (+x ? 1 : 0) [+ -1 if -x], y likewise
inline float digitalAxis(bool plus, bool minus) {
  float v = plus ? 1.0f : 0.0f;
  if (minus) v += -1.0f;
  return v;
}
// VirtualLeftStick's d-pad value: plus ? (minus ? 0 : 1) : (minus ? -1 : 0) via the {1,0}/{0,-1} tables
inline float dpadAxis(bool plus, bool minus) {
  static const float kT1[2] = {1.0f, 0.0f}, kT2[2] = {0.0f, -1.0f};  // @0x100286aa0 / @0x100286aa8
  const int c = plus ? 0 : 1;
  return minus ? kT2[c] : kT1[c];
}
// VirtualLeftStick modes 1/2 (shared-controller halves): r = 0; plus -> 1; minus -> (plus ? 0 : -1)
inline float halfAxis(bool plus, bool minus, float init) {
  float r = init, neg = -1.0f;
  if (plus) r = 1.0f, neg = 0.0f;
  if (minus) r = neg;
  return r;
}
}  // namespace

// Common::Input::ButtonHeld(int, Common::Input::Button) @0x10005e340
bool ButtonHeld(int controller, int button) { return PAD(controller)->held[button] != 0; }

// Common::Input::RightStick(int) @0x10005f0a0
Stick RightStick(int c) {
  Stick r;
  const Pad *p = PAD(c);
  if (static_cast<unsigned>(c) <= 3 && atrophy::Input_IsGamepadConnectedAt(c)) {
    for (int k = 0; k < 4; k++) r.v[k] = p->rightStick[k];
  } else {
    r.x = digitalAxis(p->held[9], p->held[0xb]);
    r.y = digitalAxis(p->held[8], p->held[0xa]);
    r.z = 0.0f;
    r.w = 0.0f;
  }
  return r;
}

// Common::Input::LeftStick(int) @0x10005f1a0
Stick LeftStick(int c) {
  Stick r;
  const Pad *p = PAD(c);
  if (static_cast<unsigned>(c) <= 3 && atrophy::Input_IsGamepadConnectedAt(c)) {
    for (int k = 0; k < 4; k++) r.v[k] = p->leftStick[k];
  } else {
    r.x = digitalAxis(p->held[1], p->held[3]);
    r.y = digitalAxis(p->held[0], p->held[2]);
    r.z = 0.0f;
    r.w = 0.0f;
  }
  return r;
}

// Common::Input::VirtualRightStick(Common::Session::Player) @0x10005f290
Stick VirtualRightStick(PoleRiders::SessionPlayer sp) {
  return sp.mode == 1 ? LeftStick(sp.pad) : RightStick(sp.pad);
}

// Common::Input::VirtualLeftStick(Common::Session::Player) @0x10005f2c0
Stick VirtualLeftStick(PoleRiders::SessionPlayer sp) {
  Stick r;
  r.z = 0.0f;
  r.w = 0.0f;
  const int c = sp.pad;
  const Pad *p = PAD(c);
  if (sp.mode == 1) {  // left half of a shared pad: buttons 4..7
    r.y = halfAxis(p->held[4], p->held[6], 0.0f);
    r.x = halfAxis(p->held[5], p->held[7], 0.0f);
    return r;
  }
  if (sp.mode == 2) {  // right half: buttons 8..0xb
    r.y = halfAxis(p->held[8], p->held[0xa], 0.0f);
    r.x = halfAxis(p->held[9], p->held[0xb], 0.0f);
    return r;
  }
  const bool analog = static_cast<unsigned>(c) <= 3 && atrophy::Input_IsGamepadConnectedAt(c);
  // x: analog stick (or its digital fallback) unless the d-pad (5 = right, 7 = left) is at least as big
  const float ax = analog ? p->leftStick[0] : digitalAxis(p->held[1], p->held[3]);
  const float dx = dpadAxis(p->held[5], p->held[7]);
  r.x = std::fabs(ax) > std::fabs(dx) ? ax : dx;  // ucomiss/jbe: NaN -> d-pad
  // y: 4 = up, 6 = down
  const float ay = analog ? p->leftStick[1] : digitalAxis(p->held[0], p->held[2]);
  const float dy = dpadAxis(p->held[4], p->held[6]);
  r.y = std::fabs(ay) > std::fabs(dy) ? ay : dy;
  return r;
}

// Common::Input::VirtualUpdateVibration(float, float, Common::Session::Player) @0x10005e740
// Rumble request into _pads (+0xf4/+0xf8); the next _UpdatePad hands it to setMotorAt and zeroes it.
void VirtualUpdateVibration(float a, float b, PoleRiders::SessionPlayer sp) {
  const float m = fmaxf(a, b);
  Pad *p = PAD(sp.pad);
  if (sp.mode == 2) {
    const float h = (a + b) * 0.5f;
    p->rumble[0] = fminf(static_cast<float>(static_cast<double>(h) * 1.4), 1.0f);  // 1.4 = 0x3ff6666666666666
  } else if (sp.mode == 1) {
    p->rumble[1] = m;
  } else if (sp.mode == 0) {
    p->rumble[0] = a;
    p->rumble[1] = b;
  }
}

// Common::Input::VirtualLeftTrigger(Common::Session::Player) @0x10005ed30 — analog LT (byte / 255)
float VirtualLeftTrigger(PoleRiders::SessionPlayer sp) {
  const int c = sp.pad;
  if (static_cast<unsigned>(c) > 3 || !atrophy::Input_IsGamepadConnectedAt(c)) return 0.0f;
  const uint8_t *g = static_cast<const uint8_t *>(atrophy::Input_GetGamepadAt(c));
  const int v = sp.mode == 2 ? g[0x162] : g[0x15a];
  return static_cast<float>(v) / 255.0f;
}

// Common::Input::VirtualPRTaunt(Common::Session::Player) @0x10005eea0 — stick-click button
// (the game calls Input_GetGamepadAt(pad) before the range check; only its result matters)
bool VirtualPRTaunt(PoleRiders::SessionPlayer sp) {
  const int c = sp.pad;
  if (static_cast<unsigned>(c) >= 4 || !atrophy::Input_IsGamepadConnectedAt(c)) return false;
  const uint8_t *g = static_cast<const uint8_t *>(atrophy::Input_GetGamepadAt(c));
  return (sp.mode == 1 ? g[0x112] : g[0x11a]) != 0;
}

// Common::Input::GetControllerNum(int) @0x10005e810 — controller index shown for a session slot.
// Mode 2 (right half of a shared pad) with a mode-1 partner on the same pad returns 7 or 8.
// Transliterated from the asm's control flow (unused in training: all slots are mode 0).
int GetControllerNum(int player) {
  const PoleRiders::SessionPlayer *s = sim::g_engine->session;
  const int ret = s[player].pad;
  if (s[player].mode != 2) return ret;
  int j = 0;
  for (; j < 4; j++)
    if (player != j && s[j].mode == 1 && s[j].pad == ret && s[j].team != 0) break;
  if (j == 4) return ret;
  const int p = player;
  int edx = 0, edi, r8, r9, r10;
  long rbx;
  bool dil = (p == 0) || (j == 0);
  if (!dil) goto L850;
  r8 = j - 1, r9 = p - 1, r10 = 0;
L893:
  edx = r10 + 1;
  if (r8 == r10 || r9 == r10) {
    r10 = edx;
    goto L893;
  }
  dil = dil || edx == 0;
  if (dil) goto L850;
  rbx = 0;
  goto L8bf;
L850:
  edi = 1, r8 = 0;
  for (;;) {
    r9 = edi + 1;
    const bool skip = (r8 == j - 1);
    r8++;
    if (skip || p == edi || edx == edi) {
      edi = r9;
      continue;
    }
    break;
  }
  rbx = edi;
L8bf:
  const PoleRiders::SessionPlayer &a = s[edx], &b = s[rbx];
  if (a.team == 0 || b.team == 0 || a.mode == 0 || b.mode == 0 || a.pad != b.pad) return 7;
  return 7 + (a.pad <= ret ? 1 : 0);
}

// Common::Input::_UpdatePad(int, float) @0x10005f660
void _UpdatePad(int c, float dt) {
  Pad *p = PAD(c);
  uint8_t prev[kNumButtons];
  for (int k = 0; k < kNumButtons; k++) {
    prev[k] = p->held[k];
    p->held[k] = 0;
  }
  if (c <= 3) {
    if (static_cast<unsigned>(c) <= 3 && atrophy::Input_IsGamepadConnectedAt(c)) {
      GamepadSDL *gp = static_cast<GamepadSDL *>(atrophy::Input_GetGamepadAt(c));
      const uint8_t *g = reinterpret_cast<const uint8_t *>(gp);
      const int type = gp->type;
      if (static_cast<unsigned>(type - 4) < 2 || type == 1)
        mapButtonsA(p, g);
      else if (static_cast<unsigned>(type - 7) <= 1)
        mapButtonsB(p, g);

      const int motors = gp->numMotors;
      if (motors == 1) atrophy::Gamepad_setMotorAt(gp, 0, (p->rumble[0] + p->rumble[1]) * 0.5f);
      if (motors >= 2) {
        atrophy::Gamepad_setMotorAt(gp, 0, p->rumble[0]);
        atrophy::Gamepad_setMotorAt(gp, 1, p->rumble[1]);
      }
      p->rumble[0] = 0.0f;
      p->rumble[1] = 0.0f;
      gp->padColor = p->padColor;

      const GamepadSDL *g2 = static_cast<GamepadSDL *>(atrophy::Input_GetGamepadAt(c));
      p->leftStick[0] = g2->sticks[0].x;
      p->leftStick[1] = g2->sticks[0].y;
      p->leftStick[2] = g2->sticks[0].z;
      p->leftStick[3] = g2->sticks[0].w;
      const float rx = g2->sticks[1].x, ry = g2->sticks[1].y;
      p->rightStick[0] = rx;
      p->rightStick[1] = ry;
      p->rightStick[2] = g2->sticks[1].z;
      p->rightStick[3] = g2->sticks[1].w;

      // stick "flick" buttons (float compares; the 0.8/0.2 ones in double with float-exact constants)
      const float ly = p->leftStick[1], lx = p->leftStick[0];
      uint8_t *P = p->held;
      if (ly > 0.4f) P[0] = 1;
      if (lx > 0.4f) P[1] = 1;
      if (-0.4f > ly) P[2] = 1;
      if (-0.4f > lx) P[3] = 1;
      if (ry > 0.4f) P[0xc] = 1;
      if (rx > 0.4f) P[0xd] = 1;
      if (-0.4f > ry) P[0xe] = 1;
      if (-0.4f > rx) P[0xf] = 1;
      const double k08 = static_cast<double>(0.8f), k02 = static_cast<double>(0.2f);
      const double dlx = lx, dly = ly, dry = ry;
      if (-k08 > dlx && k02 > static_cast<double>(std::fabs(ly))) P[0x16] = 1;
      if (dlx > k08 && k02 > static_cast<double>(std::fabs(ly))) P[0x17] = 1;
      if (-k08 > dly && k02 > static_cast<double>(std::fabs(lx))) P[0x18] = 1;
      if (dly > k08 && k02 > static_cast<double>(std::fabs(lx))) P[0x19] = 1;
      if (-k08 > dry && k02 > static_cast<double>(std::fabs(rx))) P[0x1a] = 1;
      if (dry > k08 && k02 > static_cast<double>(std::fabs(rx))) P[0x1b] = 1;
      if (ry > 0.4f) P[0x2e] = 1;
      if (rx > 0.4f) P[0x31] = 1;
      if (-0.4f > ry) P[0x2f] = 1;
      if (-0.4f > rx) P[0x30] = 1;

      if (static_cast<GamepadSDL *>(atrophy::Input_GetGamepadAt(c))->numSticks >= 3) {
        const GamepadSDL *g3 = static_cast<GamepadSDL *>(atrophy::Input_GetGamepadAt(c));
        p->triggers[0] = g3->sticks[2].x;
        p->triggers[1] = g3->sticks[2].y;
        p->triggers[2] = g3->sticks[3].x;
        p->triggers[3] = 0.0f;
      } else {
        for (int k = 0; k < 4; k++) p->triggers[k] = 0.0f;
      }
    }
    // Common::Move::Active(c) is false (PS Move stubbed by the bridge): no Move buttons / virtual d-pad.
  }
  for (int k = 0; k < kNumButtons; k++) p->held[k] = (p->held[k] | p->forced[k]) != 0;
  for (int k = 0; k < kNumButtons; k++) p->pressed[k] = prev[k] == 0 && p->held[k] != 0;
  for (int k = 0; k < kNumButtons; k++) {
    bool rep;
    if (p->held[k] != 0) {
      const float t = p->holdTime[k] + dt;
      p->holdTime[k] = t;
      const int n1 = static_cast<int>(t * 7.0f);         // cvttss2si
      const int n0 = static_cast<int>((t - dt) * 7.0f);
      rep = (n1 != n0 && t > 0.5f) || p->pressed[k] != 0;
    } else {
      p->holdTime[k] = 0.0f;
      rep = false;
    }
    p->repeat[k] = rep;
  }
}

}  // namespace Input
}  // namespace Common

namespace sim {
// Input state of a fresh bridge shadow (farm.c sh_create copies the engine's pads 0/1 and zeroes the
// axes): the bridge's virtual SDL pads as GamepadSDL::connect left them, after the idle frames the real
// match ran (sticks (0, -0) from setJoystickPosition(0, 0, -0)). Pad colours / SDL instance ids are the
// values the real game has for pads 0/1.
void DefaultInputState(EngineContext &e) {
  static const int32_t kColor[2] = {static_cast<int32_t>(0xffdbf900u), static_cast<int32_t>(0xff00ffffu)};
  for (int i = 0; i < 2; i++) {
    memset(e.gamepad[i], 0, sizeof e.gamepad[i]);
    memset(e.pads[i], 0, sizeof e.pads[i]);
    atrophy::GamepadSDL *gp = e.gp(i);
    gp->id = i;
    gp->type = 1;
    gp->connected = 1;
    for (int s = 0; s < 4; s++) {
      atrophy::GamepadStick &st = gp->sticks[s];
      st.deadZone = s < 2 ? 0.1f : 0.0f;
      for (int k = 0; k < 4; k++) st.emuButton[k] = -1;
      st.emuCurrent = -1;
    }
    gp->sticks[0].y = -0.0f;
    gp->sticks[1].y = -0.0f;
    gp->numSticks = 2;
    gp->numButtons = 17;
    gp->padColor = kColor[i];
    gp->instanceId = -1000 - i;
    Common::Input::Pad *p = e.pad(i);
    p->leftStick[1] = -0.0f;
    p->rightStick[1] = -0.0f;
    p->padColor = kColor[i];
  }
}
}  // namespace sim
