// Pole construction / destruction (package A). Pole helpers (Segment, Detach, ...) are package B
// (pole.cpp).
#include <cstring>
#include "assets.h"
#include "layout.h"
#include "../x86math.h"

namespace PoleRiders {

namespace {
// Shared poles (shared == true) tag their far handle segments with the partner's index. The original
// stores the ADDRESS OF A STACK SLOT of the Pole ctor (-0x15c(%rbp)) as the body userData, i.e. a
// dangling pointer whose contents are whatever later reuses that stack slot. We keep the value in one
// per-thread slot instead (only reachable with shared-controller modes 1/2, never in training).
thread_local int32_t g_partnerTag;
}  // namespace

// PoleRiders::Pole::Pole(float, float, b2World*, int, int, float, SpriteSheet*, bool, float, bool, bool)
// @0x10013e64c (C2; C1 thunk @0x10013ef5a)
//   Pole(x px, y px, world, team, ownerIndex, angle, sprites, hammer, lengthPx, shared, shortPole)
Pole::Pole(float x, float y, b2World *w, int teamArg, int index, float angle, SpriteSheet *sheet,
           bool hammer, float length, bool sharedArg, bool shortArg) {
  memset(this, 0, sizeof *this);  // operator new'd in the game; fields it never writes read as 0
  lengthPx = length;
  team = teamArg;
  ownerIndex = index;
  world = w;
  isHammer = hammer;
  shared = sharedArg;
  shortPole = shortArg;

  uint16_t cat, mask, maskHandle;  // -0x27e, -0x280, -0x28a
  if (teamArg == 0) {
    mask = 0xfe25, cat = 0x8, maskHandle = 0xfe01;
  } else if (teamArg == 1) {
    mask = 0xfe23, cat = 0x10, maskHandle = 0xfe01;
  } else {
    cat = 1, maskHandle = 0xffff, mask = 0xffff;
  }

  b2PolygonShape box;  // -0xe8
  b2FixtureDef fd;     // -0x118
  const float r = Assets::ratio;
  const float x0 = x / r, y0 = y / r;
  const int handleStart = Assets::HANDLE_START;
  const int handleEnd = 10 - handleStart;  // -0x2a8
  const float halfWidth = 16.0f / (r + r);
  const float r40 = r * 40.0f;
  const float segLen = length / r40;       // -0x270
  const float halfLen = 0x1.333334p+0f * segLen;  // 1.2f * segLen
  fd.shape = nullptr;
  fd.userData = nullptr;
  fd.filter.categoryBits = 1;
  fd.filter.maskBits = 0xffff;
  fd.filter.groupIndex = 0;
  fd.isSensor = false;
  fd.friction = 0x1.99999ap-2f;  // 0x3ecccccd = 0.4
  fd.density = 0x1.99999ap-3f;   // 0x3e4ccccd = 0.2
  fd.restitution = 0.0f;
  const float linDamp = Assets::poleLinearDamping;
  const float angDamp = Assets::poleAngularDamping;

  for (int i = 0; i < 10; i++) {
    b2BodyDef bd;  // -0x158
    void *ud;
    if (i > handleStart && !sharedArg) {
      ud = const_cast<int *>(&Assets::BODY_POLE);
    } else if (!sharedArg || i < handleStart) {
      ud = &ownerIndex;
    } else if (handleEnd > i) {
      ud = const_cast<int *>(&Assets::BODY_POLE);
    } else {
      g_partnerTag = (ownerIndex + 2) % 4;  // C remainder (sign of the dividend), as compiled
      ud = &g_partnerTag;
    }
    bd.userData = ud;
    const float fi = (float)(i - 1);
    float px = x86_cosf(angle) * fi * segLen;
    px = px + px;
    bd.position.x = px + x0;
    float py = x86_sinf(angle) * fi * segLen;
    py = py + py;
    bd.position.y = py + y0;
    bd.angle = angle;
    bd.angularDamping = angDamp;
    bd.linearDamping = linDamp;
    bd.type = b2_dynamicBody;
    box.SetAsBox(halfLen, halfWidth);
    fd.shape = &box;
    fd.userData = &ownerIndex;
    fd.filter.categoryBits = cat;
    fd.filter.maskBits = i < 2 ? maskHandle : mask;
    fd.filter.groupIndex = 0;
    bd.bullet = false;
    b2Body *b = world->CreateBody(&bd);
    segments[i] = b;
    b->CreateFixture(&fd);
    if (i == 9) {  // round tip
      b2CircleShape tip;  // -0x178
      tip.m_radius = halfWidth;
      tip.m_p.x = lengthPx / r40;
      tip.m_p.y = 0.0f;
      fd.shape = &tip;
      b->CreateFixture(&fd);
    }
    sprites = sheet;
  }

  for (int i = 0; i < 9; i++) {
    b2Body *a = segments[i], *b = segments[i + 1];
    b2Vec2 anchor;
    anchor.x = (a->GetPosition().x + b->GetPosition().x) * 0.5f;
    anchor.y = (a->GetPosition().y + b->GetPosition().y) * 0.5f;
    b2WeldJointDef wd;
    wd.Initialize(a, b, anchor);
    wd.collideConnected = false;
    world->CreateJoint(&wd);
  }

  if (hammer) {
    b2BodyDef bd;  // -0x1f8
    bd.angularDamping = Assets::poleAngularDamping;
    bd.linearDamping = Assets::poleLinearDamping;
    bd.position = segments[9]->GetPosition();
    bd.type = b2_dynamicBody;
    bd.userData = const_cast<int *>(&Assets::BODY_HAMMER);
    b2FixtureDef hd;  // -0x228
    hd.isSensor = false;
    hd.shape = &box;
    hd.userData = &ownerIndex;
    hd.filter.categoryBits = cat;
    hd.filter.maskBits = mask;
    hd.filter.groupIndex = 0;
    hd.density = 0x1.99999ap-4f;      // 0x3dcccccd = 0.1
    hd.friction = 0x1.333334p-2f;     // 0x3e99999a = 0.3
    hd.restitution = 0x1.333334p-1f;  // 0x3f19999a = 0.6
    bd.angle = angle;
    box.SetAsBox(24.5f / r, 56.0f / r);
    hammerBody = world->CreateBody(&bd);
    hammerBody->CreateFixture(&hd);
    // (the original then builds two more boxes, SetAsBox(19/r, 8/r, (0, -+50/r), 0), and points the
    // fixture def at them, but never creates fixtures from them: no effect)
    b2WeldJointDef wd;
    wd.Initialize(segments[9], hammerBody, hammerBody->GetPosition());
    wd.collideConnected = false;
    world->CreateJoint(&wd);
  }
}

// PoleRiders::Pole::~Pole() @0x10013e5c2
Pole::~Pole() {
  if (isHammer) world->DestroyBody(hammerBody);
  for (int i = 0; i < 10; i++) world->DestroyBody(segments[i]);
}

}  // namespace PoleRiders
