// Player per-tick logic and state (package B): accessors, ghosting, collision filters, kick / land /
// lift-off, the pole servo in Update, WallCollision, Teleport, Respawn with an existing pole.
// Construction / SpawnPlayerBody / Respawn-with-new-pole / dtor are package A.
//
// Float semantics follow the x86 code op by op (see PORTING.md). Presentation-only calls (sound,
// rumble output, dust particles) are skipped; the cheap presentation *fields* they feed (rumble,
// bend/walk volume, timers) are still updated so raw Player dumps stay close to the game's.
#include <cmath>
#include "animation_data.h"
#include "assets.h"
#include "engine.h"

namespace PoleRiders {

namespace {

constexpr float kPi = 0x1.921fb6p+1f;      // 3.14159274
constexpr float kTwoPi = 0x1.921fb6p+2f;   // 6.28318548
constexpr float kHalfPi = 0x1.921fb6p+0f;  // 1.57079637

// b2Contact::FlagForFiltering() is protected; the game's code sets m_flags |= e_filterFlag on every
// contact of the world (inlined). Reach the protected member through a derived-class member pointer.
struct ContactAccess : b2Contact {
  static constexpr uint32 b2Contact::*flags = &ContactAccess::m_flags;
  static constexpr uint32 filterFlag = b2Contact::e_filterFlag;
};
static_assert(ContactAccess::filterFlag == 0x8, "e_filterFlag");

// for (c = world->GetContactList(); c; c = c->GetNext()) c->FlagForFiltering();
inline void FlagAllContactsForFiltering(b2World *world) {
  for (b2Contact *c = world->GetContactList(); c; c = c->GetNext()) c->*ContactAccess::flags |= ContactAccess::filterFlag;
}

inline void SetFilterAll(b2Body *b, const b2Filter &f) {
  for (b2Fixture *fx = b->GetFixtureList(); fx; fx = fx->GetNext()) fx->SetFilterData(f);
}

inline b2Filter MakeFilter(uint16 cat, uint16 mask, int16 group) {
  b2Filter f;
  f.categoryBits = cat;
  f.maskBits = mask;
  f.groupIndex = group;
  return f;
}

inline int CurrentLevel() { return sim::g_engine->cfg.level; }  // Assets::currentLevel

}  // namespace

// ------------------------------------------------------------------------------------------------
// small accessors
// ------------------------------------------------------------------------------------------------

// PoleRiders::Player::GetPole() @0x10012c000
Pole *Player::GetPole() { return pole; }
// PoleRiders::Player::SharingPole() @0x10012c00a
bool Player::SharingPole() { return sharingPole; }
// PoleRiders::Player::SetRunSpeed(float) @0x10012c014
void Player::SetRunSpeed(float s) { runSpeed = s; }

// PoleRiders::Player::HammerLand() @0x10012c03c
void Player::HammerLand() {
  int n = hammerGroundContacts;
  if (n <= 0) {
    hammerGroundContacts = 0;
    n = 0;
  }
  n++;
  hammerGroundContacts = n;
  if (n >= 2) hammerGroundContacts = 1;
}

// PoleRiders::Player::HammerLiftoff() @0x10012c070
void Player::HammerLiftoff() {
  int n = hammerGroundContacts - 1;
  hammerGroundContacts = n;
  if (n <= 0) {
    hammerGroundContacts = 0;
    canPlayHammerSound = true;
  }
}

// PoleRiders::Player::OffPole() @0x10012c0a8
void Player::OffPole() {
  offPolePending = true;
  if (sharingPole) unsharePending = true;
}
// PoleRiders::Player::ShouldUnshare() @0x10012c0c2
bool Player::ShouldUnshare() { return unsharePending; }
// PoleRiders::Player::ResetUnshare() @0x10012c0d0
void Player::ResetUnshare() { unsharePending = false; }

// PoleRiders::Player::PoleContactAdjust(int) @0x10012c0ec
void Player::PoleContactAdjust(int n) {
  int v = n + poleGroundContacts;
  poleGroundContacts = v;
  if (v < 0) poleGroundContacts = 0;
}

// PoleRiders::Player::WallCollision() @0x10012c10c
// On the pole and airborne: horizontal impulse of 150 px/ratio away from the wall (toward x = 0);
// b2Body::ApplyLinearImpulse at the body's world centre is inlined in the game.
void Player::WallCollision() {
  if (!onPole || onGround) return;
  b2Body *b = body;
  float r = Assets::ratio;
  float ix;
  if (0.0f > b->GetPosition().x)
    ix = 150.0f / r;
  else
    ix = -150.0f / r;
  b->ApplyLinearImpulse(b2Vec2(ix, 0.0f), b->GetWorldCenter());
}

// PoleRiders::Player::Body() @0x10012c1fa
b2Body *Player::Body() { return body; }

// PoleRiders::Player::Boost(float) @0x10012c204
void Player::Boost(float speed) {
  boostStrength = speed;
  boostTimer = 0x1.99999ap-1f;  // 0x3f4ccccd = 0.8f
}

// PoleRiders::Player::OnPole() @0x10012c21c
bool Player::OnPole() { return onPole; }
// PoleRiders::Player::OnGround() @0x10012c226
bool Player::OnGround() { return onGround; }
// PoleRiders::Player::JustUnghosted() @0x10012c23e
bool Player::JustUnghosted() { return justUnghosted; }
// PoleRiders::Player::ShouldBallCollide() @0x10012c256
bool Player::ShouldBallCollide() { return 0.0f >= ballNoCollideTimer; }  // ucomiss; setae
// PoleRiders::Player::SetActive(bool) @0x10012c26e
void Player::SetActive(bool a) { active = a; }
// PoleRiders::Player::GetHeadNumber() @0x10012c27c
int Player::GetHeadNumber() { return session.head; }
// PoleRiders::Player::GetStats() @0x10012c30a
Stats *Player::GetStats() { return stats; }

// PoleRiders::Player::GameOverBody() @0x10012c354
void Player::GameOverBody() {
  bodyActive = false;
  emitDust = false;
  // slideSound->SetVolume(0): presentation
}

// PoleRiders::Player::GetHammer() @0x10012c61c
b2Body *Player::GetHammer() { return pole->Hammer(); }

// ------------------------------------------------------------------------------------------------
// collision filters / ghosting
// ------------------------------------------------------------------------------------------------

// PoleRiders::Player::_SetBallCollisionFilters() @0x10012fdb4
// While on the pole, the ball bit (0x20) of the mask of hull, foot and every pole-segment fixture
// follows ballNoCollideTimer (<= 0: collide with the ball).
void Player::_SetBallCollisionFilters() {
  if (!onPole) return;
  float t = ballNoCollideTimer;
  auto apply = [t](b2Fixture *fx) {
    b2Filter f = fx->GetFilterData();
    uint16 mask = f.maskBits;
    f.maskBits = (0.0f >= t) ? (uint16)(mask | 0x20) : (uint16)(mask & 0xffdf);  // cmovae (NaN: clear)
    fx->SetFilterData(f);
  };
  apply(hullFixture);
  apply(footFixture);
  for (int i = 0; i < pole->NumSegments(); i++)
    for (b2Fixture *fx = pole->Segment(i)->GetFixtureList(); fx; fx = fx->GetNext()) apply(fx);
}

// PoleRiders::Player::SetBallCollisions(bool) @0x10012ff2c
void Player::SetBallCollisions(bool on) {
  ballNoCollideTimer = on ? 0.0f : 0x1.333334p-1f;  // table {0.0f, 0.6f}[!on]
  _SetBallCollisionFilters();
}

// PoleRiders::Player::SetGhost(bool, float) @0x10012ff54
void Player::SetGhost(bool g, float t) {
  ghost = g;
  ghostTimer = t;
  b2Filter fSensor = MakeFilter(1, 0xffff, 0);  // -0x30: sensor, impale, hull (and ghost filter)
  b2Filter fFoot = MakeFilter(1, 0xffff, 0);    // -0x38: foot
  b2Filter fPole = MakeFilter(1, 0xffff, 0);    // -0x40: free pole segments (userData BODY_POLE)
  b2Filter fHand = MakeFilter(1, 0xffff, 0);    // -0x48: handle segments (userData = player index)
  if (g) {
    fSensor.maskBits = 1;
    fSensor.categoryBits = 0x100;
    SetFilterAll(body, fSensor);
    for (int i = 0; i < pole->NumSegments(); i++) SetFilterAll(pole->Segment(i), fSensor);
    justUnghosted = false;
    FlagAllContactsForFiltering(world);
    return;
  }
  if (team == 1) {
    fSensor = MakeFilter(4, 0xfe2b, 0);
    fFoot = MakeFilter(4, 0xfe22, 0);
    fHand = MakeFilter(0x10, 0xfe01, 0);
    fPole = MakeFilter(0x10, 0xfe23, 0);
  } else if (team == 0) {
    fSensor = MakeFilter(2, 0xfe35, 0);
    fFoot = MakeFilter(2, 0xfe24, 0);
    fHand = MakeFilter(8, 0xfe01, 0);
    fPole = MakeFilter(8, 0xfe25, 0);
  }
  footFixture->SetFilterData(fFoot);
  sensorFixture->SetFilterData(fSensor);
  impaleFixture->SetFilterData(fSensor);
  hullFixture->SetFilterData(fSensor);
  if (pole->NumSegments() > 0) {
    const int hs = Assets::HANDLE_START;
    int next = 0;
    do {
      int i = next;
      next = i + 1;
      bool hand;          // true: userData/filter of a handle segment
      void *handData;     // which player index the handle belongs to
      if (sharingPole) {
        if (i < hs) {
          hand = true, handData = &index;
        } else {
          int n = pole->NumSegments();
          pole->Segment(i);
          if (n - hs > i)
            hand = false, handData = nullptr;
          else
            hand = true, handData = &partnerIndex;
        }
      } else if (reversedGrip) {
        int n = pole->NumSegments();
        pole->Segment(i);
        if (n - hs >= i)
          hand = false, handData = nullptr;
        else
          hand = true, handData = &index;
      } else {
        pole->Segment(i);
        if (i < hs)
          hand = true, handData = &index;
        else
          hand = false, handData = nullptr;
      }
      b2Body *seg = pole->Segment(i);
      if (hand) {
        seg->SetUserData(handData);
        SetFilterAll(pole->Segment(i), fHand);
      } else {
        seg->SetUserData(const_cast<int *>(&Assets::BODY_POLE));
        SetFilterAll(pole->Segment(i), fPole);
      }
    } while (pole->NumSegments() > next);
  }
  justUnghosted = true;
}

// PoleRiders::Player::UpdateGhostTimer(float) @0x1001303fe
void Player::UpdateGhostTimer(float dt) {
  justUnghosted = false;
  float t = ghostTimer;
  if (!(t > 0.0f) || !onPole) return;
  t = t - dt;
  ghostTimer = t;
  if (0.0f >= t) SetGhost(false, -1.0f);
}

// PoleRiders::Player::UnsharePole() @0x100130448
void Player::UnsharePole() {
  sharingPole = false;
  SetGhost(true, 0x1.99999ap-5f);  // 0.05f
}

// PoleRiders::Player::Teleport(float, bool) @0x100130464
void Player::Teleport(float dx, bool withPole) {
  {
    b2Vec2 p(body->GetPosition().x + dx, body->GetPosition().y);
    body->SetTransform(p, body->GetAngle());
  }
  if (withPole) {
    for (int i = 0; i < pole->NumSegments(); i++) {
      b2Body *s = pole->Segment(i);
      b2Vec2 p(s->GetPosition().x + dx, s->GetPosition().y);
      s->SetTransform(p, s->GetAngle());
    }
  }
  if (pole->isHammer) {
    b2Body *h = pole->Hammer();
    b2Vec2 p(dx + h->GetPosition().x, h->GetPosition().y);
    h->SetTransform(p, h->GetAngle());
  }
}

// PoleRiders::Player::_OffPole() @0x100130546
void Player::_OffPole() {
  if (!onPole) return;
  rumbleB = fminf(rumbleB + 0x1.ccccccp-1f, 1.0f);  // +0.9f   (rumble: presentation)
  rumbleA = rumbleA + 0x1.ccccccp-2f;               // +0.45f
  onPole = false;
  world->DestroyJoint(handJoint0);
  world->DestroyJoint(handJoint1);
  pole->Detach(!sharingPole);
  b2Filter f = MakeFilter(0x80, 1, 0);
  SetFilterAll(body, f);
  hullFixture->SetFilterData(f);
  f.maskBits = 0;
  footFixture->SetFilterData(f);
  body->DestroyFixture(footFixture);  // NB: footFixture stays a dangling pointer (as in the game)
  hullFixture->SetRestitution(0x1.99999ap-3f);  // 0x3e4ccccd = 0.2f
  if (onGround) body->ApplyLinearImpulse(b2Vec2(0.0f, -7.0f), body->GetWorldCenter());  // inlined in the game
  FlagAllContactsForFiltering(world);
}

// PoleRiders::Player::_FootBoxTo(float, float) @0x100130718
// Moves the foot circle (footFixture's b2CircleShape::m_p) unless it would overlap an opponent's
// body that is not clearly above us. NB: after _OffPole the fixture is destroyed and this reads /
// writes the freed block-allocator memory exactly like the game does.
void Player::_FootBoxTo(float x, float y) {
  float fx = x + -0x1.99999ap-4f;  // x + (-0.1f)
  if (!facingLeft) fx = -fx;
  float r = Assets::ratio;
  b2CircleShape *shape = static_cast<b2CircleShape *>(footFixture->GetShape());
  b2Body *b = world->GetBodyList();
  if (b) {
    double thr = 32.0 / (double)r;
    bool move = true;
    for (; b; b = b->GetNext()) {
      const int *ud = static_cast<const int *>(b->GetUserData());
      if (!ud) continue;
      if (*ud > 3) continue;
      if (*ud == index) continue;
      for (b2Fixture *f = b->GetFixtureList(); f; f = f->GetNext()) {
        if (b2TestOverlap(f->GetShape(), footFixture->GetShape(), b->GetTransform(), body->GetTransform())) {
          if (*ud % 2 == index % 2 || onGround)
            move = false;
          else
            move = (double)b->GetPosition().y > (double)body->GetPosition().y + thr;
        }
      }
    }
    if (!move) return;
  }
  float s = 48.0f / r;
  shape->m_p.x = fx * s;
  shape->m_p.y = s * y * 0x1.666666p-1f + -0x1.99999ap-4f;  // ((s*y)*0.7f) + (-0.1f)
}

// ------------------------------------------------------------------------------------------------
// kick / land / lift-off
// ------------------------------------------------------------------------------------------------

// PoleRiders::Player::Kick(b2Vec2) @0x1001308c2
void Player::Kick(b2Vec2 from) {
  _FootBoxTo(0.0f, 0.0f);
  float dx = body->GetPosition().x - from.x;
  float dy = body->GetPosition().y - from.y;
  float len = sqrtf(dy * dy + dx * dx);
  if (!(0x1p-23f > len)) {  // b2Vec2::Normalize (b2_epsilon)
    float inv = 1.0f / len;
    dy = dy * inv;
    dx = dx * inv;
  }
  if (!facingLeft) dx = -dx;
  float a = atan2f(dy, dx) + -kHalfPi;
  while (0.0f > a) a = a + kTwoPi;
  while (a > kTwoPi) a = a + -kTwoPi;
  double d = (double)a;
  using namespace Anims;
  const AnimationData *hold, *kick;
  if (0x1.2d97c88p+0 >= d) {  // 1.1781 (3pi/8)
    hold = &kickAnimNE_hold, kick = &kickAnimNE;
  } else if (0x1.f6a7a38p+0 >= d) {  // 1.9635
    hold = &kickAnimE_hold, kick = &kickAnimE;
  } else if (0x1.5fdbbf4p+1 >= d) {  // 2.74889
    hold = &kickAnimSE_hold, kick = &kickAnimSE;
  } else if (0x1.c463accp+1 >= d) {  // 3.53429
    anim->Play(&kickAnimS);
    kicking = true;
    return;
  } else if (0x1.1475cd2p+2 >= d) {  // 4.31969
    hold = &kickAnimSW_hold, kick = &kickAnimSW;
  } else if (0x1.46b9c3ep+2 >= d) {  // 5.10509
    hold = &kickAnimW_hold, kick = &kickAnimW;
  } else {
    hold = &kickAnimNW_hold, kick = &kickAnimNW;
  }
  if (anim->GetData() == hold) return;
  anim->Play(kick);
  kicking = true;
}

// PoleRiders::Player::Land(bool) @0x100130b08
void Player::Land(bool onPoleContact) {
  standingOnPole = onPoleContact;
  kicking = false;
  int n = groundContacts;
  if (n < 0) {
    groundContacts = 0;
    n = 0;
  }
  if (n == 0) {
    if (onPole) {
      if (!onGround) {
        if (anim->GetData() != &Anims::landingAnim) anim->Play(&Anims::landingAnim);
      }
      _FootBoxTo(0.0f, 0.0f);
      onGround = true;
      n = groundContacts;
    }
    onGround = true;
  }
  groundContacts = n + 1;
  ballNoCollideTimer = 0.0f;
  _SetBallCollisionFilters();
}

// PoleRiders::Player::LiftOff(bool) @0x10013231e
void Player::LiftOff(bool) {
  int n = groundContacts - 1;
  groundContacts = n;
  if (n > 0) return;
  canPlayLandSound = true;
  stats->AddJump();
  groundContacts = 0;
  standingOnPole = false;
  if (!onPole) return;
  onGround = false;
  anim->Play(&Anims::liftoffAnim);
  // if (poleGroundContacts > 0) random() % 2 picks a jump sound: presentation (RNG never matters)
  liftoffFlag = true;
}

// ------------------------------------------------------------------------------------------------
// per tick
// ------------------------------------------------------------------------------------------------

// PoleRiders::Player::Update(float) @0x100130d10 (inlines _UpdateSticks @0x100130bee and
// _ParalyzeBody @0x10012c374)
void Player::Update(float dt) {
  using namespace Anims;

  // rumble fade (presentation)
  rumbleA = fmaxf(rumbleA - dt, 0.0f);
  rumbleB = fmaxf(rumbleB * 0x1.e66666p-1f, 0x1.99999ap-4f);  // *0.95f, floor 0.1f
  // Common::Input::VirtualUpdateVibration: rumble output into Input::_pads (+0xf4/+0xf8), which the
  // trace diff compares byte-exactly
  if (rumbleA > 0.0f && sim::g_engine->cfg.shouldRumble) {
    float rb = rumbleB;
    Common::Input::VirtualUpdateVibration(rb > 0.5f ? -0.5f + rb : 0.0f, rb, session);
  } else {
    Common::Input::VirtualUpdateVibration(0.0f, 0.0f, session);
  }

  // pole-bend sound (presentation)
  {
    int n = pole->NumSegments();
    float a = pole->Segment(n - 1)->GetAngle();
    float bend = a - pole->Segment(0)->GetAngle();
    bendVolume = fminf(0x1.99999ap-1f, (float)((double)fabsf(bend) * 0x1.3333333333333p-2 + 0x1.3333333333333p-2));
    // bendSoundA/B SetVolume(bendVolume), SetFrequency((float)(0.15*bend*bend + 0.9)): presentation
  }

  // _UpdateSticks()
  if (active) {
    Common::Input::Stick ls = Common::Input::VirtualLeftStick(session);
    leftStick[0] = ls.v[0], leftStick[1] = ls.v[1], leftStick[2] = ls.v[2], leftStick[3] = ls.v[3];
    Common::Input::Stick rs = Common::Input::VirtualRightStick(session);
    rightStick[0] = rs.v[0], rightStick[1] = rs.v[1], rightStick[2] = rs.v[2], rightStick[3] = rs.v[3];
  } else {
    for (int i = 0; i < 4; i++) leftStick[i] = 0.0f, rightStick[i] = 0.0f;
  }

  if (offPolePending) {
    _OffPole();
    offPolePending = false;
  }

  // paralyze fade-in (presentation)
  {
    float pf = paralyzeFade;
    if (pf >= 0.0f && 1.0f > pf) paralyzeFade = pf + 0.5f * dt;
  }

  // walk sound volume (presentation)
  if (onGround && onPole && (double)fabsf(leftStick[0]) > 0x1.999999999999ap-3) {
    float vx = body->GetLinearVelocity().x;
    float t = vx * vx / 300.0f * Assets::ratio;
    float wv = walkVolume;
    float add = fminf(0x1.333334p-2f, t) * 0x1.99999ap-4f;  // fminf(0.3f, t) * 0.1f
    walkVolume = wv * 0x1.ccccccp-1f + add;                  // wv*0.9f + add
  } else if (walkVolume > 0.0f) {
    walkVolume = walkVolume * 0x1.ccccccp-1f;
  }

  // rainbow flare flag (only feeds PlayState's Flare particles)
  if (CurrentLevel() == 2 && bodyActive) {
    float r = Assets::ratio;
    const b2Vec2 &p = body->GetPosition();
    if (550.0f / r > fabsf(p.x) && emitDust && p.y > 20.0f / r)
      rainbowFlare = true;
    else
      rainbowFlare = false;
  }

  // boost (b2Body::ApplyForce inlined)
  {
    float bt = boostTimer;
    if (bt > 0.0f) {
      if (body->GetType() == b2_dynamicBody) {
        float f = boostStrength * bt;
        f = f * bt;
        f = f * 100.0f;
        f = f / Assets::ratio;
        body->ApplyForce(b2Vec2(f, 0.0f), body->GetWorldCenter());
        bt = boostTimer;
      }
      boostTimer = bt - dt;
    }
  }

  if (onPole) {
    if (onGround) {
      // ---- riding on the ground ----
      _FootBoxTo(0.0f, 0.0f);
      float lx = leftStick[0];
      if (lx > 0x1.99999ap-3f || -0x1.99999ap-3f > lx) {  // |lx| > 0.2f
        // (slide sound / dust particles: presentation)
        if (body->GetType() == b2_dynamicBody) {
          float rs = runSpeed;
          float fx = lx * -groundNormal.y;
          fx = fx * rs;
          float fy = groundNormal.x * lx;
          fy = fy * rs;
          body->ApplyForce(b2Vec2(fx, fy), body->GetWorldCenter());
        }
        if (anim->GetData() != &turnAnim) {
          anim->Play(&walkAnim);
          anim->SetRate(fmaxf(fabsf(body->GetLinearVelocity().x) / 12.0f, 0x1.99999ap-4f));
        }
      } else {
        // (|ratio*vx| > 26: slide sound / dust: presentation)
        if (anim->GetData() != &landingAnim && anim->GetData() != &turnAnim) anim->Play(&idleAnim);
      }
    } else if (!kicking) {
      // ---- airborne on the pole: aim the kick with the left stick ----
      float lx = leftStick[0];
      float ly = leftStick[1];
      float len = sqrtf(lx * lx + ly * ly);
      if (len > 0x1.99999ap-1f) {  // 0.8f
        if (!(0x1p-23f > len)) {
          float inv = 1.0f / len;
          ly = ly * inv;
          lx = lx * inv;
        }
        float ang = facingLeft ? atan2f(-ly, lx) : atan2f(-ly, -lx);
        const AnimationData *d = anim->GetData();
        bool wasHold = d == &kickAnimSW_hold || d == &kickAnimSE_hold || d == &kickAnimE_hold ||
                       d == &kickAnimNE_hold || d == &kickAnimNW_hold || d == &kickAnimW_hold;
        float pos = anim->GetPosition();
        float bx, by;
        if (-0x1.5fdbc0p+1f > ang) {  // -2.7488935 (-7pi/8)
          anim->Play(&kickAnimW_hold), bx = -1.0f, by = 0.0f;
        } else if (-kHalfPi > ang) {
          anim->Play(&kickAnimNW_hold), bx = -1.0f, by = -1.0f;
        } else if (-0x1.921fb6p-2f > ang) {  // -pi/8
          anim->Play(&kickAnimNE_hold), bx = 1.0f, by = -1.0f;
        } else if (0x1.921fb6p-2f > ang) {
          anim->Play(&kickAnimE_hold), bx = 1.0f, by = 0.0f;
        } else if (kHalfPi > ang) {
          anim->Play(&kickAnimSE_hold), bx = 1.0f, by = 1.0f;
        } else if (0x1.5fdbc0p+1f > ang) {
          anim->Play(&kickAnimSW_hold), bx = -1.0f, by = 1.0f;
        } else {
          anim->Play(&kickAnimW_hold), bx = -1.0f, by = 0.0f;
        }
        _FootBoxTo(bx, by);
        if (wasHold && anim->GetData() != d) anim->SetPosition(pos);
        liftoffFlag = false;
      } else {
        anim->Play(&poleIdleAnim);
        _FootBoxTo(0.0f, 0.0f);
        liftoffFlag = true;
      }
    }
  } else {
    // ---- knocked off the pole ----
    if (onGround || anim->GetData() == &fellAnim) {
      if (0.0f > groundNormal.y)
        anim->Play(&fellAnim);
      else
        anim->Play(&fallingAnim);
    } else {
      anim->Play(&fallingAnim);
    }
    if (onGround) {
      const b2Vec2 &v = body->GetLinearVelocity();
      float s = v.x * v.x + v.y * v.y;
      if (0x1.eb851ep-6f > s && bodyActive) {  // 0.03f; _ParalyzeBody()
        bodyActive = false;
        paralyzeFade = 0.0f;
        emitDust = false;
        // slideSound->SetVolume(0): presentation
      }
    }
    // (onGround && bodyActive && |vx| > 0 && emitDust && !standingOnPole: dust particles, presentation)
  }

  // ---- pole servo: hand joints drive the handle segments toward the right-stick angle ----
  if (onPole) {
    float sx, sy;
    bool reversed;
    if (boostTimer > 0.0f) {
      sx = (team & 1) == 0 ? 1.0f : -1.0f;
      reversed = reversedGrip;
      if (reversed) sx = -sx;
      sy = 0.0f;
    } else {
      reversed = reversedGrip;
      sy = rightStick[1];
      sx = rightStick[0];
    }
    float target = atan2f(sy, sx);
    float a0, a2, w0, w2;
    if (!reversed) {
      a0 = -pole->Segment(0)->GetAngle();
      a2 = -pole->Segment(2)->GetAngle();
      w0 = pole->Segment(0)->GetAngularVelocity();
      w2 = pole->Segment(2)->GetAngularVelocity();
    } else {
      a0 = fmodf(kPi - pole->Segment(pole->NumSegments() - 3)->GetAngle(), kTwoPi);
      a2 = fmodf(kPi - pole->Segment(pole->NumSegments() - 1)->GetAngle(), kTwoPi);
      w0 = pole->Segment(pole->NumSegments() - 3)->GetAngularVelocity();
      w2 = pole->Segment(pole->NumSegments() - 1)->GetAngularVelocity();
    }
    w0 = -w0;
    w2 = -w2;
    auto wrap = [](float v) {  // fmodf(v, 2pi), sign bit set -> += 2pi
      float m = fmodf(v, kTwoPi);
      if (std::signbit(m)) m = m + kTwoPi;
      return m;
    };
    float e0, e2;
    {
      float m = wrap(a0);
      float t = wrap(target);
      e0 = wrap(m - t);
      if (e0 > kPi) e0 = e0 + -kTwoPi;
    }
    {
      float m = wrap(a2);
      float t = wrap(target);
      e2 = wrap(m - t);
      if (e2 > kPi) e2 = e2 + -kTwoPi;
    }
    float mag = sqrtf(sx * sx + sy * sy);
    if (mag > 0x1.99999ap-3f) {  // 0.2f
      float kd, kp;
      if (pole->isHammer)
        kd = 0x1.0624dep-8f, kp = 0x1.47ae14p-5f;  // 0.004f, 0.04f
      else if (!shortPole)
        kd = 0x1.0624dep-8f, kp = 0x1.eb851ep-5f;  // 0.004f, 0.06f
      else
        kd = 0x1.cac084p-7f, kp = 0x1.1eb852p-4f;  // 0.014f, 0.07f
      float u0 = e0 * kp + w0 * kd;
      float u2 = e2 * kp + w2 * kd;
      float c0 = (0.0f > u0) ? fmaxf(-0x1.99999ap-2f, u0) : fminf(0x1.99999ap-2f, u0);  // clamp +-0.4f
      float c2 = (0.0f > u2) ? fmaxf(-0x1.99999ap-2f, u2) : fminf(0x1.99999ap-2f, u2);
      // (the game re-tests mag > 0.2f here; always true)
      float ssf = Assets::spinSpeedFactor;
      handJoint0->SetMotorSpeed(c0 * ssf * 4.0f);
      float pst = Assets::poleSwingTorque;
      handJoint0->SetMaxMotorTorque(fabsf(c0) * pst * 4.0f);
      handJoint1->SetMotorSpeed(ssf * c2 * 4.0f);
      handJoint1->SetMaxMotorTorque(fabsf(c2) * pst * 4.0f);
    } else {
      handJoint0->SetMotorSpeed(0.0f);
      handJoint0->SetMaxMotorTorque(0.0f);
      handJoint1->SetMotorSpeed(0.0f);
      handJoint1->SetMaxMotorTorque(0.0f);
    }
  }

  // ---- facing: turn toward the far end of the pole (only while standing) ----
  {
    b2Body *far = reversedGrip ? pole->Segment(1) : pole->Segment(pole->NumSegments() - 1);
    if (far->GetPosition().x > body->GetPosition().x) {
      if (onGround) {
        if (facingLeft) anim->Play(&turnAnim);
        facingLeft = false;
      }
    } else if (onGround) {
      if (!facingLeft) anim->Play(&turnAnim);
      facingLeft = true;
    }
  }

  // ---- ball no-collide timer ----
  {
    float t = ballNoCollideTimer;
    if (t > 0.0f) {
      t = t - dt;
      ballNoCollideTimer = t;
      if (0.0f > t) {
        ballNoCollideTimer = 0.0f;
        _SetBallCollisionFilters();
      }
    }
  }

  // ---- animation ----
  anim->Update(dt);
  if (anim->IsFinished()) {
    const AnimationData *d = anim->GetData();
    if (d == &turnAnim) anim->Play(&idleAnim);
    if (d == &landingAnim) anim->Play(&idleAnim);
    if (d == &kickAnimW || d == &kickAnimSW || d == &kickAnimNW || d == &kickAnimE || d == &kickAnimSE ||
        d == &kickAnimNE || d == &kickAnimS) {
      anim->Play(&poleIdleAnim);
      kicking = false;
    }
  }

  // ---- timers ----
  if (tapTimer > 0.0f) {  // presentation
    float t = tapTimer + dt / -0x1.99999ap-3f;  // dt / -0.2f
    tapTimer = t;
    if (0.0f > t) tapTimer = 0.0f;
  }
  tauntFlag = false;
  if (tauntTimer > 0.0f) {
    float t = tauntTimer - dt;
    tauntTimer = t;
    if (0.0f > t) tauntTimer = 0.0f;
  }
  if (fartTimer > 0.0f) {  // presentation
    float t = fartTimer + dt / -0x1.51eb86p-1f;  // dt / -0.66f
    fartTimer = t;
    if (0.0f > t) fartTimer = 0.0f;
  }
  if (smashTimer > 0.0f) {  // presentation
    float t = smashTimer + dt / -0x1.99999ap-5f;  // dt / -0.05f
    smashTimer = t;
    if (0.0f > t) smashTimer = 0.0f;
  }
  if (smashAnim) smashAnim->Update(dt);  // presentation (ShowSmash)
  {
    float c = dt + animClock;  // presentation
    animClock = c;
    if (c > 200.0f) {
      do c = c + -200.0f;
      while (c > 200.0f);
      animClock = c;
    }
  }
}

// PoleRiders::Player::UpdateNormal(b2Contact*) @0x1001321ee
void Player::UpdateNormal(b2Contact *c) {
  if (!onPole) return;
  b2Fixture *fa = c->GetFixtureA();
  b2Fixture *fb = c->GetFixtureB();
  b2WorldManifold wm;
  wm.Initialize(c->GetManifold(), fa->GetBody()->GetTransform(), fa->GetShape()->m_radius,
                fb->GetBody()->GetTransform(), fb->GetShape()->m_radius);
  if (c->GetManifold()->pointCount > 0) {
    if (wm.normal.y > 0.0f) {
      groundNormal.x = -wm.normal.x;
      groundNormal.y = -wm.normal.y;
    } else {
      groundNormal.x = wm.normal.x;
      groundNormal.y = wm.normal.y;
    }
  } else {
    groundNormal.x = 0.0f;
    groundNormal.y = -1.0f;
  }
}

// PoleRiders::Player::Respawn(float, float, bool, Pole*) @0x100132e7a
// Respawn onto an existing pole (the team-mate's, far end, reversed grip): shared pole.
void Player::Respawn(float x, float y, bool, Pole *newPole) {
  SpawnPlayerBody(x, y);
  if (pole) delete pole;  // ~Pole + operator delete
  pole = newPole;
  sharingPole = true;
  reversedGrip = true;
  shortPole = false;
  b2Body *segA = pole->Segment(pole->NumSegments() - 2);  // grip height reference
  b2Body *segB = pole->Segment(pole->NumSegments() - 1);
  b2Body *segC = pole->Segment(pole->NumSegments() - 3);

  b2RevoluteJointDef jd;  // type revolute, everything else zero / false
  jd.bodyA = body;
  jd.bodyB = segB;
  {
    // localAnchorA = body->GetLocalPoint(b2Vec2(body.x, segA.y)) (b2MulT(R, p - pos), inlined)
    const b2Transform &xf = body->GetTransform();
    float dy = segA->GetPosition().y - xf.position.y;
    float dx = xf.position.x - xf.position.x;
    jd.localAnchorA.x = dx * xf.R.col1.x + dy * xf.R.col1.y;
    jd.localAnchorA.y = dx * xf.R.col2.x + dy * xf.R.col2.y;
  }
  const float r = Assets::ratio;
  const float off = -10.0f / r;
  auto anchorB = [&](b2Body *seg) {  // seg->GetLocalPoint(b2Vec2(body.x, segA.y)) + (-10/ratio, 0)
    const b2Transform &xf = seg->GetTransform();
    float dy = segA->GetPosition().y - xf.position.y;
    float dx = body->GetPosition().x - xf.position.x;
    float lx = dx * xf.R.col1.x + dy * xf.R.col1.y;
    float ly = dx * xf.R.col2.x + dy * xf.R.col2.y;
    jd.localAnchorB.x = lx + off;
    jd.localAnchorB.y = ly + 0.0f;
  };
  anchorB(segB);
  jd.referenceAngle = team == 0 ? 0.0f : kPi;
  jd.enableMotor = true;
  jd.maxMotorTorque = Assets::poleSwingTorque;
  jd.motorSpeed = 0.0f;
  handJoint0 = static_cast<b2RevoluteJoint *>(world->CreateJoint(&jd));
  jd.bodyB = segC;
  anchorB(segC);
  handJoint1 = static_cast<b2RevoluteJoint *>(world->CreateJoint(&jd));

  walkVolume = 0.0f;
  // walkSound / slideSound SetVolume(0): presentation
  canPlayLandSound = false;
  canPlayHammerSound = true;
  canPlayDeathSound = true;
  LiftOff(false);
  runSpeed = 2100.0f / r;
  offPolePending = false;
  unsharePending = false;
  tapTimer = 0.0f;
  tauntTimer = 0.0f;
  tauntFlag = false;
  fartTimer = 0.0f;
  standingOnPole = false;
  rainbowFlare = false;
  drawHand1 = b2Vec2(0.0f, 0.0f);
  drawHand0 = b2Vec2(0.0f, 0.0f);
}

}  // namespace PoleRiders
