// Player construction / spawning / teardown (package A): Player ctor & dtor, SpawnPlayerBody,
// Respawn with a new pole; Stats ctor & dtor. Per-tick logic is package B (player.cpp).
#include <cstring>
#include "animation_data.h"
#include "assets.h"
#include "engine.h"

namespace PoleRiders {

// PoleRiders::Stats::Stats(int) @0x100141144
Stats::Stats(int player) {
  playerNum = player;
  kicks = 0;
  jumps = 0;
  suicides = 0;
  timeLeft = -1.0f;
  timeInGoal = 0.0f, timeInEndzone = 0.0f;
  goals = 0, ownGoals = 0, offenseRatio = 0.0f;
  deaths = 0, timeNearestBall = 0.0f;
  kills = 0;
}

// PoleRiders::Stats::~Stats() @0x10014118e
Stats::~Stats() {}

// PoleRiders::Player::Player(b2World*, int, int, int, int, SpriteSheet*, Dust*) @0x100133582
//   Player(world, sessionSlot, index, team, unused, sharedSprites, dust)
// Only the gameplay field init is ported: sprite sheets and sounds (presentation) stay null, and the
// srandom(Time_GetHighResTimer()) reseed is skipped (random() never feeds gameplay).
Player::Player(b2World *w, int slot, int idx, int teamArg, int, SpriteSheet *sprites, Dust *dustArg) {
  memset(this, 0, sizeof *this);  // operator new'd in the game; fields it never writes read as 0
  session.mode = 0;
  session.team = 0;
  session.pad = -1;
  session.head = -1;
  session.unk10 = 0;
  session.unk14 = 7;
  session.unk18 = 0;
  session = sim::g_engine->session[slot];  // Common::Session::Get()[slot]
  world = w;
  index = idx;
  partnerIndex = (idx + 2) % 4;  // C remainder, as compiled
  team = teamArg;
  sharedSprites = sprites;
  dust = dustArg;
  ghostTimer = -1.0f;
  unk2c = 0;
  rainbowFlare = false;
  drawFlag21c = false;
  inputId = session.pad;
  controllerNum = Common::Input::GetControllerNum(slot);
  pole = nullptr;
  body = nullptr;
  anim = nullptr;
  smashAnim = nullptr;
  stats = new Stats(index);
}

// PoleRiders::Player::~Player() @0x100132406 (D2; D1 thunk @0x1001326e8)
Player::~Player() {
  world->DestroyBody(body);
  if (!sharingPole && pole) {
    pole->~Pole();
    operator delete(pole);
  }
  delete anim;
  delete smashAnim;
  // (sprite sheets and sounds: presentation, never created in the sim)
  if (stats) {
    stats->~Stats();
    operator delete(stats);
  }
}

// PoleRiders::Player::SpawnPlayerBody(float, float) @0x1001326f2
// (Re)creates the rider body at (x, y - 50) px with its four fixtures and a fresh walk animation.
void Player::SpawnPlayerBody(float x, float y) {
  groundContacts = 0;
  memset(rightStick, 0, sizeof rightStick);
  memset(leftStick, 0, sizeof leftStick);
  animClock = 0.0f;
  poleGroundContacts = 0;
  emitDust = true;
  boostStrength = 0.0f;
  boostTimer = 0.0f;
  bodyActive = true;
  ghost = false;
  onPole = true;
  paralyzeFade = -1.0f;
  onGround = true;
  rainbowFlare = false;
  liftoffFlag = false;
  kicking = false;
  active = true;
  ballNoCollideTimer = 0.0f;
  rumbleA = 0.0f;
  rumbleB = 0.0f;
  bendVolume = 0.0f;
  groundNormal.x = 0.0f;
  groundNormal.y = -1.0f;
  facingLeft = team != 0;

  b2BodyDef bd;  // -0x70
  const float r = Assets::ratio;

  // hull: 9-vertex rounded box (-0x168)
  b2Vec2 v[9];
  {
    float a = 16.0f / r, b = -52.0f / r;
    v[0].Set(a, b);
    v[1].x = a;
    float c = 28.0f / r;
    v[1].y = c;
    float d = 0x1.bb6c8cp+3f / r;  // 13.857f
    v[2].x = d;
    float e = 0x1.1f3b64p+5f / r;  // 35.904f
    v[2].y = e;
    v[3].x = 8.0f / r;
    float f = 0x1.4d851ep+5f / r;  // 41.69f
    v[3].y = f;
    v[4].x = 0.0f;
    v[4].y = 0x1.5e76c8p+5f / r;   // 43.808f
    v[5].x = -8.0f / r;
    v[5].y = f;
    v[6].x = -0x1.bb6c8cp+3f / r;
    v[6].y = e;
    float g = -16.0f / r;
    v[7].Set(g, c);
    v[8].Set(g, b);
  }
  b2PolygonShape hull;  // -0x120
  hull.Set(v, 9);

  uint16_t mask2, mask, cat;  // r14w (foot), r15w, r12w
  if (team == 0) {
    mask2 = 0xfe24, mask = 0xfe35, cat = 2;
  } else if (team == 1) {
    mask2 = 0xfe22, mask = 0xfe2b, cat = 4;
  } else {
    mask = 0xffff, cat = 1, mask2 = mask;
  }

  // impale sensor: thin box above the head (-0x218)
  b2PolygonShape spike;
  spike.SetAsBox(2.0f / r, 20.0f / r, b2Vec2(0.0f, -10.0f / r), 0.0f);
  b2FixtureDef fdImpale;  // -0x248
  fdImpale.userData = const_cast<int *>(&Assets::FIXTURE_IMPALE);
  fdImpale.shape = &spike;
  fdImpale.density = 0x1.0624dep-10f;  // 0x3a83126f = 0.001
  fdImpale.friction = 0x1.99999ap-1f;  // 0x3f4ccccd = 0.8
  fdImpale.restitution = 0.0f;
  fdImpale.filter.categoryBits = cat;
  fdImpale.filter.maskBits = mask;
  fdImpale.filter.groupIndex = 0;
  fdImpale.isSensor = true;

  bd.position.x = x / r;
  bd.position.y = (y + -50.0f) / r;
  bd.fixedRotation = true;
  bd.linearDamping = Assets::linearDamping;
  bd.type = b2_dynamicBody;
  bd.bullet = false;

  b2FixtureDef fdHull;  // -0x278
  fdHull.userData = nullptr;
  fdHull.isSensor = false;
  fdHull.density = 0x1.7286bep-1f;   // 0x3f39435f = 0.723684
  fdHull.friction = 0x1.99999ap-1f;  // 0.8
  fdHull.restitution = 0.0f;
  fdHull.filter.categoryBits = cat;
  fdHull.filter.maskBits = mask;
  fdHull.filter.groupIndex = 0;
  fdHull.shape = &hull;

  // ground sensor circle below the hull (-0x290)
  b2CircleShape sensor;
  const float radius = 48.0f / (3.0f * r);
  sensor.m_radius = radius;
  sensor.m_p.x = 0.0f;
  sensor.m_p.y = 100.0f / (r + r);
  b2FixtureDef fdSensor;  // -0x2c0
  fdSensor.density = 0x1.19999ap-1f;   // 0x3f0ccccd = 0.55
  fdSensor.friction = 0x1.99999ap-1f;  // 0.8
  fdSensor.restitution = 0.0f;
  fdSensor.filter.categoryBits = cat;
  fdSensor.filter.maskBits = mask;
  fdSensor.filter.groupIndex = 0;
  fdSensor.shape = &sensor;
  fdSensor.isSensor = true;
  fdSensor.userData = const_cast<int *>(&Assets::FIXTURE_SENSOR);

  // foot / kick circle at the body origin (-0x2d8; moved later by _FootBoxTo)
  b2CircleShape foot;
  foot.m_radius = radius;
  foot.m_p.SetZero();
  b2FixtureDef fdFoot;  // -0x308
  fdFoot.userData = nullptr;
  fdFoot.isSensor = false;
  fdFoot.density = 0x1.19999ap-3f;     // 0x3e0ccccd = 0.1375
  fdFoot.friction = 0x1.99999ap-1f;    // 0.8
  fdFoot.restitution = 0.5f;
  fdFoot.filter.categoryBits = cat;
  fdFoot.filter.maskBits = mask2;
  fdFoot.filter.groupIndex = 0;
  fdFoot.shape = &foot;

  if (body) world->DestroyBody(body);
  body = world->CreateBody(&bd);
  hullFixture = body->CreateFixture(&fdHull);
  sensorFixture = body->CreateFixture(&fdSensor);
  footFixture = body->CreateFixture(&fdFoot);
  impaleFixture = body->CreateFixture(&fdImpale);
  body->SetUserData(&index);

  delete anim;
  delete smashAnim;
  anim = new Animation();
  anim->Play(&Anims::walkAnim);
  smashAnim = new Animation();
  memset(unk20c, 0, sizeof unk20c);
}

// PoleRiders::Player::Respawn(float, float, bool, float, bool, bool) @0x1001331a2
//   Respawn(x px, y px, hammer, poleLengthPx, sharedPole, shortPole): new body + a NEW pole held by
//   two motorized revolute "hands" (segment 0 and 2), then LiftOff(false) and SetGhost(true, 0.3).
void Player::Respawn(float x, float y, bool hammer, float poleLength, bool sharedArg, bool shortArg) {
  SpawnPlayerBody(x, y);
  if (pole) {
    pole->~Pole();
    operator delete(pole);
  }
  sharingPole = sharedArg;
  reversedGrip = false;
  const float r = Assets::ratio;
  const float bx = body->GetPosition().x * r;
  const float by = r * body->GetPosition().y;
  static const float kAngle[2] = {0x1.921fb6p+1f, 0.0f};  // @0x10028e810: [team == 0]
  float poleAngle = kAngle[team == 0];
  if (sim::g_engine && sim::g_engine->spawnPoleUp)  // training-only pit drop: pole straight up, i.e.
    poleAngle += team == 0 ? -0x1.921fb6p+0f : 0x1.921fb6p+0f;  // raised from its forward rest angle
  else if (sim::g_engine && sim::g_engine->randomPole) {  // training-only: a random angle from forward to
    uint64_t z = (sim::g_engine->poleRng += 0x9e3779b97f4a7c15ULL);  // back-up (y down: up is negative for
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;                     // team 0, which attacks +x)
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    const float th = static_cast<float>(((z ^ (z >> 31)) >> 11) * 0x1.0p-53 * (150.0 / 180.0 * 3.141592653589793));
    poleAngle += team == 0 ? -th : th;
  }
  Pole *p = static_cast<Pole *>(operator new(sizeof(Pole)));
  new (p) Pole(bx, by + -24.0f, world, team, index, poleAngle, sharedSprites, hammer, poleLength,
               sharedArg, shortArg);
  pole = p;
  shortPole = shortArg;
  b2Body *s1 = pole->Segment(1);
  b2Body *s0 = pole->Segment(0);
  b2Body *s2 = pole->Segment(2);

  // hands at (body.x, segment1.y), in body and segment frames (inlined b2Body::GetLocalPoint)
  b2RevoluteJointDef jd;  // -0x78
  jd.userData = nullptr;
  jd.localAnchorA.x = 0.0f;
  jd.maxMotorTorque = 0.0f;
  jd.motorSpeed = 0.0f;
  jd.enableLimit = false;
  jd.localAnchorB.y = 0.0f, jd.referenceAngle = 0.0f;
  jd.localAnchorA.y = 0.0f, jd.localAnchorB.x = 0.0f;
  jd.enableMotor = false;
  jd.lowerAngle = 0.0f, jd.upperAngle = 0.0f;
  jd.collideConnected = false;
  jd.bodyA = body;
  jd.bodyB = s0;
  const b2Transform &tb = body->GetTransform();
  {
    float dy = s1->GetPosition().y - tb.position.y;
    float dx = tb.position.x - tb.position.x;  // sic: (body.x - body.x), as compiled
    jd.localAnchorA.x = tb.R.col1.x * dx + tb.R.col1.y * dy;
    jd.localAnchorA.y = dx * tb.R.col2.x + dy * tb.R.col2.y;
  }
  const float handOffset = -20.0f / r;
  {
    const b2Transform &t = s0->GetTransform();
    float dy = s1->GetPosition().y - t.position.y;
    float dx = tb.position.x - t.position.x;
    float lx = t.R.col1.x * dx + t.R.col1.y * dy;
    jd.localAnchorB.x = lx + handOffset;
    float ly = dx * t.R.col2.x + dy * t.R.col2.y;
    jd.localAnchorB.y = ly + 0.0f;
  }
  jd.referenceAngle = kAngle[team == 0];
  jd.enableMotor = true;
  jd.maxMotorTorque = Assets::poleSwingTorque;
  jd.motorSpeed = 0.0f;
  handJoint0 = static_cast<b2RevoluteJoint *>(world->CreateJoint(&jd));

  jd.bodyB = s2;
  {
    const b2Transform &t = s2->GetTransform();
    float dy = s1->GetPosition().y - t.position.y;
    float dx = body->GetPosition().x - t.position.x;
    float lx = t.R.col1.x * dx + t.R.col1.y * dy;
    jd.localAnchorB.x = lx + handOffset;
    float ly = dx * t.R.col2.x + dy * t.R.col2.y;
    jd.localAnchorB.y = ly + 0.0f;
  }
  handJoint1 = static_cast<b2RevoluteJoint *>(world->CreateJoint(&jd));

  walkVolume = 0.0f;
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
  smashTimer = 0.0f;
  rainbowFlare = false;
  drawHand1.SetZero();
  drawHand0.SetZero();
  SetGhost(true, 0x1.333334p-2f);  // 0.3f
}

}  // namespace PoleRiders
