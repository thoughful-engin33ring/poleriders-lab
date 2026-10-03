// PlayState contact listener and collision handlers (package C).
//
// Tags: every body/fixture userData is an int* (Assets::BODY_*, Player::index, Pole::ownerIndex);
// NULL decodes as 0 (= BODY_PLAYER1). Handlers are always called twice per contact with A/B swapped.
// Presentation (sounds, rumble, sparks, shockwave visuals, camera, dust, splash) is skipped; every
// gameplay field write and every call into gameplay code is kept, in order.
#include <cmath>
#include "assets.h"
#include "engine.h"
#include "layout.h"
#include "src/x86math.h"

namespace PoleRiders {

namespace {
// PlayState::ContactToInts (inlined into Begin/End/PostSolve): tag = ud ? *(int*)ud : 0
inline int TagOf(void *ud) { return ud ? *static_cast<int *>(ud) : 0; }

// b2Contact::FlagForFiltering() (m_flags |= e_filterFlag) is protected; the game inlines it.
struct ContactAccess : b2Contact {
  static constexpr uint32 b2Contact::*flags = &ContactAccess::m_flags;
  static constexpr uint32 filterFlag = b2Contact::e_filterFlag;
};
static_assert(ContactAccess::filterFlag == 0x8, "e_filterFlag");
inline void FlagForFiltering(b2Contact *c) { c->*ContactAccess::flags |= ContactAccess::filterFlag; }

// b2Vec2::Normalize as inlined by the game (sqrtss of x*x + y*y; no-op below FLT_EPSILON)
inline void Normalize(float &x, float &y) {
  const float len = sqrtf(x * x + y * y);
  if (!(0x1p-23f > len)) {  // ucomiss len, eps ; ja skip
    const float inv = 1.0f / len;
    y = y * inv;
    x = x * inv;
  }
}

// b2MulT(R, p - pos) of a body transform, in the game's operand order
inline void LocalPoint(const b2Body *h, b2Vec2 p, float &lx, float &ly) {
  const b2Transform &xf = h->GetTransform();
  const float px = p.x - xf.position.x;
  const float py = p.y - xf.position.y;
  ly = xf.R.col2.x * px + xf.R.col2.y * py;
  lx = xf.R.col1.x * px + xf.R.col1.y * py;
}
}  // namespace

// PoleRiders::PlayState::PreSolve(b2Contact*, b2Manifold const*) @0x10013544c (PreSolveInner is empty)
void PlayState::PreSolve(b2Contact *, const b2Manifold *) {}

// PoleRiders::PlayState::BeginContact(b2Contact*) @0x10013a93c
void PlayState::BeginContact(b2Contact *c) {
  b2Fixture *fa = c->GetFixtureA(), *fb = c->GetFixtureB();
  const int bodyA = TagOf(fa->GetBody()->GetUserData());
  const int bodyB = TagOf(fb->GetBody()->GetUserData());
  const int fixA = TagOf(fa->GetUserData());
  const int fixB = TagOf(fb->GetUserData());
  BeginContactInner(c, bodyA, fixA, bodyB, fixB);
  BeginContactInner(c, bodyB, fixB, bodyA, fixA);
}

// PoleRiders::PlayState::BeginContactInner(b2Contact*, int, int, int, int) @0x10013a5a4
// (bodyA, fixA, bodyB, fixB)
void PlayState::BeginContactInner(b2Contact *, int bodyA, int fixA, int bodyB, int fixB) {
  // foot sensor on ground / on a pole
  if (fixA == FIXTURE_SENSOR && bodyB == BODY_GROUND) players[bodyA]->Land(false);
  if (fixA == FIXTURE_SENSOR && bodyB == BODY_POLE) {
    if (!timerOn) timerOn = true;  // EnableTimer (inlined)
    players[bodyA]->Land(true);
  }
  if (bodyA == BODY_GROUND && bodyB == BODY_POLE) players[fixB]->PoleContactAdjust(1);
  if (bodyA == BODY_GROUND && bodyB == BODY_HAMMER) players[fixB]->HammerLand();
  if (bodyA == BODY_WALL && bodyB <= 3) players[bodyB]->WallCollision();
  if (bodyA == BODY_WALL && bodyB == BODY_BALL && 0.0f > endFade) BallWallCollision();
  if (bodyA == BODY_BALL) {
    if (!timerOn) timerOn = true;
    if (bodyB <= 3) lastTouch = bodyB;
    if ((bodyB == BODY_HAMMER || bodyB == BODY_POLE) && fixB <= 3) lastTouch = fixB;
  }
  if (bodyA == BODY_DELETER && bodyB <= 3) {
    const int i = bodyB;
    if (players[i]->OnPole() && !unlockingRainbow) players[i]->GetStats()->AddSuicide();
    players[i]->SetGhost(true, -1.0f);
    if (suddenDeath && score[0] == score[1]) {
      koWin = true;
      lastTouch = -1;
      _ScoreGoal((bodyB + 1) % 2);
    } else {
      const int lvl = level;
      b2Body *b = players[i]->Body();
      if (lvl == 5) {
        if (b->GetPosition().y > 0.0f)
          respawnTimer[i] = Assets::deathDelay;
        else
          respawnTimer[i] = 0x1.111112p-6f;  // 0x3c888889 = 1/60
      } else {
        bool unlock = false;
        if (b->GetPosition().x > 0.0f && 0.0f > players[i]->Body()->GetPosition().y && (i & 1) == 0) {
          unlock = true;
        } else if (0.0f > players[i]->Body()->GetPosition().x &&
                   0.0f > players[i]->Body()->GetPosition().y && i % 2 == 1) {
          unlock = true;
        }
        // Rainbow Highway unlock (the harness forces level5unlocked = 1, so this never runs).
        // The game also plays sndSecret and writes/saves Common::Save (+0x74b3c = 1): no files here.
        if (unlock && !sim::g_engine->cfg.level5unlocked && !unlockingRainbow) {
          unlockingRainbow = true;
          sim::g_engine->cfg.level5unlocked = true;
          sim::g_engine->cfg.level = 5;  // Assets::currentLevel = 5
          endFade = 0.0f;
        }
        if (!unlockingRainbow) respawnTimer[i] = 0x1.111112p-6f;
      }
    }
    players[i]->PlayDeathSound(0x1.333334p-1f);  // 0.6f (rumble only)
    const int n = numPlayers;
    if (n > 0) {
      bool allDead = true;  // everyone else already waiting to respawn -> respawn all next tick
      for (int k = 0; k < n; k++)
        if (k != bodyB && respawnTimer[k] <= 0.0f) allDead = false;  // ucomiss (t), 0 ; jb
      if (allDead) {
        int k = 0;
        do respawnTimer[k] = 0x1.111112p-6f;
        while (++k < numPlayers);
      }
    }
  }
  // (bodyA == BODY_SEA && bodyB <= 3) -> Splash(bodyB): Level3 water, presentation only (skipped)
}

// PoleRiders::PlayState::EndContact(b2Contact*) @0x100136348
void PlayState::EndContact(b2Contact *c) {
  b2Fixture *fa = c->GetFixtureA(), *fb = c->GetFixtureB();
  const int bodyA = TagOf(fa->GetBody()->GetUserData());
  const int bodyB = TagOf(fb->GetBody()->GetUserData());
  const int fixA = TagOf(fa->GetUserData());
  const int fixB = TagOf(fb->GetUserData());
  EndContactInner(bodyA, fixA, bodyB, fixB);
  EndContactInner(bodyB, fixB, bodyA, fixA);
}

// PoleRiders::PlayState::EndContactInner(int, int, int, int) @0x100136298 (bodyA, fixA, bodyB, fixB)
void PlayState::EndContactInner(int bodyA, int fixA, int bodyB, int fixB) {
  if (!(0.0f > endFade)) return;
  if (fixA == FIXTURE_SENSOR && (bodyB == BODY_GROUND || bodyB == BODY_POLE))
    players[bodyA]->LiftOff(bodyB == BODY_POLE);
  if (bodyA == BODY_GROUND && bodyB == BODY_POLE) players[fixB]->PoleContactAdjust(-1);
  if (bodyA == BODY_HAMMER && bodyB == BODY_GROUND) players[fixA]->HammerLiftoff();
}

// PoleRiders::PlayState::PostSolve(b2Contact*, b2ContactImpulse const*) @0x10013af72
void PlayState::PostSolve(b2Contact *c, const b2ContactImpulse *imp) {
  b2Fixture *fa = c->GetFixtureA(), *fb = c->GetFixtureB();
  const int bodyA = TagOf(fa->GetBody()->GetUserData());
  const int bodyB = TagOf(fb->GetBody()->GetUserData());
  const int fixA = TagOf(fa->GetUserData());
  const int fixB = TagOf(fb->GetUserData());
  PostSolveInner(c, imp, bodyA, fixA, bodyB, fixB);
  PostSolveInner(c, imp, bodyB, fixB, bodyA, fixA);
}

// PoleRiders::PlayState::PostSolveInner(b2Contact*, b2ContactImpulse const*, int, int, int, int)
// @0x10013ad2e (bodyA, fixA, bodyB, fixB)
void PlayState::PostSolveInner(b2Contact *c, const b2ContactImpulse *imp, int bodyA, int fixA, int bodyB,
                               int fixB) {
  if (!(0.0f > endFade)) return;
  // an enemy pole enters a rider's impale box -> rider ghosts
  if (fixB == FIXTURE_IMPALE && bodyA == BODY_POLE && fixA % 2 != bodyB % 2)
    players[bodyB]->SetGhost(true, 0.1f);
  if (bodyA < bodyB && bodyA <= 3 && bodyB <= 3 && bodyA % 2 != bodyB % 2) PlayerCollision(c, bodyA, bodyB);
  if (bodyA <= 3 && bodyB == BODY_BALL) BallPlayerCollision(imp, bodyA);
  // (BODY_BALL vs BODY_POLE) BallContact: PlayBallSound only; (BODY_POLE vs BODY_GROUND)
  // PoleGroundCollision: PlayTap only -- presentation, skipped
  if (bodyA == BODY_HAMMER && bodyB <= 3 && fixA <= 3) HammerPlayerCollision(c, imp, bodyB, fixA);
  if (bodyA == BODY_HAMMER && bodyB == BODY_BALL && fixA <= 3) HammerBallCollision(c, imp, fixA);
  if ((bodyA == BODY_HAMMER || bodyA < 3) && bodyB == BODY_GROUND) GroundContact(c, imp, bodyA, fixA);
  if (bodyA <= 3 && (bodyB == BODY_GROUND || bodyB == BODY_POLE)) players[bodyA]->UpdateNormal(c);
}

// PoleRiders::PlayState::PlayerCollision(b2Contact*, int, int) @0x100139592
// Rider vs rider of the other team (a < b). Both on poles and at least one still killable this step.
void PlayState::PlayerCollision(b2Contact *c, int a, int b) {
  if (!timerOn) timerOn = true;
  if (!c) return;
  b2WorldManifold wm;
  c->GetWorldManifold(&wm);
  b2Vec2 p;  // contact point: only feeds _ShockwaveAt's presentation part (uninitialized if no points)
  p.SetZero();
  if (c->GetManifold()->pointCount > 0) p = wm.points[0];

  const b2Vec2 pa = players[a]->Body()->GetPosition();
  const b2Vec2 pb = players[b]->Body()->GetPosition();
  float dy = pa.y - pb.y;
  float dx = pa.x - pb.x;
  Normalize(dx, dy);
  const float k = Assets::kickPlayerStrength / Assets::ratio;
  dy = dy * k;
  const float dxk = k * dx;
  const float angle = atan2f(dy, dxk);
  const int maskA = 1 << a;
  const int maskB = 1 << b;
  if (!players[a]->OnPole()) return;
  if (!players[b]->OnPole()) return;
  const int mask = maskB | maskA;
  if ((killMask & mask) == 0) return;

  const float ay_m = dy * -0.15f, ax_m = -0.15f * dxk;  // -0x70 / -0x6c
  const float ay_p = dy * 0.15f, ax_p = 0.15f * dxk;    // -0x78 / -0x74
  const float nx = -dxk, ny = -dy;                      // -0x80 / -0x7c

  bool killA;  // a dies (after b, if both)
  bool killB;
  if (fabsf(x86_cosf(angle)) > 0.5f) {
    // sideways: both must be airborne; the lower rider (larger y) dies
    if (players[a]->OnGround()) return;
    if (players[b]->OnGround()) return;
    const float ya = players[a]->Body()->GetPosition().y;
    const float yb = players[b]->Body()->GetPosition().y;
    if (yb > ya) {
      killB = true;
      killA = false;
    } else {
      killB = false;
      killA = true;
    }
  } else {
    // vertical (stomp) with a 32px tolerance: both can die
    const float off = -32.0f / Assets::ratio;
    const float ya = players[a]->Body()->GetPosition().y;
    const float ybOff = players[b]->Body()->GetPosition().y + off;
    const float yb = players[b]->Body()->GetPosition().y;
    const float yaOff = off + players[a]->Body()->GetPosition().y;
    killA = yaOff > yb;
    killB = ybOff > ya;
  }

  if (killB) {
    FlagForFiltering(c);
    players[b]->OffPole();
    players[b]->GetStats()->AddDeath();
    players[a]->GetStats()->AddKill();
    players[a]->GetStats()->AddKick();
    players[a]->Kick(players[b]->Body()->GetPosition());
    players[a]->PlayKickSound(0x1.666666p-1f);  // 0.7f (rumble only)
    if (!suddenDeath) respawnTimer[b] = Assets::deathDelay;
    killMask &= ~mask;
    _ShockwaveAt(0, p, players[a]->Body()->GetPosition());
    {
      b2Body *body = players[a]->Body();
      body->ApplyLinearImpulse(b2Vec2(ax_p, ay_p), body->GetWorldCenter());
    }
    {
      b2Body *body = players[b]->Body();
      body->ApplyLinearImpulse(b2Vec2(nx, ny), body->GetWorldCenter());
    }
    if (suddenDeath) {
      if ((b & 1) == 0) {
        if (sdLives[0] <= 1) {
          koWin = true;
          lastTouch = a;
          _ScoreGoal(1);
        } else {
          sdLives[0]--;
        }
      } else {
        if (sdLives[1] <= 1) {
          koWin = true;
          lastTouch = a;
          _ScoreGoal(0);
        } else {
          sdLives[1]--;
        }
      }
    }
  }
  if (killA) {
    FlagForFiltering(c);
    players[a]->OffPole();
    players[a]->GetStats()->AddDeath();
    players[b]->GetStats()->AddKill();
    players[b]->GetStats()->AddKick();
    players[b]->Kick(players[a]->Body()->GetPosition());
    players[b]->PlayKickSound(0x1.666666p-1f);  // 0.7f (rumble only)
    if (!suddenDeath) respawnTimer[a] = Assets::deathDelay;
    killMask &= ~mask;
    _ShockwaveAt(1, p, players[b]->Body()->GetPosition());
    {
      b2Body *body = players[a]->Body();
      body->ApplyLinearImpulse(b2Vec2(dxk, dy), body->GetWorldCenter());
    }
    {
      b2Body *body = players[b]->Body();
      body->ApplyLinearImpulse(b2Vec2(ax_m, ay_m), body->GetWorldCenter());
    }
    if (suddenDeath) {
      if ((a & 1) == 0) {
        if (sdLives[0] <= 1) {
          koWin = true;
          lastTouch = b;
          _ScoreGoal(1);
        } else {
          sdLives[0]--;
        }
      } else {
        if (sdLives[1] <= 1) {
          koWin = true;
          lastTouch = b;
          _ScoreGoal(0);
        } else {
          sdLives[1]--;
        }
      }
    }
  }
}

// PoleRiders::PlayState::HammerPlayerCollision(b2Contact*, b2ContactImpulse const*, int, int) @0x100139d6e
// Sudden-death hammer of player `owner` hits rider `victim`.
void PlayState::HammerPlayerCollision(b2Contact *c, const b2ContactImpulse *imp, int victim, int owner) {
  if (!players[victim]->OnPole()) return;
  const float s = fabsf(imp->tangentImpulses[0]) + imp->normalImpulses[0];
  if (0.5f > s) return;
  {
    const b2Vec2 v = players[owner]->GetHammer()->GetLinearVelocity();
    if (1.5f > sqrtf(v.x * v.x + v.y * v.y)) return;
  }
  b2WorldManifold wm;
  c->GetWorldManifold(&wm);
  b2Vec2 p;  // GAMEPLAY here (hit angle); the game leaves it uninitialized when there are no points
  p.SetZero();
  if (c->GetManifold()->pointCount > 0) p = wm.points[0];
  float lx, ly;
  LocalPoint(players[owner]->GetHammer(), p, lx, ly);
  const float ang = atan2f(ly, lx);
  if (-2.0f > ang) return;
  const double angd = (double)ang;
  if (angd > -1.15 && 1.15 > angd) return;  // ucomisd
  if (ang > 2.0f) return;

  const b2Vec2 vv = players[victim]->Body()->GetLinearVelocity();
  const b2Vec2 hv = players[owner]->GetHammer()->GetLinearVelocity();
  float rely = hv.y - vv.y;
  const b2Transform &hxf = players[owner]->GetHammer()->GetTransform();
  rely = rely * hxf.R.col2.y;
  const float rel = fabsf((hv.x - vv.x) * hxf.R.col2.x + rely);
  if (10.0f > rel) return;

  FlagForFiltering(c);
  players[victim]->OffPole();
  players[victim]->GetStats()->AddDeath();
  players[owner]->GetStats()->AddKill();
  _ShockwaveAt(victim, p, players[victim]->Body()->GetPosition());

  const b2Vec2 pv = players[victim]->Body()->GetPosition();
  const b2Vec2 hc = players[owner]->GetHammer()->GetWorldCenter();
  float dy = pv.y - hc.y;
  float dx = pv.x - hc.x;
  Normalize(dx, dy);
  const b2Vec2 hv2 = players[owner]->GetHammer()->GetLinearVelocity();
  b2Body *body = players[victim]->Body();
  if (body->GetType() == b2_dynamicBody) {
    const float spd = sqrtf(hv2.x * hv2.x + hv2.y * hv2.y);
    const float k = ((spd * 0.05f) * Assets::kickPlayerStrength) / Assets::ratio;
    const float h = s * 0.5f;
    const float iy = (dy * h) * k;
    const float ix = (dx * h) * k;
    body->ApplyLinearImpulse(b2Vec2(ix, iy), body->GetWorldCenter());
  }
  players[victim]->LiftOff(true);
  players[owner]->PlayKickSound(0x1.666666p-1f);  // 0.7f (rumble only)
  if ((victim & 1) == 0) {
    if (suddenDeath) {
      koWin = true;
      lastTouch = owner;
      _ScoreGoal(1);
    }
  } else {
    if (suddenDeath) {
      koWin = true;
      lastTouch = owner;
      _ScoreGoal(0);
    }
  }
}

// PoleRiders::PlayState::HammerBallCollision(b2Contact*, b2ContactImpulse const*, int) @0x10013a1c8
void PlayState::HammerBallCollision(b2Contact *c, const b2ContactImpulse *imp, int owner) {
  if (owner > 3) return;
  lastTouch = owner;
  const float s = fabsf(imp->tangentImpulses[0]) + imp->normalImpulses[0];
  if (1.0f > s) return;
  {
    b2Body *bb = ball->Body();
    float lx, ly;
    LocalPoint(players[owner]->GetHammer(), bb->GetPosition(), lx, ly);
    const double ang = (double)atan2f(ly, lx);
    if (-2.2 > ang) return;
    if (0.95 > ang && ang > -0.95) return;
    if (ang > 2.2) return;
  }
  {
    const float bx = ball->Body()->GetPosition().x;
    const float hx = players[owner]->GetHammer()->GetWorldCenter().x;
    if ((owner & 1) == 0) {
      if (hx > bx) return;
    } else {
      if (bx > hx) return;
    }
  }
  b2WorldManifold wm;
  c->GetWorldManifold(&wm);
  b2Vec2 p;  // presentation only
  p.SetZero();
  if (c->GetManifold()->pointCount > 0) p = wm.points[0];
  _ShockwaveAt(owner, p, players[owner]->GetHammer()->GetPosition());

  const b2Vec2 bp = ball->Body()->GetPosition();
  const b2Vec2 hc = players[owner]->GetHammer()->GetWorldCenter();
  float dy = bp.y - hc.y;
  float dx = bp.x - hc.x;
  Normalize(dx, dy);
  b2Body *body = ball->Body();
  if (body->GetType() == b2_dynamicBody) {
    const float k = Assets::kickBallStrength / Assets::ratio;
    const float h = s * 0.5f;
    const float iy = (dy * h) * k;
    const float ix = (dx * h) * k;
    body->ApplyLinearImpulse(b2Vec2(ix, iy), body->GetWorldCenter());
  }
  players[owner]->PlayHitSound(0x1.333334p-1f);  // 0.6f (rumble only)
  players[owner]->SetBallCollisions(false);
}

// PoleRiders::PlayState::BallWallCollision() @0x10013a4f2
void PlayState::BallWallCollision() {
  if (ballResetDelay > 0.0f) return;
  if (timeOver) return;
  ball->Ghost();
  const b2Vec2 bp = ball->Body()->GetPosition();
  const int team = !(bp.x > 0.0f);  // ucomiss 0, x ; setbe (NaN -> 1)
  _ScoreGoal(team);
  _ShockwaveAt(team ^ 1, bp, b2Vec2(bp.x + 1.0f, bp.y));
}

// PoleRiders::PlayState::BallPlayerCollision(b2ContactImpulse const*, int) @0x10013a9ec
void PlayState::BallPlayerCollision(const b2ContactImpulse *imp, int a) {
  if (!timerOn) timerOn = true;
  lastTouch = a;
  if (!ballKickAvailable) return;
  const float s = fabsf(imp->tangentImpulses[0]) + imp->normalImpulses[0];
  b2Body *pb = players[a]->Body();
  b2Body *bb = ball->Body();
  const float r = Assets::ratio;
  const float py = pb->GetPosition().y * r;
  const float by = r * bb->GetPosition().y + 60.0f;
  if (!(by > py)) return;  // ball must not be more than 60px below the rider's centre
  ballKickAvailable = false;
  if (!players[a]->ShouldBallCollide()) return;
  const float bx = bb->GetPosition().x;
  const float px = pb->GetPosition().x;
  float k = Assets::kickBallStrength;
  if (px > bx) k = -k;
  bool kick = false;
  if ((a & 1) == 0 && bx > px) kick = true;
  else if (a % 2 == 1 && px > bx) kick = true;
  if (!kick) {
    // if (s > 2.0f) PlayBallSound(0.5) -- presentation
    (void)s;
    return;
  }
  const float r2 = Assets::ratio;
  bb->ApplyLinearImpulse(b2Vec2(k / r2, 0.0f), bb->GetWorldCenter());
  pb->ApplyLinearImpulse(b2Vec2((k * -0.07f) / r2, 0.0f), pb->GetWorldCenter());
  players[a]->Kick(bb->GetPosition());
  _ShockwaveAt(a, bb->GetPosition(), pb->GetPosition());
  players[a]->PlayHitSound(0x1.333334p-1f);  // 0.6f (rumble only)
  players[a]->GetStats()->AddKick();
  players[a]->SetBallCollisions(false);
}

// PoleRiders::PlayState::_ScoreGoal(int) @0x1001389a4
void PlayState::_ScoreGoal(int team) {
  if (ballResetDelay > 0.0f) return;
  if (endFade > 0.0f) return;
  if (winBanner >= 0.0f) return;
  lastGoalTeam0 = (team == 0);
  if (team != 0)
    score[1]++;
  else
    score[0]++;
  // random()%2 goal sound, scoreArrow animation: presentation (skipped)
  goalBanner = 0.0f;
  // targets[1-team]->GoalScored(): presentation (skipped)
  for (int i = 0; i < numPlayers; i++) players[i]->RumbleBump(0x1.333334p-1f, 0.5f);  // (0.6, 0.5)
  const int lt = lastTouch;
  if (lt != -1) {
    Stats *st = players[lt]->GetStats();
    if (lt % 2 == team)
      st->AddGoal();
    else
      st->AddOwnGoal();
  }
  if (suddenDeath) {
    timeLeftShown = 0.0f;
    timeLeft = 1.0f;
    winBanner = 0.0f;
    timerOn = true;
    for (int i = 0; i < numPlayers; i++) players[i]->GetStats()->SetTimeLeft(0.0f);
    return;
  }
  const int ms = sim::g_engine->cfg.maxScore;  // Assets::maxScore
  if (score[0] >= ms) {
    players[0]->GetStats()->SetTimeLeft(timeLeft);
    if (numPlayers >= 3) players[2]->GetStats()->SetTimeLeft(timeLeft);
    players[1]->GetStats()->SetTimeLeft(0.0f);
    if (numPlayers >= 3) players[3]->GetStats()->SetTimeLeft(0.0f);
  } else if (score[1] >= ms) {
    players[1]->GetStats()->SetTimeLeft(timeLeft);
    if (numPlayers >= 3) players[3]->GetStats()->SetTimeLeft(timeLeft);
    players[0]->GetStats()->SetTimeLeft(0.0f);
    if (numPlayers >= 3) players[2]->GetStats()->SetTimeLeft(0.0f);
  } else {
    ballResetDelay = 0x1.666666p+0f;  // 0x3fb33333 = 1.4
    return;
  }
  if (timeLeft > 1.0f) timeLeftShown = timeLeft;
  timeLeft = 1.0f;
  winBanner = 0.0f;
}

// PoleRiders::PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2) @0x10013942c
// ShockWave::Trigger, Sparks::AddParticle (random() count, atan2f angle) and Camera::Shake are
// presentation; the inlined _PunchTime tail (hit-stop) is gameplay.
void PlayState::_ShockwaveAt(int, b2Vec2, b2Vec2) {
  const float h = punchHeat + 2.0f;
  punchHeat = h;
  if (5.0f > h) hitstopFrames = 7;
}

// PoleRiders::PlayState::_MakeDeleters() @0x10013560a (Level4 only)
void PlayState::_MakeDeleters() {
  b2PolygonShape box;
  b2BodyDef bd;  // defaults: static, allowSleep, awake, active, inertiaScale 1
  b2FixtureDef fd;
  const float r = Assets::ratio;
  box.SetAsBox(512.0f / r, 2000.0f / r);
  bd.userData = const_cast<int *>(&Assets::BODY_DELETER);
  const float r1 = Assets::ratio;
  bd.position.Set((levelWidth * 0.5f + 1024.0f) / r1, 0.0f / r1);
  b2Body *b1 = world->CreateBody(&bd);
  fd.isSensor = true;
  fd.filter.categoryBits = 1;
  fd.filter.maskBits = 0xfe7e;
  fd.filter.groupIndex = 0;
  fd.shape = &box;
  b1->CreateFixture(&fd);
  b1->SetType(b2_staticBody);
  const float r2 = Assets::ratio;
  bd.position.Set((levelWidth * -0.5f + -1024.0f) / r2, 0.0f / r2);
  b2Body *b2 = world->CreateBody(&bd);
  b2->CreateFixture(&fd);
  b2->SetType(b2_staticBody);
}

}  // namespace PoleRiders

// ---- PRES* Player sound helpers: rumble part only (sound choice via random() and the sound are
// skipped). The rumble accumulators reach the pad dumps through Player::Update's vibration request.
namespace PoleRiders {

// PoleRiders::Player::RumbleBump(float, float) @0x10012d9e0
void Player::RumbleBump(float a, float b) {
  rumbleB = fminf(a + rumbleB, 1.0f);
  rumbleA = b + rumbleA;
}

// PoleRiders::Player::PlayDeathSound(float) @0x10012da2e
void Player::PlayDeathSound(float) {
  if (!canPlayDeathSound) return;
  // random() % 2 -> deathSoundA/B->Play(v, 0, 1): skipped
  rumbleB = fminf(rumbleB + 0x1.99999ap-2f, 1.0f);  // + 0.4f
  rumbleA = rumbleA + 0x1.333334p-1f;               // + 0.6f
  canPlayDeathSound = false;
}

// PoleRiders::Player::PlayKickSound(float) @0x10012dada
void Player::PlayKickSound(float) {
  // random() % 2 -> kickSoundA/B->Play(v, 0, 1): skipped
  rumbleB = fminf(rumbleB + 0x1.333334p-1f, 1.0f);  // + 0.6f
  rumbleA = rumbleA + 0x1.333334p-3f;               // + 0.15f
}

// PoleRiders::Player::PlayHitSound(float) @0x10012db72
void Player::PlayHitSound(float) {
  // random() % 2 -> hitSoundA/B->Play(v, 0, 1): skipped
  rumbleB = fminf(rumbleB + 0x1.99999ap-1f, 1.0f);  // + 0.8f
  rumbleA = rumbleA + 0x1.333334p-3f;               // + 0.15f
}

}  // namespace PoleRiders

namespace PoleRiders {

// PoleRiders::Player::PlayHammerSound(float, bool) @0x10012dc0a
void Player::PlayHammerSound(float v, bool) {
  if (canPlayHammerSound) {
    // hard ? landHardSound : tapSound[random() % 3] ->Play(v, 0, 1): skipped
    canPlayHammerSound = false;
  }
  rumbleB = fminf(rumbleB + v * 0.5f, 1.0f);
  rumbleA = rumbleA + 0x1.333334p-3f;  // + 0.15f
}

// PoleRiders::Player::PlayLandSound(float, bool) @0x10012dcf4
void Player::PlayLandSound(float v, bool) {
  if (canPlayLandSound) {
    // (hard ? landHardSound : landSoftSound)->Play(...): skipped
    canPlayLandSound = false;
  }
  rumbleB = fminf(rumbleB + v * 0.5f, 1.0f);
  rumbleA = rumbleA + 0x1.333334p-3f;  // + 0.15f
}

// PoleRiders::PlayState::GroundContact(b2Contact*, b2ContactImpulse const*, int, int) @0x100138d00
// (a = body tag: rider 0..2 or BODY_HAMMER; b = fixture tag = hammer owner). Presentation: ShowSmash,
// Dust::AddParticle (random()), _Kernel_Log and the sounds are skipped; the land / hammer sound
// helpers are still called for their rumble (and sound-gate) writes.
void PlayState::GroundContact(b2Contact *, const b2ContactImpulse *imp, int a, int b) {
  const float s = fabsf(imp->tangentImpulses[0]) + imp->normalImpulses[0];
  if (a <= 3) {
    if (sim::g_engine->cfg.roundTime - timeLeft > 1.0f && players[a]->canPlayLandSound) {
      if (8.0f > s)
        players[a]->PlayLandSound(fminf(s * 0.125f, 0x1.99999ap-2f), false);  // min(.., 0.4)
      else
        players[a]->PlayLandSound(fminf(s / 10.0f, 0.5f), true);
    }
  } else if (a == BODY_HAMMER && b <= 3 && players[b]->canPlayHammerSound) {
    if (s > 6.0f)
      players[b]->PlayHammerSound(fminf(s * 0.125f, 0x1.333334p-1f), true);  // min(.., 0.6)
    else if (s > 1.0f)
      players[b]->PlayHammerSound(fminf(s / 3.0f, 0x1.99999ap-1f), false);  // min(.., 0.8)
  }
}

}  // namespace PoleRiders
