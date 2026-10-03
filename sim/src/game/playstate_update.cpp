// PlayState::Update (package C): the per-tick match driver.
// Presentation is skipped (Background volumes, Camera, ShockWave/Dust/Sparks/Flare updates and the
// 8x random() Flare particles, Target/score-arrow/gull animations, sounds, achievements, Taunting ->
// TriggerFart); every gameplay field write and every call into gameplay code is kept, in order.
#include <cmath>
#include "assets.h"
#include "engine.h"
#include "layout.h"

namespace PoleRiders {

namespace {
const float kSide[2] = {1.0f, -1.0f};  // @0x10028b430, indexed by (i even) / (ball x < 0)
}

// PoleRiders::PlayState::Update() @0x10013b8c8 (inlines _GoSuddenDeath)
void PlayState::Update() {
  float dt = this->dt;  // -0x64: this tick's dt (0 during hit-stop)

  // punch heat / hit-stop
  float ph = punchHeat;
  if (ph > 0.0f) {
    ph = ph - dt;
    punchHeat = ph;
  }
  if (0.0f > ph) punchHeat = 0.0f;
  if (hitstopFrames > 0) {
    hitstopFrames = hitstopFrames - 1;
    dt = 0.0f;
  }

  // fade-in / music volume ramp (presentation fields; Background::SetVolume skipped)
  if (fadeIn > 0.0f) fadeIn = fadeIn + -2.0f * dt;
  if (0.0f > endFade) {
    float v = volumeRamp;
    if (1.0f > v)
      v = v + 0.1f * dt;
    else
      v = 1.0f;
    volumeRamp = v;
  }

  // 2v2 sudden death: a rider whose shared pole got knocked off un-shares its partner
  int n = numPlayers;
  if (n == 4 && suddenDeath && n > 0) {
    for (int i = 0; i < numPlayers; i++) {
      if (players[i]->ShouldUnshare()) {
        players[i]->ResetUnshare();
        players[(i + 2) % 4]->UnsharePole();
      }
    }
    n = numPlayers;
  }

  // hidden button 0x4a held on both teams -> sudden death (_GoSuddenDeath inlined)
  if (n > 0) {
    bool team0 = false, team1 = false;
    for (int i = 0; i < numPlayers; i++) {
      if (Common::Input::ButtonHeld(players[i]->inputId, 0x4a)) {
        if ((i & 1) == 0)
          team0 = true;
        else
          team1 = true;
      }
    }
    if (team0 && team1 && !suddenDeath) {
      score[0] = 0;
      score[1] = 0;
      suddenDeath = true;
      timeLeft = 0.0f;
      sdBanner = 0.0f;
      // sndGoldenGoal->Play(0.8, 0, 1) skipped
      ballResetPending = true;
    }
  }

  // match clock
  if (timerOn && timeLeft > 0.0f) {
    const float t = timeLeft - dt;
    timeLeft = t;
    if (t <= 0.0f) {  // ucomiss t, 0 ; jb (skips NaN)
      const int s0 = score[0], s1 = score[1];
      if (s0 == s1) {
        suddenDeath = true;
        sdBanner = 0.0f;
        // sndGoldenGoal->Play skipped
        ballResetPending = true;
      } else if (s0 > s1 || s1 > s0) {
        matchDecided = true;
        const bool sd = suddenDeath;
        const int ms = sim::g_engine->cfg.maxScore;  // Assets::maxScore
        if (!sd && s0 < ms && s1 < ms) timeUpBanner = 0.0f;
        if (!sd && s0 < ms && s1 < ms) winBanner = 0.0f;
        timeOver = true;
      }
      // (level == 5 && matchDecided): achievements skipped
      if (matchDecided) {
        for (int i = 0; i < numPlayers; i++) {
          players[i]->SetActive(false);
          players[i]->GameOverBody();
        }
        endSlowmoTimer = 3.0f;
      }
    }
  }

  // end fade -> game over (Camera::Shake(0,0,0) skipped)
  if (endFade >= 0.0f) {
    const float e = endFade + 0x1.861864p-7f;  // 0x3c430c32 = 1/84
    endFade = e;
    if (e > 1.0f) gameOver = true;
  }

  // per-step contact budgets
  ballKickAvailable = true;
  killMask = 0xf;

  for (int i = 0; i < numPlayers; i++) players[i]->UpdateGhostTimer(dt);

  // a rider that just un-ghosted while overlapping an enemy rider/pole ghosts again for 0.1 s
  for (int i = 0; i < numPlayers; i++) {
    if (!players[i]->JustUnghosted()) continue;
    if (!(numPlayers > 0)) continue;
    const int iTeam = i % 2;
    bool overlap = false;
    int j = 0;
    do {
      if (j % 2 != iTeam) {
        b2Body *bi = players[i]->Body();
        // rider i's non-sensor fixtures vs rider j's body and pole
        for (b2Fixture *fa = bi->GetFixtureList(); fa; fa = fa->GetNext()) {
          if (fa->IsSensor()) continue;
          b2Body *bj = players[j]->Body();
          for (b2Fixture *fb = bj->GetFixtureList(); fb; fb = fb->GetNext())
            if (b2TestOverlap(fa->GetShape(), fb->GetShape(), players[i]->Body()->GetTransform(),
                              players[j]->Body()->GetTransform()))
              overlap = true;
          if (players[j]->GetPole()->NumSegments() > 0) {
            int s = 0;
            do {
              for (b2Fixture *fb = players[j]->GetPole()->Segment(s)->GetFixtureList(); fb; fb = fb->GetNext())
                if (b2TestOverlap(fa->GetShape(), fb->GetShape(), players[i]->Body()->GetTransform(),
                                  players[j]->GetPole()->Segment(s)->GetTransform()))
                  overlap = true;
              s++;
            } while (players[j]->GetPole()->NumSegments() > s);
          }
        }
        // rider i's pole segments (all fixtures) vs rider j's body and pole
        if (players[i]->GetPole()->NumSegments() > 0) {
          int si = 0;
          do {
            for (b2Fixture *fa = players[i]->GetPole()->Segment(si)->GetFixtureList(); fa; fa = fa->GetNext()) {
              for (b2Fixture *fb = players[j]->Body()->GetFixtureList(); fb; fb = fb->GetNext())
                if (b2TestOverlap(fa->GetShape(), fb->GetShape(), players[i]->GetPole()->Segment(si)->GetTransform(),
                                  players[j]->Body()->GetTransform()))
                  overlap = true;
              if (players[j]->GetPole()->NumSegments() > 0) {
                int s = 0;
                do {
                  for (b2Fixture *fb = players[j]->GetPole()->Segment(s)->GetFixtureList(); fb; fb = fb->GetNext())
                    if (b2TestOverlap(fa->GetShape(), fb->GetShape(),
                                      players[i]->GetPole()->Segment(si)->GetTransform(),
                                      players[j]->GetPole()->Segment(s)->GetTransform()))
                      overlap = true;
                  s++;
                } while (players[j]->GetPole()->NumSegments() > s);
              }
            }
            si++;
          } while (players[i]->GetPole()->NumSegments() > si);
        }
      }
      j++;
    } while (numPlayers > j);
    if (overlap) players[i]->SetGhost(true, 0.1f);
  }

  // physics
  world->SetContinuousPhysics(false);  // b2World+0x19365 = 0
  world->Step(dt, 60, 18);
  world->ClearForces();

  for (int i = 0; i < numPlayers; i++) players[i]->Update(dt);
  // ShockWave::Update(1/60) x numPlayers, Dust/Sparks Update or Fade, Flare::Update (level 2): skipped
  ball->Update(dt);
  // Target::Update x2, score-arrow / gull Animation updates: presentation, skipped

  // ball reset countdown after a goal
  if (ballResetDelay > 0.0f) {
    const float d = ballResetDelay - dt;
    ballResetDelay = d;
    if (d <= 0.0f) {  // ucomiss d, 0 ; jb
      ballResetPending = true;
      ballResetDelay = 0.0f;
    }
  }
  if (ballResetPending) {
    ballResetPending = false;
    ballResetDelay = 0.0f;
    if (ball) delete ball;  // ~Ball leaks the anchor body, as in the game
    Ball *b = new Ball(0.0f, ballHeight, world, sprites, additiveSprites, levelWidth);
    ball = b;
    b->CreateBody();
    if (numPlayers > 0) {
      const float t = 0.5f * dt;
      int i = 0;
      do respawnTimer[i] = t;
      while (numPlayers > ++i);
    }
  }
  // Taunting() && OnPole() -> TriggerFart on an opponent + sound: presentation, skipped. (Taunting
  // itself only writes Player::tauntTimer when LT and R3 are held, which the sim's input never does.)

  // Princeton: riders leaving the arena sideways wrap around (teleport), and ghost near the edges
  if (level == 4) {
    if (numPlayers > 0) {
      const float r = Assets::ratio;
      int i = 0;
      do {
        const float x = players[i]->Body()->GetPosition().x;
        const float lim = (levelWidth * 0.5f + 412.0f) / r;
        const float nlim = -lim;
        if (nlim > x) {
          if (players[i]->SharingPole()) {
            const int p = (i + 2) % 4;
            if (nlim > players[p]->Body()->GetPosition().x) {
              players[i]->Teleport((levelWidth + 824.0f) / r, true);
              players[p]->Teleport((levelWidth + 824.0f) / r, false);
            }
          } else {
            players[i]->Teleport((levelWidth + 824.0f) / r, true);
          }
        } else if (x > lim) {
          if (players[i]->SharingPole()) {
            const int p = (i + 2) % 4;
            if (players[p]->Body()->GetPosition().x > lim) {
              players[i]->Teleport(-((levelWidth + 824.0f) / r), true);
              players[p]->Teleport(-((levelWidth + 824.0f) / r), false);
            }
          } else {
            players[i]->Teleport(-((levelWidth + 824.0f) / r), true);
          }
        }
        i++;
      } while (numPlayers > i);
    }
    if (numPlayers > 0) {
      const double r = (double)Assets::ratio;
      int i = 0;
      do {
        const float x = players[i]->Body()->GetPosition().x;
        const float h = (float)(((double)(levelWidth * 0.5f) + -100.0) / r);
        const float nh = -h;
        if (nh > x || x > h) players[i]->SetGhost(true, 0.1f);
        i++;
      } while (numPlayers > i);
    }
  }
  // (level == 2): OnGround && ShouldAddFlare -> 8x random() + Flare::AddParticle: presentation, skipped

  // respawns
  int sdRespawn = 0;  // %ebx: numPlayers once a sudden-death respawn happened
  if (numPlayers > 0) {
    const float rainbowLen = Assets::rainbowPoleLength;
    const float stdLen = Assets::standardPoleLength;
    const float shortLen = -147.0f + stdLen;
    int i = 0;
    do {
      float t = respawnTimer[i];
      if (t > 0.0f) {
        t = t - dt;
        respawnTimer[i] = t;
        if (t <= 0.0f) {  // ucomiss t, 0 ; jb
          const int row = (i - i % 2) / 2;
          const bool even = (i & 1) == 0;
          if (suddenDeath) {
            sdRespawn = numPlayers;
          } else {
            const float side = kSide[even];
            bool shortPole;
            float len;
            if (players[i]->GetHeadNumber() == 7) {
              shortPole = true;
              len = shortLen;
            } else if (level == 5) {
              shortPole = false;
              len = rainbowLen;
            } else {
              shortPole = false;
              len = stdLen;
            }
            const float y = spawnHeight - (float)(row << 6);
            const float x = ((levelWidth + 64.0f) * side) * 0.5f;
            players[i]->Respawn(x, y, false, len, false, shortPole);
            players[i]->Boost((-spawnBoost - (float)(row * 20)) * side);
          }
          respawnTimer[i] = 0.0f;
        }
      }
      i++;
    } while (numPlayers > i);
  }
  if (sdRespawn > 0) {
    if (sdRespawn == 4) {
      // 2v2 sudden death: one shared hammer pole per team
      const float len = Assets::standardPoleLength + 128.0f;
      players[0]->Respawn((levelWidth + 64.0f) * -0.5f, spawnHeight, false, len, true, false);
      Pole *p0 = players[0]->GetPole();
      players[2]->Respawn(((levelWidth + 64.0f) + -700.0f) * -0.5f, spawnHeight, false, p0);
      players[0]->Boost(spawnBoost);
      players[2]->Boost(spawnBoost);
      const float rs = 2500.0f / Assets::ratio;
      players[0]->SetRunSpeed(rs);
      players[2]->SetRunSpeed(rs);
      sdLives[0] = 2;
      players[1]->Respawn((levelWidth + 64.0f) * 0.5f, spawnHeight, false, len, true, false);
      Pole *p1 = players[1]->GetPole();
      const float x3 = (64.0f + levelWidth) + -700.0f;
      players[3]->Respawn(x3 * 0.5f, spawnHeight, false, p1);
      players[1]->Boost(-spawnBoost);
      players[3]->Boost(-spawnBoost);
      players[1]->SetRunSpeed(rs);
      players[3]->SetRunSpeed(rs);
      sdLives[1] = 2;
      if (level == 4) _MakeDeleters();
    } else {
      // 1v1 sudden death: hammer poles
      float len = ballHeight * -2.0f;
      if (level == 2)
        len = len + 32.0f;
      else
        len = len + -80.0f;
      players[0]->Respawn((levelWidth + 64.0f) * -0.5f, spawnHeight, true, len, false, false);
      players[0]->Boost((float)((double)spawnBoost * 1.3));
      players[1]->Respawn((64.0f + levelWidth) * 0.5f, spawnHeight, true, len, false, false);
      players[1]->Boost((float)((double)(-spawnBoost) * 1.3));
    }
  }

  // end-of-match slow motion
  const bool md = matchDecided;
  if (md && dt > 0x1.5d867cp-13f) this->dt = this->dt / 1.5f;  // 0x392ec33e
  if (md && endSlowmoTimer > 0.0f) {
    const float e = endSlowmoTimer + -0x1.111112p-6f;  // 0xbc888889 = -1/60
    endSlowmoTimer = e;
    if (0.0f > e) endFade = 0.0f;
  }
  // Camera focus / update: skipped

  // banners (winBanner >= 0 blocks _ScoreGoal)
  if (goalBanner >= 0.0f) {
    const float g = goalBanner + dt / 1.7f;
    goalBanner = g;
    if (g > 1.0f) goalBanner = -1.0f;
  }
  if (sdBanner >= 0.0f) {
    const float s = sdBanner + dt / 0.3f;
    sdBanner = s;
    if (s > 1.0f) sdBanner = 1.0f;
  }
  if (timeUpBanner >= 0.0f) {
    const float s = timeUpBanner + 0x1.24924ap-7f;  // 0x3c124925 = 1/112
    timeUpBanner = s;
    if (s > 1.0f) timeUpBanner = 1.0f;
  }
  if (winBanner >= 0.0f) {
    const float s = winBanner + 0x1.3b13b2p-8f;  // 0x3b9d89d9 = 1/208
    winBanner = s;
    if (s > 1.0f) winBanner = 1.0f;
  }

  // stats
  if (!timerOn || timeOver) return;
  int nearest = -1;
  if (numPlayers > 0) {
    float best = 10000.0f;
    int i = 0;
    do {
      b2Body *bb = ball->Body();
      b2Body *pb = players[i]->Body();
      float dy = pb->GetPosition().y - bb->GetPosition().y;
      dy = dy * dy;
      float dx = pb->GetPosition().x - bb->GetPosition().x;
      dx = dx * dx;
      const float d = dx + dy;
      if (best > d) {
        nearest = i;
        best = d;
      }
      i++;
    } while (numPlayers > i);
  }
  // (the game indexes players[-1] if nobody is within 100 m of the ball; unreachable in practice)
  if (nearest >= 0) players[nearest]->GetStats()->IncrementTimeNearestBall(dt);
  {
    const float bx = ball->Body()->GetPosition().x;
    if ((double)fabsf(bx) > (double)targetWidth / 3.0) {
      const float s = kSide[0.0f > ball->Body()->GetPosition().x];
      const float a = s * dt;
      players[0]->GetStats()->IncrementOffenseRatio(a);
      const float b = -s * dt;
      players[1]->GetStats()->IncrementOffenseRatio(b);
      if (numPlayers > 2) {
        players[2]->GetStats()->IncrementOffenseRatio(a);
        players[3]->GetStats()->IncrementOffenseRatio(b);
      }
    }
  }
  for (int i = 0; i < numPlayers; i++) {
    const double x = (double)players[i]->Body()->GetPosition().x;
    if ((i & 1) == 0) {
      if (x > (double)targetWidth / 3.0) {
        players[i]->GetStats()->IncrementTimeInEndzone(dt);
      } else if ((double)targetWidth / -3.0 > (double)players[i]->Body()->GetPosition().x) {
        players[i]->GetStats()->IncrementTimeInGoal(dt);
      }
    } else {
      if ((double)targetWidth / -3.0 > x) {
        players[i]->GetStats()->IncrementTimeInEndzone(dt);
      } else if ((double)players[i]->Body()->GetPosition().x > (double)targetWidth / 3.0) {
        players[i]->GetStats()->IncrementTimeInGoal(dt);
      }
    }
  }
}

}  // namespace PoleRiders
