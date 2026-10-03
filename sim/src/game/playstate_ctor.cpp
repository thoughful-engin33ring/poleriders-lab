// PlayState construction / teardown (package A). PlayState::Update and the contact listener are
// package C (playstate_update.cpp, contacts.cpp).
//
// Skipped presentation (pointers stay null): camera, targets, sprite sheets, Dust/Sparks/Flare,
// score-arrow / gull animations, sounds, shockwaves, backgrounds, reverb/compressor, DebugRender,
// Christmas / Chinese-new-year / snake / pencil easter-egg flags (time() and random() based), the
// srandom(Time_GetHighResTimer()) reseed and the Level-4 Christmas achievement.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "assets.h"
#include "engine.h"

namespace PoleRiders {

// PoleRiders::PlayState::PlayState(bool) @0x100134190 (C2; C1 thunk @0x10013d64c). Inlines
// PlayState::_SetupWorld() (@0x10013b032).
PlayState::PlayState(bool fromRainbow) {
  // The bridge shadow (and so the oracle) allocates PlayState with calloc: every field the ctor
  // doesn't write is 0. Keep the vptr.
  memset(reinterpret_cast<char *>(this) + sizeof(void *), 0, sizeof(PlayState) - sizeof(void *));
  const sim::MatchConfig &cfg = sim::g_engine->cfg;

  timerOn = false;
  timeLeft = cfg.roundTime;  // Assets::roundTime
  timeLeftShown = 0.0f;
  suddenDeath = false;
  sdLives[0] = 0;
  sdLives[1] = 0;
  matchDecided = false;
  gameOver = false;
  endFade = -1.0f;
  fadeIn = 1.0f;
  volumeRamp = 0.0f;
  fromRainbowUnlock = fromRainbow;
  ballResetPending = false;
  ballResetDelay = 0.0f;
  dt = 0x1.111112p-6f;  // 0x3c888889 = 1/60
  score[0] = 0;
  score[1] = 0;
  lastTouch = -1;
  koWin = false;
  timeOver = false;
  unlockingRainbow = false;
  killMask = 0xf;
  ballKickAvailable = true;

  // _SetupWorld()
  b2Vec2 gravity(0.0f, 20.0f);
  world = new b2World(gravity, true);
  world->SetContactListener(this);

  switch (cfg.level) {  // Assets::currentLevel
    case 2:
      levelWidth = 3520.0f;   // Level2::_levelWidth
      ballHeight = -394.0f;   // Level2::_ballHeight
      targetWidth = 2600.0f;  // Level2::_targetWidth
      spawnHeight = -782.0f;  // Level2::_spawnHeight
      spawnBoost = 50.0f;     // sic: Level1::_spawnBoost for every level (original bug)
      level2 = new Level2(0.0f, 0.0f, world);
      level = 2;
      break;
    case 3: case 4: case 5:
      fprintf(stderr, "PlayState: level %d not ported (only 1 = Kalgoorlie, 2 = Flatirons)\n", cfg.level);
      abort();
    default:  // 1, and (as in the original's jump-table default) anything out of 1..5
      levelWidth = 2800.0f;   // Level1::_levelWidth
      ballHeight = -394.0f;   // Level1::_ballHeight
      targetWidth = 2606.0f;  // Level1::_targetWidth
      spawnHeight = -612.0f;  // Level1::_spawnHeight
      spawnBoost = 50.0f;     // Level1::_spawnBoost
      level1 = new Level1(0.0f, 0.0f, world);
      level = 1;
      break;
  }

  // Common::Save: level5unlocked/showIcons -> Assets (engine config in the sim; nothing to copy)
  const PoleRiders::SessionPlayer *sess = sim::g_engine->session;  // Common::Session::Get()
  numPlayers = 0;
  for (int i = 0; i < 4; i++)
    if (sess[i].team != 0) numPlayers++;

  float spawnX = 0.8f;  // -0x24c (reused: the camera zoom literal until a level case sets it)
  const float rainbowLen = Assets::rainbowPoleLength;
  const float standardLen = Assets::standardPoleLength;
  const float shortLen = -147.0f + standardLen;
  bool firstLeft = true, firstRight = true;
  for (int slot = 0; slot < 4; slot++) {
    int t = sess[slot].team;
    if (t == 0) continue;
    int idx;
    if (t == -1) {
      idx = firstLeft ? 0 : 2;
      firstLeft = false;
    } else {
      idx = firstRight ? 1 : 3;
      firstRight = false;
    }
    int side = t == -1 ? -1 : 1;
    int teamArg = t != -1;
    switch (level) {  // player spawn x base (jump table @0x100135378)
      case 1: case 5: spawnX = 400.0f; break;
      case 2: case 3: spawnX = 800.0f; break;
      case 4: spawnX = 1000.0f; break;
      default: break;
    }
    Player *p = new Player(world, slot, idx, teamArg, 0, sprites, dust);
    players[idx] = p;
    float x = (float)side * ((float)((idx << 7) & 0x100) + spawnX);
    if (players[idx]->GetHeadNumber() == 7)
      players[idx]->Respawn(x, -5.0f, false, shortLen, false, true);
    else if (level == 5)
      players[idx]->Respawn(x, -5.0f, false, rainbowLen, false, false);
    else
      players[idx]->Respawn(x, -5.0f, false, standardLen, false, false);
    respawnTimer[idx] = 0.0f;
  }

  ball = new Ball(0.0f, ballHeight, world, sprites, additiveSprites, levelWidth);
  ball->CreateBody();

  goalBanner = -1.0f;
  sdBanner = -1.0f;
  timeUpBanner = -1.0f;
  winBanner = -1.0f;
  bannerFrame = 0;
  debugRender = nullptr;  // Assets::debug == 0
  hitstopFrames = 0;
  punchHeat = 0.0f;
  reverbOn = true;
  compressorOn = true;
}

// PoleRiders::PlayState::~PlayState() @0x10013b4ba (D2; D0 @0x10013b098 = this + operator delete)
PlayState::~PlayState() {
  for (int i = 0; i < numPlayers; i++) {
    if (players[i]) {
      players[i]->~Player();
      operator delete(players[i]);
    }
  }
  if (ball) {
    ball->~Ball();
    operator delete(ball);
  }
  switch (level) {  // Level objects have no dtor: plain operator delete
    case 1: operator delete(level1); break;
    case 2: operator delete(level2); break;
    case 3: operator delete(level3); break;
    case 4: operator delete(level4); break;
    case 5: operator delete(level5); break;
    default: break;
  }
  delete world;
}

// PoleRiders::PlayState::LoadComplete() @0x10013d574
// Presentation only: plays the fanfare, starts every player's walk/slide loops (Player::LoadComplete)
// and the level background's LoadComplete.
void PlayState::LoadComplete() {}

}  // namespace PoleRiders
