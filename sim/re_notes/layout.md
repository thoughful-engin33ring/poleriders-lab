# Game object memory layout (Super Pole Riders)

Recovered from `Sportsfriends.orig` (x86_64, unstripped, optimized). Compilable form with
`static_assert`s: `sim/src/game/layout.h`. Method: every `PoleRiders::*` function was scanned
with a small type-propagating register tracker (this-pointer in `%rdi`, typed pointer fields,
stack spills, `operator new` sizes, ctor calls); all `disp(%reg)` accesses on typed objects were
collected and then checked by hand against the ctors, `PlayState::Update`, `Player::Update`,
the contact handlers, `_ScoreGoal`, `Respawn`, `SpawnPlayerBody`, `SetGhost`, `Pole::Pole`,
`Ball::CreateBody/Update` (scratch scripts: `$SCRATCH/re-layout/{scan,rep,ctx,mktab}.py`).

"Writers"/"readers" columns are the functions that access the field *directly* (inlined code
included); ctor = the class name itself. Functions reached only through accessors (e.g.
`Player::OnPole()`) are listed under the accessor. *(presentation)* = only drawing, sound,
particles, rumble, UI read it. `?` = meaning uncertain.

Sizes (from `operator new` at the allocation sites):

| class | size | vtable | allocated by | freed by |
|---|---|---|---|---|
| PlayState | 0x228 | yes (b2ContactListener, primary base; `this` is the listener) | `PoleRiders::Begin()` / `PoleRiders::Update()` (global `_playState`) | same |
| Player | 0x240 | no | PlayState ctor (per active session slot) | ~PlayState |
| Pole | 0x78 | no | `Player::Respawn(x,y,bool,float,bool,bool)` (deletes the old one first) | Respawn, ~Player (unless sharing) |
| Ball | 0x48 | no | PlayState ctor, and `PlayState::Update` on every ball reset (after each goal / sudden death) | Update (ball reset), ~PlayState |
| Stats | 0x34 | no | end of Player ctor | ~Player |
| Level1..5 | 0x08 | no | PlayState ctor (only the one for `level`) | ~PlayState (switch on `level`) |
| Target | 0x20 | no | PlayState ctor (x2) | ~PlayState |
| Animation | 0x18 | no | PlayState ctor (x3), Target ctor, `Player::SpawnPlayerBody` (x2, old ones deleted) | owners |
| b2World | 0x19368 | - | PlayState ctor (inlined `_SetupWorld`), gravity (0, 20), doSleep=true | ~PlayState |
| Camera 0x44, Dust 0x3410, Sparks 0x3010, Flare 0xd010, ShockWave 0x20 (x numPlayers), Background 0x358 / _Level2 0x48 / _Level3 0x68 / _Level4 0x88 / _Level5 0x88, DebugRender 0x40, SpriteSheet 0x250, Sound::Effect 0x18, Sound::Persistent 0x20 | | | PlayState / Player ctors | owners |


## PlayState (0x228)

`PlayState(bool fromRainbowUnlock)` @0x100134190 (C2). Player order in `players[]`: team 0 at 0,2; team 1 at 1,3.

| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `void*` | _vtbl | vtable for PoleRiders::PlayState (b2ContactListener: BeginContact/EndContact/PreSolve/PostSolve) | PlayState, ~PlayState | Box2D (virtual dispatch) |
| 0x008 | `float` | levelWidth | px, LevelN::_levelWidth (Flatirons 3520) | PlayState | Draw, PlayState, Update, _MakeDeleters |
| 0x00c | `float` | targetWidth | px, LevelN::_targetWidth (endzone stats: /x/ > targetWidth/3) | PlayState | PlayState, Update, _DrawWire |
| 0x010 | `float` | ballHeight | px, LevelN::_ballHeight (Ball spawn y) | PlayState | PlayState, Update, _DrawWire |
| 0x014 | `float` | spawnHeight | px, LevelN::_spawnHeight (respawn y) | PlayState | Update |
| 0x018 | `float` | spawnBoost | px, ALWAYS Level1::_spawnBoost (50) -- binary bug, see notes | PlayState | Update |
| 0x020 | `b2World*` | world | new b2World(gravity (0,20)), contact listener = this | PlayState, _SetupWorld | Draw, PlayState, Update, _MakeDeleters, ~PlayState |
| 0x028 | `int32_t[2]` | score | [team]; _ScoreGoal(team) increments; zeroed on sudden death | PlayState, Update, _GoSuddenDeath, _ScoreGoal | BeginContactInner, CalculateAward, Draw, GetResult, Update, _ScoreGoal |
| 0x030 | `int32_t` | lastTouch | player index that last touched the ball (-1 none): goal/own-goal credit | BallPlayerCollision, BeginContactInner, HammerBallCollision, HammerPlayerCollision, PlayState, PlayerCollision | _ScoreGoal |
| 0x034 | `bool` | koWin | sudden death decided by a kill/deleter (Draw "KOText") | BeginContactInner, HammerPlayerCollision, PlayState, PlayerCollision | Draw |
| 0x035 | `bool` | unlockingRainbow | Unlocking(): rainbow-highway unlock in progress | BeginContactInner, PlayState | BeginContactInner, Draw, Unlocking |
| 0x036 | `bool` | timeOver | TimeOver()/SetTimeOver(); set when the clock ends with a winner | PlayState, SetTimeOver, Update | BallWallCollision, TimeOver, Update |
| 0x038 | `Animation*[2]` | scoreArrowAnim | [team] scoreArrow1/2 *(presentation)* | PlayState | Draw, _ScoreGoal, ~PlayState |
| 0x048 | `Animation*` | gullAnim |  *(presentation)* | PlayState | Draw, Update, ~PlayState |
| 0x050 | `Player*[4]` | players | [index]; team 0 = 0,2  team 1 = 1,3; only numPlayers valid | PlayState | BallContact, BallPlayerCollision, BeginContactInner, CalculateAward, Draw, EndContactInner, GetResult, GroundContact, HammerBallCollision, HammerPlayerCollision, LoadComplete, PlayState, PlayerCollision, PoleGroundCollision, PostSolveInner, Silence, Splash, Update, _CalculateBoxingDay, _ScoreGoal, ~PlayState |
| 0x070 | `Camera*` | camera |  *(presentation)* | PlayState | DisableScreenShake, Draw, Update, _ShockwaveAt, ~PlayState |
| 0x078 | `Ball*` | ball | recreated after every goal (ballResetPending) | PlayState, Update | BallPlayerCollision, Draw, HammerBallCollision, Update, _DrawWire, ~PlayState |
| 0x080 | `Target*[2]` | targets | [0] left, [1] right; targets[1-team]->GoalScored() *(presentation)* | PlayState | _ScoreGoal, ~PlayState |
| 0x090 | `ShockWave*[4]` | shockwaves | [player] (numPlayers allocated) *(presentation)* | PlayState | Draw, Update, _ShockwaveAt, ~PlayState |
| 0x0b0 | `Dust*` | dust |  *(presentation)* | PlayState | Draw, GroundContact, PlayState, Update, ~PlayState |
| 0x0b8 | `Sparks*` | sparks |  *(presentation)* | PlayState | Draw, Update, _ShockwaveAt, ~PlayState |
| 0x0c0 | `Flare*` | flare |  *(presentation)* | PlayState | Draw, Update, ~PlayState |
| 0x0d0 | `Common::Sound::Effect*` | sndFanfare |  *(presentation)* | PlayState | LoadComplete, ~PlayState |
| 0x0d8 | `Common::Sound::Effect*[4]` | sndGoal | Goal1A,Goal1B,Goal2A,Goal2B (random() picks) *(presentation)* | PlayState | _ScoreGoal, ~PlayState |
| 0x0f8 | `Common::Sound::Effect*` | sndGoldenGoal |  *(presentation)* | PlayState | Update, _GoSuddenDeath, ~PlayState |
| 0x100 | `Common::Sound::Effect*` | sndFart |  *(presentation)* | PlayState | Update, ~PlayState |
| 0x108 | `Common::Sound::Effect*` | sndSecret |  *(presentation)* | PlayState | BeginContactInner, ~PlayState |
| 0x110 | `bool` | ballResetPending | set when ballResetDelay expires / sudden death starts; Update rebuilds ball + respawns everyone | PlayState, Update, _GoSuddenDeath | Update |
| 0x114 | `float` | ballResetDelay | 1.4 s after a goal; >0 blocks _ScoreGoal and BallWallCollision | PlayState, Update, _ScoreGoal | BallWallCollision, Update, _ScoreGoal |
| 0x118 | `bool` | lastGoalTeam0 | last goal was by team 0 (camera focus) *(presentation)* | _ScoreGoal | Update |
| 0x120 | `Level1*` | level1 | only the slot for `level` is written (others uninitialized) | PlayState | ~PlayState |
| 0x128 | `Level2*` | level2 | (see level1) | PlayState | ~PlayState |
| 0x130 | `Level3*` | level3 | (see level1) | PlayState | ~PlayState |
| 0x138 | `Level4*` | level4 | (see level1) | PlayState | ~PlayState |
| 0x140 | `Level5*` | level5 | (see level1) | PlayState | ~PlayState |
| 0x148 | `Background*` | background1 |  *(presentation)* | PlayState | Draw, LoadComplete, Update, ~PlayState |
| 0x150 | `Background_Level2*` | background2 |  *(presentation)* | PlayState | Draw, Update, ~PlayState |
| 0x158 | `Background_Level3*` | background3 | (Splash() uses it for water droplets) *(presentation)* | PlayState | Draw, Splash, Update, ~PlayState |
| 0x160 | `Background_Level4*` | background4 |  *(presentation)* | PlayState | Draw, Update, ~PlayState |
| 0x168 | `Background_Level5*` | background5 |  *(presentation)* | PlayState | Draw, Update, ~PlayState |
| 0x170 | `int32_t` | killMask | bit i = player i may still die in PlayerCollision this tick; reset to 0xf each Update | PlayState, PlayerCollision, Update | PlayerCollision |
| 0x174 | `bool` | ballKickAvailable | one BallPlayerCollision kick per tick; reset to 1 each Update | BallPlayerCollision, PlayState, Update | BallPlayerCollision |
| 0x175 | `bool` | suddenDeath | poles are hammers; next goal / kill wins | PlayState, Update, _GoSuddenDeath | BeginContactInner, CalculateAward, Draw, HammerPlayerCollision, PlayerCollision, Update, _GoSuddenDeath, _ScoreGoal |
| 0x178 | `int32_t[2]` | sdLives | [team] sudden-death kills needed-1 (set 2 when 4 players respawn); 0 in 1v1 | PlayState, PlayerCollision, Update | PlayerCollision |
| 0x180 | `float` | timeLeft | match clock (s), runs only while timerOn; forced to 1.0 on a winning goal | PlayState, Update, _GoSuddenDeath, _ScoreGoal | Draw, GroundContact, Update, _ScoreGoal |
| 0x184 | `float` | timeLeftShown | clock snapshot shown on the win screen *(presentation)* | PlayState, _ScoreGoal | Draw |
| 0x188 | `bool` | timerOn | set on first ball/pole touch (EnableTimer) | BallPlayerCollision, BeginContactInner, EnableTimer, PlayState, PlayerCollision, _ScoreGoal | BallPlayerCollision, BeginContactInner, EnableTimer, PlayerCollision, Update |
| 0x18c | `float` | dt | 1/60; divided by 1.5 per tick once matchDecided (slow-mo) | PlayState, Update | Update |
| 0x190 | `float` | endSlowmoTimer | 3 s after matchDecided, then endFade starts | Update | Update |
| 0x194 | `bool` | matchDecided | clock ran out with a winner: players deactivated | PlayState, Update | Update |
| 0x198 | `DebugRender*` | debugRender | NULL unless Assets::debug *(presentation)* | PlayState | Draw, PlayState, ~PlayState |
| 0x1a0 | `float[4]` | respawnTimer | [player] >0: counting down; at <=0 the player respawns at spawn + Boost. Set to deathDelay (3 s) by kills (PlayerCollision, not in sudden death), 1/60 by deleters (Level5: deathDelay if y > 0), 0.5*dt for everyone on ball reset | BeginContactInner, PlayState, PlayerCollision, Update | BeginContactInner, Update |
| 0x1b0 | `SpriteSheet*` | sprites | sprites.ssht (shared with players/ball/dust) *(presentation)* | PlayState | Draw, PlayState, Update, ~PlayState |
| 0x1b8 | `SpriteSheet*` | messageSprites | messages.ssht *(presentation)* | PlayState | Draw, ~PlayState |
| 0x1c0 | `SpriteSheet*` | numberSprites | numbers.ssht *(presentation)* | PlayState | Draw, ~PlayState |
| 0x1c8 | `SpriteSheet*` | scoreArrowSprites | scorearrow.ssht *(presentation)* | PlayState | Draw, ~PlayState |
| 0x1d0 | `SpriteSheet*` | shockwaveSprites | shockwave.ssht *(presentation)* | PlayState | PlayState, ~PlayState |
| 0x1d8 | `SpriteSheet*` | additiveSprites | additive.ssht *(presentation)* | PlayState | PlayState, Update, ~PlayState |
| 0x1e0 | `SpriteSheet*` | wireSprites | wire.ssht *(presentation)* | PlayState | _DrawWire, ~PlayState |
| 0x1e8 | `float` | goalBanner | -1 idle; 0 on goal, +dt/1.7 to 1 then -1 *(presentation)* | PlayState, Update, _ScoreGoal | Draw, Update |
| 0x1ec | `float` | sdBanner | -1 idle; 0 on sudden death, +dt/0.3 to 1 *(presentation)* | PlayState, Update, _GoSuddenDeath | Draw, Update |
| 0x1f0 | `float` | timeUpBanner | -1 idle; 0 when decided on time, +1/112 per tick *(presentation)* | PlayState, Update | Draw, Update |
| 0x1f4 | `float` | winBanner | -1 idle; 0 on winning goal / decision, +1/208 per tick; >=0 BLOCKS _ScoreGoal | PlayState, Update, _ScoreGoal | Draw, Update, _ScoreGoal |
| 0x1f8 | `int32_t` | bannerFrame | Draw-only frame counter (mod 18) *(presentation)* | Draw, PlayState | Draw |
| 0x1fc | `bool` | gameOver | GameOver(): set when endFade passes 1 | PlayState, Update | GameOver |
| 0x200 | `float` | endFade | -1 = playing; Reset()/rainbow unlock/slow-mo end set 0; +1/84 per tick; >0 blocks goals | BeginContactInner, PlayState, Reset, Update | BeginContactInner, Draw, EndContactInner, PostSolveInner, Update, _ScoreGoal |
| 0x204 | `float` | fadeIn | 1 -> 0 at 2/s *(presentation)* | PlayState, Update | Draw, Update |
| 0x208 | `float` | volumeRamp | music/background volume ramp *(presentation)* | PlayState, Update | Update |
| 0x20c | `bool` | fromRainbowUnlock | ctor arg (_unlockingRainbowHighway): skip fade-in *(presentation)* | PlayState | Draw |
| 0x210 | `int32_t` | numPlayers | count of active session slots (2 or 4 in practice) | PlayState | BeginContactInner, CalculateAward, Draw, GetResult, LoadComplete, PlayState, Silence, Update, _CalculateBoxingDay, _ScoreGoal, ~PlayState |
| 0x214 | `int32_t` | level | 1..5 (Flatirons/Level2 = 2); selects LevelN/Background slots | PlayState | BeginContactInner, Draw, LoadComplete, PlayState, Update, ~PlayState |
| 0x218 | `int32_t` | hitstopFrames | >0: dt forced to 0 for this many ticks (_PunchTime / _ShockwaveAt set 7) | PlayState, Update, _PunchTime, _ShockwaveAt | Update |
| 0x21c | `float` | punchHeat | +2 per punch, -dt per tick; hit-stop only if < 5 after adding | PlayState, Update, _PunchTime, _ShockwaveAt | Update, _PunchTime, _ShockwaveAt |
| 0x220 | `bool` | reverbOn | ctor 1 *(presentation)* | PlayState | - |
| 0x221 | `bool` | compressorOn | ctor 1, passed to Sound::SetCompressor *(presentation)* | PlayState | PlayState |

## Player (0x240)

`Player(b2World*, int sessionSlot, int index, int team, int /*unused*/, SpriteSheet*, Dust*)` @0x100133582. The embedded `SessionPlayer` (+0x220) is a by-value copy of `Common::Session::Get()[slot]` (7 ints) and is passed by value to `Common::Input::VirtualLeftStick/RightStick/UpdateVibration/ButtonHeld`.

| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `Stats*` | stats | new Stats(index) (last thing the ctor does) | Player | GetStats, LiftOff, ~Player |
| 0x010 | `b2World*` | world | ctor arg | Player | Respawn, SetGhost, SpawnPlayerBody, _FootBoxTo, _OffPole, ~Player |
| 0x018 | `Pole*` | pole | owned unless sharingPole; GetPole() | Player, Respawn | Draw, DrawHandsAgain, GetHammer, GetPole, Respawn, SetGhost, Teleport, Update, _OffPole, _SetBallCollisionFilters, ~Player |
| 0x020 | `bool` | sharingPole | SharingPole(); set by Respawn(bool), cleared by UnsharePole | Respawn, UnsharePole | OffPole, SharingPole, _OffPole, ~Player |
| 0x021 | `bool` | reversedGrip | hands on the far pole end (shared-pole partner); picks segments in Update/SetGhost | Respawn | Draw, DrawHandsAgain, Update |
| 0x022 | `bool` | bodyActive | set by SpawnPlayerBody; cleared by GameOverBody/_ParalyzeBody/when an off-pole body settles | GameOverBody, SpawnPlayerBody, Update, _ParalyzeBody | Update |
| 0x023 | `bool` | shortPole | head #7: shorter pole, stiffer arm servo gains (0.014/0.07 vs 0.004/0.06) | Respawn | Update |
| 0x024 | `float` | runSpeed | run force scale (Respawn: 2100/ratio; SetRunSpeed) | Respawn, SetRunSpeed | Update |
| 0x028 | `bool` | onPole | OnPole(); SpawnPlayerBody=1, _OffPole=0 | SpawnPlayerBody, _OffPole | Draw, DrawHandsAgain, Land, LiftOff, OnPole, Update, UpdateGhostTimer, UpdateNormal, WallCollision, _OffPole, _SetBallCollisionFilters |
| 0x029 | `bool` | onGround | OnGround(); Land/LiftOff | Land, LiftOff, SpawnPlayerBody | Land, OnGround, Update, WallCollision, _FootBoxTo, _OffPole |
| 0x02a | `bool` | standingOnPole | StandingOnPole(); Land(bool onPole) | Land, LiftOff, Respawn | StandingOnPole, Update |
| 0x02c | `int32_t` | unk2c | ctor=0, never read ? | Player | - |
| 0x030 | `b2Body*` | body | rider body; userData = &index; Body() | Player, SpawnPlayerBody | Body, Draw, DrawIconsFront, Kick, Respawn, SetGhost, SpawnPlayerBody, Teleport, TriggerFart, Update, WallCollision, _FootBoxTo, _OffPole, ~Player |
| 0x038 | `SpriteSheet*` | bodySprites | body{1,2,1b,2b}_white.ssht *(presentation)* | Player | Draw, Player, ~Player |
| 0x040 | `SpriteSheet*` | headSprites | heads_white.ssht *(presentation)* | Player | Draw, ~Player |
| 0x048 | `SpriteSheet*` | fartSprites | fart.ssht *(presentation)* | Player | Draw, ~Player |
| 0x050 | `SpriteSheet*` | numberSprites | playernumbers.ssht *(presentation)* | Player | Draw, DrawIconsFront, ~Player |
| 0x058 | `SpriteSheet*` | sharedSprites | ctor arg (PlayState::sprites), handed to Pole *(presentation)* | Player | Draw, DrawHandsAgain, DrawIconsFront, Respawn, _DrawForeArm |
| 0x060 | `Dust*` | dust | ctor arg (PlayState::dust) *(presentation)* | Player | Update |
| 0x068 | `Animation*` | anim | body animation; kick state machine compares anim->data | Player, SpawnPlayerBody | Draw, Kick, Land, LiftOff, SpawnPlayerBody, Update, ~Player |
| 0x070 | `Animation*` | smashAnim | ShowSmash *(presentation)* | Player, SpawnPlayerBody | Draw, ShowSmash, SpawnPlayerBody, Update, ~Player |
| 0x078 | `b2RevoluteJoint*` | handJoint0 | body<->pole segment 0 (or n-2 if reversed), motor = arm servo | Respawn | Update, _OffPole |
| 0x080 | `b2RevoluteJoint*` | handJoint1 | body<->pole segment 2 (or n-3 if reversed), motor = arm servo | Respawn | Update, _OffPole |
| 0x088 | `b2Vec2` | drawHand0 | smoothed hand position for Draw *(presentation)* | Draw, Respawn | Draw |
| 0x090 | `b2Vec2` | drawHand1 |  *(presentation)* | Draw, Respawn | Draw |
| 0x098 | `int32_t` | team | 0 or 1 (ctor arg) | Player | Player, Respawn, SpawnPlayerBody, Update |
| 0x09c | `int32_t` | index | player index 0..3; body userData points HERE | Player | Draw, DrawIconsFront, Player, Respawn, _FootBoxTo |
| 0x0a0 | `int32_t` | partnerIndex | (index+2)%4 = teammate; userData of far handle segs of a shared pole | Player | - |
| 0x0a4 | `int32_t` | controllerNum | Input::GetControllerNum(slot) *(presentation)* | Player | Draw, DrawIconsFront |
| 0x0a8 | `int32_t` | inputId | = session.pad; ButtonHeld(inputId,...), achievements | Player | PlayState::PlayState, PlayState::Update, PlayState::_CalculateBoxingDay |
| 0x0ac | `b2Vec2` | groundNormal | contact normal (UpdateNormal); (0,-1) default; run force is along its tangent | SpawnPlayerBody, UpdateNormal | Update |
| 0x0b4 | `float` | boostStrength | Boost(f) | Boost, SpawnPlayerBody | Update |
| 0x0b8 | `float` | boostTimer | Boost sets 0.8, Update: force while >0, -= dt | Boost, SpawnPlayerBody, Update | Update |
| 0x0c8 | `b2Fixture*` | hullFixture | 9-vertex body polygon (_OffPole sets its m_restitution (+0x34) to 0.2) | SpawnPlayerBody | _OffPole, _SetBallCollisionFilters |
| 0x0d0 | `b2Fixture*` | footFixture | circle "foot box" (moved by _FootBoxTo; destroyed by _OffPole) | SpawnPlayerBody | _FootBoxTo, _OffPole, _SetBallCollisionFilters |
| 0x0d8 | `b2Fixture*` | sensorFixture | sensor circle, userData FIXTURE_SENSOR | SpawnPlayerBody | - |
| 0x0e0 | `b2Fixture*` | impaleFixture | sensor box above head, userData FIXTURE_IMPALE | SpawnPlayerBody | - |
| 0x0e8 | `Common::Sound::Persistent*` | bendSoundA |  *(presentation)* | Player | LiftOff, Update, ~Player |
| 0x0f0 | `Common::Sound::Persistent*` | bendSoundB |  *(presentation)* | Player | LiftOff, Update, ~Player |
| 0x0f8 | `Common::Sound::Effect*` | deathSoundA |  *(presentation)* | Player | PlayDeathSound, ~Player |
| 0x100 | `Common::Sound::Effect*` | deathSoundB |  *(presentation)* | Player | PlayDeathSound, ~Player |
| 0x108 | `Common::Sound::Effect*` | landHardSound |  *(presentation)* | Player | PlayHammerSound, PlayLandSound, ~Player |
| 0x110 | `Common::Sound::Effect*` | landSoftSound |  *(presentation)* | Player | PlayLandSound, ~Player |
| 0x118 | `Common::Sound::Effect*` | hitSoundA |  *(presentation)* | Player | PlayHitSound, ~Player |
| 0x120 | `Common::Sound::Effect*` | hitSoundB |  *(presentation)* | Player | PlayHitSound, ~Player |
| 0x128 | `Common::Sound::Effect*` | kickSoundA |  *(presentation)* | Player | PlayKickSound, ~Player |
| 0x130 | `Common::Sound::Effect*` | kickSoundB |  *(presentation)* | Player | PlayKickSound, ~Player |
| 0x138 | `Common::Sound::Effect*[3]` | tapSound |  *(presentation)* | Player | PlayHammerSound, PlayTap, ~Player |
| 0x150 | `Common::Sound::Effect*[4]` | ballSound |  *(presentation)* | Player | PlayBallSound, ~Player |
| 0x170 | `Common::Sound::Persistent*` | walkSound |  *(presentation)* | Player | LoadComplete, Respawn, Silence, Update, ~Player |
| 0x178 | `Common::Sound::Persistent*` | slideSound |  *(presentation)* | Player | GameOverBody, LoadComplete, Respawn, Silence, Update, _ParalyzeBody, ~Player |
| 0x180 | `float` | rumbleA | rumble accumulator (-dt per Update) *(presentation)* | PlayDeathSound, PlayHammerSound, PlayHitSound, PlayKickSound, PlayLandSound, RumbleBump, SpawnPlayerBody, Update, _OffPole | PlayDeathSound, PlayHammerSound, PlayHitSound, PlayKickSound, PlayLandSound, RumbleBump, Update, _OffPole |
| 0x184 | `float` | rumbleB | rumble level (x0.95 per Update) *(presentation)* | PlayDeathSound, PlayHammerSound, PlayHitSound, PlayKickSound, PlayLandSound, RumbleBump, SpawnPlayerBody, Update, _OffPole | PlayDeathSound, PlayHammerSound, PlayHitSound, PlayKickSound, PlayLandSound, RumbleBump, Update, _OffPole |
| 0x188 | `bool` | canPlayLandSound | (sound gating; sound fns call random()) *(presentation)* | DisableLandSound, LiftOff, PlayLandSound, Respawn | CanPlayLandSound, PlayLandSound |
| 0x189 | `bool` | canPlayHammerSound |  *(presentation)* | HammerLiftoff, PlayHammerSound, Respawn | CanPlayHammerSound, PlayHammerSound, ShowSmash |
| 0x18a | `bool` | canPlayDeathSound |  *(presentation)* | PlayDeathSound, Respawn | PlayDeathSound |
| 0x18c | `float` | bendVolume | pole-bend sound volume *(presentation)* | SpawnPlayerBody, Update | Update |
| 0x190 | `bool` | emitDust | ? dust/slide effects allowed (cleared with bodyActive) | GameOverBody, SpawnPlayerBody, Update, _ParalyzeBody | Update |
| 0x191 | `bool` | rainbowFlare | ShouldAddFlare(): on level 2 near centre, high up; PlayState::Update then calls random() 8x/player | Player, Respawn, SpawnPlayerBody, Update | ShouldAddFlare, Update |
| 0x194 | `int32_t` | groundContacts | Land ++ / LiftOff -- (LiftOff at <=0 => airborne) | Land, LiftOff, SpawnPlayerBody | Land, LiftOff |
| 0x198 | `int32_t` | hammerGroundContacts | HammerLand / HammerLiftoff | HammerLand, HammerLiftoff | HammerLand, HammerLiftoff |
| 0x19c | `float[4]` | rightStick | VirtualRightStick() result: [0]=x [1]=y (pole-angle setpoint), [2],[3] never read | SpawnPlayerBody, Update, _UpdateSticks | Draw, Update |
| 0x1ac | `float[4]` | leftStick | VirtualLeftStick(): [0]=x (run) [1]=y, [2],[3] never read | SpawnPlayerBody, Update, _UpdateSticks | Update |
| 0x1bc | `float` | animClock | += dt, wraps at 200 s *(presentation)* | SpawnPlayerBody, Update | Draw, Update |
| 0x1c0 | `int32_t` | poleGroundContacts | PoleContactAdjust(+-1), clamped >= 0 | PoleContactAdjust | LiftOff, PoleContactAdjust |
| 0x1c4 | `bool` | liftoffFlag | ? written by LiftOff/Update/SpawnPlayerBody, never read | LiftOff, SpawnPlayerBody, Update | - |
| 0x1c5 | `bool` | kicking | set by Kick, cleared by Land / when kick anim finishes; blocks kick-aim | Kick, Land, SpawnPlayerBody, Update | Update |
| 0x1c6 | `bool` | facingLeft | init team!=0; flips on turns; mirrors kick/foot box | SpawnPlayerBody, Update | Draw, DrawHandsAgain, DrawIconsFront, Kick, TriggerFart, Update, _FootBoxTo |
| 0x1c7 | `bool` | offPolePending | OffPole(): _OffPole() runs at next Update | OffPole, Respawn, Update | Update |
| 0x1c8 | `bool` | unsharePending | OffPole() while sharing; ShouldUnshare() | OffPole, ResetUnshare, Respawn | ShouldUnshare |
| 0x1cc | `float` | ballNoCollideTimer | >0 => ball collisions disabled (SetBallCollisions); -= dt | Land, SetBallCollisions, SpawnPlayerBody, Update | Draw, ShouldBallCollide, Update, _SetBallCollisionFilters |
| 0x1d0 | `bool` | active | SetActive(); false => sticks read as 0 | SetActive, SpawnPlayerBody | Update, _UpdateSticks |
| 0x1d4 | `float` | tapTimer | PlayTap cooldown *(presentation)* | PlayTap, Respawn, Update | PlayTap, Update |
| 0x1d8 | `float` | walkVolume | walk sound volume *(presentation)* | Respawn, Update | Update |
| 0x1dc | `float` | tauntTimer | Taunting() cooldown (0.8 s) | Respawn, Taunting, Update | Taunting, Update |
| 0x1e0 | `bool` | tauntFlag | ? cleared every Update / Respawn | Respawn, Update | - |
| 0x1e4 | `float` | fartTimer | TriggerFart = 1 *(presentation)* | Respawn, TriggerFart, Update | Draw, Update |
| 0x1e8 | `b2Vec2` | fartPos |  *(presentation)* | TriggerFart | Draw |
| 0x1f0 | `float` | smashTimer | ShowSmash = 1 *(presentation)* | Respawn, ShowSmash, Update | Draw, Update |
| 0x1f4 | `b2Vec2` | smashPos |  *(presentation)* | ShowSmash | Draw |
| 0x1fc | `bool` | fartFlip |  *(presentation)* | TriggerFart | Draw |
| 0x1fd | `bool` | ghost | GetGhost(); SetGhost(b, t) | SetGhost, SpawnPlayerBody | GetGhost |
| 0x200 | `float` | ghostTimer | GetGhostTimer(); counts down only while onPole; ctor -1 | Player, SetGhost, UpdateGhostTimer | Draw, GetGhostTimer, UpdateGhostTimer |
| 0x204 | `bool` | justUnghosted | JustUnghosted(); set by SetGhost(false), cleared by UpdateGhostTimer | SetGhost, UpdateGhostTimer | JustUnghosted |
| 0x208 | `float` | paralyzeFade | -1 = alive; 0..1 (+0.5*dt) after _ParalyzeBody *(presentation)* | SpawnPlayerBody, Update, _ParalyzeBody | Draw, DrawHandsAgain, Update |
| 0x20c | `float[4]` | unk20c | ? zeroed by SpawnPlayerBody, never read | SpawnPlayerBody | - |
| 0x21c | `bool` | drawFlag21c | ? Draw only *(presentation)* | Draw, Player | Draw |
| 0x220 | `SessionPlayer` | session | copy of Common::Session::Get()[slot] | Player | Draw, GetHeadNumber, Player, Taunting, Update, _UpdateSticks, _UpdateVibration |

## Pole (0x78)

`Pole(x, y, world, team, index, angle, sprites, isHammer, lengthPx, shared, shortPole)` @0x10013e64c. Collision filters come from `team` (see ctor). Segments are `new`ed in index order 0..9, then 9 weld joints, then (hammer) head body + weld to segment 9.

| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `bool` | isHammer | ctor arg 8 (sudden death): Hammer() returns hammerBody iff set | Pole | Detach, Draw, Hammer, Player::Draw, Player::DrawHandsAgain, Player::Teleport, Player::Update, ~Pole |
| 0x001 | `bool` | shared | ctor stack arg 1: pole shared by two riders (mode 1/2); changes segment userData | Pole | Draw |
| 0x002 | `bool` | shortPole | ctor stack arg 2 (= Player::shortPole) *(presentation)* (Pole::Draw only) | Pole | Draw |
| 0x004 | `float` | lengthPx | pole length in px (Assets::standardPoleLength etc.); hammer head offset | Pole | Draw, Pole |
| 0x008 | `b2Body*[10]` | segments | segment bodies, welded chain; Segment(i), NumSegments()==10 | Pole | Draw, Pole, Segment |
| 0x058 | `SpriteSheet*` | sprites | (= Player::sharedSprites) *(presentation)* | Pole | Draw |
| 0x060 | `int32_t` | team | ctor arg (selects collision filter bits) | Pole | - |
| 0x064 | `int32_t` | ownerIndex | player index; &ownerIndex is the userData of every pole fixture | Pole | Draw, Pole |
| 0x068 | `b2Body*` | hammerBody | hammer head body (BODY_HAMMER), only if isHammer (else uninitialized) | Pole | Draw, Hammer, Pole, ~Pole |
| 0x070 | `b2World*` | world | ctor arg | Pole | Pole, ~Pole |

## Ball (0x48)

`Ball(x, heightPx, world, sprites, additive, levelWidthPx)` @0x1001234e0 only stores fields; `CreateBody()` @0x1001235e0 builds ball, trolley, distance joint (freq 2, damping 0.2, via +0xb8/+0xbc), anchor, prismatic joint.

| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `b2Body*` | body | the ball (BODY_BALL, circle r=24px); Body() | CreateBody | Body, CreateBody, Draw, Ghost, Update, ~Ball |
| 0x008 | `b2Body*` | trolley | zipline trolley (BODY_BALL, r=10px, filter cat 0x100 mask 0) | CreateBody | CreateBody, Ghost, ~Ball |
| 0x010 | `b2DistanceJoint*` | wire | ball<->trolley distance joint; Update sets its frequencyHz (+0xb8) | CreateBody | Update |
| 0x018 | `b2World*` | world | ctor arg | Ball | CreateBody, ~Ball |
| 0x020 | `float` | heightPx | ctor arg (PlayState::ballHeight): spawn y in px | Ball | CreateBody |
| 0x028 | `SpriteSheet*` | sprites |  *(presentation)* | Ball | Draw |
| 0x030 | `SpriteSheet*` | additive |  *(presentation)* | Ball | Draw |
| 0x038 | `float` | spin | += dt, wrapped to 2pi (Draw) *(presentation)* | Ball, Update | Draw, Update |
| 0x03c | `float` | fadeIn | 0 -> 1 by dt (Draw) *(presentation)* | Ball, Update | Draw, Update |
| 0x040 | `float` | levelWidthPx | ctor arg (PlayState::levelWidth); Update force/spring scale | Ball | Update |
| 0x044 | `bool` | frozen | Ghost() sets; Update then SetType(static); Frozen() | Ball, Ghost | Draw, Frozen, Update |

## Stats (0x34)

Pure counters, never read by gameplay (only by the victory screen), but written from gameplay code.

| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `int32_t` | playerNum | Stats(int) arg = Player::index | Stats | PlayerNum |
| 0x004 | `int32_t` | kicks | AddKick | AddKick, Stats | AddKick, Kicks |
| 0x008 | `int32_t` | jumps | AddJump (Player::LiftOff) | AddJump, Stats | AddJump, Jumps |
| 0x00c | `int32_t` | suicides | AddSuicide (deleter contact while on pole) | AddSuicide, Stats | AddSuicide, Suicides |
| 0x010 | `float` | timeLeft | SetTimeLeft (clock snapshot at match end); ctor -1 | SetTimeLeft, Stats | TimeLeft |
| 0x014 | `int32_t` | kills | AddKill | AddKill, Stats | AddKill, Kills |
| 0x018 | `int32_t` | deaths | AddDeath | AddDeath | AddDeath, Deaths |
| 0x01c | `float` | timeNearestBall | IncrementTimeNearestBall | IncrementTimeNearestBall, Stats | IncrementTimeNearestBall, TimeNearestBall |
| 0x020 | `int32_t` | goals | AddGoal | AddGoal | AddGoal, Goals |
| 0x024 | `int32_t` | ownGoals | AddOwnGoal | AddOwnGoal, Stats | AddOwnGoal, OwnGoals |
| 0x028 | `float` | offenseRatio | IncrementOffenseRatio (+-dt by ball side) | IncrementOffenseRatio | IncrementOffenseRatio, OffenseRatio |
| 0x02c | `float` | timeInGoal | IncrementTimeInGoal | IncrementTimeInGoal, Stats | IncrementTimeInGoal, TimeInGoal |
| 0x030 | `float` | timeInEndzone | IncrementTimeInEndzone | IncrementTimeInEndzone | IncrementTimeInEndzone, TimeInEndzone |

## Target (0x20)

Presentation only; `_ScoreGoal` calls `targets[1-team]->GoalScored()`.

| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `float` | x | px (ctor arg) *(presentation)* | Target | Draw |
| 0x004 | `float` | y | px (ctor arg) *(presentation)* | Target | Draw |
| 0x008 | `bool` | mirrored | ctor bool: false = left target (target.ssht), true = right (target2.ssht) *(presentation)* | Target | Draw |
| 0x010 | `Animation*` | anim | animWait / animHit *(presentation)* | Target | Draw, GoalScored, Update, ~Target |
| 0x018 | `SpriteSheet*` | sprites |  *(presentation)* | Target | Draw, ~Target |

## Animation (0x18)

Methods are not inlined; gameplay-relevant through `Player::anim`.

| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `const AnimationData*` | data | current clip (ctor: &nullAnimation); Play() swaps it | Animation, Play | GetData, GetFrame, Play, Update |
| 0x008 | `float` | position | normalized 0..1 (>=1 => IsFinished) | Play, SetPosition, Update | GetFrame, GetPosition, IsFinished, Update |
| 0x00c | `float` | rate | playback rate multiplier (Play resets to 1.0) | Play, SetRate | Update |
| 0x010 | `bool` | playing | IsPlaying(); cleared by GetFrame() at the end *(presentation)* | Animation, GetFrame, Play | IsPlaying |

## AnimationData (static tables)

No accessor functions; read by `Animation::Update/GetFrame`.

| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `int32_t[32]` | frames | sprite frame indices *(presentation)* | - | - |
| 0x080 | `int32_t` | numFrames | frame count (Update: pos += dt*rate*fps/numFrames) | - | - |
| 0x084 | `float` | fps | frames per second | - | - |
| 0x088 | `bool` | loop | wraps position back below 1.0 when set | - | - |

## Common::Session::Player (0x1c, embedded in Player at +0x220)



| off | type | name | meaning | writers | readers |
|---|---|---|---|---|---|
| 0x000 | `int32_t` | mode | 0 = whole pad; 1/2 = shared-controller halves (unused in training) | - | - |
| 0x004 | `int32_t` | team | 0 = empty slot, -1 = team 0, else team 1 (PlayState ctor reads it) | - | - |
| 0x008 | `int32_t` | pad | gamepad index (Input::_pads[pad]); also Player::inputId | - | - |
| 0x00c | `int32_t` | head | head/character number (GetHeadNumber; 7 = short pole) | - | - |
| 0x010 | `int32_t` | unk10 | ? | - | - |
| 0x014 | `int32_t` | unk14 | ? (default 7) | - | - |
| 0x018 | `int32_t` | unk18 | ? | - | - |

## Ownership graph

```
PoleRiders::_playState (global) --new--> PlayState (0x228)
  PlayState.world ----------> b2World (0x19368)   [all bodies/joints below live in it]
  PlayState.levelN ---------> LevelN {world}      -> CreateBody(0,0): ground/wall/deleter/sea bodies (not stored)
  PlayState.players[i] -----> Player (0x240)
     Player.stats ----------> Stats (0x34)
     Player.body -----------> b2Body  (SpawnPlayerBody: DestroyBody(old) + CreateBody; 4 fixtures:
                                       hullFixture, sensorFixture, footFixture, impaleFixture, created in that order)
     Player.pole -----------> Pole (0x78)         (Respawn: ~Pole + delete old, new Pole; shared poles are not owned)
        Pole.segments[10] --> b2Body x10, welded by 9 b2WeldJoints (not stored)
        Pole.hammerBody ----> b2Body (only when isHammer) + weld joint to segments[9] (not stored)
     Player.handJoint0/1 ---> b2RevoluteJoint x2 (body<->pole), destroyed by _OffPole
     Player.anim/smashAnim -> Animation x2 (re-newed in SpawnPlayerBody)
     sprites / sounds ------> SpriteSheet x4, Sound::Effect x13, Sound::Persistent x4   (presentation)
  PlayState.ball -----------> Ball (0x48)
     Ball.body, Ball.trolley -> b2Body x2 (BODY_BALL), Ball.wire -> b2DistanceJoint
     + static BODY_ANCHOR body and a b2PrismaticJoint anchor<->trolley: NOT stored, and ~Ball
       only destroys body+trolley => every ball reset LEAKS one static anchor body into the world
  PlayState.targets[2] -> Target, shockwaves[n] -> ShockWave, camera, dust, sparks, flare,
  backgroundN, 7 SpriteSheets, 8 Sound::Effects, 3 Animations, debugRender (presentation)
```

Construction order inside `PlayState::PlayState` (matters for Box2D body order): world ->
LevelN (+ ground bodies) -> Background -> Camera -> Targets -> sprite sheets -> Dust/Sparks/Flare
-> for each session slot 0..3 (in slot order): `new Player` then `Player::Respawn` (SpawnPlayerBody
creates the rider body, then `new Pole` creates 10 segments [+hammer], then 2 revolute joints)
-> Animations -> `new Ball` + `Ball::CreateBody` (ball, trolley, distance joint, anchor, prismatic
joint) -> sounds -> ShockWaves. Player index for slot k: team `-1` slots get 0 then 2, other
slots get 1 then 3, so `players[]` may be filled out of slot order.

## Box2D userData conventions

`PoleRiders::Assets::BODY_*` / `FIXTURE_*` are `static int`s at 0x10028e06c.. whose value is
their own ordinal: BODY_PLAYER1..4 = 0..3, GROUND 4, WALL 5, POLE 6, BALL 7, ANCHOR 8, NONE 9,
DELETER 10, HAMMER 11, SEA 12, FIXTURE_SENSOR 13, FIXTURE_NONE 14, FIXTURE_HAMMERBODY1/2 15/16
(unused by PoleRiders), FIXTURE_IMPALE 17. `HANDLE_START` (0x10028e0e8) = 2.

userData always holds an `int*`; decoding (`PlayState::BeginContact/EndContact/PostSolve/
ContactToInts`) is `tag = ud ? *(int*)ud : 0` for `fixture->m_body->m_userData` (body tag) and
`fixture->m_userData` (fixture tag), for both fixtures, then `XxxInner(contact, bodyA, fixA,
bodyB, fixB)` is called twice with A/B swapped. **A NULL userData decodes as 0 = BODY_PLAYER1.**

| object | body userData | fixture userData |
|---|---|---|
| rider body (`Player.body`) | `&player->index` (Player+0x9c) -> 0..3 | hull: NULL(0); foot circle: NULL(0); sensor circle: `&FIXTURE_SENSOR`; impale box: `&FIXTURE_IMPALE` |
| pole segment i (ctor) | i <= HANDLE_START (0,1,2): `&pole->ownerIndex` (Pole+0x64); else `&BODY_POLE`. Shared pole: i < 2 -> `&ownerIndex`, 2 <= i < 8 -> BODY_POLE, i >= 8 -> **address of a stack local in Pole::Pole** holding the partner index (dangling!) | all: `&pole->ownerIndex` |
| pole segment i (after `SetGhost(false)`, i.e. on every un-ghost) | rewritten: i < HANDLE_START (0,1) -> `&player->index`; reversed grip: i > 10-2 -> `&player->index`; shared: far end (i >= 8) -> `&player->partnerIndex` (Player+0xa0); else `&BODY_POLE` (**note: segment 2 changes from player-tag to BODY_POLE here**) | unchanged |
| hammer head (`Pole.hammerBody`) | `&BODY_HAMMER` | `&pole->ownerIndex` |
| ball, trolley | `&BODY_BALL` | NULL |
| zipline anchor | `&BODY_ANCHOR` | NULL |
| level bodies | `&BODY_GROUND` / `&BODY_WALL` / `&BODY_DELETER` / `&BODY_SEA` / `&BODY_NONE` (some set after creation via `body->m_userData`) | NULL |
| Level4 deleters (`PlayState::_MakeDeleters`) | `&BODY_DELETER` | NULL |

So "body tag 0..3" = a rider body **or a pole handle segment** of that rider, and "fixture tag
0..3" on a POLE/HAMMER body = the pole's owner. `_FootBoxTo` also walks `world->m_bodyList` and
dereferences every non-NULL body userData as an int.

Box2D internals the game writes directly (inlined): `b2World::m_continuousPhysics`
(+0x19365) = false every `PlayState::Update` before `Step(dt, 60, 18)`; `b2Contact::m_flags |= 8`
(filter flag) on all contacts of `world->m_contactList` (+0x192f8) in `_OffPole`/`SetGhost(true)`
and on the contact in `HammerPlayerCollision`; `b2Fixture` shape of `footFixture` (`b2CircleShape::m_p`,
+0x10/+0x14) moved by `_FootBoxTo`; `hullFixture->m_restitution`(+0x34)=0.2 in `_OffPole`;
`b2DistanceJoint::m_frequencyHz` (+0xb8) in `Ball::Update`; `b2Body` force/velocity/awake
fields via inlined `ApplyForce/ApplyLinearImpulse/SetAwake`.

## Surprises / things the port must reproduce

1. **`PlayState::spawnBoost` (+0x18) is always `Level1::_spawnBoost` (50)** — all five level
   branches of the ctor load `Level1::_spawnBoost`; `Level2::_spawnBoost` (80, quoted in HANDOFF)
   and the other levels' values are never referenced. It is gameplay: respawn `Boost()` =
   `-(spawnBoost + 20*row) * side`.
2. **Level id**: the Level2/Flatirons branch sets `PlayState::level` (+0x214) = **2**
   (HANDOFF says "Flatirons reports 1" — worth re-checking in the bridge; `Assets::currentLevel`
   is a separate global that `Player::Update` tests `== 2` for the rainbow flare and `== 4`).
3. **Ball reset leaks the zipline anchor**: `~Ball` destroys only `body` and `trolley`; the static
   BODY_ANCHOR body from each `Ball::CreateBody` stays in the world (its prismatic joint dies with
   the trolley). After n goals the world has n extra static bodies (affects body list order,
   broadphase proxies, and thus bit-exactness).
4. **Dangling userData** for shared poles (mode 1/2 only; unused in training): far-end handle
   segments point into `Pole::Pole`'s stack frame until the next `SetGhost(false)` re-points them.
5. Handle segment range differs between `Pole::Pole` (i <= 2) and `Player::SetGhost(false)` (i < 2):
   segment 2 is a player-tagged body until the rider first un-ghosts (Respawn ends with
   `SetGhost(true, 0.3)`, so this happens 0.3 s of on-pole time after every spawn).
6. **`Ball::Update` calls libm `pow(double, 3.0)`** (x-position-cubed centring force, x -35) —
   not guaranteed bit-identical on arm64 (PORTING.md list). Other libm in gameplay: `atan2f`,
   `fmodf`, `fminf`, `fmaxf`, `sqrtf` (identical), `cosf`/`sinf` (in `Pole::Pole` and
   `PlayState::PlayerCollision`: use x86_sinf/x86_cosf).
7. **random()/srandom()**: `PlayState::PlayState` and **each `Player::Player`** call
   `srandom(Time_GetHighResTimer())`; the ctor then draws 2 values (snake/pencil). Afterwards
   random() is consumed by presentation paths that are gated by gameplay state: `_ScoreGoal`
   (goal sound), `_ShockwaveAt` (spark count), `Player::LiftOff` (when poleGroundContacts > 0),
   sound helpers (`PlayDeathSound/KickSound/HitSound/HammerSound/PlayTap`), `GroundContact`,
   `Splash`, `Dust::AddParticle` (18 per call), `Sparks/Flare`, and `PlayState::Update` (8 per
   player per tick when `Player::rainbowFlare`). No gameplay value reads random() after the ctor.
8. Per-step contact budgets live in PlayState: `killMask` (+0x170, reset to 0xF each tick; a
   victim's bit is cleared so a rider dies at most once per step) and `ballKickAvailable`
   (+0x174, one ball kick per step). Both are reset at the *start* of `Update`, before `Step`.
9. `winBanner` (+0x1f4, presentation-looking) is **gameplay**: `_ScoreGoal` returns early while
   it is >= 0, as it does while `ballResetDelay` (+0x114) > 0 or `endFade` (+0x200) > 0.
10. A winning goal does not end the match directly: `_ScoreGoal` forces `timeLeft = 1.0`; the
    normal clock expiry 1 s later sets `matchDecided`, after which `dt` is divided by 1.5 every
    tick (slow motion) and `endSlowmoTimer` (3 s) -> `endFade` (84 ticks) -> `gameOver`.
11. After every goal, when the ball is rebuilt, `respawnTimer[i] = 0.5*dt` for all players:
    everybody respawns on the next tick.
12. `Player::anim` (+0x68) is gameplay state: the kick/aim logic compares `anim->data` with the
    static `kickAnim*`/`kickAnim*_hold`/`turnAnim`/`landingAnim`/`poleIdleAnim` tables and clears
    `kicking` when a kick clip `IsFinished()`. The `AnimationData` tables (numFrames/fps/loop at
    +0x80/+0x84/+0x88, stride 0xa0 at 0x10028ed40..) must be copied into the sim.
13. Unused slots of `PlayState::level1..5` / `background1..5` are never initialized (heap garbage).
14. `Player` fields never accessed: 0x08..0x0f, 0xbc..0xc7; `PlayState` 0x1c, 0xc8..0xcf.
    `Player.unk20c[4]` is zeroed by `SpawnPlayerBody` but never read; `liftoffFlag` (+0x1c4) and
    `unk2c` are write-only.

## Level subclasses

All five `LevelN` are 8 bytes `{b2World* world}`, no vtable; `LevelN::LevelN(x, y, world)` stores
the world and tail-calls `LevelN::CreateBody(x, y)` (called with 0,0). Level-specific data is in
class statics (`LevelN::_levelWidth/_ballHeight/_targetWidth/_spawnHeight/_spawnBoost/_friction/
_restitution/_density/_angle`), not in the object. Per-level constants copied into PlayState:

| level | class | levelWidth | ballHeight | targetWidth | spawnHeight | (own _spawnBoost, unused) | initial spawn |x| px | background |
|---|---|---|---|---|---|---|---|---|
| 1 | Level1 | 2800 | -394 | 2606 | -612 | 50 (used by all) | 400 | Background (0x358) |
| 2 | Level2 (Flatirons) | 3520 | -394 | 2600 | -782 | 80 | 800 | Background_Level2 |
| 3 | Level3 | 2930 | -364 | 2294 | -652 | 55 | 800 | Background_Level3 (water: BODY_SEA, Splash) |
| 4 | Level4 | 4652 | -421 | 3600 | -512 | 150 | 1000 | Background_Level4; `_MakeDeleters` adds deleter bodies on respawn; players teleported across at |x| > levelWidth/2+412 px |
| 5 | Level5 (Rainbow Highway) | 2926 | -394 | 2600 | -652 | 70 | 400 | Background_Level5; rainbowPoleLength; death delay only when y > 0 |

Tag usage in `CreateBody`: Level1 GROUND x3, WALL x2, DELETER, NONE; Level2 GROUND x3, WALL x4,
DELETER, NONE; Level3 adds SEA and a 4th GROUND; Level4 GROUND x1, WALL x3; Level5 GROUND x4,
WALL x2. (Geometry/filters/friction: see the function inventory.)

